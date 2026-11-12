"""Native in-process DuckDB and PyTorch execution strategy for Gemma-4-SQL ETL, SFT, DPO, and evaluation."""

import json
import logging
import math
from pathlib import Path

from t1d_analytics.errors import InvalidTableNameError
from t1d_analytics.gemma_library_adapter import (
    DpoRunResult,
    EtlRunResult,
    EvalRunResult,
    SftRunResult,
)
from t1d_analytics.sql_agent import score_sql_completion_logprob
from t1d_analytics.sql_utils import escape_sql_identifier, normalize_sql

logger = logging.getLogger(__name__)


class NativeDuckDbStrategy:
    """Native in-process DuckDB and PyTorch execution strategy implementing `GemmaExecutionStrategy`."""

    @property
    def execution_mode(self) -> str:
        """
        Return canonical execution mode identifier.

        Returns:
            str: Always `'native_in_process'`.

        """
        return "native_in_process"

    def run_etl(
        self,
        stage_lower: str,
        db_path: Path,
        duckdb_table: str,
        output_dir: str | Path | None = None,
        timeout: float = 300.0,
    ) -> EtlRunResult:
        """
        Execute native in-process DuckDB ETL exporting JSONL and Parquet shards.

        Args:
            stage_lower: Normalized stage identifier ('pretrain', 'sft', 'posttrain').
            db_path: Validated path to the source DuckDB database.
            duckdb_table: Source table identifier.
            output_dir: Optional output directory path.
            timeout: Unused execution timeout parameter for strategy parity.

        Returns:
            EtlRunResult: Structured ETL execution result.

        Raises:
            InvalidTableNameError: If `duckdb_table` is not a valid SQL identifier.

        """
        import duckdb

        _ = timeout
        if not duckdb_table.isidentifier():
            raise InvalidTableNameError(
                f"Invalid DuckDB table identifier: '{duckdb_table}'"
            )

        escaped_table = escape_sql_identifier(duckdb_table)
        out_p = (
            Path(output_dir)
            if output_dir
            else db_path.parent / f"{duckdb_table}_{stage_lower}"
        )
        out_p.mkdir(parents=True, exist_ok=True)

        conn = duckdb.connect(str(db_path), read_only=True)
        try:
            rel = conn.execute(f'SELECT * FROM "{escaped_table}"')
            cols = [d[0] for d in rel.description] if rel.description else []
            rows = rel.fetchall()
            row_count = len(rows)

            col_projection = ", ".join(f'"{escape_sql_identifier(c)}"' for c in cols)
            default_sql = (
                f'SELECT {col_projection} FROM "{escaped_table}"'
                if col_projection
                else f'SELECT * FROM "{escaped_table}"'
            )

            jsonl_file = out_p / f"{stage_lower}_data.jsonl"
            with open(jsonl_file, "w", encoding="utf-8") as f:
                for r in rows:
                    rec = dict(zip(cols, r, strict=False))
                    prompt_val = str(
                        rec.get("prompt")
                        or rec.get("question")
                        or f"Retrieve {', '.join(cols)} from clinical table {duckdb_table}"
                    )
                    completion_val = str(
                        rec.get("completion") or rec.get("sql") or default_sql
                    )
                    f.write(
                        json.dumps(
                            {
                                "prompt": prompt_val,
                                "completion": completion_val,
                                "data": rec,
                            },
                            default=str,
                        )
                        + "\n"
                    )

            parquet_file = out_p / f"{stage_lower}_data.parquet"
            escaped_pq = str(parquet_file.resolve()).replace("'", "''")
            copy_sql = (
                f'COPY (SELECT * FROM "{escaped_table}") '
                f"TO '{escaped_pq}' (FORMAT PARQUET)"
            )
            conn.execute(copy_sql)
        finally:
            conn.close()

        return {
            "status": "completed",
            "execution_mode": self.execution_mode,
            "result": {
                "table": duckdb_table,
                "rows": row_count,
                "output_dir": str(out_p.resolve()),
            },
        }

    def run_sft(
        self,
        model_name: str,
        ds_p: Path,
        out_p: Path,
        hyperparameters: dict[str, int | float | str | bool] | None = None,
        timeout: float = 600.0,
        dry_run: bool = True,
    ) -> SftRunResult:
        """
        Execute native in-process SFT training fallback via LocalCpuRunner.

        Args:
            model_name: Target model name.
            ds_p: Validated dataset path.
            out_p: Validated output directory path.
            hyperparameters: Optional hyperparameter dictionary.
            timeout: Unused execution timeout parameter for strategy parity.
            dry_run: Whether to run in dry-run validation mode.

        Returns:
            SftRunResult: Structured SFT execution result.

        """
        from t1d_analytics.models import (
            TrainingBackend,
            TrainingHyperparameters,
            TrainingJobConfig,
        )
        from t1d_analytics.training_runner import get_training_runner

        _ = timeout
        runner = get_training_runner(TrainingBackend.LOCAL_CPU)
        lr_val = 2e-5
        epochs_val = 1
        use_dry_run = dry_run
        use_tiny = not dry_run
        if hyperparameters:
            lr_val = float(hyperparameters.get("learning_rate", 2e-5))
            epochs_val = int(hyperparameters.get("num_epochs", 1))
            if "dry_run" in hyperparameters:
                use_dry_run = bool(hyperparameters["dry_run"])
            if "use_tiny_model" in hyperparameters:
                use_tiny = bool(hyperparameters["use_tiny_model"])

        train_cfg = TrainingJobConfig(
            model_name=model_name,
            backend=TrainingBackend.LOCAL_CPU,
            dataset_path=str(ds_p.resolve()),
            output_dir=str(out_p.resolve()),
            dry_run=use_dry_run,
            use_tiny_model=use_tiny,
            hyperparameters=TrainingHyperparameters(
                learning_rate=lr_val,
                num_epochs=epochs_val,
            ),
        )
        res_runner = runner.run_training(train_cfg)
        return {
            "status": "completed",
            "execution_mode": self.execution_mode,
            "result": res_runner,
        }

    def run_dpo(
        self,
        model_name: str,
        ds_p: Path,
        out_p: Path,
        beta: float = 0.1,
        timeout: float = 600.0,
        dry_run: bool = True,
    ) -> DpoRunResult:
        """
        Execute native in-process DPO alignment using the Bradley-Terry preference objective.

        Args:
            model_name: Target model name.
            ds_p: Validated preference dataset path.
            out_p: Validated output directory path.
            beta: DPO temperature hyperparameter.
            timeout: Unused execution timeout parameter for strategy parity.
            dry_run: Whether to run in dry-run validation mode.

        Returns:
            DpoRunResult: Structured DPO execution result.

        """
        from t1d_analytics.job_manager import JobMetricValue
        from t1d_analytics.training_runner import optimize_dpo_preference_pairs

        _ = timeout
        pairs: list[dict[str, str]] = []
        with open(ds_p, encoding="utf-8") as f:
            for raw_line in f:
                raw_line = raw_line.strip()
                if not raw_line:
                    continue
                try:
                    row = json.loads(raw_line)
                    if isinstance(row, dict):
                        prompt_str = str(
                            row.get("prompt") or row.get("question") or ""
                        ).strip()
                        chosen_str = str(
                            row.get("chosen_sql")
                            or row.get("chosen")
                            or row.get("completion")
                            or row.get("sql")
                            or ""
                        ).strip()
                        rejected_str = str(
                            row.get("rejected_sql") or row.get("rejected") or ""
                        ).strip()
                        if chosen_str:
                            pairs.append(
                                {
                                    "prompt": prompt_str,
                                    "chosen_sql": chosen_str,
                                    "rejected_sql": rejected_str,
                                }
                            )
                except json.JSONDecodeError:
                    continue

        ckpt_dir = out_p / "checkpoint-final"
        if not dry_run and pairs:
            dpo_opt = optimize_dpo_preference_pairs(
                pairs=pairs,
                beta=beta,
                num_steps=1,
                checkpoint_dir=ckpt_dir,
            )
            dpo_losses = dpo_opt.losses
            reward_margins = dpo_opt.margins
        else:
            dpo_losses = []
            reward_margins = []
            for pair in pairs:
                logp_chosen = score_sql_completion_logprob(
                    pair["prompt"], pair["chosen_sql"]
                )
                logp_rejected = score_sql_completion_logprob(
                    pair["prompt"], pair["rejected_sql"]
                )
                margin = beta * (logp_chosen - logp_rejected)
                loss_val = -math.log(1.0 / (1.0 + math.exp(-margin)))
                dpo_losses.append(round(loss_val, 4))
                reward_margins.append(round(margin, 4))

        avg_dpo_loss = (
            round(sum(dpo_losses) / len(dpo_losses), 4) if dpo_losses else 0.0
        )
        avg_reward_margin = (
            round(sum(reward_margins) / len(reward_margins), 4)
            if reward_margins
            else 0.0
        )
        res_runner: dict[str, JobMetricValue] = {
            "status": "dry_run_completed" if dry_run else "completed",
            "backend": "local-cpu",
            "model_name": model_name,
            "samples_processed": len(pairs),
            "dpo_pairs": len(pairs),
            "beta": beta,
            "dpo_loss": avg_dpo_loss,
            "reward_margin": avg_reward_margin,
            "output_dir": str(out_p.resolve()),
        }
        if not dry_run:
            res_runner["checkpoint_path"] = str(ckpt_dir.resolve())
        return {
            "status": "completed",
            "execution_mode": self.execution_mode,
            "result": res_runner,
        }

    def evaluate_sql(
        self,
        model_name: str,
        tc_p: Path,
        db_p: Path,
        timeout: float = 300.0,
    ) -> EvalRunResult:
        """
        Execute native in-process SQL evaluation comparing normalized SQL and DuckDB result sets.

        Args:
            model_name: Unused evaluated model name for strategy parity.
            tc_p: Validated test cases JSONL path.
            db_p: Validated DuckDB database path.
            timeout: Unused execution timeout parameter for strategy parity.

        Returns:
            EvalRunResult: Structured evaluation result with exact_match and execution_accuracy.

        """
        import duckdb

        _ = (model_name, timeout)
        cases: list[dict[str, str]] = []
        with open(tc_p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        cases.append({str(k): str(v) for k, v in obj.items()})
                except json.JSONDecodeError as e:
                    logger.debug(f"Invalid JSON in test cases: {e}")

        if not cases:
            return {
                "status": "completed",
                "execution_mode": self.execution_mode,
                "accuracy": {
                    "exact_match": 0.0,
                    "execution_accuracy": 0.0,
                    "total": 0,
                },
            }

        exact_matches = 0
        exec_matches = 0
        conn = duckdb.connect(str(db_p), read_only=True)
        try:
            for case in cases:
                gold_sql = str(
                    case.get("gold_sql")
                    or case.get("sql")
                    or case.get("completion")
                    or ""
                ).strip()
                pred_sql = str(
                    case.get("predicted_sql")
                    or case.get("pred_sql")
                    or case.get("prediction")
                    or ""
                ).strip()

                if not gold_sql or not pred_sql:
                    continue

                if normalize_sql(gold_sql) == normalize_sql(pred_sql):
                    exact_matches += 1

                try:
                    gold_res = conn.execute(gold_sql).fetchall()
                    pred_res = conn.execute(pred_sql).fetchall()
                    if gold_res == pred_res:
                        exec_matches += 1
                except duckdb.Error as e:
                    logger.debug(f"Execution failed during evaluation: {e}")
                except (OSError, RuntimeError, ValueError, TypeError) as e:
                    logger.debug(f"Unexpected error during evaluation: {e}")
        finally:
            conn.close()

        total = len(cases)
        return {
            "status": "completed",
            "execution_mode": self.execution_mode,
            "accuracy": {
                "exact_match": round(exact_matches / total, 4),
                "execution_accuracy": round(exec_matches / total, 4),
                "total": total,
            },
        }
