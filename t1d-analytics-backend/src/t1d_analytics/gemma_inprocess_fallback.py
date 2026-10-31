"""Native in-process DuckDB and PyTorch fallback engine for Gemma-4-SQL ETL, SFT, DPO, and evaluation."""

import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from t1d_analytics.errors import InvalidIdentifierError
from t1d_analytics.gemma_cli_toolchain import resolve_local_gemma_sql_repo
from t1d_analytics.gemma_library_adapter import (
    DpoRunResult,
    EtlRunResult,
    EvalRunResult,
    SftRunResult,
)
from t1d_analytics.sql_agent import (
    score_sql_completion_logprob as _score_sql_completion_logprob_impl,
)
from t1d_analytics.sql_utils import escape_sql_identifier, normalize_sql

logger = logging.getLogger(__name__)


def score_sql_completion_logprob(prompt: str, sql_text: str) -> float:
    """
    Compute a structural and lexical log-probability score for a candidate SQL completion.

    Args:
    ----
        prompt: Natural language question or prompt string.
        sql_text: Candidate SQL completion string.

    Returns:
    -------
        float: Estimated log-probability score (higher is better, <= 0.0).

    """
    return _score_sql_completion_logprob_impl(
        prompt=prompt,
        sql_text=sql_text,
        repo_resolver=resolve_local_gemma_sql_repo,
    )


class InProcessDuckDbGemmaFallback:
    """Native in-process DuckDB fallback engine for ETL, SFT, DPO, and SQL evaluation."""

    def run_etl(
        self,
        stage_lower: str,
        db_path: Path,
        duckdb_table: str,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> EtlRunResult:
        """
        Execute native in-process DuckDB ETL exporting JSONL and Parquet shards.

        Args:
        ----
            stage_lower: Normalized stage identifier ('pretrain', 'sft', 'posttrain').
            db_path: Validated path to the source DuckDB database.
            duckdb_table: Source table identifier.
            output_dir: Optional output directory path.

        Returns:
        -------
            EtlRunResult: Structured ETL execution result.

        Raises:
        ------
            InvalidIdentifierError: If `duckdb_table` is not a valid SQL identifier.

        """
        import duckdb

        if not duckdb_table.isidentifier():
            raise InvalidIdentifierError(
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
                    rec = dict(zip(cols, r))
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
                        + chr(10)
                    )

            parquet_file = out_p / f"{stage_lower}_data.parquet"
            escaped_pq = str(parquet_file.resolve()).replace("'", "''")
            conn.execute(
                f"COPY (SELECT * FROM \"{escaped_table}\") TO '{escaped_pq}' (FORMAT PARQUET)"
            )
        finally:
            conn.close()

        return {
            "status": "completed",
            "execution_mode": "native_in_process",
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
        hyperparameters: Optional[Dict[str, Union[int, float, str, bool]]] = None,
        dry_run: bool = True,
    ) -> SftRunResult:
        """
        Execute native in-process SFT training fallback via LocalCpuRunner.

        Args:
        ----
            model_name: Target model name.
            ds_p: Validated dataset path.
            out_p: Validated output directory path.
            hyperparameters: Optional hyperparameter dictionary.
            dry_run: Whether to run in dry-run validation mode.

        Returns:
        -------
            SftRunResult: Structured SFT execution result.

        """
        from t1d_analytics.models import (
            TrainingBackend,
            TrainingHyperparameters,
            TrainingJobConfig,
        )
        from t1d_analytics.training_runner import get_training_runner

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
            "execution_mode": "native_in_process",
            "result": res_runner,
        }

    @staticmethod
    def _optimize_dpo_preference_tensors(
        pairs: List[Dict[str, str]],
        beta: float,
        num_steps: int = 1,
    ) -> Tuple[List[float], List[float]]:
        """
        Execute a genuine PyTorch Bradley-Terry preference optimization pass over `(prompt, chosen_sql, rejected_sql)`.

        Computes token-level log-probabilities for chosen and rejected SQL completions under a
        causal LM and minimizes `-log(sigmoid(beta * (logp_chosen - logp_rejected)))`.

        Args:
        ----
            pairs: Preference dataset triplets with `'prompt'`, `'chosen_sql'`, and `'rejected_sql'`.
            beta: DPO temperature hyperparameter.
            num_steps: Number of optimization epochs over the preference pairs.

        Returns:
        -------
            Tuple[List[float], List[float]]: `(dpo_losses, reward_margins)` per preference pair step.

        """
        import torch

        from t1d_analytics.dataset_loader import _tokenize_pair_with_masking
        from t1d_analytics.hf_models import (
            TinyCausalLMFactory,
            TokenizerProtocol,
            _extract_output_logits,
        )

        class _ByteTokenizer(TokenizerProtocol):
            """Lightweight byte-level tokenizer for in-process DPO optimization."""

            def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
                """Encode UTF-8 text into byte token IDs in `[1, 999]`."""
                _ = add_special_tokens
                return [min(999, max(1, b)) for b in text.encode("utf-8")]

        tokenizer = _ByteTokenizer()
        factory = TinyCausalLMFactory()
        model = factory.create_model("tiny-dpo", vocab_size=1000, embed_dim=32)
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)

        losses: List[float] = []
        margins: List[float] = []
        for _ in range(max(1, num_steps)):
            for pair in pairs:
                c_in, c_lbl = _tokenize_pair_with_masking(
                    pair["prompt"], pair["chosen_sql"], tokenizer, max_seq_length=256
                )
                r_in, r_lbl = _tokenize_pair_with_masking(
                    pair["prompt"],
                    pair["rejected_sql"] or "SELECT 0",
                    tokenizer,
                    max_seq_length=256,
                )

                optimizer.zero_grad()
                c_logits = _extract_output_logits(model(c_in.unsqueeze(0))).view(
                    -1, 1000
                )
                r_logits = _extract_output_logits(model(r_in.unsqueeze(0))).view(
                    -1, 1000
                )

                c_token_nll = loss_fn(c_logits, c_lbl.view(-1))
                r_token_nll = loss_fn(r_logits, r_lbl.view(-1))
                logp_chosen = -float(c_token_nll.item())
                logp_rejected = -float(r_token_nll.item())
                margin_val = beta * (logp_chosen - logp_rejected)
                sigmoid_neg_margin = 1.0 / (1.0 + math.exp(margin_val))
                dpo_grad_scale = max(1e-6, beta * sigmoid_neg_margin)
                (c_token_nll / (1.0 / dpo_grad_scale)).backward()
                ((-r_token_nll) / (1.0 / dpo_grad_scale)).backward()
                optimizer.step()

                dpo_loss_val = -math.log(1.0 / (1.0 + math.exp(-margin_val)))

                losses.append(round(dpo_loss_val, 4))
                margins.append(round(margin_val, 4))

        return losses, margins

    def run_dpo(
        self,
        model_name: str,
        ds_p: Path,
        out_p: Path,
        beta: float = 0.1,
        dry_run: bool = True,
    ) -> DpoRunResult:
        """
        Execute native in-process DPO alignment fallback computing Bradley-Terry log-margins.

        Args:
        ----
            model_name: Target model name.
            ds_p: Validated preference dataset path.
            out_p: Validated output directory path.
            beta: DPO temperature hyperparameter.
            dry_run: Whether to run in dry-run validation mode.

        Returns:
        -------
            DpoRunResult: Structured DPO execution result.

        """
        from t1d_analytics.models import (
            TrainingBackend,
            TrainingHyperparameters,
            TrainingJobConfig,
        )
        from t1d_analytics.training_runner import get_training_runner

        pairs: List[Dict[str, str]] = []
        with open(ds_p, "r", encoding="utf-8") as f:
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

        if not dry_run and pairs:
            dpo_losses, reward_margins = self._optimize_dpo_preference_tensors(
                pairs, beta
            )
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

        runner = get_training_runner(TrainingBackend.LOCAL_CPU)
        train_cfg = TrainingJobConfig(
            model_name=model_name,
            backend=TrainingBackend.LOCAL_CPU,
            dataset_path=str(ds_p.resolve()),
            output_dir=str(out_p.resolve()),
            dry_run=dry_run,
            use_tiny_model=not dry_run,
            hyperparameters=TrainingHyperparameters(
                learning_rate=1e-6,
                num_epochs=1,
            ),
        )
        res_runner = runner.run_training(train_cfg)
        res_runner["dpo_pairs"] = len(pairs)
        res_runner["beta"] = beta
        res_runner["dpo_loss"] = (
            round(sum(dpo_losses) / len(dpo_losses), 4) if dpo_losses else 0.0
        )
        res_runner["reward_margin"] = (
            round(sum(reward_margins) / len(reward_margins), 4)
            if reward_margins
            else 0.0
        )
        return {
            "status": "completed",
            "execution_mode": "native_in_process",
            "result": res_runner,
        }

    def evaluate_sql(
        self,
        tc_p: Path,
        db_p: Path,
    ) -> EvalRunResult:
        """
        Execute native in-process SQL evaluation comparing normalized SQL and DuckDB result sets.

        Args:
        ----
            tc_p: Validated test cases JSONL path.
            db_p: Validated DuckDB database path.

        Returns:
        -------
            EvalRunResult: Structured evaluation result with exact_match and execution_accuracy.

        """
        import duckdb

        cases: List[Dict[str, str]] = []
        with open(tc_p, "r", encoding="utf-8") as f:
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
                "execution_mode": "native_in_process",
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
                except Exception as e:
                    logger.debug(f"Unexpected error during evaluation: {e}")
        finally:
            conn.close()

        total = len(cases)
        return {
            "status": "completed",
            "execution_mode": "native_in_process",
            "accuracy": {
                "exact_match": round(exact_matches / total, 4),
                "execution_accuracy": round(exec_matches / total, 4),
                "total": total,
            },
        }
