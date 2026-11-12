"""Strategy-backed Gemma-4-SQL pipeline engine coordinating CLI, Python library, and native DuckDB strategies."""

import importlib
from pathlib import Path
from types import ModuleType

from t1d_analytics.errors import (
    DatabaseNotFoundError,
    GemmaToolchainError,
    InvalidStageError,
)
from t1d_analytics.gemma_cli_toolchain import (
    SUPPORTED_STAGES as SUPPORTED_STAGES,
)
from t1d_analytics.gemma_cli_toolchain import (
    CliStrategy as CliStrategy,
)
from t1d_analytics.gemma_cli_toolchain import (
    GemmaMetricsDict as GemmaMetricsDict,
)
from t1d_analytics.gemma_cli_toolchain import (
    SubprocessGemmaToolchain as SubprocessGemmaToolchain,
)
from t1d_analytics.gemma_cli_toolchain import (
    check_gemma_sql_installed as check_gemma_sql_installed,
)
from t1d_analytics.gemma_cli_toolchain import (
    get_gemma_sql_binary as get_gemma_sql_binary,
)
from t1d_analytics.gemma_cli_toolchain import (
    parse_gemma_sql_metrics as parse_gemma_sql_metrics,
)
from t1d_analytics.gemma_cli_toolchain import (
    resolve_local_gemma_sql_repo as resolve_local_gemma_sql_repo,
)
from t1d_analytics.gemma_cli_toolchain import (
    run_gemma_sql_etl as run_gemma_sql_etl,
)
from t1d_analytics.gemma_cli_toolchain import (
    run_gemma_sql_train as run_gemma_sql_train,
)
from t1d_analytics.gemma_inprocess_fallback import (
    NativeDuckDbStrategy as NativeDuckDbStrategy,
)
from t1d_analytics.gemma_library_adapter import (
    DpoRunResult as DpoRunResult,
)
from t1d_analytics.gemma_library_adapter import (
    EtlPayloadDict as EtlPayloadDict,
)
from t1d_analytics.gemma_library_adapter import (
    EtlRunResult as EtlRunResult,
)
from t1d_analytics.gemma_library_adapter import (
    EvalAccuracyDict as EvalAccuracyDict,
)
from t1d_analytics.gemma_library_adapter import (
    EvalRunResult as EvalRunResult,
)
from t1d_analytics.gemma_library_adapter import (
    GemmaExecutionStrategy as GemmaExecutionStrategy,
)
from t1d_analytics.gemma_library_adapter import (
    LibraryStrategy as LibraryStrategy,
)
from t1d_analytics.gemma_library_adapter import (
    SftRunResult as SftRunResult,
)


class GemmaSqlPipelineEngine:
    """
    Programmatic pipeline engine delegating ETL, SFT, DPO, and SQL evaluation to a `GemmaExecutionStrategy`.

    Supports explicit injection of `CliStrategy`, `LibraryStrategy`, or `NativeDuckDbStrategy`, or
    auto-selects `LibraryStrategy` when `gemma_4_sql` is importable and `NativeDuckDbStrategy` when
    `allow_native_fallback=True`.
    """

    def __init__(
        self,
        allow_native_fallback: bool = False,
        strategy: GemmaExecutionStrategy | None = None,
    ) -> None:
        """
        Initialize the pipeline engine with an explicit or auto-discovered `GemmaExecutionStrategy`.

        Args:
            allow_native_fallback: Whether to select `NativeDuckDbStrategy` when `gemma_4_sql` is absent.
            strategy: Optional explicit `GemmaExecutionStrategy` implementation (`CliStrategy`,
                `LibraryStrategy`, or `NativeDuckDbStrategy`).

        Raises:
            GemmaToolchainError: If `strategy` is None, `gemma_4_sql` is not installed, and
                `allow_native_fallback` is False.

        """
        if strategy is not None:
            self._strategy: GemmaExecutionStrategy = strategy
            return

        try:
            mod: ModuleType = importlib.import_module("gemma_4_sql")
            self._strategy = LibraryStrategy(mod)
        except ImportError as exc:
            if not allow_native_fallback:
                raise GemmaToolchainError(
                    "The 'gemma_4_sql' Python package is required. "
                    "Install via 'pip install -e /path/to/gemma-4-sql[all]'."
                ) from exc
            self._strategy = NativeDuckDbStrategy()

    @property
    def strategy(self) -> GemmaExecutionStrategy:
        """
        Return the active `GemmaExecutionStrategy` bound to this engine.

        Returns:
            GemmaExecutionStrategy: Active execution strategy instance.

        """
        return self._strategy

    @property
    def is_library_available(self) -> bool:
        """
        Check whether the active strategy executes via the in-process `gemma_4_sql` library.

        Returns:
            bool: True when `strategy.execution_mode == 'library'`, False otherwise.

        """
        return self._strategy.execution_mode == "library"

    def run_etl(
        self,
        stage: str,
        duckdb_path: str | Path,
        duckdb_table: str,
        output_dir: str | Path | None = None,
        timeout: float = 300.0,
    ) -> EtlRunResult:
        """
        Validate inputs and delegate ETL execution to the active `GemmaExecutionStrategy`.

        Args:
            stage: Pipeline stage ('pretrain', 'sft', 'posttrain').
            duckdb_path: Path to input DuckDB database.
            duckdb_table: Name of database table to ingest.
            output_dir: Optional destination directory for output dataset shards.
            timeout: Execution timeout in seconds.

        Returns:
            EtlRunResult: Execution status, output path, and stage metadata.

        Raises:
            InvalidStageError: If `stage` is invalid.
            DatabaseNotFoundError: If `duckdb_path` does not exist.

        """
        stage_lower = stage.lower().strip()
        if stage_lower not in SUPPORTED_STAGES:
            raise InvalidStageError(
                f"Invalid ETL stage '{stage}'. Supported stages are: {', '.join(SUPPORTED_STAGES)}"
            )

        db_path = Path(duckdb_path)
        if not db_path.exists():
            raise DatabaseNotFoundError(f"DuckDB database file not found: {db_path}")

        return self._strategy.run_etl(
            stage_lower=stage_lower,
            db_path=db_path,
            duckdb_table=duckdb_table,
            output_dir=output_dir,
            timeout=timeout,
        )

    def run_sft(
        self,
        model_name: str,
        dataset_path: str | Path,
        output_dir: str | Path,
        hyperparameters: dict[str, int | float | str | bool] | None = None,
        timeout: float = 600.0,
        dry_run: bool = True,
    ) -> SftRunResult:
        """
        Validate inputs and delegate SFT training to the active `GemmaExecutionStrategy`.

        Args:
            model_name: Base model name.
            dataset_path: Path to formatted training dataset.
            output_dir: Output checkpoint directory.
            hyperparameters: Optional dictionary of training hyperparameters.
            timeout: Execution timeout in seconds.
            dry_run: Whether to execute a dry-run validation pass when supported.

        Returns:
            SftRunResult: SFT training results and parsed metrics.

        Raises:
            DatabaseNotFoundError: If `dataset_path` does not exist.

        """
        ds_p = Path(dataset_path)
        if not ds_p.exists():
            raise DatabaseNotFoundError(f"Training dataset not found: {ds_p}")

        out_p = Path(output_dir)
        out_p.mkdir(parents=True, exist_ok=True)

        return self._strategy.run_sft(
            model_name=model_name,
            ds_p=ds_p,
            out_p=out_p,
            hyperparameters=hyperparameters,
            timeout=timeout,
            dry_run=dry_run,
        )

    def run_dpo(
        self,
        model_name: str,
        preference_dataset_path: str | Path,
        output_dir: str | Path,
        beta: float = 0.1,
        timeout: float = 600.0,
        dry_run: bool = True,
    ) -> DpoRunResult:
        """
        Validate inputs and delegate DPO alignment to the active `GemmaExecutionStrategy`.

        Args:
            model_name: Base model name.
            preference_dataset_path: Path to preference dataset containing prompt/chosen/rejected.
            output_dir: Destination checkpoint directory.
            beta: DPO temperature hyperparameter.
            timeout: Execution timeout in seconds.
            dry_run: Whether to execute a dry-run validation pass when supported.

        Returns:
            DpoRunResult: DPO training run metrics and status.

        Raises:
            DatabaseNotFoundError: If `preference_dataset_path` does not exist.

        """
        ds_p = Path(preference_dataset_path)
        if not ds_p.exists():
            raise DatabaseNotFoundError(f"Preference dataset not found: {ds_p}")

        out_p = Path(output_dir)
        out_p.mkdir(parents=True, exist_ok=True)

        return self._strategy.run_dpo(
            model_name=model_name,
            ds_p=ds_p,
            out_p=out_p,
            beta=beta,
            timeout=timeout,
            dry_run=dry_run,
        )

    def evaluate_sql(
        self,
        model_name: str,
        test_cases_path: str | Path,
        duckdb_path: str | Path,
        timeout: float = 300.0,
    ) -> EvalRunResult:
        """
        Validate inputs and delegate SQL evaluation to the active `GemmaExecutionStrategy`.

        Args:
            model_name: Evaluated model name or path.
            test_cases_path: Path to evaluation test cases file (.jsonl).
            duckdb_path: Path to validation DuckDB database.
            timeout: Execution timeout in seconds.

        Returns:
            EvalRunResult: Execution accuracy, semantic pass rate, and metrics.

        Raises:
            DatabaseNotFoundError: If `test_cases_path` or `duckdb_path` does not exist.

        """
        tc_p = Path(test_cases_path)
        if not tc_p.exists():
            raise DatabaseNotFoundError(f"Test cases file not found: {tc_p}")

        db_p = Path(duckdb_path)
        if not db_p.exists():
            raise DatabaseNotFoundError(f"DuckDB database file not found: {db_p}")

        return self._strategy.evaluate_sql(
            model_name=model_name,
            tc_p=tc_p,
            db_p=db_p,
            timeout=timeout,
        )
