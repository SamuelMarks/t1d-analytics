"""Bridge interface connecting T1D Analytics data pipelines to external gemma-4-sql toolchains."""

import importlib
import subprocess
import threading
from pathlib import Path
from types import ModuleType
from typing import Dict, List, Optional, Sequence, Union

from t1d_analytics.api_schemas import TableSchemaModel
from t1d_analytics.errors import (
    DatabaseNotFoundError,
    GemmaToolchainError,
    InvalidStageError,
)
from t1d_analytics.gemma_cli_toolchain import (
    SUPPORTED_STAGES as SUPPORTED_STAGES,
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
    run_gemma_sql_etl as _run_gemma_sql_etl_impl,
)
from t1d_analytics.gemma_cli_toolchain import (
    run_gemma_sql_train as _run_gemma_sql_train_impl,
)
from t1d_analytics.gemma_inprocess_fallback import (
    InProcessDuckDbGemmaFallback as InProcessDuckDbGemmaFallback,
)
from t1d_analytics.gemma_inprocess_fallback import (
    score_sql_completion_logprob as score_sql_completion_logprob,
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
    GemmaLibraryAdapter as GemmaLibraryAdapter,
)
from t1d_analytics.gemma_library_adapter import (
    SftRunResult as SftRunResult,
)
from t1d_analytics.gemma_library_adapter import (
    _GemmaDpoModuleProtocol as _GemmaDpoModuleProtocol,
)
from t1d_analytics.gemma_library_adapter import (
    _GemmaEtlModuleProtocol as _GemmaEtlModuleProtocol,
)
from t1d_analytics.gemma_library_adapter import (
    _GemmaEvalModuleProtocol as _GemmaEvalModuleProtocol,
)
from t1d_analytics.gemma_library_adapter import (
    _GemmaLibraryProtocol as _GemmaLibraryProtocol,
)
from t1d_analytics.gemma_library_adapter import (
    _GemmaSftModuleProtocol as _GemmaSftModuleProtocol,
)
from t1d_analytics.sql_agent import (
    AgenticSqlResult as AgenticSqlResult,
)
from t1d_analytics.sql_agent import (
    _classify_sql_error_hint as _classify_sql_error_hint,
)
from t1d_analytics.sql_agent import (
    _extract_schema_map_from_text as _extract_schema_map_from_text,
)
from t1d_analytics.sql_agent import (
    execute_agentic_nl_to_sql as execute_agentic_nl_to_sql,
)
from t1d_analytics.sql_agent import (
    prune_schema_with_gemma_rag as _prune_schema_with_gemma_rag_impl,
)
from t1d_analytics.sql_agent import (
    validate_sql_tokens as _validate_sql_tokens_impl,
)


def prune_schema_with_gemma_rag(
    nl_query: str,
    full_schema_text: Union[str, Sequence[TableSchemaModel]],
    top_k_tables: int = 4,
) -> str:
    """
    Augment or prune a database schema description using `gemma_4_sql.sdk.rag.retrieve_relevant_schema`.

    Args:
    ----
        nl_query: Natural language clinical question.
        full_schema_text: Full database schema text or sequence of structured `TableSchemaModel` objects.
        top_k_tables: Maximum number of top-ranked tables to highlight.

    Returns:
    -------
        str: Schema string enriched with prioritized relevant table context.

    """
    return _prune_schema_with_gemma_rag_impl(
        nl_query=nl_query,
        full_schema_text=full_schema_text,
        top_k_tables=top_k_tables,
        repo_resolver=resolve_local_gemma_sql_repo,
    )


def validate_sql_tokens(sql: str) -> List[int]:
    """
    Tokenize a SQL query string using `gemma_4_sql.tokenization.SQLTokenizer`.

    Falls back to deterministic UTF-8 byte-level token IDs if `gemma_4_sql` is not installed.

    Args:
    ----
        sql: SQL query string to tokenize.

    Returns:
    -------
        List[int]: List of integer token IDs.

    """
    return _validate_sql_tokens_impl(sql, repo_resolver=resolve_local_gemma_sql_repo)


def run_gemma_sql_etl(
    stage: str,
    duckdb_path: Union[str, Path],
    duckdb_table: str,
    output_dir: Optional[Union[str, Path]] = None,
    extra_args: Optional[List[str]] = None,
    timeout: float = 300.0,
) -> subprocess.CompletedProcess[str]:
    """
    Execute gemma-4-sql ETL ingestion to convert DuckDB tables into Grain/MaxText dataset shards.

    Args:
    ----
        stage: Pipeline stage, one of ('pretrain', 'sft', 'posttrain').
        duckdb_path: Path to source DuckDB database containing clinical/training pairs.
        duckdb_table: Name of database table to ingest.
        output_dir: Optional destination directory for output shards.
        extra_args: Optional list of additional command line flags to append.
        timeout: Maximum execution timeout in seconds.

    Returns:
    -------
        subprocess.CompletedProcess[str]: Process execution result with stdout and stderr.

    """
    return _run_gemma_sql_etl_impl(
        stage=stage,
        duckdb_path=duckdb_path,
        duckdb_table=duckdb_table,
        output_dir=output_dir,
        extra_args=extra_args,
        timeout=timeout,
        binary_resolver=lambda: get_gemma_sql_binary(),
    )


def run_gemma_sql_train(
    stage: str,
    config_path: Union[str, Path],
    extra_args: Optional[List[str]] = None,
    timeout: float = 600.0,
    cancel_event: Optional[threading.Event] = None,
) -> subprocess.CompletedProcess[str]:
    """
    Execute gemma-4-sql training entrypoint for MaxText or local training pipelines.

    Args:
    ----
        stage: Pipeline training stage, one of ('pretrain', 'sft', 'posttrain').
        config_path: Path to configuration YAML or training parameter file.
        extra_args: Optional list of additional CLI flags.
        timeout: Maximum execution timeout in seconds.
        cancel_event: Optional cooperative cancellation event.

    Returns:
    -------
        subprocess.CompletedProcess[str]: Process execution result.

    """
    return _run_gemma_sql_train_impl(
        stage=stage,
        config_path=config_path,
        extra_args=extra_args,
        timeout=timeout,
        cancel_event=cancel_event,
        binary_resolver=lambda: get_gemma_sql_binary(),
    )


class GemmaSqlPipelineEngine:
    """
    Direct Python programmatic engine for gemma-4-sql data preparation and model training.

    Executes ETL, SFT, DPO, and evaluation directly via the in-process 'gemma_4_sql' Python package,
    delegating to `InProcessDuckDbGemmaFallback` when running in standalone mode.
    """

    def __init__(self, allow_native_fallback: bool = False) -> None:
        """
        Initialize the pipeline engine and load the gemma_4_sql Python package.

        Args:
        ----
            allow_native_fallback: Whether to allow in-process native DuckDB execution
                when external gemma_4_sql package is not installed.

        Raises:
        ------
            GemmaToolchainError: If gemma_4_sql package is not installed and allow_native_fallback is False.

        """
        self._fallback = InProcessDuckDbGemmaFallback()
        try:
            mod: ModuleType = importlib.import_module("gemma_4_sql")
            self._adapter: Optional[GemmaLibraryAdapter] = GemmaLibraryAdapter(mod)
        except ImportError as e:
            if not allow_native_fallback:
                raise GemmaToolchainError(
                    "The 'gemma_4_sql' Python package is required. "
                    "Install via 'pip install -e /path/to/gemma-4-sql[all]'."
                ) from e
            self._adapter = None

    @property
    def is_library_available(self) -> bool:
        """
        Check if the direct in-process gemma_4_sql Python package is available.

        Returns
        -------
            bool: True if Python module is imported, False otherwise.

        """
        return self._adapter is not None

    def run_etl(
        self,
        stage: str,
        duckdb_path: Union[str, Path],
        duckdb_table: str,
        output_dir: Optional[Union[str, Path]] = None,
        timeout: float = 300.0,
    ) -> EtlRunResult:
        """
        Execute ETL pipeline stage directly via the in-process gemma_4_sql module or native fallback.

        Args:
        ----
            stage: Pipeline stage ('pretrain', 'sft', 'posttrain').
            duckdb_path: Path to input DuckDB database.
            duckdb_table: Name of database table to ingest.
            output_dir: Optional destination directory for output dataset shards.
            timeout: Execution timeout parameter for compatibility.

        Returns:
        -------
            EtlRunResult: Execution status, output path, and stage metadata.

        Raises:
        ------
            DatabaseNotFoundError: If the DuckDB database does not exist.
            InvalidStageError: If the stage is invalid.
            GemmaToolchainError: If execution fails or module lacks run_etl.

        """
        stage_lower = stage.lower().strip()
        if stage_lower not in SUPPORTED_STAGES:
            raise InvalidStageError(
                f"Invalid ETL stage '{stage}'. Supported stages are: {', '.join(SUPPORTED_STAGES)}"
            )

        db_path = Path(duckdb_path)
        if not db_path.exists():
            raise DatabaseNotFoundError(f"DuckDB database file not found: {db_path}")

        if self._adapter is not None:
            res = self._adapter.run_etl(
                stage_lower=stage_lower,
                duckdb_path=str(db_path.resolve()),
                duckdb_table=duckdb_table,
                output_dir=str(Path(output_dir).resolve()) if output_dir else None,
            )
            return {
                "status": "completed",
                "execution_mode": "library",
                "result": res,
            }

        return self._fallback.run_etl(
            stage_lower=stage_lower,
            db_path=db_path,
            duckdb_table=duckdb_table,
            output_dir=output_dir,
        )

    def run_sft(
        self,
        model_name: str,
        dataset_path: Union[str, Path],
        output_dir: Union[str, Path],
        hyperparameters: Optional[Dict[str, Union[int, float, str, bool]]] = None,
        timeout: float = 600.0,
        dry_run: bool = True,
    ) -> SftRunResult:
        """
        Execute supervised fine-tuning (SFT) workflow via native gemma_4_sql package or fallback.

        Args:
        ----
            model_name: Base model name.
            dataset_path: Path to formatted training dataset.
            output_dir: Output checkpoint directory.
            hyperparameters: Optional dictionary of training hyperparameters.
            timeout: Execution timeout parameter for compatibility.
            dry_run: Whether to execute a dry-run validation pass in native fallback mode.

        Returns:
        -------
            SftRunResult: SFT training results and parsed metrics.

        Raises:
        ------
            DatabaseNotFoundError: If dataset path does not exist.
            GemmaToolchainError: If training fails or module lacks run_sft.

        """
        ds_p = Path(dataset_path)
        if not ds_p.exists():
            raise DatabaseNotFoundError(f"Training dataset not found: {ds_p}")

        out_p = Path(output_dir)
        out_p.mkdir(parents=True, exist_ok=True)

        if self._adapter is not None:
            res = self._adapter.run_sft(
                model_name=model_name,
                dataset_path=str(ds_p.resolve()),
                output_dir=str(out_p.resolve()),
                hyperparameters=hyperparameters or {},
            )
            return {
                "status": "completed",
                "execution_mode": "library",
                "result": res,
            }

        return self._fallback.run_sft(
            model_name=model_name,
            ds_p=ds_p,
            out_p=out_p,
            hyperparameters=hyperparameters,
            dry_run=dry_run,
        )

    def run_dpo(
        self,
        model_name: str,
        preference_dataset_path: Union[str, Path],
        output_dir: Union[str, Path],
        beta: float = 0.1,
        timeout: float = 600.0,
        dry_run: bool = True,
    ) -> DpoRunResult:
        """
        Execute Direct Preference Optimization (DPO) alignment workflow via gemma_4_sql package or fallback.

        Args:
        ----
            model_name: Base model name.
            preference_dataset_path: Path to preference dataset containing prompt/chosen/rejected.
            output_dir: Destination checkpoint directory.
            beta: DPO temperature hyperparameter.
            timeout: Execution timeout parameter for compatibility.
            dry_run: Whether to execute a dry-run validation pass in native fallback mode.

        Returns:
        -------
            DpoRunResult: DPO training run metrics and status.

        Raises:
        ------
            DatabaseNotFoundError: If preference dataset file does not exist.
            GemmaToolchainError: If training fails or module lacks run_dpo.

        """
        ds_p = Path(preference_dataset_path)
        if not ds_p.exists():
            raise DatabaseNotFoundError(f"Preference dataset not found: {ds_p}")

        out_p = Path(output_dir)
        out_p.mkdir(parents=True, exist_ok=True)

        if self._adapter is not None:
            res = self._adapter.run_dpo(
                model_name=model_name,
                dataset_path=str(ds_p.resolve()),
                output_dir=str(out_p.resolve()),
                beta=beta,
            )
            return {
                "status": "completed",
                "execution_mode": "library",
                "result": res,
            }

        return self._fallback.run_dpo(
            model_name=model_name,
            ds_p=ds_p,
            out_p=out_p,
            beta=beta,
            dry_run=dry_run,
        )

    def evaluate_sql(
        self,
        model_name: str,
        test_cases_path: Union[str, Path],
        duckdb_path: Union[str, Path],
        timeout: float = 300.0,
    ) -> EvalRunResult:
        """
        Execute evaluation suite testing model SQL accuracy directly via gemma_4_sql package or fallback.

        Args:
        ----
            model_name: Evaluated model name or path.
            test_cases_path: Path to evaluation test cases file (.jsonl).
            duckdb_path: Path to validation DuckDB database.
            timeout: Execution timeout parameter for compatibility.

        Returns:
        -------
            EvalRunResult: Execution accuracy, semantic pass rate, and metrics.

        Raises:
        ------
            DatabaseNotFoundError: If test cases or DuckDB file does not exist.
            GemmaToolchainError: If evaluation fails or module lacks evaluate_sql.

        """
        tc_p = Path(test_cases_path)
        if not tc_p.exists():
            raise DatabaseNotFoundError(f"Test cases file not found: {tc_p}")

        db_p = Path(duckdb_path)
        if not db_p.exists():
            raise DatabaseNotFoundError(f"DuckDB database file not found: {db_p}")

        if self._adapter is not None:
            res = self._adapter.evaluate_sql(
                model_name=model_name,
                test_cases_path=str(tc_p.resolve()),
                duckdb_path=str(db_p.resolve()),
            )
            return {
                "status": "completed",
                "execution_mode": "library",
                "accuracy": res,
            }

        return self._fallback.evaluate_sql(tc_p=tc_p, db_p=db_p)
