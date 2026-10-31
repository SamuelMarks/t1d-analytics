"""Typed library adapter and structural protocols for in-process `gemma_4_sql` execution."""

from types import ModuleType
from typing import Callable, Dict, Optional, Protocol, Union, runtime_checkable

from typing_extensions import TypedDict

from t1d_analytics.errors import GemmaToolchainError
from t1d_analytics.job_manager import JobMetricValue


class EtlPayloadDict(TypedDict, total=False):
    """Payload returned by an ETL pipeline execution."""

    table: str
    rows: int
    output_dir: str
    shards: int


class EtlRunResult(TypedDict):
    """Structured result dictionary for an ETL pipeline stage."""

    status: str
    execution_mode: str
    result: EtlPayloadDict


class SftRunResult(TypedDict):
    """Structured result dictionary for an SFT training run."""

    status: str
    execution_mode: str
    result: Dict[str, JobMetricValue]


class DpoRunResult(TypedDict):
    """Structured result dictionary for a DPO alignment run."""

    status: str
    execution_mode: str
    result: Dict[str, JobMetricValue]


class EvalAccuracyDict(TypedDict):
    """Structured accuracy summary for SQL evaluation."""

    exact_match: float
    execution_accuracy: float
    total: int


class EvalRunResult(TypedDict):
    """Structured result dictionary for a SQL accuracy evaluation run."""

    status: str
    execution_mode: str
    accuracy: Union[EvalAccuracyDict, float]


@runtime_checkable
class _GemmaEtlModuleProtocol(Protocol):
    """Runtime-checkable protocol for a module exposing run_etl."""

    @property
    def run_etl(self) -> Callable[..., EtlPayloadDict]:
        """Return run_etl callable."""
        ...


@runtime_checkable
class _GemmaSftModuleProtocol(Protocol):
    """Runtime-checkable protocol for a module exposing run_sft."""

    @property
    def run_sft(self) -> Callable[..., Dict[str, JobMetricValue]]:
        """Return run_sft callable."""
        ...


@runtime_checkable
class _GemmaDpoModuleProtocol(Protocol):
    """Runtime-checkable protocol for a module exposing run_dpo."""

    @property
    def run_dpo(self) -> Callable[..., Dict[str, JobMetricValue]]:
        """Return run_dpo callable."""
        ...


@runtime_checkable
class _GemmaEvalModuleProtocol(Protocol):
    """Runtime-checkable protocol for a module exposing evaluate_sql."""

    @property
    def evaluate_sql(
        self,
    ) -> Callable[..., Union[EvalAccuracyDict, float]]:
        """Return evaluate_sql callable."""
        ...


class _GemmaLibraryProtocol(
    _GemmaEtlModuleProtocol,
    _GemmaSftModuleProtocol,
    _GemmaDpoModuleProtocol,
    _GemmaEvalModuleProtocol,
    Protocol,
):
    """Combined structural protocol for gemma_4_sql library hooks."""


class GemmaLibraryAdapter:
    """Typed adapter wrapping an imported `gemma_4_sql` Python module."""

    def __init__(self, module: Union[ModuleType, _GemmaLibraryProtocol]) -> None:
        """
        Initialize the adapter with a loaded `gemma_4_sql` module.

        Args:
        ----
            module: Imported `gemma_4_sql` module or protocol implementation.

        """
        self._module = module

    def run_etl(
        self,
        stage_lower: str,
        duckdb_path: str,
        duckdb_table: str,
        output_dir: Optional[str],
    ) -> EtlPayloadDict:
        """
        Invoke `gemma_4_sql.run_etl` after verifying callable protocol compliance.

        Args:
        ----
            stage_lower: Normalized stage string.
            duckdb_path: Resolved DuckDB database path.
            duckdb_table: Source DuckDB table name.
            output_dir: Optional resolved output directory path.

        Returns:
        -------
            EtlPayloadDict: Library ETL execution payload.

        Raises:
        ------
            GemmaToolchainError: If `run_etl` is missing or not callable.

        """
        if not isinstance(self._module, _GemmaEtlModuleProtocol) or not callable(
            self._module.run_etl
        ):
            raise GemmaToolchainError(
                "The 'gemma_4_sql' package does not expose a callable 'run_etl' function."
            )
        return self._module.run_etl(
            stage_lower,
            duckdb_path=duckdb_path,
            duckdb_table=duckdb_table,
            output_dir=output_dir,
        )

    def run_sft(
        self,
        model_name: str,
        dataset_path: str,
        output_dir: str,
        hyperparameters: Dict[str, Union[int, float, str, bool]],
    ) -> Dict[str, JobMetricValue]:
        """
        Invoke `gemma_4_sql.run_sft` after verifying callable protocol compliance.

        Args:
        ----
            model_name: Target model name.
            dataset_path: Resolved dataset path.
            output_dir: Resolved output directory path.
            hyperparameters: Training hyperparameters dictionary.

        Returns:
        -------
            Dict[str, JobMetricValue]: Library SFT execution metrics.

        Raises:
        ------
            GemmaToolchainError: If `run_sft` is missing or not callable.

        """
        if not isinstance(self._module, _GemmaSftModuleProtocol) or not callable(
            self._module.run_sft
        ):
            raise GemmaToolchainError(
                "The 'gemma_4_sql' package does not expose a callable 'run_sft' function."
            )
        return self._module.run_sft(
            model=model_name,
            dataset=dataset_path,
            output_dir=output_dir,
            hyperparameters=hyperparameters,
        )

    def run_dpo(
        self,
        model_name: str,
        dataset_path: str,
        output_dir: str,
        beta: float,
    ) -> Dict[str, JobMetricValue]:
        """
        Invoke `gemma_4_sql.run_dpo` after verifying callable protocol compliance.

        Args:
        ----
            model_name: Target model name.
            dataset_path: Resolved preference dataset path.
            output_dir: Resolved output directory path.
            beta: DPO beta temperature parameter.

        Returns:
        -------
            Dict[str, JobMetricValue]: Library DPO execution metrics.

        Raises:
        ------
            GemmaToolchainError: If `run_dpo` is missing or not callable.

        """
        if not isinstance(self._module, _GemmaDpoModuleProtocol) or not callable(
            self._module.run_dpo
        ):
            raise GemmaToolchainError(
                "The 'gemma_4_sql' package does not expose a callable 'run_dpo' function."
            )
        return self._module.run_dpo(
            model=model_name,
            dataset=dataset_path,
            output_dir=output_dir,
            beta=beta,
        )

    def evaluate_sql(
        self,
        model_name: str,
        test_cases_path: str,
        duckdb_path: str,
    ) -> Union[EvalAccuracyDict, float]:
        """
        Invoke `gemma_4_sql.evaluate_sql` after verifying callable protocol compliance.

        Args:
        ----
            model_name: Evaluated model name.
            test_cases_path: Resolved test cases path.
            duckdb_path: Resolved DuckDB database path.

        Returns:
        -------
            Union[EvalAccuracyDict, float]: Library evaluation accuracy result.

        Raises:
        ------
            GemmaToolchainError: If `evaluate_sql` is missing or not callable.

        """
        if not isinstance(self._module, _GemmaEvalModuleProtocol) or not callable(
            self._module.evaluate_sql
        ):
            raise GemmaToolchainError(
                "The 'gemma_4_sql' package does not expose a callable 'evaluate_sql' function."
            )
        return self._module.evaluate_sql(
            model=model_name,
            test_cases=test_cases_path,
            duckdb_path=duckdb_path,
        )
