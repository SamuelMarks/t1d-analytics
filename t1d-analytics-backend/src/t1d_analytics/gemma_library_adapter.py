"""Gemma execution strategy protocol and in-process `gemma_4_sql` library strategy."""

from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Protocol, runtime_checkable

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
    result: dict[str, JobMetricValue]


class DpoRunResult(TypedDict):
    """Structured result dictionary for a DPO alignment run."""

    status: str
    execution_mode: str
    result: dict[str, JobMetricValue]


class EvalAccuracyDict(TypedDict):
    """Structured accuracy summary for SQL evaluation."""

    exact_match: float
    execution_accuracy: float
    total: int


class EvalRunResult(TypedDict):
    """Structured result dictionary for a SQL accuracy evaluation run."""

    status: str
    execution_mode: str
    accuracy: EvalAccuracyDict | float


@runtime_checkable
class GemmaExecutionStrategy(Protocol):
    """Unified execution strategy protocol for Gemma-4-SQL ETL, SFT, DPO, and SQL evaluation."""

    @property
    def execution_mode(self) -> str:
        """Return canonical execution mode identifier ('cli', 'library', 'native_in_process')."""
        ...

    def run_etl(
        self,
        stage_lower: str,
        db_path: Path,
        duckdb_table: str,
        output_dir: str | Path | None = None,
        timeout: float = 300.0,
    ) -> EtlRunResult:
        """Execute an ETL pipeline stage."""
        ...

    def run_sft(
        self,
        model_name: str,
        ds_p: Path,
        out_p: Path,
        hyperparameters: dict[str, int | float | str | bool] | None = None,
        timeout: float = 600.0,
        dry_run: bool = True,
    ) -> SftRunResult:
        """Execute a Supervised Fine-Tuning (SFT) run."""
        ...

    def run_dpo(
        self,
        model_name: str,
        ds_p: Path,
        out_p: Path,
        beta: float = 0.1,
        timeout: float = 600.0,
        dry_run: bool = True,
    ) -> DpoRunResult:
        """Execute a Direct Preference Optimization (DPO) alignment run."""
        ...

    def evaluate_sql(
        self,
        model_name: str,
        tc_p: Path,
        db_p: Path,
        timeout: float = 300.0,
    ) -> EvalRunResult:
        """Evaluate SQL accuracy against DuckDB ground-truth results."""
        ...


@runtime_checkable
class GemmaEtlModuleProtocol(Protocol):
    """Runtime-checkable protocol for a module exposing run_etl."""

    @property
    def run_etl(self) -> Callable[..., EtlPayloadDict]:
        """Return run_etl callable."""
        ...


@runtime_checkable
class GemmaSftModuleProtocol(Protocol):
    """Runtime-checkable protocol for a module exposing run_sft."""

    @property
    def run_sft(self) -> Callable[..., dict[str, JobMetricValue]]:
        """Return run_sft callable."""
        ...


@runtime_checkable
class GemmaDpoModuleProtocol(Protocol):
    """Runtime-checkable protocol for a module exposing run_dpo."""

    @property
    def run_dpo(self) -> Callable[..., dict[str, JobMetricValue]]:
        """Return run_dpo callable."""
        ...


@runtime_checkable
class GemmaEvalModuleProtocol(Protocol):
    """Runtime-checkable protocol for a module exposing evaluate_sql."""

    @property
    def evaluate_sql(
        self,
    ) -> Callable[..., EvalAccuracyDict | float]:
        """Return evaluate_sql callable."""
        ...


class GemmaLibraryProtocol(
    GemmaEtlModuleProtocol,
    GemmaSftModuleProtocol,
    GemmaDpoModuleProtocol,
    GemmaEvalModuleProtocol,
    Protocol,
):
    """Combined structural protocol for gemma_4_sql library hooks."""


class LibraryStrategy:
    """In-process Python library execution strategy wrapping an imported `gemma_4_sql` module."""

    def __init__(self, module: ModuleType | GemmaLibraryProtocol) -> None:
        """
        Initialize the library strategy with a loaded `gemma_4_sql` module.

        Args:
            module: Imported `gemma_4_sql` module or protocol implementation.

        """
        self._module = module

    @property
    def execution_mode(self) -> str:
        """
        Return canonical execution mode identifier.

        Returns:
            str: Always `'library'`.

        """
        return "library"

    def run_etl(
        self,
        stage_lower: str,
        db_path: Path,
        duckdb_table: str,
        output_dir: str | Path | None = None,
        timeout: float = 300.0,
    ) -> EtlRunResult:
        """
        Invoke `gemma_4_sql.run_etl` after verifying callable protocol compliance.

        Args:
            stage_lower: Normalized stage string.
            db_path: Validated DuckDB database path.
            duckdb_table: Source DuckDB table name.
            output_dir: Optional output directory path.
            timeout: Execution timeout parameter in seconds.

        Returns:
            EtlRunResult: Structured library ETL execution result.

        Raises:
            GemmaToolchainError: If `run_etl` is missing or not callable.

        """
        _ = timeout
        if not isinstance(self._module, GemmaEtlModuleProtocol) or not callable(
            self._module.run_etl
        ):
            raise GemmaToolchainError(
                "The 'gemma_4_sql' package does not expose a callable 'run_etl' function."
            )
        payload = self._module.run_etl(
            stage_lower,
            duckdb_path=str(db_path.resolve()),
            duckdb_table=duckdb_table,
            output_dir=str(Path(output_dir).resolve()) if output_dir else None,
        )
        return {
            "status": "completed",
            "execution_mode": self.execution_mode,
            "result": payload,
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
        Invoke `gemma_4_sql.run_sft` after verifying callable protocol compliance.

        Args:
            model_name: Target model name.
            ds_p: Validated dataset path.
            out_p: Validated output directory path.
            hyperparameters: Optional training hyperparameters dictionary.
            timeout: Execution timeout parameter in seconds.
            dry_run: Unused compatibility flag for dry-run parity.

        Returns:
            SftRunResult: Structured library SFT execution result.

        Raises:
            GemmaToolchainError: If `run_sft` is missing or not callable.

        """
        _ = (timeout, dry_run)
        if not isinstance(self._module, GemmaSftModuleProtocol) or not callable(
            self._module.run_sft
        ):
            raise GemmaToolchainError(
                "The 'gemma_4_sql' package does not expose a callable 'run_sft' function."
            )
        metrics = self._module.run_sft(
            model=model_name,
            dataset=str(ds_p.resolve()),
            output_dir=str(out_p.resolve()),
            hyperparameters=hyperparameters or {},
        )
        return {
            "status": "completed",
            "execution_mode": self.execution_mode,
            "result": metrics,
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
        Invoke `gemma_4_sql.run_dpo` after verifying callable protocol compliance.

        Args:
            model_name: Target model name.
            ds_p: Validated preference dataset path.
            out_p: Validated output directory path.
            beta: DPO beta temperature parameter.
            timeout: Execution timeout parameter in seconds.
            dry_run: Unused compatibility flag for dry-run parity.

        Returns:
            DpoRunResult: Structured library DPO execution result.

        Raises:
            GemmaToolchainError: If `run_dpo` is missing or not callable.

        """
        _ = (timeout, dry_run)
        if not isinstance(self._module, GemmaDpoModuleProtocol) or not callable(
            self._module.run_dpo
        ):
            raise GemmaToolchainError(
                "The 'gemma_4_sql' package does not expose a callable 'run_dpo' function."
            )
        metrics = self._module.run_dpo(
            model=model_name,
            dataset=str(ds_p.resolve()),
            output_dir=str(out_p.resolve()),
            beta=beta,
        )
        return {
            "status": "completed",
            "execution_mode": self.execution_mode,
            "result": metrics,
        }

    def evaluate_sql(
        self,
        model_name: str,
        tc_p: Path,
        db_p: Path,
        timeout: float = 300.0,
    ) -> EvalRunResult:
        """
        Invoke `gemma_4_sql.evaluate_sql` after verifying callable protocol compliance.

        Args:
            model_name: Evaluated model name.
            tc_p: Validated test cases path.
            db_p: Validated DuckDB database path.
            timeout: Execution timeout parameter in seconds.

        Returns:
            EvalRunResult: Structured library SQL evaluation result.

        Raises:
            GemmaToolchainError: If `evaluate_sql` is missing or not callable.

        """
        _ = timeout
        if not isinstance(self._module, GemmaEvalModuleProtocol) or not callable(
            self._module.evaluate_sql
        ):
            raise GemmaToolchainError(
                "The 'gemma_4_sql' package does not expose a callable 'evaluate_sql' function."
            )
        acc = self._module.evaluate_sql(
            model=model_name,
            test_cases=str(tc_p.resolve()),
            duckdb_path=str(db_p.resolve()),
        )
        return {
            "status": "completed",
            "execution_mode": self.execution_mode,
            "accuracy": acc,
        }
