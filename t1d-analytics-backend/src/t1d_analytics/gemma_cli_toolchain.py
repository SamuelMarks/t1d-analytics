"""CLI toolchain discovery and subprocess execution for external gemma-4-sql binaries."""

import importlib.util
import logging
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable, List, Optional, Union

from typing_extensions import TypedDict

from t1d_analytics.errors import (
    ConfigFileNotFoundError,
    DatabaseNotFoundError,
    GemmaToolchainError,
    InvalidStageError,
    SubprocessTimeoutError,
    TrainingExecutionError,
)
from t1d_analytics.i18n import get_translator

logger = logging.getLogger(__name__)

SUPPORTED_STAGES = ("pretrain", "sft", "posttrain")


class GemmaMetricsDict(TypedDict):
    """Parsed step, loss, and learning rate metrics from gemma-4-sql training logs."""

    steps: Optional[int]
    loss: Optional[float]
    learning_rate: Optional[float]


def resolve_local_gemma_sql_repo(
    search_root: Optional[Path] = None,
    register_sys_path: bool = True,
) -> Optional[Path]:
    """
    Locate a sibling `gemma-4-sql/src` directory relative to the repository.

    Checks up to six parent levels from `search_root` (or this module's path) for a sibling
    `gemma-4-sql/src/gemma_4_sql` package, optionally registering it on `sys.path`.

    Args:
    ----
        search_root: Optional starting directory for upward search.
        register_sys_path: Whether to prepend the discovered path onto `sys.path`.

    Returns:
    -------
        Optional[Path]: Resolved path to `gemma-4-sql/src` if found, otherwise None.

    """
    base = (search_root or Path(__file__).resolve()).resolve()
    current = base if base.is_dir() else base.parent
    for _ in range(6):
        candidate_src = current / "gemma-4-sql" / "src"
        if (candidate_src / "gemma_4_sql" / "__init__.py").exists():
            src_str = str(candidate_src)
            if register_sys_path and src_str not in sys.path:
                sys.path.insert(0, src_str)
            return candidate_src
        if current.parent == current:
            break
        current = current.parent
    return None


def _split_gemma_command(gemma_bin: str) -> List[str]:
    """
    Convert a resolved gemma-4-sql binary specification into an argv list safe for paths with spaces.

    Args:
    ----
        gemma_bin: Binary path string or module invocation string.

    Returns:
    -------
        List[str]: Argument vector suitable for subprocess.run.

    """
    if gemma_bin == "python3 -m gemma_4_sql":
        return ["python3", "-m", "gemma_4_sql"]
    return [gemma_bin]


def parse_gemma_sql_metrics(stdout: str) -> GemmaMetricsDict:
    """
    Parse standard output metrics from gemma-4-sql execution logs.

    Extracts step counts, training loss, learning rate, and evaluation metrics.

    Args:
    ----
        stdout: Raw output string captured from gemma-4-sql execution.

    Returns:
    -------
        GemmaMetricsDict: Parsed metrics dictionary containing 'steps', 'loss', and 'learning_rate'.

    """
    metrics: GemmaMetricsDict = {
        "steps": None,
        "loss": None,
        "learning_rate": None,
    }
    for line in stdout.splitlines():
        line = line.strip()
        loss_match = re.search(r"\bloss[=:\s]+([0-9]*\.?[0-9]+)", line, re.IGNORECASE)
        if loss_match:
            metrics["loss"] = float(loss_match.group(1))

        step_match = re.search(r"\bstep[s=:\s]+([0-9]+)", line, re.IGNORECASE)
        if step_match:
            metrics["steps"] = int(step_match.group(1))

        lr_match = re.search(
            r"\b(learning_rate|lr)[=:\s]+([0-9]*\.?[0-9]+(?:e-?[0-9]+)?)",
            line,
            re.IGNORECASE,
        )
        if lr_match:
            metrics["learning_rate"] = float(lr_match.group(2))

    return metrics


def check_gemma_sql_installed() -> bool:
    """
    Check whether the gemma-4-sql CLI binary or Python package is installed.

    Returns
    -------
        bool: True if gemma-4-sql CLI or Python module is discoverable, False otherwise.

    """
    if shutil.which("gemma-4-sql") is not None:
        return True
    return importlib.util.find_spec("gemma_4_sql") is not None


def get_gemma_sql_binary() -> str:
    """
    Resolve the executable binary path for gemma-4-sql.

    Returns
    -------
        str: Name or path of the binary.

    Raises
    ------
        GemmaToolchainError: If gemma-4-sql is not installed or discoverable.

    """
    bin_path = shutil.which("gemma-4-sql")
    if bin_path is not None:
        return bin_path

    if importlib.util.find_spec("gemma_4_sql") is not None:
        return "python3 -m gemma_4_sql"

    raise GemmaToolchainError(
        "gemma-4-sql is not installed in the current environment. "
        "Install it via 'pip install -e /path/to/gemma-4-sql[all]' or ensure 'gemma-4-sql' is in PATH."
    )


class SubprocessGemmaToolchain:
    """Wraps external gemma-4-sql CLI binary discovery and subprocess execution."""

    def __init__(
        self,
        binary_resolver: Callable[[], str] = get_gemma_sql_binary,
    ) -> None:
        """
        Initialize the CLI toolchain wrapper.

        Args:
        ----
            binary_resolver: Callable resolving the `gemma-4-sql` executable path.

        """
        self._binary_resolver = binary_resolver

    def run_etl(
        self,
        stage: str,
        duckdb_path: Union[str, Path],
        duckdb_table: str,
        output_dir: Optional[Union[str, Path]] = None,
        extra_args: Optional[List[str]] = None,
        timeout: float = 300.0,
    ) -> subprocess.CompletedProcess[str]:
        """
        Execute gemma-4-sql ETL ingestion subprocess to convert DuckDB tables into dataset shards.

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

        Raises:
        ------
            InvalidStageError: If stage is invalid.
            DatabaseNotFoundError: If database file is missing.
            SubprocessTimeoutError: If subprocess execution times out.
            TrainingExecutionError: If gemma-4-sql subprocess exits with non-zero code.

        """
        _ = get_translator()
        stage_lower = stage.lower().strip()
        if stage_lower not in SUPPORTED_STAGES:
            raise InvalidStageError(
                f"Invalid ETL stage '{stage}'. Supported stages are: {', '.join(SUPPORTED_STAGES)}"
            )

        db_path = Path(duckdb_path)
        if not db_path.exists():
            raise DatabaseNotFoundError(f"DuckDB database file not found: {db_path}")

        gemma_bin = self._binary_resolver()

        cmd: List[str] = [
            *_split_gemma_command(gemma_bin),
            "etl",
            stage_lower,
            "--duckdb-path",
            str(db_path.resolve()),
            "--duckdb-table",
            duckdb_table,
        ]

        if output_dir is not None:
            out_p = Path(output_dir)
            out_p.mkdir(parents=True, exist_ok=True)
            cmd.extend(["--output-dir", str(out_p.resolve())])

        if extra_args:
            cmd.extend(extra_args)

        logger.info("Executing gemma-4-sql ETL: %s", " ".join(cmd))
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise SubprocessTimeoutError(
                f"gemma-4-sql ETL command timed out after {timeout} seconds."
            ) from exc

        if proc.returncode != 0:
            nl = chr(10)
            raise TrainingExecutionError(
                f"gemma-4-sql ETL command failed with code {proc.returncode}:{nl}{proc.stderr}"
            )

        return proc

    def _run_cancellable_popen(
        self,
        cmd: List[str],
        stage_lower: str,
        timeout: float,
        cancel_event: threading.Event,
    ) -> subprocess.CompletedProcess[str]:
        """
        Execute a training subprocess via `subprocess.Popen` while polling `cancel_event`.

        Args:
        ----
            cmd: Command argument vector.
            stage_lower: Normalized training stage name.
            timeout: Maximum execution timeout in seconds.
            cancel_event: Cooperative cancellation event.

        Returns:
        -------
            subprocess.CompletedProcess[str]: Completed process output.

        Raises:
        ------
            TrainingExecutionError: If cancelled or child process exits non-zero.
            SubprocessTimeoutError: If child process exceeds timeout.

        """
        popen_proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        start = time.time()
        while True:
            if cancel_event.is_set():
                popen_proc.terminate()
                try:
                    popen_proc.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    popen_proc.kill()
                raise TrainingExecutionError(
                    f"gemma-4-sql {stage_lower} training cancelled."
                )
            ret = popen_proc.poll()
            if ret is not None:
                out_str, err_str = popen_proc.communicate()
                return subprocess.CompletedProcess(
                    args=cmd,
                    returncode=ret,
                    stdout=out_str or "",
                    stderr=err_str or "",
                )
            if time.time() - start >= timeout:
                popen_proc.kill()
                raise SubprocessTimeoutError(
                    f"gemma-4-sql {stage_lower} training timed out after {timeout} seconds."
                )
            time.sleep(0.01)

    def run_train(
        self,
        stage: str,
        config_path: Union[str, Path],
        extra_args: Optional[List[str]] = None,
        timeout: float = 600.0,
        cancel_event: Optional[threading.Event] = None,
    ) -> subprocess.CompletedProcess[str]:
        """
        Execute gemma-4-sql training entrypoint subprocess for MaxText or local training pipelines.

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

        Raises:
        ------
            InvalidStageError: If stage is invalid.
            ConfigFileNotFoundError: If configuration file does not exist.
            SubprocessTimeoutError: If training subprocess times out.
            TrainingExecutionError: If training subprocess exits with non-zero code or is cancelled.

        """
        stage_lower = stage.lower().strip()
        if stage_lower not in SUPPORTED_STAGES:
            raise InvalidStageError(
                f"Invalid training stage '{stage}'. Supported stages are: {', '.join(SUPPORTED_STAGES)}"
            )

        cfg_p = Path(config_path)
        if not cfg_p.exists():
            raise ConfigFileNotFoundError(
                f"Training configuration file not found: {cfg_p}"
            )

        gemma_bin = self._binary_resolver()

        cmd: List[str] = [
            *_split_gemma_command(gemma_bin),
            stage_lower,
            "--config",
            str(cfg_p.resolve()),
        ]

        if extra_args:
            cmd.extend(extra_args)

        logger.info("Executing gemma-4-sql training: %s", " ".join(cmd))
        if cancel_event is not None:
            proc = self._run_cancellable_popen(
                cmd=cmd,
                stage_lower=stage_lower,
                timeout=timeout,
                cancel_event=cancel_event,
            )
        else:
            try:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                raise SubprocessTimeoutError(
                    f"gemma-4-sql {stage_lower} training timed out after {timeout} seconds."
                ) from exc

        if proc.returncode != 0:
            nl = chr(10)
            raise TrainingExecutionError(
                f"gemma-4-sql {stage_lower} training failed with code {proc.returncode}:{nl}{proc.stderr}"
            )

        return proc


def run_gemma_sql_etl(
    stage: str,
    duckdb_path: Union[str, Path],
    duckdb_table: str,
    output_dir: Optional[Union[str, Path]] = None,
    extra_args: Optional[List[str]] = None,
    timeout: float = 300.0,
    binary_resolver: Callable[[], str] = get_gemma_sql_binary,
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
        binary_resolver: Callable resolving the `gemma-4-sql` executable path.

    Returns:
    -------
        subprocess.CompletedProcess[str]: Process execution result with stdout and stderr.

    """
    return SubprocessGemmaToolchain(binary_resolver=binary_resolver).run_etl(
        stage=stage,
        duckdb_path=duckdb_path,
        duckdb_table=duckdb_table,
        output_dir=output_dir,
        extra_args=extra_args,
        timeout=timeout,
    )


def run_gemma_sql_train(
    stage: str,
    config_path: Union[str, Path],
    extra_args: Optional[List[str]] = None,
    timeout: float = 600.0,
    cancel_event: Optional[threading.Event] = None,
    binary_resolver: Callable[[], str] = get_gemma_sql_binary,
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
        binary_resolver: Callable resolving the `gemma-4-sql` executable path.

    Returns:
    -------
        subprocess.CompletedProcess[str]: Process execution result.

    """
    return SubprocessGemmaToolchain(binary_resolver=binary_resolver).run_train(
        stage=stage,
        config_path=config_path,
        extra_args=extra_args,
        timeout=timeout,
        cancel_event=cancel_event,
    )
