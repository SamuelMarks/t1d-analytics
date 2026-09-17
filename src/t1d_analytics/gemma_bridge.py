"""Bridge interface connecting T1D Analytics data pipelines to external gemma-4-sql toolchains."""

import importlib.util
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from t1d_analytics.i18n import get_translator

logger = logging.getLogger(__name__)

SUPPORTED_STAGES = ("pretrain", "sft", "posttrain")


def parse_gemma_sql_metrics(stdout: str) -> Dict[str, Any]:
    """
    Parse standard output metrics from gemma-4-sql execution logs.

    Extracts step counts, training loss, learning rate, and evaluation metrics.

    Args:
    ----
        stdout: Raw output string captured from gemma-4-sql execution.

    Returns:
    -------
        Dict[str, Any]: Parsed metrics dictionary containing 'steps', 'loss', and 'learning_rate'.

    """
    import re

    metrics: Dict[str, Any] = {
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
    if os.environ.get("T1D_MOCK_GEMMA_SQL") == "1":
        return True
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
        RuntimeError: If gemma-4-sql is not installed or discoverable.

    """
    if os.environ.get("T1D_MOCK_GEMMA_SQL") == "1":
        return "gemma-4-sql"

    bin_path = shutil.which("gemma-4-sql")
    if bin_path is not None:
        return bin_path

    if importlib.util.find_spec("gemma_4_sql") is not None:
        return "python3 -m gemma_4_sql"

    raise RuntimeError(
        "gemma-4-sql is not installed in the current environment. "
        "Install it via 'pip install -e /path/to/gemma-4-sql[all]' or ensure 'gemma-4-sql' is in PATH."
    )


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

    Raises:
    ------
        ValueError: If stage is invalid or database file is missing.
        RuntimeError: If gemma-4-sql is not installed or execution fails.

    """
    _ = get_translator()
    stage_lower = stage.lower().strip()
    if stage_lower not in SUPPORTED_STAGES:
        raise ValueError(
            f"Invalid ETL stage '{stage}'. Supported stages are: {', '.join(SUPPORTED_STAGES)}"
        )

    db_path = Path(duckdb_path)
    if not db_path.exists():
        raise FileNotFoundError(f"DuckDB database file not found: {db_path}")

    gemma_bin = get_gemma_sql_binary()

    cmd: List[str] = [
        *gemma_bin.split(),
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

    if os.environ.get("T1D_MOCK_GEMMA_SQL") == "1":
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=0,
            stdout=f"[MOCK] Executed gemma-4-sql etl {stage_lower} for {duckdb_table}\n",
            stderr="",
        )

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
        raise TimeoutError(
            f"gemma-4-sql ETL command timed out after {timeout} seconds."
        ) from exc

    if proc.returncode != 0:
        raise RuntimeError(
            f"gemma-4-sql ETL command failed with code {proc.returncode}:\n{proc.stderr}"
        )

    return proc


def run_gemma_sql_train(
    stage: str,
    config_path: Union[str, Path],
    extra_args: Optional[List[str]] = None,
    timeout: float = 600.0,
) -> subprocess.CompletedProcess[str]:
    """
    Execute gemma-4-sql training entrypoint for MaxText or local training pipelines.

    Args:
    ----
        stage: Pipeline training stage, one of ('pretrain', 'sft', 'posttrain').
        config_path: Path to configuration YAML or training parameter file.
        extra_args: Optional list of additional CLI flags.
        timeout: Maximum execution timeout in seconds.

    Returns:
    -------
        subprocess.CompletedProcess[str]: Process execution result.

    Raises:
    ------
        ValueError: If stage is invalid or configuration file does not exist.
        RuntimeError: If gemma-4-sql is not installed or execution fails.

    """
    stage_lower = stage.lower().strip()
    if stage_lower not in SUPPORTED_STAGES:
        raise ValueError(
            f"Invalid training stage '{stage}'. Supported stages are: {', '.join(SUPPORTED_STAGES)}"
        )

    cfg_p = Path(config_path)
    if not cfg_p.exists():
        raise FileNotFoundError(f"Training configuration file not found: {cfg_p}")

    gemma_bin = get_gemma_sql_binary()

    cmd: List[str] = [
        *gemma_bin.split(),
        stage_lower,
        "--config",
        str(cfg_p.resolve()),
    ]

    if extra_args:
        cmd.extend(extra_args)

    if os.environ.get("T1D_MOCK_GEMMA_SQL") == "1":
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=0,
            stdout=f"[MOCK] Executed gemma-4-sql {stage_lower} with config {config_path}\n",
            stderr="",
        )

    logger.info("Executing gemma-4-sql training: %s", " ".join(cmd))
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(
            f"gemma-4-sql {stage_lower} training timed out after {timeout} seconds."
        ) from exc

    if proc.returncode != 0:
        raise RuntimeError(
            f"gemma-4-sql {stage_lower} training failed with code {proc.returncode}:\n{proc.stderr}"
        )

    return proc


class GemmaSqlPipelineEngine:
    """
    Direct Python programmatic engine for gemma-4-sql data preparation and model training.

    Executes ETL, SFT, DPO, and evaluation directly via the in-process 'gemma_4_sql' Python package.
    """

    def __init__(self) -> None:
        """
        Initialize the pipeline engine and load the gemma_4_sql Python package.

        Raises
        ------
            RuntimeError: If gemma_4_sql package is not installed.

        """
        try:
            import importlib

            self._library_module = importlib.import_module("gemma_4_sql")
        except ImportError as e:
            raise RuntimeError(
                "The 'gemma_4_sql' Python package is required. "
                "Install via 'pip install -e /path/to/gemma-4-sql[all]'."
            ) from e

    @property
    def is_library_available(self) -> bool:
        """
        Check if the direct in-process gemma_4_sql Python package is available.

        Returns
        -------
            bool: True if Python module is imported, False otherwise.

        """
        return self._library_module is not None

    def run_etl(
        self,
        stage: str,
        duckdb_path: Union[str, Path],
        duckdb_table: str,
        output_dir: Optional[Union[str, Path]] = None,
        timeout: float = 300.0,
    ) -> Dict[str, Any]:
        """
        Execute ETL pipeline stage directly via the in-process gemma_4_sql module.

        Args:
        ----
            stage: Pipeline stage ('pretrain', 'sft', 'posttrain').
            duckdb_path: Path to input DuckDB database.
            duckdb_table: Name of database table to ingest.
            output_dir: Optional destination directory for output dataset shards.
            timeout: Execution timeout parameter for compatibility.

        Returns:
        -------
            Dict[str, Any]: Execution status, output path, and stage metadata.

        Raises:
        ------
            FileNotFoundError: If the DuckDB database does not exist.
            ValueError: If the stage is invalid.
            RuntimeError: If execution fails or module lacks run_etl.

        """
        stage_lower = stage.lower().strip()
        if stage_lower not in SUPPORTED_STAGES:
            raise ValueError(
                f"Invalid ETL stage '{stage}'. Supported stages are: {', '.join(SUPPORTED_STAGES)}"
            )

        db_path = Path(duckdb_path)
        if not db_path.exists():
            raise FileNotFoundError(f"DuckDB database file not found: {db_path}")

        fn = getattr(self._library_module, "run_etl", None)
        if not callable(fn):
            raise RuntimeError(
                "The 'gemma_4_sql' package does not expose a callable 'run_etl' function."
            )

        res = fn(
            stage_lower,
            duckdb_path=str(db_path.resolve()),
            duckdb_table=duckdb_table,
            output_dir=str(Path(output_dir).resolve()) if output_dir else None,
        )
        return {
            "status": "completed",
            "execution_mode": "library",
            "result": res,
        }

    def run_sft(
        self,
        model_name: str,
        dataset_path: Union[str, Path],
        output_dir: Union[str, Path],
        hyperparameters: Optional[Dict[str, Any]] = None,
        timeout: float = 600.0,
    ) -> Dict[str, Any]:
        """
        Execute supervised fine-tuning (SFT) workflow via native gemma_4_sql package.

        Args:
        ----
            model_name: Base model name.
            dataset_path: Path to formatted training dataset.
            output_dir: Output checkpoint directory.
            hyperparameters: Optional dictionary of training hyperparameters.
            timeout: Execution timeout parameter for compatibility.

        Returns:
        -------
            Dict[str, Any]: SFT training results and parsed metrics.

        Raises:
        ------
            FileNotFoundError: If dataset path does not exist.
            RuntimeError: If training fails or module lacks run_sft.

        """
        ds_p = Path(dataset_path)
        if not ds_p.exists():
            raise FileNotFoundError(f"Training dataset not found: {ds_p}")

        out_p = Path(output_dir)
        out_p.mkdir(parents=True, exist_ok=True)

        fn = getattr(self._library_module, "run_sft", None)
        if not callable(fn):
            raise RuntimeError(
                "The 'gemma_4_sql' package does not expose a callable 'run_sft' function."
            )

        res = fn(
            model=model_name,
            dataset=str(ds_p.resolve()),
            output_dir=str(out_p.resolve()),
            hyperparameters=hyperparameters or {},
        )
        return {
            "status": "completed",
            "execution_mode": "library",
            "result": res,
        }

    def run_dpo(
        self,
        model_name: str,
        preference_dataset_path: Union[str, Path],
        output_dir: Union[str, Path],
        beta: float = 0.1,
        timeout: float = 600.0,
    ) -> Dict[str, Any]:
        """
        Execute Direct Preference Optimization (DPO) alignment workflow via native gemma_4_sql package.

        Args:
        ----
            model_name: Base model name.
            preference_dataset_path: Path to preference dataset containing prompt/chosen/rejected.
            output_dir: Destination checkpoint directory.
            beta: DPO temperature hyperparameter.
            timeout: Execution timeout parameter for compatibility.

        Returns:
        -------
            Dict[str, Any]: DPO training run metrics and status.

        Raises:
        ------
            FileNotFoundError: If preference dataset file does not exist.
            RuntimeError: If training fails or module lacks run_dpo.

        """
        ds_p = Path(preference_dataset_path)
        if not ds_p.exists():
            raise FileNotFoundError(f"Preference dataset not found: {ds_p}")

        out_p = Path(output_dir)
        out_p.mkdir(parents=True, exist_ok=True)

        fn = getattr(self._library_module, "run_dpo", None)
        if not callable(fn):
            raise RuntimeError(
                "The 'gemma_4_sql' package does not expose a callable 'run_dpo' function."
            )

        res = fn(
            model=model_name,
            dataset=str(ds_p.resolve()),
            output_dir=str(out_p.resolve()),
            beta=beta,
        )
        return {
            "status": "completed",
            "execution_mode": "library",
            "result": res,
        }

    def evaluate_sql(
        self,
        model_name: str,
        test_cases_path: Union[str, Path],
        duckdb_path: Union[str, Path],
        timeout: float = 300.0,
    ) -> Dict[str, Any]:
        """
        Execute evaluation suite testing model SQL accuracy directly via gemma_4_sql package.

        Args:
        ----
            model_name: Evaluated model name or path.
            test_cases_path: Path to evaluation test cases file (.jsonl).
            duckdb_path: Path to validation DuckDB database.
            timeout: Execution timeout parameter for compatibility.

        Returns:
        -------
            Dict[str, Any]: Execution accuracy, semantic pass rate, and metrics.

        Raises:
        ------
            FileNotFoundError: If test cases or DuckDB file does not exist.
            RuntimeError: If evaluation fails or module lacks evaluate_sql.

        """
        tc_p = Path(test_cases_path)
        if not tc_p.exists():
            raise FileNotFoundError(f"Test cases file not found: {tc_p}")

        db_p = Path(duckdb_path)
        if not db_p.exists():
            raise FileNotFoundError(f"DuckDB database file not found: {db_p}")

        fn = getattr(self._library_module, "evaluate_sql", None)
        if not callable(fn):
            raise RuntimeError(
                "The 'gemma_4_sql' package does not expose a callable 'evaluate_sql' function."
            )

        res = fn(
            model=model_name,
            test_cases=str(tc_p.resolve()),
            duckdb_path=str(db_p.resolve()),
        )
        return {
            "status": "completed",
            "execution_mode": "library",
            "accuracy": res,
        }
