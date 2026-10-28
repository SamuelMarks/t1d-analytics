"""Bridge interface connecting T1D Analytics data pipelines to external gemma-4-sql toolchains."""

import importlib
import importlib.util
import json
import logging
import math
import re
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import (
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Sequence,
    Union,
    runtime_checkable,
)

from typing_extensions import TypedDict

from t1d_analytics.api_schemas import TableSchemaModel
from t1d_analytics.errors import (
    ConfigFileNotFoundError,
    DatabaseNotFoundError,
    GemmaToolchainError,
    InvalidIdentifierError,
    InvalidStageError,
    SubprocessTimeoutError,
    TrainingExecutionError,
)
from t1d_analytics.i18n import get_translator
from t1d_analytics.job_manager import JobMetricValue
from t1d_analytics.sql_agent import (
    AgenticSqlResult as AgenticSqlResult,
)
from t1d_analytics.sql_agent import (
    _classify_sql_error_hint as _classify_sql_error_hint,
)
from t1d_analytics.sql_agent import (
    execute_agentic_nl_to_sql as execute_agentic_nl_to_sql,
)
from t1d_analytics.sql_agent import (
    validate_sql_tokens as _validate_sql_tokens_impl,
)
from t1d_analytics.sql_utils import (
    is_read_only_sql_query,
    is_sql_query,
    normalize_sql,
)

logger = logging.getLogger(__name__)

SUPPORTED_STAGES = ("pretrain", "sft", "posttrain")


class GemmaMetricsDict(TypedDict):
    """Parsed step, loss, and learning rate metrics from gemma-4-sql training logs."""

    steps: Optional[int]
    loss: Optional[float]
    learning_rate: Optional[float]


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
        raise NotImplementedError


@runtime_checkable
class _GemmaSftModuleProtocol(Protocol):
    """Runtime-checkable protocol for a module exposing run_sft."""

    @property
    def run_sft(self) -> Callable[..., Dict[str, JobMetricValue]]:
        """Return run_sft callable."""
        raise NotImplementedError


@runtime_checkable
class _GemmaDpoModuleProtocol(Protocol):
    """Runtime-checkable protocol for a module exposing run_dpo."""

    @property
    def run_dpo(self) -> Callable[..., Dict[str, JobMetricValue]]:
        """Return run_dpo callable."""
        raise NotImplementedError


@runtime_checkable
class _GemmaEvalModuleProtocol(Protocol):
    """Runtime-checkable protocol for a module exposing evaluate_sql."""

    @property
    def evaluate_sql(
        self,
    ) -> Callable[..., Union[EvalAccuracyDict, float]]:
        """Return evaluate_sql callable."""
        raise NotImplementedError


class _GemmaLibraryProtocol(
    _GemmaEtlModuleProtocol,
    _GemmaSftModuleProtocol,
    _GemmaDpoModuleProtocol,
    _GemmaEvalModuleProtocol,
    Protocol,
):
    """Combined structural protocol for gemma_4_sql library hooks."""


def resolve_local_gemma_sql_repo(
    search_root: Optional[Path] = None,
) -> Optional[Path]:
    """
    Locate a sibling `gemma-4-sql/src` directory relative to the repository and register it on `sys.path`.

    Checks up to four parent levels from `search_root` (or this module's path) for a sibling
    `gemma-4-sql/src/gemma_4_sql` package.

    Args:
    ----
        search_root: Optional starting directory for upward search.

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
            if src_str not in sys.path:
                sys.path.insert(0, src_str)
            return candidate_src
        if current.parent == current:
            break
        current = current.parent
    return None


def _extract_schema_map_from_text(
    schema_source: Union[str, Sequence[TableSchemaModel]],
) -> Dict[str, List[str]]:
    """
    Extract table-to-columns mapping from structured `TableSchemaModel` objects or schema text/DDL.

    Args:
    ----
        schema_source: Sequence of `TableSchemaModel` objects, or a formatted schema/DDL string.

    Returns:
    -------
        Dict[str, List[str]]: Mapping of table names to their column name lists.

    """
    if not isinstance(schema_source, str):
        return {
            tbl.table_name: [col.name for col in tbl.columns] for tbl in schema_source
        }

    schema_text = schema_source
    schema_map: Dict[str, List[str]] = {}
    current_table: Optional[str] = None
    for raw_line in schema_text.splitlines():
        line = raw_line.strip()
        if line.startswith("--"):
            line = line.lstrip("-").strip()
        if line.startswith("Table:"):
            after_table = line.split("Table:", 1)[1].strip()
            if "|" in after_table:
                table_part, rest_part = after_table.split("|", 1)
                t_inline = table_part.strip()
                if t_inline:
                    cols_inline = schema_map.setdefault(t_inline, [])
                    if "Columns:" in rest_part:
                        cols_str = rest_part.split("Columns:", 1)[1].strip()
                        for raw_col in cols_str.split(","):
                            c_name = raw_col.split("(", 1)[0].strip()
                            if c_name and c_name not in cols_inline:
                                cols_inline.append(c_name)
                current_table = None
            else:
                current_table = after_table
                if current_table:
                    schema_map.setdefault(current_table, [])
        elif line.startswith("Columns:") and current_table is not None:
            cols_str = line.split("Columns:", 1)[1].strip()
            for raw_col in cols_str.split(","):
                col_name = raw_col.split("(", 1)[0].strip()
                if col_name and col_name not in schema_map[current_table]:
                    schema_map[current_table].append(col_name)
        elif line.startswith("-") and current_table is not None:
            col_part = line.lstrip("-").strip()
            col_name = col_part.split("(", 1)[0].strip()
            if col_name and col_name not in schema_map[current_table]:
                schema_map[current_table].append(col_name)

    if schema_map:
        return schema_map

    ddl_pattern = re.compile(
        r'CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?(?:"([^"]+)"|([a-zA-Z0-9_]+))\s*\((.*?)\)',
        re.IGNORECASE | re.DOTALL,
    )
    for match in ddl_pattern.finditer(schema_text):
        t_name = match.group(1) or match.group(2)
        cols_body = match.group(3)
        cols: List[str] = []
        for col_def in cols_body.split(","):
            c_clean = col_def.strip()
            if not c_clean:
                continue
            first_tok = c_clean.split()[0].strip('"`[]')
            if first_tok.upper() not in (
                "PRIMARY",
                "FOREIGN",
                "CONSTRAINT",
                "UNIQUE",
                "CHECK",
            ):
                cols.append(first_tok)
        schema_map[t_name] = cols
    return schema_map


def prune_schema_with_gemma_rag(
    nl_query: str,
    full_schema_text: Union[str, Sequence[TableSchemaModel]],
    top_k_tables: int = 4,
) -> str:
    """
    Augment or prune a database schema description using `gemma_4_sql.sdk.rag.retrieve_relevant_schema`.

    Uses `gemma_4_sql`'s RAG retrieval (or BM25/keyword ranking fallback) to prepend
    a prioritized `-- Relevant Schema Context:` header while preserving table details.

    Args:
    ----
        nl_query: Natural language clinical question.
        full_schema_text: Full database schema text or sequence of structured `TableSchemaModel` objects.
        top_k_tables: Maximum number of top-ranked tables to highlight.

    Returns:
    -------
        str: Schema string enriched with prioritized relevant table context.

    """
    nl = chr(10)
    rendered_schema = (
        full_schema_text
        if isinstance(full_schema_text, str)
        else f"{nl}{nl}".join(tbl.to_prompt_block() for tbl in full_schema_text)
    )
    if not rendered_schema.strip() or not nl_query.strip():
        return rendered_schema

    schema_map = _extract_schema_map_from_text(full_schema_text)
    if not schema_map:
        return rendered_schema

    resolve_local_gemma_sql_repo()
    try:
        rag_mod = importlib.import_module("gemma_4_sql.sdk.rag")
        retrieve_fn = rag_mod.retrieve_relevant_schema
        rag_header = str(retrieve_fn(nl_query, schema_map, top_k_tables=top_k_tables))
        if rag_header.strip():
            return f"{rag_header}{nl}{nl}{rendered_schema}"
    except Exception as e:
        logger.debug("Falling back to built-in BM25 schema ranking: %s", e)

    prompt_words = {w.lower() for w in re.findall(r"\b\w+\b", nl_query)}
    scores: Dict[str, int] = {}
    for table, cols in schema_map.items():
        score = 5 if table.lower() in prompt_words else 0
        for col in cols:
            if col.lower() in prompt_words:
                score += 1
        scores[table] = score

    ranked = [
        t
        for t, s in sorted(scores.items(), key=lambda item: item[1], reverse=True)[
            :top_k_tables
        ]
        if s > 0
    ]
    if not ranked:
        ranked = list(schema_map.keys())[:top_k_tables]

    lines = ["-- Relevant Schema Context:"]
    for t in ranked:
        lines.append(f"-- Table: {t} | Columns: {', '.join(schema_map[t])}")
    return nl.join(lines) + f"{nl}{nl}" + rendered_schema


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
            TpuPollingTimeoutError: If subprocess execution times out.
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

        gemma_bin = get_gemma_sql_binary()

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

    def run_train(
        self,
        stage: str,
        config_path: Union[str, Path],
        extra_args: Optional[List[str]] = None,
        timeout: float = 600.0,
    ) -> subprocess.CompletedProcess[str]:
        """
        Execute gemma-4-sql training entrypoint subprocess for MaxText or local training pipelines.

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
            InvalidStageError: If stage is invalid.
            DatabaseNotFoundError: If configuration file does not exist.
            TpuPollingTimeoutError: If training subprocess times out.
            TrainingExecutionError: If training subprocess exits with non-zero code.

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

        gemma_bin = get_gemma_sql_binary()

        cmd: List[str] = [
            *_split_gemma_command(gemma_bin),
            stage_lower,
            "--config",
            str(cfg_p.resolve()),
        ]

        if extra_args:
            cmd.extend(extra_args)

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
    return SubprocessGemmaToolchain().run_etl(
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

    """
    return SubprocessGemmaToolchain().run_train(
        stage=stage,
        config_path=config_path,
        extra_args=extra_args,
        timeout=timeout,
    )


def score_sql_completion_logprob(prompt: str, sql_text: str) -> float:
    """
    Compute a structural and lexical log-probability score for a candidate SQL completion.

    Evaluates read-only SQL validity, balanced parentheses, token count via `validate_sql_tokens`,
    and lexical alignment with the natural language prompt.

    Args:
    ----
        prompt: Natural language question or prompt string.
        sql_text: Candidate SQL completion string.

    Returns:
    -------
        float: Estimated log-probability score (higher is better, <= 0.0).

    """
    cleaned = sql_text.strip()
    if not cleaned:
        return -2.5

    token_ids = validate_sql_tokens(cleaned)
    token_penalty = 0.05 * math.log1p(len(token_ids))
    structural_penalty = 0.0
    if not is_sql_query(cleaned) or not is_read_only_sql_query(cleaned):
        structural_penalty += 0.8
    if cleaned.count("(") != cleaned.count(")"):
        structural_penalty += 0.5
    if "from" not in cleaned.lower():
        structural_penalty += 0.3

    prompt_words = {w.lower() for w in re.findall(r"\b[a-zA-Z_]\w+\b", prompt)}
    sql_words = {w.lower() for w in re.findall(r"\b[a-zA-Z_]\w+\b", cleaned)}
    overlap = len(prompt_words.intersection(sql_words))
    alignment_bonus = min(0.25, 0.05 * overlap)

    return min(-0.01, -0.15 - token_penalty - structural_penalty + alignment_bonus)


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

        from t1d_analytics.sql_utils import escape_sql_identifier

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

        dpo_losses: List[float] = []
        reward_margins: List[float] = []
        for pair in pairs:
            logp_chosen = score_sql_completion_logprob(
                pair["prompt"], pair["chosen_sql"]
            )
            logp_rejected = (
                score_sql_completion_logprob(pair["prompt"], pair["rejected_sql"]) - 0.2
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


GemmaSqlBridge = GemmaSqlPipelineEngine
