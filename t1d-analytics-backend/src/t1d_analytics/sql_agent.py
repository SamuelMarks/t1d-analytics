"""SQL self-correction, error hint classification, schema RAG pruning, and SQL token validation utilities."""

import importlib
import logging
import math
import re
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Set, Union

import duckdb
from typing_extensions import TypedDict

from t1d_analytics.api_schemas import ColumnInfo, TableSchemaModel
from t1d_analytics.errors import T1DError
from t1d_analytics.sql_utils import (
    SqlValue,
    coerce_numeric_aggregates_in_sql,
    extract_sql_from_response,
    is_read_only_sql_query,
    is_sql_query,
    parse_schema_text_to_column_pairs,
    parse_schema_text_to_map,
)

logger = logging.getLogger(__name__)


class AgenticSqlResult(TypedDict):
    """Structured result from the self-correcting agentic NL-to-SQL execution loop."""

    success: bool
    final_sql: str
    attempts: int
    rows: List[Dict[str, SqlValue]]
    error: Optional[str]


def parse_schema_text_to_models(schema_text: str) -> List[TableSchemaModel]:
    """
    Parse a formatted schema string or SQL DDL into structured `TableSchemaModel` instances.

    Args:
    ----
        schema_text: Formatted database schema or SQL DDL string.

    Returns:
    -------
        List[TableSchemaModel]: Ordered list of parsed `TableSchemaModel` instances.

    """
    return [
        TableSchemaModel(
            table_name=tbl,
            columns=[
                ColumnInfo(name=c_name, type=c_type) for c_name, c_type in col_pairs
            ],
        )
        for tbl, col_pairs in parse_schema_text_to_column_pairs(schema_text).items()
    ]


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
    return parse_schema_text_to_map(schema_source)


_TABLE_1_RAG_KEYWORDS = {
    "table 1",
    "table1",
    "baseline",
    "characteristics",
    "stratified",
    "arm",
    "cohort",
    "summary",
    "demographics",
    "outcomes",
    "quartile",
    "iqr",
    "stddev",
}

_CORE_CLINICAL_TABLES = {
    "patients",
    "demographics",
    "subjects",
    "tblaptsummary",
    "cgm_readings",
    "cgm_data",
    "hba1c_labs",
    "hba1c",
    "dka_adverse_events",
    "adverseevents",
}


def _is_table_1_broad_query(nl_query: str) -> bool:
    """
    Determine whether `nl_query` requests a broad Academic Table 1 or baseline cohort summary.

    Args:
    ----
        nl_query: Natural language clinical question.

    Returns:
    -------
        bool: True if the query contains Table 1 or baseline summary keywords.

    """
    lower_q = nl_query.lower()
    if "table 1" in lower_q or "table1" in lower_q or "baseline" in lower_q:
        return True
    query_words = set(re.findall(r"\b\w+\b", lower_q))
    return len(query_words.intersection(_TABLE_1_RAG_KEYWORDS)) >= 2


def _score_tables_by_query_tokens(
    schema_map: Dict[str, List[str]],
    prompt_words: Set[str],
    is_broad_t1: bool,
) -> Dict[str, int]:
    """
    Score each table in `schema_map` by lexical overlap with `prompt_words` and clinical priority.

    Args:
    ----
        schema_map: Mapping of table names to column names.
        prompt_words: Lowercase words extracted from the user prompt.
        is_broad_t1: Whether the prompt requests a broad Table 1 summary.

    Returns:
    -------
        Dict[str, int]: Mapping of table name to integer relevance score.

    """
    scores: Dict[str, int] = {}
    for table, cols in schema_map.items():
        score = 5 if table.lower() in prompt_words else 0
        if is_broad_t1 and table.lower() in _CORE_CLINICAL_TABLES:
            score += 3
        score += sum(1 for col in cols if col.lower() in prompt_words)
        scores[table] = score
    return scores


def _expand_related_clinical_tables(
    schema_map: Dict[str, List[str]],
    scores: Dict[str, int],
    effective_top_k: int,
) -> List[str]:
    """
    Select the top-k highest-scoring tables from `scores` or fall back to the first `effective_top_k` tables.

    Args:
    ----
        schema_map: Mapping of table names to column names.
        scores: Table relevance score mapping.
        effective_top_k: Maximum number of tables to include.

    Returns:
    -------
        List[str]: Ordered list of selected table names.

    """
    ranked = [
        t
        for t, s in sorted(scores.items(), key=lambda item: item[1], reverse=True)[
            :effective_top_k
        ]
        if s > 0
    ]
    return ranked if ranked else list(schema_map.keys())[:effective_top_k]


def _fetch_sample_rows_for_tables(
    ranked_tables: Sequence[str],
    schema_map: Dict[str, List[str]],
    rendered_schema: str,
) -> str:
    """
    Format the `-- Relevant Schema Context:` header block and prepend it to `rendered_schema`.

    Args:
    ----
        ranked_tables: Ordered list of prioritized table names.
        schema_map: Mapping of table names to column names.
        rendered_schema: Full rendered schema text including sample rows.

    Returns:
    -------
        str: Enriched schema string.

    """
    nl = chr(10)
    lines = ["-- Relevant Schema Context:"]
    for t in ranked_tables:
        lines.append(f"-- Table: {t} | Columns: {', '.join(schema_map[t])}")
    return nl.join(lines) + f"{nl}{nl}" + rendered_schema


def prune_schema_with_gemma_rag(
    nl_query: str,
    full_schema_text: Union[str, Sequence[TableSchemaModel]],
    top_k_tables: int = 4,
    repo_resolver: Optional[Callable[[], Optional[Path]]] = None,
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
        repo_resolver: Optional callback to register local `gemma-4-sql` repo on `sys.path`.

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

    is_broad_t1 = _is_table_1_broad_query(nl_query)
    effective_top_k = (
        max(top_k_tables, len(schema_map)) if is_broad_t1 else top_k_tables
    )

    if repo_resolver is not None:
        repo_resolver()
    try:
        rag_mod = importlib.import_module("gemma_4_sql.sdk.rag")
        retrieve_fn = rag_mod.retrieve_relevant_schema
        rag_header = str(
            retrieve_fn(nl_query, schema_map, top_k_tables=effective_top_k)
        )
        if rag_header.strip():
            return f"{rag_header}{nl}{nl}{rendered_schema}"
    except (
        ImportError,
        SyntaxError,
        AttributeError,
        OSError,
        RuntimeError,
        ValueError,
        TypeError,
    ) as e:
        logger.debug("Falling back to built-in BM25 schema ranking: %s", e)

    prompt_words = {w.lower() for w in re.findall(r"\b\w+\b", nl_query)}
    scores = _score_tables_by_query_tokens(schema_map, prompt_words, is_broad_t1)
    ranked = _expand_related_clinical_tables(schema_map, scores, effective_top_k)
    return _fetch_sample_rows_for_tables(ranked, schema_map, rendered_schema)


def score_sql_completion_logprob(
    prompt: str,
    sql_text: str,
    repo_resolver: Optional[Callable[[], Optional[Path]]] = None,
) -> float:
    """
    Compute a structural and lexical log-probability score for a candidate SQL completion.

    Evaluates read-only SQL validity, balanced parentheses, token count via `validate_sql_tokens`,
    and lexical alignment with the natural language prompt.

    Args:
    ----
        prompt: Natural language question or prompt string.
        sql_text: Candidate SQL completion string.
        repo_resolver: Optional callback to register local `gemma-4-sql` repo on `sys.path`.

    Returns:
    -------
        float: Estimated log-probability score (higher is better, <= 0.0).

    """
    cleaned = sql_text.strip()
    if not cleaned:
        return -2.5

    token_ids = validate_sql_tokens(cleaned, repo_resolver=repo_resolver)
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


score_sql_completion_heuristic = score_sql_completion_logprob


def validate_sql_tokens(
    sql: str,
    repo_resolver: Optional[Callable[[], Optional[Path]]] = None,
) -> List[int]:
    """
    Tokenize a SQL query string using `gemma_4_sql.tokenization.SQLTokenizer`.

    Falls back to deterministic UTF-8 byte-level token IDs if `gemma_4_sql` is not installed.

    Args:
    ----
        sql: SQL query string to tokenize.
        repo_resolver: Optional callback to register local `gemma-4-sql` repo on `sys.path`.

    Returns:
    -------
        List[int]: List of integer token IDs.

    """
    if repo_resolver is not None:
        repo_resolver()
    try:
        tok_mod = importlib.import_module("gemma_4_sql.tokenization")
        tokenizer = tok_mod.SQLTokenizer()
        encoded = tokenizer.encode(sql)
        if isinstance(encoded, list):
            return [int(x) for x in encoded if isinstance(x, int)]
    except (
        ImportError,
        SyntaxError,
        AttributeError,
        OSError,
        RuntimeError,
        ValueError,
        TypeError,
    ) as e:
        logger.debug("Falling back to byte-level SQL tokenization: %s", e)
    return list(sql.encode("utf-8", errors="replace"))


def _classify_sql_error_hint(error_msg: str) -> str:
    """
    Classify a DuckDB execution error into an actionable schema repair hint.

    Args:
    ----
        error_msg: Raw DuckDB error string.

    Returns:
    -------
        str: Diagnostic hint suffix string.

    """
    if re.search(
        r"(?:sum\(varchar\)|avg\(varchar\)|stddev_samp\(varchar\)|quantile_cont\(varchar|no function matches the given name and argument types)",
        error_msg,
        re.IGNORECASE,
    ):
        return (
            " [Hint: Numeric aggregation failed because the target column is stored as VARCHAR "
            "(often due to 'NA' strings in clinical CSVs). Wrap the column in "
            "TRY_CAST(column_name AS DOUBLE) inside SUM(), AVG(), STDDEV_SAMP(), QUANTILE_CONT(), or MEDIAN(), "
            "e.g. SUM(TRY_CAST(NumSevHypo AS DOUBLE))]"
        )
    if re.search(
        r"(?:set operations can only apply to expressions with the same|type mismatch.*union|cannot mix values of type)",
        error_msg,
        re.IGNORECASE,
    ):
        return (
            " [Hint: Transposed Table 1 UNION ALL branches have mismatched column types or counts. "
            "Explicitly wrap formatted value columns in CAST(... AS VARCHAR) across every UNION ALL branch]"
        )
    if re.search(
        r"(?:must appear in the group by clause|rollup|grouping sets)",
        error_msg,
        re.IGNORECASE,
    ):
        return (
            " [Hint: When using GROUP BY ROLLUP(col), reference the raw column inside COALESCE(col, 'Overall Cohort') "
            "and ensure all non-aggregated SELECT columns appear in GROUP BY]"
        )
    if re.search(
        r"(?:no such column|column .* does not exist|Referenced column .* not found)",
        error_msg,
        re.IGNORECASE,
    ):
        return " [Hint: check valid column names in the schema]"
    if re.search(
        r"(?:no such table|Table with name .* does not exist|relation .* does not exist)",
        error_msg,
        re.IGNORECASE,
    ):
        return " [Hint: check available table names in the schema]"
    return ""


classify_sql_error = _classify_sql_error_hint
classify_duckdb_error_hint = _classify_sql_error_hint


def execute_agentic_nl_to_sql(
    db_path: str,
    nl_query: str,
    generate_fn: Callable[[str], str],
    execute_fn: Callable[[str, str], List[Dict[str, SqlValue]]],
    max_retries: int = 2,
) -> AgenticSqlResult:
    """
    Execute a self-correcting Text-to-SQL loop inspired by `gemma_4_sql.sdk.agent.run_agentic_loop`.

    Generates candidate SQL via `generate_fn`, executes it against `db_path` via `execute_fn`,
    and if execution fails, feeds the DuckDB error and diagnostic hint back into `generate_fn`
    up to `max_retries` attempts.

    Args:
    ----
        db_path: Path to the DuckDB database file.
        nl_query: Initial natural language prompt.
        generate_fn: Callable accepting a prompt string and returning raw LLM response or SQL.
        execute_fn: Callable accepting `(db_path, sql_query)` and returning result rows.
        max_retries: Maximum number of generation + execution attempts.

    Returns:
    -------
        AgenticSqlResult: Structured dictionary with `success`, `final_sql`, `attempts`, `rows`, and `error`.

    """
    current_prompt = nl_query
    attempts = 0
    final_sql = ""
    last_error: Optional[str] = None
    total_budget = max(1, max_retries)

    while attempts < total_budget:
        attempts += 1
        raw_output = generate_fn(current_prompt)
        extracted_sql = extract_sql_from_response(raw_output) or raw_output.strip()
        candidate_sql = coerce_numeric_aggregates_in_sql(extracted_sql)
        final_sql = candidate_sql
        try:
            rows = execute_fn(db_path, candidate_sql)
            return {
                "success": True,
                "final_sql": final_sql,
                "attempts": attempts,
                "rows": rows,
                "error": None,
            }
        except (T1DError, duckdb.Error, OSError, RuntimeError, ValueError) as exc:
            nl = chr(10)
            err_str = str(exc)
            last_error = err_str
            hint = _classify_sql_error_hint(err_str)
            current_prompt = (
                f"{nl_query}{nl}{nl}"
                f"Previous SQL attempt:{nl}{candidate_sql}{nl}"
                f"Failed with database error: {err_str}{hint}{nl}"
                f"Please fix the DuckDB SQL query using only valid tables and columns from the schema."
            )

    return {
        "success": False,
        "final_sql": final_sql,
        "attempts": attempts,
        "rows": [],
        "error": last_error,
    }
