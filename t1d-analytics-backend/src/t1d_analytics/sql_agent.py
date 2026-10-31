"""SQL self-correction, error hint classification, schema RAG pruning, and SQL token validation utilities."""

import importlib
import logging
import math
import re
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Union

from typing_extensions import TypedDict

from t1d_analytics.api_schemas import TableSchemaModel
from t1d_analytics.sql_utils import (
    SqlValue,
    extract_sql_from_response,
    is_read_only_sql_query,
    is_sql_query,
)

logger = logging.getLogger(__name__)


class AgenticSqlResult(TypedDict):
    """Structured result from the self-correcting agentic NL-to-SQL execution loop."""

    success: bool
    final_sql: str
    attempts: int
    rows: List[Dict[str, SqlValue]]
    error: Optional[str]


def _append_parsed_columns(cols_list: List[str], cols_str: str) -> None:
    """
    Parse comma-separated column descriptors (`col (TYPE)`) and append unique column names.

    Args:
    ----
        cols_list: Target list of column names to mutate.
        cols_str: Raw comma-separated column specifications.

    """
    for raw_col in cols_str.split(","):
        c_name = raw_col.split("(", 1)[0].strip()
        if c_name and c_name not in cols_list:
            cols_list.append(c_name)


def _extract_ddl_schema_map(schema_text: str) -> Dict[str, List[str]]:
    """
    Extract table-to-columns mapping from `CREATE TABLE` DDL statements.

    Args:
    ----
        schema_text: Raw SQL DDL string.

    Returns:
    -------
        Dict[str, List[str]]: Mapping of table names to column name lists.

    """
    schema_map: Dict[str, List[str]] = {}
    ddl_pattern = re.compile(
        r'CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?(?:"([^"]+)"|([a-zA-Z0-9_]+))\s*\((.*?)\)',
        re.IGNORECASE | re.DOTALL,
    )
    excluded_keywords = {"PRIMARY", "FOREIGN", "CONSTRAINT", "UNIQUE", "CHECK"}
    for match in ddl_pattern.finditer(schema_text):
        t_name = match.group(1) or match.group(2)
        cols_body = match.group(3)
        cols: List[str] = []
        for col_def in cols_body.split(","):
            c_clean = col_def.strip()
            if not c_clean:
                continue
            first_tok = c_clean.split()[0].strip('"`[]')
            if first_tok.upper() not in excluded_keywords:
                cols.append(first_tok)
        schema_map[t_name] = cols
    return schema_map


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
                        _append_parsed_columns(
                            cols_inline, rest_part.split("Columns:", 1)[1].strip()
                        )
                current_table = None
            else:
                current_table = after_table
                if current_table:
                    schema_map.setdefault(current_table, [])
        elif line.startswith("Columns:") and current_table is not None:
            _append_parsed_columns(
                schema_map[current_table], line.split("Columns:", 1)[1].strip()
            )
        elif line.startswith("-") and current_table is not None:
            _append_parsed_columns(schema_map[current_table], line.lstrip("-").strip())

    return schema_map if schema_map else _extract_ddl_schema_map(schema_text)


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

    if repo_resolver is not None:
        repo_resolver()
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
    except Exception as e:
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
        candidate_sql = extract_sql_from_response(raw_output) or raw_output.strip()
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
        except Exception as exc:
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
