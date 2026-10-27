"""SQL self-correction, error hint classification, and SQL token validation utilities."""

import importlib
import logging
import re
from pathlib import Path
from typing import Callable, Dict, List, Optional

from typing_extensions import TypedDict

from t1d_analytics.sql_utils import SqlValue, extract_sql_from_response

logger = logging.getLogger(__name__)


class AgenticSqlResult(TypedDict):
    """Structured result from the self-correcting agentic NL-to-SQL execution loop."""

    success: bool
    final_sql: str
    attempts: int
    rows: List[Dict[str, SqlValue]]
    error: Optional[str]


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
