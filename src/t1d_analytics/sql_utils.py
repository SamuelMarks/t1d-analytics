"""Shared SQL sanitization, extraction, normalization, and DuckDB connection utilities."""

import datetime
import decimal
import math
import re
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional, Union

import duckdb

from t1d_analytics.errors import (
    EmptyQueryError,
    MultipleStatementsError,
    ReadOnlyViolationError,
)

SqlValue = Union[str, int, float, bool, None]

READ_ONLY_SQL_PREFIXES = (
    "select",
    "with",
    "show",
    "describe",
    "pragma",
    "summarize",
    "explain",
    "values",
    "from",
    "table",
)

ALL_SQL_KEYWORDS = READ_ONLY_SQL_PREFIXES + (
    "call",
    "create",
    "drop",
    "alter",
    "insert",
    "update",
    "delete",
    "copy",
    "attach",
    "detach",
    "begin",
    "commit",
    "rollback",
    "checkpoint",
)


def normalize_sql(sql: str) -> str:
    """
    Normalize a SQL query string for exact-match comparison.

    Args:
    ----
        sql: Raw SQL query string.

    Returns:
    -------
        str: Lowercase, whitespace-normalized SQL string without trailing semicolons.

    """
    cleaned = sql.strip().rstrip(";")
    return re.sub(r"\s+", " ", cleaned).strip().lower()


def strip_sql_comments_and_parens(query: str) -> str:
    """
    Strip SQL line/block comments and outer parentheses from a query string.

    Args:
    ----
        query: Raw SQL or user input string.

    Returns:
    -------
        str: Cleaned query string.

    """
    cleaned = re.sub(r"--.*$", " ", query, flags=re.MULTILINE)
    cleaned = re.sub(r"/\*.*?\*/", " ", cleaned, flags=re.DOTALL).strip()
    while cleaned.startswith("(") and cleaned.endswith(")"):
        cleaned = cleaned[1:-1].strip()
    return cleaned


def is_read_only_sql_query(query: str) -> bool:
    """
    Check whether a SQL query starts with a read-only statement keyword.

    Args:
    ----
        query: SQL query string to inspect.

    Returns:
    -------
        bool: True if the statement starts with a read-only SQL keyword, False otherwise.

    """
    unwrapped = strip_sql_comments_and_parens(query)
    if not unwrapped:
        return False
    first_word = unwrapped.split()[0].lower().rstrip(";").strip()
    return first_word in READ_ONLY_SQL_PREFIXES


def validate_read_only_sql(query: str) -> str:
    """
    Validate that a SQL query is non-empty, contains a single statement, and is read-only.

    Args:
    ----
        query: SQL query string to validate.

    Returns:
    -------
        str: Stripped, validated SQL query string.

    Raises:
    ------
        EmptyQueryError: If the query is empty or contains only comments/whitespace.
        MultipleStatementsError: If the query contains multiple semicolon-separated statements.
        ReadOnlyViolationError: If the query is not a read-only SQL statement.

    """
    unwrapped = strip_sql_comments_and_parens(query)
    if not unwrapped:
        raise EmptyQueryError("SQL query cannot be empty.")

    without_literals = re.sub(r"'(?:''|[^'])*'", "''", unwrapped)
    without_literals = re.sub(r'"(?:""|[^"])*"', '""', without_literals)
    segments = [seg.strip() for seg in without_literals.split(";") if seg.strip()]
    if len(segments) > 1:
        raise MultipleStatementsError(
            "Multiple SQL statements are not allowed in a single read-only query."
        )

    if not is_read_only_sql_query(unwrapped):
        raise ReadOnlyViolationError(f"Query is not read-only: {query.strip()}")

    return query.strip()


def is_sql_query(query: str, conn: Optional[duckdb.DuckDBPyConnection] = None) -> bool:
    """
    Determine whether a user input string is a SQL query rather than natural language.

    Args:
    ----
        query: The raw query string entered by the user.
        conn: Optional DuckDB connection to assist in query syntax verification.

    Returns:
    -------
        bool: True if the query appears to be a SQL statement, False if natural language.

    """
    unwrapped = strip_sql_comments_and_parens(query)
    if not unwrapped:
        return False

    first_word = unwrapped.split()[0].lower().rstrip(";").strip()
    if first_word not in ALL_SQL_KEYWORDS:
        return False

    if first_word == "show":
        words = unwrapped.lower().split()
        nl_indicators = {
            "me",
            "the",
            "a",
            "an",
            "us",
            "our",
            "all",
            "how",
            "what",
            "where",
            "which",
            "any",
            "patients",
            "trials",
            "datasets",
        }
        if len(words) > 1 and words[1] in nl_indicators:
            return False

    temp_conn = conn or duckdb.connect(":memory:")
    try:
        if type(temp_conn).__name__ == "DuckDBPyConnection" and hasattr(
            temp_conn, "extract_statements"
        ):
            stmts = temp_conn.extract_statements(unwrapped)
            return len(stmts) > 0

        if first_word in (
            "checkpoint",
            "attach",
            "detach",
            "call",
            "pragma",
            "create",
            "drop",
            "alter",
            "insert",
            "update",
            "delete",
        ):
            return True

        temp_conn.execute(f"EXPLAIN {unwrapped}")
        return True
    except duckdb.ParserException:
        return False
    except Exception:
        return True
    finally:
        if conn is None:
            temp_conn.close()


def extract_sql_from_response(full_response: str) -> str:
    """
    Extract a SQL query from markdown code blocks or plain text in an LLM completion.

    Args:
    ----
        full_response: Raw text response from the LLM.

    Returns:
    -------
        str: Extracted SQL query string.

    """
    if not isinstance(full_response, str):
        full_response = str(full_response)

    blocks = re.findall(
        r"```(?:[a-zA-Z0-9_-]+[ \t]*[\r\n]+)?(.*?)[\r\n]*```",
        full_response,
        re.DOTALL | re.IGNORECASE,
    )
    if blocks:
        sql_candidate = blocks[-1].strip()
        for block in reversed(blocks):
            clean = block.strip()
            if clean.lower().startswith(
                (
                    "select",
                    "with",
                    "show",
                    "describe",
                    "pragma",
                    "summarize",
                    "explain",
                )
            ):
                sql_candidate = clean
                break
        return str(sql_candidate)

    sql_query = full_response
    lines = sql_query.splitlines()
    if lines and lines[0].strip().lower() in ("sql", "duckdb"):
        sql_query = "\n".join(lines[1:])
    return sql_query.replace("```", "").strip()


def sanitize_sql_value(val: Any) -> SqlValue:
    """
    Sanitize an arbitrary database column value into a JSON-compliant SqlValue.

    Args:
    ----
        val: The raw value returned from the database driver.

    Returns:
    -------
        SqlValue: A JSON-compliant and Pydantic-compatible value.

    """
    if val is None:
        return None
    if isinstance(val, (datetime.date, datetime.datetime, datetime.time)):
        return val.isoformat()
    if isinstance(val, decimal.Decimal):
        return float(val)
    if isinstance(val, uuid.UUID):
        return str(val)
    if isinstance(val, float):
        if math.isnan(val) or math.isinf(val):
            return None
        return val
    if isinstance(val, (int, bool, str)):
        return val
    return str(val)


def sanitize_spreadsheet_cell(val: Any) -> Any:
    """
    Sanitize a cell value against spreadsheet formula injection (CSV/XLSX).

    Leading characters '=', '+', '-', '@', tab, and carriage-return on non-numeric strings
    are prefixed with a single quote so spreadsheet engines treat them as text.

    Args:
    ----
        val: Raw database value.

    Returns:
    -------
        Any: Safe cell value for spreadsheet export.

    """
    if val is None:
        return None
    if isinstance(val, (int, float, bool)):
        return val
    s_val = str(val)
    if s_val and s_val[0] in ("=", "+", "-", "@", chr(9), chr(13)):
        is_numeric = False
        if s_val[0] in ("+", "-"):
            try:
                float(s_val)
                is_numeric = True
            except ValueError:
                is_numeric = False
        if not is_numeric:
            s_val = f"'{s_val}"
    return s_val


@contextmanager
def open_sandboxed_duckdb(
    db_path: Union[str, Path],
    read_only: bool = True,
    disable_external_access: bool = True,
) -> Iterator[duckdb.DuckDBPyConnection]:
    """
    Context manager opening a DuckDB connection with optional security sandboxing.

    Args:
    ----
        db_path: Path to the DuckDB database file.
        read_only: Whether to open the database in read-only mode.
        disable_external_access: Whether to execute 'SET enable_external_access = false'.

    Yields:
    ------
        duckdb.DuckDBPyConnection: Configured DuckDB connection guaranteed to close on exit.

    """
    conn = duckdb.connect(str(db_path), read_only=read_only)
    try:
        if disable_external_access:
            conn.execute("SET enable_external_access = false")
        yield conn
    finally:
        conn.close()


INTERNAL_TABLE_NAMES = {"chat_sessions", "pretrain", "sft", "dpo"}
INTERNAL_TABLE_PREFIXES = ("_t1d_", "pretrain_", "sft_", "dpo_")


def is_internal_table(table_name: str, exclude_training_tables: bool = True) -> bool:
    """
    Check if a DuckDB table is an internal manifest, session, or training artifact table.

    Args:
    ----
        table_name: Name of the table to inspect.
        exclude_training_tables: Whether to also exclude chat_sessions and sft/dpo/pretrain tables.

    Returns:
    -------
        bool: True if the table is an internal system table, False otherwise.

    """
    if table_name.startswith("_t1d_"):
        return True
    if not exclude_training_tables:
        return False
    lower_name = table_name.lower()
    return lower_name in INTERNAL_TABLE_NAMES or lower_name.startswith(
        INTERNAL_TABLE_PREFIXES
    )


def extract_table_schemas(
    conn: duckdb.DuckDBPyConnection,
    exclude_training_tables: bool = True,
) -> dict[str, str]:
    """
    Extract per-table textual schema descriptions from an active DuckDB connection.

    Args:
    ----
        conn: Active DuckDB connection.
        exclude_training_tables: Whether to exclude internal chat_sessions and training tables.

    Returns:
    -------
        dict[str, str]: Mapping from table name to formatted column listing.

    """
    schema: dict[str, str] = {}
    tables = conn.execute("SHOW TABLES").fetchall()
    for row in tables:
        table_name = str(row[0])
        if is_internal_table(
            table_name, exclude_training_tables=exclude_training_tables
        ):
            continue
        columns = conn.execute(f'DESCRIBE "{table_name}"').fetchall()
        schema_desc = f"Table: {table_name}\nColumns:\n"
        for col in columns:
            schema_desc += f"- {col[0]} ({col[1]})\n"
        schema[table_name] = schema_desc
    return schema
