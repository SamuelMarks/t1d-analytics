"""Shared SQL sanitization, extraction, normalization, and DuckDB connection utilities."""

import datetime
import decimal
import math
import re
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import duckdb

from t1d_analytics.errors import (
    EmptyQueryError,
    MultipleStatementsError,
    ReadOnlyViolationError,
)

SqlValue = Union[str, int, float, bool, None]
SqlRawInputValue = Union[
    str,
    int,
    float,
    bool,
    None,
    decimal.Decimal,
    datetime.date,
    datetime.datetime,
    datetime.time,
    uuid.UUID,
    bytes,
    Sequence[SqlValue],
]

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

ALL_SQL_KEYWORDS = (
    *READ_ONLY_SQL_PREFIXES,
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


def escape_sql_identifier(identifier: str) -> str:
    """
    Escape double quotes inside a SQL identifier for safe ANSI double-quoted interpolation.

    Args:
    ----
        identifier: Raw table or column identifier string.

    Returns:
    -------
        str: Escaped identifier string with internal double quotes doubled.

    """
    return identifier.replace('"', '""')


def escape_sql_literal(literal: str) -> str:
    """
    Escape single quotes inside a SQL string literal for safe ANSI single-quoted interpolation.

    Args:
    ----
        literal: Raw string value to embed inside single quotes.

    Returns:
    -------
        str: Escaped string value with internal single quotes doubled.

    """
    return literal.replace("'", "''")


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


_SQL_IDENT_PATTERN = (
    r'(?:"[^"]+"|[a-zA-Z_][a-zA-Z0-9_]*)'
    r'(?:\s*\.\s*(?:"[^"]+"|[a-zA-Z_][a-zA-Z0-9_]*))?'
)

_NUMERIC_AGG_RE = re.compile(
    r"\b(SUM|AVG|STDDEV|STDDEV_SAMP|STDDEV_POP|VARIANCE|VAR_SAMP|VAR_POP|MEDIAN)"
    r"\s*\(\s*(DISTINCT\s+)?"
    r"(?:\(\s*)?(" + _SQL_IDENT_PATTERN + r")(?:\s*\))?"
    r"\s*\)",
    re.IGNORECASE,
)

_NUMERIC_AGG_COALESCE_RE = re.compile(
    r"\b(SUM|AVG|STDDEV|STDDEV_SAMP|STDDEV_POP|VARIANCE|VAR_SAMP|VAR_POP|MEDIAN)"
    r"\s*\(\s*(DISTINCT\s+)?"
    r"(COALESCE|IFNULL|NULLIF)\s*\(\s*(" + _SQL_IDENT_PATTERN + r")\s*,"
    r"([^()]+)\)\s*\)",
    re.IGNORECASE,
)

_QUANTILE_AGG_RE = re.compile(
    r"\b(QUANTILE_CONT|QUANTILE_DISC)\s*\(\s*("
    + _SQL_IDENT_PATTERN
    + r")\s*,\s*([^()]+)\)",
    re.IGNORECASE,
)

_STRICT_NUMERIC_CAST_RE = re.compile(
    r"(?<!TRY_)\bCAST\s*\(\s*("
    + _SQL_IDENT_PATTERN
    + r")\s+AS\s+"
    + r"(DOUBLE(?:\s+PRECISION)?|FLOAT\d*|REAL|INT(?:EGER|\d+)?|BIGINT|SMALLINT|TINYINT|HUGEINT|DECIMAL(?:\s*\([^)]+\))?|NUMERIC(?:\s*\([^)]+\))?)"
    + r"\s*\)",
    re.IGNORECASE,
)


def coerce_numeric_aggregates_in_sql(query: str) -> str:
    """
    Rewrite numeric aggregate calls on column identifiers to use ``TRY_CAST(... AS DOUBLE)``.

    Public T1D clinical CSV/SAS/SPSS datasets (such as JAEB ``tblAPTSummary.csv``) frequently
    encode missing values in numeric columns (e.g. ``NumSevHypo``, ``AgeAsOfRandDt``, ``HbA1c``)
    as empty strings, ``"NA"``, ``"."``, or ``"Unknown"``, causing DuckDB ``read_csv`` to infer
    ``VARCHAR`` column types. Rewriting ``SUM(col)``, ``AVG(col)``, ``SUM(COALESCE(col, 0))``,
    and strict ``CAST(col AS DOUBLE/INT)`` to ``TRY_CAST`` prevents ``Binder Error: No function
    matches the given name and argument types 'sum(VARCHAR)'`` and ``Conversion Error``.

    Args:
    ----
        query: Original SQL query string.

    Returns:
    -------
        str: SQL query with numeric aggregate column arguments wrapped in ``TRY_CAST(... AS DOUBLE)``.

    """

    def _replace_coalesce(match: re.Match[str]) -> str:
        """
        Format a matched numeric aggregate containing COALESCE/IFNULL/NULLIF with TRY_CAST.

        Args:
        ----
            match: Regex match for aggregate wrapping COALESCE/IFNULL/NULLIF.

        Returns:
        -------
            str: Rewritten SQL aggregate expression.

        """
        func_name = match.group(1)
        distinct_kw = match.group(2) or ""
        null_fn = match.group(3)
        col_ident = match.group(4)
        fallback_arg = match.group(5).strip()
        return (
            f"{func_name}({distinct_kw}{null_fn}("
            f"TRY_CAST({col_ident} AS DOUBLE), {fallback_arg}))"
        )

    def _replace_match(match: re.Match[str]) -> str:
        """
        Format a matched numeric aggregate call with TRY_CAST around the identifier.

        Args:
        ----
            match: Regex match containing function name, optional DISTINCT, and column identifier.

        Returns:
        -------
            str: Rewritten SQL aggregate expression.

        """
        func_name = match.group(1)
        distinct_kw = match.group(2) or ""
        col_ident = match.group(3)
        return f"{func_name}({distinct_kw}TRY_CAST({col_ident} AS DOUBLE))"

    def _replace_quantile(match: re.Match[str]) -> str:
        """
        Format a matched QUANTILE_CONT/QUANTILE_DISC call with TRY_CAST around the identifier.

        Args:
        ----
            match: Regex match containing quantile function name, column identifier, and quantile literal.

        Returns:
        -------
            str: Rewritten SQL quantile expression.

        """
        func_name = match.group(1)
        col_ident = match.group(2)
        q_arg = match.group(3).strip()
        return f"{func_name}(TRY_CAST({col_ident} AS DOUBLE), {q_arg})"

    def _replace_strict_cast(match: re.Match[str]) -> str:
        """
        Rewrite a strict CAST(ident AS <numeric_type>) into TRY_CAST(ident AS <numeric_type>).

        Args:
        ----
            match: Regex match containing column identifier and target numeric SQL type.

        Returns:
        -------
            str: Safe TRY_CAST SQL expression.

        """
        col_ident = match.group(1)
        target_type = match.group(2)
        return f"TRY_CAST({col_ident} AS {target_type})"

    coerced = _NUMERIC_AGG_COALESCE_RE.sub(_replace_coalesce, query)
    coerced = _NUMERIC_AGG_RE.sub(_replace_match, coerced)
    coerced = _QUANTILE_AGG_RE.sub(_replace_quantile, coerced)
    return _STRICT_NUMERIC_CAST_RE.sub(_replace_strict_cast, coerced)


rewrite_varchar_numeric_aggregates = coerce_numeric_aggregates_in_sql


def strip_sql_comments_and_parens(query: str) -> str:
    """
    Strip SQL line/block comments (including nested block comments) and outer parentheses from a query string.

    Args:
    ----
        query: Raw SQL or user input string.

    Returns:
    -------
        str: Cleaned query string.

    """
    cleaned = re.sub(r"--.*$", " ", query, flags=re.MULTILINE)
    prev = None
    while prev != cleaned:
        prev = cleaned
        cleaned = re.sub(r"/\*(?:(?!\*/|/\*).)*\*/", " ", cleaned, flags=re.DOTALL)
    cleaned = cleaned.strip()
    while cleaned.startswith("(") and cleaned.endswith(")"):
        cleaned = cleaned[1:-1].strip()
    return cleaned


_FORBIDDEN_WRITE_STATEMENT_RE = re.compile(
    r"\b(?:"
    r"INSERT\s+INTO|"
    r"UPDATE\s+\S+\s+SET|"
    r"DELETE\s+FROM|"
    r"DROP\s+(?:TABLE|VIEW|INDEX|SCHEMA|SEQUENCE|MACRO|TYPE)|"
    r"ALTER\s+(?:TABLE|VIEW)|"
    r"CREATE\s+(?:OR\s+REPLACE\s+)?(?:TEMP(?:ORARY)?\s+)?(?:TABLE|VIEW|INDEX|SCHEMA|SEQUENCE|MACRO|TYPE)|"
    r"REPLACE\s+INTO|"
    r"TRUNCATE(?:\s+TABLE)?|"
    r"COPY\s+\S+\s+TO|"
    r"ATTACH\b|"
    r"DETACH\b|"
    r"VACUUM\b|"
    r"GRANT\b|"
    r"REVOKE\b|"
    r"READ_CSV(?:_AUTO)?\s*\(|"
    r"READ_PARQUET\s*\(|"
    r"READ_JSON(?:_AUTO)?\s*\(|"
    r"READ_TEXT\s*\(|"
    r"READ_BLOB\s*\(|"
    r"GLOB\s*\(|"
    r"SNIFF_CSV\s*\(|"
    r"POSTGRES_SCAN\s*\(|"
    r"SQLITE_SCAN\s*\("
    r")",
    re.IGNORECASE,
)


def strip_sql_literals_and_identifiers(query: str) -> str:
    """
    Replace single-quoted string literals, dollar-quoted strings, and double-quoted identifiers with empty placeholders.

    Args:
    ----
        query: SQL query string.

    Returns:
    -------
        str: SQL query with quoted literals and identifiers neutralized.

    """
    without_dollar = re.sub(
        r"\$([A-Za-z0-9_]*)\$.*?\$\1\$", "''", query, flags=re.DOTALL
    )
    without_single = re.sub(r"'(?:''|[^'])*'", "''", without_dollar)
    return re.sub(r'"(?:""|[^"])*"', '""', without_single)


def is_read_only_sql_query(query: str) -> bool:
    """
    Check whether a SQL query starts with a read-only statement keyword and contains no write/DDL clauses.

    Single-quoted string literals (e.g., ``SELECT 'update'``) and scalar function calls
    (e.g., ``SELECT REPLACE(col, 'a', 'b')``) are permitted, while multi-statement queries
    and embedded write/DDL clauses (e.g., ``WITH d AS (DELETE FROM t) SELECT * FROM d``) are rejected.

    Args:
    ----
        query: SQL query string to inspect.

    Returns:
    -------
        bool: True if the statement is a single read-only SQL query, False otherwise.

    """
    unwrapped = strip_sql_comments_and_parens(query)
    if not unwrapped:
        return False
    first_word = unwrapped.split()[0].lower().rstrip(";").strip()
    if first_word not in READ_ONLY_SQL_PREFIXES:
        return False

    without_literals = strip_sql_literals_and_identifiers(unwrapped)
    segments = [seg.strip() for seg in without_literals.split(";") if seg.strip()]
    if len(segments) > 1:
        return False

    return _FORBIDDEN_WRITE_STATEMENT_RE.search(without_literals) is None


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

    without_literals = strip_sql_literals_and_identifiers(unwrapped)
    segments = [seg.strip() for seg in without_literals.split(";") if seg.strip()]
    if len(segments) > 1:
        raise MultipleStatementsError(
            "Multiple SQL statements are not allowed in a single read-only query."
        )

    if not is_read_only_sql_query(unwrapped):
        raise ReadOnlyViolationError(f"Query is not read-only: {query.strip()}")

    return query.strip()


ensure_read_only_sql = validate_read_only_sql


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
        if isinstance(temp_conn, duckdb.DuckDBPyConnection):
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


def extract_sql_from_response(full_response: SqlValue) -> str:
    """
    Extract a SQL query from markdown code blocks or plain text in an LLM completion.

    Args:
    ----
        full_response: Raw text response or scalar value from the LLM.

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


def sanitize_sql_value(val: SqlRawInputValue) -> SqlValue:
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


def sanitize_spreadsheet_cell(val: SqlRawInputValue) -> SqlValue:
    """
    Sanitize a cell value against spreadsheet formula injection (CSV/XLSX).

    Leading characters '=', '+', '-', '@', tab, and carriage-return on non-numeric strings
    are prefixed with a single quote so spreadsheet engines treat them as text.

    Args:
    ----
        val: Raw database value.

    Returns:
    -------
        SqlValue: Safe cell value for spreadsheet export.

    """
    if val is None:
        return None
    if isinstance(val, (int, float, bool)):
        return val
    s_val = str(val)
    if not s_val:
        return s_val
    first_char = s_val[0]
    stripped_leading = s_val.lstrip(" ")
    trigger_char = stripped_leading[0] if stripped_leading else ""
    if first_char in (chr(9), chr(13), chr(10)) or trigger_char in (
        "=",
        "+",
        "-",
        "@",
        chr(9),
        chr(13),
        chr(10),
    ):
        is_numeric = False
        if trigger_char in ("+", "-"):
            try:
                float(stripped_leading)
                is_numeric = True
            except ValueError:
                is_numeric = False
        if not is_numeric:
            s_val = f"'{s_val}"
    return s_val


def _is_duckdb_lock_error(exc: duckdb.IOException) -> bool:
    """
    Determine whether a `duckdb.IOException` was caused by transient file lock contention.

    Args:
    ----
        exc: The DuckDB I/O exception to inspect.

    Returns:
    -------
        bool: True if the error indicates file lock contention, False otherwise.

    """
    err_msg = str(exc).lower()
    return (
        "lock" in err_msg
        or "conflicting" in err_msg
        or "resource temporarily unavailable" in err_msg
    )


@contextmanager
def connect_duckdb_with_retry(
    db_path: Union[str, Path],
    read_only: bool = True,
    max_attempts: int = 5,
    base_delay_seconds: float = 0.05,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> Iterator[duckdb.DuckDBPyConnection]:
    """
    Acquire a DuckDB connection with retry backoff on lock contention and close on context exit.

    Connection acquisition is strictly separated from the `yield` block so that exceptions
    raised inside the caller's `with` body never trigger generator re-entry errors.

    Args:
    ----
        db_path: Path to the DuckDB database file.
        read_only: Whether to open the database in read-only mode.
        max_attempts: Maximum number of connection attempts on lock contention.
        base_delay_seconds: Initial backoff delay in seconds.
        sleep_fn: Injectable sleep callable for backoff delays.

    Yields:
    ------
        duckdb.DuckDBPyConnection: Open DuckDB connection guaranteed to close on exit.

    Raises:
    ------
        duckdb.IOException: If a non-lock I/O error occurs or lock retries are exhausted.

    """
    attempt = 0
    while True:
        try:
            conn = duckdb.connect(str(db_path), read_only=read_only)
            break
        except duckdb.IOException as e:
            if not _is_duckdb_lock_error(e) or attempt >= max_attempts - 1:
                raise
            sleep_fn(base_delay_seconds * (attempt + 1))
            attempt += 1

    try:
        yield conn
    finally:
        conn.close()


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


class ReadOnlyDuckDBPool:
    """Thread-safe, mtime-validated connection pool for read-only DuckDB databases."""

    def __init__(self, max_pool_size_per_db: int = 8) -> None:
        """
        Initialize the read-only DuckDB connection pool.

        Args:
        ----
            max_pool_size_per_db: Maximum idle connections retained per database file.

        """
        self._max_pool_size_per_db = max(1, max_pool_size_per_db)
        self._lock = threading.Lock()
        self._pools: Dict[str, List[duckdb.DuckDBPyConnection]] = {}
        self._mtimes: Dict[str, Tuple[int, int]] = {}

    @staticmethod
    def _stat_signature(path_str: str) -> Tuple[int, int]:
        """
        Compute the `(st_mtime_ns, st_size)` signature for `path_str` on disk.

        Args:
        ----
            path_str: Target filesystem path string.

        Returns:
        -------
            Tuple[int, int]: Modification time in nanoseconds and file size in bytes.

        """
        try:
            st = Path(path_str).stat()
            return (st.st_mtime_ns, st.st_size)
        except OSError:
            return (-1, -1)

    def acquire(
        self,
        db_path: Union[str, Path],
        connect_fn: Callable[..., duckdb.DuckDBPyConnection] = duckdb.connect,
        disable_external_access: bool = True,
    ) -> duckdb.DuckDBPyConnection:
        """
        Acquire a read-only DuckDB connection from the pool or open a new one.

        Args:
        ----
            db_path: Path to the DuckDB database file.
            connect_fn: DuckDB connection factory callable.
            disable_external_access: Whether to execute 'SET enable_external_access = false'.

        Returns:
        -------
            duckdb.DuckDBPyConnection: Open read-only DuckDB connection.

        """
        key = str(Path(db_path).resolve())
        sig = self._stat_signature(key)
        stale_conns: List[duckdb.DuckDBPyConnection] = []
        with self._lock:
            prev_sig = self._mtimes.get(key)
            if prev_sig is not None and prev_sig != sig:
                stale_conns = self._pools.pop(key, [])
            self._mtimes[key] = sig
            bucket = self._pools.get(key)
            if bucket:
                conn = bucket.pop()
            else:
                conn = None

        for stale in stale_conns:
            stale.close()

        if conn is not None:
            return conn

        new_conn = connect_fn(str(db_path), read_only=True)
        if disable_external_access:
            new_conn.execute("SET enable_external_access = false")
        return new_conn

    def release(
        self,
        db_path: Union[str, Path],
        conn: duckdb.DuckDBPyConnection,
        discard: bool = False,
    ) -> None:
        """
        Return a read-only DuckDB connection to the pool or close it if discarded/full.

        Args:
        ----
            db_path: Path to the DuckDB database file.
            conn: Connection instance to release.
            discard: If True, close `conn` instead of returning it to the pool.

        """
        if discard:
            conn.close()
            return
        key = str(Path(db_path).resolve())
        sig = self._stat_signature(key)
        should_close = False
        with self._lock:
            if sig == (-1, -1) or self._mtimes.get(key) != sig:
                should_close = True
            else:
                bucket = self._pools.setdefault(key, [])
                if len(bucket) < self._max_pool_size_per_db:
                    bucket.append(conn)
                else:
                    should_close = True
        if should_close:
            conn.close()

    def close_path(self, db_path: Union[str, Path]) -> None:
        """
        Close and evict all idle pooled read-only connections for a specific database file.

        Args:
        ----
            db_path: Path to the DuckDB database file.

        """
        key = str(Path(db_path).resolve())
        with self._lock:
            to_close = self._pools.pop(key, [])
            self._mtimes.pop(key, None)
        for conn in to_close:
            conn.close()

    def close_all(self) -> None:
        """
        Close all idle pooled connections across all database files.

        Returns
        -------
            None

        """
        to_close: List[duckdb.DuckDBPyConnection] = []
        with self._lock:
            for bucket in self._pools.values():
                to_close.extend(bucket)
            self._pools.clear()
            self._mtimes.clear()
        for conn in to_close:
            conn.close()


global_read_only_duckdb_pool = ReadOnlyDuckDBPool()


@contextmanager
def connect_duckdb_read_only_pooled(
    db_path: Union[str, Path],
    connect_fn: Callable[..., duckdb.DuckDBPyConnection] = duckdb.connect,
    disable_external_access: bool = True,
    pool: Optional[ReadOnlyDuckDBPool] = None,
) -> Iterator[duckdb.DuckDBPyConnection]:
    """
    Context manager checking out a read-only DuckDB connection from `ReadOnlyDuckDBPool`.

    Args:
    ----
        db_path: Path to the DuckDB database file.
        connect_fn: DuckDB connection factory callable.
        disable_external_access: Whether to disable external filesystem access on new connections.
        pool: Optional custom `ReadOnlyDuckDBPool` instance (defaults to `global_read_only_duckdb_pool`).

    Yields:
    ------
        duckdb.DuckDBPyConnection: Pooled read-only DuckDB connection.

    """
    active_pool = pool if pool is not None else global_read_only_duckdb_pool
    use_pool = connect_fn is duckdb.connect
    if not use_pool:
        conn = connect_fn(str(db_path), read_only=True)
        try:
            if disable_external_access:
                conn.execute("SET enable_external_access = false")
            yield conn
        finally:
            conn.close()
        return

    conn = active_pool.acquire(
        db_path=db_path,
        connect_fn=connect_fn,
        disable_external_access=disable_external_access,
    )
    discard = False
    try:
        yield conn
    except BaseException:
        discard = True
        raise
    finally:
        active_pool.release(db_path=db_path, conn=conn, discard=discard)


def close_read_only_duckdb_pool() -> None:
    """
    Close all connections held in `global_read_only_duckdb_pool`.

    Returns
    -------
        None

    """
    global_read_only_duckdb_pool.close_all()


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


def extract_table_columns_map(
    conn: duckdb.DuckDBPyConnection,
    exclude_internal: bool = False,
    exclude_training_tables: bool = False,
) -> dict[str, list[tuple[str, str]]]:
    """
    Extract ordered `(column_name, column_type)` tuples per table from an active DuckDB connection.

    Args:
    ----
        conn: Active DuckDB connection.
        exclude_internal: Whether to exclude `_t1d_` internal tables.
        exclude_training_tables: Whether to also exclude `chat_sessions` and training tables.

    Returns:
    -------
        dict[str, list[tuple[str, str]]]: Mapping from table name to list of `(column_name, column_type)` pairs.

    """
    result: dict[str, list[tuple[str, str]]] = {}
    tables = conn.execute("SHOW TABLES").fetchall()
    for row in tables:
        table_name = str(row[0])
        if (exclude_internal or exclude_training_tables) and is_internal_table(
            table_name, exclude_training_tables=exclude_training_tables
        ):
            continue
        escaped_name = escape_sql_identifier(table_name)
        columns = conn.execute(f'DESCRIBE "{escaped_name}"').fetchall()
        result[table_name] = [(str(col[0]), str(col[1])) for col in columns]
    return result


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
    cols_map = extract_table_columns_map(
        conn,
        exclude_internal=True,
        exclude_training_tables=exclude_training_tables,
    )
    for table_name, columns in cols_map.items():
        schema_desc = f"Table: {table_name}\nColumns:\n"
        for col_name, col_type in columns:
            schema_desc += f"- {col_name} ({col_type})\n"
        schema[table_name] = schema_desc
    return schema


def execute_duckdb_batch_write_with_retry(
    db_path: Union[str, Path],
    statements: Sequence[Tuple[str, Optional[Sequence[SqlRawInputValue]]]],
    max_attempts: int = 20,
    base_delay_seconds: float = 0.05,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> None:
    """
    Execute a batch of write statements against a DuckDB database within a single connection and retry on lock contention.

    Args:
    ----
        db_path: Path to the target DuckDB database.
        statements: Sequence of (query, optional_params) tuples to execute in order.
        max_attempts: Maximum number of connection/execution attempts.
        base_delay_seconds: Base delay in seconds multiplied by attempt index.
        sleep_fn: Sleep callable used between lock retries.

    Raises:
    ------
        duckdb.IOException: If a non-lock I/O error occurs or lock retries are exhausted.

    """
    global_read_only_duckdb_pool.close_path(db_path)
    for attempt in range(max_attempts):
        try:
            conn = duckdb.connect(str(db_path), read_only=False)
            try:
                for query, params in statements:
                    if params is not None:
                        conn.execute(query, list(params))
                    else:
                        conn.execute(query)
            finally:
                conn.close()
            return
        except duckdb.IOException as e:
            if "lock" in str(e).lower():
                sleep_fn(base_delay_seconds * (attempt + 1))
            else:
                raise
    raise duckdb.IOException("database is locked after retries")


def execute_duckdb_write_with_retry(
    db_path: Union[str, Path],
    query: str,
    params: Optional[Sequence[SqlRawInputValue]] = None,
    max_attempts: int = 20,
    base_delay_seconds: float = 0.05,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> None:
    """
    Execute a write statement against a DuckDB database with retry backoff on lock contention.

    Args:
    ----
        db_path: Path to the target DuckDB database.
        query: Parameterized SQL statement to execute.
        params: Optional sequence of bind parameters.
        max_attempts: Maximum number of connection/execution attempts.
        base_delay_seconds: Base delay in seconds multiplied by attempt index.
        sleep_fn: Sleep callable used between lock retries.

    Raises:
    ------
        duckdb.IOException: If a non-lock I/O error occurs or lock retries are exhausted.

    """
    execute_duckdb_batch_write_with_retry(
        db_path=db_path,
        statements=[(query, params)],
        max_attempts=max_attempts,
        base_delay_seconds=base_delay_seconds,
        sleep_fn=sleep_fn,
    )
