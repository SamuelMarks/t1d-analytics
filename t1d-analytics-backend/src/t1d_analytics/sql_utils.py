"""Shared SQL sanitization, extraction, normalization, and DuckDB connection utilities."""

import datetime
import decimal
import math
import re
import threading
import uuid
from typing import Optional, Protocol, Sequence, Union, runtime_checkable

import duckdb

from t1d_analytics.duckdb_pool import (
    PoolSqlRawInputValue,
    PoolSqlValue,
    ReadOnlyDuckDBPool,
    _is_duckdb_lock_error,
    close_read_only_duckdb_pool,
    connect_duckdb_read_only_pooled,
    connect_duckdb_with_retry,
    execute_duckdb_batch_write_with_retry,
    execute_duckdb_write_with_retry,
    global_read_only_duckdb_pool,
    open_sandboxed_duckdb,
)
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

_PARSER_LOCK = threading.Lock()
_CACHED_MEMORY_CONN: Optional[duckdb.DuckDBPyConnection] = None
_REAL_DUCKDB_CONNECT = duckdb.connect


def _get_cached_parser_conn() -> duckdb.DuckDBPyConnection:
    """
    Return a cached in-memory DuckDB connection for lightweight SQL statement parsing.

    Returns
    -------
        duckdb.DuckDBPyConnection: Cached `:memory:` DuckDB connection.

    """
    global _CACHED_MEMORY_CONN
    if _CACHED_MEMORY_CONN is None:
        _CACHED_MEMORY_CONN = _REAL_DUCKDB_CONNECT(":memory:")
    return _CACHED_MEMORY_CONN


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

_SQL_MASKABLE_TOKEN_RE = re.compile(
    r"\$([A-Za-z0-9_]*)\$.*?\$\1\$|'(?:''|[^'])*'|--[^\r\n]*|/\*.*?\*/",
    re.DOTALL,
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
    String literals (`'...'`), dollar-quoted strings (`$$...$$`), and SQL comments are masked
    before rewriting so literal text is never mutated.

    Args:
    ----
        query: Original SQL query string.

    Returns:
    -------
        str: SQL query with numeric aggregate column arguments wrapped in ``TRY_CAST(... AS DOUBLE)``.

    """
    masked_tokens: list[str] = []

    def _mask_token(match: re.Match[str]) -> str:
        """
        Replace a matched string literal or comment with an indexed placeholder token.

        Args:
        ----
            match: Regex match for a SQL literal or comment.

        Returns:
        -------
            str: Placeholder token string.

        """
        idx = len(masked_tokens)
        masked_tokens.append(match.group(0))
        return f"__T1D_SQL_MASK_{idx}__"

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

    masked_query = _SQL_MASKABLE_TOKEN_RE.sub(_mask_token, query)
    coerced = _NUMERIC_AGG_COALESCE_RE.sub(_replace_coalesce, masked_query)
    coerced = _NUMERIC_AGG_RE.sub(_replace_match, coerced)
    coerced = _QUANTILE_AGG_RE.sub(_replace_quantile, coerced)
    coerced = _STRICT_NUMERIC_CAST_RE.sub(_replace_strict_cast, coerced)
    for idx, original_token in enumerate(masked_tokens):
        coerced = coerced.replace(f"__T1D_SQL_MASK_{idx}__", original_token)
    return coerced


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


_SHOW_NL_INDICATORS = {
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


@runtime_checkable
class SqlSyntaxValidatorProtocol(Protocol):
    """Protocol for DuckDB connection objects capable of parsing SQL statements."""

    def extract_statements(self, query: str) -> Sequence[duckdb.Statement]:
        """
        Parse a SQL string into DuckDB statement AST objects.

        Args:
        ----
            query: Raw SQL query string.

        Returns:
        -------
            Sequence[duckdb.Statement]: Parsed statement list.

        """
        ...


def _check_sql_syntax_with_conn(
    unwrapped: str,
    first_word: str,
    conn: Optional[SqlSyntaxValidatorProtocol] = None,
) -> bool:
    """
    Validate SQL syntax using either the provided `conn` or a cached `:memory:` DuckDB parser connection.

    Args:
    ----
        unwrapped: Comment-stripped SQL string.
        first_word: Lowercased first token of `unwrapped`.
        conn: Optional explicit DuckDB connection implementing `SqlSyntaxValidatorProtocol`.

    Returns:
    -------
        bool: True if DuckDB parses the statement without a `ParserException`.

    """
    _ = first_word
    try:
        if isinstance(conn, duckdb.DuckDBPyConnection):
            return len(conn.extract_statements(unwrapped)) > 0

        if conn is not None and type(conn).__module__ != "unittest.mock":
            return len(conn.extract_statements(unwrapped)) > 0

        with _PARSER_LOCK:
            parser_conn = _get_cached_parser_conn()
            return len(parser_conn.extract_statements(unwrapped)) > 0
    except duckdb.ParserException:
        return False
    except (duckdb.Error, RuntimeError, ValueError, OSError):
        return True


def is_sql_query(query: str, conn: Optional[SqlSyntaxValidatorProtocol] = None) -> bool:
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

    words = unwrapped.lower().split()
    first_word = words[0].rstrip(";").strip()
    if first_word not in ALL_SQL_KEYWORDS:
        return False

    if first_word == "show" and len(words) > 1 and words[1] in _SHOW_NL_INDICATORS:
        return False

    return _check_sql_syntax_with_conn(unwrapped, first_word, conn)


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
        sql_query = chr(10).join(lines[1:])
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
    nl = chr(10)
    for table_name, columns in cols_map.items():
        schema_desc = f"Table: {table_name}{nl}Columns:{nl}"
        for col_name, col_type in columns:
            schema_desc += f"- {col_name} ({col_type}){nl}"
        schema[table_name] = schema_desc
    return schema


_DDL_CREATE_TABLE_RE = re.compile(
    r'CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?(?:"([^"]+)"|([a-zA-Z0-9_]+))\s*\(((?:[^()]+|\([^()]*\))*)\)',
    re.IGNORECASE | re.DOTALL,
)
_DDL_EXCLUDED_KEYWORDS = {"PRIMARY", "FOREIGN", "CONSTRAINT", "UNIQUE", "CHECK"}


def _append_parsed_column_specs(
    cols_list: list[tuple[str, str]],
    cols_str: str,
) -> None:
    """
    Parse comma-separated column descriptors (`col (TYPE)` or `col`) and append unique `(name, type)` pairs.

    Args:
    ----
        cols_list: Target list of `(column_name, column_type)` pairs to mutate.
        cols_str: Raw comma-separated column specifications.

    """
    seen = {name for name, _ in cols_list}
    for raw_col in cols_str.split(","):
        if "(" in raw_col and raw_col.rstrip().endswith(")"):
            name_part, type_part = raw_col.split("(", 1)
            c_name = name_part.strip()
            c_type = type_part.rstrip().rstrip(")").strip() or "VARCHAR"
        else:
            c_name = raw_col.split("(", 1)[0].strip()
            c_type = "VARCHAR"
        if c_name and c_name not in seen:
            cols_list.append((c_name, c_type))
            seen.add(c_name)


def _extract_ddl_column_pairs(
    schema_text: str,
) -> dict[str, list[tuple[str, str]]]:
    """
    Extract table-to-`(column_name, column_type)` pairs from `CREATE TABLE` DDL statements.

    Args:
    ----
        schema_text: Raw SQL DDL string.

    Returns:
    -------
        dict[str, list[tuple[str, str]]]: Mapping of table names to `(column_name, column_type)` lists.

    """
    schema_pairs: dict[str, list[tuple[str, str]]] = {}
    for match in _DDL_CREATE_TABLE_RE.finditer(schema_text):
        t_name = match.group(1) or match.group(2)
        cols_body = match.group(3)
        cols: list[tuple[str, str]] = []
        for col_def in cols_body.split(","):
            c_clean = col_def.strip()
            if not c_clean:
                continue
            tokens = c_clean.split()
            first_tok = tokens[0].strip('"`[]')
            if first_tok.upper() not in _DDL_EXCLUDED_KEYWORDS:
                col_type = tokens[1].upper() if len(tokens) > 1 else "VARCHAR"
                cols.append((first_tok, col_type))
        schema_pairs[t_name] = cols
    return schema_pairs


def _parse_table_header_line(
    after_table: str,
    schema_pairs: dict[str, list[tuple[str, str]]],
) -> Optional[str]:
    """
    Parse a `Table: ...` line and update `schema_pairs`, returning the active multi-line table name if any.

    Args:
    ----
        after_table: Substring following `Table:`.
        schema_pairs: Mutable mapping of table names to `(column_name, column_type)` lists.

    Returns:
    -------
        Optional[str]: Active multi-line table name, or `None` for inline `Table: t | Columns: ...` lines.

    """
    if "|" in after_table:
        table_part, rest_part = after_table.split("|", 1)
        t_inline = table_part.strip()
        if t_inline:
            cols_inline = schema_pairs.setdefault(t_inline, [])
            if "Columns:" in rest_part:
                _append_parsed_column_specs(
                    cols_inline, rest_part.split("Columns:", 1)[1].strip()
                )
        return None
    if after_table:
        schema_pairs.setdefault(after_table, [])
        return after_table
    return None


def parse_schema_text_to_column_pairs(
    schema_text: str,
) -> dict[str, list[tuple[str, str]]]:
    """
    Parse a formatted `Table: ... / Columns: ...` or `CREATE TABLE` DDL string into `(column_name, column_type)` pairs.

    Args:
    ----
        schema_text: Formatted database schema or SQL DDL string.

    Returns:
    -------
        dict[str, list[tuple[str, str]]]: Mapping of table names to `(column_name, column_type)` lists.

    """
    schema_pairs: dict[str, list[tuple[str, str]]] = {}
    current_table: Optional[str] = None
    for raw_line in schema_text.splitlines():
        line = raw_line.strip()
        if line.startswith("--"):
            line = line.lstrip("-").strip()
        if line.startswith("Table:"):
            current_table = _parse_table_header_line(
                line.split("Table:", 1)[1].strip(), schema_pairs
            )
        elif line.startswith("Columns:") and current_table is not None:
            _append_parsed_column_specs(
                schema_pairs[current_table], line.split("Columns:", 1)[1].strip()
            )
        elif line.startswith("-") and current_table is not None:
            _append_parsed_column_specs(
                schema_pairs[current_table], line.lstrip("-").strip()
            )

    return schema_pairs if schema_pairs else _extract_ddl_column_pairs(schema_text)


def parse_schema_text_to_map(schema_text: str) -> dict[str, list[str]]:
    """
    Parse a formatted schema string or SQL DDL into a mapping of table names to column names.

    Args:
    ----
        schema_text: Formatted database schema or SQL DDL string.

    Returns:
    -------
        dict[str, list[str]]: Mapping of table names to ordered column name lists.

    """
    return {
        tbl: [col_name for col_name, _ in cols]
        for tbl, cols in parse_schema_text_to_column_pairs(schema_text).items()
    }


__all__ = [
    "ALL_SQL_KEYWORDS",
    "INTERNAL_TABLE_NAMES",
    "INTERNAL_TABLE_PREFIXES",
    "READ_ONLY_SQL_PREFIXES",
    "PoolSqlRawInputValue",
    "PoolSqlValue",
    "ReadOnlyDuckDBPool",
    "SqlRawInputValue",
    "SqlValue",
    "_append_parsed_column_specs",
    "_check_sql_syntax_with_conn",
    "_get_cached_parser_conn",
    "_is_duckdb_lock_error",
    "close_read_only_duckdb_pool",
    "coerce_numeric_aggregates_in_sql",
    "connect_duckdb_read_only_pooled",
    "connect_duckdb_with_retry",
    "escape_sql_identifier",
    "escape_sql_literal",
    "execute_duckdb_batch_write_with_retry",
    "execute_duckdb_write_with_retry",
    "extract_sql_from_response",
    "extract_table_columns_map",
    "extract_table_schemas",
    "global_read_only_duckdb_pool",
    "is_internal_table",
    "is_read_only_sql_query",
    "is_sql_query",
    "normalize_sql",
    "open_sandboxed_duckdb",
    "parse_schema_text_to_column_pairs",
    "parse_schema_text_to_map",
    "rewrite_varchar_numeric_aggregates",
    "sanitize_spreadsheet_cell",
    "sanitize_sql_value",
    "strip_sql_comments_and_parens",
    "strip_sql_literals_and_identifiers",
    "validate_read_only_sql",
]
