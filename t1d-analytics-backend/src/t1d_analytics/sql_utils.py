"""Shared SQL sanitization, extraction, normalization, and DuckDB connection utilities."""

import atexit
import datetime
import decimal
import math
import re
import threading
import uuid
from collections.abc import Sequence
from typing import Literal, Protocol, runtime_checkable

import duckdb

from t1d_analytics.duckdb_pool import (
    JAEB_ATTACH_CANDIDATES,
    PoolSqlRawInputValue,
    PoolSqlValue,
    ReadOnlyDuckDbCursorProxy,
    ReadOnlyDuckDBPool,
    close_read_only_duckdb_pool,
    connect_duckdb_read_only_cursor,
    connect_duckdb_read_only_pooled,
    connect_duckdb_with_retry,
    create_read_only_cursor_proxy,
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

SqlValue = str | int | float | bool | None
SqlRawInputValue = (
    str
    | int
    | float
    | bool
    | None
    | decimal.Decimal
    | datetime.date
    | datetime.datetime
    | datetime.time
    | uuid.UUID
    | bytes
    | Sequence[SqlValue]
)

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
_CACHED_MEMORY_CONN: duckdb.DuckDBPyConnection | None = None
_REAL_DUCKDB_CONNECT = duckdb.connect


def _get_cached_parser_conn() -> duckdb.DuckDBPyConnection:
    """
    Return a cached in-memory DuckDB connection for lightweight SQL statement parsing.

    Caller must hold `_PARSER_LOCK` or allow this helper to initialize safely under lock.

    Returns:
        duckdb.DuckDBPyConnection: Cached `:memory:` DuckDB connection.

    """
    global _CACHED_MEMORY_CONN
    if _CACHED_MEMORY_CONN is None:
        _CACHED_MEMORY_CONN = _REAL_DUCKDB_CONNECT(":memory:")
    return _CACHED_MEMORY_CONN


def close_cached_parser_conn() -> None:
    """
    Close and clear the cached in-memory SQL parser DuckDB connection.

    Returns:
        None

    """
    global _CACHED_MEMORY_CONN
    with _PARSER_LOCK:
        if _CACHED_MEMORY_CONN is not None:
            _CACHED_MEMORY_CONN.close()
            _CACHED_MEMORY_CONN = None


atexit.register(close_cached_parser_conn)


def escape_sql_identifier(identifier: str) -> str:
    """
    Escape double quotes inside a SQL identifier for safe ANSI double-quoted interpolation.

    Args:
        identifier: Raw table or column identifier string.

    Returns:
        str: Escaped identifier string with internal double quotes doubled.

    """
    return identifier.replace('"', '""')


def escape_sql_literal(literal: str) -> str:
    """
    Escape single quotes inside a SQL string literal for safe ANSI single-quoted interpolation.

    Args:
        literal: Raw string value to embed inside single quotes.

    Returns:
        str: Escaped string value with internal single quotes doubled.

    """
    return literal.replace("'", "''")


def normalize_sql(sql: str) -> str:
    """
    Normalize a SQL query string for exact-match comparison.

    Args:
        sql: Raw SQL query string.

    Returns:
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
        query: Original SQL query string.

    Returns:
        str: SQL query with numeric aggregate column arguments wrapped in ``TRY_CAST(... AS DOUBLE)``.

    """
    masked_tokens: list[str] = []

    def _mask_token(match: re.Match[str]) -> str:
        """
        Replace a matched string literal or comment with an indexed placeholder token.

        Args:
            match: Regex match for a SQL literal or comment.

        Returns:
            str: Placeholder token string.

        """
        idx = len(masked_tokens)
        masked_tokens.append(match.group(0))
        return f"__T1D_SQL_MASK_{idx}__"

    def _replace_coalesce(match: re.Match[str]) -> str:
        """
        Format a matched numeric aggregate containing COALESCE/IFNULL/NULLIF with TRY_CAST.

        Args:
            match: Regex match for aggregate wrapping COALESCE/IFNULL/NULLIF.

        Returns:
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
            match: Regex match containing function name, optional DISTINCT, and column identifier.

        Returns:
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
            match: Regex match containing quantile function name, column identifier, and quantile literal.

        Returns:
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
            match: Regex match containing column identifier and target numeric SQL type.

        Returns:
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


_SQL_COMMENT_OR_QUOTED_RE = re.compile(
    r"'(?:''|[^'])*'|\"(?:\"\"|[^\"])*\"|--[^\n]*|/\*(?:(?!\*/|/\*).)*\*/",
    re.DOTALL,
)


def _has_enclosing_outer_parens(text: str) -> bool:
    """
    Check whether index `0` `'('` and index `-1` `')'` form a single balanced pair enclosing `text`.

    Prevents stripping outer parentheses on compound queries such as `(SELECT 1) UNION ALL (SELECT 2)`.

    Args:
        text: Trimmed SQL candidate string.

    Returns:
        bool: True only when `text[0]` closes at `text[-1]` with balanced parentheses.

    """
    if len(text) < 2 or text[0] != "(" or text[-1] != ")":
        return False
    masked = strip_sql_literals_and_identifiers(text)
    depth = 0
    last_idx = len(masked) - 1
    for idx, ch in enumerate(masked):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0 and idx < last_idx:
                return False
    return depth == 0


def strip_sql_comments_and_parens(query: str) -> str:
    """
    Strip SQL line/block comments outside quoted literals and unwrap balanced enclosing parentheses.

    Args:
        query: Raw SQL or user input string.

    Returns:
        str: Cleaned query string preserving compound parenthesized set operations.

    """

    def _replace_comment(match: re.Match[str]) -> str:
        """
        Preserve quoted SQL literals/identifiers while replacing comment tokens with a single space.

        Args:
            match: Regex match object for a quoted token or SQL comment.

        Returns:
            str: Unchanged quoted token or `' '` for comments.

        """
        token = match.group(0)
        if token[0] in ("'", '"'):
            return token
        return " "

    cleaned = query
    prev = None
    while prev != cleaned:
        prev = cleaned
        cleaned = _SQL_COMMENT_OR_QUOTED_RE.sub(_replace_comment, cleaned)
    cleaned = cleaned.strip()
    while _has_enclosing_outer_parens(cleaned):
        cleaned = cleaned[1:-1].strip()
    return cleaned


_SAFE_READ_ONLY_PRAGMAS = frozenset(
    {
        "collations",
        "database_list",
        "database_size",
        "functions",
        "metadata_info",
        "platform",
        "show_tables",
        "show_tables_expanded",
        "storage_info",
        "table_info",
        "types",
        "user_agent",
        "version",
    }
)

_PRAGMA_NAME_RE = re.compile(
    r"^pragma\s+([a-zA-Z_][a-zA-Z0-9_]*)(?:\s*\([^;]*\))?\s*;?$", re.IGNORECASE
)

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
    r"COPY\b|"
    r"ATTACH\b|"
    r"DETACH\b|"
    r"EXPORT\s+DATABASE\b|"
    r"IMPORT\s+DATABASE\b|"
    r"(?:FORCE\s+)?INSTALL\s+\S+|"
    r"LOAD\s+\S+|"
    r"SET\s+(?:GLOBAL\s+|SESSION\s+|LOCAL\s+)?[A-Za-z_][A-Za-z0-9_]*\s*(?:=|TO\b)|"
    r"RESET\s+[A-Za-z_][A-Za-z0-9_]*|"
    r"CALL\s+[A-Za-z_]|"
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
        query: SQL query string.

    Returns:
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
        query: SQL query string to inspect.

    Returns:
        bool: True if the statement is a single read-only SQL query, False otherwise.

    """
    unwrapped = strip_sql_comments_and_parens(query)
    core = unwrapped.lstrip("(").strip()
    if not core:
        return False
    first_word = core.split()[0].lower().rstrip(";").strip()
    if first_word not in READ_ONLY_SQL_PREFIXES:
        return False

    without_literals = strip_sql_literals_and_identifiers(unwrapped)
    segments = [seg.strip() for seg in without_literals.split(";") if seg.strip()]
    if len(segments) > 1:
        return False

    if first_word == "pragma":
        pragma_match = _PRAGMA_NAME_RE.match(without_literals.strip())
        if (
            pragma_match is None
            or pragma_match.group(1).lower() not in _SAFE_READ_ONLY_PRAGMAS
        ):
            return False

    return _FORBIDDEN_WRITE_STATEMENT_RE.search(without_literals) is None


def validate_read_only_sql(query: str) -> str:
    """
    Validate that a SQL query is non-empty, contains a single statement, and is read-only.

    Args:
        query: SQL query string to validate.

    Returns:
        str: Stripped, validated SQL query string.

    Raises:
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

VALID_SHOW_SECOND_TOKENS = {
    "tables",
    "databases",
    "schemas",
    "variables",
    "all",
    "columns",
}

_VALID_EXPLAIN_SECOND_TOKENS = {
    "select",
    "with",
    "analyze",
    "from",
    "show",
    "summarize",
    "describe",
    "values",
    "table",
}

_FROM_NL_DETERMINERS = {
    "the",
    "a",
    "an",
    "our",
    "all",
    "each",
    "patients",
    "subjects",
}

_WITH_CTE_PREFIX_RE = re.compile(
    r'^with\s+(?:recursive\s+)?[a-zA-Z_"][a-zA-Z0-9_"]*\s+as\s*\(',
    re.IGNORECASE,
)

_SELECT_NL_IMPERATIVE_RE = re.compile(
    r"^select\s+(?:patients|subjects|participants|all\s+patients|the\s+|a\s+|an\s+|which\s+|who\s+)\b",
    re.IGNORECASE,
)


def _is_disqualified_describe_or_summarize(
    first_word: str,
    second_word: str,
    tokens: Sequence[str],
) -> bool:
    """
    Determine whether a `DESCRIBE` or `SUMMARIZE` candidate is natural-language prose.

    Args:
        first_word: Lowercased first token (`"describe"` or `"summarize"`).
        second_word: Lowercased second token without trailing semicolons.
        tokens: Whitespace-split tokens of the input query.

    Returns:
        bool: True if the input is natural language rather than a valid `DESCRIBE` or `SUMMARIZE` query.

    """
    if len(tokens) == 2 and second_word:
        return False
    if first_word == "summarize" and second_word == "select":
        return False
    return not (first_word == "describe" and second_word in ("select", "table"))


def _is_disqualified_from_clause(
    unwrapped: str,
    second_word: str,
    tokens: Sequence[str],
) -> bool:
    """
    Determine whether a `FROM` candidate is natural-language prose rather than DuckDB `FROM` shorthand.

    Args:
        unwrapped: Comment-stripped input query string.
        second_word: Lowercased second token without trailing semicolons.
        tokens: Whitespace-split tokens of `unwrapped`.

    Returns:
        bool: True if the input is natural language rather than a valid DuckDB `FROM` query.

    """
    if not second_word:
        return True
    if second_word in _FROM_NL_DETERMINERS and (
        len(tokens) > 2 or second_word not in ("patients", "subjects")
    ):
        has_trailing_select = any(
            tok.lower().rstrip(";") == "select" for tok in tokens[2:]
        )
        return not (has_trailing_select or ";" in unwrapped)
    return False


def has_natural_language_sql_disqualifier(
    unwrapped: str,
    first_word: str,
    tokens: Sequence[str],
) -> bool:
    """
    Check if a string starting with a SQL keyword exhibits natural-language English prose patterns.

    Args:
        unwrapped: Comment-stripped input query string.
        first_word: Lowercased first token without trailing semicolons.
        tokens: Whitespace-split tokens of `unwrapped`.

    Returns:
        bool: True if the input matches a natural-language disqualifier and cannot be literal SQL.

    """
    second_word = tokens[1].lower().rstrip(";").strip() if len(tokens) > 1 else ""
    third_word = tokens[2].lower().rstrip(";").strip() if len(tokens) > 2 else ""

    if first_word == "show":
        if not second_word or second_word not in VALID_SHOW_SECOND_TOKENS:
            return True
        return second_word == "all" and third_word != "tables"

    if first_word == "table":
        return len(tokens) != 2 or not second_word or second_word[0].isdigit()

    if first_word == "with":
        return _WITH_CTE_PREFIX_RE.match(unwrapped) is None

    if first_word in ("summarize", "describe"):
        return _is_disqualified_describe_or_summarize(first_word, second_word, tokens)

    if first_word == "explain":
        return not second_word or second_word not in _VALID_EXPLAIN_SECOND_TOKENS

    if first_word == "from":
        return _is_disqualified_from_clause(unwrapped, second_word, tokens)

    if first_word == "select" and _SELECT_NL_IMPERATIVE_RE.match(unwrapped):
        return not any(tok.lower().rstrip(";") == "from" for tok in tokens[1:])

    return False


@runtime_checkable
class SqlSyntaxValidatorProtocol(Protocol):
    """Protocol for DuckDB connection objects capable of parsing SQL statements."""

    def extract_statements(self, query: str) -> Sequence[duckdb.Statement]:
        """
        Parse a SQL string into DuckDB statement AST objects.

        Args:
            query: Raw SQL query string.

        Returns:
            Sequence[duckdb.Statement]: Parsed statement list.

        """
        ...


def check_sql_syntax_with_conn(
    unwrapped: str,
    first_word: str,
    conn: SqlSyntaxValidatorProtocol | None = None,
) -> bool:
    """
    Validate SQL syntax using either the provided `conn` or a cached `:memory:` DuckDB parser connection.

    Args:
        unwrapped: Comment-stripped SQL string.
        first_word: Lowercased first token of `unwrapped`.
        conn: Optional explicit DuckDB connection implementing `SqlSyntaxValidatorProtocol`.

    Returns:
        bool: True if DuckDB parses the statement without a `ParserException`.

    """
    _ = first_word
    try:
        if isinstance(conn, duckdb.DuckDBPyConnection):
            return len(conn.extract_statements(unwrapped)) > 0

        if isinstance(conn, SqlSyntaxValidatorProtocol):
            return len(conn.extract_statements(unwrapped)) > 0

        with _PARSER_LOCK:
            parser_conn = _get_cached_parser_conn()
            return len(parser_conn.extract_statements(unwrapped)) > 0
    except duckdb.ParserException:
        return False
    except (duckdb.Error, RuntimeError, ValueError, OSError):
        return True


SqlInputKind = Literal[
    "natural_language",
    "valid_read_only_sql",
    "malformed_read_only_sql",
    "non_read_only_sql",
]


def classify_sql_input(
    query: str,
    conn: SqlSyntaxValidatorProtocol | None = None,
) -> SqlInputKind:
    """
    Classify raw user input into natural language vs. AST-verified read-only/malformed/mutating SQL.

    Consolidates lexical comment/paren normalization, English prose disqualifiers,
    read-only policy guards, and DuckDB AST syntax parsing into one authoritative classifier.

    Args:
        query: Raw input string entered by the user.
        conn: Optional DuckDB connection or `SqlSyntaxValidatorProtocol` implementation.

    Returns:
        SqlInputKind: One of `'natural_language'`, `'valid_read_only_sql'`,
            `'malformed_read_only_sql'`, or `'non_read_only_sql'`.

    """
    unwrapped = strip_sql_comments_and_parens(query)
    core = unwrapped.lstrip("(").strip()
    if not core:
        return "natural_language"

    tokens = core.split()
    first_word = tokens[0].lower().rstrip(";").strip()
    if first_word not in ALL_SQL_KEYWORDS:
        return "natural_language"

    if has_natural_language_sql_disqualifier(core, first_word, tokens):
        return "natural_language"

    is_ast_valid = check_sql_syntax_with_conn(unwrapped, first_word, conn)
    if first_word in READ_ONLY_SQL_PREFIXES and is_read_only_sql_query(unwrapped):
        return "valid_read_only_sql" if is_ast_valid else "malformed_read_only_sql"

    if is_ast_valid or first_word in READ_ONLY_SQL_PREFIXES:
        return "non_read_only_sql"

    return "natural_language"


def is_valid_read_only_sql(
    query: str,
    conn: SqlSyntaxValidatorProtocol | None = None,
) -> bool:
    """
    Return True when `query` is an AST-valid, single-statement read-only SQL query.

    Args:
        query: Raw user query string.
        conn: Optional DuckDB syntax validator connection.

    Returns:
        bool: True iff `classify_sql_input(query, conn)` equals `'valid_read_only_sql'`.

    """
    return classify_sql_input(query, conn=conn) == "valid_read_only_sql"


def is_sql_query(query: str, conn: SqlSyntaxValidatorProtocol | None = None) -> bool:
    """
    Determine whether a user input string is an AST-valid SQL statement rather than natural language.

    Args:
        query: The raw query string entered by the user.
        conn: Optional DuckDB connection to assist in query syntax verification.

    Returns:
        bool: True if the query is an AST-valid SQL statement, False otherwise.

    """
    kind = classify_sql_input(query, conn=conn)
    if kind == "valid_read_only_sql":
        return True
    if kind == "non_read_only_sql":
        unwrapped = strip_sql_comments_and_parens(query)
        first_word = (
            unwrapped.lstrip("(").strip().split()[0].lower().rstrip(";").strip()
        )
        return check_sql_syntax_with_conn(unwrapped, first_word, conn)
    return False


def _clean_extracted_sql_candidate(raw_sql: str) -> str:
    """
    Strip leading markdown language markers (`sql`, `duckdb`, `sql ...`, `...`) and DDL preambles when followed by SELECT/WITH.

    Args:
        raw_sql: Candidate SQL block text.

    Returns:
        str: Cleaned SQL query string.

    """
    cleaned = raw_sql.strip()
    if cleaned.startswith("-- Synthesized schema-guided fallback SQL"):
        non_comment_lines = [
            line
            for line in cleaned.splitlines()
            if not line.strip().startswith("-- Synthesized schema-guided fallback SQL")
        ]
        cleaned = "\n".join(non_comment_lines).strip()
    cleaned = re.sub(
        r"^(?:(?:sql|duckdb)(?:[ \t]+\.{2,})?|\.{2,})(?:[ \t]*[\r\n]+|[ \t]+)",
        "",
        cleaned,
        flags=re.IGNORECASE,
    ).strip()
    ddl_with_select = re.search(
        r"^(?:--[^\r\n]*[\r\n]+\s*)*(?:CREATE\s+(?:OR\s+REPLACE\s+)?(?:TEMP(?:ORARY)?\s+)?TABLE\b[\s\S]*?;+\s*(?:--[^\r\n]*[\r\n]+\s*)*)+((?:WITH|SELECT)\b[\s\S]*)$",
        cleaned,
        flags=re.IGNORECASE,
    )
    if ddl_with_select:
        return ddl_with_select.group(1).strip()
    return cleaned


def extract_sql_from_response(full_response: SqlValue) -> str:
    """
    Extract a SQL query from markdown code blocks or plain text in an LLM completion.

    Args:
        full_response: Raw text response or scalar value from the LLM.

    Returns:
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
        sql_candidate = _clean_extracted_sql_candidate(blocks[-1])
        for block in reversed(blocks):
            cleaned_block = _clean_extracted_sql_candidate(block)
            uncommented = strip_sql_comments_and_parens(cleaned_block)
            if uncommented.lower().startswith(
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
                sql_candidate = cleaned_block
                break
        return str(sql_candidate)

    unclosed_fence = re.search(
        r"```(?:sql|duckdb)?[ \t]*[\r\n]+([\s\S]+)$",
        full_response,
        flags=re.IGNORECASE,
    )
    if unclosed_fence:
        return _clean_extracted_sql_candidate(
            unclosed_fence.group(1).replace("```", "")
        )

    sql_query = full_response.replace("```", "").strip()
    return _clean_extracted_sql_candidate(sql_query)


def sanitize_sql_value(val: SqlRawInputValue) -> SqlValue:
    """
    Sanitize an arbitrary database column value into a JSON-compliant SqlValue.

    Args:
        val: The raw value returned from the database driver.

    Returns:
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


_SAFE_SIGNED_DECIMAL_RE = re.compile(r"^[+-](?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")


def sanitize_spreadsheet_cell(val: SqlRawInputValue) -> SqlValue:
    """
    Sanitize a cell value against spreadsheet formula injection (CSV/XLSX).

    Leading characters '=', '+', '-', '@', tab, carriage-return, and line-feed
    (even after leading spaces) on non-numeric strings are prefixed with a single
    quote so spreadsheet engines treat them as text.

    Args:
        val: Raw database value.

    Returns:
        SqlValue: Safe cell value for spreadsheet export.

    """
    if val is None:
        return None
    if isinstance(val, (int, float, bool)):
        return val
    s_val = str(val)
    if not s_val:
        return s_val
    first_code = ord(s_val[0])
    stripped_leading = s_val.lstrip(" \u00a0")
    trigger_char = stripped_leading[0] if stripped_leading else ""
    trigger_code = ord(trigger_char) if trigger_char else 32
    if first_code < 32 or trigger_code < 32 or trigger_char in ("=", "+", "-", "@"):
        is_numeric = False
        if trigger_char in ("+", "-") and _SAFE_SIGNED_DECIMAL_RE.match(
            stripped_leading
        ):
            parsed_num = float(stripped_leading)
            is_numeric = math.isfinite(parsed_num)
        if not is_numeric:
            s_val = f"'{s_val}"
    return s_val


INTERNAL_TABLE_NAMES = {"chat_sessions", "pretrain", "sft", "dpo"}
INTERNAL_TABLE_PREFIXES = ("_t1d_", "pretrain_", "sft_", "dpo_")


def is_internal_table(table_name: str, exclude_training_tables: bool = True) -> bool:
    """
    Check if a DuckDB table is an internal manifest, session, or training artifact table.

    Args:
        table_name: Name of the table to inspect.
        exclude_training_tables: Whether to also exclude chat_sessions and sft/dpo/pretrain tables.

    Returns:
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
    conn: duckdb.DuckDBPyConnection | ReadOnlyDuckDbCursorProxy,
    exclude_internal: bool = False,
    exclude_training_tables: bool = False,
) -> dict[str, list[tuple[str, str]]]:
    """
    Extract ordered `(column_name, column_type)` tuples per table from an active DuckDB connection or cursor.

    Args:
        conn: Active DuckDB connection or read-only cursor proxy.
        exclude_internal: Whether to exclude `_t1d_` internal tables.
        exclude_training_tables: Whether to also exclude `chat_sessions` and training tables.

    Returns:
        dict[str, list[tuple[str, str]]]: Mapping from table name to list of `(column_name, column_type)` pairs.

    """
    cursor_proxy = (
        conn
        if isinstance(conn, ReadOnlyDuckDbCursorProxy)
        else create_read_only_cursor_proxy(conn)
    )
    should_close_proxy = not isinstance(conn, ReadOnlyDuckDbCursorProxy)
    try:
        result: dict[str, list[tuple[str, str]]] = {}
        tables = cursor_proxy.execute("SHOW TABLES").fetchall()
        for row in tables:
            table_name = str(row[0])
            if (exclude_internal or exclude_training_tables) and is_internal_table(
                table_name, exclude_training_tables=exclude_training_tables
            ):
                continue
            escaped_name = escape_sql_identifier(table_name)
            columns = cursor_proxy.execute(f'DESCRIBE "{escaped_name}"').fetchall()
            result[table_name] = [(str(col[0]), str(col[1])) for col in columns]
        return result
    finally:
        if should_close_proxy:
            cursor_proxy.close()


def extract_table_schemas(
    conn: duckdb.DuckDBPyConnection | ReadOnlyDuckDbCursorProxy,
    exclude_training_tables: bool = True,
) -> dict[str, str]:
    """
    Extract per-table textual schema descriptions from an active DuckDB connection or cursor.

    Args:
        conn: Active DuckDB connection or read-only cursor proxy.
        exclude_training_tables: Whether to exclude internal chat_sessions and training tables.

    Returns:
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
        schema_text: Raw SQL DDL string.

    Returns:
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
) -> str | None:
    """
    Parse a `Table: ...` line and update `schema_pairs`, returning the active multi-line table name if any.

    Args:
        after_table: Substring following `Table:`.
        schema_pairs: Mutable mapping of table names to `(column_name, column_type)` lists.

    Returns:
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
        schema_text: Formatted database schema or SQL DDL string.

    Returns:
        dict[str, list[tuple[str, str]]]: Mapping of table names to `(column_name, column_type)` lists.

    """
    schema_pairs: dict[str, list[tuple[str, str]]] = {}
    current_table: str | None = None
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
        schema_text: Formatted database schema or SQL DDL string.

    Returns:
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
    "JAEB_ATTACH_CANDIDATES",
    "READ_ONLY_SQL_PREFIXES",
    "VALID_SHOW_SECOND_TOKENS",
    "PoolSqlRawInputValue",
    "PoolSqlValue",
    "ReadOnlyDuckDBPool",
    "ReadOnlyDuckDbCursorProxy",
    "SqlInputKind",
    "SqlRawInputValue",
    "SqlSyntaxValidatorProtocol",
    "SqlValue",
    "check_sql_syntax_with_conn",
    "classify_sql_input",
    "close_cached_parser_conn",
    "close_read_only_duckdb_pool",
    "coerce_numeric_aggregates_in_sql",
    "connect_duckdb_read_only_cursor",
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
    "has_natural_language_sql_disqualifier",
    "is_internal_table",
    "is_read_only_sql_query",
    "is_sql_query",
    "is_valid_read_only_sql",
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
