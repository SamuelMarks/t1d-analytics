"""Unit tests for shared SQL utilities in t1d_analytics.sql_utils."""

import datetime
import decimal
import uuid
from pathlib import Path
from unittest.mock import patch

import duckdb
import pytest

from t1d_analytics.sql_utils import (
    coerce_numeric_aggregates_in_sql,
    connect_duckdb_with_retry,
    extract_sql_from_response,
    extract_table_schemas,
    is_internal_table,
    is_read_only_sql_query,
    is_sql_query,
    normalize_sql,
    open_sandboxed_duckdb,
    sanitize_spreadsheet_cell,
    sanitize_sql_value,
    strip_sql_comments_and_parens,
    validate_read_only_sql,
)


def test_normalize_sql_and_strip_comments() -> None:
    """Test normalize_sql, coerce_numeric_aggregates_in_sql, and strip_sql_comments_and_parens."""
    assert normalize_sql("  SELECT   *  FROM  patients ; ") == "select * from patients"
    coerced = coerce_numeric_aggregates_in_sql(
        "SELECT SUM(NumSevHypo), AVG(DISTINCT t.AgeAsOfRandDt), "
        "STDDEV_SAMP(hba1c), MEDIAN(tir), QUANTILE_CONT(tbr, 0.25), "
        "SUM(COALESCE(NumSevHypo, 0)), AVG(DISTINCT IFNULL(t.AgeAsOfRandDt, 0)), "
        "SUM((NumSevHypo)), CAST(NumSevHypo AS INT), CAST(t.AgeAsOfRandDt AS DOUBLE) "
        "FROM tblaptsummary t"
    )
    assert "SUM(TRY_CAST(NumSevHypo AS DOUBLE))" in coerced
    assert "AVG(DISTINCT TRY_CAST(t.AgeAsOfRandDt AS DOUBLE))" in coerced
    assert "STDDEV_SAMP(TRY_CAST(hba1c AS DOUBLE))" in coerced
    assert "MEDIAN(TRY_CAST(tir AS DOUBLE))" in coerced
    assert "QUANTILE_CONT(TRY_CAST(tbr AS DOUBLE), 0.25)" in coerced
    assert "SUM(COALESCE(TRY_CAST(NumSevHypo AS DOUBLE), 0))" in coerced
    assert "AVG(DISTINCT IFNULL(TRY_CAST(t.AgeAsOfRandDt AS DOUBLE), 0))" in coerced
    assert "TRY_CAST(NumSevHypo AS INT)" in coerced
    assert "TRY_CAST(t.AgeAsOfRandDt AS DOUBLE)" in coerced
    # Verify idempotency
    assert coerce_numeric_aggregates_in_sql(coerced) == coerced
    assert (
        strip_sql_comments_and_parens("""-- comment
(/* block */ (SELECT 1))""")
        == "SELECT 1"
    )
    assert (
        strip_sql_comments_and_parens("/* outer /* nested */ outer */ SELECT 1")
        == "SELECT 1"
    )
    assert (
        strip_sql_comments_and_parens("""   -- only comment
""")
        == ""
    )


def test_is_read_only_sql_query() -> None:
    """Test read-only SQL classification."""
    assert is_read_only_sql_query("SELECT * FROM tbl") is True
    assert is_read_only_sql_query("WITH cte AS (SELECT 1) SELECT * FROM cte") is True
    assert is_read_only_sql_query("((SELECT 1))") is True
    assert (
        is_read_only_sql_query(
            "SELECT REPLACE(name, 'a', 'b'), 'DROP TABLE tbl' AS msg FROM tbl"
        )
        is True
    )
    assert (
        is_read_only_sql_query("""   -- comment
""")
        is False
    )
    assert is_read_only_sql_query("DROP TABLE tbl") is False
    assert is_read_only_sql_query("INSERT INTO tbl VALUES (1)") is False
    assert is_read_only_sql_query("SELECT 1; DROP TABLE tbl") is False
    assert (
        is_read_only_sql_query("WITH d AS (DELETE FROM tbl) SELECT * FROM d") is False
    )
    assert is_read_only_sql_query("SELECT $$DROP TABLE tbl;$$ AS msg FROM tbl") is True
    assert (
        is_read_only_sql_query("SELECT * FROM read_parquet('secret.parquet')") is False
    )
    assert is_read_only_sql_query("SELECT * FROM read_csv_auto('secret.csv')") is False


def test_is_sql_query_branches() -> None:
    """Test all branches of is_sql_query including comments, parens, show NL, real connections, and protocol fakes."""
    from collections.abc import Sequence

    from t1d_analytics.sql_utils import (
        SqlSyntaxValidatorProtocol,
        classify_sql_input,
        is_valid_read_only_sql,
    )

    assert is_sql_query("") is False
    assert is_sql_query("()") is False
    assert is_sql_query("((SELECT 1)") is False
    assert is_sql_query("(SELECT 1) UNION ALL (SELECT 2)") is True
    assert is_valid_read_only_sql("(SELECT 1) UNION ALL (SELECT 2)") is True
    assert is_valid_read_only_sql("SELECT '--not a comment' AS lbl") is True
    assert classify_sql_input("SELECT FROM WHERE INVALID") == "malformed_read_only_sql"
    assert classify_sql_input("Create a Table 1 for patients") == "natural_language"
    assert is_sql_query("What is the average glucose?") is False
    assert is_sql_query("Show me the patients") is False
    assert is_sql_query("SHOW") is False
    assert is_sql_query("SHOW ALL") is False
    assert is_sql_query("SHOW ALL TABLES") is True
    assert is_sql_query("SHOW TABLES") is True
    assert is_sql_query("TABLE patients;") is True
    assert is_sql_query("Table 1 comparing Closed-Loop vs Control") is False
    assert is_sql_query("TABLE") is False
    assert is_sql_query("WITH cte AS (SELECT 1) SELECT * FROM cte") is True
    assert is_sql_query("With Closed-Loop vs Control patients, what is TIR?") is False
    assert is_sql_query("SUMMARIZE patients;") is True
    assert is_sql_query("SUMMARIZE SELECT * FROM patients;") is True
    assert is_sql_query("Summarize baseline HbA1c by arm") is False
    assert is_sql_query("DESCRIBE patients;") is True
    assert is_sql_query("DESCRIBE TABLE patients;") is True
    assert is_sql_query("DESCRIBE SELECT * FROM patients;") is True
    assert is_sql_query("Describe the cohort demographics") is False
    assert is_sql_query("EXPLAIN SELECT 1") is True
    assert is_sql_query("Explain the relationship between HbA1c and TIR") is False
    assert is_sql_query("FROM") is False
    assert is_sql_query("FROM patients") is True
    assert is_sql_query("FROM patients in cohort") is False
    assert is_sql_query("From the patients in our cohort") is False
    assert is_sql_query("FROM patients SELECT *") is True
    assert is_sql_query("FROM patients;") is True
    assert is_sql_query("SELECT 1") is True
    assert is_sql_query("SELECT * FROM patients") is True
    assert is_sql_query("Select patients who had DKA events.") is False
    assert is_sql_query("SELECT patients FROM cohort") is True
    assert is_sql_query("SELECT FROM WHERE INVALID") is False

    real_conn = duckdb.connect(":memory:")
    try:
        assert is_sql_query("CREATE TABLE x (id INT)", conn=real_conn) is True
        assert is_sql_query("SELECT 1", conn=real_conn) is True
        assert is_sql_query("SELECT FROM WHERE INVALID", conn=real_conn) is False
    finally:
        real_conn.close()

    class _RuntimeErrorValidator(SqlSyntaxValidatorProtocol):
        """Protocol fake raising RuntimeError during statement extraction."""

        def extract_statements(self, query: str) -> Sequence[duckdb.Statement]:
            """Raise RuntimeError to verify non-ParserException fallback."""
            _ = query
            raise RuntimeError("other db error")

    class _ListReturningValidator(SqlSyntaxValidatorProtocol):
        """Protocol fake returning a concrete statement tuple."""

        def extract_statements(self, query: str) -> Sequence[duckdb.Statement]:
            """Delegate to in-memory DuckDB connection and return tuple."""
            with duckdb.connect(":memory:") as tmp_c:
                return tuple(tmp_c.extract_statements(query))

    from unittest.mock import MagicMock

    assert is_sql_query("SELECT 1", conn=_RuntimeErrorValidator()) is True
    assert is_sql_query("SELECT 1", conn=_ListReturningValidator()) is True
    assert is_sql_query("SELECT 1", conn=MagicMock()) is True


def test_extract_sql_from_response_branches() -> None:
    """Test SQL extraction from fenced blocks, non-strings, and plain text."""
    assert extract_sql_from_response(12345) == "12345"
    fenced = """Thought
```sql
SELECT * FROM p;
```
```text
summary
```"""
    assert extract_sql_from_response(fenced) == "SELECT * FROM p;"
    fenced_non_sql = """```json
{"a": 1}
```"""
    assert extract_sql_from_response(fenced_non_sql) == '{"a": 1}'
    assert (
        extract_sql_from_response("""sql
SELECT 1""")
        == "SELECT 1"
    )
    assert (
        extract_sql_from_response("""duckdb
SELECT 2""")
        == "SELECT 2"
    )
    assert extract_sql_from_response("SELECT 3") == "SELECT 3"


def test_sanitize_sql_value_and_spreadsheet_cell() -> None:
    """Test value sanitization for JSON and spreadsheet formula-injection defense."""
    assert sanitize_sql_value(None) is None
    assert sanitize_sql_value(datetime.date(2026, 1, 2)) == "2026-01-02"
    assert sanitize_sql_value(decimal.Decimal("12.5")) == 12.5
    u = uuid.uuid4()
    assert sanitize_sql_value(u) == str(u)
    assert sanitize_sql_value(float("nan")) is None
    assert sanitize_sql_value(float("inf")) is None
    assert sanitize_sql_value(3.14) == 3.14
    assert sanitize_sql_value(42) == 42
    assert sanitize_sql_value(True) is True
    assert sanitize_sql_value("hello") == "hello"
    assert sanitize_sql_value(["a", "b"]) == "['a', 'b']"

    assert sanitize_spreadsheet_cell(None) is None
    assert sanitize_spreadsheet_cell(10) == 10
    assert sanitize_spreadsheet_cell(True) is True
    assert sanitize_spreadsheet_cell("normal") == "normal"
    assert sanitize_spreadsheet_cell("") == ""
    assert sanitize_spreadsheet_cell("=SUM(A1:A2)") == "'=SUM(A1:A2)"
    assert sanitize_spreadsheet_cell("@cmd") == "'@cmd"
    assert sanitize_spreadsheet_cell("+cmd") == "'+cmd"
    assert sanitize_spreadsheet_cell("-cmd") == "'-cmd"
    assert sanitize_spreadsheet_cell("+12.5") == "+12.5"
    assert sanitize_spreadsheet_cell("-42") == "-42"


def test_open_sandboxed_duckdb(tmp_path: Path) -> None:
    """Test open_sandboxed_duckdb context manager."""
    db_file = tmp_path / "sandbox.duckdb"
    with open_sandboxed_duckdb(
        db_file, read_only=False, disable_external_access=False
    ) as conn:
        conn.execute("CREATE TABLE t (x INT)")
        conn.execute("CREATE TABLE chat_sessions (id VARCHAR)")
        conn.execute("INSERT INTO t VALUES (7)")

    with open_sandboxed_duckdb(
        db_file, read_only=True, disable_external_access=True
    ) as conn:
        row = conn.execute("SELECT x FROM t").fetchone()
        assert row == (7,)
        with pytest.raises(duckdb.Error):
            conn.execute("DROP TABLE t")
        schemas = extract_table_schemas(conn)
        assert "t" in schemas
        assert "chat_sessions" not in schemas

    assert is_internal_table("_t1d_ingestion_manifest") is True
    assert is_internal_table("chat_sessions", exclude_training_tables=True) is True
    assert is_internal_table("chat_sessions", exclude_training_tables=False) is False
    assert is_internal_table("patients") is False


def test_validate_read_only_sql() -> None:
    """Test validate_read_only_sql accepts valid read-only queries and rejects empty, multi-statement, and mutating SQL."""
    from t1d_analytics.errors import (
        EmptyQueryError,
        MultipleStatementsError,
        ReadOnlyViolationError,
    )
    from t1d_analytics.sql_utils import escape_sql_literal

    assert escape_sql_literal("O'Brien") == "O''Brien"

    assert (
        validate_read_only_sql("  SELECT * FROM patients WHERE name = 'a;b'; ")
        == "SELECT * FROM patients WHERE name = 'a;b';"
    )
    assert validate_read_only_sql("PRAGMA version;") == "PRAGMA version;"
    assert (
        validate_read_only_sql("PRAGMA table_info('patients')")
        == "PRAGMA table_info('patients')"
    )

    with pytest.raises(EmptyQueryError):
        validate_read_only_sql("   -- only a comment\n ")

    for multi_vec in (
        "SELECT 1; DROP TABLE patients;",
        "SELECT 1; ATTACH 'evil.db'",
        "SELECT 1; COPY patients TO '/tmp/leak.csv'",
    ):
        with pytest.raises(MultipleStatementsError) as multi_exc:
            validate_read_only_sql(multi_vec)
        assert multi_exc.value.error_code == "backend.sqlMultipleStatements"

    for forbidden_vec in (
        "DROP TABLE patients",
        "/*! DROP TABLE patients */",
        "-- comment\nDROP TABLE patients",
        "ATTACH ':memory:' AS extra",
        "DETACH extra",
        "COPY patients TO 'out.csv'",
        "EXPORT DATABASE 'out_dir'",
        "IMPORT DATABASE 'in_dir'",
        "LOAD httpfs",
        "INSTALL httpfs",
        "SET memory_limit = '1GB'",
        "RESET memory_limit",
        "CALL duckdb_tables()",
        "PRAGMA enable_profiling",
        "PRAGMA memory_limit='1GB'",
        "WITH sneaky AS (SELECT 1) COPY sneaky TO 'leak.csv'",
        "SELECT * FROM read_csv_auto('/etc/passwd')",
        "SELECT * FROM read_parquet('secret.parquet')",
        "SELECT * FROM glob('*')",
        "WITH mutated AS (DELETE FROM patients RETURNING *) SELECT * FROM mutated",
        "WITH ins AS (INSERT INTO patients VALUES (1)) SELECT * FROM ins",
    ):
        with pytest.raises(ReadOnlyViolationError):
            validate_read_only_sql(forbidden_vec)


def test_execute_duckdb_batch_write_with_retry(tmp_path: Path) -> None:
    """Test execute_duckdb_batch_write_with_retry and execute_duckdb_write_with_retry."""
    from unittest.mock import patch

    from t1d_analytics.sql_utils import (
        execute_duckdb_batch_write_with_retry,
        execute_duckdb_write_with_retry,
    )

    db_file = tmp_path / "batch_write.duckdb"
    execute_duckdb_batch_write_with_retry(
        db_file,
        [
            ("CREATE TABLE items (id INT, label VARCHAR)", None),
            ("INSERT INTO items VALUES (?, ?)", [1, "alpha"]),
        ],
    )
    execute_duckdb_write_with_retry(
        db_file,
        "INSERT INTO items VALUES (?, ?)",
        params=[2, "beta"],
    )

    conn = duckdb.connect(str(db_file), read_only=True)
    rows = conn.execute("SELECT id, label FROM items ORDER BY id").fetchall()
    conn.close()
    assert rows == [(1, "alpha"), (2, "beta")]

    # Atomic transaction rollback when second statement in batch fails
    with pytest.raises(duckdb.Error):
        execute_duckdb_batch_write_with_retry(
            db_file,
            [
                ("INSERT INTO items VALUES (?, ?)", [3, "rolled_back"]),
                ("INSERT INTO nonexistent_table VALUES (1)", None),
            ],
        )
    conn_after_rb = duckdb.connect(str(db_file), read_only=True)
    rows_after_rb = conn_after_rb.execute(
        "SELECT id, label FROM items ORDER BY id"
    ).fetchall()
    conn_after_rb.close()
    assert rows_after_rb == [(1, "alpha"), (2, "beta")]

    # Test lock retry and DatabaseLockedError exhaustion
    from t1d_analytics.errors import DatabaseLockedError
    from t1d_analytics.sql_utils import close_cached_parser_conn, is_sql_query

    assert is_sql_query("SELECT 1") is True
    close_cached_parser_conn()
    close_cached_parser_conn()
    assert is_sql_query("SELECT 1") is True

    delays: list[float] = []
    with patch(
        "duckdb.connect",
        side_effect=duckdb.IOException("Could not set lock on file"),
    ):
        with pytest.raises(DatabaseLockedError, match="database is locked"):
            execute_duckdb_write_with_retry(
                db_file,
                "SELECT 1",
                max_attempts=2,
                base_delay_seconds=0.01,
                sleep_fn=delays.append,
            )
    assert len(delays) == 2

    # Test non-lock IOException re-raises immediately
    with patch(
        "duckdb.connect",
        side_effect=duckdb.IOException("Disk full error"),
    ):
        with pytest.raises(duckdb.IOException, match="Disk full"):
            execute_duckdb_write_with_retry(db_file, "SELECT 1", max_attempts=2)

    # Leading whitespace formula injection defense
    assert sanitize_spreadsheet_cell("  =1+1") == "'  =1+1"
    assert sanitize_spreadsheet_cell(" @SUM(A1:A2)") == "' @SUM(A1:A2)"
    assert sanitize_spreadsheet_cell("   ") == "   "
    assert sanitize_spreadsheet_cell("") == ""

    # connect_duckdb_with_retry: lock retry and clean caller exception propagation
    connect_calls = 0
    real_connect = duckdb.connect

    def _flaky_connect(path: str, read_only: bool = True) -> duckdb.DuckDBPyConnection:
        """Fail once with lock error then connect."""
        nonlocal connect_calls
        connect_calls += 1
        if connect_calls == 1:
            raise duckdb.IOException("Conflicting lock on file")
        return real_connect(path, read_only=read_only)

    slept_intervals: list[float] = []
    with patch("duckdb.connect", side_effect=_flaky_connect):
        with connect_duckdb_with_retry(
            db_file, read_only=True, max_attempts=3, sleep_fn=slept_intervals.append
        ) as conn_retry:
            row = conn_retry.execute("SELECT 42").fetchone()
            assert row == (42,)
    assert len(slept_intervals) == 1

    # Verify an IOException raised INSIDE the caller block propagates cleanly without RuntimeError
    with pytest.raises(duckdb.IOException, match="Caller query IO failure"):
        with connect_duckdb_with_retry(db_file, read_only=True) as conn_inner:
            _ = conn_inner
            raise duckdb.IOException("Caller query IO failure")

    with patch(
        "duckdb.connect",
        side_effect=duckdb.IOException("Unrecoverable permission denied"),
    ):
        with pytest.raises(duckdb.IOException, match="Unrecoverable"):
            with connect_duckdb_with_retry(db_file, read_only=True, max_attempts=2):
                pass


def test_read_only_duckdb_pool(tmp_path: Path) -> None:
    """Test ReadOnlyDuckDBPool connection reuse, mtime invalidation, capacity overflow, and custom connect_fn."""
    from t1d_analytics.sql_utils import (
        ReadOnlyDuckDBPool,
        close_read_only_duckdb_pool,
        connect_duckdb_read_only_pooled,
    )

    db_file = tmp_path / "pooled.duckdb"
    wconn = duckdb.connect(str(db_file))
    wconn.execute("CREATE TABLE t (x INT)")
    wconn.execute("INSERT INTO t VALUES (10)")
    wconn.close()

    pool = ReadOnlyDuckDBPool(max_pool_size_per_db=1)
    c1 = pool.acquire(db_file, disable_external_access=False)
    c2 = pool.acquire(db_file, disable_external_access=True)
    pool.release(db_file, c1)
    # Pool max is 1, so releasing c2 closes it
    pool.release(db_file, c2)

    # Re-acquiring pops c1 from bucket
    c1_reused = pool.acquire(db_file)
    assert c1_reused is c1
    pool.release(db_file, c1_reused)

    # Simulate mtime signature change on disk -> acquire evicts stale pooled connection
    pool._mtimes[str(db_file.resolve())] = (0, 0)
    with connect_duckdb_read_only_pooled(db_file, pool=pool) as conn_fresh:
        rows = conn_fresh.execute("SELECT x FROM t ORDER BY x").fetchall()
        assert rows == [(10,)]

    # Exception inside context manager discards connection
    with pytest.raises(RuntimeError, match="boom"):
        with connect_duckdb_read_only_pooled(db_file, pool=pool) as _conn_err:
            raise RuntimeError("boom")

    # Custom connect_fn with disable_external_access=False and True
    custom_called = 0

    def _custom_connect(path: str, read_only: bool = True) -> duckdb.DuckDBPyConnection:
        """Open DuckDB connection via custom wrapper."""
        nonlocal custom_called
        custom_called += 1
        return duckdb.connect(path, read_only=read_only)

    with connect_duckdb_read_only_pooled(
        db_file, connect_fn=_custom_connect, disable_external_access=False, pool=pool
    ) as conn_custom:
        assert conn_custom.execute("SELECT COUNT(*) FROM t").fetchone() == (1,)
    with connect_duckdb_read_only_pooled(
        db_file, connect_fn=_custom_connect, disable_external_access=True, pool=pool
    ) as conn_custom2:
        assert conn_custom2.execute("SELECT COUNT(*) FROM t").fetchone() == (1,)
    assert custom_called == 2

    # close_path evicts idle connections for db_file so write connection can open
    c_pooled = pool.acquire(db_file)
    pool.release(db_file, c_pooled)
    pool.close_path(db_file)

    # Release when file has been deleted from disk
    c_last = pool.acquire(db_file)
    db_file.unlink()
    pool.release(db_file, c_last)

    # close_all closes any remaining idle connections in the pool
    db_file_2 = tmp_path / "pooled2.duckdb"
    wconn2 = duckdb.connect(str(db_file_2))
    wconn2.execute("CREATE TABLE t2 (x INT)")
    wconn2.close()
    c_for_close_all = pool.acquire(db_file_2)
    pool.release(db_file_2, c_for_close_all)
    pool.close_all()
    close_read_only_duckdb_pool()


def test_coerce_numeric_aggregates_masks_literals_and_comments() -> None:
    """Test coerce_numeric_aggregates_in_sql preserves string literals and comments untouched."""
    from t1d_analytics.sql_agent import parse_schema_text_to_models
    from t1d_analytics.sql_utils import (
        coerce_numeric_aggregates_in_sql,
        parse_schema_text_to_column_pairs,
        parse_schema_text_to_map,
    )

    q = (
        "SELECT 'SUM(NumSevHypo)' AS label, $$AVG(HbA1c)$$ AS dlabel, "
        "SUM(NumSevHypo) AS total -- comment with AVG(x)\n"
        "FROM tblaptsummary /* CAST(x AS DOUBLE) */"
    )
    rewritten = coerce_numeric_aggregates_in_sql(q)
    assert "'SUM(NumSevHypo)' AS label" in rewritten
    assert "$$AVG(HbA1c)$$ AS dlabel" in rewritten
    assert "-- comment with AVG(x)" in rewritten
    assert "/* CAST(x AS DOUBLE) */" in rewritten
    assert "SUM(TRY_CAST(NumSevHypo AS DOUBLE)) AS total" in rewritten

    schema_text = (
        "-- Table: inline_tbl | Columns: a (INTEGER), b\n"
        "Table: multi_tbl\n"
        "Columns: c (DOUBLE)\n"
        "- d (VARCHAR)\n"
        "Table:\n"
        "Table: |\n"
    )
    pairs = parse_schema_text_to_column_pairs(schema_text)
    assert pairs["inline_tbl"] == [("a", "INTEGER"), ("b", "VARCHAR")]
    assert pairs["multi_tbl"] == [("c", "DOUBLE"), ("d", "VARCHAR")]

    models = parse_schema_text_to_models(schema_text)
    assert [m.table_name for m in models] == ["inline_tbl", "multi_tbl"]
    assert models[0].column_names() == ["a", "b"]

    ddl_text = 'CREATE TABLE "patients" (id INT, PRIMARY KEY (id), flag)'
    ddl_map = parse_schema_text_to_map(ddl_text)
    assert ddl_map == {"patients": ["id", "flag"]}


def test_read_only_duckdb_cursor_and_jaeb_attachment(tmp_path: Path) -> None:
    """Test ReadOnlyDuckDbCursorProxy, connect_duckdb_read_only_cursor, and read-only Jaeb DuckDB attachment."""
    from t1d_analytics.analytics import get_database_schema
    from t1d_analytics.duckdb_pool import (
        ReadOnlyDuckDbCursorProxy,
        ReadOnlyDuckDBPool,
        _noop_interrupt,
        _resolve_jaeb_attach_path,
        attach_jaeb_read_only_if_present,
        connect_duckdb_read_only_cursor,
        create_read_only_cursor_proxy,
    )
    from t1d_analytics.sql_executor import execute_duckdb_query
    from t1d_analytics.sql_utils import extract_table_columns_map, extract_table_schemas

    _noop_interrupt()

    primary_db = tmp_path / "t1d_analytics.duckdb"
    jaeb_dir = tmp_path / "mnt_data_t1d" / "jaeb"
    jaeb_dir.mkdir(parents=True)
    jaeb_db = jaeb_dir / "t1d.duckdb"

    w_primary = duckdb.connect(str(primary_db))
    w_primary.execute("CREATE TABLE main_patients (id INT, label VARCHAR)")
    w_primary.execute("INSERT INTO main_patients VALUES (1, 'primary')")
    w_primary.close()

    w_jaeb = duckdb.connect(str(jaeb_db))
    w_jaeb.execute("CREATE TABLE jaeb_cohort (pt_id INT, hba1c DOUBLE)")
    w_jaeb.execute("INSERT INTO jaeb_cohort VALUES (101, 6.8), (102, 7.4)")
    w_jaeb.close()

    # _resolve_jaeb_attach_path returns resolved jaeb_db when distinct, None when same or missing
    assert (
        _resolve_jaeb_attach_path(primary_db, jaeb_candidates=(jaeb_db,))
        == jaeb_db.resolve()
    )
    assert _resolve_jaeb_attach_path(jaeb_db, jaeb_candidates=(jaeb_db,)) is None
    assert (
        _resolve_jaeb_attach_path(
            primary_db, jaeb_candidates=(tmp_path / "nonexistent.duckdb",)
        )
        is None
    )

    with (
        patch.object(Path, "resolve", side_effect=OSError("resolve failed")),
        patch.object(Path, "is_file", side_effect=OSError("is_file failed")),
    ):
        assert _resolve_jaeb_attach_path(primary_db, jaeb_candidates=(jaeb_db,)) is None

    # Test _attach_jaeb_read_only_if_present attaches jaeb in READ_ONLY mode and sets search_path
    with patch("t1d_analytics.duckdb_pool.JAEB_ATTACH_CANDIDATES", (jaeb_db,)):
        pool = ReadOnlyDuckDBPool(max_pool_size_per_db=2)
        with connect_duckdb_read_only_cursor(primary_db, pool=pool) as cur:
            assert isinstance(cur, ReadOnlyDuckDbCursorProxy)
            assert isinstance(cur.connection, duckdb.DuckDBPyConnection)
            assert isinstance(cur.raw_cursor, duckdb.DuckDBPyConnection)
            assert cur.raw_cursor is not cur.connection

            # Query attached jaeb table via search_path ('main,jaeb') and parameterized query
            res = cur.execute(
                "SELECT pt_id, hba1c FROM jaeb_cohort WHERE pt_id = ?", [101]
            )
            assert res is not None
            assert cur.description == [("pt_id",), ("hba1c",)]
            assert cur.fetchone() == (101, 6.8)

            cur.execute("SELECT pt_id FROM jaeb_cohort ORDER BY pt_id")
            assert cur.fetchmany(1) == [(101,)]
            assert cur.fetchall() == [(102,)]

            # Verify jaeb attachment is strictly READ_ONLY
            with pytest.raises(duckdb.Error):
                cur.execute("INSERT INTO jaeb.jaeb_cohort VALUES (103, 8.0)")

            # Schema helpers accept ReadOnlyDuckDbCursorProxy directly
            cols_from_cur = extract_table_columns_map(cur)
            assert "main_patients" in cols_from_cur
            schemas_from_cur = extract_table_schemas(cur)
            assert "main_patients" in schemas_from_cur
            schema_str = get_database_schema(cur)
            assert "main_patients" in schema_str

            # Interrupt and idempotent close
            cur.interrupt()
            cur.close()
            cur.close()

        # Execute query via execute_duckdb_query against attached Jaeb data
        rows = execute_duckdb_query(
            str(primary_db),
            "SELECT pt_id, hba1c FROM jaeb.jaeb_cohort ORDER BY pt_id",
        )
        assert rows == [
            {"pt_id": 101, "hba1c": 6.8},
            {"pt_id": 102, "hba1c": 7.4},
        ]
        pool.close_all()

    # _attach_jaeb_read_only_if_present swallows duckdb/RuntimeError gracefully on broken connection
    class _FailingAttachConn:
        """Minimal connection stub raising RuntimeError on ATTACH."""

        def execute(self, _sql: str) -> "_FailingAttachConn":
            """Raise RuntimeError."""
            raise RuntimeError("attach unsupported")

    attach_jaeb_read_only_if_present(
        _FailingAttachConn(),
        primary_db,
        jaeb_candidates=(jaeb_db,),
    )

    # create_read_only_cursor_proxy with custom cursor-capable protocol object and fallback object
    from collections.abc import Sequence as AbcSequence

    from t1d_analytics.duckdb_pool import PoolSqlRawInputValue

    class _CustomCursorObj:
        """Custom cursor double without close method implementing DuckDbCursorObjectProtocol."""

        description: AbcSequence[tuple[str, ...]] | None = None

        def execute(
            self,
            _query: str,
            _parameters: AbcSequence[PoolSqlRawInputValue] | None = None,
        ) -> duckdb.DuckDBPyConnection:
            """Open an in-memory DuckDB relation."""
            return duckdb.connect(":memory:")

        def fetchone(self) -> tuple[PoolSqlRawInputValue, ...] | None:
            """Return None."""
            return None

        def fetchall(self) -> list[tuple[PoolSqlRawInputValue, ...]]:
            """Return empty row list."""
            return []

        def fetchmany(self, _size: int = 1) -> list[tuple[PoolSqlRawInputValue, ...]]:
            """Return empty row list."""
            return []

    class _CustomConnWithCursor(_CustomCursorObj):
        """Custom connection double implementing .cursor()."""

        def __init__(self) -> None:
            """Initialize with a dedicated cursor instance."""
            self.cur = _CustomCursorObj()
            self.interrupt = "not-callable"

        def cursor(self) -> _CustomCursorObj:
            """Return dedicated cursor double."""
            return self.cur

    custom_conn = _CustomConnWithCursor()
    proxy_custom = create_read_only_cursor_proxy(custom_conn)
    assert id(proxy_custom.raw_cursor) == id(custom_conn.cur)
    assert proxy_custom.description is None
    proxy_custom.interrupt()
    proxy_custom.close()

    class _SelfCursorConn(_CustomCursorObj):
        """Connection double whose .cursor() returns self."""

        def cursor(self) -> "_SelfCursorConn":
            """Return self."""
            return self

    self_conn = _SelfCursorConn()
    proxy_self = create_read_only_cursor_proxy(self_conn)
    assert id(proxy_self.raw_cursor) == id(self_conn)
    proxy_self.interrupt()
    proxy_self.close()


def test_csv_formula_injection_backend_frontend_parity() -> None:
    """Verify `sql_utils.sanitize_spreadsheet_cell` and `exporters.ts:sanitizeSpreadsheetCell` produce identical OWASP outputs."""
    import json
    import subprocess
    from pathlib import Path

    from t1d_analytics.sql_utils import sanitize_spreadsheet_cell

    vectors: list[str] = [
        "",
        "   ",
        "normal clinical note",
        "=SUM(A1:A10)",
        "+cmd|' /C calc'!A0",
        "-cmd|' /C calc'!A0",
        "@SUM(1+1)*cmd|' /C calc'!A0",
        '   =HYPERLINK("http://evil.example")',
        "\u00a0@SUM(A1:A2)",
        "  +IMPORTXML(A1,A2)",
        "  -2+3",
        "\t=1+1",
        "\r=1+1",
        "\n=1+1",
        "\x00=1+1",
        "\x0b+1+1",
        "\x0c@1+1",
        "   \t=1+1",
        "   \x1f-cmd",
        "-12.5",
        "+3.25",
        "  -42.0",
        "  +0.125e2",
        "+1e999",
        "-1e999",
    ]

    py_outputs = [sanitize_spreadsheet_cell(v) for v in vectors]

    web_dir = Path(__file__).resolve().parents[2] / "t1d-analytics-web"
    exporters_ts = web_dir / "src" / "exporters.ts"
    node_script = """
import fs from "node:fs";
import vm from "node:vm";
import ts from "./node_modules/typescript/lib/typescript.js";

const vectors = JSON.parse(process.argv[1]);
const src = fs.readFileSync("./src/exporters.ts", "utf8");
const transpiled = ts.transpileModule(src, {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 },
}).outputText;
const mod = { exports: {} };
vm.runInNewContext(transpiled, { module: mod, exports: mod.exports, Blob, URL, document: {} });
const out = vectors.map((v) => mod.exports.sanitizeSpreadsheetCell(v));
process.stdout.write(JSON.stringify(out));
"""
    assert exporters_ts.is_file()
    proc = subprocess.run(
        ["node", "--input-type=module", "-e", node_script, json.dumps(vectors)],
        cwd=str(web_dir),
        check=True,
        capture_output=True,
        text=True,
    )
    ts_outputs = json.loads(proc.stdout)
    assert py_outputs == ts_outputs
    assert py_outputs[3] == "'=SUM(A1:A10)"
    assert py_outputs[19] == "-12.5"
    assert py_outputs[23] == "'+1e999"
