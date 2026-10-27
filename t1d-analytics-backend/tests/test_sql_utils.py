"""Unit tests for shared SQL utilities in t1d_analytics.sql_utils."""

import datetime
import decimal
import uuid
from pathlib import Path
from unittest.mock import MagicMock

import duckdb
import pytest

from t1d_analytics.sql_utils import (
    coerce_numeric_aggregates_in_sql,
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
        "SELECT SUM(NumSevHypo), AVG(DISTINCT t.AgeAsOfRandDt) FROM tblaptsummary t"
    )
    assert "SUM(TRY_CAST(NumSevHypo AS DOUBLE))" in coerced
    assert "AVG(DISTINCT TRY_CAST(t.AgeAsOfRandDt AS DOUBLE))" in coerced
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
    """Test all branches of is_sql_query including comments, parens, show NL, and mocks."""
    assert is_sql_query("") is False
    assert is_sql_query("()") is False
    assert is_sql_query("What is the average glucose?") is False
    assert is_sql_query("Show me the patients") is False
    assert is_sql_query("SHOW TABLES") is True
    assert is_sql_query("SELECT 1") is True
    assert is_sql_query("SELECT FROM WHERE INVALID") is False

    # Mock connection without extract_statements
    mock_conn = MagicMock()
    del mock_conn.extract_statements
    assert is_sql_query("CREATE TABLE x (id INT)", conn=mock_conn) is True
    assert is_sql_query("SELECT 1", conn=mock_conn) is True

    mock_conn.execute.side_effect = duckdb.ParserException("syntax error")
    assert is_sql_query("SELECT 1", conn=mock_conn) is False

    mock_conn.execute.side_effect = RuntimeError("other db error")
    assert is_sql_query("SELECT 1", conn=mock_conn) is True


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

    assert (
        validate_read_only_sql("  SELECT * FROM patients WHERE name = 'a;b'; ")
        == "SELECT * FROM patients WHERE name = 'a;b';"
    )

    with pytest.raises(EmptyQueryError):
        validate_read_only_sql("   -- only a comment\n ")

    with pytest.raises(MultipleStatementsError):
        validate_read_only_sql("SELECT 1; DROP TABLE patients;")

    with pytest.raises(ReadOnlyViolationError):
        validate_read_only_sql("DROP TABLE patients")


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

    # Test lock retry and exhaustion
    delays: list[float] = []
    with patch(
        "duckdb.connect",
        side_effect=duckdb.IOException("Could not set lock on file"),
    ):
        with pytest.raises(duckdb.IOException, match="database is locked"):
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
