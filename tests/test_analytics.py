"""Tests for the analytics module."""

import io
import os
import sys
import types
import typing
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import duckdb
import pytest
from t1d_analytics.analytics import (
    extract_zips,
    get_database_schema,
    handle_natural_language,
    load_data_to_duckdb,
    run_query_repl,
)

if "any_llm" not in sys.modules:
    sys.modules["any_llm"] = types.ModuleType("any_llm")
    setattr(sys.modules["any_llm"], "AnyLLM", MagicMock())


def test_extract_zips(tmp_path: Path) -> None:
    """Test extracting zip files."""
    zip_path = tmp_path / "test_data.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("test.csv", "id,name\n1,test")
    extract_zips(str(tmp_path))
    extract_dir = tmp_path / "test_data"
    assert extract_dir.exists()
    assert (extract_dir / "test.csv").exists()


def test_extract_zips_bad_zip(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test handling of bad zip files."""
    bad_zip_path = tmp_path / "bad.zip"
    bad_zip_path.write_text("Not a real zip file")
    extract_zips(str(tmp_path))
    assert "Bad Zip File" in capsys.readouterr().out


def test_extract_zips_invalid_dir(capsys: pytest.CaptureFixture[str]) -> None:
    """Test extracting with invalid directory."""
    extract_zips("/path/does/not/exist/12345")
    assert "does not exist" in capsys.readouterr().out


def test_extract_zips_no_zips(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test extracting when no zips are present."""
    extract_zips(str(tmp_path))
    assert "No zip files found to extract" in capsys.readouterr().out


def test_load_data_to_duckdb(tmp_path: Path) -> None:
    """Test loading CSV data into DuckDB."""
    (tmp_path / "data1.csv").write_text("id,val\n1,10\n2,20")
    (tmp_path / "123_data.csv").write_text("id,val\n1,10")
    db_path = str(tmp_path / "test.duckdb")
    load_data_to_duckdb(str(tmp_path), db_path)

    conn = duckdb.connect(db_path)
    tables = [r[0] for r in conn.execute("SHOW TABLES").fetchall()]
    assert "data1" in tables
    assert "t_123_data" in tables
    res = conn.execute("SELECT sum(val) FROM data1").fetchone()
    assert res is not None
    assert res[0] == 30
    conn.close()


def test_load_data_to_duckdb_no_csv(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test loading when no CSVs are present."""
    db_path = str(tmp_path / "test.duckdb")
    load_data_to_duckdb(str(tmp_path), db_path)
    assert "No tabular files found." in capsys.readouterr().out


def test_load_data_to_duckdb_tab_delimited(tmp_path: Path) -> None:
    """Test loading tab delimited data."""
    (tmp_path / "data_tab.txt").write_text("id\tval\n1\t10\n2\t20")
    db_path = str(tmp_path / "test.duckdb")
    load_data_to_duckdb(str(tmp_path), db_path)
    conn = duckdb.connect(db_path)
    res = conn.execute("SELECT sum(val) FROM data_tab").fetchone()
    assert res is not None
    assert res[0] == 30
    conn.close()


def test_load_data_to_duckdb_utf16_pipe(tmp_path: Path) -> None:
    """Test loading utf-16 pipe delimited data."""
    file_path = tmp_path / "data_utf16.txt"
    with open(file_path, "wb") as f:
        f.write(b"\xff\xfe")  # BOM
        f.write("id|val\n1|10\n2|20".encode("utf-16le"))

    db_path = str(tmp_path / "test.duckdb")
    load_data_to_duckdb(str(tmp_path), db_path)
    conn = duckdb.connect(db_path)
    res = conn.execute("SELECT sum(val) FROM data_utf16").fetchone()
    assert res is not None
    assert res[0] == 30
    conn.close()


def test_load_data_to_duckdb_invalid_dir(capsys: pytest.CaptureFixture[str]) -> None:
    """Test loading with invalid directory."""
    load_data_to_duckdb("/path/does/not/exist/12345", "test.db")
    assert "does not exist" in capsys.readouterr().out


def test_load_data_to_duckdb_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test loading CSV data with an error."""
    (tmp_path / "data.csv").mkdir()
    db_path = str(tmp_path / "test.duckdb")
    load_data_to_duckdb(str(tmp_path), db_path)
    assert "Failed to load data.csv" in capsys.readouterr().out


def test_get_database_schema() -> None:
    """Test extracting database schema."""
    conn = duckdb.connect(":memory:")
    conn.execute("CREATE TABLE test_tab (id INTEGER, name VARCHAR)")
    schema = get_database_schema(conn)
    assert "Table: test_tab" in schema
    assert "id (INTEGER)" in schema
    assert "name (VARCHAR)" in schema
    conn.close()


@patch("t1d_analytics.analytics.get_database_schema", return_value="mock schema")
@patch("any_llm.AnyLLM.create")
def test_handle_natural_language_success(
    mock_create: MagicMock,
    mock_get_schema: MagicMock,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Test successful natural language query handling."""
    conn = duckdb.connect(":memory:")
    conn.execute("CREATE TABLE users (name VARCHAR)")
    conn.execute("INSERT INTO users VALUES ('Alice')")

    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "```sql\nSELECT name FROM users;\n```"
    mock_llm.completion.return_value = mock_response
    mock_create.return_value = mock_llm

    handle_natural_language(conn, "get all users")
    output = capsys.readouterr().out
    assert "SELECT name FROM users;" in output
    assert "Alice" in output
    conn.close()


@patch("t1d_analytics.analytics.get_database_schema", return_value="mock schema")
@patch("any_llm.AnyLLM.create")
def test_handle_natural_language_error(
    mock_create: MagicMock,
    mock_get_schema: MagicMock,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Test natural language error handling."""
    conn = duckdb.connect(":memory:")
    mock_llm = MagicMock()
    mock_llm.completion.side_effect = Exception("API error")
    mock_create.return_value = mock_llm

    handle_natural_language(conn, "get all users")
    assert "Failed to generate or execute query: API error" in capsys.readouterr().out
    conn.close()


def test_handle_natural_language_no_module(capsys: pytest.CaptureFixture[str]) -> None:
    """Test natural language when any-llm is missing."""
    import sys

    # Temporarily remove any_llm from modules to test ImportError
    original_module = sys.modules.get("any_llm")
    sys.modules["any_llm"] = None  # type: ignore

    conn = duckdb.connect(":memory:")
    handle_natural_language(conn, "test")
    assert "any-llm-sdk[ollama] is not installed" in capsys.readouterr().out
    conn.close()

    if original_module:
        sys.modules["any_llm"] = original_module


def test_handle_natural_language_provider_readiness_error(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test natural language when requested provider is missing API credentials."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    conn = duckdb.connect(":memory:")
    handle_natural_language(conn, "test query", model="openai/gpt-4o")
    output = capsys.readouterr().out
    assert "Failed to generate or execute query:" in output
    assert "OPENAI_API_KEY" in output
    conn.close()


@patch("t1d_analytics.analytics.get_database_schema", return_value="mock schema")
@patch("any_llm.AnyLLM.create")
def test_handle_natural_language_multi_provider_success(
    mock_create: MagicMock,
    mock_get_schema: MagicMock,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test natural language handling using an external provider with API key."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-123")
    conn = duckdb.connect(":memory:")
    conn.execute("CREATE TABLE users (name VARCHAR)")
    conn.execute("INSERT INTO users VALUES ('Bob')")

    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "```sql\nSELECT name FROM users;\n```"
    mock_llm.completion.return_value = mock_response
    mock_create.return_value = mock_llm

    handle_natural_language(conn, "get users", model="gpt-4o", provider="openai")
    output = capsys.readouterr().out
    assert "SELECT name FROM users;" in output
    assert "Bob" in output
    mock_create.assert_called_once_with("openai")
    mock_llm.completion.assert_called_once()
    assert mock_llm.completion.call_args[1]["model"] == "gpt-4o"
    conn.close()


@patch("t1d_analytics.analytics.input", side_effect=["", "SELECT 1 AS val;", "exit"])
def test_run_query_repl_sql(
    mock_input: MagicMock, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test REPL with a standard SQL query and empty input."""
    db_path = str(tmp_path / "test.duckdb")
    # Create a valid duckdb file instead of touching it
    conn = duckdb.connect(db_path)
    conn.close()

    run_query_repl(db_path)
    output = capsys.readouterr().out
    assert "val" in output
    assert "1" in output
    assert "Exiting." in output


@patch(
    "t1d_analytics.analytics.input", side_effect=["SELECT * FROM not_exist;", "exit"]
)
def test_run_query_repl_sql_error(
    mock_input: MagicMock, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test REPL with an invalid SQL query."""
    db_path = str(tmp_path / "test.duckdb")
    conn = duckdb.connect(db_path)
    conn.close()

    run_query_repl(db_path)
    output = capsys.readouterr().out
    assert "SQL Error" in output


@patch("t1d_analytics.analytics.handle_natural_language")
@patch("t1d_analytics.analytics.input", side_effect=["what is the count?", "quit"])
def test_run_query_repl_nl(
    mock_input: MagicMock,
    mock_handle_nl: MagicMock,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Test REPL routing natural language queries."""
    db_path = str(tmp_path / "test.duckdb")
    conn = duckdb.connect(db_path)
    conn.close()

    run_query_repl(db_path)
    mock_handle_nl.assert_called_once()
    assert "Exiting." in capsys.readouterr().out


@patch("t1d_analytics.analytics.input", side_effect=KeyboardInterrupt)
def test_run_query_repl_interrupt(
    mock_input: MagicMock, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test REPL handling of KeyboardInterrupt."""
    db_path = str(tmp_path / "test.duckdb")
    conn = duckdb.connect(db_path)
    conn.close()

    run_query_repl(db_path)
    assert "Exiting." in capsys.readouterr().out


@patch("t1d_analytics.analytics.input", side_effect=EOFError)
def test_run_query_repl_eof(
    mock_input: MagicMock, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test REPL handling of EOF."""
    db_path = str(tmp_path / "test.duckdb")
    conn = duckdb.connect(db_path)
    conn.close()

    run_query_repl(db_path)
    assert "Exiting." in capsys.readouterr().out


def test_run_query_repl_invalid_db(capsys: pytest.CaptureFixture[str]) -> None:
    """Test REPL with invalid DB path."""
    run_query_repl("/path/does/not/exist/12345.db")
    assert "does not exist" in capsys.readouterr().out


@patch("builtins.input", side_effect=["quit"])
def test_run_query_repl_missing_initial_data(
    mock_input: MagicMock, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test REPL warning when DB has tables but lacks T1D trial datasets."""
    db_path = str(tmp_path / "custom.duckdb")
    conn = duckdb.connect(db_path)
    conn.execute("CREATE TABLE custom (id INT)")
    conn.close()

    run_query_repl(db_path)
    out = capsys.readouterr().out
    assert "lacks standard initial clinical trial datasets" in out


@patch("t1d_analytics.analytics.get_database_schema", return_value="mock schema")
@patch("any_llm.AnyLLM.create")
def test_handle_natural_language_edge_cases(
    mock_create: MagicMock,
    mock_get_schema: MagicMock,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Test natural language edge cases for SQL generation."""
    conn = duckdb.connect(":memory:")
    conn.execute("CREATE TABLE users (name VARCHAR)")
    conn.execute("INSERT INTO users VALUES ('Alice')")

    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_llm.completion.return_value = mock_response
    mock_create.return_value = mock_llm

    # Case 1: startswith("```") but only 1 line
    mock_response.choices[0].message.content = "```SELECT name FROM users;```"
    handle_natural_language(conn, "get all users")
    output = capsys.readouterr().out
    assert "SELECT name FROM users;" in output

    # Case 2: lower().startswith("sql\\n")
    mock_response.choices[0].message.content = "sql\nSELECT name FROM users;"
    handle_natural_language(conn, "get all users")
    output = capsys.readouterr().out
    assert "SELECT name FROM users;" in output

    # Case 3: lower().startswith("duckdb\\n")
    mock_response.choices[0].message.content = "duckdb\nSELECT name FROM users;"
    handle_natural_language(conn, "get all users")
    output = capsys.readouterr().out
    assert "SELECT name FROM users;" in output

    conn.close()


def test_get_database_schema_fetch_error() -> None:
    """Test extracting database schema with a fetch error."""
    mock_conn = MagicMock()

    # Setup mock returns
    # 1st call: SHOW TABLES -> fetchall returns [('test_tab',)]
    # 2nd call: DESCRIBE test_tab -> fetchall returns [('id', 'INTEGER'), ('name', 'VARCHAR')]
    # 3rd call: SELECT * FROM test_tab LIMIT 1 -> raises Exception

    show_tables_mock = MagicMock()
    show_tables_mock.fetchall.return_value = [("test_tab",)]

    describe_mock = MagicMock()
    describe_mock.fetchall.return_value = [("id", "INTEGER"), ("name", "VARCHAR")]

    select_mock = MagicMock()
    select_mock.fetchone.side_effect = Exception("Mock fetch error")

    def mock_execute(query: str) -> typing.Any:
        """Mock execute function."""
        if query.startswith("SHOW TABLES"):
            return show_tables_mock
        elif query.startswith("DESCRIBE"):
            return describe_mock
        elif query.startswith("SELECT * FROM"):
            return select_mock
        return MagicMock()

    mock_conn.execute.side_effect = mock_execute

    schema = get_database_schema(mock_conn)

    assert "Table: test_tab" in schema
    assert "(could not fetch)" in schema


def test_load_data_to_duckdb_duplicate_mapped_columns(tmp_path: Path) -> None:
    """Test loading data with columns that map to the same standard name."""
    # Pt_NumHospDKA and NumHospDKA both map to numhospdka in likely_matches.json
    (tmp_path / "data_dupes.csv").write_text(
        "Pt_NumHospDKA,NumHospDKA,NumHospDKA\n1,2,3"
    )
    db_path = str(tmp_path / "test.duckdb")
    load_data_to_duckdb(str(tmp_path), db_path)

    conn = duckdb.connect(db_path)
    cols = [r[0] for r in conn.execute("DESCRIBE data_dupes").fetchall()]
    assert "numhospdka" in cols
    assert "NumHospDKA_2" in cols
    conn.close()


def test_load_data_to_duckdb_existing_table(tmp_path: Path) -> None:
    """Test loading data when the table already exists."""
    (tmp_path / "data_exist.csv").write_text("id,val\n1,10\n2,20")
    db_path = str(tmp_path / "test.duckdb")

    # Create the table beforehand
    conn = duckdb.connect(db_path)
    conn.execute("CREATE TABLE data_exist (dummy INT)")
    conn.close()

    # Load data
    load_data_to_duckdb(str(tmp_path), db_path)

    # Verify table structure wasn't overwritten
    conn = duckdb.connect(db_path)
    cols = [r[0] for r in conn.execute("DESCRIBE data_exist").fetchall()]
    assert "dummy" in cols
    assert "id" not in cols
    conn.close()


def test_extract_zips_already_extracted(tmp_path: Path) -> None:
    """Test extract_zips when the directory already exists."""
    import zipfile

    from t1d_analytics.analytics import extract_zips

    zip_path = tmp_path / "data.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("test.txt", "content")

    extract_dir = tmp_path / "data"
    extract_dir.mkdir()

    extract_zips(str(tmp_path))
    # It should skip extraction, so "test.txt" won't be inside extract_dir
    assert not (extract_dir / "test.txt").exists()


def test_load_data_no_matches_file(tmp_path: Path, mocker: typing.Any) -> None:
    """Test load_data_to_duckdb when matches file does not exist."""
    import duckdb
    from t1d_analytics.analytics import load_data_to_duckdb

    mocker.patch(
        "pathlib.Path.exists",
        autospec=True,
        side_effect=lambda self: False if "likely_matches" in str(self) else True,
    )

    (tmp_path / "data.csv").write_text("A,B\n1,2")
    db_path = str(tmp_path / "test.duckdb")
    load_data_to_duckdb(str(tmp_path), db_path)
    conn = duckdb.connect(db_path)
    assert "data" in [r[0] for r in conn.execute("SHOW TABLES").fetchall()]


def test_handle_natural_language_no_result(mocker: typing.Any) -> None:
    """Test handle_natural_language when query returns no result."""
    from t1d_analytics.analytics import handle_natural_language

    # Mock any-llm
    mock_llm = mocker.MagicMock()
    mock_llm.return_value.generate.return_value = "```sql\nSELECT 1;\n```"
    mocker.patch.dict("sys.modules", {"any_llm": mocker.MagicMock(AnyLLM=mock_llm)})

    conn = mocker.MagicMock()
    conn.sql.return_value = None  # No result

    handle_natural_language(conn, "test")
    conn.sql.assert_called_once()


def test_run_query_repl_no_result(mocker: typing.Any, tmp_path: Path) -> None:
    """Test run_query_repl when sql returns no result and exercises natural language branch."""
    from t1d_analytics.analytics import run_query_repl

    db_path = tmp_path / "test.duckdb"
    init_conn = duckdb.connect(str(db_path))
    init_conn.execute("CREATE TABLE t (id INT)")
    init_conn.close()

    inputs = ["show me patients", "SELECT 1 WHERE 1=0", "exit"]
    mocker.patch("builtins.input", side_effect=inputs)
    mock_nl = mocker.patch("t1d_analytics.analytics.handle_natural_language")

    run_query_repl(str(db_path))
    mock_nl.assert_called_once()


def test_run_query_repl_with_result(mocker: typing.Any, tmp_path: Path) -> None:
    """Test run_query_repl displays results when query returns non-empty result."""
    from t1d_analytics.analytics import run_query_repl

    db_path = tmp_path / "test.duckdb"
    db_path.touch()

    inputs = ["SELECT 1", "exit"]
    mocker.patch("builtins.input", side_effect=inputs)

    mock_conn = mocker.MagicMock()
    mock_result = mocker.MagicMock()
    mock_conn.sql.return_value = mock_result
    mocker.patch("duckdb.connect", return_value=mock_conn)

    run_query_repl(str(db_path))
    mock_conn.sql.assert_called_once()
    mock_result.show.assert_called_once()


def test_extract_zips_path_traversal_skipped(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test that zip slip path traversal entries are skipped safely."""
    zip_path = tmp_path / "malicious.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("../../evil.txt", "evil content")
        zf.writestr("safe.txt", "safe content")
    extract_zips(str(tmp_path))
    assert not (tmp_path.parent / "evil.txt").exists()
    extract_dir = tmp_path / "malicious"
    assert (extract_dir / "safe.txt").exists()
    assert "path traversal detected" in capsys.readouterr().out


def test_is_sql_query() -> None:
    """Test is_sql_query classification logic."""
    from t1d_analytics.analytics import is_sql_query

    assert is_sql_query("SELECT * FROM users")
    assert is_sql_query("WITH cte AS (SELECT 1) SELECT * FROM cte")
    assert is_sql_query("DESCRIBE users")
    assert is_sql_query("PRAGMA version")
    assert is_sql_query("EXPLAIN SELECT 1")
    assert is_sql_query("SHOW TABLES")
    assert is_sql_query("SHOW columns")
    assert is_sql_query("-- comment\nSELECT * FROM users")
    assert is_sql_query("/* block comment */ SELECT * FROM users")
    assert not is_sql_query("Show me the first 5 patients")
    assert not is_sql_query("Show the average HbA1c")
    assert not is_sql_query("Show all patients with low blood glucose")
    assert not is_sql_query("What is the average age?")
    assert not is_sql_query("")
    assert not is_sql_query("   ")
    assert not is_sql_query("-- only comments\n")


def test_load_data_to_duckdb_bom_and_delimiters(tmp_path: Path) -> None:
    """Test loading data files with UTF-8 BOM, tabs, and pipes."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    db_path = str(tmp_path / "test.duckdb")

    # UTF-8 BOM
    bom_file = data_dir / "bom_data.csv"
    with open(bom_file, "wb") as f:
        f.write(b"\xef\xbb\xbfid,name\n1,Alpha\n")

    # Tab delimited
    tsv_file = data_dir / "tab_data.csv"
    tsv_file.write_text("id\tval\n1\t100\n2\t200\n", encoding="utf-8")

    # Pipe delimited with commas inside cells
    pipe_file = data_dir / "pipe_data.csv"
    pipe_file.write_text(
        'id|description\n1|"apple, banana, cherry"\n', encoding="utf-8"
    )

    load_data_to_duckdb(str(data_dir), db_path)

    conn = duckdb.connect(db_path, read_only=True)
    tables = [r[0] for r in conn.execute("SHOW TABLES").fetchall()]
    assert "bom_data" in tables
    assert "tab_data" in tables
    assert "pipe_data" in tables
    pipe_rows = conn.execute("SELECT * FROM pipe_data").fetchall()
    assert len(pipe_rows) == 1
    assert "apple, banana, cherry" in pipe_rows[0][1]
    conn.close()


def test_run_query_repl_custom_model_routing(
    mocker: typing.Any, tmp_path: Path
) -> None:
    """Test run_query_repl routes 'Show me...' to handle_natural_language with custom model."""
    from t1d_analytics.analytics import run_query_repl

    db_path = tmp_path / "test.duckdb"
    conn = duckdb.connect(str(db_path))
    conn.execute("CREATE TABLE patients (id INT)")
    conn.close()

    inputs = ["Show me the first 5 patients", "exit"]
    mocker.patch("builtins.input", side_effect=inputs)
    mock_hnl = mocker.patch("t1d_analytics.analytics.handle_natural_language")

    run_query_repl(str(db_path), model="custom-gemma", provider="openai")
    mock_hnl.assert_called_once()
    assert mock_hnl.call_args[1]["model"] == "custom-gemma"
    assert mock_hnl.call_args[1]["provider"] == "openai"


def test_load_data_to_duckdb_fallback_delimiters(
    tmp_path: Path, mocker: typing.Any
) -> None:
    """Test load_data_to_duckdb delimiter fallback logic when Sniffer raises an exception."""
    data_dir = tmp_path / "data_fb"
    data_dir.mkdir()
    db_path = str(tmp_path / "test_fb.duckdb")

    # Pipe fallback
    pipe_file = data_dir / "pipe_fb.csv"
    pipe_file.write_text("colA|colB\n1|2\n")

    # Tab fallback
    tab_file = data_dir / "tab_fb.csv"
    tab_file.write_text("colA\tcolB\n1\t2\n")

    # Mock csv.Sniffer to raise Exception
    mock_sniffer = mocker.patch("csv.Sniffer")
    mock_sniffer.return_value.sniff.side_effect = Exception("Sniff failed")

    load_data_to_duckdb(str(data_dir), db_path)

    conn = duckdb.connect(db_path, read_only=True)
    tables = [r[0] for r in conn.execute("SHOW TABLES").fetchall()]
    assert "pipe_fb" in tables
    assert "tab_fb" in tables
    conn.close()


def test_load_data_to_duckdb_empty_file(tmp_path: Path) -> None:
    """Test load_data_to_duckdb with an empty CSV file."""
    data_dir = tmp_path / "data_empty"
    data_dir.mkdir()
    db_path = str(tmp_path / "test_empty.duckdb")

    empty_file = data_dir / "empty.csv"
    empty_file.write_text("")

    load_data_to_duckdb(str(data_dir), db_path)


@patch("t1d_analytics.analytics.get_database_schema", return_value="mock schema")
@patch("any_llm.AnyLLM.create")
def test_handle_natural_language_ollama_host_prefix(
    mock_create: MagicMock,
    mock_get_schema: MagicMock,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Test handle_natural_language prefixes http:// to OLLAMA_HOST if missing."""
    conn = duckdb.connect(":memory:")
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "SELECT 1;"
    mock_llm.completion.return_value = mock_response
    mock_create.return_value = mock_llm

    with patch.dict(os.environ, {"OLLAMA_HOST": "custom-host:11434"}):
        handle_natural_language(conn, "test query")

    output = capsys.readouterr().out
    assert "SELECT 1;" in output
    conn.close()


def test_load_data_to_duckdb_subdirectories_collision(tmp_path: Path) -> None:
    """Test load_data_to_duckdb disambiguates duplicate stems across subdirectories."""
    data_dir = tmp_path / "multi_study"
    study_a = data_dir / "study_a"
    study_b = data_dir / "study_b"
    study_a.mkdir(parents=True)
    study_b.mkdir(parents=True)
    db_path = str(tmp_path / "multi.duckdb")

    (study_a / "demographics.csv").write_text("id,age\n1,25\n")
    (study_b / "demographics.csv").write_text("id,age\n2,30\n")

    # With automatic collision detection
    load_data_to_duckdb(str(data_dir), db_path)

    conn = duckdb.connect(db_path, read_only=True)
    tables = [r[0] for r in conn.execute("SHOW TABLES").fetchall()]
    assert any("study_a_demographics" in t for t in tables)
    assert any("study_b_demographics" in t for t in tables)
    conn.close()


def test_load_data_to_duckdb_numeric_stem_and_root_collision(tmp_path: Path) -> None:
    """Test load_data_to_duckdb with numeric file stems and repeated table names."""
    data_dir = tmp_path / "numeric_study"
    data_dir.mkdir()
    db_path = str(tmp_path / "numeric.duckdb")

    # Numeric start
    (data_dir / "123_results.csv").write_text("id,val\n1,10\n")
    # Duplicate stem in same root folder
    (data_dir / "results.csv").write_text("id,val\n2,20\n")
    (data_dir / "results.txt").write_text("id,val\n3,30\n")

    load_data_to_duckdb(str(data_dir), db_path, prefix_subdirs=True)

    conn = duckdb.connect(db_path, read_only=True)
    tables = [r[0] for r in conn.execute("SHOW TABLES").fetchall()]
    assert "t_123_results" in tables
    assert "results" in tables or "results_2" in tables
    conn.close()


def test_is_sql_query_advanced_heuristics() -> None:
    """Test SQL vs Natural language heuristics including DuckDB shorthand and boundary keywords."""
    from t1d_analytics.analytics import is_sql_query

    # Natural language beginning with SQL keywords
    assert not is_sql_query("Select patients with severe hypo events")
    assert not is_sql_query(
        "Describe the distribution of CGM glucose in pediatric patients"
    )
    assert not is_sql_query("Explain why patient 10 had glycemic variability")

    # DuckDB shorthand statements
    assert not is_sql_query("()")
    assert is_sql_query("FROM patients")
    assert is_sql_query("(SELECT 1)")
    assert is_sql_query("SUMMARIZE patients")
    assert is_sql_query("CALL current_setting('threads')")
    assert is_sql_query("VALUES (1, 'a'), (2, 'b')")
    assert is_sql_query("TABLE patients")


def test_extract_zips_sibling_prefix_path_traversal(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test extract_zips rejects zip entries targeting sibling directories with matching prefixes."""
    data_dir = tmp_path / "zip_test"
    data_dir.mkdir()
    zip_path = data_dir / "sibling_attack.zip"

    import io

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zf:
        # Traversal to sibling folder data_dir_evil
        zf.writestr("../zip_test_evil/malicious.txt", "evil")

    zip_path.write_bytes(zip_buffer.getvalue())

    extract_zips(str(data_dir))
    out = capsys.readouterr().out
    assert "path traversal detected" in out.lower()


def test_load_data_to_duckdb_parquet_and_gz(tmp_path: Path) -> None:
    """Test load_data_to_duckdb ingests Parquet and gzip compressed CSV files."""
    import gzip

    data_dir = tmp_path / "multi_format"
    data_dir.mkdir()
    db_path = str(tmp_path / "mf.duckdb")

    # Write parquet file
    conn_tmp = duckdb.connect(":memory:")
    conn_tmp.execute("CREATE TABLE temp_src (id INT, metric VARCHAR)")
    conn_tmp.execute("INSERT INTO temp_src VALUES (101, 'A1c')")
    pq_path = str(data_dir / "lab_metrics.parquet")
    conn_tmp.execute(f"COPY temp_src TO '{pq_path}' (FORMAT PARQUET)")
    conn_tmp.close()

    # Write .csv.gz file
    gz_path = data_dir / "cgm_readings.csv.gz"
    with gzip.open(gz_path, "wt", encoding="utf-8") as gf:
        gf.write("patient_id,glucose\n201,135\n")

    load_data_to_duckdb(str(data_dir), db_path)

    conn = duckdb.connect(db_path, read_only=True)
    tables = [r[0] for r in conn.execute("SHOW TABLES").fetchall()]
    assert "lab_metrics" in tables
    assert "cgm_readings" in tables
    pq_rows = conn.execute("SELECT * FROM lab_metrics").fetchall()
    assert pq_rows == [(101, "A1c")]
    gz_rows = conn.execute("SELECT * FROM cgm_readings").fetchall()
    assert gz_rows == [(201, 135)]
    conn.close()


def test_load_data_to_duckdb_semicolon_fallback(
    tmp_path: Path, monkeypatch: typing.Any
) -> None:
    """Test loading semicolon-delimited and comma fallback CSV when csv.Sniffer fails."""
    import csv

    data_dir = tmp_path / "semi_data"
    data_dir.mkdir()
    (data_dir / "readings.csv").write_text("patient_id;val\n1;99\n2;105\n")
    (data_dir / "plain.csv").write_text("patient_id,val\n1,10\n2,20\n")
    db_path = str(tmp_path / "semi.duckdb")

    # Force csv.Sniffer to raise an exception to trigger the fallback
    def mock_sniff(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
        raise csv.Error("Sniffer failed")

    monkeypatch.setattr(csv.Sniffer, "sniff", mock_sniff)

    load_data_to_duckdb(str(data_dir), db_path)

    conn = duckdb.connect(db_path, read_only=True)
    res = conn.execute("SELECT sum(val) FROM readings").fetchone()
    assert res is not None
    assert res[0] == 204
    res_plain = conn.execute("SELECT sum(val) FROM plain").fetchone()
    assert res_plain is not None
    assert res_plain[0] == 30
    conn.close()


def test_load_data_to_duckdb_uppercase_extensions(tmp_path: Path) -> None:
    """Test loading files with uppercase extensions like .CSV and .TXT on POSIX."""
    data_dir = tmp_path / "upper_data"
    data_dir.mkdir()
    (data_dir / "UPPER.CSV").write_text("id,val\n1,50\n")
    (data_dir / "TEXT.TXT").write_text("id,val\n2,60\n")
    db_path = str(tmp_path / "upper.duckdb")

    load_data_to_duckdb(str(data_dir), db_path)

    conn = duckdb.connect(db_path, read_only=True)
    tables = [r[0] for r in conn.execute("SHOW TABLES").fetchall()]
    assert "upper" in tables
    assert "text" in tables
    conn.close()


def test_load_data_to_duckdb_repeated_and_disambiguation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test load_data_to_duckdb skips already loaded file and disambiguates collisions from different files."""
    data_dir_1 = tmp_path / "dir1"
    data_dir_1.mkdir()
    (data_dir_1 / "trial.csv").write_text("id,score\n1,10\n")

    db_path = str(tmp_path / "idempotent.duckdb")

    # First load
    load_data_to_duckdb(str(data_dir_1), db_path)
    conn = duckdb.connect(db_path)
    assert "trial" in [r[0] for r in conn.execute("SHOW TABLES").fetchall()]
    conn.close()

    # Second load on exact same directory: should skip
    load_data_to_duckdb(str(data_dir_1), db_path)
    out = capsys.readouterr().out
    assert "already exists, skipping" in out

    # Third load with a different file mapping to same table stem: should disambiguate as trial_2
    data_dir_2 = tmp_path / "dir2"
    data_dir_2.mkdir()
    (data_dir_2 / "trial.csv").write_text("id,score\n2,20\n")

    load_data_to_duckdb(str(data_dir_2), db_path)
    conn = duckdb.connect(db_path)
    tables = [r[0] for r in conn.execute("SHOW TABLES").fetchall()]
    assert "trial" in tables
    assert "trial_2" in tables
    conn.close()


def test_is_sql_query_ddl_and_dml() -> None:
    """Test is_sql_query recognizes standard DDL and DML statements."""
    from t1d_analytics.analytics import is_sql_query

    assert is_sql_query("CREATE TABLE test (id INT, name VARCHAR);")
    assert is_sql_query("DROP TABLE IF EXISTS test;")
    assert is_sql_query("ALTER TABLE test ADD COLUMN age INT;")
    assert is_sql_query("INSERT INTO test VALUES (1, 'Alice', 30);")
    assert is_sql_query("UPDATE test SET age = 31 WHERE id = 1;")
    assert is_sql_query("DELETE FROM test WHERE id = 1;")
    assert is_sql_query("COPY test TO 'out.csv' (FORMAT CSV);")
    assert is_sql_query("CHECKPOINT;")


def test_is_sql_query_without_extract_statements() -> None:
    """Test is_sql_query fallback when extract_statements is absent on connection."""
    from t1d_analytics.analytics import is_sql_query

    mock_conn = MagicMock(spec=["execute", "close"])
    mock_conn.execute.return_value = None

    assert not hasattr(mock_conn, "extract_statements")
    assert is_sql_query("SELECT 42", conn=mock_conn)
    assert is_sql_query("EXPLAIN SELECT 42", conn=mock_conn)

    # ParserException simulation
    mock_conn.execute.side_effect = duckdb.ParserException("syntax error")
    assert not is_sql_query("INVALID SYNTAX", conn=mock_conn)

    # General Exception simulation (e.g. CatalogException) returns True
    mock_conn.execute.side_effect = Exception("Catalog error")
    assert is_sql_query("SELECT * FROM missing_table", conn=mock_conn)


def test_extract_tar_and_nested_archives(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test extracting .tar.gz, .tgz, .tar.bz2, and nested archives recursively."""
    import tarfile

    from t1d_analytics.analytics import extract_zips

    # 1. Create a tar.gz archive
    tar_gz_path = tmp_path / "sample.tar.gz"
    inner_file = tmp_path / "tar_content.txt"
    inner_file.write_text("tar content")
    with tarfile.open(tar_gz_path, "w:gz") as tar:
        tar.add(inner_file, arcname="tar_content.txt")

    # 2. Create a nested zip archive: outer.zip containing inner.zip
    nested_dir = tmp_path / "nested_source"
    nested_dir.mkdir()
    inner_zip_path = nested_dir / "inner.zip"
    with zipfile.ZipFile(inner_zip_path, "w") as zf:
        zf.writestr("nested_data.txt", "deep data")

    outer_zip_path = tmp_path / "outer.zip"
    with zipfile.ZipFile(outer_zip_path, "w") as zf:
        zf.write(inner_zip_path, arcname="inner.zip")

    # 3. Create .tgz and .tar.bz2 archives
    tgz_path = tmp_path / "archive.tgz"
    with tarfile.open(tgz_path, "w:gz") as tar:
        tar.add(inner_file, arcname="tgz_content.txt")

    tbz_path = tmp_path / "archive.tar.bz2"
    with tarfile.open(tbz_path, "w:bz2") as tar:
        tar.add(inner_file, arcname="tbz_content.txt")

    extract_zips(str(tmp_path), max_depth=3)

    assert (tmp_path / "sample" / "tar_content.txt").exists()
    assert (tmp_path / "archive" / "tgz_content.txt").exists()
    assert (tmp_path / "outer" / "inner" / "nested_data.txt").exists()


def test_extract_tar_path_traversal(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test path traversal Zip Slip prevention in tar archives."""
    import tarfile

    from t1d_analytics.analytics import extract_zips

    tar_path = tmp_path / "traversal.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        ti = tarfile.TarInfo(name="../../evil.sh")
        ti.size = 9
        tar.addfile(ti, io.BytesIO(b"echo evil"))

    extract_zips(str(tmp_path))
    assert not (tmp_path.parent / "evil.sh").exists()
    assert "path traversal detected" in capsys.readouterr().out


def test_extract_bad_tar_archive(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test handling corrupted tar archives."""
    from t1d_analytics.analytics import extract_zips

    bad_tar = tmp_path / "corrupt.tar.gz"
    bad_tar.write_bytes(b"not a valid tar gzip stream")

    extract_zips(str(tmp_path))
    assert "Bad Archive File" in capsys.readouterr().out


def test_clinical_data_ingestion_and_manifest(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test SAS (.xpt) and SPSS (.sav) data loading, type conversions, and manifest tracking."""
    import numpy as np
    import pandas as pd  # type: ignore[import-untyped]
    import pyreadstat  # type: ignore[import-untyped]
    from t1d_analytics.analytics import get_ingestion_manifest, load_data_to_duckdb

    data_dir = tmp_path / "clinical_data"
    data_dir.mkdir()
    db_path = str(tmp_path / "clinical.duckdb")

    # Create synthetic SPSS dataset with dates and NaNs
    df_spss = pd.DataFrame(
        {
            "PATIENT_ID": [101, 102],
            "VISIT_DATE": pd.date_range("2024-01-01", periods=2),
            "GLUCOSE": [115.5, np.nan],
        }
    )
    spss_file = data_dir / "trial_spss.sav"
    pyreadstat.write_sav(df_spss, str(spss_file))

    # Create synthetic SAS XPT dataset
    df_xpt = pd.DataFrame(
        {
            "PATIENT_ID": [201, 202],
            "HBA1C": [6.5, np.nan],
        }
    )
    xpt_file = data_dir / "trial_sas.xpt"
    pyreadstat.write_xport(df_xpt, str(xpt_file))

    # First load with clinical flags
    load_data_to_duckdb(
        str(data_dir),
        db_path,
        include_sas=True,
        include_spss=True,
    )

    manifest_records = get_ingestion_manifest(db_path)
    assert len(manifest_records) == 2
    table_names = [r["table_name"] for r in manifest_records]
    assert "trial_spss" in table_names
    assert "trial_sas" in table_names

    conn = duckdb.connect(db_path)
    spss_rows = conn.execute(
        "SELECT subject_id, glucose_value FROM trial_spss ORDER BY subject_id"
    ).fetchall()
    assert spss_rows[0] == (101, 115.5)
    assert spss_rows[1] == (102, None)  # NaN translated to NULL

    sas_rows = conn.execute(
        "SELECT subject_id, hba1c FROM trial_sas ORDER BY subject_id"
    ).fetchall()
    assert sas_rows[0] == (201, 6.5)
    assert sas_rows[1] == (202, None)
    conn.close()

    # Second load: identical files should skip via manifest
    capsys.readouterr()
    load_data_to_duckdb(
        str(data_dir),
        db_path,
        include_sas=True,
        include_spss=True,
    )
    assert "already exists, skipping" in capsys.readouterr().out


def test_read_clinical_dataframe_edge_cases(tmp_path: Path, mocker: MagicMock) -> None:
    """Test error handling in _read_clinical_dataframe."""
    from t1d_analytics.analytics import _read_clinical_dataframe

    # Unsupported format
    unsupported = tmp_path / "test.unsupported"
    unsupported.write_text("sample")
    with pytest.raises(ValueError, match="Unsupported clinical file format"):
        _read_clinical_dataframe(unsupported)

    # Empty/None returned
    mocker.patch("pyreadstat.read_sav", return_value=(None, None))
    sav_file = tmp_path / "empty.sav"
    sav_file.write_text("data")
    with pytest.raises(ValueError, match="Could not read clinical file"):
        _read_clinical_dataframe(sav_file)


def test_get_ingestion_manifest_edge_cases(tmp_path: Path) -> None:
    """Test get_ingestion_manifest with non-existent database or missing table."""
    from t1d_analytics.analytics import get_ingestion_manifest

    assert get_ingestion_manifest(str(tmp_path / "nonexistent.duckdb")) == []

    empty_db = str(tmp_path / "empty.duckdb")
    conn = duckdb.connect(empty_db)
    conn.execute("CREATE TABLE dummy (id INT)")
    conn.close()
    assert get_ingestion_manifest(empty_db) == []


def test_read_clinical_dataframe_sas7bdat_and_fallbacks(
    tmp_path: Path, mocker: MagicMock
) -> None:
    """Test reading .sas7bdat, decoding bytes in columns, and fallback to pandas readers."""
    import pandas as pd
    from t1d_analytics.analytics import _read_clinical_dataframe

    # 1. sas7bdat success with byte data to decode
    raw_df = pd.DataFrame({b"col_bytes": [b"val_bytes", "str_val"]})
    mocker.patch("pyreadstat.read_sas7bdat", return_value=(raw_df, None))
    sas_file = tmp_path / "test.sas7bdat"
    sas_file.write_text("data")
    res_df = _read_clinical_dataframe(sas_file)
    assert "col_bytes" in res_df.columns
    assert res_df["col_bytes"].iloc[0] == "val_bytes"

    # 2. SAS pyreadstat failure falling back to pd.read_sas
    mocker.patch("pyreadstat.read_sas7bdat", side_effect=Exception("Read failure"))
    mock_pandas_sas = mocker.patch(
        "pandas.read_sas", return_value=pd.DataFrame({"a": [1]})
    )
    fallback_sas_df = _read_clinical_dataframe(sas_file)
    assert fallback_sas_df["a"].iloc[0] == 1
    mock_pandas_sas.assert_called_once_with(str(sas_file), format="sas7bdat")

    # 3. SPSS pyreadstat failure falling back to pd.read_spss
    sav_file = tmp_path / "test.sav"
    sav_file.write_text("data")
    mocker.patch("pyreadstat.read_sav", side_effect=Exception("Read failure"))
    mock_pandas_spss = mocker.patch(
        "pandas.read_spss", return_value=pd.DataFrame({"b": [2]})
    )
    fallback_sav_df = _read_clinical_dataframe(sav_file)
    assert fallback_sav_df["b"].iloc[0] == 2
    mock_pandas_spss.assert_called_once_with(str(sav_file))


def test_extract_zips_max_depth_loop_completion(tmp_path: Path) -> None:
    """Test that extract_zips terminates cleanly when max_depth iterations are exhausted."""
    from t1d_analytics.analytics import extract_zips

    # Create a zip that produces a nested zip
    outer_zip = tmp_path / "depth_test.zip"
    with zipfile.ZipFile(outer_zip, "w") as zf:
        zf.writestr("test.txt", "content")

    # With max_depth=1, loop completes without reaching break on empty unprocessed
    extract_zips(str(tmp_path), max_depth=1)
    assert (tmp_path / "depth_test" / "test.txt").exists()


def test_extract_sql_from_response_variations() -> None:
    """Test extract_sql_from_response under various LLM output formats."""
    from t1d_analytics.analytics import extract_sql_from_response

    # Non-string conversion
    assert extract_sql_from_response(12345) == "12345"  # type: ignore[arg-type]

    # Multi-block markdown with reasoning
    response = """Here is my reasoning.
```duckdb
SELECT 1;
```
And here is the final query:
```sql
SELECT * FROM users WHERE age > 18;
```
Hope this helps!"""
    assert extract_sql_from_response(response) == "SELECT * FROM users WHERE age > 18;"

    # Plain sql\n and duckdb\n prefixes
    assert extract_sql_from_response("sql\nSELECT 2;") == "SELECT 2;"
    assert extract_sql_from_response("duckdb\nSELECT 3;") == "SELECT 3;"


def test_get_clinical_variable_mappings(mocker: typing.Any) -> None:
    """Test get_clinical_variable_mappings handling existing, missing, and malformed files."""
    from t1d_analytics.analytics import get_clinical_variable_mappings

    # Normal case
    mappings = get_clinical_variable_mappings()
    assert "subject_id" in mappings
    assert "hba1c" in mappings

    # Missing file
    mocker.patch("pathlib.Path.exists", return_value=False)
    assert get_clinical_variable_mappings() == {}

    # Malformed JSON
    mocker.patch("pathlib.Path.exists", return_value=True)
    mocker.patch("builtins.open", mocker.mock_open(read_data="not json"))
    assert get_clinical_variable_mappings() == {}

    # JSON that is not a dictionary (e.g. list)
    mocker.patch("builtins.open", mocker.mock_open(read_data='["not", "a", "dict"]'))
    assert get_clinical_variable_mappings() == {}


def test_profile_table(tmp_path: Path) -> None:
    """Test profile_table data quality metrics, null calculation, and error cases."""
    from t1d_analytics.analytics import profile_table

    db_file = tmp_path / "profile_test.duckdb"
    conn = duckdb.connect(str(db_file))
    conn.execute(
        """
        CREATE TABLE metrics (
            id INT,
            val FLOAT,
            notes VARCHAR
        )
        """
    )
    conn.execute(
        "INSERT INTO metrics VALUES (1, 10.5, 'note1'), (2, 20.5, NULL), (3, NULL, 'note3')"
    )
    conn.close()

    # Success
    report = profile_table(str(db_file), "metrics")
    assert report["table_name"] == "metrics"
    assert report["total_rows"] == 3
    assert report["column_count"] == 3

    cols = {c["column_name"]: c for c in report["columns"]}
    assert cols["id"]["null_count"] == 0
    assert cols["id"]["min"] == 1
    assert cols["id"]["max"] == 3
    assert cols["val"]["null_count"] == 1
    assert cols["val"]["min"] == 10.5
    assert cols["notes"]["null_count"] == 1

    # Empty table profile
    empty_db = tmp_path / "empty.duckdb"
    conn2 = duckdb.connect(str(empty_db))
    conn2.execute("CREATE TABLE empty_tbl (id INT)")
    conn2.execute("CREATE TABLE all_nulls (num FLOAT)")
    conn2.execute("INSERT INTO all_nulls VALUES (NULL), (NULL)")
    conn2.close()
    empty_report = profile_table(str(empty_db), "empty_tbl")
    assert empty_report["total_rows"] == 0
    assert empty_report["columns"][0]["null_percentage"] == 0.0

    null_report = profile_table(str(empty_db), "all_nulls")
    assert null_report["columns"][0]["mean"] is None

    # Invalid table name
    with pytest.raises(ValueError, match="Invalid table identifier"):
        profile_table(str(db_file), "metrics; DROP TABLE metrics;")

    # Non-existent DB
    with pytest.raises(FileNotFoundError, match="Database file not found"):
        profile_table(str(tmp_path / "nonexistent.duckdb"), "metrics")

    # Non-existent table in DB
    with pytest.raises(ValueError, match="does not exist in database"):
        profile_table(str(db_file), "missing_table")


def test_load_data_to_duckdb_missing_dir(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Test load_data_to_duckdb gracefully returns when data_dir does not exist."""
    from t1d_analytics.analytics import load_data_to_duckdb

    load_data_to_duckdb(
        "/nonexistent/directory/path/that/does/not/exist", "test_dummy.duckdb"
    )
    assert "does not exist" in capsys.readouterr().out


def test_profile_table_no_numeric_columns(tmp_path: Path) -> None:
    """Test profile_table on table with only non-numeric columns."""
    from t1d_analytics.analytics import profile_table

    db = tmp_path / "str_only.duckdb"
    conn = duckdb.connect(str(db))
    conn.execute("CREATE TABLE text_tbl (name VARCHAR)")
    conn.execute("INSERT INTO text_tbl VALUES ('alice'), ('bob')")
    conn.close()

    res = profile_table(str(db), "text_tbl")
    assert "min" not in res["columns"][0]


def test_extract_sql_from_response_blocks_without_keywords() -> None:
    """Test extract_sql_from_response when code block has non-standard statement."""
    from t1d_analytics.analytics import extract_sql_from_response

    text = "```\nCREATE TABLE foo (a INT);\n```"
    assert extract_sql_from_response(text) == "CREATE TABLE foo (a INT);"


def test_profile_special_characters_and_quotes(tmp_path: Path) -> None:
    """Test profiling table with complex quoted and unicode column names."""
    from t1d_analytics.analytics import profile_table

    db = tmp_path / "special_cols.duckdb"
    conn = duckdb.connect(str(db))
    conn.execute(
        'CREATE TABLE "t_special" ("col""quoted""" INT, "Glucose (mg/dL)" FLOAT, "Patient #" VARCHAR, "日本語" VARCHAR)'
    )
    conn.execute(
        "INSERT INTO \"t_special\" VALUES (10, 115.5, 'P-1', 'テスト'), (NULL, 130.0, 'P-2', NULL)"
    )
    conn.close()

    res = profile_table(str(db), "t_special")
    assert res["table_name"] == "t_special"
    assert res["total_rows"] == 2
    assert res["column_count"] == 4
    col_names = [c["column_name"] for c in res["columns"]]
    assert 'col"quoted"' in col_names
    assert "Glucose (mg/dL)" in col_names
    assert "Patient #" in col_names
    assert "日本語" in col_names


def test_extract_tar_symlink_and_hardlink_skipped(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test that tar archive members with symlinks or hardlinks are skipped safely."""
    import io
    import tarfile

    from t1d_analytics.analytics import extract_zips

    tar_path = tmp_path / "links.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        # Regular file
        regular = tarfile.TarInfo(name="regular.txt")
        regular.size = 5
        tar.addfile(regular, io.BytesIO(b"hello"))

        # Symlink
        sym = tarfile.TarInfo(name="symlink.txt")
        sym.type = tarfile.SYMTYPE
        sym.linkname = "/etc/passwd"
        tar.addfile(sym)

        # Hardlink
        lnk = tarfile.TarInfo(name="hardlink.txt")
        lnk.type = tarfile.LNKTYPE
        lnk.linkname = "regular.txt"
        tar.addfile(lnk)

    extract_zips(str(tmp_path))
    out = capsys.readouterr().out
    assert "symlink or hardlink detected" in out
    extract_dir = tmp_path / "links"
    assert (extract_dir / "regular.txt").exists()
    assert not (extract_dir / "symlink.txt").exists()


def test_extract_zip_symlink_skipped(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test that zip archive entries with symlink attribute mode are skipped safely."""
    import zipfile

    from t1d_analytics.analytics import extract_zips

    zip_path = tmp_path / "symlink.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        # Regular entry
        zf.writestr("normal.txt", "normal file")
        # Symlink entry
        zi = zipfile.ZipInfo("sym_target.txt")
        # Mode 0o120777: S_IFLNK (0o120000) | 0o777 permissions
        zi.external_attr = 0o120777 << 16
        zf.writestr(zi, "/outside/path")

    extract_zips(str(tmp_path))
    out = capsys.readouterr().out
    assert "symlink detected" in out
    extract_dir = tmp_path / "symlink"
    assert (extract_dir / "normal.txt").exists()


def test_is_sql_query_fallback_side_effect_free() -> None:
    """Test is_sql_query fallback logic when extract_statements is absent on custom connection."""
    from unittest.mock import MagicMock

    from t1d_analytics.analytics import is_sql_query

    mock_conn = MagicMock(spec=[])
    assert is_sql_query("CHECKPOINT", conn=mock_conn)
    assert is_sql_query("ATTACH 'foo.db'", conn=mock_conn)
    assert is_sql_query("CALL my_proc()", conn=mock_conn)


def test_profile_table_no_columns(tmp_path: Path, mocker: typing.Any) -> None:
    """Test profile_table when table describe returns empty columns."""
    from t1d_analytics.analytics import profile_table

    db = tmp_path / "empty_cols.duckdb"
    conn = duckdb.connect(str(db))
    conn.execute("CREATE TABLE t_empty (id INT)")
    conn.close()

    mock_conn = mocker.MagicMock()
    mock_conn.execute.return_value.fetchall.side_effect = [
        [("t_empty",)],  # SHOW TABLES
        [],  # DESCRIBE returns empty
    ]
    mock_conn.execute.return_value.fetchone.return_value = [0]
    mocker.patch("duckdb.connect", return_value=mock_conn)

    res = profile_table(str(db), "t_empty")
    assert res["column_count"] == 0
    assert res["columns"] == []


def test_export_table_validations(tmp_path: Path) -> None:
    """Test validation errors for export_table (invalid ID, missing file, missing table, bad format)."""
    from t1d_analytics.analytics import export_table

    db_file = tmp_path / "validations.duckdb"
    conn = duckdb.connect(str(db_file))
    conn.execute("CREATE TABLE test_tbl (id INT)")
    conn.close()

    # Invalid table identifier
    with pytest.raises(ValueError, match="Invalid table identifier"):
        export_table(str(db_file), "bad table;", tmp_path / "out.csv", "csv")

    # Missing DB file
    with pytest.raises(FileNotFoundError, match="Database file not found"):
        export_table(
            str(tmp_path / "nonexistent.duckdb"),
            "test_tbl",
            tmp_path / "out.csv",
            "csv",
        )

    # Unsupported format
    with pytest.raises(ValueError, match="Unsupported export format"):
        export_table(str(db_file), "test_tbl", tmp_path / "out.xml", "xml")

    # Nonexistent table
    with pytest.raises(ValueError, match="does not exist in database"):
        export_table(str(db_file), "missing_tbl", tmp_path / "out.csv", "csv")


def test_export_table_parquet_and_csv(tmp_path: Path) -> None:
    """Test exporting table to Parquet and CSV formats."""
    from t1d_analytics.analytics import export_table

    db_file = tmp_path / "formats.duckdb"
    conn = duckdb.connect(str(db_file))
    conn.execute("CREATE TABLE cohort AS SELECT 1 AS id, 'Alice' AS name, 5.5 AS a1c")
    conn.close()

    # Parquet export
    pq_out = tmp_path / "cohort.parquet"
    export_table(str(db_file), "cohort", pq_out, "parquet")
    assert pq_out.exists()
    conn_pq = duckdb.connect()
    rows_pq = conn_pq.execute(f"SELECT * FROM read_parquet('{pq_out}')").fetchall()
    assert len(rows_pq) == 1
    assert rows_pq[0] == (1, "Alice", 5.5)
    conn_pq.close()

    # CSV export
    csv_out = tmp_path / "nested" / "cohort.csv"
    export_table(str(db_file), "cohort", csv_out, "csv")
    assert csv_out.exists()
    content = csv_out.read_text()
    assert "id,name,a1c" in content
    assert "1,Alice,5.5" in content


def test_export_table_excel_formula_sanitization(tmp_path: Path) -> None:
    """Test exporting table to Excel with formula injection sanitization."""
    from openpyxl import load_workbook
    from t1d_analytics.analytics import export_table

    db_file = tmp_path / "excel.duckdb"
    conn = duckdb.connect(str(db_file))
    conn.execute(
        """
        CREATE TABLE clinical (
            id INT,
            val_float DOUBLE,
            null_col VARCHAR,
            flag BOOLEAN,
            note VARCHAR,
            formula_inj VARCHAR,
            num_negative VARCHAR,
            plus_bad VARCHAR,
            end_col VARCHAR
        )
    """
    )
    conn.execute(
        """
        INSERT INTO clinical VALUES
        (1, 3.14, NULL, TRUE, 'normal note', '=HYPERLINK("http://evil.com")', '-42.5', '+bad_formula', 'done')
    """
    )
    conn.close()

    xlsx_out = tmp_path / "clinical.xlsx"
    export_table(str(db_file), "clinical", xlsx_out, "excel")
    assert xlsx_out.exists()

    wb = load_workbook(str(xlsx_out), read_only=True)
    ws = wb.active
    rows = list(ws.iter_rows(values_only=True))
    assert rows[0] == (
        "id",
        "val_float",
        "null_col",
        "flag",
        "note",
        "formula_inj",
        "num_negative",
        "plus_bad",
        "end_col",
    )
    row1 = rows[1]
    assert row1[0] == 1
    assert row1[1] == 3.14
    assert row1[2] is None
    assert row1[3] is True
    assert row1[4] == "normal note"
    # Escaped formula
    assert row1[5].startswith("'=")
    # -42.5 is numeric, so it should not be escaped with apostrophe
    assert row1[6] == "-42.5"
    # +bad_formula is non-numeric, so it should be escaped
    assert row1[7].startswith("'+")
    assert row1[8] == "done"
    wb.close()
