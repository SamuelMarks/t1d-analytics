"""Tests for the analytics module."""

import sys
import types
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

if "any_llm" not in sys.modules:
    sys.modules["any_llm"] = types.ModuleType("any_llm")
    sys.modules["any_llm"].AnyLLM = MagicMock()

import duckdb
import pytest

from t1d_analytics.analytics import (
    extract_zips,
    get_database_schema,
    handle_natural_language,
    load_data_to_duckdb,
    run_query_repl,
)


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
    assert conn.execute("SELECT sum(val) FROM data1").fetchone()[0] == 30
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
    assert conn.execute("SELECT sum(val) FROM data_tab").fetchone()[0] == 30
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
    assert conn.execute("SELECT sum(val) FROM data_utf16").fetchone()[0] == 30
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
    sys.modules["any_llm"] = None

    conn = duckdb.connect(":memory:")
    handle_natural_language(conn, "test")
    assert "any-llm-sdk[ollama] is not installed" in capsys.readouterr().out
    conn.close()

    if original_module:
        sys.modules["any_llm"] = original_module


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

    def mock_execute(query):
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

def test_load_data_no_matches_file(tmp_path: Path, mocker) -> None:
    """Test load_data_to_duckdb when matches file does not exist."""
    import duckdb

    from t1d_analytics.analytics import load_data_to_duckdb
    
    mocker.patch("pathlib.Path.exists", autospec=True, side_effect=lambda self: False if "likely_matches" in str(self) else True)
    
    (tmp_path / "data.csv").write_text("A,B\n1,2")
    db_path = str(tmp_path / "test.duckdb")
    load_data_to_duckdb(str(tmp_path), db_path)
    conn = duckdb.connect(db_path)
    assert "data" in [r[0] for r in conn.execute("SHOW TABLES").fetchall()]

def test_handle_natural_language_no_result(mocker) -> None:
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

def test_run_query_repl_no_result(mocker, tmp_path) -> None:
    """Test run_query_repl when sql returns no result."""
    from t1d_analytics.analytics import run_query_repl
    
    db_path = tmp_path / "test.duckdb"
    db_path.touch()
    
    inputs = ["SELECT 1", "exit"]
    mocker.patch("builtins.input", side_effect=inputs)
    
    mock_conn = mocker.MagicMock()
    mock_conn.sql.return_value = None
    mocker.patch("duckdb.connect", return_value=mock_conn)
    
    run_query_repl(str(db_path))
    mock_conn.sql.assert_called_once()
