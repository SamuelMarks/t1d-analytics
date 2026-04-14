"""Tests for the FastAPI application."""

import sys
import types
from unittest.mock import MagicMock, patch

if "any_llm" not in sys.modules:
    sys.modules["any_llm"] = types.ModuleType("any_llm")
    sys.modules["any_llm"].AnyLLM = MagicMock()

import duckdb
import pytest
from fastapi.testclient import TestClient

from t1d_analytics.api import app, execute_sql, generate_sql_from_nl

client = TestClient(app)


@pytest.fixture
def mock_db(tmp_path) -> str:
    """Fixture to provide a test duckdb database path."""
    db_path = str(tmp_path / "test.duckdb")
    conn = duckdb.connect(db_path)
    conn.execute("CREATE TABLE users (id INTEGER, name VARCHAR)")
    conn.execute("INSERT INTO users VALUES (1, 'Alice'), (2, 'Bob')")
    conn.close()
    return db_path


def test_execute_sql(mock_db: str) -> None:
    """Test standard SQL execution."""
    res = execute_sql(mock_db, "SELECT * FROM users")
    assert len(res) == 2
    assert res[0]["name"] == "Alice"


def test_execute_sql_no_result(mock_db: str) -> None:
    """Test SQL execution that returns no rows."""
    res = execute_sql(
        mock_db, "SELECT * FROM duckdb_tables() WHERE table_name = 'nonexistent_dummy'"
    )
    assert res == []


def test_execute_sql_error(mock_db: str) -> None:
    """Test SQL execution with invalid query."""
    with pytest.raises(ValueError, match="SQL execution error"):
        execute_sql(mock_db, "SELECT * FROM non_existent")


@patch("any_llm.AnyLLM.create")
def test_generate_sql_from_nl_success(mock_create: MagicMock, mock_db: str) -> None:
    """Test natural language to SQL translation."""
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "```sql\nSELECT * FROM users;\n```"
    mock_llm.completion.return_value = mock_response
    mock_create.return_value = mock_llm

    sql = generate_sql_from_nl(mock_db, "get users")
    assert sql == "SELECT * FROM users;"


@patch("any_llm.AnyLLM.create")
def test_generate_sql_from_nl_single_line_codeblock(
    mock_create: MagicMock, mock_db: str
) -> None:
    """Test NL to SQL with single line codeblock."""
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "```SELECT * FROM users```"
    mock_llm.completion.return_value = mock_response
    mock_create.return_value = mock_llm

    sql = generate_sql_from_nl(mock_db, "get users")
    assert sql == "SELECT * FROM users"


@patch("any_llm.AnyLLM.create")
def test_generate_sql_from_nl_prefixes(mock_create: MagicMock, mock_db: str) -> None:
    """Test NL to SQL with 'sql' and 'duckdb' prefixes."""
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_llm.completion.return_value = mock_response
    mock_create.return_value = mock_llm

    # Test sql prefix
    mock_response.choices[0].message.content = "sql\nSELECT * FROM a"
    sql = generate_sql_from_nl(mock_db, "test")
    assert sql == "SELECT * FROM a"

    # Test duckdb prefix
    mock_response.choices[0].message.content = "duckdb\nSELECT * FROM b"
    sql = generate_sql_from_nl(mock_db, "test")
    assert sql == "SELECT * FROM b"


def test_generate_sql_from_nl_no_module(mock_db: str) -> None:
    """Test handling of missing any_llm module."""
    orig = sys.modules.get("any_llm")
    sys.modules["any_llm"] = None  # Force ImportError
    try:
        with pytest.raises(RuntimeError, match="not installed"):
            generate_sql_from_nl(mock_db, "test")
    finally:
        if orig is not None:
            sys.modules["any_llm"] = orig
        else:
            sys.modules.pop("any_llm", None)


@patch("t1d_analytics.api.duckdb.connect")
def test_generate_sql_from_nl_schema_error(mock_connect: MagicMock) -> None:
    """Test handling of DB connection error."""
    mock_connect.side_effect = Exception("DB error")
    with pytest.raises(ValueError, match="Failed to read schema: DB error"):
        generate_sql_from_nl("dummy.db", "test")


@patch("any_llm.AnyLLM.create")
def test_generate_sql_from_nl_llm_error(mock_create: MagicMock, mock_db: str) -> None:
    """Test handling of LLM completion error."""
    mock_create.side_effect = Exception("API limit")
    with pytest.raises(RuntimeError, match="LLM translation error: API limit"):
        generate_sql_from_nl(mock_db, "test")


def test_chat_endpoint_empty_message() -> None:
    """Test API rejects empty messages."""
    response = client.post("/api/chat", json={"message": "   "})
    assert response.status_code == 400
    assert "cannot be empty" in response.json()["detail"]


def test_chat_endpoint_sql_success(mock_db: str) -> None:
    """Test API handles direct SQL queries successfully."""
    response = client.post(
        "/api/chat",
        json={"message": "SELECT * FROM users", "model": "sql", "db_path": mock_db},
    )
    assert response.status_code == 200
    data = response.json()
    assert "Executed literal SQL" in data["content"]
    assert len(data["sqlResult"]) == 2
    assert data["sqlResult"][0]["name"] == "Alice"


@patch("t1d_analytics.api.generate_sql_from_nl")
def test_chat_endpoint_nl_success(mock_generate: MagicMock, mock_db: str) -> None:
    """Test API handles NLP queries successfully."""
    mock_generate.return_value = "SELECT * FROM users"
    response = client.post(
        "/api/chat",
        json={"message": "give me users", "model": "gemma4", "db_path": mock_db},
    )
    assert response.status_code == 200
    data = response.json()
    assert "Generated SQL" in data["content"]
    assert data["sqlResult"] is None
    assert data["sqlQuery"] == "SELECT * FROM users"


@patch("t1d_analytics.api.execute_sql")
def test_chat_endpoint_value_error(mock_execute: MagicMock, mock_db: str) -> None:
    """Test API handles ValueError (SQL syntax error)."""
    mock_execute.side_effect = ValueError("Syntax error")
    response = client.post(
        "/api/chat",
        json={"message": "SELECT invalid", "model": "sql", "db_path": mock_db},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["error"] == "Syntax error"
    assert "Error executing database operation" in data["content"]


@patch("t1d_analytics.api.generate_sql_from_nl")
def test_chat_endpoint_runtime_error(mock_generate: MagicMock, mock_db: str) -> None:
    """Test API handles RuntimeError (LLM failure)."""
    mock_generate.side_effect = RuntimeError("LLM offline")
    response = client.post(
        "/api/chat",
        json={"message": "give me users", "model": "gemma4", "db_path": mock_db},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["error"] == "LLM offline"
    assert "Error during NLP translation" in data["content"]


@patch("t1d_analytics.api.generate_sql_from_nl")
def test_chat_endpoint_generic_exception(
    mock_generate: MagicMock, mock_db: str
) -> None:
    """Test API handles generic unexpected exceptions."""
    mock_generate.side_effect = Exception("System failure")
    response = client.post(
        "/api/chat",
        json={"message": "give me users", "model": "gemma4", "db_path": mock_db},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["error"] == "System failure"
    assert "unexpected error" in data["content"]


def test_list_models_success(monkeypatch):
    """Test successful model listing from local Ollama."""
    import json
    import urllib.request

    class MockResponse:
        def read(self):
            return json.dumps(
                {
                    "models": [
                        {"name": "gemma4", "size": 12345},
                        {"name": "llama3", "size": 67890},
                    ]
                }
            ).encode()

    class MockUrlopen:
        def __init__(self, req, timeout=None):
            self.req = req

        def __enter__(self):
            return MockResponse()

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

    monkeypatch.setattr(urllib.request, "urlopen", MockUrlopen)

    response = client.get("/api/models")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert len(data["models"]) == 2
    assert data["models"][0]["name"] == "gemma4"
    assert data["models"][1]["name"] == "llama3"


def test_list_models_url_error(monkeypatch):
    """Test fallback when Ollama is unreachable via URLError."""
    import urllib.error
    import urllib.request

    def mock_urlopen(*args, **kwargs):
        raise urllib.error.URLError("Connection refused")

    monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)

    response = client.get("/api/models")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert len(data["models"]) == 1
    assert data["models"][0]["name"] == "gemma4"


def test_list_models_general_error(monkeypatch):
    """Test fallback when an unexpected exception occurs."""
    import urllib.request

    def mock_urlopen(*args, **kwargs):
        raise Exception("Unexpected boom")

    monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)

    response = client.get("/api/models")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert len(data["models"]) == 1
    assert data["models"][0]["name"] == "gemma4"


@patch("t1d_analytics.api.duckdb.connect")
def test_get_schema_success(mock_connect: MagicMock) -> None:
    """Test schema endpoint returns structured schema."""
    mock_conn = MagicMock()
    # Mock SHOW TABLES
    mock_conn.execute.return_value.fetchall.side_effect = [
        [("users",)],
        [("id", "INTEGER"), ("name", "VARCHAR")],
    ]
    mock_connect.return_value = mock_conn

    response = client.get("/api/schema")
    assert response.status_code == 200
    data = response.json()
    assert len(data["tables"]) == 1
    assert data["tables"][0]["name"] == "users"
    assert data["tables"][0]["columns"][0]["name"] == "id"
    assert data["tables"][0]["columns"][0]["type"] == "INTEGER"


@patch("t1d_analytics.api.duckdb.connect")
def test_get_schema_error(mock_connect: MagicMock) -> None:
    """Test schema endpoint handles errors gracefully."""
    mock_connect.side_effect = Exception("DB connection failed")

    response = client.get("/api/schema")
    assert response.status_code == 200
    data = response.json()
    assert len(data["tables"]) == 0


def test_get_table_data(mock_db, monkeypatch):
    """Test fetching valid table data with pagination."""
    monkeypatch.setenv("T1D_DB_PATH", mock_db)
    response = client.get("/api/table/users")
    assert response.status_code == 200
    data = response.json()
    assert "rows" in data
    assert len(data["rows"]) == 2
    assert data["rows"][0]["id"] == 1
    assert data["rows"][0]["name"] == "Alice"


def test_get_table_data_invalid_table(mock_db, monkeypatch):
    """Test fetching data from a table that doesn't exist."""
    monkeypatch.setenv("T1D_DB_PATH", mock_db)
    response = client.get("/api/table/non_existent_table")
    assert response.status_code == 404


def test_get_table_data_sql_injection(mock_db, monkeypatch):
    """Test protection against basic SQL injection in table name."""
    monkeypatch.setenv("T1D_DB_PATH", mock_db)
    response = client.get("/api/table/invalid table name;")
    assert response.status_code == 400


def test_get_table_data_exception(mock_db, monkeypatch):
    """Test server error handling when DB operations fail."""
    monkeypatch.setenv("T1D_DB_PATH", mock_db)
    from unittest.mock import patch

    with patch("duckdb.connect", side_effect=Exception("DB Failure")):
        response = client.get("/api/table/users")
        assert response.status_code == 500
        assert "DB Failure" in response.json()["detail"]
