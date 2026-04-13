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
    res = execute_sql(mock_db, "CREATE TABLE dummy (id INTEGER)")
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
    assert len(data["sqlResult"]) == 2


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
