"""Tests for the FastAPI application."""

import io
import json
import os
import sys
import time
import types
import typing
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import duckdb
import pytest
from fastapi.testclient import TestClient

from t1d_analytics.api import app, execute_sql, generate_sql_from_nl

if "any_llm" not in sys.modules:
    sys.modules["any_llm"] = types.ModuleType("any_llm")
    setattr(sys.modules["any_llm"], "AnyLLM", MagicMock())

client = TestClient(app)


@pytest.fixture
def mock_db(tmp_path: typing.Any) -> str:
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
    with pytest.raises(ValueError, match=r"{\"error_code\": \"backend.sqlExecution.*"):
        execute_sql(mock_db, "SELECT * FROM non_existent")


def test_execute_sql_sandboxed_external_access(mock_db: str) -> None:
    """Test SQL execution sandboxing blocks external filesystem access."""
    with pytest.raises(ValueError, match=r"backend\.sqlExecution"):
        execute_sql(mock_db, "SELECT * FROM read_csv('/etc/passwd')")


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
    assert sql[1] == "SELECT * FROM users;"


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
    assert sql[1] == "SELECT * FROM users"


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
    assert sql[1] == "SELECT * FROM a"

    # Test duckdb prefix
    mock_response.choices[0].message.content = "duckdb\nSELECT * FROM b"
    sql = generate_sql_from_nl(mock_db, "test")
    assert sql[1] == "SELECT * FROM b"


def test_generate_sql_from_nl_no_module(mock_db: str) -> None:
    """Test handling of missing any_llm module."""
    orig = sys.modules.get("any_llm")
    sys.modules["any_llm"] = None  # type: ignore  # Force ImportError
    try:
        with pytest.raises(RuntimeError, match=r"backend\.missingSdk.*"):
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
    with pytest.raises(ValueError, match=r"backend\.readSchemaFailed.*"):
        generate_sql_from_nl("dummy.db", "test")


@patch("any_llm.AnyLLM.create")
def test_generate_sql_from_nl_llm_error(mock_create: MagicMock, mock_db: str) -> None:
    """Test handling of LLM completion error."""
    mock_create.side_effect = Exception("API limit")
    with pytest.raises(RuntimeError, match=r"backend\.llmTranslationError.*"):
        generate_sql_from_nl(mock_db, "test")


def test_chat_endpoint_empty_message() -> None:
    """Test API rejects empty messages."""
    response = client.post("/api/chat", json={"message": "   "})
    assert response.status_code == 400
    assert response.json()["detail"]["error_code"] == "backend.emptyMessage"


def test_chat_endpoint_sql_success(mock_db: str) -> None:
    """Test API handles direct SQL queries successfully."""
    response = client.post(
        "/api/chat",
        json={"message": "SELECT * FROM users", "model": "sql", "db_path": mock_db},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["content"] == "backend.literalSql"
    assert len(data["sqlResult"]) == 2
    assert data["sqlResult"][0]["name"] == "Alice"


@patch("t1d_analytics.api.generate_sql_from_nl")
def test_chat_endpoint_nl_success(mock_generate: MagicMock, mock_db: str) -> None:
    """Test API handles NLP queries successfully and auto-executes the query."""
    mock_generate.return_value = ("Generated Markdown", "SELECT * FROM users")
    response = client.post(
        "/api/chat",
        json={"message": "give me users", "model": "gemma4", "db_path": mock_db},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["content"] == "Generated Markdown"
    assert data["sqlResult"] is not None
    assert len(data["sqlResult"]) == 2
    assert data["sqlQuery"] == "SELECT * FROM users"


@patch("t1d_analytics.api.generate_sql_from_nl")
def test_chat_endpoint_nl_sql_execution_failure(
    mock_generate: MagicMock, mock_db: str
) -> None:
    """Test API handles NLP queries where generated SQL execution fails."""
    mock_generate.return_value = (
        "Generated Markdown",
        "SELECT * FROM nonexistent_table",
    )
    response = client.post(
        "/api/chat",
        json={"message": "give me users", "model": "gemma4", "db_path": mock_db},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["content"] == "Generated Markdown"
    assert data["sqlResult"] is None
    assert data["error"] is not None
    assert data["error"]["error_code"] == "backend.sqlExecution"


@patch("t1d_analytics.api.generate_sql_from_nl")
def test_chat_endpoint_nl_no_sql_query(mock_generate: MagicMock, mock_db: str) -> None:
    """Test API handles NLP response that contains no SQL query."""
    mock_generate.return_value = ("Just an explanation, no query.", "")
    response = client.post(
        "/api/chat",
        json={"message": "hello", "model": "gemma4", "db_path": mock_db},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["content"] == "Just an explanation, no query."
    assert data["sqlResult"] is None
    assert data["sqlQuery"] == ""


def test_chat_endpoint_value_error(mock_db: str) -> None:
    """Test API handles ValueError (SQL syntax error)."""
    response = client.post(
        "/api/chat",
        json={"message": "SELECT invalid", "model": "sql", "db_path": mock_db},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["error"]["error_code"] == "backend.sqlExecution"
    assert data["content"] == "backend.errorDbExecution"


@patch("any_llm.AnyLLM.create")
def test_chat_endpoint_runtime_error(mock_create: MagicMock, mock_db: str) -> None:
    """Test API handles RuntimeError (LLM failure)."""
    mock_create.side_effect = Exception("LLM offline")
    response = client.post(
        "/api/chat",
        json={"message": "give me users", "model": "gemma4", "db_path": mock_db},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["error"]["error_code"] == "backend.llmTranslationError"
    assert data["content"] == "backend.errorNlpTranslation"


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
    assert data["error"]["error_code"] == "backend.errorUnexpected"
    assert data["error"]["params"]["error"] == "System failure"
    assert data["content"] == "backend.errorUnexpected"


def test_list_models_success(monkeypatch: typing.Any) -> None:
    """Test successful model listing from local Ollama."""
    import json
    import urllib.request

    class MockResponse:
        """Mock HTTP response object."""

        def read(self) -> bytes:
            """Return mock JSON byte payload."""
            return json.dumps(
                {
                    "models": [
                        {"name": "gemma4", "size": 12345},
                        {"name": "llama3", "size": 67890},
                    ]
                }
            ).encode()

    class MockUrlopen:
        """Mock context manager for urllib urlopen."""

        def __init__(self, req: typing.Any, timeout: typing.Any = None) -> None:
            """Initialize MockUrlopen."""
            self.req = req

        def __enter__(self) -> typing.Any:
            """Enter the context manager."""
            return MockResponse()

        def __exit__(
            self, exc_type: typing.Any, exc_val: typing.Any, exc_tb: typing.Any
        ) -> None:
            """Exit the context manager."""
            pass

    monkeypatch.setattr(urllib.request, "urlopen", MockUrlopen)

    response = client.get("/api/models")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    # 2 Ollama models + 6 cloud models
    assert len(data["models"]) == 8
    assert data["models"][0]["name"] == "gemma4"
    assert data["models"][0]["provider"] == "ollama"
    assert data["models"][1]["name"] == "llama3"
    assert any(
        m["name"] == "openai/gpt-4o" and m["provider"] == "openai"
        for m in data["models"]
    )

    # Without cloud models
    resp_local_only = client.get("/api/models?include_cloud=false")
    assert resp_local_only.status_code == 200
    assert len(resp_local_only.json()["models"]) == 2


def test_list_models_url_error(monkeypatch: typing.Any) -> None:
    """Test when Ollama is unreachable via URLError."""
    import urllib.error
    import urllib.request

    def mock_urlopen(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
        """Simulate urlopen throwing URLError."""
        raise urllib.error.URLError("Connection refused")

    monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)

    response = client.get("/api/models?include_cloud=false")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    # No fake placeholder models returned when Ollama is offline
    assert len(data["models"]) == 0

    # With cloud models included
    resp_cloud = client.get("/api/models?include_cloud=true")
    assert resp_cloud.status_code == 200
    data_cloud = resp_cloud.json()
    assert len(data_cloud["models"]) > 0
    for m in data_cloud["models"]:
        assert "available" in m
        assert "reachable" in m


def test_list_models_general_error(monkeypatch: typing.Any) -> None:
    """Test when an unexpected exception occurs during Ollama fetch."""
    import urllib.request

    def mock_urlopen(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
        """Simulate urlopen throwing a generic exception."""
        raise Exception("Unexpected boom")

    monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)

    response = client.get("/api/models?include_cloud=false")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    # No fake placeholder models returned
    assert len(data["models"]) == 0


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


def test_get_table_data(mock_db: str, monkeypatch: typing.Any) -> None:
    """Test fetching valid table data with pagination, sorting, and searching."""
    monkeypatch.setenv("T1D_DB_PATH", mock_db)
    response = client.get("/api/table/users")
    assert response.status_code == 200
    data = response.json()
    assert "rows" in data
    assert len(data["rows"]) == 2
    assert data["rows"][0]["id"] == 1
    assert data["rows"][0]["name"] == "Alice"

    # Test sorting desc
    resp_desc = client.get("/api/table/users?sort_by=name&order=desc")
    assert resp_desc.status_code == 200
    assert resp_desc.json()["rows"][0]["name"] == "Bob"

    # Test invalid sort order
    resp_bad_order = client.get("/api/table/users?order=invalid")
    assert resp_bad_order.status_code == 400
    assert resp_bad_order.json()["detail"]["error_code"] == "backend.invalidSortOrder"

    # Test invalid sort column
    resp_bad_col = client.get("/api/table/users?sort_by=nonexistent")
    assert resp_bad_col.status_code == 400
    assert resp_bad_col.json()["detail"]["error_code"] == "backend.invalidSortColumn"

    # Test search filtering match
    resp_search = client.get("/api/table/users?search=alice")
    assert resp_search.status_code == 200
    assert len(resp_search.json()["rows"]) == 1
    assert resp_search.json()["rows"][0]["name"] == "Alice"
    assert resp_search.json()["total_count"] == 1

    # Test search filtering no match
    resp_no_match = client.get("/api/table/users?search=nonexistent")
    assert resp_no_match.status_code == 200
    assert len(resp_no_match.json()["rows"]) == 0
    assert resp_no_match.json()["total_count"] == 0


def test_get_table_data_invalid_table(mock_db: str, monkeypatch: typing.Any) -> None:
    """Test fetching data from a table that doesn't exist."""
    monkeypatch.setenv("T1D_DB_PATH", mock_db)
    response = client.get("/api/table/non_existent_table")
    assert response.status_code == 404


def test_get_table_data_sql_injection(mock_db: str, monkeypatch: typing.Any) -> None:
    """Test protection against basic SQL injection in table name."""
    monkeypatch.setenv("T1D_DB_PATH", mock_db)
    response = client.get("/api/table/invalid table name;")
    assert response.status_code == 400


def test_get_table_data_exception(mock_db: str, monkeypatch: typing.Any) -> None:
    """Test server error handling when DB operations fail."""
    monkeypatch.setenv("T1D_DB_PATH", mock_db)
    from unittest.mock import patch

    with patch("duckdb.connect", side_effect=Exception("DB Failure")):
        response = client.get("/api/table/users")
        assert response.status_code == 500
        assert response.json()["detail"]["error_code"] == "backend.serverError"
        assert "DB Failure" in response.json()["detail"]["params"]["error"]


def test_execute_sql_endpoint_success(mock_db: str) -> None:
    """Test execute_sql_endpoint returns successfully."""
    response = client.post(
        "/api/execute-sql",
        json={"query": "SELECT * FROM users", "db_path": mock_db},
    )
    assert response.status_code == 200
    data = response.json()
    assert "sqlResult" in data
    assert len(data["sqlResult"]) == 2
    assert data["sqlResult"][0]["name"] == "Alice"


def test_execute_sql_endpoint_empty(mock_db: str) -> None:
    """Test execute_sql_endpoint empty query."""
    response = client.post(
        "/api/execute-sql",
        json={"query": "   ", "db_path": mock_db},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["error"]["error_code"] == "backend.emptyMessage"


def test_execute_sql_endpoint_value_error(mock_db: str) -> None:
    """Test execute_sql_endpoint catching ValueError."""
    response = client.post(
        "/api/execute-sql",
        json={"query": "BAD SQL", "db_path": mock_db},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["error"]["error_code"] == "backend.sqlExecution"


@patch("t1d_analytics.api.execute_sql")
def test_execute_sql_endpoint_exception(mock_execute: MagicMock, mock_db: str) -> None:
    """Test execute_sql_endpoint catching general Exception."""
    mock_execute.side_effect = Exception("System crash")
    response = client.post(
        "/api/execute-sql",
        json={"query": "SELECT * FROM users", "db_path": mock_db},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["error"]["error_code"] == "backend.errorUnexpected"
    assert data["error"]["params"]["error"] == "System crash"


def test_generate_sql_fallback_no_prefix(mocker: typing.Any) -> None:
    """Test SQL fallback when prefix is missing."""
    from t1d_analytics.api import generate_sql_from_nl

    mock_llm_instance = mocker.MagicMock()
    mock_choice = mocker.MagicMock()
    mock_choice.message.content = "SELECT * FROM test;"
    mock_llm_instance.completion.return_value.choices = [mock_choice]

    mock_llm_class = mocker.MagicMock()
    mock_llm_class.create.return_value = mock_llm_instance

    mocker.patch.dict(
        "sys.modules", {"any_llm": mocker.MagicMock(AnyLLM=mock_llm_class)}
    )

    mocker.patch("t1d_analytics.api.duckdb.connect")
    mocker.patch("t1d_analytics.api.get_database_schema", return_value="dummy_schema")
    full, sql = generate_sql_from_nl("dummy_db", "test query")
    assert sql == "SELECT * FROM test;"


def test_global_exception_handler() -> None:
    """Test that unhandled exceptions are caught by the global exception handler."""
    from fastapi import APIRouter
    from fastapi.testclient import TestClient

    from t1d_analytics.api import app

    router = APIRouter()

    @router.get("/error_endpoint_for_test")
    def error_endpoint() -> None:
        """Endpoint designed to raise an unhandled exception."""
        raise Exception("Trigger unhandled exception")

    app.include_router(router)

    test_client = TestClient(app, raise_server_exceptions=False)
    response = test_client.get(
        "/error_endpoint_for_test", headers={"Accept-Language": "ja-JP,ja;q=0.9"}
    )
    assert response.status_code == 500
    data = response.json()
    assert "detail" in data
    assert data["detail"]["error_code"] == "backend.serverError"
    assert "Trigger unhandled exception" in data["detail"]["params"]["error"]


@pytest.mark.anyio
async def test_lifespan() -> None:
    """Test lifespan startup diagnostics execution."""
    from t1d_analytics.api import lifespan

    with patch("t1d_analytics.api.log_startup_diagnostics") as mock_diag:
        async with lifespan(app):
            pass
        mock_diag.assert_called_once()


def test_execute_sql_db_not_found() -> None:
    """Test execute_sql when database file is missing."""
    with pytest.raises(ValueError, match=r"backend\.dbNotFound.*"):
        execute_sql("missing_file.duckdb", "SELECT 1")


@patch("any_llm.AnyLLM.create")
def test_generate_sql_from_nl_db_not_found(mock_create: MagicMock) -> None:
    """Test generate_sql_from_nl when database file is missing."""
    with pytest.raises(ValueError, match=r"backend\.dbNotFound.*"):
        generate_sql_from_nl("missing_file.duckdb", "SELECT 1")


def test_get_status_endpoint() -> None:
    """Test /api/status and /api/health endpoints."""
    response = client.get("/api/status", headers={"Accept-Language": "ja"})
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "database" in data
    assert "ollama" in data

    resp_health = client.get("/api/health")
    assert resp_health.status_code == 200


def test_execute_sql_max_rows(mock_db: str) -> None:
    """Test execute_sql respects max_rows constraint."""
    res = execute_sql(mock_db, "SELECT * FROM users", max_rows=1)
    assert len(res) == 1
    assert res[0]["name"] == "Alice"


def test_get_table_data_pagination_metadata(mock_db: str) -> None:
    """Test get_table_data returns total_count, page, and total_pages."""
    response = client.get(f"/api/table/users?limit=1&offset=0&db_path={mock_db}")
    assert response.status_code == 200
    data = response.json()
    assert data["total_count"] == 2
    assert data["page"] == 1
    assert data["total_pages"] == 2
    assert len(data["rows"]) == 1


def test_get_schema_with_db_path(mock_db: str) -> None:
    """Test get_schema respects db_path query parameter."""
    response = client.get(f"/api/schema?db_path={mock_db}")
    assert response.status_code == 200
    data = response.json()
    assert len(data["tables"]) >= 1
    table_names = [t["name"] for t in data["tables"]]
    assert "users" in table_names


@patch("urllib.request.urlopen")
def test_list_models_with_ollama_host(mock_urlopen: MagicMock) -> None:
    """Test list_models respects OLLAMA_HOST env variable."""
    mock_response = MagicMock()
    mock_response.read.return_value = json.dumps(
        {"models": [{"name": "gemma4-custom", "size": 12345}]}
    ).encode("utf-8")
    mock_urlopen.return_value.__enter__.return_value = mock_response

    with patch.dict(os.environ, {"OLLAMA_HOST": "remote-ollama:11434"}):
        response = client.get("/api/models")
        assert response.status_code == 200
        data = response.json()
        assert any(m["name"] == "gemma4-custom" for m in data["models"])


def test_sanitize_sql_value() -> None:
    """Test sanitize_sql_value with diverse Python and DuckDB data types."""
    import datetime
    import decimal
    import uuid

    from t1d_analytics.api import sanitize_sql_value

    assert sanitize_sql_value(None) is None
    assert sanitize_sql_value(datetime.date(2023, 1, 15)) == "2023-01-15"
    assert (
        sanitize_sql_value(datetime.datetime(2023, 1, 15, 12, 30, 45))
        == "2023-01-15T12:30:45"
    )
    assert sanitize_sql_value(datetime.time(8, 30)) == "08:30:00"
    assert sanitize_sql_value(decimal.Decimal("123.45")) == 123.45
    u = uuid.uuid4()
    assert sanitize_sql_value(u) == str(u)
    assert sanitize_sql_value(float("nan")) is None
    assert sanitize_sql_value(float("inf")) is None
    assert sanitize_sql_value(float("-inf")) is None
    assert sanitize_sql_value(12.34) == 12.34
    assert sanitize_sql_value(42) == 42
    assert sanitize_sql_value(True) is True
    assert sanitize_sql_value("test string") == "test string"
    assert sanitize_sql_value([1, 2, 3]) == "[1, 2, 3]"


def test_execute_sql_complex_types(tmp_path: typing.Any) -> None:
    """Test execute_sql and TableDataResponse with dates, timestamps, decimals, and nan."""
    db_path = str(tmp_path / "complex.duckdb")
    conn = duckdb.connect(db_path)
    conn.execute(
        """
        CREATE TABLE clinical (
            id INTEGER,
            reading_date DATE,
            reading_ts TIMESTAMP,
            val DECIMAL(5, 2),
            nan_val DOUBLE
        )
        """
    )
    conn.execute(
        """
        INSERT INTO clinical VALUES (
            1,
            '2023-05-10'::DATE,
            '2023-05-10 14:20:00'::TIMESTAMP,
            120.50,
            'nan'::DOUBLE
        )
        """
    )
    conn.close()

    res = execute_sql(db_path, "SELECT * FROM clinical")
    assert len(res) == 1
    row = res[0]
    assert row["id"] == 1
    assert row["reading_date"] == "2023-05-10"
    assert row["reading_ts"] == "2023-05-10T14:20:00"
    assert row["val"] == 120.5
    assert row["nan_val"] is None

    # Test via API endpoints
    table_resp = client.get(f"/api/table/clinical?db_path={db_path}")
    assert table_resp.status_code == 200
    t_data = table_resp.json()
    assert t_data["total_count"] == 1
    assert t_data["rows"][0]["reading_date"] == "2023-05-10"

    chat_resp = client.post(
        "/api/chat",
        json={
            "message": "SELECT * FROM clinical",
            "model": "sql",
            "db_path": db_path,
        },
    )
    assert chat_resp.status_code == 200
    c_data = chat_resp.json()
    assert c_data["sqlResult"][0]["val"] == 120.5


def test_list_databases_endpoint(tmp_path: typing.Any, monkeypatch: typing.Any) -> None:
    """Test /api/databases endpoint returns database list and current DB."""
    db1 = tmp_path / "custom.duckdb"
    conn = duckdb.connect(str(db1))
    conn.execute("CREATE TABLE t (id INT)")
    conn.close()

    monkeypatch.chdir(tmp_path)
    # First call when data/ does not exist (covers s_dir.exists() == False)
    resp1 = client.get("/api/databases")
    assert resp1.status_code == 200

    # Second call when data/ exists with duplicate symlink (covers s_dir.exists() == True and duplicate check)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    try:
        (data_dir / "custom.duckdb").symlink_to(db1)
    except OSError:
        pass

    resp2 = client.get("/api/databases")
    assert resp2.status_code == 200
    data = resp2.json()
    assert "databases" in data
    assert "current_db" in data
    assert any(d["name"] == "custom.duckdb" for d in data["databases"])


def test_execute_sql_timeout(mock_db: str) -> None:
    """Test execute_sql timeout triggers ValueError with queryTimeout error code."""
    with patch("duckdb.connect") as mock_connect:
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn
        mock_conn.execute.side_effect = [
            None,
            duckdb.InterruptException("Query interrupted"),
        ]
        with pytest.raises(ValueError, match=r"backend\.queryTimeout"):
            execute_sql(mock_db, "SELECT 1", timeout_seconds=0.001)


def test_chat_stream_empty_message() -> None:
    """Test /api/chat/stream returns error event on empty message."""
    resp = client.post("/api/chat/stream", json={"message": "", "model": "sql"})
    assert resp.status_code == 200
    assert "backend.emptyMessage" in resp.text


def test_chat_stream_literal_sql_success(mock_db: str) -> None:
    """Test /api/chat/stream executes literal SQL and streams results."""
    resp = client.post(
        "/api/chat/stream",
        json={
            "message": "SELECT * FROM users",
            "model": "sql",
            "db_path": mock_db,
        },
    )
    assert resp.status_code == 200
    text = resp.text
    assert "backend.literalSql" in text
    assert "Alice" in text


def test_chat_stream_literal_sql_error(mock_db: str) -> None:
    """Test /api/chat/stream returns error event on invalid SQL."""
    resp = client.post(
        "/api/chat/stream",
        json={
            "message": "SELECT * FROM nonexistent",
            "model": "sql",
            "db_path": mock_db,
        },
    )
    assert resp.status_code == 200
    assert "backend.errorDbExecution" in resp.text


@patch("t1d_analytics.api.stream_llm_tokens")
def test_chat_stream_nl_success(mock_stream: MagicMock, mock_db: str) -> None:
    """Test /api/chat/stream streams NL tokens and executes generated SQL."""
    mock_stream.return_value = iter(
        ["Here is the query: ", "```sql\n", "SELECT * FROM users;\n", "```"]
    )
    resp = client.post(
        "/api/chat/stream",
        json={
            "message": "show all users",
            "model": "gemma4",
            "db_path": mock_db,
        },
    )
    assert resp.status_code == 200
    text = resp.text
    assert "token" in text
    assert "Alice" in text


@patch("t1d_analytics.api.stream_llm_tokens")
def test_chat_stream_nl_no_sql_generated(mock_stream: MagicMock, mock_db: str) -> None:
    """Test /api/chat/stream when NL prompt does not yield a SQL query."""
    mock_stream.return_value = iter(["I cannot answer this with SQL"])
    resp = client.post(
        "/api/chat/stream",
        json={"message": "hello", "model": "gemma4", "db_path": mock_db},
    )
    assert resp.status_code == 200
    assert "I cannot answer this with SQL" in resp.text


@patch("t1d_analytics.api.stream_llm_tokens")
def test_chat_stream_nl_sql_execution_error(
    mock_stream: MagicMock, mock_db: str
) -> None:
    """Test /api/chat/stream handles generated SQL execution failure."""
    mock_stream.return_value = iter(
        ["Here is the query: ", "```sql\n", "SELECT * FROM missing_table;\n", "```"]
    )
    resp = client.post(
        "/api/chat/stream",
        json={"message": "query", "model": "gemma4", "db_path": mock_db},
    )
    assert resp.status_code == 200
    assert "missing_table" in resp.text or "sql_err" in resp.text


@patch("t1d_analytics.api.stream_llm_tokens")
def test_chat_stream_nl_translation_error(mock_stream: MagicMock, mock_db: str) -> None:
    """Test /api/chat/stream handles LLM translation failure."""
    mock_stream.side_effect = RuntimeError("Ollama offline")
    resp = client.post(
        "/api/chat/stream",
        json={"message": "query", "model": "gemma4", "db_path": mock_db},
    )
    assert resp.status_code == 200
    assert "backend.errorNlpTranslation" in resp.text


def test_validate_db_path_null_byte() -> None:
    """Test validate_db_path rejects null byte injection."""
    from fastapi import HTTPException

    from t1d_analytics.api import validate_db_path

    with pytest.raises(HTTPException) as exc_info:
        validate_db_path("test\0.duckdb")
    assert exc_info.value.status_code == 400
    assert isinstance(exc_info.value.detail, dict)
    assert exc_info.value.detail["error_code"] == "backend.invalidDatabasePath"


def test_validate_db_path_traversal() -> None:
    """Test validate_db_path rejects paths outside allowed directory containment."""
    from pathlib import Path

    from fastapi import HTTPException

    from t1d_analytics.api import validate_db_path

    allowed = [Path("/mock/allowed/dir")]
    with pytest.raises(HTTPException) as exc_info:
        validate_db_path("/etc/passwd", allowed_dirs=allowed)
    assert exc_info.value.status_code == 400
    assert isinstance(exc_info.value.detail, dict)
    assert exc_info.value.detail["error_code"] == "backend.invalidDatabasePath"


def test_validate_db_path_env_var(monkeypatch: typing.Any, tmp_path: Path) -> None:
    """Test validate_db_path respects T1D_DB_PATH environment variable."""
    from t1d_analytics.api import validate_db_path

    env_db = tmp_path / "env_test.duckdb"
    monkeypatch.setenv("T1D_DB_PATH", str(env_db))
    resolved = validate_db_path()
    assert resolved == str(env_db.resolve())


def test_chat_stream_invalid_db_path() -> None:
    """Test /api/chat/stream handles invalid db_path."""
    from unittest.mock import patch

    with patch("t1d_analytics.api.validate_db_path") as mock_val:
        from fastapi import HTTPException

        mock_val.side_effect = HTTPException(status_code=400)
        resp = client.post(
            "/api/chat/stream",
            json={"message": "hello", "db_path": "../../evil.duckdb"},
        )
        assert resp.status_code == 200
        assert "backend.invalidDatabasePath" in resp.text


def test_get_table_data_negative_pagination(mock_db: str) -> None:
    """Test get_table_data rejects negative or zero limit and negative offset."""
    # Negative limit
    resp_neg_lim = client.get("/api/table/users?limit=-1&db_path=" + mock_db)
    assert resp_neg_lim.status_code == 400
    assert resp_neg_lim.json()["detail"]["error_code"] == "backend.invalidPagination"

    # Zero limit
    resp_zero_lim = client.get("/api/table/users?limit=0&db_path=" + mock_db)
    assert resp_zero_lim.status_code == 400
    assert resp_zero_lim.json()["detail"]["error_code"] == "backend.invalidPagination"

    # Negative offset
    resp_neg_off = client.get("/api/table/users?offset=-5&db_path=" + mock_db)
    assert resp_neg_off.status_code == 400
    assert resp_neg_off.json()["detail"]["error_code"] == "backend.invalidPagination"


def test_execute_sql_endpoint_invalid_db() -> None:
    """Test /api/execute-sql returns 400 for out-of-workspace db_path."""
    from unittest.mock import patch

    from fastapi import HTTPException

    with patch("t1d_analytics.api.validate_db_path") as mock_val:
        mock_val.side_effect = HTTPException(
            status_code=400,
            detail={"error_code": "backend.invalidDatabasePath", "params": {}},
        )
        resp = client.post(
            "/api/execute-sql",
            json={"query": "SELECT 1", "db_path": "/etc/shadow"},
        )
        assert resp.status_code == 400
        assert resp.json()["detail"]["error_code"] == "backend.invalidDatabasePath"


@patch("any_llm.AnyLLM.create")
def test_generate_sql_from_nl_multi_block(mock_create: MagicMock, mock_db: str) -> None:
    """Test generate_sql_from_nl extracts the final SQL query block when multiple markdown blocks exist."""
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = """
    Here is some preliminary exploration:
    ```sql
    DESCRIBE users;
    ```
    And here is the final query answering your question:
    ```sql
    SELECT id, name FROM users WHERE id > 0;
    ```
    """
    mock_llm.completion.return_value = mock_response
    mock_create.return_value = mock_llm

    full_resp, sql = generate_sql_from_nl(mock_db, "get users")
    assert sql == "SELECT id, name FROM users WHERE id > 0;"


@patch("any_llm.AnyLLM.create")
def test_generate_sql_from_nl_non_query_block(
    mock_create: MagicMock, mock_db: str
) -> None:
    """Test generate_sql_from_nl falls back to last block if no keyword matched."""
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = """
    ```text
    random text block
    ```
    """
    mock_llm.completion.return_value = mock_response
    mock_create.return_value = mock_llm

    full_resp, sql = generate_sql_from_nl(mock_db, "get users")
    assert sql == "random text block"


@patch("any_llm.AnyLLM.create")
def test_stream_llm_tokens_branches(mock_create: MagicMock) -> None:
    """Test stream_llm_tokens across stream generator, message content, string chunk, and fallback."""
    from t1d_analytics.api import stream_llm_tokens

    mock_llm = MagicMock()
    mock_create.return_value = mock_llm

    # 1. stream with delta.content
    chunk1 = MagicMock()
    chunk1.choices = [MagicMock()]
    chunk1.choices[0].delta.content = "token1 "
    chunk1.choices[0].message = None
    mock_llm.completion.return_value = [chunk1]

    tokens = list(stream_llm_tokens("prompt", "gemma4"))
    assert tokens == ["token1 "]

    # 2. stream with message.content
    chunk2 = MagicMock()
    chunk2.choices = [MagicMock()]
    del chunk2.choices[0].delta
    chunk2.choices[0].message.content = "token2 "
    mock_llm.completion.return_value = [chunk2]

    tokens2 = list(stream_llm_tokens("prompt", "gemma4"))
    assert tokens2 == ["token2 "]

    # 3. stream with string chunks
    mock_llm.completion.return_value = ["chunk_a", "chunk_b", ""]
    tokens3 = list(stream_llm_tokens("prompt", "gemma4"))
    assert tokens3 == ["chunk_a", "chunk_b"]

    # 3b. stream with empty delta and empty message choices
    empty_chunk = MagicMock()
    empty_chunk.choices = [MagicMock()]
    empty_chunk.choices[0].delta.content = ""
    empty_chunk.choices[0].message.content = ""
    mock_llm.completion.return_value = [empty_chunk, object()]
    assert list(stream_llm_tokens("prompt", "gemma4")) == []

    # 3c. stream_resp without __iter__ falls through
    mock_resp_obj = MagicMock()
    mock_resp_obj.choices = [MagicMock()]
    mock_resp_obj.choices[0].message.content = "non_iter fallback"
    mock_llm.completion.side_effect = [object(), mock_resp_obj]
    assert list(stream_llm_tokens("prompt", "gemma4")) == ["non_iter fallback"]

    # 4. stream=True fails, falling back to sync completion
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock()]
    mock_resp.choices[0].message.content = "fallback words stream"

    def fail_streaming(**kwargs: typing.Any) -> typing.Any:
        if kwargs.get("stream"):
            raise RuntimeError("Streaming not supported")
        return mock_resp

    mock_llm.completion.side_effect = fail_streaming
    tokens4 = list(stream_llm_tokens("prompt", "gemma4"))
    assert "".join(tokens4) == "fallback words stream"


@patch("t1d_analytics.api.stream_llm_tokens")
def test_chat_stream_true_tokens_and_events(
    mock_stream: MagicMock, mock_db: str
) -> None:
    """Test /api/chat/stream streams actual tokens and yields result payload."""
    mock_stream.return_value = iter(
        [
            "```sql\n",
            "SELECT * ",
            "FROM users",
            "\n```",
        ]
    )
    resp = client.post(
        "/api/chat/stream",
        json={"message": "get all users", "model": "gemma4", "db_path": mock_db},
    )
    assert resp.status_code == 200
    text = resp.text
    assert "token" in text
    assert "result" in text
    assert "Alice" in text


def test_sessions_crud_endpoints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test conversation history session persistence CRUD endpoints."""
    from t1d_analytics.api import _get_sessions_db_conn

    test_db = str(tmp_path / "sessions_test.duckdb")
    monkeypatch.setattr(
        "t1d_analytics.api._get_sessions_db_conn",
        lambda: _get_sessions_db_conn(test_db),
    )

    # 1. Create a session with auto-generated title
    create_payload = {
        "messages": [
            {"role": "user", "content": "What is HbA1c?"},
            {"role": "assistant", "content": "HbA1c is glycated hemoglobin."},
        ]
    }
    resp1 = client.post("/api/sessions", json=create_payload)
    assert resp1.status_code == 200
    data1 = resp1.json()
    sess_id = data1["session_id"]
    assert sess_id.startswith("session-")
    assert data1["title"] == "What is HbA1c?"
    assert len(data1["messages"]) == 2

    # 2. Update session with custom title
    update_payload = {
        "session_id": sess_id,
        "title": "Custom Session Title",
        "messages": [
            {"role": "user", "content": "Updated content"},
        ],
    }
    resp_up = client.post("/api/sessions", json=update_payload)
    assert resp_up.status_code == 200
    assert resp_up.json()["title"] == "Custom Session Title"

    # 3. Create session with empty messages (default title)
    resp_empty = client.post("/api/sessions", json={"messages": []})
    assert resp_empty.status_code == 200
    assert resp_empty.json()["title"] == "New Chat"

    # 4. List sessions
    resp_list = client.get("/api/sessions")
    assert resp_list.status_code == 200
    sessions = resp_list.json()["sessions"]
    assert len(sessions) == 2
    assert any(s["session_id"] == sess_id for s in sessions)

    # 5. Get session by ID
    resp_get = client.get(f"/api/sessions/{sess_id}")
    assert resp_get.status_code == 200
    assert resp_get.json()["session_id"] == sess_id

    # 6. Get non-existent session
    resp_404 = client.get("/api/sessions/nonexistent_id")
    assert resp_404.status_code == 404

    # 7. Delete session
    resp_del = client.delete(f"/api/sessions/{sess_id}")
    assert resp_del.status_code == 200
    assert resp_del.json()["status"] == "deleted"

    # Verify deleted
    resp_check = client.get("/api/sessions")
    assert not any(s["session_id"] == sess_id for s in resp_check.json()["sessions"])


def test_sessions_malformed_json_and_default_conn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test sessions endpoints with corrupted messages_json and default db_path."""
    from t1d_analytics.api import _get_sessions_db_conn

    # 1. Default _get_sessions_db_conn() path
    monkeypatch.chdir(tmp_path)
    conn = _get_sessions_db_conn()
    conn.execute(
        "INSERT INTO chat_sessions VALUES ('bad-json-id', 'Bad JSON', now(), now(), 'not-json{')"
    )
    conn.close()

    # list_sessions handles malformed JSON
    resp_list = client.get("/api/sessions")
    assert resp_list.status_code == 200
    item = next(
        s for s in resp_list.json()["sessions"] if s["session_id"] == "bad-json-id"
    )
    assert item["messages"] == []

    # get_session handles malformed JSON
    resp_get = client.get("/api/sessions/bad-json-id")
    assert resp_get.status_code == 200
    assert resp_get.json()["messages"] == []


def test_concurrency_and_lock_contention(mock_db: str) -> None:
    """TEST-API-01: Execute 50 parallel requests to /api/execute-sql and /api/chat."""
    import concurrent.futures

    def make_sql_request(idx: int) -> int:
        res = client.post(
            "/api/execute-sql",
            json={
                "query": "SELECT id, name FROM users WHERE id = 1",
                "db_path": mock_db,
            },
        )
        return res.status_code

    def make_chat_sql_request(idx: int) -> int:
        res = client.post(
            "/api/chat",
            json={
                "message": "SELECT COUNT(*) FROM users",
                "model": "sql",
                "db_path": mock_db,
            },
        )
        return res.status_code

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = []
        for i in range(25):
            futures.append(executor.submit(make_sql_request, i))
            futures.append(executor.submit(make_chat_sql_request, i))

        results = [f.result() for f in concurrent.futures.as_completed(futures)]

    assert len(results) == 50
    assert all(code == 200 for code in results)


def test_query_timeout_cartesian_product(mock_db: str) -> None:
    """TEST-API-02: Interruption of pathological Cartesian cross-join query within 1.0s."""
    from t1d_analytics.api import execute_sql

    pathological_query = (
        "WITH RECURSIVE t(n) AS (VALUES (1) UNION ALL SELECT n+1 FROM t WHERE n < 100000000) "
        "SELECT a.n, b.n FROM t a CROSS JOIN t b"
    )
    with pytest.raises(ValueError) as exc_info:
        execute_sql(mock_db, pathological_query, timeout_seconds=0.1)

    assert "backend.queryTimeout" in str(exc_info.value)


def test_sse_streaming_httpx_integration(mock_db: str) -> None:
    """TEST-API-03: Server-Sent Events (SSE) Streaming Integration via HTTP streaming client."""
    with client.stream(
        "POST",
        "/api/chat/stream",
        json={"message": "SELECT * FROM users", "model": "sql", "db_path": mock_db},
    ) as response:
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]
        events = []
        for line in response.iter_lines():
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))

    assert len(events) >= 2
    assert events[0]["event"] == "token"
    assert events[-1]["event"] == "result"
    assert events[-1]["sqlResult"][0]["name"] == "Alice"


def test_get_status_with_db_path(mock_db: str) -> None:
    """Test get_status with explicit valid db_path."""
    resp = client.get(f"/api/status?db_path={mock_db}")
    assert resp.status_code == 200
    data = resp.json()
    assert "status" in data
    assert data["database"]["connected"] is True


def test_multi_turn_history_in_prompt_and_chat(
    mock_db: str, mocker: typing.Any
) -> None:
    """Test _build_sql_prompt and chat endpoints with multi-turn conversation history."""
    from t1d_analytics.api import ChatMessageModel, _build_sql_prompt

    history = [
        ChatMessageModel(
            role="user",
            content="Show me all patients",
            sqlQuery="SELECT * FROM patients",
        ),
        ChatMessageModel(
            role="assistant",
            content="Here are the patients.",
            sqlQuery="SELECT * FROM patients",
        ),
        ChatMessageModel(role="user", content="Now filter by age < 18"),
    ]

    prompt = _build_sql_prompt("schema text", "Show only females", history=history)
    assert "Prior Conversation History:" in prompt
    assert "Show me all patients (SQL: SELECT * FROM patients)" in prompt
    assert "Now filter by age < 18" in prompt

    # Empty history branch
    prompt_empty = _build_sql_prompt("schema text", "Show patients", history=[])
    assert "Prior Conversation History:" not in prompt_empty

    # Chat endpoint with history
    mock_llm_res = ("Explanation", "SELECT * FROM users WHERE age < 18")
    mocker.patch("t1d_analytics.api.generate_sql_from_nl", return_value=mock_llm_res)
    resp = client.post(
        "/api/chat",
        json={
            "message": "Now filter by age < 18",
            "model": "gemma4",
            "db_path": mock_db,
            "history": [m.model_dump() for m in history],
        },
    )
    assert resp.status_code == 200
    assert resp.json()["sqlQuery"] == "SELECT * FROM users WHERE age < 18"

    # Streaming endpoint with history
    with client.stream(
        "POST",
        "/api/chat/stream",
        json={
            "message": "Now filter by age < 18",
            "model": "gemma4",
            "db_path": mock_db,
            "history": [m.model_dump() for m in history],
        },
    ) as stream_resp:
        assert stream_resp.status_code == 200


def test_sessions_concurrent_access_and_locking() -> None:
    """Test concurrent multi-threaded requests on sessions endpoints."""
    import concurrent.futures

    def do_session_op(idx: int) -> int:
        if idx % 3 == 0:
            res = client.post(
                "/api/sessions",
                json={
                    "session_id": f"conc-session-{idx}",
                    "title": f"Chat {idx}",
                    "messages": [{"role": "user", "content": f"msg {idx}"}],
                },
            )
            return res.status_code
        elif idx % 3 == 1:
            res = client.get("/api/sessions")
            return res.status_code
        else:
            # Delete if exists, otherwise save
            client.post(
                "/api/sessions",
                json={
                    "session_id": f"conc-session-del-{idx}",
                    "title": f"Chat {idx}",
                },
            )
            res = client.delete(f"/api/sessions/conc-session-del-{idx}")
            return res.status_code

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(do_session_op, i) for i in range(25)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]

    assert len(results) == 25
    assert all(code == 200 for code in results)


def test_delete_session_not_found() -> None:
    """Test delete_session returns 404 for non-existent session ID."""
    resp = client.delete("/api/sessions/non-existent-session-xyz")
    assert resp.status_code == 404
    assert resp.json()["detail"] == "Session not found"


def test_get_table_profile_endpoint(mock_db: str, mocker: typing.Any) -> None:
    """Test /api/table/{table_name}/profile endpoint success and error conditions."""
    # 1. Success
    resp = client.get(f"/api/table/users/profile?db_path={mock_db}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["table_name"] == "users"
    assert data["total_rows"] == 2

    # 2. Missing DB file
    resp_missing_db = client.get("/api/table/users/profile?db_path=missing_file.duckdb")
    assert resp_missing_db.status_code == 404

    # 3. Invalid table identifier
    resp_bad_id = client.get(f"/api/table/users;drop/profile?db_path={mock_db}")
    assert resp_bad_id.status_code == 400

    # 4. Table not found in DB
    resp_no_tbl = client.get(f"/api/table/missing_tbl/profile?db_path={mock_db}")
    assert resp_no_tbl.status_code == 404

    # 5. FileNotFoundError from profile_table
    mocker.patch(
        "t1d_analytics.api.profile_table",
        side_effect=FileNotFoundError("not found"),
    )
    resp_fnf = client.get(f"/api/table/users/profile?db_path={mock_db}")
    assert resp_fnf.status_code == 404

    # 6. Generic 500 error
    mocker.patch(
        "t1d_analytics.api.profile_table",
        side_effect=Exception("Database crash"),
    )
    resp_err = client.get(f"/api/table/users/profile?db_path={mock_db}")
    assert resp_err.status_code == 500


def test_export_excel_endpoint(mock_db: str, mocker: typing.Any) -> None:
    """Test /api/export/excel endpoint streaming and error branches."""
    import openpyxl

    # 1. Success
    resp = client.get(f"/api/export/excel?table_name=users&db_path={mock_db}")
    assert resp.status_code == 200
    assert (
        resp.headers["content-type"]
        == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    assert "users_export.xlsx" in resp.headers["content-disposition"]
    wb = openpyxl.load_workbook(io.BytesIO(resp.content))
    ws = wb.active
    assert ws is not None
    rows = list(ws.iter_rows(values_only=True))
    assert rows[0] == ("id", "name")
    assert any("Alice" in str(r) for r in rows)

    # 1b. Test numeric preservation and formula injection escaping
    conn = duckdb.connect(mock_db)
    conn.execute(
        "CREATE TABLE formulas (id INT, val_neg DOUBLE, val_pos VARCHAR, formula1 VARCHAR, formula2 VARCHAR, null_col VARCHAR)"
    )
    conn.execute(
        "INSERT INTO formulas VALUES (1, -5.2, '+10', '=cmd|'' /C calc''!A0', '-cmd', NULL)"
    )
    conn.close()

    resp_f = client.get(f"/api/export/excel?table_name=formulas&db_path={mock_db}")
    assert resp_f.status_code == 200
    wb_f = openpyxl.load_workbook(io.BytesIO(resp_f.content))
    ws_f = wb_f.active
    assert ws_f is not None
    rows_f = list(ws_f.iter_rows(values_only=True))
    data_row = rows_f[1]
    assert data_row[1] == -5.2
    assert data_row[2] == "+10"
    assert data_row[3] == "'=cmd|' /C calc'!A0"
    assert data_row[4] == "'-cmd"
    assert data_row[5] is None

    # 2. Missing DB
    resp_no_db = client.get("/api/export/excel?table_name=users&db_path=no_db.duckdb")
    assert resp_no_db.status_code == 404

    # 3. Invalid table name
    resp_inv = client.get(f"/api/export/excel?table_name=users;drop&db_path={mock_db}")
    assert resp_inv.status_code == 400

    # 4. Table not found
    resp_no_tbl = client.get(f"/api/export/excel?table_name=missing&db_path={mock_db}")
    assert resp_no_tbl.status_code == 404

    # 5. Server error
    mocker.patch(
        "duckdb.connect",
        side_effect=Exception("IO Failure"),
    )
    resp_500 = client.get(f"/api/export/excel?table_name=users&db_path={mock_db}")
    assert resp_500.status_code == 500

    # 5b. Server error during query execution when conn is open
    mock_conn_exc = mocker.MagicMock()
    mock_conn_exc.execute.side_effect = [
        mocker.MagicMock(fetchall=lambda: [("users",)]),
        Exception("Execution failure"),
    ]
    mocker.patch("duckdb.connect", return_value=mock_conn_exc)
    resp_500b = client.get(f"/api/export/excel?table_name=users&db_path={mock_db}")
    assert resp_500b.status_code == 500


def test_sessions_db_env_var(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test T1D_SESSIONS_DB environment variable controls session database path."""
    from t1d_analytics.api import _get_sessions_db_conn

    env_db = str(tmp_path / "env_sessions.duckdb")
    monkeypatch.setenv("T1D_SESSIONS_DB", env_db)
    conn = _get_sessions_db_conn()
    conn.close()
    assert Path(env_db).exists()


def test_sessions_db_lock_retry(tmp_path: Path, mocker: typing.Any) -> None:
    """Test _get_sessions_db_conn retries upon IOException lock contention."""
    from t1d_analytics.api import _get_sessions_db_conn

    real_conn = duckdb.connect(str(tmp_path / "retry.duckdb"))
    attempts: list[int] = []

    def mock_connect(
        *args: typing.Any, **kwargs: typing.Any
    ) -> duckdb.DuckDBPyConnection:
        if len(attempts) < 2:
            attempts.append(1)
            raise duckdb.IOException("Could not set lock on file")
        return real_conn

    mocker.patch("duckdb.connect", side_effect=mock_connect)
    conn = _get_sessions_db_conn(str(tmp_path / "retry.duckdb"))
    conn.close()
    assert len(attempts) == 2


def test_sessions_db_lock_failure(tmp_path: Path, mocker: typing.Any) -> None:
    """Test _get_sessions_db_conn raises if lock cannot be acquired after retries."""
    from t1d_analytics.api import _get_sessions_db_conn

    mocker.patch(
        "duckdb.connect",
        side_effect=duckdb.IOException("Fatal lock error"),
    )
    with pytest.raises(duckdb.IOException):
        _get_sessions_db_conn(str(tmp_path / "fail.duckdb"))


def test_sessions_db_non_lock_error(tmp_path: Path, mocker: typing.Any) -> None:
    """Test _get_sessions_db_conn immediately raises for non-lock IOException."""
    from t1d_analytics.api import _get_sessions_db_conn

    mocker.patch(
        "duckdb.connect",
        side_effect=duckdb.IOException("Permission denied"),
    )
    with pytest.raises(duckdb.IOException):
        _get_sessions_db_conn(str(tmp_path / "perm.duckdb"))


def test_export_parquet_endpoint(mock_db: str, mocker: typing.Any) -> None:
    """Test /api/export/parquet endpoint for data export in columnar parquet format."""
    # 1. Successful export
    resp = client.get(f"/api/export/parquet?table_name=users&db_path={mock_db}")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/vnd.apache.parquet"
    assert "users_export.parquet" in resp.headers["content-disposition"]
    assert len(resp.content) > 0

    # 2. Missing DB
    resp_no_db = client.get("/api/export/parquet?table_name=users&db_path=no_db.duckdb")
    assert resp_no_db.status_code == 404

    # 3. Invalid table name
    resp_inv = client.get(
        f"/api/export/parquet?table_name=users;drop&db_path={mock_db}"
    )
    assert resp_inv.status_code == 400

    # 4. Table not found
    resp_no_tbl = client.get(
        f"/api/export/parquet?table_name=missing&db_path={mock_db}"
    )
    assert resp_no_tbl.status_code == 404

    # 5. Server error
    mocker.patch(
        "duckdb.connect",
        side_effect=Exception("Parquet export failure"),
    )
    resp_500 = client.get(f"/api/export/parquet?table_name=users&db_path={mock_db}")
    assert resp_500.status_code == 500

    # 5b. Server error during COPY execution when conn is open and temp file exists
    mock_conn_p = mocker.MagicMock()
    mock_conn_p.execute.side_effect = [
        mocker.MagicMock(fetchall=lambda: [("users",)]),
        Exception("Disk error during COPY"),
    ]
    mocker.patch("duckdb.connect", return_value=mock_conn_p)
    resp_500b = client.get(f"/api/export/parquet?table_name=users&db_path={mock_db}")
    assert resp_500b.status_code == 500


def test_export_csv_endpoint(mock_db: str, mocker: typing.Any) -> None:
    """Test /api/export/csv endpoint streaming, formula escaping, and error handling."""
    # 1. Successful export
    resp = client.get(f"/api/export/csv?table_name=users&db_path={mock_db}")
    assert resp.status_code == 200
    assert "text/csv" in resp.headers["content-type"]
    assert "users_export.csv" in resp.headers["content-disposition"]
    text = resp.content.decode("utf-8")
    assert text.startswith("\ufeff")  # BOM
    assert '"id","name"' in text or '"id"' in text

    # 1b. Test with formula injection, nulls, and numeric leading signs
    conn = duckdb.connect(mock_db)
    conn.execute("CREATE TABLE csv_formulas (id INT, val VARCHAR, num VARCHAR)")
    conn.execute(
        "INSERT INTO csv_formulas VALUES (1, '=CMD', NULL), (2, '@SUM', '-99.9'), (3, '+CMD', '-CALC')"
    )
    conn.execute("CREATE TABLE multi_chunk (id INT, txt VARCHAR)")
    conn.execute(
        "INSERT INTO multi_chunk SELECT i, repeat('X', 120) FROM range(800) tbl(i)"
    )
    conn.close()
    resp_f = client.get(f"/api/export/csv?table_name=csv_formulas&db_path={mock_db}")
    assert resp_f.status_code == 200
    text_f = resp_f.content.decode("utf-8")
    assert "'=CMD" in text_f
    assert '""' in text_f  # Null value encoded as empty quoted string
    assert "'+42.5" not in text_f
    assert "'+99.9" not in text_f

    # Multi-chunk export to exhaust while loop chunk iteration
    resp_multi = client.get(f"/api/export/csv?table_name=multi_chunk&db_path={mock_db}")
    assert resp_multi.status_code == 200
    assert len(resp_multi.content) > 70000

    # 2. Missing DB
    resp_no_db = client.get("/api/export/csv?table_name=users&db_path=no_db.duckdb")
    assert resp_no_db.status_code == 404

    # 3. Invalid table name
    resp_inv = client.get(f"/api/export/csv?table_name=users;drop&db_path={mock_db}")
    assert resp_inv.status_code == 400

    # 4. Table not found
    resp_no_tbl = client.get(f"/api/export/csv?table_name=missing&db_path={mock_db}")
    assert resp_no_tbl.status_code == 404

    # 5. Server error on connect
    mocker.patch("duckdb.connect", side_effect=Exception("CSV export failure"))
    resp_500 = client.get(f"/api/export/csv?table_name=users&db_path={mock_db}")
    assert resp_500.status_code == 500

    # 5b. Server error with open conn during SHOW TABLES
    mock_conn_open = mocker.MagicMock()
    mock_conn_open.execute.side_effect = Exception("Show tables error")
    mocker.patch("duckdb.connect", return_value=mock_conn_open)
    resp_500b = client.get(f"/api/export/csv?table_name=users&db_path={mock_db}")
    assert resp_500b.status_code == 500
    mock_conn_open.close.assert_called()

    # 5c. Server error during fetch
    mock_conn = mocker.MagicMock()
    mock_res = mocker.MagicMock()
    mock_res.description = [("id",)]
    mock_res.fetchmany.side_effect = Exception("Read failure")
    mock_conn.execute.side_effect = [
        mocker.MagicMock(fetchall=lambda: [("users",)]),
        mock_res,
    ]
    mocker.patch("duckdb.connect", return_value=mock_conn)
    resp_500c = client.get(f"/api/export/csv?table_name=users&db_path={mock_db}")
    assert resp_500c.status_code == 500


def test_export_json_endpoint(mock_db: str, mocker: typing.Any) -> None:
    """Test /api/export/json endpoint streaming JSON array and error handling."""
    # 1. Successful export
    resp = client.get(f"/api/export/json?table_name=users&db_path={mock_db}")
    assert resp.status_code == 200
    assert "application/json" in resp.headers["content-type"]
    assert "users_export.json" in resp.headers["content-disposition"]
    data = json.loads(resp.content.decode("utf-8"))
    assert isinstance(data, list)
    assert len(data) > 0

    # Multi-chunk export to exhaust while loop chunk iteration
    conn = duckdb.connect(mock_db)
    conn.execute("CREATE TABLE multi_chunk (id INT, txt VARCHAR)")
    conn.execute(
        "INSERT INTO multi_chunk SELECT i, repeat('X', 120) FROM range(800) tbl(i)"
    )
    conn.close()
    resp_multi_j = client.get(
        f"/api/export/json?table_name=multi_chunk&db_path={mock_db}"
    )
    assert resp_multi_j.status_code == 200
    assert len(resp_multi_j.content) > 70000

    # 2. Missing DB
    resp_no_db = client.get("/api/export/json?table_name=users&db_path=no_db.duckdb")
    assert resp_no_db.status_code == 404

    # 3. Invalid table name
    resp_inv = client.get(f"/api/export/json?table_name=users;drop&db_path={mock_db}")
    assert resp_inv.status_code == 400

    # 4. Table not found
    resp_no_tbl = client.get(f"/api/export/json?table_name=missing&db_path={mock_db}")
    assert resp_no_tbl.status_code == 404

    # 5. Server error on connect
    mocker.patch("duckdb.connect", side_effect=Exception("JSON export failure"))
    resp_500 = client.get(f"/api/export/json?table_name=users&db_path={mock_db}")
    assert resp_500.status_code == 500

    # 5b. Server error with open conn during SHOW TABLES
    mock_conn_open = mocker.MagicMock()
    mock_conn_open.execute.side_effect = Exception("JSON show tables error")
    mocker.patch("duckdb.connect", return_value=mock_conn_open)
    resp_500b = client.get(f"/api/export/json?table_name=users&db_path={mock_db}")
    assert resp_500b.status_code == 500
    mock_conn_open.close.assert_called()

    # 5c. Server error during fetch
    mock_conn = mocker.MagicMock()
    mock_res = mocker.MagicMock()
    mock_res.description = [("id",)]
    mock_res.fetchmany.side_effect = Exception("JSON read failure")
    mock_conn.execute.side_effect = [
        mocker.MagicMock(fetchall=lambda: [("users",)]),
        mock_res,
    ]
    mocker.patch("duckdb.connect", return_value=mock_conn)
    resp_500c = client.get(f"/api/export/json?table_name=users&db_path={mock_db}")
    assert resp_500c.status_code == 500


def test_export_excel_with_null_and_formula(tmp_path: typing.Any) -> None:
    """Test /api/export/excel with null values and formula injection prefix."""
    import openpyxl

    db_path = str(tmp_path / "formula.duckdb")
    conn = duckdb.connect(db_path)
    conn.execute("CREATE TABLE items (id INTEGER, formula VARCHAR, note VARCHAR)")
    conn.execute("INSERT INTO items VALUES (1, '=1+1', NULL)")
    conn.close()

    resp = client.get(f"/api/export/excel?table_name=items&db_path={db_path}")
    assert resp.status_code == 200
    assert (
        resp.headers["content-type"]
        == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    wb = openpyxl.load_workbook(io.BytesIO(resp.content))
    ws = wb.active
    assert ws is not None
    rows = list(ws.iter_rows(values_only=True))
    assert rows[1][1] == "'=1+1"
    assert rows[1][2] is None


def test_get_table_data_missing_db_and_db_error(
    mock_db: str, mocker: typing.Any
) -> None:
    """Test get_table_data 404 on missing DB file and connection error."""
    resp_no_db = client.get("/api/table/users?db_path=nonexistent.duckdb")
    assert resp_no_db.status_code == 404

    # DuckDB cannot open file exception
    mocker.patch("duckdb.connect", side_effect=Exception("Cannot open file"))
    resp_open_err = client.get(f"/api/table/users?db_path={mock_db}")
    assert resp_open_err.status_code == 404


def test_api_special_characters_and_quotes(tmp_path: Path) -> None:
    """Test get_schema, get_table_data, and export_excel with quotes and special characters."""
    db = tmp_path / "api_special.duckdb"
    conn = duckdb.connect(str(db))
    conn.execute(
        'CREATE TABLE "users_special" ("id" INT, "Glucose (mg/dL)" FLOAT, "col""quoted""" VARCHAR)'
    )
    conn.execute(
        "INSERT INTO \"users_special\" VALUES (1, 105.0, 'quote \"inside\" text'), (2, 120.0, 'normal')"
    )
    conn.close()

    # 1. get_schema
    resp_schema = client.get(f"/api/schema?db_path={db}")
    assert resp_schema.status_code == 200
    schema_data = resp_schema.json()
    assert any(t["name"] == "users_special" for t in schema_data["tables"])

    # 2. get_table_data
    resp_data = client.get(f"/api/table/users_special?db_path={db}")
    assert resp_data.status_code == 200
    assert len(resp_data.json()["rows"]) == 2

    # 3. export_excel
    import openpyxl

    resp_export = client.get(f"/api/export/excel?table_name=users_special&db_path={db}")
    assert resp_export.status_code == 200
    assert (
        resp_export.headers["content-type"]
        == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    wb_sp = openpyxl.load_workbook(io.BytesIO(resp_export.content))
    ws_sp = wb_sp.active
    assert ws_sp is not None
    rows_sp = list(ws_sp.iter_rows(values_only=True))
    assert rows_sp[0] == ("id", "Glucose (mg/dL)", 'col"quoted"')
    assert rows_sp[1] == (1, 105.0, 'quote "inside" text')


def test_api_auth_middleware(monkeypatch: typing.Any, mock_db: str) -> None:
    """Test API authentication middleware with valid, invalid, and missing API keys."""
    monkeypatch.setenv("T1D_API_KEY", "secret-test-key-123")
    monkeypatch.setenv("T1D_DB_PATH", mock_db)

    # Health route should be exempt
    resp_health = client.get("/api/health")
    assert resp_health.status_code == 200

    # Protected route without key should return 401
    resp_unauth = client.get("/api/schema")
    assert resp_unauth.status_code == 401
    assert resp_unauth.json()["detail"]["error_code"] == "backend.unauthorized"

    # Protected route with wrong key should return 401
    resp_wrong = client.get("/api/schema", headers={"X-API-Key": "wrong-key"})
    assert resp_wrong.status_code == 401

    # Protected route with correct X-API-Key should succeed
    resp_ok_header = client.get(
        "/api/schema", headers={"X-API-Key": "secret-test-key-123"}
    )
    assert resp_ok_header.status_code == 200

    # Protected route with correct Bearer token should succeed
    resp_ok_bearer = client.get(
        "/api/schema", headers={"Authorization": "Bearer secret-test-key-123"}
    )
    assert resp_ok_bearer.status_code == 200


def test_sliding_window_rate_limiting(monkeypatch: typing.Any, mock_db: str) -> None:
    """Test sliding-window rate limiting on execute_sql endpoint."""
    from t1d_analytics.api import RATE_LIMIT_STORE, check_rate_limit

    RATE_LIMIT_STORE.clear()
    ip = "192.168.1.100"

    # Check that first 3 requests are allowed
    assert check_rate_limit(ip, limit=3, window_seconds=60.0) is True
    assert check_rate_limit(ip, limit=3, window_seconds=60.0) is True
    assert check_rate_limit(ip, limit=3, window_seconds=60.0) is True

    # 4th request exceeds limit
    assert check_rate_limit(ip, limit=3, window_seconds=60.0) is False

    # After window passes, should allow again
    RATE_LIMIT_STORE[ip] = [time.time() - 70.0]
    assert check_rate_limit(ip, limit=3, window_seconds=60.0) is True

    # Test endpoint triggers 429 when rate limit is exceeded
    monkeypatch.setenv("T1D_DB_PATH", mock_db)
    monkeypatch.setattr("t1d_analytics.api.check_rate_limit", lambda *a, **kw: False)

    resp_sql = client.post("/api/execute-sql", json={"query": "SELECT 1"})
    assert resp_sql.status_code == 429
    assert resp_sql.json()["detail"]["error_code"] == "backend.rateLimitExceeded"

    resp_chat = client.post("/api/chat", json={"message": "hello", "model": "sql"})
    assert resp_chat.status_code == 429

    resp_stream = client.post(
        "/api/chat/stream", json={"message": "hello", "model": "sql"}
    )
    assert resp_stream.status_code == 429
    RATE_LIMIT_STORE.clear()


def test_rate_limiter_backends_and_retry_after(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test MemoryRateLimiter and DuckDbRateLimiter features, retry-after headers, and reset."""
    from concurrent.futures import ThreadPoolExecutor

    from fastapi import HTTPException

    from t1d_analytics.api import (
        DuckDbRateLimiter,
        MemoryRateLimiter,
        enforce_rate_limit,
        get_rate_limiter,
    )

    # 1. MemoryRateLimiter retry-after and reset
    mem_limiter = MemoryRateLimiter()
    mem_limiter.reset()
    assert mem_limiter.get_retry_after("10.0.0.1", window_seconds=60.0) == 1
    assert mem_limiter.is_allowed("10.0.0.1", limit=1, window_seconds=60.0) is True
    assert mem_limiter.is_allowed("10.0.0.1", limit=1, window_seconds=60.0) is False
    assert mem_limiter.get_retry_after("10.0.0.1", window_seconds=60.0) >= 1
    mem_limiter.reset()
    assert mem_limiter.is_allowed("10.0.0.1", limit=1, window_seconds=60.0) is True

    # 2. DuckDbRateLimiter features
    db_path = tmp_path / "rate_limits.duckdb"
    duck_limiter = DuckDbRateLimiter(str(db_path))
    duck_limiter.reset()
    assert duck_limiter.get_retry_after("10.0.0.2", window_seconds=60.0) == 1
    assert duck_limiter.is_allowed("10.0.0.2", limit=2, window_seconds=60.0) is True
    assert duck_limiter.is_allowed("10.0.0.2", limit=2, window_seconds=60.0) is True
    assert duck_limiter.is_allowed("10.0.0.2", limit=2, window_seconds=60.0) is False
    assert duck_limiter.get_retry_after("10.0.0.2", window_seconds=60.0) >= 1
    duck_limiter.reset()
    assert duck_limiter.is_allowed("10.0.0.2", limit=2, window_seconds=60.0) is True

    # 3. get_rate_limiter backend routing
    monkeypatch.setenv("T1D_RATE_LIMITER_BACKEND", "duckdb")
    limiter_duck = get_rate_limiter()
    assert isinstance(limiter_duck, DuckDbRateLimiter)

    monkeypatch.setenv("T1D_RATE_LIMITER_BACKEND", "memory")
    limiter_mem = get_rate_limiter()
    assert isinstance(limiter_mem, MemoryRateLimiter)

    # 4. enforce_rate_limit Retry-After header
    mock_request = MagicMock()
    mock_request.client.host = "10.0.0.99"
    limiter_mem.reset()
    enforce_rate_limit(mock_request, limit=1, window_seconds=60.0)
    with pytest.raises(HTTPException) as exc_info:
        enforce_rate_limit(mock_request, limit=1, window_seconds=60.0)
    assert exc_info.value.status_code == 429
    assert (
        exc_info.value.headers is not None and "Retry-After" in exc_info.value.headers
    )

    # 5. Concurrent flood testing
    concurrent_ip = "10.0.0.50"
    limiter_mem.reset()

    def send_req() -> bool:
        return limiter_mem.is_allowed(concurrent_ip, limit=10, window_seconds=60.0)

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: send_req(), range(30)))
    assert results.count(True) == 10
    assert results.count(False) == 20


def test_multiprovider_llm_validation(
    monkeypatch: pytest.MonkeyPatch, mock_db: str
) -> None:
    """Test multi-provider parsing, missing API key, and missing SDK error handling."""
    from t1d_analytics.api import generate_sql_from_nl, stream_llm_tokens

    # 1. stream_llm_tokens with provider prefix
    monkeypatch.setenv("OPENAI_API_KEY", "test-stream-key")
    mock_llm = MagicMock()
    mock_llm.completion.return_value = ["token1 ", "token2"]
    with patch("any_llm.AnyLLM.create", return_value=mock_llm):
        tokens = list(stream_llm_tokens("test prompt", "openai/gpt-4o"))
        assert len(tokens) > 0

    # 2. generate_sql_from_nl missing external API key
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError) as exc_info:
        generate_sql_from_nl(mock_db, "Count users", model_name="openai/gpt-4o")
    assert "backend.missingApiKey" in str(exc_info.value)

    # 3. generate_sql_from_nl missing SDK
    with patch.dict("sys.modules", {"any_llm": None}):
        with pytest.raises(RuntimeError) as exc_info:
            generate_sql_from_nl(
                mock_db, "Count users", model_name="anthropic/claude-3"
            )
        assert "backend.missingSdk" in str(exc_info.value)

    # 4. generate_sql_from_nl successful with provider prefix
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    mock_llm2 = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "```sql\nSELECT count(*) FROM users;\n```"
    mock_resp = MagicMock()
    mock_resp.choices = [mock_choice]
    mock_llm2.completion.return_value = mock_resp
    with patch("any_llm.AnyLLM.create", return_value=mock_llm2):
        full, sql = generate_sql_from_nl(
            mock_db, "Count users", model_name="openai/gpt-4o"
        )
        assert "SELECT count(*) FROM users" in sql


def test_get_providers_status_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test /api/providers/status reports health and credential readiness for all providers."""
    # Test with keys configured and Ollama mock success
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test12345")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test12345")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-test-key")

    class MockResp:
        """Mock HTTP response."""

        status = 200

        def __enter__(self) -> "MockResp":
            """Enter context."""
            return self

        def __exit__(self, *args: typing.Any) -> None:
            """Exit context."""
            pass

    with patch("urllib.request.urlopen", return_value=MockResp()):
        resp = client.get("/api/providers/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "providers" in data
        provs = {p["provider"]: p for p in data["providers"]}
        assert provs["ollama"]["healthy"] is True
        assert provs["ollama"]["reachable"] is True
        assert provs["ollama"]["error_code"] == "PROVIDER_ONLINE"
        assert provs["openai"]["configured"] is True
        assert provs["openai"]["error_code"] == "PROVIDER_ONLINE"
        assert provs["anthropic"]["configured"] is True
        assert provs["google"]["configured"] is True

    # Test with keys unconfigured and Ollama error
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    with patch("urllib.request.urlopen", side_effect=Exception("Connection refused")):
        resp = client.get("/api/providers/status")
        assert resp.status_code == 200
        data = resp.json()
        provs = {p["provider"]: p for p in data["providers"]}
        assert provs["ollama"]["healthy"] is False
        assert provs["ollama"]["reachable"] is False
        assert provs["ollama"]["error_code"] == "PROVIDER_OFFLINE"
        assert provs["openai"]["configured"] is False
        assert provs["openai"]["error_code"] == "MISSING_API_KEY"
        assert provs["anthropic"]["configured"] is False
        assert provs["google"]["configured"] is False


def test_extract_stream_delta_various_providers() -> None:
    """Test extract_stream_delta across string, dict, OpenAI, Anthropic, Google, and fallback formats."""
    from t1d_analytics.api import extract_stream_delta

    # 1. String chunk
    assert extract_stream_delta("plain token", "ollama") == "plain token"

    # 2. Dict formats
    assert (
        extract_stream_delta({"choices": [{"delta": {"content": "tok1"}}]}, "openai")
        == "tok1"
    )
    assert extract_stream_delta({"choices": [{"text": "tok2"}]}, "openai") == "tok2"
    assert extract_stream_delta({"message": {"content": "tok3"}}, "ollama") == "tok3"
    assert extract_stream_delta({"response": "tok4"}, "ollama") == "tok4"
    assert extract_stream_delta({"text": "tok5"}, "google") == "tok5"

    # 3. OpenAI / Ollama object formats
    chunk_openai_delta = MagicMock()
    choice_delta = MagicMock()
    choice_delta.delta.content = "openai_delta"
    chunk_openai_delta.choices = [choice_delta]
    assert extract_stream_delta(chunk_openai_delta, "openai") == "openai_delta"

    chunk_openai_msg = MagicMock()
    choice_msg = MagicMock(spec=["message"])
    choice_msg.message.content = "openai_msg"
    chunk_openai_msg.choices = [choice_msg]
    assert extract_stream_delta(chunk_openai_msg, "ollama") == "openai_msg"

    chunk_openai_txt = MagicMock()
    choice_txt = MagicMock(spec=["text"])
    choice_txt.text = "openai_text"
    chunk_openai_txt.choices = [choice_txt]
    assert extract_stream_delta(chunk_openai_txt, "openai") == "openai_text"

    # 4. Anthropic object formats
    chunk_anthropic = MagicMock()
    chunk_anthropic.delta.text = "anthropic_tok"
    assert extract_stream_delta(chunk_anthropic, "anthropic") == "anthropic_tok"

    chunk_anthropic_content = MagicMock(spec=["delta"])
    chunk_anthropic_content.delta = MagicMock(spec=["content"])
    chunk_anthropic_content.delta.content = "anthropic_content"
    assert (
        extract_stream_delta(chunk_anthropic_content, "anthropic")
        == "anthropic_content"
    )

    chunk_anthropic_choice = MagicMock(spec=["choices"])
    choice_ant = MagicMock()
    choice_ant.delta.text = "choice_ant_tok"
    chunk_anthropic_choice.choices = [choice_ant]
    assert extract_stream_delta(chunk_anthropic_choice, "anthropic") == "choice_ant_tok"

    # 5. Google / Gemini object formats
    chunk_google = MagicMock()
    part = MagicMock()
    part.text = "google_tok"
    cand = MagicMock()
    cand.content.parts = [part]
    chunk_google.candidates = [cand]
    assert extract_stream_delta(chunk_google, "google") == "google_tok"

    chunk_google_text = MagicMock(spec=["text"])
    chunk_google_text.text = "google_direct_text"
    assert extract_stream_delta(chunk_google_text, "google") == "google_direct_text"

    # 6. Fallback formats
    chunk_fallback_delta = MagicMock(spec=["choices"])
    c_fall = MagicMock()
    c_fall.delta.content = "fallback_content"
    chunk_fallback_delta.choices = [c_fall]
    assert (
        extract_stream_delta(chunk_fallback_delta, "other_provider")
        == "fallback_content"
    )

    chunk_fallback_text = MagicMock(spec=["text"])
    chunk_fallback_text.text = "fallback_text"
    assert (
        extract_stream_delta(chunk_fallback_text, "other_provider") == "fallback_text"
    )

    # 7. Unmatched chunk
    assert extract_stream_delta(object(), "openai") == ""


def test_stream_llm_tokens_native_and_fallback() -> None:
    """Test stream_llm_tokens yields native stream chunks or falls back to full completion block."""
    from t1d_analytics.api import stream_llm_tokens

    # Native stream success
    mock_llm = MagicMock()
    mock_chunk1 = MagicMock()
    mock_chunk1.choices = [MagicMock(delta=MagicMock(content="word1 "))]
    mock_chunk2 = MagicMock()
    mock_chunk2.choices = [MagicMock(delta=MagicMock(content="word2"))]
    mock_llm.completion.return_value = [mock_chunk1, mock_chunk2]

    with patch("any_llm.AnyLLM.create", return_value=mock_llm):
        tokens = list(stream_llm_tokens("test prompt", "gemma4", provider="ollama"))
        assert tokens == ["word1 ", "word2"]

    # Native stream raises error, single-block fallback
    mock_llm_err = MagicMock()

    def fail_streaming(**kwargs: typing.Any) -> typing.Any:
        if kwargs.get("stream"):
            raise RuntimeError("Streaming not supported")
        resp = MagicMock()
        resp.choices = [
            MagicMock(message=MagicMock(content="Single full response block."))
        ]
        return resp

    mock_llm_err.completion.side_effect = fail_streaming
    with patch("any_llm.AnyLLM.create", return_value=mock_llm_err):
        tokens2 = list(
            stream_llm_tokens("test prompt", "openai/gpt-4o", api_key="sk-test")
        )
        assert tokens2 == ["Single full response block."]


@pytest.mark.anyio
async def test_stream_chat_events_client_disconnect(mock_db: str) -> None:
    """Test stream_chat_events detects client disconnection and aborts streaming promptly."""
    from t1d_analytics.api import ChatRequest, stream_chat_events

    # 1. Disconnected before starting
    req_disc_before = MagicMock()
    req_disc_before.is_disconnected = AsyncMock(return_value=True)

    chat_req = ChatRequest(message="SELECT 1", model="sql", db_path=mock_db)
    events: list[str] = []
    async for evt in stream_chat_events(chat_req, req=req_disc_before):
        events.append(evt)
    assert len(events) == 0

    # 2. Disconnected during token streaming
    req_disc_during = MagicMock()
    req_disc_during.is_disconnected = AsyncMock(side_effect=[False, True])

    chat_req_nl = ChatRequest(message="Count users", model="gemma4", db_path=mock_db)
    with patch(
        "t1d_analytics.api.stream_llm_tokens", return_value=iter(["tok1", "tok2"])
    ):
        events_during: list[str] = []
        async for evt in stream_chat_events(chat_req_nl, req=req_disc_during):
            events_during.append(evt)
        assert len(events_during) == 0


def test_chat_endpoints_with_provider_headers(mock_db: str) -> None:
    """Test /api/chat and /api/chat/stream accept x-provider and x-provider-api-key headers."""
    # /api/chat with headers
    with patch(
        "t1d_analytics.api.generate_sql_from_nl", return_value=("answer", "SELECT 1")
    ):
        resp = client.post(
            "/api/chat",
            json={"message": "Count users", "db_path": mock_db},
            headers={"x-provider": "openai", "x-provider-api-key": "sk-header-test"},
        )
        assert resp.status_code == 200

    # /api/chat/stream with headers
    with patch("t1d_analytics.api.stream_llm_tokens", return_value=iter(["Done"])):
        resp_stream = client.post(
            "/api/chat/stream",
            json={"message": "Count users", "db_path": mock_db},
            headers={"x-provider": "anthropic", "x-provider-api-key": "sk-ant-header"},
        )
        assert resp_stream.status_code == 200
        assert "Done" in resp_stream.text


def test_is_provider_configured_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test is_provider_configured handles known and unknown providers and environment keys."""
    from t1d_analytics.api import is_provider_configured

    assert is_provider_configured("ollama") is True

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert is_provider_configured("openai") is False
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    assert is_provider_configured("openai") is True

    assert is_provider_configured("unknown_provider_xyz") is False


def test_get_providers_status_unprefixed_host_and_non_200(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test get_providers_status with unprefixed OLLAMA_HOST and non-200 HTTP status."""
    monkeypatch.setenv("OLLAMA_HOST", "localhost:11434")

    class MockStatus500:
        """Mock HTTP 500 response."""

        status = 500

        def __enter__(self) -> "MockStatus500":
            """Enter context."""
            return self

        def __exit__(self, *args: typing.Any) -> None:
            """Exit context."""
            pass

    with patch("urllib.request.urlopen", return_value=MockStatus500()):
        resp = client.get("/api/providers/status")
        assert resp.status_code == 200
        data = resp.json()
        provs = {p["provider"]: p for p in data["providers"]}
        assert provs["ollama"]["healthy"] is False


def test_generate_sql_from_nl_with_direct_api_key(
    monkeypatch: pytest.MonkeyPatch, mock_db: str
) -> None:
    """Test generate_sql_from_nl passes client-supplied api_key to AnyLLM without polluting environment."""
    from t1d_analytics.api import generate_sql_from_nl

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    mock_llm = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "```sql\nSELECT 1;\n```"
    mock_llm.completion.return_value = MagicMock(choices=[mock_choice])

    with patch("any_llm.AnyLLM.create", return_value=mock_llm) as mock_create:
        full, sql = generate_sql_from_nl(
            mock_db,
            "Count users",
            model_name="openai/gpt-4o",
            api_key="direct_secret_key_123",
        )
        assert sql == "SELECT 1;"
        mock_create.assert_called_with("openai", api_key="direct_secret_key_123")
        assert os.environ.get("OPENAI_API_KEY") is None


def test_stream_chat_events_nl_without_sql(mock_db: str) -> None:
    """Test stream_chat_events when LLM response contains plain answer without SQL code blocks."""
    with patch(
        "t1d_analytics.api.stream_llm_tokens",
        return_value=iter(["I cannot answer this question."]),
    ):
        resp = client.post(
            "/api/chat/stream",
            json={
                "message": "What is the meaning of life?",
                "model": "gemma4",
                "db_path": mock_db,
            },
        )
        assert resp.status_code == 200
        assert "I cannot answer this question." in resp.text
        assert "done" in resp.text
        assert "result" in resp.text


def test_extract_stream_delta_dict_empty() -> None:
    """Test extract_stream_delta with dict containing no text."""
    from t1d_analytics.api import extract_stream_delta

    assert extract_stream_delta({"empty": "dict"}, "ollama") == ""
    assert extract_stream_delta({"choices": ["not_a_dict"]}, "openai") == ""
    assert extract_stream_delta({"choices": [{"other": 1}]}, "openai") == ""
    assert extract_stream_delta({"choices": [{"delta": {}}]}, "openai") == ""

    assert extract_stream_delta({"choices": []}, "openai") == ""
    assert extract_stream_delta({"choices": [{"delta": "not_a_dict"}]}, "openai") == ""

    # Google empty candidates / parts, and text fallback
    chunk_google_empty_parts = MagicMock()
    chunk_google_empty_parts.candidates = [MagicMock(content=MagicMock(parts=[]))]
    chunk_google_empty_parts.text = "google_cand_text_fallback"
    assert (
        extract_stream_delta(chunk_google_empty_parts, "google")
        == "google_cand_text_fallback"
    )

    # Google parts with empty or non-string text
    chunk_google_non_str = MagicMock()
    chunk_google_non_str.candidates = [
        MagicMock(content=MagicMock(parts=[MagicMock(text=None)]))
    ]
    chunk_google_non_str.text = None
    assert extract_stream_delta(chunk_google_non_str, "google") == ""

    # Google no candidates and no text
    chunk_google_none = MagicMock(candidates=None, text=None)
    assert extract_stream_delta(chunk_google_none, "google") == ""

    # Anthropic empty delta text/content
    chunk_ant_none = MagicMock(
        delta=MagicMock(text=None, content=None), choices=None, text=None
    )
    assert extract_stream_delta(chunk_ant_none, "anthropic") == ""

    # Anthropic choices with empty text
    chunk_ant_choices_empty = MagicMock(
        delta=None, choices=[MagicMock(delta=MagicMock(text=None))], text=None
    )
    assert extract_stream_delta(chunk_ant_choices_empty, "anthropic") == ""

    # OpenAI choices with delta.content string
    chunk_openai_str = MagicMock(
        choices=[MagicMock(delta=MagicMock(content="real_delta"))], text=None
    )
    assert extract_stream_delta(chunk_openai_str, "openai") == "real_delta"

    # OpenAI choices with empty delta, message, text
    chunk_openai_none = MagicMock(
        choices=[
            MagicMock(
                delta=MagicMock(content=None, text=None),
                message=MagicMock(content=None),
                text=None,
            )
        ],
        text=None,
    )
    assert extract_stream_delta(chunk_openai_none, "openai") == ""

    # Unknown provider with arbitrary object
    assert extract_stream_delta(object(), "unknown_provider") == ""


def test_stream_llm_tokens_missing_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test stream_llm_tokens raises RuntimeError when external cloud provider key is missing."""
    from t1d_analytics.api import stream_llm_tokens

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="backend.missingApiKey"):
        list(stream_llm_tokens("test prompt", "openai/gpt-4o"))


def test_stream_llm_tokens_with_direct_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test stream_llm_tokens accepts direct api_key without checking or polluting os.environ."""
    from t1d_analytics.api import stream_llm_tokens

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    mock_llm = MagicMock()
    mock_llm.completion.return_value = ["token1", "token2"]

    with patch("any_llm.AnyLLM.create", return_value=mock_llm) as mock_create:
        tokens = list(
            stream_llm_tokens(
                "test prompt",
                "openai/gpt-4o",
                api_key="sk-test-stream-key",
            )
        )
        assert tokens == ["token1", "token2"]
        mock_create.assert_called_with("openai", api_key="sk-test-stream-key")
        assert os.environ.get("OPENAI_API_KEY") is None


def test_get_table_data_zero_columns(tmp_path: Path) -> None:
    """Test get_table_data gracefully handles tables when table_info reports 0 columns."""
    from typing import Any

    import duckdb

    db_file = tmp_path / "zero_cols.duckdb"
    conn = duckdb.connect(str(db_file))
    conn.execute("CREATE TABLE dummy_tbl (id INT)")
    conn.close()

    orig_connect = duckdb.connect

    class MockConn:
        def __init__(self, real_conn: Any) -> None:
            self._real = real_conn

        def execute(self, query: str, params: Any = None) -> Any:
            if "PRAGMA table_info" in query or "DESCRIBE" in query:
                mock_res = MagicMock()
                mock_res.fetchall.return_value = []
                mock_res.description = []
                return mock_res
            if params is not None:
                return self._real.execute(query, params)
            return self._real.execute(query)

        def close(self) -> None:
            self._real.close()

    def mock_conn_factory(path: str, **kwargs: Any) -> Any:
        real = orig_connect(path, **kwargs)
        return MockConn(real)

    with patch("duckdb.connect", side_effect=mock_conn_factory):
        resp = client.get(
            "/api/table/dummy_tbl",
            params={"db_path": str(db_file)},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["rows"] == []
        assert data["total_count"] == 0


def test_verify_api_auth_with_bearer_token(
    monkeypatch: pytest.MonkeyPatch, mock_db: str
) -> None:
    """Test API authentication middleware with T1D_API_BEARER_TOKEN."""
    # 1. Unset auth -> allows access
    monkeypatch.delenv("T1D_API_KEY", raising=False)
    monkeypatch.delenv("T1D_API_BEARER_TOKEN", raising=False)
    resp = client.get(f"/api/table/users?db_path={mock_db}")
    assert resp.status_code == 200

    # 2. Set T1D_API_BEARER_TOKEN -> rejects missing and invalid tokens
    monkeypatch.setenv("T1D_API_BEARER_TOKEN", "secret-bearer-999")
    resp_unauth = client.get(f"/api/table/users?db_path={mock_db}")
    assert resp_unauth.status_code == 401

    resp_bad = client.get(
        f"/api/table/users?db_path={mock_db}",
        headers={"Authorization": "Bearer wrong-token"},
    )
    assert resp_bad.status_code == 401

    # 3. Valid Bearer token
    resp_ok = client.get(
        f"/api/table/users?db_path={mock_db}",
        headers={"Authorization": "Bearer secret-bearer-999"},
    )
    assert resp_ok.status_code == 200

    # 4. Valid x-api-key header
    resp_ok_hdr = client.get(
        f"/api/table/users?db_path={mock_db}",
        headers={"x-api-key": "secret-bearer-999"},
    )
    assert resp_ok_hdr.status_code == 200

    # 5. Exempt health endpoint
    resp_health = client.get("/api/health")
    assert resp_health.status_code == 200


def test_training_job_endpoints(tmp_path: Path, mocker: typing.Any) -> None:
    """Test /api/training/jobs lifecycle: submit, list, get, and cancel."""
    payload = {
        "dataset_path": str(tmp_path / "ds.jsonl"),
        "output_dir": str(tmp_path / "out"),
        "backend": "cpu",
        "model_name": "gemma4-sql",
        "dry_run": True,
    }
    mocker.patch(
        "t1d_analytics.training_runner.LocalCpuRunner.run_training",
        return_value={"loss": 0.12},
    )
    resp_post = client.post("/api/training/jobs", json=payload)
    assert resp_post.status_code == 200
    job_id = resp_post.json()["job_id"]
    assert job_id.startswith("job-")

    # Shorthand backends (gpu, tpu)
    resp_gpu = client.post("/api/training/jobs", json={**payload, "backend": "gpu"})
    assert resp_gpu.status_code == 200
    resp_tpu = client.post("/api/training/jobs", json={**payload, "backend": "tpu"})
    assert resp_tpu.status_code == 200

    # Invalid backend -> 400
    resp_inv = client.post(
        "/api/training/jobs", json={**payload, "backend": "quantum-processor"}
    )
    assert resp_inv.status_code == 400

    # 2. List jobs
    resp_list = client.get("/api/training/jobs")
    assert resp_list.status_code == 200
    jobs = resp_list.json()["jobs"]
    assert any(j["job_id"] == job_id for j in jobs)

    # 3. Get job details
    resp_get = client.get(f"/api/training/jobs/{job_id}")
    assert resp_get.status_code == 200
    assert resp_get.json()["job_id"] == job_id

    # 4. Get non-existent job -> 404
    resp_404 = client.get("/api/training/jobs/non-existent-job-id")
    assert resp_404.status_code == 404

    # 5. Cancel non-existent job -> 404
    resp_cancel_404 = client.delete("/api/training/jobs/non-existent-job-id")
    assert resp_cancel_404.status_code == 404

    # 6. Cancel existing job
    resp_cancel = client.delete(f"/api/training/jobs/{job_id}")
    assert resp_cancel.status_code == 200
    assert resp_cancel.json()["status"] == "cancelled"


def test_multiturn_history_truncation_50_plus_turns() -> None:
    """Test conversations with 50+ turns properly truncate to sliding window without token overflow."""
    from t1d_analytics.api import ChatMessageModel, _build_sql_prompt

    # Create 60 turns of conversation history
    history = [
        ChatMessageModel(
            role="user" if i % 2 == 0 else "assistant",
            content=f"Turn message {i}",
            sqlQuery=f"SELECT {i}" if i % 2 == 1 else None,
        )
        for i in range(60)
    ]
    prompt = _build_sql_prompt("schema text", "Final query", history=history)
    # Verify older turns (e.g. Turn message 0 - 50) are truncated
    assert "Turn message 0" not in prompt
    assert "Turn message 40" not in prompt
    assert "Turn message 50" not in prompt
    # Verify recent turns (54 to 59) are present
    assert "Turn message 58" in prompt
    assert "Turn message 59" in prompt
    assert "Final query" in prompt


def test_malformed_prior_assistant_responses() -> None:
    """Test SQL extraction robustly handles unclosed code fences and malformed markdown."""
    from t1d_analytics.api import _extract_sql_from_response

    # Unclosed code fence
    malformed_unclosed = (
        "Here is the query:\n```sql\nSELECT * FROM patients WHERE id = 1"
    )
    sql1 = _extract_sql_from_response(malformed_unclosed)
    assert "SELECT * FROM patients" in sql1

    # Multiple code blocks with mixed fences
    mixed_fences = (
        "Some thoughts...\n```\nSELECT 1;\n```\nAnd another:\n```sql\nSELECT 2;\n```"
    )
    sql2 = _extract_sql_from_response(mixed_fences)
    assert "SELECT 2" in sql2 or "SELECT 1" in sql2

    # Plain text without fences but valid SELECT
    plain_select = "You can run this query:\nSELECT id, glucose FROM cgm LIMIT 10;"
    sql3 = _extract_sql_from_response(plain_select)
    assert "SELECT id, glucose FROM cgm" in sql3
