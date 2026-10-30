"""Tests for FastAPI chat, NL-to-SQL translation, SSE streaming, and SQL execution endpoints."""

import json
import os
import types
import typing
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import duckdb
import pytest
from fastapi.testclient import TestClient

from t1d_analytics.api import app, execute_sql, generate_sql_from_nl
from t1d_analytics.sql_utils import SqlValue

client = TestClient(app)


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
    with pytest.raises(ValueError, match=r"backend\.sqlExecution"):
        execute_sql(mock_db, "SELECT * FROM non_existent")


def test_execute_sql_sandboxed_external_access(mock_db: str) -> None:
    """Test SQL execution sandboxing blocks external filesystem access."""
    with pytest.raises(ValueError, match=r"backend\.sqlExecution"):
        execute_sql(mock_db, "SELECT * FROM read_csv('/etc/passwd')")


def test_execute_sql_db_not_found() -> None:
    """Test execute_sql when database file is missing."""
    with pytest.raises(ValueError, match=r"backend\.dbNotFound.*"):
        execute_sql("missing_file.duckdb", "SELECT 1")


def test_execute_sql_max_rows(mock_db: str) -> None:
    """Test execute_sql respects max_rows constraint."""
    res = execute_sql(mock_db, "SELECT * FROM users", max_rows=1)
    assert len(res) == 1
    assert res[0]["name"] == "Alice"


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


def test_execute_sql_timer_safe_interrupt_branch(mock_db: str) -> None:
    """Test the timer callback when connection is already closed to cover false branch."""
    import threading

    timer_callback = None
    original_timer = threading.Timer

    def mock_timer(
        interval: float, function: typing.Callable[[], None]
    ) -> threading.Timer:
        """Capture timer callback and return a real threading.Timer."""
        nonlocal timer_callback
        timer_callback = function
        return original_timer(interval, function)

    with patch("threading.Timer", side_effect=mock_timer):
        execute_sql(mock_db, "SELECT 1")

    assert timer_callback is not None
    timer_callback()


def test_query_timeout_cartesian_product(mock_db: str) -> None:
    """TEST-API-02: Interruption of pathological Cartesian cross-join query within 1.0s."""
    pathological_query = (
        "WITH RECURSIVE t(n) AS (VALUES (1) UNION ALL SELECT n+1 FROM t WHERE n < 100000000) "
        "SELECT a.n, b.n FROM t a CROSS JOIN t b"
    )
    with pytest.raises(ValueError) as exc_info:
        execute_sql(mock_db, pathological_query, timeout_seconds=0.1)

    assert "backend.queryTimeout" in str(exc_info.value)


@patch("any_llm.AnyLLM.create")
def test_generate_sql_from_nl_success(mock_create: MagicMock, mock_db: str) -> None:
    """Test natural language to SQL translation."""
    mock_llm = MagicMock()
    mock_msg = MagicMock(
        content="""```sql
SELECT * FROM users;
```"""
    )
    mock_choice = MagicMock(message=mock_msg)
    mock_response = MagicMock(choices=[mock_choice])
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
    mock_msg = MagicMock(content="```SELECT * FROM users```")
    mock_choice = MagicMock(message=mock_msg)
    mock_response = MagicMock(choices=[mock_choice])
    mock_llm.completion.return_value = mock_response
    mock_create.return_value = mock_llm

    sql = generate_sql_from_nl(mock_db, "get users")
    assert sql[1] == "SELECT * FROM users"


@patch("any_llm.AnyLLM.create")
def test_generate_sql_from_nl_prefixes(mock_create: MagicMock, mock_db: str) -> None:
    """Test NL to SQL with 'sql' and 'duckdb' prefixes."""
    mock_llm = MagicMock()
    mock_msg = MagicMock(
        content="""sql
SELECT * FROM a"""
    )
    mock_choice = MagicMock(message=mock_msg)
    mock_response = MagicMock(choices=[mock_choice])
    mock_llm.completion.return_value = mock_response
    mock_create.return_value = mock_llm

    sql = generate_sql_from_nl(mock_db, "test")
    assert sql[1] == "SELECT * FROM a"

    mock_msg.content = """duckdb
SELECT * FROM b"""
    sql = generate_sql_from_nl(mock_db, "test")
    assert sql[1] == "SELECT * FROM b"


def test_generate_sql_from_nl_no_module(mock_db: str) -> None:
    """Test handling of missing any_llm module."""
    with patch.dict("sys.modules", {"any_llm": None}):
        with pytest.raises(RuntimeError, match=r"backend\.missingSdk.*"):
            generate_sql_from_nl(mock_db, "test")


def test_generate_sql_from_nl_schema_error() -> None:
    """Test handling of DB connection error."""

    def failing_connect(path: str, read_only: bool = True) -> duckdb.DuckDBPyConnection:
        """Raise generic exception simulating schema read failure."""
        raise Exception("DB error")

    with pytest.raises(ValueError, match=r"backend\.readSchemaFailed.*"):
        generate_sql_from_nl("dummy.db", "test", connect_fn=failing_connect)


@patch("any_llm.AnyLLM.create")
def test_generate_sql_from_nl_llm_error(mock_create: MagicMock, mock_db: str) -> None:
    """Test handling of LLM completion error."""
    mock_create.side_effect = Exception("API limit")
    with pytest.raises(RuntimeError, match=r"backend\.llmTranslationError.*"):
        generate_sql_from_nl(mock_db, "test")


@patch("any_llm.AnyLLM.create")
def test_generate_sql_from_nl_db_not_found(mock_create: MagicMock) -> None:
    """Test generate_sql_from_nl when database file is missing."""
    with pytest.raises(ValueError, match=r"backend\.dbNotFound.*"):
        generate_sql_from_nl("missing_file.duckdb", "SELECT 1")


def test_generate_sql_fallback_no_prefix(mocker: MagicMock) -> None:
    """Test SQL fallback when prefix is missing."""
    mock_llm_instance = mocker.MagicMock()
    mock_msg = mocker.MagicMock(content="SELECT * FROM test;")
    mock_choice = mocker.MagicMock(message=mock_msg)
    mock_llm_instance.completion.return_value = mocker.MagicMock(choices=[mock_choice])

    mock_conn = mocker.MagicMock()
    _full, sql = generate_sql_from_nl(
        "dummy_db",
        "test query",
        connect_fn=lambda path, read_only=True: mock_conn,
        schema_fn=lambda conn: "dummy_schema",
        llm_factory=lambda provider, api_key=None: mock_llm_instance,
    )
    assert sql == "SELECT * FROM test;"


@patch("any_llm.AnyLLM.create")
def test_generate_sql_from_nl_multi_block(mock_create: MagicMock, mock_db: str) -> None:
    """Test generate_sql_from_nl extracts the final SQL query block when multiple markdown blocks exist."""
    mock_llm = MagicMock()
    mock_msg = MagicMock(
        content="""
    Here is some preliminary exploration:
    ```sql
    DESCRIBE users;
    ```
    And here is the final query answering your question:
    ```sql
    SELECT id, name FROM users WHERE id > 0;
    ```
    """
    )
    mock_choice = MagicMock(message=mock_msg)
    mock_response = MagicMock(choices=[mock_choice])
    mock_llm.completion.return_value = mock_response
    mock_create.return_value = mock_llm

    _full_resp, sql = generate_sql_from_nl(mock_db, "get users")
    assert sql == "SELECT id, name FROM users WHERE id > 0;"


@patch("any_llm.AnyLLM.create")
def test_generate_sql_from_nl_non_query_block(
    mock_create: MagicMock, mock_db: str
) -> None:
    """Test generate_sql_from_nl falls back to last block if no keyword matched."""
    mock_llm = MagicMock()
    mock_msg = MagicMock(
        content="""
    ```text
    random text block
    ```
    """
    )
    mock_choice = MagicMock(message=mock_msg)
    mock_response = MagicMock(choices=[mock_choice])
    mock_llm.completion.return_value = mock_response
    mock_create.return_value = mock_llm

    _full_resp, sql = generate_sql_from_nl(mock_db, "get users")
    assert sql == "random text block"


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


def test_execute_sql_endpoint_invalid_db() -> None:
    """Test /api/execute-sql returns 400 for out-of-workspace db_path."""
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
        [
            "Here is the query: ",
            """```sql
""",
            """SELECT * FROM users;
""",
            "```",
        ]
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
        [
            "Here is the query: ",
            """```sql
""",
            """SELECT * FROM missing_table;
""",
            "```",
        ]
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


def test_chat_stream_invalid_db_path() -> None:
    """Test /api/chat and /api/chat/stream handle invalid db_path."""
    resp_chat = client.post(
        "/api/chat",
        json={"message": "SELECT 1", "model": "sql", "db_path": "../../evil.duckdb"},
    )
    assert resp_chat.status_code == 400
    assert resp_chat.json()["detail"]["error_code"] == "backend.invalidDatabasePath"

    resp = client.post(
        "/api/chat/stream",
        json={"message": "hello", "db_path": "../../evil.duckdb"},
    )
    assert resp.status_code == 200
    assert "backend.invalidDatabasePath" in resp.text


def test_server_auth_header_not_leaked_to_llm_provider(
    monkeypatch: pytest.MonkeyPatch, mock_db: str
) -> None:
    """Verify server x-api-key header is used only for server auth and never forwarded as an LLM provider API key."""
    monkeypatch.setenv("T1D_API_KEY", "server-auth-secret-token")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with patch("t1d_analytics.api.generate_sql_from_nl") as mock_gen:
        mock_gen.return_value = ("Here is the query", "SELECT 1")
        resp = client.post(
            "/api/chat",
            headers={"x-api-key": "server-auth-secret-token"},
            json={
                "message": "Count patients",
                "model": "openai/gpt-4o",
                "db_path": mock_db,
            },
        )
        assert resp.status_code == 200
        _, kwargs = mock_gen.call_args
        assert kwargs.get("api_key") is None

        resp_with_prov = client.post(
            "/api/chat",
            headers={
                "x-api-key": "server-auth-secret-token",
                "x-provider-api-key": "sk-explicit-provider-key",
            },
            json={
                "message": "Count patients",
                "model": "openai/gpt-4o",
                "db_path": mock_db,
            },
        )
        assert resp_with_prov.status_code == 200
        _, kwargs_prov = mock_gen.call_args
        assert kwargs_prov.get("api_key") == "sk-explicit-provider-key"


@patch("any_llm.AnyLLM.create")
def test_stream_llm_tokens_branches(mock_create: MagicMock) -> None:
    """Test stream_llm_tokens across stream generator, message content, string chunk, and fallback."""
    from t1d_analytics.api import stream_llm_tokens

    mock_llm = MagicMock()
    mock_create.return_value = mock_llm

    chunk1 = MagicMock(
        choices=[MagicMock(delta=MagicMock(content="token1 "), message=None)]
    )
    mock_llm.completion.return_value = [chunk1]

    tokens = list(stream_llm_tokens("prompt", "gemma4"))
    assert tokens == ["token1 "]

    choice2 = MagicMock(message=MagicMock(content="token2 "))
    del choice2.delta
    chunk2 = MagicMock(choices=[choice2])
    mock_llm.completion.return_value = [chunk2]

    tokens2 = list(stream_llm_tokens("prompt", "gemma4"))
    assert tokens2 == ["token2 "]

    mock_llm.completion.return_value = ["chunk_a", "chunk_b", ""]
    tokens3 = list(stream_llm_tokens("prompt", "gemma4"))
    assert tokens3 == ["chunk_a", "chunk_b"]

    empty_chunk = MagicMock(
        choices=[
            MagicMock(
                delta=MagicMock(content=""),
                message=MagicMock(content=""),
            )
        ]
    )
    mock_llm.completion.return_value = [empty_chunk, types.SimpleNamespace()]
    assert list(stream_llm_tokens("prompt", "gemma4")) == []

    mock_resp_obj = MagicMock(
        choices=[MagicMock(message=MagicMock(content="non_iter fallback"))]
    )
    mock_llm.completion.side_effect = [types.SimpleNamespace(), mock_resp_obj]
    assert list(stream_llm_tokens("prompt", "gemma4")) == ["non_iter fallback"]

    mock_resp = MagicMock(
        choices=[MagicMock(message=MagicMock(content="fallback words stream"))]
    )

    def fail_streaming(
        **kwargs: typing.Union[SqlValue, list[dict[str, str]]],
    ) -> MagicMock:
        """Raise error when stream=True, else return mock_resp."""
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
            """```sql
""",
            "SELECT * ",
            "FROM users",
            """
```""",
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


def test_multi_turn_history_in_prompt_and_chat(mock_db: str, mocker: MagicMock) -> None:
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

    prompt_empty = _build_sql_prompt("schema text", "Show patients", history=[])
    assert "Prior Conversation History:" not in prompt_empty

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


def test_multiprovider_llm_validation(
    monkeypatch: pytest.MonkeyPatch, mock_db: str
) -> None:
    """Test multi-provider parsing, missing API key, and missing SDK error handling."""
    from t1d_analytics.api import stream_llm_tokens

    monkeypatch.setenv("OPENAI_API_KEY", "test-stream-key")
    mock_llm = MagicMock()
    mock_llm.completion.return_value = ["token1 ", "token2"]
    with patch("any_llm.AnyLLM.create", return_value=mock_llm):
        tokens = list(stream_llm_tokens("test prompt", "openai/gpt-4o"))
        assert len(tokens) > 0

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError) as exc_info:
        generate_sql_from_nl(mock_db, "Count users", model_name="openai/gpt-4o")
    assert "backend.missingApiKey" in str(exc_info.value)

    with patch.dict("sys.modules", {"any_llm": None}):
        with pytest.raises(RuntimeError) as exc_info:
            generate_sql_from_nl(
                mock_db, "Count users", model_name="anthropic/claude-3"
            )
        assert "backend.missingSdk" in str(exc_info.value)

    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    mock_llm2 = MagicMock()
    mock_choice = MagicMock(
        message=MagicMock(
            content="""```sql
SELECT count(*) FROM users;
```"""
        )
    )
    mock_resp = MagicMock(choices=[mock_choice])
    mock_llm2.completion.return_value = mock_resp
    with patch("any_llm.AnyLLM.create", return_value=mock_llm2):
        _full, sql = generate_sql_from_nl(
            mock_db, "Count users", model_name="openai/gpt-4o"
        )
        assert "SELECT count(*) FROM users" in sql


def test_extract_stream_delta_various_providers() -> None:
    """Test extract_stream_delta across string, dict, OpenAI, Anthropic, Google, and fallback formats."""
    from t1d_analytics.api import extract_stream_delta

    assert extract_stream_delta("plain token", "ollama") == "plain token"
    assert (
        extract_stream_delta({"choices": [{"delta": {"content": "tok1"}}]}, "openai")
        == "tok1"
    )
    assert extract_stream_delta({"choices": [{"text": "tok2"}]}, "openai") == "tok2"
    assert extract_stream_delta({"message": {"content": "tok3"}}, "ollama") == "tok3"
    assert extract_stream_delta({"response": "tok4"}, "ollama") == "tok4"
    assert extract_stream_delta({"text": "tok5"}, "google") == "tok5"

    choice_delta = MagicMock(delta=MagicMock(content="openai_delta"))
    chunk_openai_delta = MagicMock(choices=[choice_delta])
    assert extract_stream_delta(chunk_openai_delta, "openai") == "openai_delta"

    choice_msg = MagicMock(spec=["message"])
    choice_msg.message = MagicMock(content="openai_msg")
    chunk_openai_msg = MagicMock(choices=[choice_msg])
    assert extract_stream_delta(chunk_openai_msg, "ollama") == "openai_msg"

    choice_txt = MagicMock(spec=["text"])
    choice_txt.text = "openai_text"
    chunk_openai_txt = MagicMock(choices=[choice_txt])
    assert extract_stream_delta(chunk_openai_txt, "openai") == "openai_text"

    chunk_anthropic = MagicMock(delta=MagicMock(text="anthropic_tok"))
    assert extract_stream_delta(chunk_anthropic, "anthropic") == "anthropic_tok"

    chunk_anthropic_content = MagicMock(spec=["delta"])
    chunk_anthropic_content.delta = MagicMock(spec=["content"])
    chunk_anthropic_content.delta.content = "anthropic_content"
    assert (
        extract_stream_delta(chunk_anthropic_content, "anthropic")
        == "anthropic_content"
    )

    chunk_anthropic_choice = MagicMock(spec=["choices"])
    choice_ant = MagicMock(delta=MagicMock(text="choice_ant_tok"))
    chunk_anthropic_choice.choices = [choice_ant]
    assert extract_stream_delta(chunk_anthropic_choice, "anthropic") == "choice_ant_tok"

    part = MagicMock(text="google_tok")
    cand = MagicMock(content=MagicMock(parts=[part]))
    chunk_google = MagicMock(candidates=[cand])
    assert extract_stream_delta(chunk_google, "google") == "google_tok"

    chunk_google_text = MagicMock(spec=["text"])
    chunk_google_text.text = "google_direct_text"
    assert extract_stream_delta(chunk_google_text, "google") == "google_direct_text"

    chunk_fallback_delta = MagicMock(spec=["choices"])
    c_fall = MagicMock(delta=MagicMock(content="fallback_content"))
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
    assert extract_stream_delta(types.SimpleNamespace(), "openai") == ""


def test_stream_llm_tokens_native_and_fallback() -> None:
    """Test stream_llm_tokens yields native stream chunks or falls back to full completion block."""
    from t1d_analytics.api import stream_llm_tokens

    mock_llm = MagicMock()
    mock_chunk1 = MagicMock()
    mock_chunk1.choices = [MagicMock(delta=MagicMock(content="word1 "))]
    mock_chunk2 = MagicMock()
    mock_chunk2.choices = [MagicMock(delta=MagicMock(content="word2"))]
    mock_llm.completion.return_value = [mock_chunk1, mock_chunk2]

    with patch("any_llm.AnyLLM.create", return_value=mock_llm):
        tokens = list(stream_llm_tokens("test prompt", "gemma4", provider="ollama"))
        assert tokens == ["word1 ", "word2"]

    mock_llm_err = MagicMock()

    def fail_streaming(
        **kwargs: typing.Union[SqlValue, list[dict[str, str]]],
    ) -> MagicMock:
        """Raise RuntimeError when stream=True, else return single response block."""
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

    req_disc_before = MagicMock()
    req_disc_before.is_disconnected = AsyncMock(return_value=True)
    chat_req = ChatRequest(message="SELECT 1", model="sql", db_path=mock_db)

    with pytest.raises(StopAsyncIteration):
        gen1 = stream_chat_events(chat_req, req=req_disc_before)
        await gen1.__anext__()

    req_disc_during = MagicMock()
    req_disc_during.is_disconnected = AsyncMock(side_effect=[False, True])

    chat_req_nl = ChatRequest(message="Count users", model="gemma4", db_path=mock_db)
    with patch(
        "t1d_analytics.api.stream_llm_tokens", return_value=iter(["tok1", "tok2"])
    ):
        gen = stream_chat_events(chat_req_nl, req=req_disc_during)
        with pytest.raises(StopAsyncIteration):
            await gen.__anext__()


def test_chat_endpoints_with_provider_headers(mock_db: str) -> None:
    """Test /api/chat and /api/chat/stream accept x-provider and x-provider-api-key headers."""
    with patch(
        "t1d_analytics.api.generate_sql_from_nl", return_value=("answer", "SELECT 1")
    ):
        resp = client.post(
            "/api/chat",
            json={"message": "Count users", "db_path": mock_db},
            headers={"x-provider": "openai", "x-provider-api-key": "sk-header-test"},
        )
        assert resp.status_code == 200

    with patch("t1d_analytics.api.stream_llm_tokens", return_value=iter(["Done"])):
        resp_stream = client.post(
            "/api/chat/stream",
            json={"message": "Count users", "db_path": mock_db},
            headers={"x-provider": "anthropic", "x-provider-api-key": "sk-ant-header"},
        )
        assert resp_stream.status_code == 200
        assert "Done" in resp_stream.text


def test_generate_sql_from_nl_with_direct_api_key(
    monkeypatch: pytest.MonkeyPatch, mock_db: str
) -> None:
    """Test generate_sql_from_nl passes client-supplied api_key to AnyLLM without polluting environment."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    mock_llm = MagicMock()
    mock_choice = MagicMock(
        message=MagicMock(
            content="""```sql
SELECT 1;
```"""
        )
    )
    mock_llm.completion.return_value = MagicMock(choices=[mock_choice])

    with patch("any_llm.AnyLLM.create", return_value=mock_llm) as mock_create:
        _full, sql = generate_sql_from_nl(
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

    chunk_google_empty_parts = MagicMock()
    chunk_google_empty_parts.candidates = [MagicMock(content=MagicMock(parts=[]))]
    chunk_google_empty_parts.text = "google_cand_text_fallback"
    assert (
        extract_stream_delta(chunk_google_empty_parts, "google")
        == "google_cand_text_fallback"
    )

    chunk_google_non_str = MagicMock()
    chunk_google_non_str.candidates = [
        MagicMock(content=MagicMock(parts=[MagicMock(text=None)]))
    ]
    chunk_google_non_str.text = None
    assert extract_stream_delta(chunk_google_non_str, "google") == ""

    chunk_google_none = MagicMock(candidates=None, text=None)
    assert extract_stream_delta(chunk_google_none, "google") == ""

    chunk_ant_none = MagicMock(
        delta=MagicMock(text=None, content=None), choices=None, text=None
    )
    assert extract_stream_delta(chunk_ant_none, "anthropic") == ""

    chunk_ant_choices_empty = MagicMock(
        delta=None, choices=[MagicMock(delta=MagicMock(text=None))], text=None
    )
    assert extract_stream_delta(chunk_ant_choices_empty, "anthropic") == ""

    chunk_openai_str = MagicMock(
        choices=[MagicMock(delta=MagicMock(content="real_delta"))], text=None
    )
    assert extract_stream_delta(chunk_openai_str, "openai") == "real_delta"

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
    assert extract_stream_delta(types.SimpleNamespace(), "unknown_provider") == ""


def test_stream_llm_tokens_missing_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test stream_llm_tokens raises RuntimeError when external cloud provider key is missing."""
    from t1d_analytics.api import stream_llm_tokens

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match=r"backend\.missingApiKey"):
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


def test_multiturn_history_truncation_50_plus_turns() -> None:
    """Test conversations with 50+ turns properly truncate to sliding window without token overflow."""
    from t1d_analytics.api import ChatMessageModel, _build_sql_prompt

    history = [
        ChatMessageModel(
            role="user" if i % 2 == 0 else "assistant",
            content=f"Turn message {i}",
            sqlQuery=f"SELECT {i}" if i % 2 == 1 else None,
        )
        for i in range(60)
    ]
    prompt = _build_sql_prompt("schema text", "Final query", history=history)
    assert "Turn message 0" not in prompt
    assert "Turn message 40" not in prompt
    assert "Turn message 50" not in prompt
    assert "Turn message 58" in prompt
    assert "Turn message 59" in prompt
    assert "Final query" in prompt


def test_malformed_prior_assistant_responses() -> None:
    """Test SQL extraction robustly handles unclosed code fences and malformed markdown."""
    from t1d_analytics.api import _extract_sql_from_response

    malformed_unclosed = """Here is the query:
```sql
SELECT * FROM patients WHERE id = 1"""
    sql1 = _extract_sql_from_response(malformed_unclosed)
    assert "SELECT * FROM patients" in sql1

    mixed_fences = """Some thoughts...
```
SELECT 1;
```
And another:
```sql
SELECT 2;
```"""
    sql2 = _extract_sql_from_response(mixed_fences)
    assert "SELECT 2" in sql2 or "SELECT 1" in sql2

    plain_select = """You can run this query:
SELECT id, glucose FROM cgm LIMIT 10;"""
    sql3 = _extract_sql_from_response(plain_select)
    assert "SELECT id, glucose FROM cgm" in sql3


def test_chat_auto_execute_false_and_varchar_sum_coercion(tmp_path: Path) -> None:
    """Test auto_execute=False on /api/chat and /api/chat/stream and SUM(VARCHAR) coercion in execute_sql."""
    db_file = tmp_path / "varchar_hypo.duckdb"
    conn = duckdb.connect(str(db_file))
    conn.execute(
        "CREATE TABLE tblaptsummary (PtID INT, TxGroup VARCHAR, Gender VARCHAR, AgeAsOfRandDt VARCHAR, NumSevHypo VARCHAR)"
    )
    conn.execute(
        "INSERT INTO tblaptsummary VALUES (1, 'Control', 'M', '20', '2'), (2, 'Control', 'F', 'NA', '3')"
    )
    conn.close()

    default_sql = """SELECT
    TxGroup,
    Gender,
    COUNT(DISTINCT PtID) as patient_count,
    ROUND(AVG(AgeAsOfRandDt), 1) as avg_age,
    SUM(NumSevHypo) as total_severe_hypo_events
FROM tblaptsummary
GROUP BY TxGroup, Gender"""
    rows = execute_sql(str(db_file), default_sql)
    assert len(rows) == 2
    total_hypo = sum(float(r["total_severe_hypo_events"] or 0) for r in rows)
    assert total_hypo == 5.0

    with patch(
        "t1d_analytics.api.generate_sql_from_nl",
        return_value=(
            """```sql
SELECT * FROM tblaptsummary
```""",
            "SELECT * FROM tblaptsummary",
        ),
    ):
        resp = client.post(
            "/api/chat",
            json={
                "message": "Show summary",
                "model": "gemma4",
                "db_path": str(db_file),
                "auto_execute": False,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["sqlQuery"] == "SELECT * FROM tblaptsummary"
        assert data["sqlResult"] is None

    with patch(
        "t1d_analytics.api.stream_llm_tokens",
        return_value=iter(
            [
                """```sql
SELECT * FROM tblaptsummary
```"""
            ]
        ),
    ):
        resp_stream = client.post(
            "/api/chat/stream",
            json={
                "message": "Show summary",
                "model": "gemma4",
                "db_path": str(db_file),
                "auto_execute": False,
            },
        )
        assert resp_stream.status_code == 200
        assert "sql_generated" in resp_stream.text
        assert "Executing SQL query..." not in resp_stream.text


def test_ollama_crash_fallback_and_readonly_sql_routing(mock_db: str) -> None:
    """Test synthesize_fallback_clinical_sql, _is_recoverable_local_ollama_crash, and read-only SQL auto-routing."""
    from t1d_analytics.chat_service import (
        _is_recoverable_local_ollama_crash,
        synthesize_fallback_clinical_sql,
        translate_nl_to_sql,
    )

    assert (
        _is_recoverable_local_ollama_crash(
            "openai", RuntimeError("insufficient memory")
        )
        is False
    )
    assert (
        _is_recoverable_local_ollama_crash(
            "ollama",
            RuntimeError(
                "llama-server process has terminated: exit status 1: insufficient memory"
            ),
        )
        is True
    )
    assert (
        _is_recoverable_local_ollama_crash("ollama", RuntimeError("generic error"))
        is False
    )

    # 1. Empty schema map
    resp_empty, sql_empty = synthesize_fallback_clinical_sql("", "How many patients?")
    assert 'SELECT COUNT(*) AS count FROM "patients";' in sql_empty
    assert "```sql" in resp_empty

    # 2. Schema with matching table and column words + "first 5"
    schema_text = (
        "Table: demographics\nColumns: patient_id (BIGINT), age (DOUBLE), gender (VARCHAR)\n\n"
        "Table: cgm_readings\nColumns: subject_id (BIGINT), glucose_value (DOUBLE)\n"
    )
    _, sql_match = synthesize_fallback_clinical_sql(
        schema_text, "Show me the first 5 patients in the demographics table age"
    )
    assert sql_match == 'SELECT * FROM "demographics" LIMIT 5;'

    # 3. Schema with zero matching words -> falls back to preferred table ("patients")
    schema_pref = (
        "Table: other_tbl\nColumns: x (INT)\n\nTable: patients\nColumns: id (INT)\n"
    )
    _, sql_pref = synthesize_fallback_clinical_sql(schema_pref, "unrelated question")
    assert sql_pref == 'SELECT * FROM "patients" LIMIT 10;'

    # 4. Schema with no preferred table and zero matching words -> first table in schema
    schema_first = "Table: custom_alpha\nColumns: a (INT)\n\nTable: custom_beta\nColumns: b (INT)\n"
    _, sql_first = synthesize_fallback_clinical_sql(schema_first, "unrelated question")
    assert sql_first == 'SELECT * FROM "custom_alpha" LIMIT 10;'

    # 5. translate_nl_to_sql fallback on Ollama OOM crash
    with patch(
        "t1d_analytics.chat_service.complete_llm_prompt",
        side_effect=RuntimeError(
            "llama-server process has terminated: unable to allocate AMX buffer"
        ),
    ):
        fb_full, fb_sql = translate_nl_to_sql(
            mock_db, "Show me the first 5 users", model_name="gemma4:latest"
        )
        assert fb_sql == 'SELECT * FROM "users" LIMIT 5;'
        assert fb_sql in fb_full

    # 6. /api/chat/stream recovers when stream_fn raises a local Ollama OOM / sync-in-async crash
    with patch(
        "t1d_analytics.api.stream_llm_tokens",
        side_effect=RuntimeError(
            "Cannot use the `sync` API in an `async` context. Use the `async` API instead."
        ),
    ):
        resp_crash_stream = client.post(
            "/api/chat/stream",
            json={
                "message": "Show me the first 5 users",
                "model": "gemma4:latest",
                "db_path": mock_db,
            },
        )
        assert resp_crash_stream.status_code == 200
        assert "sql_generated" in resp_crash_stream.text
        assert 'SELECT * FROM \\"users\\" LIMIT 5;' in resp_crash_stream.text

    # 7. is_likely_sql_statement and _resolve_sql_mode_query in batch /api/chat and /api/chat/stream
    from t1d_analytics.chat_service import is_likely_sql_statement

    assert is_likely_sql_statement("   ") is False
    assert is_likely_sql_statement("SELECT * FROM users") is True
    assert is_likely_sql_statement("SHOW TABLES") is True
    assert is_likely_sql_statement("Show me the first 5 users") is False
    assert is_likely_sql_statement("How many users?") is False

    resp_nl_sql_mode = client.post(
        "/api/chat",
        json={
            "message": "Show me the first 5 users",
            "model": "sql",
            "db_path": mock_db,
        },
    )
    assert resp_nl_sql_mode.status_code == 200
    assert resp_nl_sql_mode.json()["sqlQuery"] == 'SELECT * FROM "users" LIMIT 5;'

    resp_nl_sql_stream = client.post(
        "/api/chat/stream",
        json={
            "message": "Show me the first 5 users",
            "model": "sql",
            "db_path": mock_db,
        },
    )
    assert resp_nl_sql_stream.status_code == 200
    assert 'SELECT * FROM \\"users\\" LIMIT 5;' in resp_nl_sql_stream.text
