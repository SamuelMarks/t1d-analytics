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
from t1d_analytics.sql_fallback_synthesizer import TranslationResult
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
    from t1d_analytics.errors import SqlExecutionError

    with pytest.raises(SqlExecutionError, match=r"backend\.sqlExecution"):
        execute_sql(mock_db, "SELECT * FROM read_csv('/etc/passwd')")


def test_execute_sql_db_not_found() -> None:
    """Test execute_sql when database file is missing."""
    from t1d_analytics.errors import DatabaseNotFoundError

    with pytest.raises(DatabaseNotFoundError, match=r"backend\.dbNotFound.*"):
        execute_sql("missing_file.duckdb", "SELECT 1")


def test_execute_sql_max_rows(mock_db: str) -> None:
    """Test execute_sql respects max_rows constraint."""
    res = execute_sql(mock_db, "SELECT * FROM users", max_rows=1)
    assert len(res) == 1
    assert res[0]["name"] == "Alice"


def test_execute_sql_timeout(mock_db: str) -> None:
    """Test execute_sql timeout triggers SqlTimeoutError and evicts the interrupted connection from the pool."""
    from t1d_analytics.errors import SqlTimeoutError
    from t1d_analytics.sql_utils import global_read_only_duckdb_pool

    global_read_only_duckdb_pool.close_path(mock_db)
    with patch("duckdb.connect") as mock_connect:
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn
        mock_conn.execute.side_effect = [
            None,
            duckdb.InterruptException("Query interrupted"),
        ]
        with pytest.raises(SqlTimeoutError, match=r"backend\.queryTimeout"):
            execute_sql(mock_db, "SELECT 1", timeout_seconds=0.001)
        mock_conn.close.assert_called_once()
        resolved_key = str(Path(mock_db).resolve())
        assert global_read_only_duckdb_pool._pools.get(resolved_key, []) == []


def test_execute_sql_timer_safe_interrupt_branch(mock_db: str) -> None:
    """Test that timeout timer firing concurrently after connection release does not call interrupt() on a closed connection."""
    import threading

    fired = threading.Event()
    original_timer = threading.Timer

    def post_close_timer(
        interval: float, function: typing.Callable[[], None]
    ) -> threading.Timer:
        """Return a real threading.Timer that also invokes function once after timer.cancel()."""
        _ = interval

        def _wrapped() -> None:
            """Execute the timer callback."""
            function()
            fired.set()

        t = original_timer(10.0, _wrapped)
        orig_cancel = t.cancel

        def _cancel_and_fire_late() -> None:
            """Cancel timer and simulate late callback execution after active_conn is cleared."""
            orig_cancel()
            threading.Thread(target=_wrapped, daemon=True).start()

        t.cancel = _cancel_and_fire_late  # type: ignore[method-assign]
        return t

    with patch("threading.Timer", side_effect=post_close_timer):
        rows = execute_sql(mock_db, "SELECT id, name FROM users ORDER BY id")

    assert fired.wait(timeout=2.0) is True
    assert rows == [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]


def test_query_timeout_cartesian_product(mock_db: str) -> None:
    """TEST-API-02: Interruption of pathological Cartesian cross-join query within 1.0s."""
    from t1d_analytics.errors import SqlTimeoutError

    pathological_query = (
        "WITH RECURSIVE t(n) AS (VALUES (1) UNION ALL SELECT n+1 FROM t WHERE n < 100000000) "
        "SELECT a.n, b.n FROM t a CROSS JOIN t b"
    )
    with pytest.raises(SqlTimeoutError) as exc_info:
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
    assert sql.sql_query == "SELECT * FROM users;"
    assert sql.is_rule_based_sql is False


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
    assert sql.sql_query == "SELECT * FROM users"


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
    assert sql.sql_query == "SELECT * FROM a"

    mock_msg.content = """duckdb
SELECT * FROM b"""
    sql = generate_sql_from_nl(mock_db, "test")
    assert sql.sql_query == "SELECT * FROM b"


def test_generate_sql_from_nl_no_module(mock_db: str) -> None:
    """Test handling of missing any_llm module."""
    from t1d_analytics.errors import MissingSdkError

    with patch.dict("sys.modules", {"any_llm": None}):
        with pytest.raises(MissingSdkError, match=r"backend\.missingSdk.*"):
            generate_sql_from_nl(mock_db, "test")


def test_generate_sql_from_nl_schema_error() -> None:
    """Test handling of DB connection error."""
    from t1d_analytics.errors import SchemaReadError

    def failing_connect(path: str, read_only: bool = True) -> duckdb.DuckDBPyConnection:
        """Raise duckdb.Error simulating schema read failure."""
        _ = (path, read_only)
        raise duckdb.Error("DB error")

    with pytest.raises(SchemaReadError, match=r"backend\.readSchemaFailed.*"):
        generate_sql_from_nl("dummy.db", "test", connect_fn=failing_connect)


@patch("any_llm.AnyLLM.create")
def test_generate_sql_from_nl_llm_error(mock_create: MagicMock, mock_db: str) -> None:
    """Test handling of LLM completion error."""
    from t1d_analytics.errors import LlmTranslationError

    mock_create.side_effect = RuntimeError("API limit")
    with pytest.raises(LlmTranslationError, match=r"backend\.llmTranslationError.*"):
        generate_sql_from_nl(mock_db, "test")


@patch("any_llm.AnyLLM.create")
def test_generate_sql_from_nl_db_not_found(mock_create: MagicMock) -> None:
    """Test generate_sql_from_nl when database file is missing."""
    from t1d_analytics.errors import DatabaseNotFoundError

    with pytest.raises(DatabaseNotFoundError, match=r"backend\.dbNotFound.*"):
        generate_sql_from_nl("missing_file.duckdb", "SELECT 1")


def test_generate_sql_fallback_no_prefix(mocker: MagicMock) -> None:
    """Test SQL fallback when prefix is missing."""
    mock_llm_instance = mocker.MagicMock()
    mock_msg = mocker.MagicMock(content="SELECT * FROM test;")
    mock_choice = mocker.MagicMock(message=mock_msg)
    mock_llm_instance.completion.return_value = mocker.MagicMock(choices=[mock_choice])

    mock_conn = mocker.MagicMock()
    res = generate_sql_from_nl(
        "dummy_db",
        "test query",
        connect_fn=lambda path, read_only=True: mock_conn,
        schema_fn=lambda conn: "dummy_schema",
        llm_factory=lambda provider, api_key=None: mock_llm_instance,
    )
    assert res.sql_query == "SELECT * FROM test;"


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

    res = generate_sql_from_nl(mock_db, "get users")
    assert res.sql_query == "SELECT id, name FROM users WHERE id > 0;"


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

    res = generate_sql_from_nl(mock_db, "get users")
    assert res.sql_query == "random text block"


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


@patch("t1d_analytics.api_dependencies.translate_nl_to_sql")
def test_chat_endpoint_nl_success(mock_generate: MagicMock, mock_db: str) -> None:
    """Test API handles NLP queries successfully, passes expected arguments, and auto-executes the query."""
    mock_generate.return_value = TranslationResult(
        full_response="Generated Markdown", sql_query="SELECT * FROM users"
    )
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
    mock_generate.assert_called_once()
    call_args, call_kwargs = mock_generate.call_args
    assert Path(call_args[0]).resolve() == Path(mock_db).resolve()
    assert call_args[1] == "give me users"
    assert call_kwargs["model_name"] == "gemma4"


@patch("t1d_analytics.api_dependencies.translate_nl_to_sql")
def test_chat_endpoint_nl_sql_execution_failure(
    mock_generate: MagicMock, mock_db: str
) -> None:
    """Test API handles NLP queries where generated SQL execution fails."""
    mock_generate.return_value = TranslationResult(
        full_response="Generated Markdown",
        sql_query="SELECT * FROM nonexistent_table",
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


@patch("t1d_analytics.api_dependencies.translate_nl_to_sql")
def test_chat_endpoint_nl_no_sql_query(mock_generate: MagicMock, mock_db: str) -> None:
    """Test API handles NLP response that contains no SQL query."""
    mock_generate.return_value = TranslationResult(
        full_response="Just an explanation, no query.", sql_query=""
    )
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
    mock_create.side_effect = RuntimeError("LLM offline")
    response = client.post(
        "/api/chat",
        json={"message": "give me users", "model": "gemma4", "db_path": mock_db},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["error"]["error_code"] == "backend.llmTranslationError"
    assert data["content"] == "backend.errorNlpTranslation"


@patch("t1d_analytics.api_dependencies.translate_nl_to_sql")
def test_chat_endpoint_generic_exception(
    mock_generate: MagicMock, mock_db: str
) -> None:
    """Test API handles generic unexpected exceptions."""
    mock_generate.side_effect = OSError("System failure")
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


@patch("t1d_analytics.api_dependencies.execute_sql")
def test_execute_sql_endpoint_exception(mock_execute: MagicMock, mock_db: str) -> None:
    """Test execute_sql_endpoint catching general Exception."""
    mock_execute.side_effect = OSError("System crash")
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


@patch("t1d_analytics.api_dependencies.stream_llm_completion_tokens")
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


@patch("t1d_analytics.api_dependencies.stream_llm_completion_tokens")
def test_chat_stream_nl_no_sql_generated(mock_stream: MagicMock, mock_db: str) -> None:
    """Test /api/chat/stream when NL prompt does not yield a SQL query."""
    mock_stream.return_value = iter(["I cannot answer this with SQL"])
    resp = client.post(
        "/api/chat/stream",
        json={"message": "hello", "model": "gemma4", "db_path": mock_db},
    )
    assert resp.status_code == 200
    assert "I cannot answer this with SQL" in resp.text

    mock_stream.return_value = iter(["```sql\n```"])
    resp_empty_sql = client.post(
        "/api/chat/stream",
        json={"message": "hello", "model": "gemma4", "db_path": mock_db},
    )
    assert resp_empty_sql.status_code == 200
    assert "sql_generated" not in resp_empty_sql.text


@patch("t1d_analytics.api_dependencies.stream_llm_completion_tokens")
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


@patch("t1d_analytics.api_dependencies.stream_llm_completion_tokens")
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

    with patch("t1d_analytics.api_dependencies.translate_nl_to_sql") as mock_gen:
        mock_gen.return_value = TranslationResult(
            full_response="Here is the query", sql_query="SELECT 1"
        )
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


@patch("t1d_analytics.api_dependencies.stream_llm_completion_tokens")
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
    """Test build_sql_prompt_with_history and chat endpoints with multi-turn conversation history."""
    from t1d_analytics.api_schemas import ChatMessageModel
    from t1d_analytics.chat_service import build_sql_prompt_with_history

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

    prompt = build_sql_prompt_with_history(
        "schema text", "Show only females", history=history
    )
    assert "Prior Conversation History:" in prompt
    assert "Show me all patients (SQL: SELECT * FROM patients)" in prompt
    assert "Now filter by age < 18" in prompt

    prompt_empty = build_sql_prompt_with_history(
        "schema text", "Show patients", history=[]
    )
    assert "Prior Conversation History:" not in prompt_empty

    mock_llm_res = TranslationResult(
        full_response="Explanation",
        sql_query="SELECT * FROM users WHERE age < 18",
    )
    mocker.patch(
        "t1d_analytics.api_dependencies.translate_nl_to_sql", return_value=mock_llm_res
    )
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
        stream_resp.read()


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
        res_mp = generate_sql_from_nl(
            mock_db, "Count users", model_name="openai/gpt-4o"
        )
        assert "SELECT count(*) FROM users" in res_mp.sql_query


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
        "t1d_analytics.api_dependencies.stream_llm_completion_tokens",
        return_value=iter(["tok1", "tok2"]),
    ):
        gen = stream_chat_events(chat_req_nl, req=req_disc_during)
        with pytest.raises(StopAsyncIteration):
            await gen.__anext__()


def test_chat_endpoints_with_provider_headers(mock_db: str) -> None:
    """Test /api/chat and /api/chat/stream forward x-provider and x-provider-api-key headers to LLM translation."""
    with patch(
        "t1d_analytics.api_dependencies.translate_nl_to_sql",
        return_value=TranslationResult(full_response="answer", sql_query="SELECT 1"),
    ) as mock_gen:
        resp = client.post(
            "/api/chat",
            json={"message": "Count users", "db_path": mock_db},
            headers={"x-provider": "openai", "x-provider-api-key": "sk-header-test"},
        )
        assert resp.status_code == 200
        mock_gen.assert_called_once()
        _, gen_kwargs = mock_gen.call_args
        assert gen_kwargs["provider"] == "openai"
        assert gen_kwargs["api_key"] == "sk-header-test"

    with patch(
        "t1d_analytics.api_dependencies.stream_llm_completion_tokens",
        return_value=iter(["Done"]),
    ) as mock_stream:
        resp_stream = client.post(
            "/api/chat/stream",
            json={"message": "Count users", "db_path": mock_db},
            headers={"x-provider": "anthropic", "x-provider-api-key": "sk-ant-header"},
        )
        assert resp_stream.status_code == 200
        assert "Done" in resp_stream.text
        mock_stream.assert_called_once()
        _, stream_kwargs = mock_stream.call_args
        assert stream_kwargs["provider"] == "anthropic"
        assert stream_kwargs["api_key"] == "sk-ant-header"


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
        res_direct = generate_sql_from_nl(
            mock_db,
            "Count users",
            model_name="openai/gpt-4o",
            api_key="direct_secret_key_123",
        )
        assert res_direct.sql_query == "SELECT 1;"
        mock_create.assert_called_with("openai", api_key="direct_secret_key_123")
        assert os.environ.get("OPENAI_API_KEY") is None


def test_stream_chat_events_nl_without_sql(mock_db: str) -> None:
    """Test stream_chat_events when LLM response contains plain answer without SQL code blocks."""
    with patch(
        "t1d_analytics.api_dependencies.stream_llm_completion_tokens",
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
    from t1d_analytics.api_schemas import ChatMessageModel
    from t1d_analytics.chat_service import build_sql_prompt_with_history

    history = [
        ChatMessageModel(
            role="user" if i % 2 == 0 else "assistant",
            content=f"Turn message {i}",
            sqlQuery=f"SELECT {i}" if i % 2 == 1 else None,
        )
        for i in range(60)
    ]
    prompt = build_sql_prompt_with_history(
        "schema text", "Final query", history=history
    )
    assert "Turn message 0" not in prompt
    assert "Turn message 40" not in prompt
    assert "Turn message 50" not in prompt
    assert "Turn message 58" in prompt
    assert "Turn message 59" in prompt
    assert "Final query" in prompt


def test_malformed_prior_assistant_responses() -> None:
    """Test SQL extraction robustly handles unclosed code fences and malformed markdown."""
    from t1d_analytics.sql_utils import extract_sql_from_response

    malformed_unclosed = """Here is the query:
```sql
SELECT * FROM patients WHERE id = 1"""
    sql1 = extract_sql_from_response(malformed_unclosed)
    assert "SELECT * FROM patients" in sql1

    mixed_fences = """Some thoughts...
```
SELECT 1;
```
And another:
```sql
SELECT 2;
```"""
    sql2 = extract_sql_from_response(mixed_fences)
    assert "SELECT 2" in sql2 or "SELECT 1" in sql2

    plain_select = """You can run this query:
SELECT id, glucose FROM cgm LIMIT 10;"""
    sql3 = extract_sql_from_response(plain_select)
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
        "t1d_analytics.api_dependencies.translate_nl_to_sql",
        return_value=TranslationResult(
            full_response="""```sql
SELECT * FROM tblaptsummary
```""",
            sql_query="SELECT * FROM tblaptsummary",
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
        "t1d_analytics.api_dependencies.stream_llm_completion_tokens",
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
        _is_recoverable_local_ollama_crash(
            "ollama",
            RuntimeError(
                "Failed to connect to Ollama. Please check that Ollama is downloaded, running and accessible. https://ollama.com/download"
            ),
        )
        is True
    )
    assert (
        _is_recoverable_local_ollama_crash("ollama", RuntimeError("generic error"))
        is False
    )

    # 1. Empty schema map + TranslationResult contract
    tr_res = synthesize_fallback_clinical_sql("", "How many patients?")
    assert isinstance(tr_res, TranslationResult)
    assert tr_res.used_fallback is True
    assert tr_res.is_rule_based_sql is True
    assert tr_res.generation_mode == "rule_based_fallback"
    assert 'SELECT COUNT(*) AS count FROM "patients";' in tr_res.sql_query
    assert "```sql" in tr_res.full_response

    # 2. Schema with matching table and column words + "first 5"
    schema_text = (
        "Table: demographics\nColumns: patient_id (BIGINT), age (DOUBLE), gender (VARCHAR)\n\n"
        "Table: cgm_readings\nColumns: subject_id (BIGINT), glucose_value (DOUBLE)\n"
    )
    res_match = synthesize_fallback_clinical_sql(
        schema_text, "Show me the first 5 patients in the demographics table age"
    )
    assert res_match.sql_query == 'SELECT * FROM "demographics" LIMIT 5;'

    # 3. Schema with zero matching words -> falls back to preferred table ("patients")
    schema_pref = (
        "Table: other_tbl\nColumns: x (INT)\n\nTable: patients\nColumns: id (INT)\n"
    )
    res_pref = synthesize_fallback_clinical_sql(schema_pref, "unrelated question")
    assert res_pref.sql_query == 'SELECT * FROM "patients" LIMIT 10;'

    # 4. Schema with no preferred table and zero matching words -> first table in schema
    schema_first = "Table: custom_alpha\nColumns: a (INT)\n\nTable: custom_beta\nColumns: b (INT)\n"
    res_first = synthesize_fallback_clinical_sql(schema_first, "unrelated question")
    assert res_first.sql_query == 'SELECT * FROM "custom_alpha" LIMIT 10;'

    # 4b. Table 1 fallback SQL synthesis with and without grouping column
    schema_t1 = (
        "Table: patients\n"
        "Columns: subject_id (BIGINT), age (BIGINT), TxGroup (VARCHAR), hba1c (DOUBLE), tir (DOUBLE)\n"
    )
    res_t1_grouped = synthesize_fallback_clinical_sql(
        schema_t1, "Generate a Table 1 baseline characteristics summary"
    )
    assert 'GROUP BY "TxGroup"' in res_t1_grouped.sql_query
    assert "mean_age_yr" in res_t1_grouped.sql_query
    assert "mean_hba1c_pct" in res_t1_grouped.sql_query

    res_t1_ungrouped = synthesize_fallback_clinical_sql(
        schema_first, "Create a cohort summary table"
    )
    assert (
        res_t1_ungrouped.sql_query
        == 'SELECT COUNT(*) AS n_participants FROM "custom_alpha";'
    )

    # 4c. _resolve_db_provenance_flags error fallback when duckdb connection fails
    from t1d_analytics.chat_service import _resolve_db_provenance_flags

    prov_err, is_synth_err = _resolve_db_provenance_flags(
        "nonexistent.duckdb",
        connect_fn=MagicMock(side_effect=RuntimeError("cannot open")),
    )
    assert prov_err == "synthetic_seed"
    assert is_synth_err is True

    # 5. translate_nl_to_sql fallback on Ollama OOM crash and Failed to connect to Ollama
    with patch(
        "t1d_analytics.chat_service.complete_llm_prompt",
        side_effect=RuntimeError(
            "llama-server process has terminated: unable to allocate AMX buffer"
        ),
    ):
        fb_res = translate_nl_to_sql(
            mock_db, "Show me the first 5 users", model_name="gemma4:latest"
        )
        assert fb_res.sql_query == 'SELECT * FROM "users" LIMIT 5;'
        assert fb_res.sql_query in fb_res.full_response

    with patch(
        "t1d_analytics.chat_service.complete_llm_prompt",
        side_effect=RuntimeError(
            "Failed to connect to Ollama. Please check that Ollama is downloaded, running and accessible. https://ollama.com/download"
        ),
    ):
        fb_conn_res = translate_nl_to_sql(
            mock_db, "Show me the first 5 users", model_name="custom-local:7b"
        )
        assert fb_conn_res.sql_query == 'SELECT * FROM "users" LIMIT 5;'
        assert fb_conn_res.sql_query in fb_conn_res.full_response

    # 5b. _resolve_and_select_local_ollama_model and retry-once with lightweight fallback model
    from t1d_analytics.api_schemas import ModelInfo
    from t1d_analytics.chat_service import (
        _probe_installed_ollama_names,
        _resolve_and_select_local_ollama_model,
        stream_llm_completion_tokens,
    )

    with patch(
        "t1d_analytics.chat_service.probe_ollama_tags_and_version",
        return_value=([ModelInfo(name="gemma:2b")], "0.5.0"),
    ):
        assert _probe_installed_ollama_names() == ["gemma:2b"]

    p_empty, fb_empty = _resolve_and_select_local_ollama_model(
        "gemma:2b",
        probe_fn=MagicMock(side_effect=RuntimeError("offline")),
    )
    assert p_empty == "gemma:2b"
    assert fb_empty is None

    p_resolved, fb_resolved = _resolve_and_select_local_ollama_model(
        "gemma2:27b",
        probe_fn=lambda: ["gemma2:27b", "gemma:2b", "gemma4:latest"],
    )
    assert p_resolved == "gemma2:27b"
    assert fb_resolved == "gemma:2b"

    p_unmatched, fb_unmatched = _resolve_and_select_local_ollama_model(
        "missing:model",
        probe_fn=lambda: ["gemma2:27b", "gemma:2b"],
    )
    assert p_unmatched == "gemma:2b"
    assert fb_unmatched is None

    with (
        patch(
            "t1d_analytics.chat_service._resolve_and_select_local_ollama_model",
            return_value=("gemma2:27b", "gemma:2b"),
        ),
        patch(
            "t1d_analytics.chat_service.complete_llm_prompt",
            side_effect=[
                RuntimeError("Request timed out on gemma2:27b"),
                "```sql\nSELECT COUNT(*) FROM users;\n```",
            ],
        ) as mock_comp_retry,
    ):
        retry_res = translate_nl_to_sql(
            mock_db, "Count users", model_name="ollama/gemma2:27b"
        )
        assert retry_res.sql_query == "SELECT COUNT(*) FROM users;"
        assert "SELECT COUNT(*)" in retry_res.full_response
        assert mock_comp_retry.call_count == 2

    mock_retry_transport = MagicMock()
    mock_retry_transport.completion.side_effect = [
        RuntimeError("stream failed"),
        RuntimeError("non-stream primary failed"),
        "SELECT 42;",
    ]
    with (
        patch(
            "t1d_analytics.chat_service._resolve_and_select_local_ollama_model",
            return_value=("gemma2:27b", "gemma:2b"),
        ),
        patch(
            "t1d_analytics.chat_service.create_any_llm_transport",
            return_value=mock_retry_transport,
        ),
    ):
        retry_tokens = list(
            stream_llm_completion_tokens("Count users", "ollama/gemma2:27b")
        )
        assert retry_tokens == ["SELECT 42;"]

    mock_fail_transport = MagicMock()
    mock_fail_transport.completion.side_effect = [
        RuntimeError("stream failed"),
        RuntimeError("non-stream failed without fallback"),
    ]
    with (
        patch(
            "t1d_analytics.chat_service._resolve_and_select_local_ollama_model",
            return_value=("gemma:2b", None),
        ),
        patch(
            "t1d_analytics.chat_service.create_any_llm_transport",
            return_value=mock_fail_transport,
        ),
    ):
        with pytest.raises(RuntimeError, match="non-stream failed without fallback"):
            list(stream_llm_completion_tokens("Count users", "ollama/gemma:2b"))

    # 6. /api/chat/stream recovers when stream_fn raises a local Ollama OOM / sync-in-async crash
    with patch(
        "t1d_analytics.api_dependencies.stream_llm_completion_tokens",
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

    with patch(
        "t1d_analytics.api_dependencies.stream_llm_completion_tokens",
        side_effect=RuntimeError(
            "Failed to connect to Ollama. Please check that Ollama is downloaded, running and accessible. https://ollama.com/download"
        ),
    ):
        resp_conn_stream = client.post(
            "/api/chat/stream",
            json={
                "message": "Show me the first 5 users",
                "model": "gemma:2b",
                "db_path": mock_db,
            },
        )
        assert resp_conn_stream.status_code == 200
        assert "sql_generated" in resp_conn_stream.text
        assert 'SELECT * FROM \\"users\\" LIMIT 5;' in resp_conn_stream.text

    # 7. is_likely_sql_statement and _resolve_sql_mode_query in batch /api/chat and /api/chat/stream
    from t1d_analytics.chat_service import is_likely_sql_statement
    from t1d_analytics.sql_fallback_synthesizer import _lookup_table_1_curated_sql
    from t1d_analytics.training_data import TABLE_1_BENCHMARK_SUITE

    assert _lookup_table_1_curated_sql("unknown random query") is None
    t1_q = TABLE_1_BENCHMARK_SUITE[0]["question"]
    assert _lookup_table_1_curated_sql(t1_q) == TABLE_1_BENCHMARK_SUITE[0]["gold_sql"]

    assert is_likely_sql_statement("   ") is False
    assert is_likely_sql_statement("SELECT * FROM users") is True
    assert is_likely_sql_statement("SHOW TABLES") is True
    assert is_likely_sql_statement("SHOW ALL TABLES") is True
    assert is_likely_sql_statement("Show all users") is False
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


def test_eval_nl_to_sql_endpoint_and_chat_gold_comparison(mock_db: str) -> None:
    """Test POST /api/eval/nl-to-sql and gold_comparison enrichment in /api/chat and /api/chat/stream."""
    # 1. Empty question returns 400
    empty_resp = client.post(
        "/api/eval/nl-to-sql",
        json={"question": "   ", "model": "gemma:2b", "db_path": mock_db},
    )
    assert empty_resp.status_code == 400

    # 2. Successful eval with explicit gold_sql and header overrides
    with patch(
        "t1d_analytics.api_dependencies.translate_nl_to_sql",
        return_value=TranslationResult(
            full_response="```sql\nSELECT id, name FROM users;\n```",
            sql_query="SELECT id, name FROM users;",
        ),
    ):
        eval_resp = client.post(
            "/api/eval/nl-to-sql",
            headers={
                "X-Provider": "ollama",
                "X-Provider-Api-Key": "test-key",
            },
            json={
                "question": "List all users",
                "gold_sql": "SELECT id, name FROM users",
                "model": "gemma:2b",
                "db_path": mock_db,
            },
        )
        assert eval_resp.status_code == 200
        eval_data = eval_resp.json()
        assert eval_data["sqlQuery"] == "SELECT id, name FROM users;"
        assert eval_data["gold_comparison"] is not None
        assert eval_data["gold_comparison"]["exact_match"] is True
        assert eval_data["gold_comparison"]["execution_accuracy"] is True
        assert eval_data["gold_comparison"]["column_overlap_pct"] == 100.0

    # 3. Eval when generated SQL fails execution on DuckDB read-only cursor
    with patch(
        "t1d_analytics.api_dependencies.translate_nl_to_sql",
        return_value=TranslationResult(
            full_response="```sql\nSELECT * FROM bad_table;\n```",
            sql_query="SELECT * FROM bad_table;",
        ),
    ):
        eval_bad_sql = client.post(
            "/api/eval/nl-to-sql",
            json={
                "question": "List all users",
                "gold_sql": "SELECT id, name FROM users",
                "model": "gemma:2b",
                "provider": "ollama",
                "api_key": "explicit",
                "db_path": mock_db,
            },
        )
        assert eval_bad_sql.status_code == 200
        bad_data = eval_bad_sql.json()
        assert bad_data["error"] is not None
        assert bad_data["gold_comparison"]["syntax_error"] is True

    # 4. Eval when LLM returns empty SQL query (no SQL generated)
    with patch(
        "t1d_analytics.api_dependencies.translate_nl_to_sql",
        return_value=TranslationResult(full_response="No SQL generated", sql_query=""),
    ):
        eval_no_sql = client.post(
            "/api/eval/nl-to-sql",
            json={
                "question": "Hello",
                "gold_sql": "SELECT id FROM users",
                "model": "gemma:2b",
                "db_path": mock_db,
            },
        )
        assert eval_no_sql.status_code == 200
        assert eval_no_sql.json()["gold_comparison"]["syntax_error"] is True

    # 5. Eval when translate_fn raises RuntimeError
    with patch(
        "t1d_analytics.api_dependencies.translate_nl_to_sql",
        side_effect=RuntimeError("LLM transport failure"),
    ):
        eval_err = client.post(
            "/api/eval/nl-to-sql",
            json={
                "question": "List users",
                "model": "gemma:2b",
                "db_path": mock_db,
            },
        )
        assert eval_err.status_code == 200
        assert eval_err.json()["content"] == "backend.errorNlpTranslation"
        assert eval_err.json()["error"] is not None

    # 6. /api/chat and /api/chat/stream with gold_sql populate gold_comparison
    with patch(
        "t1d_analytics.api_dependencies.translate_nl_to_sql",
        return_value=TranslationResult(
            full_response="```sql\nSELECT * FROM users;\n```",
            sql_query="SELECT * FROM users;",
        ),
    ):
        chat_gold = client.post(
            "/api/chat",
            json={
                "message": "Get all users",
                "model": "gemma:2b",
                "db_path": mock_db,
                "gold_sql": "SELECT * FROM users",
            },
        )
        assert chat_gold.status_code == 200
        assert chat_gold.json()["gold_comparison"]["execution_accuracy"] is True

    with patch(
        "t1d_analytics.api_dependencies.stream_llm_completion_tokens",
        return_value=iter(["```sql\nSELECT * FROM users;\n```"]),
    ):
        stream_gold = client.post(
            "/api/chat/stream",
            json={
                "message": "Get all users",
                "model": "gemma:2b",
                "db_path": mock_db,
                "gold_sql": "SELECT * FROM users",
            },
        )
        assert stream_gold.status_code == 200
        assert '"gold_comparison":' in stream_gold.text
        assert '"execution_accuracy": true' in stream_gold.text


def test_format_llm_failure_reason_and_chat_response_to_dict() -> None:
    """Test format_llm_failure_reason across all exception/string branches and ChatResponse.to_dict()."""
    import email.message
    import urllib.error

    from t1d_analytics.api_schemas import ApiError, ChatResponse, SqlGoldComparisonModel
    from t1d_analytics.errors import LlmProviderError
    from t1d_analytics.sql_fallback_synthesizer import (
        extract_fallback_error_from_text,
        format_llm_failure_reason,
        is_fallback_response_text,
        synthesize_fallback_clinical_sql,
    )

    assert is_fallback_response_text("SELECT 1;") is False
    assert (
        is_fallback_response_text("Local LLM inference was unavailable; synthesized")
        is True
    )

    assert format_llm_failure_reason(None) == (None, None)
    assert format_llm_failure_reason("   ") == (None, None)
    assert format_llm_failure_reason("Model returned empty response") == (
        "Model returned empty response",
        "backend.llmInvalidSqlOutput",
    )

    t1d_err = LlmProviderError(params={"error": "Missing API key for provider"})
    msg_t1d, code_t1d = format_llm_failure_reason(
        t1d_err, model_name="gpt-4o", provider="openai"
    )
    assert msg_t1d == "Missing API key for provider"
    assert code_t1d == t1d_err.error_code

    t1d_empty_param = LlmProviderError(params={})
    msg_t1d_empty, _ = format_llm_failure_reason(t1d_empty_param)
    assert msg_t1d_empty is not None

    hdrs = email.message.Message()
    http_err = urllib.error.HTTPError(
        url="http://127.0.0.1:11434/api/chat",
        code=404,
        msg="Not Found",
        hdrs=hdrs,
        fp=None,
    )
    msg_http, code_http = format_llm_failure_reason(
        http_err, model_name="gemma:2b", provider="ollama"
    )
    assert msg_http == "HTTP 404 Not Found (ollama) for model 'gemma:2b'"
    assert code_http == "backend.llmHttpError"

    url_err = urllib.error.URLError("Connection refused")
    msg_url, code_url = format_llm_failure_reason(
        url_err, model_name="gemma:2b", provider="ollama"
    )
    assert (
        msg_url == "Connection failed (ollama) for model 'gemma:2b': Connection refused"
    )
    assert code_url == "backend.llmConnectionError"

    timeout_err = TimeoutError("Read timed out after 45s")
    msg_to, code_to = format_llm_failure_reason(
        timeout_err, model_name="gemma:2b", provider="ollama"
    )
    assert "Inference timed out" in (msg_to or "")
    assert code_to == "backend.llmTimeoutError"

    generic_err = ValueError("Unexpected token sk-secret12345678 in response")
    msg_gen, code_gen = format_llm_failure_reason(
        generic_err, model_name="gemma:2b", provider="ollama"
    )
    assert "[REDACTED]" in (msg_gen or "")
    assert code_gen == "backend.llmTranslationError"

    # synthesize_fallback_clinical_sql with include_diagnostic_note=True and llm_error=None
    fb_no_err = synthesize_fallback_clinical_sql(
        "Table: users\nColumns: id (INT)\n",
        "Show users",
        include_diagnostic_note=True,
        llm_error=None,
    )
    assert "Note: Local LLM inference was unavailable;" in fb_no_err.full_response
    assert fb_no_err.llm_error is None
    assert extract_fallback_error_from_text(fb_no_err.full_response) is None

    fb_with_err = synthesize_fallback_clinical_sql(
        "Table: users\nColumns: id (INT)\n",
        "Show users",
        include_diagnostic_note=True,
        llm_error="Connection refused",
        llm_error_code=None,
    )
    assert "(Connection refused)" in fb_with_err.full_response
    assert fb_with_err.llm_error == "Connection refused"
    assert fb_with_err.llm_error_code == "backend.llmTranslationError"
    assert (
        extract_fallback_error_from_text(fb_with_err.full_response)
        == "Connection refused"
    )

    # ChatResponse.to_dict() with and without error/gold_comparison
    resp_plain = ChatResponse(
        content="ok",
        sqlQuery="SELECT 1;",
        sqlResult=[{"a": 1}],
        is_rule_based_sql=True,
        generation_mode="rule_based_fallback",
        llm_error="Connection refused",
        llm_error_code="backend.llmConnectionError",
    )
    d_plain = resp_plain.to_dict()
    assert d_plain["llm_error"] == "Connection refused"
    assert d_plain["error"] is None
    assert d_plain["gold_comparison"] is None

    resp_full = ChatResponse(
        content="err",
        error=ApiError(error_code="backend.sqlError", params={"error": "syntax"}),
        gold_comparison=SqlGoldComparisonModel(
            gold_sql="SELECT 1",
            pred_sql="SELECT 2",
            exact_match=False,
            execution_accuracy=False,
            syntax_error=True,
            gold_row_count=1,
            pred_row_count=0,
            gold_columns=["a"],
            pred_columns=[],
            column_overlap_pct=0.0,
            gold_rows_preview=[{"a": 1}],
            error_message="syntax",
        ),
    )
    d_full = resp_full.to_dict()
    assert isinstance(d_full["error"], dict)
    assert isinstance(d_full["gold_comparison"], dict)
