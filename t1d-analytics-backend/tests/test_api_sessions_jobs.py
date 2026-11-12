"""Tests for FastAPI session persistence, training job lifecycle, concurrency, rate limiting, and authentication."""

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import duckdb
import pytest
from fastapi.testclient import TestClient

from t1d_analytics.api import app

client = TestClient(app)


def test_sessions_crud_endpoints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test conversation history session persistence CRUD endpoints."""
    test_db = str(tmp_path / "sessions_test.duckdb")
    monkeypatch.setenv("T1D_SESSIONS_DB_PATH", test_db)
    _run_sessions_crud_assertions()


def _run_sessions_crud_assertions() -> None:
    """Execute assertions for session persistence CRUD endpoints."""
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

    resp_empty = client.post("/api/sessions", json={"messages": []})
    assert resp_empty.status_code == 200
    assert resp_empty.json()["title"] == "New Chat"

    resp_list = client.get("/api/sessions")
    assert resp_list.status_code == 200
    sessions = resp_list.json()["sessions"]
    assert len(sessions) == 2
    assert any(s["session_id"] == sess_id for s in sessions)

    resp_get = client.get(f"/api/sessions/{sess_id}")
    assert resp_get.status_code == 200
    assert resp_get.json()["session_id"] == sess_id

    resp_404 = client.get("/api/sessions/nonexistent_id")
    assert resp_404.status_code == 404

    resp_del = client.delete(f"/api/sessions/{sess_id}")
    assert resp_del.status_code == 200
    assert resp_del.json()["status"] == "deleted"

    resp_check = client.get("/api/sessions")
    assert not any(s["session_id"] == sess_id for s in resp_check.json()["sessions"])


def test_sessions_malformed_json_and_default_conn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test sessions endpoints with corrupted messages_json, legacy sum(VARCHAR) sanitization, and default db_path."""
    import json

    from t1d_analytics.seed import (
        sanitize_legacy_error_sessions,
        seed_sessions_database,
    )
    from t1d_analytics.session_store import (
        clear_initialized_session_dbs,
        open_sessions_db_conn,
    )

    clear_initialized_session_dbs()
    monkeypatch.delenv("T1D_SESSIONS_DB_PATH", raising=False)
    monkeypatch.chdir(tmp_path)
    conn = open_sessions_db_conn()
    conn.execute(
        "INSERT INTO chat_sessions VALUES ('bad-json-id', 'Bad JSON', now(), now(), 'not-json{')"
    )
    legacy_msgs = [
        {
            "role": "assistant",
            "content": "Binder Error: No function matches the given name and argument types 'sum(VARCHAR)'.",
        },
        {"role": "user", "content": "SELECT SUM(NumSevHypo) FROM tblaptsummary"},
        {
            "role": "assistant",
            "content": "Binder Error: No function matches the given name and argument types 'sum(VARCHAR)'.",
        },
        {"role": "user", "content": "SELECT SUM(NumSevHypo) FROM tblaptsummary"},
        {"role": "assistant", "content": "Executed SQL query."},
        {
            "role": "assistant",
            "content": "Binder Error: No function matches the given name and argument types 'sum(VARCHAR)'.",
        },
        {
            "role": "assistant",
            "content": "Some non-binder error message",
        },
        {"role": "user", "content": "SELECT AVG(AgeAsOfRandDt) FROM tblaptsummary"},
        {
            "role": "assistant",
            "content": "Binder Error: No function matches the given name and argument types 'avg(VARCHAR)'.",
        },
    ]
    conn.execute(
        "INSERT INTO chat_sessions VALUES ('legacy-err-id', 'Legacy Errors', now(), now(), ?)",
        [json.dumps(legacy_msgs)],
    )
    conn.execute(
        "INSERT INTO chat_sessions VALUES ('session-default-welcome', 'Welcome', now(), now(), ?)",
        [
            json.dumps(
                [
                    {
                        "role": "assistant",
                        "content": "Error details: Binder Error",
                        "isError": True,
                    }
                ]
            )
        ],
    )
    sanitized_cnt = sanitize_legacy_error_sessions(conn)
    assert sanitized_cnt == 2
    # Calling a second time has 0 changes
    assert sanitize_legacy_error_sessions(conn) == 0
    conn.close()

    assert seed_sessions_database(str(tmp_path / "data" / "sessions.duckdb")) == 0

    resp_list = client.get("/api/sessions")
    assert resp_list.status_code == 200
    item = next(
        s for s in resp_list.json()["sessions"] if s["session_id"] == "bad-json-id"
    )
    assert item["messages"] == []
    legacy_item = next(
        s for s in resp_list.json()["sessions"] if s["session_id"] == "legacy-err-id"
    )
    assert len(legacy_item["messages"]) == 5

    resp_get = client.get("/api/sessions/bad-json-id")
    assert resp_get.status_code == 200
    assert resp_get.json()["messages"] == []


def test_sessions_concurrent_access_and_locking() -> None:
    """Test concurrent multi-threaded requests on sessions endpoints."""
    import concurrent.futures

    def do_session_op(idx: int) -> int:
        """Execute a session create, list, or delete operation."""
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
    detail = resp.json()["detail"]
    assert detail["error_code"] == "backend.sessionNotFound"


def test_sessions_db_env_var(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test T1D_SESSIONS_DB environment variable controls session database path."""
    from t1d_analytics.session_store import open_sessions_db_conn

    env_db = str(tmp_path / "env_sessions.duckdb")
    monkeypatch.setenv("T1D_SESSIONS_DB", env_db)
    conn = open_sessions_db_conn()
    conn.close()
    assert Path(env_db).exists()


def test_sessions_db_path_traversal_and_sandboxing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify GET/POST/DELETE /api/sessions reject path traversal, /tmp escapes, symlinks, bad extensions, and missing parent dirs."""
    from t1d_analytics.db_security import validate_sessions_db_path
    from t1d_analytics.errors import InvalidDatabasePathError
    from t1d_analytics.session_store import open_sessions_db_conn

    untrusted_tmp_target = Path("/tmp/untrusted_t1d_audit_probe/outside.duckdb")
    traversal_target = "../../outside.duckdb"
    bad_ext_target = str(tmp_path / "sessions.txt")
    missing_parent_target = str(tmp_path / "missing_subdir" / "sessions.duckdb")

    for bad_path in (
        traversal_target,
        str(untrusted_tmp_target),
        "   ",
        bad_ext_target,
    ):
        resp_get = client.get("/api/sessions", params={"db_path": bad_path})
        assert resp_get.status_code == 400
        assert resp_get.json()["detail"]["error_code"] == "backend.invalidDatabasePath"

        resp_one = client.get("/api/sessions/s1", params={"db_path": bad_path})
        assert resp_one.status_code == 400

        resp_post = client.post(
            "/api/sessions",
            params={"db_path": bad_path},
            json={"messages": [{"role": "user", "content": "hi"}]},
        )
        assert resp_post.status_code == 400

        resp_del = client.delete("/api/sessions/s1", params={"db_path": bad_path})
        assert resp_del.status_code == 400

    assert not untrusted_tmp_target.parent.exists()

    # Symlink escaping allowed root must also fail containment
    allowed_sub = tmp_path / "allowed_sess"
    allowed_sub.mkdir()
    escape_link = allowed_sub / "escaped_link.duckdb"
    escape_link.symlink_to("/etc/passwd")
    with pytest.raises(InvalidDatabasePathError):
        validate_sessions_db_path(str(escape_link), allowed_dirs=[allowed_sub])

    # Explicit db_path inside allowed root whose parent directory does not exist must not auto-mkdir
    resp_missing_parent = client.post(
        "/api/sessions",
        params={"db_path": missing_parent_target},
        json={"messages": []},
    )
    assert resp_missing_parent.status_code == 400
    assert not Path(missing_parent_target).parent.exists()

    with pytest.raises(InvalidDatabasePathError):
        open_sessions_db_conn(db_path=missing_parent_target)


def test_sessions_db_lock_retry(tmp_path: Path, mocker: MagicMock) -> None:
    """Test _get_sessions_db_conn retries upon IOException lock contention."""
    from t1d_analytics.session_store import open_sessions_db_conn

    real_conn = duckdb.connect(str(tmp_path / "retry.duckdb"))
    attempts: list[int] = []
    sleep_delays: list[float] = []

    def mock_connect(
        database: str = ":memory:", read_only: bool = False
    ) -> duckdb.DuckDBPyConnection:
        """Raise lock IOException twice before returning real_conn."""
        if len(attempts) < 2:
            attempts.append(1)
            raise duckdb.IOException("Could not set lock on file")
        return real_conn

    mocker.patch("duckdb.connect", side_effect=mock_connect)
    conn = open_sessions_db_conn(
        str(tmp_path / "retry.duckdb"),
        sleep_fn=lambda d: sleep_delays.append(d),
    )
    conn.close()
    assert len(attempts) == 2
    assert len(sleep_delays) == 2


def test_sessions_db_lock_failure(tmp_path: Path, mocker: MagicMock) -> None:
    """Test _get_sessions_db_conn raises if lock cannot be acquired after retries."""
    from t1d_analytics.session_store import open_sessions_db_conn

    mocker.patch(
        "duckdb.connect",
        side_effect=duckdb.IOException("Fatal lock error"),
    )
    with pytest.raises(duckdb.IOException):
        open_sessions_db_conn(
            str(tmp_path / "fail.duckdb"),
            sleep_fn=lambda _: None,
        )


def test_sessions_db_non_lock_error(tmp_path: Path, mocker: MagicMock) -> None:
    """Test open_sessions_db_conn immediately raises for non-lock IOException."""
    from t1d_analytics.session_store import open_sessions_db_conn

    mocker.patch(
        "duckdb.connect",
        side_effect=duckdb.IOException("Permission denied"),
    )
    with pytest.raises(duckdb.IOException):
        open_sessions_db_conn(str(tmp_path / "perm.duckdb"))


def test_concurrency_and_lock_contention(mock_db: str) -> None:
    """TEST-API-01: Execute 50 parallel requests to /api/execute-sql and /api/chat."""
    import concurrent.futures

    def make_sql_request(idx: int) -> int:
        """Send a concurrent SQL execution request."""
        res = client.post(
            "/api/execute-sql",
            json={
                "query": "SELECT id, name FROM users WHERE id = 1",
                "db_path": mock_db,
            },
        )
        return res.status_code

    def make_chat_sql_request(idx: int) -> int:
        """Send a concurrent chat SQL execution request."""
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


def test_api_auth_middleware(monkeypatch: pytest.MonkeyPatch, mock_db: str) -> None:
    """Test API authentication middleware with valid, invalid, and missing API keys."""
    monkeypatch.setenv("T1D_API_KEY", "secret-test-key-123")
    monkeypatch.setenv("T1D_DB_PATH", mock_db)

    resp_health = client.get("/api/health")
    assert resp_health.status_code == 200

    resp_unauth = client.get("/api/schema")
    assert resp_unauth.status_code == 401
    assert resp_unauth.json()["detail"]["error_code"] == "backend.unauthorized"

    resp_wrong = client.get("/api/schema", headers={"X-API-Key": "wrong-key"})
    assert resp_wrong.status_code == 401

    resp_ok_header = client.get(
        "/api/schema", headers={"X-API-Key": "secret-test-key-123"}
    )
    assert resp_ok_header.status_code == 200

    resp_ok_bearer = client.get(
        "/api/schema", headers={"Authorization": "Bearer secret-test-key-123"}
    )
    assert resp_ok_bearer.status_code == 200


def test_verify_api_auth_with_bearer_token(
    monkeypatch: pytest.MonkeyPatch, mock_db: str
) -> None:
    """Test API authentication middleware with T1D_API_BEARER_TOKEN."""
    monkeypatch.delenv("T1D_API_KEY", raising=False)
    monkeypatch.delenv("T1D_API_BEARER_TOKEN", raising=False)
    resp = client.get(f"/api/table/users?db_path={mock_db}")
    assert resp.status_code == 200

    monkeypatch.setenv("T1D_API_BEARER_TOKEN", "secret-bearer-999")
    resp_unauth = client.get(f"/api/table/users?db_path={mock_db}")
    assert resp_unauth.status_code == 401

    resp_bad = client.get(
        f"/api/table/users?db_path={mock_db}",
        headers={"Authorization": "Bearer wrong-token"},
    )
    assert resp_bad.status_code == 401

    resp_ok = client.get(
        f"/api/table/users?db_path={mock_db}",
        headers={"Authorization": "Bearer secret-bearer-999"},
    )
    assert resp_ok.status_code == 200

    resp_ok_hdr = client.get(
        f"/api/table/users?db_path={mock_db}",
        headers={"x-api-key": "secret-bearer-999"},
    )
    assert resp_ok_hdr.status_code == 200

    monkeypatch.setenv("T1D_DB_PATH", mock_db)
    resp_health = client.get("/api/health")
    assert resp_health.status_code == 200


def test_sliding_window_rate_limiting(
    monkeypatch: pytest.MonkeyPatch, mock_db: str
) -> None:
    """Test sliding-window rate limiting across middleware route tiers."""
    from t1d_analytics.rate_limiter import RATE_LIMIT_STORE, check_rate_limit

    RATE_LIMIT_STORE.clear()
    ip = "192.168.1.100"

    assert check_rate_limit(ip, limit=3, window_seconds=60.0) is True
    assert check_rate_limit(ip, limit=3, window_seconds=60.0) is True
    assert check_rate_limit(ip, limit=3, window_seconds=60.0) is True
    assert check_rate_limit(ip, limit=3, window_seconds=60.0) is False

    RATE_LIMIT_STORE[ip] = [time.time() - 70.0]
    assert check_rate_limit(ip, limit=3, window_seconds=60.0) is True

    monkeypatch.setenv("T1D_DB_PATH", mock_db)
    with patch.object(app.state.rate_limiter, "is_allowed", return_value=False):
        resp_sql = client.post("/api/execute-sql", json={"query": "SELECT 1"})
        assert resp_sql.status_code == 429
        assert resp_sql.json()["detail"]["error_code"] == "backend.rateLimitExceeded"

        resp_chat = client.post("/api/chat", json={"message": "hello", "model": "sql"})
        assert resp_chat.status_code == 429

        resp_stream = client.post(
            "/api/chat/stream", json={"message": "hello", "model": "sql"}
        )
        assert resp_stream.status_code == 429

        assert client.get("/api/sessions").status_code == 429
        assert client.get(f"/api/table/users?db_path={mock_db}").status_code == 429
        assert (
            client.get(
                f"/api/export/csv?table_name=users&db_path={mock_db}"
            ).status_code
            == 429
        )
        assert client.get("/api/training/jobs").status_code == 429
        assert client.get("/api/status").status_code == 429
        assert client.get("/api/health").status_code == 200
    RATE_LIMIT_STORE.clear()


def test_rate_limiter_backends_and_retry_after(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test MemoryRateLimiter and DuckDbRateLimiter features, retry-after headers, and reset."""
    from concurrent.futures import ThreadPoolExecutor

    from t1d_analytics.errors import RateLimitExceededError
    from t1d_analytics.rate_limiter import (
        RATE_LIMIT_STORE,
        DuckDbRateLimiter,
        MemoryRateLimiter,
        close_all_rate_limiters,
        enforce_rate_limit,
        get_rate_limiter,
    )

    isolated_a = MemoryRateLimiter()
    isolated_b = MemoryRateLimiter()
    assert isolated_a._store is not RATE_LIMIT_STORE
    assert isolated_a._store is not isolated_b._store

    with MemoryRateLimiter(store=RATE_LIMIT_STORE) as mem_limiter:
        mem_limiter.reset()
        assert mem_limiter.get_retry_after("10.0.0.1", window_seconds=60.0) == 1
        assert mem_limiter.is_allowed("10.0.0.1", limit=1, window_seconds=60.0) is True
        assert mem_limiter.is_allowed("10.0.0.1", limit=1, window_seconds=60.0) is False
        assert mem_limiter.get_retry_after("10.0.0.1", window_seconds=60.0) >= 1
        RATE_LIMIT_STORE["stale.ip.1"] = [0.0]
        assert mem_limiter.is_allowed("10.0.0.2", limit=5, window_seconds=60.0) is True
        assert "stale.ip.1" not in RATE_LIMIT_STORE
        mem_limiter.reset()
        assert mem_limiter.is_allowed("10.0.0.1", limit=1, window_seconds=60.0) is True

    db_path = tmp_path / "rate_limits.duckdb"
    with DuckDbRateLimiter(str(db_path)) as duck_limiter:
        duck_limiter.reset()
        assert duck_limiter.get_retry_after("10.0.0.2", window_seconds=60.0) == 1
        assert duck_limiter.is_allowed("10.0.0.2", limit=2, window_seconds=60.0) is True
        assert duck_limiter.is_allowed("10.0.0.2", limit=2, window_seconds=60.0) is True
        assert (
            duck_limiter.is_allowed("10.0.0.2", limit=2, window_seconds=60.0) is False
        )
        assert duck_limiter.get_retry_after("10.0.0.2", window_seconds=60.0) >= 1
        duck_limiter.reset()
        assert duck_limiter.is_allowed("10.0.0.2", limit=2, window_seconds=60.0) is True

    monkeypatch.setenv("T1D_RATE_LIMITER_BACKEND", "duckdb")
    limiter_duck = get_rate_limiter()
    assert isinstance(limiter_duck, DuckDbRateLimiter)

    monkeypatch.setenv("T1D_RATE_LIMITER_BACKEND", "memory")
    limiter_mem = get_rate_limiter()
    assert isinstance(limiter_mem, MemoryRateLimiter)
    close_all_rate_limiters()
    limiter_mem = get_rate_limiter()

    mock_request = MagicMock()
    mock_request.client.host = "10.0.0.99"
    limiter_mem.reset()
    enforce_rate_limit(mock_request, limit=1, window_seconds=60.0)
    with pytest.raises(RateLimitExceededError) as exc_info:
        enforce_rate_limit(mock_request, limit=1, window_seconds=60.0)
    assert exc_info.value.status_code == 429
    assert (
        exc_info.value.headers is not None and "Retry-After" in exc_info.value.headers
    )

    concurrent_ip = "10.0.0.50"
    limiter_mem.reset()

    def send_req() -> bool:
        """Check rate limit allowance for concurrent_ip."""
        return limiter_mem.is_allowed(concurrent_ip, limit=10, window_seconds=60.0)

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: send_req(), range(30)))
    assert results.count(True) == 10
    assert results.count(False) == 20
    close_all_rate_limiters()


def test_training_job_endpoints(tmp_path: Path, mocker: MagicMock) -> None:
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

    resp_gpu = client.post("/api/training/jobs", json={**payload, "backend": "gpu"})
    assert resp_gpu.status_code == 200
    resp_tpu = client.post("/api/training/jobs", json={**payload, "backend": "tpu"})
    assert resp_tpu.status_code == 200

    resp_inv = client.post(
        "/api/training/jobs", json={**payload, "backend": "quantum-processor"}
    )
    assert resp_inv.status_code == 400

    resp_list = client.get("/api/training/jobs")
    assert resp_list.status_code == 200
    jobs = resp_list.json()["jobs"]
    assert any(j["job_id"] == job_id for j in jobs)

    resp_get = client.get(f"/api/training/jobs/{job_id}")
    assert resp_get.status_code == 200
    assert resp_get.json()["job_id"] == job_id

    resp_404 = client.get("/api/training/jobs/non-existent-job-id")
    assert resp_404.status_code == 404

    resp_cancel_404 = client.delete("/api/training/jobs/non-existent-job-id")
    assert resp_cancel_404.status_code == 404

    resp_cancel = client.delete(f"/api/training/jobs/{job_id}")
    assert resp_cancel.status_code == 200
    assert resp_cancel.json()["status"] in ["cancelled", "completed"]

    from t1d_analytics.api_dependencies import resolve_training_job_manager
    from t1d_analytics.session_store import (
        SessionStore,
        managed_sessions_db_conn,
        open_sessions_db_conn,
    )

    mgr = resolve_training_job_manager(app)
    pruned = mgr.prune_completed_jobs(max_retained=0)
    assert pruned >= 0

    sess_db = str(tmp_path / "managed_sess.duckdb")
    with managed_sessions_db_conn(sess_db) as m_conn:
        m_conn.execute("SELECT 1")
    default_store = SessionStore()
    assert default_store.list_sessions(db_path=sess_db).sessions == []

    from t1d_analytics.api_schemas import ChatMessageModel, CreateSessionRequest

    saved = default_store.upsert_session(
        CreateSessionRequest(
            session_id="norm-s1",
            title="Normalized Session",
            messages=[ChatMessageModel(role="user", content="Hello normalized")],
        ),
        db_path=sess_db,
    )
    assert len(saved.messages) == 1
    meta_only = default_store.list_sessions(db_path=sess_db, include_messages=False)
    assert len(meta_only.sessions) == 1
    assert meta_only.sessions[0].messages == []

    resp_meta_http = client.get(
        "/api/sessions",
        params={"db_path": sess_db, "include_messages": "false"},
    )
    assert resp_meta_http.status_code == 200
    assert resp_meta_http.json()["sessions"][0]["messages"] == []

    with managed_sessions_db_conn(sess_db) as m_conn:
        m_conn.execute(
            "INSERT INTO chat_messages VALUES ('norm-s1', 99, 'user', 'bad', '{bad-json')"
        )
    hydrated = default_store.get_session("norm-s1", db_path=sess_db)
    assert len(hydrated.messages) == 1
    listed_full = default_store.list_sessions(db_path=sess_db, include_messages=True)
    assert len(listed_full.sessions[0].messages) == 1

    from collections.abc import Iterator, Sequence
    from contextlib import contextmanager

    from t1d_analytics.session_store import SessionDbConnectionProtocol

    rollback_calls: list[str] = []

    class _FailingTxConn:
        """Connection double raising mid-transaction to verify ROLLBACK handling."""

        def __init__(self, fail_on_prefix: str) -> None:
            """Initialize with SQL prefix that triggers a failure."""
            self._fail_on = fail_on_prefix

        def execute(
            self,
            sql: str,
            _params: Sequence[str | int] | None = None,
        ) -> SessionDbConnectionProtocol:
            """Record SQL and fail on configured prefix."""
            rollback_calls.append(sql)
            if sql.startswith(self._fail_on):
                raise duckdb.Error("simulated tx failure")
            return self

        def fetchone(self) -> tuple[str | int | None, ...] | None:
            """Return non-empty existence row."""
            return (1,)

        def fetchall(self) -> list[tuple[str | int | None, ...]]:
            """Return empty row list."""
            return []

    @contextmanager
    def _failing_upsert_factory(
        _p: str | None,
    ) -> Iterator[SessionDbConnectionProtocol]:
        """Yield failing upsert connection double."""
        yield _FailingTxConn("UPDATE chat_sessions")

    @contextmanager
    def _failing_delete_factory(
        _p: str | None,
    ) -> Iterator[SessionDbConnectionProtocol]:
        """Yield failing delete connection double."""
        yield _FailingTxConn("DELETE FROM chat_messages")

    with pytest.raises(duckdb.Error, match="simulated tx failure"):
        SessionStore(conn_factory=_failing_upsert_factory).upsert_session(
            CreateSessionRequest(session_id="s-fail", title="Fail")
        )
    assert "ROLLBACK" in rollback_calls

    rollback_calls.clear()
    with pytest.raises(duckdb.Error, match="simulated tx failure"):
        SessionStore(conn_factory=_failing_delete_factory).delete_session("s-fail")
    assert "ROLLBACK" in rollback_calls

    bad_conn = MagicMock()
    bad_conn.execute.side_effect = duckdb.Error("CREATE TABLE failed")
    mocker.patch("duckdb.connect", return_value=bad_conn)
    with pytest.raises(duckdb.Error, match="CREATE TABLE failed"):
        open_sessions_db_conn(str(tmp_path / "bad_schema.duckdb"))
    bad_conn.close.assert_called_once()


def test_rate_limiter_abstract_stubs(tmp_path: Path) -> None:
    """Verify concrete rate limiters and LocalCpuRunner satisfy their base contracts."""
    from t1d_analytics.models import TrainingBackend, TrainingJobConfig
    from t1d_analytics.rate_limiter import BaseRateLimiter, MemoryRateLimiter
    from t1d_analytics.training_runner import LocalCpuRunner

    limiter = MemoryRateLimiter()
    assert isinstance(limiter, BaseRateLimiter)
    limiter.reset()
    assert limiter.is_allowed("127.0.0.1") is True
    assert limiter.get_retry_after("127.0.0.1") >= 1

    ds_file = tmp_path / "ds.jsonl"
    ds_file.write_text('{"prompt": "q", "sql": "SELECT 1"}\n', encoding="utf-8")
    cfg = TrainingJobConfig(
        model_name="gemma4-sql",
        backend=TrainingBackend.LOCAL_CPU,
        dataset_path=str(ds_file),
        output_dir=str(tmp_path / "out"),
        dry_run=True,
        use_tiny_model=True,
    )
    mock_tok = MagicMock()
    mock_tok.encode.return_value = [1, 2, 3]
    with patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tok):
        res = LocalCpuRunner().run_training(cfg)
    assert res["status"] == "dry_run_completed"


def test_job_managers_ttl_eviction_and_shutdown(tmp_path: Path) -> None:
    """Verify TTL expiration eviction and executor shutdown on TrainingJobManager and ChatTurnJobManager."""
    from t1d_analytics.api_schemas import ChatRequest, ChatResponse
    from t1d_analytics.job_manager import ChatTurnJobManager, TrainingJobManager
    from t1d_analytics.models import TrainingBackend, TrainingJobConfig
    from t1d_analytics.training_runner import get_training_runner

    now_val = [100.0]
    ds_file = tmp_path / "ttl_ds.jsonl"
    ds_file.write_text('{"prompt": "q", "sql": "SELECT 1"}\n', encoding="utf-8")

    train_mgr = TrainingJobManager(
        runner_resolver=get_training_runner,
        max_workers=1,
        max_retained=10,
        ttl_seconds=30.0,
        clock_fn=lambda: now_val[0],
    )
    cfg = TrainingJobConfig(
        model_name="gemma4-sql",
        backend=TrainingBackend.LOCAL_CPU,
        dataset_path=str(ds_file),
        output_dir=str(tmp_path / "ttl_out"),
        dry_run=True,
        use_tiny_model=True,
    )
    mock_tok = MagicMock()
    mock_tok.encode.return_value = [1, 2]
    with patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tok):
        jid = train_mgr.submit_job(cfg)
        train_mgr.wait_for_job(jid, timeout=5.0)
    now_val[0] = 200.0
    assert train_mgr.prune_completed_jobs() == 1
    assert train_mgr.get_job(jid) is None
    train_mgr.shutdown(wait=True)
    with patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tok):
        jid_after = train_mgr.submit_job(cfg)
        train_mgr.wait_for_job(jid_after, timeout=5.0)
    assert train_mgr.get_job(jid_after) is not None
    train_mgr.shutdown(wait=True)

    chat_mgr = ChatTurnJobManager(
        max_workers=1,
        max_retained=10,
        ttl_seconds=30.0,
        clock_fn=lambda: now_val[0],
    )
    ticket = chat_mgr.submit_chat_job(
        ChatRequest(message="SELECT 1", model="sql"),
        lambda _req, _prog: ChatResponse(content="ok"),
    )
    cjid = str(ticket["job_id"])
    chat_mgr.wait_for_chat_job(cjid, timeout=5.0)
    now_val[0] = 300.0
    assert chat_mgr.prune_completed_chat_jobs() == 1
    assert chat_mgr.get_chat_job(cjid) is None
    chat_mgr.shutdown(wait=True)
    ticket_after = chat_mgr.submit_chat_job(
        ChatRequest(message="SELECT 1", model="sql"),
        lambda _req, _prog: ChatResponse(content="ok2"),
    )
    cjid_after = str(ticket_after["job_id"])
    chat_mgr.wait_for_chat_job(cjid_after, timeout=5.0)
    assert chat_mgr.get_chat_job(cjid_after) is not None
    chat_mgr.shutdown(wait=True)
