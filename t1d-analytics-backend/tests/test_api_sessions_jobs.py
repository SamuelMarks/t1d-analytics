"""Tests for FastAPI session persistence, training job lifecycle, concurrency, rate limiting, and authentication."""

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import duckdb
import pytest
from fastapi.testclient import TestClient

from t1d_analytics.api import app

client = TestClient(app)


def test_sessions_crud_endpoints(tmp_path: Path) -> None:
    """Test conversation history session persistence CRUD endpoints."""
    from t1d_analytics.api import _get_sessions_db_conn

    test_db = str(tmp_path / "sessions_test.duckdb")
    p_sess = patch(
        "t1d_analytics.api._get_sessions_db_conn",
        lambda: _get_sessions_db_conn(test_db),
    )
    p_sess.start()
    try:
        _run_sessions_crud_assertions()
    finally:
        p_sess.stop()


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
    """Test sessions endpoints with corrupted messages_json and default db_path."""
    from t1d_analytics.api import _get_sessions_db_conn

    monkeypatch.delenv("T1D_SESSIONS_DB_PATH", raising=False)
    monkeypatch.chdir(tmp_path)
    conn = _get_sessions_db_conn()
    conn.execute(
        "INSERT INTO chat_sessions VALUES ('bad-json-id', 'Bad JSON', now(), now(), 'not-json{')"
    )
    conn.close()

    resp_list = client.get("/api/sessions")
    assert resp_list.status_code == 200
    item = next(
        s for s in resp_list.json()["sessions"] if s["session_id"] == "bad-json-id"
    )
    assert item["messages"] == []

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
    from t1d_analytics.api import _get_sessions_db_conn

    env_db = str(tmp_path / "env_sessions.duckdb")
    monkeypatch.setenv("T1D_SESSIONS_DB", env_db)
    conn = _get_sessions_db_conn()
    conn.close()
    assert Path(env_db).exists()


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
    """Test _get_sessions_db_conn immediately raises for non-lock IOException."""
    from t1d_analytics.api import _get_sessions_db_conn

    mocker.patch(
        "duckdb.connect",
        side_effect=duckdb.IOException("Permission denied"),
    )
    with pytest.raises(duckdb.IOException):
        _get_sessions_db_conn(str(tmp_path / "perm.duckdb"))


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
    """Test sliding-window rate limiting on execute_sql endpoint."""
    from t1d_analytics.api import RATE_LIMIT_STORE, check_rate_limit

    RATE_LIMIT_STORE.clear()
    ip = "192.168.1.100"

    assert check_rate_limit(ip, limit=3, window_seconds=60.0) is True
    assert check_rate_limit(ip, limit=3, window_seconds=60.0) is True
    assert check_rate_limit(ip, limit=3, window_seconds=60.0) is True
    assert check_rate_limit(ip, limit=3, window_seconds=60.0) is False

    RATE_LIMIT_STORE[ip] = [time.time() - 70.0]
    assert check_rate_limit(ip, limit=3, window_seconds=60.0) is True

    monkeypatch.setenv("T1D_DB_PATH", mock_db)
    with patch("t1d_analytics.api.check_rate_limit", return_value=False):
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

    from t1d_analytics.api import (
        RATE_LIMIT_STORE,
        DuckDbRateLimiter,
        MemoryRateLimiter,
        enforce_rate_limit,
        get_rate_limiter,
    )
    from t1d_analytics.errors import RateLimitExceededError

    mem_limiter = MemoryRateLimiter()
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
    duck_limiter = DuckDbRateLimiter(str(db_path))
    duck_limiter.reset()
    assert duck_limiter.get_retry_after("10.0.0.2", window_seconds=60.0) == 1
    assert duck_limiter.is_allowed("10.0.0.2", limit=2, window_seconds=60.0) is True
    assert duck_limiter.is_allowed("10.0.0.2", limit=2, window_seconds=60.0) is True
    assert duck_limiter.is_allowed("10.0.0.2", limit=2, window_seconds=60.0) is False
    assert duck_limiter.get_retry_after("10.0.0.2", window_seconds=60.0) >= 1
    duck_limiter.reset()
    assert duck_limiter.is_allowed("10.0.0.2", limit=2, window_seconds=60.0) is True

    monkeypatch.setenv("T1D_RATE_LIMITER_BACKEND", "duckdb")
    limiter_duck = get_rate_limiter()
    assert isinstance(limiter_duck, DuckDbRateLimiter)

    monkeypatch.setenv("T1D_RATE_LIMITER_BACKEND", "memory")
    limiter_mem = get_rate_limiter()
    assert isinstance(limiter_mem, MemoryRateLimiter)

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
