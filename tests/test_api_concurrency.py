"""Concurrent multi-tenant API, session mutation, and thread safety tests."""

import concurrent.futures
import sys
import threading
import types
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

if "any_llm" not in sys.modules:
    sys.modules["any_llm"] = types.ModuleType("any_llm")
    setattr(sys.modules["any_llm"], "AnyLLM", MagicMock())

import duckdb
from fastapi.testclient import TestClient

from t1d_analytics.api import app

client = TestClient(app)


def test_concurrent_chat_api_key_isolation(tmp_path: Path) -> None:
    """Test 20 parallel chat requests with distinct API keys verify zero cross-tenant key leakage."""
    db_file = tmp_path / "concurrent_chat.duckdb"
    conn = duckdb.connect(str(db_file))
    conn.execute("CREATE TABLE users (id INT, name VARCHAR)")
    conn.execute("INSERT INTO users VALUES (1, 'Alice'), (2, 'Bob')")
    conn.close()

    recorded_keys: Dict[int, str] = {}
    lock = threading.Lock()

    def mock_create(provider: str, **kwargs: Any) -> MagicMock:
        api_key = kwargs.get("api_key", "")
        mock_instance = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = (
            f"```sql\nSELECT COUNT(*) FROM users -- {api_key}\n```"
        )
        mock_instance.completion.return_value = MagicMock(choices=[mock_choice])
        return mock_instance

    with patch("any_llm.AnyLLM.create", side_effect=mock_create):

        def worker(thread_idx: int) -> bool:
            secret_key = f"tenant_secret_key_{thread_idx:04d}"
            resp = client.post(
                "/api/chat",
                json={
                    "message": "Count users",
                    "model": "openai/gpt-4o",
                    "db_path": str(db_file),
                },
                headers={
                    "x-provider": "openai",
                    "x-provider-api-key": secret_key,
                },
            )
            if resp.status_code != 200:
                return False
            data = resp.json()
            sql_query = data.get("sqlQuery") or data.get("content") or ""
            with lock:
                recorded_keys[thread_idx] = sql_query
            return secret_key in sql_query

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            results = list(executor.map(worker, range(20)))

    assert all(results)
    assert len(recorded_keys) == 20
    for idx, query in recorded_keys.items():
        expected_key = f"tenant_secret_key_{idx:04d}"
        assert expected_key in query


def test_concurrent_session_mutations(tmp_path: Path) -> None:
    """Test high-concurrency read/write operations against sessions.duckdb under thread contention."""
    db_file = tmp_path / "sessions_concurrent.duckdb"

    def session_worker(thread_idx: int) -> bool:
        session_id = f"session-tenant-{thread_idx}"
        # 1. Create or update session
        post_resp = client.post(
            "/api/sessions",
            params={"db_path": str(db_file)},
            json={
                "session_id": session_id,
                "title": f"Chat Title {thread_idx}",
                "messages": [{"role": "user", "content": f"Message from {thread_idx}"}],
            },
        )
        if post_resp.status_code != 200:
            return False

        # 2. Read back individual session
        get_resp = client.get(
            f"/api/sessions/{session_id}",
            params={"db_path": str(db_file)},
        )
        if get_resp.status_code != 200:
            return False
        session_data = get_resp.json()
        return bool(session_data["title"] == f"Chat Title {thread_idx}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        outcomes = list(executor.map(session_worker, range(20)))

    assert all(outcomes)

    # 3. List all sessions
    list_resp = client.get("/api/sessions", params={"db_path": str(db_file)})
    assert list_resp.status_code == 200
    all_sessions = list_resp.json()["sessions"]
    assert len(all_sessions) == 20


def test_concurrent_session_deletion_race(tmp_path: Path) -> None:
    """Test concurrent deletion requests against the same sessions do not corrupt database state."""
    db_file = tmp_path / "sessions_race.duckdb"

    # Pre-populate 5 sessions
    for i in range(5):
        client.post(
            "/api/sessions",
            params={"db_path": str(db_file)},
            json={
                "session_id": f"race-session-{i}",
                "title": f"Race {i}",
                "messages": [],
            },
        )

    def delete_worker(target_id: str) -> int:
        resp = client.delete(
            f"/api/sessions/{target_id}",
            params={"db_path": str(db_file)},
        )
        return resp.status_code

    targets: List[str] = [f"race-session-{i % 5}" for i in range(25)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        status_codes = list(executor.map(delete_worker, targets))

    # All deletions must return either 200 OK or 404 Not Found (if already deleted)
    for code in status_codes:
        assert code in (200, 404)

    # Verify all 5 sessions were deleted
    remaining = client.get("/api/sessions", params={"db_path": str(db_file)}).json()
    assert len(remaining["sessions"]) == 0
