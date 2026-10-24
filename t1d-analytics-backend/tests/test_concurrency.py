"""Concurrency and multi-threading contention tests for DuckDB."""

import concurrent.futures
import typing
from pathlib import Path
from unittest.mock import patch

import duckdb
import pytest
from fastapi.testclient import TestClient

from t1d_analytics.api import app, execute_sql
from t1d_analytics.sql_utils import execute_duckdb_write_with_retry


def test_concurrent_duckdb_reads(tmp_path: Path) -> None:
    """
    Test multiple threads executing concurrent read queries against DuckDB.

    Args:
    ----
        tmp_path: Pytest temporary directory fixture.

    """
    db_path = str(tmp_path / "concurrent_read.duckdb")
    conn = duckdb.connect(db_path)
    conn.execute("CREATE TABLE records (id INTEGER, metric DOUBLE)")
    for i in range(100):
        conn.execute(f"INSERT INTO records VALUES ({i}, {i * 1.5})")
    conn.close()

    def run_read(thread_id: int) -> int:
        """Execute a concurrent read query."""
        query = f"SELECT count(*) FROM records WHERE id >= {thread_id}"
        res = execute_sql(db_path, query)
        return len(res)

    workers = 10
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(run_read, i) for i in range(workers)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]

    assert len(results) == workers
    assert all(r == 1 for r in results)


def test_concurrent_api_session_writes(tmp_path: Path) -> None:
    """
    Test multiple concurrent requests saving sessions to verify locking resilience.

    Args:
    ----
        tmp_path: Pytest temporary directory fixture.

    """
    client = TestClient(app)
    db_path = str(tmp_path / "sessions_concurrent.duckdb")

    def create_session(idx: int) -> int:
        """Create a chat session via POST /api/sessions."""
        resp = client.post(
            f"/api/sessions?db_path={db_path}",
            json={
                "session_id": f"thread_sess_{idx}",
                "title": f"Session {idx}",
                "messages": [{"role": "user", "content": f"Query {idx}"}],
            },
        )
        return resp.status_code

    workers = 8
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(create_session, i) for i in range(workers)]
        statuses = [f.result() for f in concurrent.futures.as_completed(futures)]

    assert all(status == 200 for status in statuses)

    list_resp = client.get(f"/api/sessions?db_path={db_path}")
    assert list_resp.status_code == 200
    sessions = list_resp.json().get("sessions", [])
    session_ids = {s["session_id"] for s in sessions}
    assert all(f"thread_sess_{i}" in session_ids for i in range(workers))


def _process_worker_fn(db_path: str, idx: int) -> int:
    """
    Worker function executed in separate process to write DuckDB with retry backoff.

    Args:
    ----
        db_path: Path to the target DuckDB database.
        idx: Value to insert.

    Returns:
    -------
        int: The inserted value.

    """
    execute_duckdb_write_with_retry(
        db_path, "INSERT INTO proc_records VALUES (?)", [idx]
    )
    return idx


def test_multiprocess_duckdb_writes(tmp_path: Path) -> None:
    """
    Test multiple distinct processes writing sequentially and retry backoff on lock contention.

    Args:
    ----
        tmp_path: Pytest temporary directory fixture.

    """
    db_path = str(tmp_path / "proc_test.duckdb")
    conn = duckdb.connect(db_path)
    conn.execute("CREATE TABLE proc_records (val INTEGER)")
    conn.close()

    assert _process_worker_fn(db_path, 999) == 999
    execute_duckdb_write_with_retry(db_path, "INSERT INTO proc_records VALUES (1000)")

    original_connect = duckdb.connect
    calls: list[int] = []

    def mock_connect(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
        """Raise lock IOException on first call then delegate to duckdb.connect."""
        calls.append(1)
        if len(calls) == 1:
            raise duckdb.IOException("database is locked")
        return original_connect(*args, **kwargs)

    with patch("duckdb.connect", side_effect=mock_connect):
        execute_duckdb_write_with_retry(
            db_path,
            "INSERT INTO proc_records VALUES (?)",
            [1001],
            sleep_fn=lambda _: None,
        )

    with patch("duckdb.connect", side_effect=duckdb.IOException("disk I/O failure")):
        with pytest.raises(duckdb.IOException, match="disk I/O failure"):
            execute_duckdb_write_with_retry(
                db_path,
                "INSERT INTO proc_records VALUES (1)",
                sleep_fn=lambda _: None,
            )

    with patch("duckdb.connect", side_effect=duckdb.IOException("database is locked")):
        with pytest.raises(
            duckdb.IOException, match="database is locked after retries"
        ):
            execute_duckdb_write_with_retry(
                db_path,
                "INSERT INTO proc_records VALUES (1)",
                max_attempts=2,
                sleep_fn=lambda _: None,
            )

    workers = 4
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_process_worker_fn, db_path, i) for i in range(workers)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]

    assert len(results) == workers
    conn = duckdb.connect(db_path, read_only=True)
    count_row = conn.execute("SELECT count(*) FROM proc_records").fetchone()
    assert count_row is not None
    count = count_row[0]
    conn.close()
    assert count == workers + 3
