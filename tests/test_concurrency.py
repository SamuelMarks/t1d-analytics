"""Concurrency and multi-threading contention tests for DuckDB."""

import concurrent.futures
from pathlib import Path

import duckdb
from fastapi.testclient import TestClient

from t1d_analytics.api import app, execute_sql


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

    # Verify all created thread sessions are present
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
    import time

    import duckdb

    for attempt in range(20):
        try:
            conn = duckdb.connect(db_path, read_only=False)
            conn.execute("INSERT INTO proc_records VALUES (?)", [idx])
            conn.close()
            return idx
        except duckdb.IOException as e:
            if "lock" in str(e).lower() and attempt < 19:
                time.sleep(0.05 * (attempt + 1))
            else:
                raise
    return idx


def test_multiprocess_duckdb_writes(tmp_path: Path) -> None:
    """
    Test multiple distinct processes writing sequentially to verify process database persistence.

    Args:
    ----
        tmp_path: Pytest temporary directory fixture.

    """
    db_path = str(tmp_path / "proc_test.duckdb")
    conn = duckdb.connect(db_path)
    conn.execute("CREATE TABLE proc_records (val INTEGER)")
    conn.close()

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
    assert count == workers
