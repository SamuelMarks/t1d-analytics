"""Thread-safe DuckDB read-only connection pooling, sandboxing, and retryable write helpers."""

import datetime
import decimal
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import duckdb

PoolSqlValue = Union[str, int, float, bool, None]
PoolSqlRawInputValue = Union[
    str,
    int,
    float,
    bool,
    None,
    decimal.Decimal,
    datetime.date,
    datetime.datetime,
    datetime.time,
    uuid.UUID,
    bytes,
    Sequence[PoolSqlValue],
]


def _is_duckdb_lock_error(exc: duckdb.IOException) -> bool:
    """
    Determine whether a `duckdb.IOException` was caused by transient file lock contention.

    Args:
    ----
        exc: The DuckDB I/O exception to inspect.

    Returns:
    -------
        bool: True if the error indicates file lock contention, False otherwise.

    """
    err_msg = str(exc).lower()
    return (
        "lock" in err_msg
        or "conflicting" in err_msg
        or "resource temporarily unavailable" in err_msg
    )


@contextmanager
def connect_duckdb_with_retry(
    db_path: Union[str, Path],
    read_only: bool = True,
    max_attempts: int = 5,
    base_delay_seconds: float = 0.05,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> Iterator[duckdb.DuckDBPyConnection]:
    """
    Acquire a DuckDB connection with retry backoff on lock contention and close on context exit.

    Connection acquisition is strictly separated from the `yield` block so that exceptions
    raised inside the caller's `with` body never trigger generator re-entry errors.

    Args:
    ----
        db_path: Path to the DuckDB database file.
        read_only: Whether to open the database in read-only mode.
        max_attempts: Maximum number of connection attempts on lock contention.
        base_delay_seconds: Initial backoff delay in seconds.
        sleep_fn: Injectable sleep callable for backoff delays.

    Yields:
    ------
        duckdb.DuckDBPyConnection: Open DuckDB connection guaranteed to close on exit.

    Raises:
    ------
        duckdb.IOException: If a non-lock I/O error occurs or lock retries are exhausted.

    """
    attempt = 0
    while True:
        try:
            conn = duckdb.connect(str(db_path), read_only=read_only)
            break
        except duckdb.IOException as e:
            if not _is_duckdb_lock_error(e) or attempt >= max_attempts - 1:
                raise
            sleep_fn(base_delay_seconds * (attempt + 1))
            attempt += 1

    try:
        yield conn
    finally:
        conn.close()


@contextmanager
def open_sandboxed_duckdb(
    db_path: Union[str, Path],
    read_only: bool = True,
    disable_external_access: bool = True,
) -> Iterator[duckdb.DuckDBPyConnection]:
    """
    Context manager opening a DuckDB connection with optional security sandboxing.

    Args:
    ----
        db_path: Path to the DuckDB database file.
        read_only: Whether to open the database in read-only mode.
        disable_external_access: Whether to execute 'SET enable_external_access = false'.

    Yields:
    ------
        duckdb.DuckDBPyConnection: Configured DuckDB connection guaranteed to close on exit.

    """
    conn = duckdb.connect(str(db_path), read_only=read_only)
    try:
        if disable_external_access:
            conn.execute("SET enable_external_access = false")
        yield conn
    finally:
        conn.close()


class ReadOnlyDuckDBPool:
    """Thread-safe, mtime-validated connection pool for read-only DuckDB databases."""

    def __init__(self, max_pool_size_per_db: int = 8) -> None:
        """
        Initialize the read-only DuckDB connection pool.

        Args:
        ----
            max_pool_size_per_db: Maximum idle connections retained per database file.

        """
        self._max_pool_size_per_db = max(1, max_pool_size_per_db)
        self._lock = threading.Lock()
        self._pools: Dict[str, List[duckdb.DuckDBPyConnection]] = {}
        self._mtimes: Dict[str, Tuple[int, int]] = {}

    @staticmethod
    def _stat_signature(path_str: str) -> Tuple[int, int]:
        """
        Compute the `(st_mtime_ns, st_size)` signature for `path_str` on disk.

        Args:
        ----
            path_str: Target filesystem path string.

        Returns:
        -------
            Tuple[int, int]: Modification time in nanoseconds and file size in bytes.

        """
        try:
            st = Path(path_str).stat()
            return (st.st_mtime_ns, st.st_size)
        except OSError:
            return (-1, -1)

    def acquire(
        self,
        db_path: Union[str, Path],
        connect_fn: Callable[..., duckdb.DuckDBPyConnection] = duckdb.connect,
        disable_external_access: bool = True,
    ) -> duckdb.DuckDBPyConnection:
        """
        Acquire a read-only DuckDB connection from the pool or open a new one.

        Args:
        ----
            db_path: Path to the DuckDB database file.
            connect_fn: DuckDB connection factory callable.
            disable_external_access: Whether to execute 'SET enable_external_access = false'.

        Returns:
        -------
            duckdb.DuckDBPyConnection: Open read-only DuckDB connection.

        """
        key = str(Path(db_path).resolve())
        sig = self._stat_signature(key)
        stale_conns: List[duckdb.DuckDBPyConnection] = []
        with self._lock:
            prev_sig = self._mtimes.get(key)
            if prev_sig is not None and prev_sig != sig:
                stale_conns = self._pools.pop(key, [])
            self._mtimes[key] = sig
            bucket = self._pools.get(key)
            conn = bucket.pop() if bucket else None

        for stale in stale_conns:
            stale.close()

        if conn is not None:
            return conn

        new_conn = connect_fn(str(db_path), read_only=True)
        if disable_external_access:
            new_conn.execute("SET enable_external_access = false")
        return new_conn

    def release(
        self,
        db_path: Union[str, Path],
        conn: duckdb.DuckDBPyConnection,
        discard: bool = False,
    ) -> None:
        """
        Return a read-only DuckDB connection to the pool or close it if discarded/full.

        Args:
        ----
            db_path: Path to the DuckDB database file.
            conn: Connection instance to release.
            discard: If True, close `conn` instead of returning it to the pool.

        """
        if discard:
            conn.close()
            return
        key = str(Path(db_path).resolve())
        sig = self._stat_signature(key)
        should_close = False
        with self._lock:
            if sig == (-1, -1) or self._mtimes.get(key) != sig:
                should_close = True
            else:
                bucket = self._pools.setdefault(key, [])
                if len(bucket) < self._max_pool_size_per_db:
                    bucket.append(conn)
                else:
                    should_close = True
        if should_close:
            conn.close()

    def close_path(self, db_path: Union[str, Path]) -> None:
        """
        Close and evict all idle pooled read-only connections for a specific database file.

        Args:
        ----
            db_path: Path to the DuckDB database file.

        """
        key = str(Path(db_path).resolve())
        with self._lock:
            to_close = self._pools.pop(key, [])
            self._mtimes.pop(key, None)
        for conn in to_close:
            conn.close()

    def close_all(self) -> None:
        """
        Close all idle pooled connections across all database files.

        Returns
        -------
            None

        """
        to_close: List[duckdb.DuckDBPyConnection] = []
        with self._lock:
            for bucket in self._pools.values():
                to_close.extend(bucket)
            self._pools.clear()
            self._mtimes.clear()
        for conn in to_close:
            conn.close()


global_read_only_duckdb_pool = ReadOnlyDuckDBPool()


@contextmanager
def connect_duckdb_read_only_pooled(
    db_path: Union[str, Path],
    connect_fn: Callable[..., duckdb.DuckDBPyConnection] = duckdb.connect,
    disable_external_access: bool = True,
    pool: Optional[ReadOnlyDuckDBPool] = None,
) -> Iterator[duckdb.DuckDBPyConnection]:
    """
    Context manager checking out a read-only DuckDB connection from `ReadOnlyDuckDBPool`.

    Args:
    ----
        db_path: Path to the DuckDB database file.
        connect_fn: DuckDB connection factory callable.
        disable_external_access: Whether to disable external filesystem access on new connections.
        pool: Optional custom `ReadOnlyDuckDBPool` instance (defaults to `global_read_only_duckdb_pool`).

    Yields:
    ------
        duckdb.DuckDBPyConnection: Pooled read-only DuckDB connection.

    """
    active_pool = pool if pool is not None else global_read_only_duckdb_pool
    use_pool = connect_fn is duckdb.connect
    if not use_pool:
        conn = connect_fn(str(db_path), read_only=True)
        try:
            if disable_external_access:
                conn.execute("SET enable_external_access = false")
            yield conn
        finally:
            conn.close()
        return

    conn = active_pool.acquire(
        db_path=db_path,
        connect_fn=connect_fn,
        disable_external_access=disable_external_access,
    )
    discard = False
    try:
        yield conn
    except BaseException:
        discard = True
        raise
    finally:
        active_pool.release(db_path=db_path, conn=conn, discard=discard)


def close_read_only_duckdb_pool() -> None:
    """
    Close all connections held in `global_read_only_duckdb_pool`.

    Returns
    -------
        None

    """
    global_read_only_duckdb_pool.close_all()


def execute_duckdb_batch_write_with_retry(
    db_path: Union[str, Path],
    statements: Sequence[Tuple[str, Optional[Sequence[PoolSqlRawInputValue]]]],
    max_attempts: int = 20,
    base_delay_seconds: float = 0.05,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> None:
    """
    Execute a batch of write statements against a DuckDB database within a single connection and retry on lock contention.

    Args:
    ----
        db_path: Path to the target DuckDB database.
        statements: Sequence of (query, optional_params) tuples to execute in order.
        max_attempts: Maximum number of connection/execution attempts.
        base_delay_seconds: Base delay in seconds multiplied by attempt index.
        sleep_fn: Sleep callable used between lock retries.

    Raises:
    ------
        duckdb.IOException: If a non-lock I/O error occurs or lock retries are exhausted.

    """
    global_read_only_duckdb_pool.close_path(db_path)
    for attempt in range(max_attempts):
        try:
            conn = duckdb.connect(str(db_path), read_only=False)
            try:
                for query, params in statements:
                    if params is not None:
                        conn.execute(query, list(params))
                    else:
                        conn.execute(query)
            finally:
                conn.close()
            return
        except duckdb.IOException as e:
            if _is_duckdb_lock_error(e):
                sleep_fn(base_delay_seconds * (attempt + 1))
            else:
                raise
    raise duckdb.IOException("database is locked after retries")


def execute_duckdb_write_with_retry(
    db_path: Union[str, Path],
    query: str,
    params: Optional[Sequence[PoolSqlRawInputValue]] = None,
    max_attempts: int = 20,
    base_delay_seconds: float = 0.05,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> None:
    """
    Execute a write statement against a DuckDB database with retry backoff on lock contention.

    Args:
    ----
        db_path: Path to the target DuckDB database.
        query: Parameterized SQL statement to execute.
        params: Optional sequence of bind parameters.
        max_attempts: Maximum number of connection/execution attempts.
        base_delay_seconds: Base delay in seconds multiplied by attempt index.
        sleep_fn: Sleep callable used between lock retries.

    Raises:
    ------
        duckdb.IOException: If a non-lock I/O error occurs or lock retries are exhausted.

    """
    execute_duckdb_batch_write_with_retry(
        db_path=db_path,
        statements=[(query, params)],
        max_attempts=max_attempts,
        base_delay_seconds=base_delay_seconds,
        sleep_fn=sleep_fn,
    )


__all__ = [
    "PoolSqlRawInputValue",
    "PoolSqlValue",
    "ReadOnlyDuckDBPool",
    "_is_duckdb_lock_error",
    "close_read_only_duckdb_pool",
    "connect_duckdb_read_only_pooled",
    "connect_duckdb_with_retry",
    "execute_duckdb_batch_write_with_retry",
    "execute_duckdb_write_with_retry",
    "global_read_only_duckdb_pool",
    "open_sandboxed_duckdb",
]
