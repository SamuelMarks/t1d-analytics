"""Thread-safe DuckDB read-only connection pooling, sandboxing, and retryable write helpers."""

import datetime
import decimal
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import (
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Union,
    runtime_checkable,
)

import duckdb

JAEB_ATTACH_CANDIDATES: Tuple[Path, ...] = (
    Path("/mnt/data/t1d/jaeb/t1d.duckdb"),
    Path("/mnt/data/t1d/t1d.duckdb"),
)

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


def _resolve_jaeb_attach_path(
    db_path: Union[str, Path],
    jaeb_candidates: Optional[Sequence[Path]] = None,
) -> Optional[Path]:
    """
    Resolve an existing `/mnt/data/t1d` Jaeb DuckDB file when it exists on disk and is distinct from `db_path`.

    Args:
    ----
        db_path: Primary database path being opened.
        jaeb_candidates: Optional candidate filesystem paths for the Jaeb DuckDB file.

    Returns:
    -------
        Optional[Path]: Resolved Jaeb DuckDB path to attach in read-only mode, or `None`.

    """
    candidates = (
        jaeb_candidates if jaeb_candidates is not None else JAEB_ATTACH_CANDIDATES
    )
    try:
        primary_resolved = Path(db_path).resolve()
    except OSError:
        primary_resolved = Path(db_path)

    for candidate in candidates:
        try:
            if candidate.is_file():
                cand_resolved = candidate.resolve()
                if cand_resolved != primary_resolved:
                    return cand_resolved
        except OSError:
            continue
    return None


def _attach_jaeb_read_only_if_present(
    conn: duckdb.DuckDBPyConnection,
    db_path: Union[str, Path],
    jaeb_candidates: Optional[Sequence[Path]] = None,
) -> None:
    """
    Attach `/mnt/data/t1d/jaeb/t1d.duckdb` (or `/mnt/data/t1d/t1d.duckdb`) in `READ_ONLY` mode if present and distinct.

    Executes `ATTACH IF NOT EXISTS '<jaeb_path>' AS jaeb (READ_ONLY)` and `SET search_path = 'main,jaeb'`
    prior to locking `enable_external_access = false`.

    Args:
    ----
        conn: Open read-only DuckDB connection.
        db_path: Primary database path opened on `conn`.
        jaeb_candidates: Optional candidate filesystem paths for the Jaeb DuckDB file.

    """
    jaeb_path = _resolve_jaeb_attach_path(db_path, jaeb_candidates=jaeb_candidates)
    if jaeb_path is None:
        return
    escaped_jaeb = str(jaeb_path).replace("'", "''")
    try:
        conn.execute(f"ATTACH IF NOT EXISTS '{escaped_jaeb}' AS jaeb (READ_ONLY)")
        conn.execute("SET search_path = 'main,jaeb'")
    except (duckdb.Error, OSError, RuntimeError, ValueError):
        pass


@runtime_checkable
class _DuckDbCursorCapableProtocol(Protocol):
    """Protocol for DuckDB connections that expose a `.cursor()` factory method."""

    def cursor(self) -> duckdb.DuckDBPyConnection:  # pragma: no cover
        """Create an isolated DuckDB cursor bound to the underlying connection."""


def _noop_interrupt() -> None:
    """Perform a no-op interrupt when a test double connection does not implement `.interrupt()`."""


class ReadOnlyDuckDbCursorProxy:
    """
    Read-only DuckDB cursor wrapper delegating query execution to a dedicated cursor and interrupts to the parent connection.

    Ensures concurrent analytical queries execute on isolated cursors without thread-safety contention
    while `cursor.close()` closes only the cursor rather than the pooled parent connection.
    """

    def __init__(
        self,
        conn: duckdb.DuckDBPyConnection,
        cursor: duckdb.DuckDBPyConnection,
        has_dedicated_cursor: bool,
    ) -> None:
        """
        Initialize the read-only DuckDB cursor proxy.

        Args:
        ----
            conn: Parent pooled read-only DuckDB connection.
            cursor: Dedicated DuckDB cursor (or `conn` fallback for minimal test doubles).
            has_dedicated_cursor: Whether `cursor` is a distinct cursor object that should be closed on exit.

        """
        self._conn = conn
        self._cursor = cursor
        self._has_dedicated_cursor = has_dedicated_cursor
        self._closed = False

    def execute(
        self,
        query: str,
        parameters: Optional[Sequence[PoolSqlRawInputValue]] = None,
    ) -> duckdb.DuckDBPyConnection:
        """
        Execute a read-only SQL query on the underlying DuckDB cursor.

        Args:
        ----
            query: SQL statement to execute.
            parameters: Optional positional bind parameters.

        Returns:
        -------
            duckdb.DuckDBPyConnection: Executed DuckDB cursor relation.

        """
        if parameters is not None:
            return self._cursor.execute(query, list(parameters))
        return self._cursor.execute(query)

    def fetchone(
        self,
    ) -> Optional[Tuple[PoolSqlRawInputValue, ...]]:
        """
        Fetch the next row from the current query result set.

        Returns
        -------
            Optional[Tuple[PoolSqlRawInputValue, ...]]: Single result tuple or `None`.

        """
        return self._cursor.fetchone()

    def fetchall(
        self,
    ) -> List[Tuple[PoolSqlRawInputValue, ...]]:
        """
        Fetch all remaining rows from the current query result set.

        Returns
        -------
            List[Tuple[PoolSqlRawInputValue, ...]]: All result tuples.

        """
        return self._cursor.fetchall()

    def fetchmany(
        self,
        size: int = 1,
    ) -> List[Tuple[PoolSqlRawInputValue, ...]]:
        """
        Fetch up to `size` rows from the current query result set.

        Args:
        ----
            size: Maximum number of rows to fetch.

        Returns:
        -------
            List[Tuple[PoolSqlRawInputValue, ...]]: Fetched result tuples.

        """
        return self._cursor.fetchmany(size)

    @property
    def description(
        self,
    ) -> Optional[Sequence[Tuple[str, ...]]]:
        """
        Return the cursor column description metadata for the most recent query.

        Returns
        -------
            Optional[Sequence[Tuple[str, ...]]]: Sequence of column descriptor tuples or `None`.

        """
        raw_desc = self._cursor.description
        if raw_desc is None:
            return None
        return [(str(col[0]),) for col in raw_desc]

    def interrupt(self) -> None:
        """Interrupt an in-flight query on the active DuckDB cursor and parent connection."""
        cursor_interrupt = getattr(self._cursor, "interrupt", None)
        if callable(cursor_interrupt):
            cursor_interrupt()
        if self._cursor is not self._conn:
            conn_interrupt = getattr(self._conn, "interrupt", _noop_interrupt)
            if callable(conn_interrupt):
                conn_interrupt()

    def close(self) -> None:
        """Close the dedicated DuckDB cursor idempotently without closing the pooled parent connection."""
        if self._closed:
            return
        self._closed = True
        if self._has_dedicated_cursor:
            close_attr = getattr(self._cursor, "close", None)
            if callable(close_attr):
                close_attr()

    @property
    def connection(self) -> duckdb.DuckDBPyConnection:
        """
        Return the underlying parent DuckDB connection.

        Returns
        -------
            duckdb.DuckDBPyConnection: Parent DuckDB connection.

        """
        return self._conn

    @property
    def raw_cursor(self) -> duckdb.DuckDBPyConnection:
        """
        Return the underlying raw DuckDB cursor object.

        Returns
        -------
            duckdb.DuckDBPyConnection: Underlying DuckDB cursor (or fallback connection).

        """
        return self._cursor


def _create_read_only_cursor_proxy(
    conn: duckdb.DuckDBPyConnection,
) -> ReadOnlyDuckDbCursorProxy:
    """
    Create a `ReadOnlyDuckDbCursorProxy` from `conn`, using `conn.cursor()` when supported by a real DuckDB connection.

    Falls back cleanly to `conn` for minimal test doubles or mock connections where `.execute()` is stubbed directly on `conn`.

    Args:
    ----
        conn: Open read-only DuckDB connection or test double.

    Returns:
    -------
        ReadOnlyDuckDbCursorProxy: Initialized read-only cursor proxy.

    """
    if isinstance(conn, duckdb.DuckDBPyConnection):
        dedicated_cursor = conn.cursor()
        dbs = conn.execute(
            "SELECT database_name FROM duckdb_databases() WHERE database_name = 'jaeb'"
        ).fetchall()
        if dbs:
            dedicated_cursor.execute("SET search_path = 'main,jaeb'")
        return ReadOnlyDuckDbCursorProxy(
            conn=conn,
            cursor=dedicated_cursor,
            has_dedicated_cursor=True,
        )
    if (
        isinstance(conn, _DuckDbCursorCapableProtocol)
        and type(conn).__module__ != "unittest.mock"
    ):
        cursor_obj = conn.cursor()
        if cursor_obj is not None and cursor_obj is not conn:
            return ReadOnlyDuckDbCursorProxy(
                conn=conn,
                cursor=cursor_obj,
                has_dedicated_cursor=True,
            )
    return ReadOnlyDuckDbCursorProxy(
        conn=conn,
        cursor=conn,
        has_dedicated_cursor=False,
    )


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
        _attach_jaeb_read_only_if_present(new_conn, db_path)
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
            _attach_jaeb_read_only_if_present(conn, db_path)
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


@contextmanager
def connect_duckdb_read_only_cursor(
    db_path: Union[str, Path],
    connect_fn: Callable[..., duckdb.DuckDBPyConnection] = duckdb.connect,
    disable_external_access: bool = True,
    pool: Optional[ReadOnlyDuckDBPool] = None,
) -> Iterator[ReadOnlyDuckDbCursorProxy]:
    """
    Context manager checking out a pooled read-only DuckDB connection and yielding an isolated read-only cursor.

    Checks out a read-only connection via `connect_duckdb_read_only_pooled`, attaches
    `/mnt/data/t1d/jaeb/t1d.duckdb` (or `/mnt/data/t1d/t1d.duckdb`) in `READ_ONLY` mode
    if present and distinct from `db_path`, creates a dedicated DuckDB cursor, and closes
    the cursor deterministically before returning the connection to the read-only pool.

    Args:
    ----
        db_path: Path to the DuckDB database file.
        connect_fn: DuckDB connection factory callable.
        disable_external_access: Whether to disable external filesystem access on new connections.
        pool: Optional custom `ReadOnlyDuckDBPool` instance.

    Yields:
    ------
        ReadOnlyDuckDbCursorProxy: Isolated read-only DuckDB cursor proxy.

    """
    with connect_duckdb_read_only_pooled(
        db_path=db_path,
        connect_fn=connect_fn,
        disable_external_access=disable_external_access,
        pool=pool,
    ) as conn:
        cursor_proxy = _create_read_only_cursor_proxy(conn)
        try:
            yield cursor_proxy
        finally:
            cursor_proxy.close()


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
    "JAEB_ATTACH_CANDIDATES",
    "PoolSqlRawInputValue",
    "PoolSqlValue",
    "ReadOnlyDuckDBPool",
    "ReadOnlyDuckDbCursorProxy",
    "_attach_jaeb_read_only_if_present",
    "_create_read_only_cursor_proxy",
    "_is_duckdb_lock_error",
    "_resolve_jaeb_attach_path",
    "close_read_only_duckdb_pool",
    "connect_duckdb_read_only_cursor",
    "connect_duckdb_read_only_pooled",
    "connect_duckdb_with_retry",
    "execute_duckdb_batch_write_with_retry",
    "execute_duckdb_write_with_retry",
    "global_read_only_duckdb_pool",
    "open_sandboxed_duckdb",
]
