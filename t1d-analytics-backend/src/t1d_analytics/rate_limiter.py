"""Rate limiting utilities and middleware helpers for the T1D Analytics REST API."""

from __future__ import annotations

import math
import os
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from types import TracebackType

import duckdb
from fastapi import Request

from t1d_analytics.errors import RateLimitExceededError

RATE_LIMIT_STORE: dict[str, list[float]] = {}
RATE_LIMIT_LOCK = threading.Lock()


class BaseRateLimiter(ABC):
    """Abstract base class defining the contract for rate limiters."""

    @abstractmethod
    def is_allowed(
        self, client_ip: str, limit: int = 120, window_seconds: float = 60.0
    ) -> bool:
        """
        Check if a client IP has exceeded the sliding-window rate limit.

        Args:
            client_ip: The client IP address string.
            limit: Maximum allowed requests within the window.
            window_seconds: Duration of the sliding window in seconds.

        Returns:
            bool: True if allowed, False if rate-limited.

        """
        ...

    @abstractmethod
    def get_retry_after(self, client_ip: str, window_seconds: float = 60.0) -> int:
        """
        Calculate remaining seconds before the client can retry.

        Args:
            client_ip: The client IP address string.
            window_seconds: Duration of the sliding window in seconds.

        Returns:
            int: Seconds until the rate limit expires.

        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset rate limiter state."""
        ...

    def close(self) -> None:
        """Release any underlying rate limiter resources."""
        return None

    def __enter__(self) -> BaseRateLimiter:
        """
        Enter the runtime context for the rate limiter.

        Returns:
            BaseRateLimiter: The active rate limiter instance.

        """
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """
        Exit the runtime context and close the rate limiter.

        Args:
            exc_type: Exception type if raised inside the context block.
            exc_val: Exception instance if raised inside the context block.
            exc_tb: Exception traceback if raised inside the context block.

        """
        _ = (exc_type, exc_val, exc_tb)
        self.close()


class MemoryRateLimiter(BaseRateLimiter):
    """In-memory sliding-window rate limiter using thread synchronization."""

    def __init__(
        self,
        store: dict[str, list[float]] | None = None,
        lock: threading.Lock | None = None,
    ) -> None:
        """
        Initialize an in-memory rate limiter with an encapsulated timestamp dictionary and lock.

        Args:
            store: Optional backing dictionary mapping client IPs to request timestamps.
            lock: Optional threading.Lock for synchronizing concurrent access.

        """
        super().__init__()
        self._store: dict[str, list[float]] = store if store is not None else {}
        self._lock: threading.Lock = lock if lock is not None else threading.Lock()

    def is_allowed(
        self, client_ip: str, limit: int = 120, window_seconds: float = 60.0
    ) -> bool:
        """
        Check and record request timestamp in memory.

        Args:
            client_ip: The client IP address string.
            limit: Maximum allowed requests within the window.
            window_seconds: Duration of the sliding window in seconds.

        Returns:
            bool: True if allowed, False if rate-limited.

        """
        now = time.time()
        with self._lock:
            expired_ips = [
                ip
                for ip, ts_list in self._store.items()
                if not ts_list or (now - ts_list[-1] >= window_seconds)
            ]
            for ip in expired_ips:
                self._store.pop(ip, None)

            timestamps = self._store.get(client_ip, [])
            valid_timestamps = [t for t in timestamps if now - t < window_seconds]
            if len(valid_timestamps) >= limit:
                self._store[client_ip] = valid_timestamps
                return False
            valid_timestamps.append(now)
            self._store[client_ip] = valid_timestamps
            return True

    def get_retry_after(self, client_ip: str, window_seconds: float = 60.0) -> int:
        """
        Calculate retry-after delay based on oldest valid timestamp.

        Args:
            client_ip: The client IP address string.
            window_seconds: Duration of the sliding window in seconds.

        Returns:
            int: Seconds remaining until window expiry.

        """
        now = time.time()
        with self._lock:
            timestamps = self._store.get(client_ip, [])
            valid = [t for t in timestamps if now - t < window_seconds]
            if not valid:
                self._store.pop(client_ip, None)
                return 1
            oldest = min(valid)
            return max(1, math.ceil(window_seconds - (now - oldest)))

    def reset(self) -> None:
        """Clear all stored timestamps."""
        with self._lock:
            self._store.clear()


class DuckDbRateLimiter(BaseRateLimiter):
    """Shared persistent rate limiter backed by a DuckDB table."""

    def __init__(self, db_path: str | None = None) -> None:
        """
        Initialize DuckDB-backed rate limiter with a persistent thread-safe connection.

        Args:
            db_path: Path to DuckDB file, or ':memory:' if None.

        """
        super().__init__()
        self.db_path = db_path or os.environ.get("T1D_RATE_LIMITER_DB", ":memory:")
        self._lock = threading.Lock()
        self._last_prune_ts: float = 0.0
        self._conn = duckdb.connect(self.db_path)
        with self._lock:
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS api_rate_limits (client_ip VARCHAR, ts DOUBLE)"
            )

    def is_allowed(
        self, client_ip: str, limit: int = 120, window_seconds: float = 60.0
    ) -> bool:
        """
        Check and record request in DuckDB, pruning expired rows periodically rather than on every request.

        Args:
            client_ip: The client IP address string.
            limit: Maximum allowed requests within the window.
            window_seconds: Duration of the sliding window in seconds.

        Returns:
            bool: True if allowed, False if rate-limited.

        """
        now = time.time()
        cutoff = now - window_seconds
        prune_interval = max(1.0, min(5.0, window_seconds / 2.0))
        with self._lock:
            if now - self._last_prune_ts >= prune_interval:
                self._conn.execute("DELETE FROM api_rate_limits WHERE ts < ?", [cutoff])
                self._last_prune_ts = now
            count_res = self._conn.execute(
                "SELECT COUNT(*) FROM api_rate_limits WHERE client_ip = ? AND ts >= ?",
                [client_ip, cutoff],
            ).fetchone()
            count = count_res[0] if count_res else 0
            if count >= limit:
                return False
            self._conn.execute(
                "INSERT INTO api_rate_limits VALUES (?, ?)", [client_ip, now]
            )
            return True

    def get_retry_after(self, client_ip: str, window_seconds: float = 60.0) -> int:
        """
        Calculate retry-after delay based on oldest valid record in DuckDB.

        Args:
            client_ip: The client IP address string.
            window_seconds: Duration of the sliding window in seconds.

        Returns:
            int: Seconds remaining until window expiry.

        """
        now = time.time()
        cutoff = now - window_seconds
        with self._lock:
            res = self._conn.execute(
                "SELECT MIN(ts) FROM api_rate_limits WHERE client_ip = ? AND ts >= ?",
                [client_ip, cutoff],
            ).fetchone()
            if not res or res[0] is None:
                return 1
            min_ts = float(res[0])
            return max(1, math.ceil(window_seconds - (now - min_ts)))

    def reset(self) -> None:
        """Drop or clear all records from rate limits table."""
        with self._lock:
            self._conn.execute("DELETE FROM api_rate_limits")

    def close(self) -> None:
        """Close the underlying DuckDB connection in a thread-safe manner."""
        with self._lock:
            self._conn.close()


_RATE_LIMITER_INSTANCES: dict[str, BaseRateLimiter] = {}
_RATE_LIMITER_INSTANCES_LOCK = threading.Lock()


def close_all_rate_limiters() -> None:
    """Close and remove all cached rate limiter instances."""
    with _RATE_LIMITER_INSTANCES_LOCK:
        for limiter in _RATE_LIMITER_INSTANCES.values():
            limiter.close()
        _RATE_LIMITER_INSTANCES.clear()


def get_rate_limiter() -> BaseRateLimiter:
    """
    Resolve the configured rate limiter instance based on environment configuration.

    Returns:
        BaseRateLimiter: MemoryRateLimiter or DuckDbRateLimiter instance.

    """
    backend = os.environ.get("T1D_RATE_LIMITER_BACKEND", "memory").lower().strip()
    with _RATE_LIMITER_INSTANCES_LOCK:
        if backend not in _RATE_LIMITER_INSTANCES:
            if backend == "duckdb":
                _RATE_LIMITER_INSTANCES[backend] = DuckDbRateLimiter()
            else:
                _RATE_LIMITER_INSTANCES[backend] = MemoryRateLimiter(
                    store=RATE_LIMIT_STORE,
                    lock=RATE_LIMIT_LOCK,
                )
        return _RATE_LIMITER_INSTANCES[backend]


def check_rate_limit(
    client_ip: str, limit: int = 120, window_seconds: float = 60.0
) -> bool:
    """
    Check if a client IP has exceeded the sliding-window rate limit.

    Args:
        client_ip: The client IP address string.
        limit: Maximum number of allowed requests in the window.
        window_seconds: The duration of the sliding window in seconds.

    Returns:
        bool: True if the request is allowed, False if rate limited.

    """
    return get_rate_limiter().is_allowed(
        client_ip, limit=limit, window_seconds=window_seconds
    )


def enforce_rate_limit(
    request: Request,
    limit: int = 120,
    window_seconds: float = 60.0,
    limiter_check_fn: Callable[[str, int, float], bool] | None = None,
    limiter: BaseRateLimiter | None = None,
) -> None:
    """
    Enforce sliding-window rate limiting on a specific endpoint.

    Args:
        request: The incoming FastAPI HTTP request.
        limit: Maximum requests allowed in the window (default: 120).
        window_seconds: Window size in seconds (default: 60.0).
        limiter_check_fn: Optional explicit callable checking whether the client IP is allowed.
        limiter: Optional isolated `BaseRateLimiter` instance; defaults to `get_rate_limiter()`.

    Raises:
        RateLimitExceededError: 429 Too Many Requests if the limit is exceeded.

    """
    client_ip = request.client.host if request.client else "unknown"
    bucket_key = f"{client_ip}:{limit}:{int(window_seconds)}"
    active_limiter = limiter if limiter is not None else get_rate_limiter()
    allowed = (
        limiter_check_fn(client_ip, limit, window_seconds)
        if limiter_check_fn is not None
        else active_limiter.is_allowed(
            bucket_key, limit=limit, window_seconds=window_seconds
        )
    )
    if not allowed:
        retry_after = active_limiter.get_retry_after(
            bucket_key, window_seconds=window_seconds
        )
        raise RateLimitExceededError(
            retry_after=retry_after,
            params={"limit": limit, "window": int(window_seconds)},
        )
