"""Rate limiting utilities and middleware helpers for the T1D Analytics REST API."""

import math
import os
import threading
import time
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional

import duckdb
from fastapi import Request

from t1d_analytics.errors import RateLimitExceededError

RATE_LIMIT_STORE: Dict[str, List[float]] = {}
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
        ----
            client_ip: The client IP address string.
            limit: Maximum allowed requests within the window.
            window_seconds: Duration of the sliding window in seconds.

        Returns:
        -------
            bool: True if allowed, False if rate-limited.

        """
        raise NotImplementedError

    @abstractmethod
    def get_retry_after(self, client_ip: str, window_seconds: float = 60.0) -> int:
        """
        Calculate remaining seconds before the client can retry.

        Args:
        ----
            client_ip: The client IP address string.
            window_seconds: Duration of the sliding window in seconds.

        Returns:
        -------
            int: Seconds until the rate limit expires.

        """
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        """Reset rate limiter state."""
        raise NotImplementedError


class MemoryRateLimiter(BaseRateLimiter):
    """In-memory sliding-window rate limiter using thread synchronization."""

    def is_allowed(
        self, client_ip: str, limit: int = 120, window_seconds: float = 60.0
    ) -> bool:
        """
        Check and record request timestamp in memory.

        Args:
        ----
            client_ip: The client IP address string.
            limit: Maximum allowed requests within the window.
            window_seconds: Duration of the sliding window in seconds.

        Returns:
        -------
            bool: True if allowed, False if rate-limited.

        """
        now = time.time()
        with RATE_LIMIT_LOCK:
            expired_ips = [
                ip
                for ip, ts_list in RATE_LIMIT_STORE.items()
                if not ts_list or (now - ts_list[-1] >= window_seconds)
            ]
            for ip in expired_ips:
                RATE_LIMIT_STORE.pop(ip, None)

            timestamps = RATE_LIMIT_STORE.get(client_ip, [])
            valid_timestamps = [t for t in timestamps if now - t < window_seconds]
            if len(valid_timestamps) >= limit:
                RATE_LIMIT_STORE[client_ip] = valid_timestamps
                return False
            valid_timestamps.append(now)
            RATE_LIMIT_STORE[client_ip] = valid_timestamps
            return True

    def get_retry_after(self, client_ip: str, window_seconds: float = 60.0) -> int:
        """
        Calculate retry-after delay based on oldest valid timestamp.

        Args:
        ----
            client_ip: The client IP address string.
            window_seconds: Duration of the sliding window in seconds.

        Returns:
        -------
            int: Seconds remaining until window expiry.

        """
        now = time.time()
        with RATE_LIMIT_LOCK:
            timestamps = RATE_LIMIT_STORE.get(client_ip, [])
            valid = [t for t in timestamps if now - t < window_seconds]
            if not valid:
                RATE_LIMIT_STORE.pop(client_ip, None)
                return 1
            oldest = min(valid)
            return max(1, math.ceil(window_seconds - (now - oldest)))

    def reset(self) -> None:
        """Clear all stored timestamps."""
        with RATE_LIMIT_LOCK:
            RATE_LIMIT_STORE.clear()


class DuckDbRateLimiter(BaseRateLimiter):
    """Shared persistent rate limiter backed by a DuckDB table."""

    def __init__(self, db_path: Optional[str] = None) -> None:
        """
        Initialize DuckDB-backed rate limiter with a persistent thread-safe connection.

        Args:
        ----
            db_path: Path to DuckDB file, or ':memory:' if None.

        """
        super().__init__()
        self.db_path = db_path or os.environ.get("T1D_RATE_LIMITER_DB", ":memory:")
        self._lock = threading.Lock()
        self._conn = duckdb.connect(self.db_path)
        with self._lock:
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS api_rate_limits (client_ip VARCHAR, ts DOUBLE)"
            )

    def is_allowed(
        self, client_ip: str, limit: int = 120, window_seconds: float = 60.0
    ) -> bool:
        """
        Check and record request in DuckDB.

        Args:
        ----
            client_ip: The client IP address string.
            limit: Maximum allowed requests within the window.
            window_seconds: Duration of the sliding window in seconds.

        Returns:
        -------
            bool: True if allowed, False if rate-limited.

        """
        now = time.time()
        cutoff = now - window_seconds
        with self._lock:
            self._conn.execute("DELETE FROM api_rate_limits WHERE ts < ?", [cutoff])
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
        ----
            client_ip: The client IP address string.
            window_seconds: Duration of the sliding window in seconds.

        Returns:
        -------
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


_RATE_LIMITER_INSTANCES: Dict[str, BaseRateLimiter] = {}
_RATE_LIMITER_INSTANCES_LOCK = threading.Lock()


def get_rate_limiter() -> BaseRateLimiter:
    """
    Resolve the configured rate limiter instance based on environment configuration.

    Returns
    -------
        BaseRateLimiter: MemoryRateLimiter or DuckDbRateLimiter instance.

    """
    backend = os.environ.get("T1D_RATE_LIMITER_BACKEND", "memory").lower().strip()
    with _RATE_LIMITER_INSTANCES_LOCK:
        if backend not in _RATE_LIMITER_INSTANCES:
            if backend == "duckdb":
                _RATE_LIMITER_INSTANCES[backend] = DuckDbRateLimiter()
            else:
                _RATE_LIMITER_INSTANCES[backend] = MemoryRateLimiter()
        return _RATE_LIMITER_INSTANCES[backend]


def check_rate_limit(
    client_ip: str, limit: int = 120, window_seconds: float = 60.0
) -> bool:
    """
    Check if a client IP has exceeded the sliding-window rate limit.

    Args:
    ----
        client_ip: The client IP address string.
        limit: Maximum number of allowed requests in the window.
        window_seconds: The duration of the sliding window in seconds.

    Returns:
    -------
        bool: True if the request is allowed, False if rate limited.

    """
    return get_rate_limiter().is_allowed(
        client_ip, limit=limit, window_seconds=window_seconds
    )


def enforce_rate_limit(
    request: Request,
    limit: int = 120,
    window_seconds: float = 60.0,
    limiter_check_fn: Callable[[str, int, float], bool] = check_rate_limit,
) -> None:
    """
    Enforce sliding-window rate limiting on a specific endpoint.

    Args:
    ----
        request: The incoming FastAPI HTTP request.
        limit: Maximum requests allowed in the window (default: 120).
        window_seconds: Window size in seconds (default: 60.0).
        limiter_check_fn: Callable checking whether the client IP is allowed.

    Raises:
    ------
        RateLimitExceededError: 429 Too Many Requests if the limit is exceeded.

    """
    client_ip = request.client.host if request.client else "unknown"
    if not limiter_check_fn(client_ip, limit, window_seconds):
        retry_after = get_rate_limiter().get_retry_after(
            client_ip, window_seconds=window_seconds
        )
        raise RateLimitExceededError(
            retry_after=retry_after,
            params={"limit": limit, "window": int(window_seconds)},
        )
