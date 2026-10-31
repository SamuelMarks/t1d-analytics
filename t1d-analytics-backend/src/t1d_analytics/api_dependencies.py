"""Shared FastAPI middleware, security dependencies, and store/job providers for T1D Analytics."""

import hmac
import logging
import os
import threading
from contextlib import contextmanager
from typing import Callable, Dict, Iterator, List, Optional

import duckdb
from fastapi import FastAPI, Request

from t1d_analytics.api_schemas import ApiError
from t1d_analytics.db_security import (
    discover_workspace_databases as discover_workspace_databases,
)
from t1d_analytics.db_security import (
    get_allowed_db_dirs as get_allowed_db_dirs,
)
from t1d_analytics.db_security import (
    validate_db_path as validate_db_path,
)
from t1d_analytics.db_security import (
    validate_workspace_artifact_path as validate_workspace_artifact_path,
)
from t1d_analytics.errors import (
    UnauthorizedApiError,
    exception_to_error_detail,
)
from t1d_analytics.job_manager import TrainingJobManager
from t1d_analytics.rate_limiter import (
    BaseRateLimiter as BaseRateLimiter,
)
from t1d_analytics.rate_limiter import (
    DuckDbRateLimiter as DuckDbRateLimiter,
)
from t1d_analytics.rate_limiter import (
    MemoryRateLimiter as MemoryRateLimiter,
)
from t1d_analytics.rate_limiter import (
    check_rate_limit as check_rate_limit,
)
from t1d_analytics.rate_limiter import (
    enforce_rate_limit as _enforce_rate_limit,
)
from t1d_analytics.rate_limiter import (
    get_rate_limiter as get_rate_limiter,
)
from t1d_analytics.session_store import SessionStore, open_sessions_db_conn
from t1d_analytics.sql_executor import execute_duckdb_query
from t1d_analytics.sql_utils import SqlValue

_sessions_lock = threading.Lock()
_get_sessions_db_conn = open_sessions_db_conn


def configure_api_logging() -> None:
    """
    Configure default root logging formatting for the T1D Analytics API if not already configured.

    Returns
    -------
        None

    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def get_cors_origins() -> List[str]:
    """
    Resolve allowed CORS origins from T1D_CORS_ORIGINS environment variable.

    Returns
    -------
        List[str]: List of allowed origin URLs for cross-origin requests.

    """
    raw = os.environ.get(
        "T1D_CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000,http://127.0.0.1:3000",
    )
    return [o.strip() for o in raw.split(",") if o.strip()]


def get_request_locale(request: Request) -> str:
    """
    Extract the primary two-letter language code from an incoming HTTP request's `Accept-Language` header.

    Args:
    ----
        request: The incoming FastAPI HTTP request.

    Returns:
    -------
        str: Primary locale code (e.g. `'en'`, `'ja'`, `'ar'`, `'he'`).

    """
    lang_header = request.headers.get("accept-language", "en")
    return lang_header.split(",")[0].split("-")[0].split(";")[0]


def verify_api_auth(request: Request) -> None:
    """
    Validate optional API key or Bearer token authentication.

    If T1D_API_KEY or T1D_API_BEARER_TOKEN is configured in the environment,
    requests must provide either a matching 'X-API-Key' header or
    'Authorization: Bearer <token>' header. Public health check endpoints
    (/api/health, /api/ready) are exempt.

    Args:
    ----
        request: The incoming FastAPI HTTP request.

    Raises:
    ------
        UnauthorizedApiError: 401 Unauthorized if auth fails or token is missing.

    """
    required_key = os.environ.get("T1D_API_KEY") or os.environ.get(
        "T1D_API_BEARER_TOKEN"
    )
    if not required_key:
        return

    exempt_paths = {"/api/health", "/api/ready", "/docs", "/openapi.json"}
    if request.url.path in exempt_paths:
        return

    api_key_header = request.headers.get("x-api-key")
    auth_header = request.headers.get("authorization")

    bearer_token = None
    if auth_header and auth_header.lower().startswith("bearer "):
        bearer_token = auth_header[7:].strip()

    provided_token = api_key_header or bearer_token
    if not provided_token or not hmac.compare_digest(
        provided_token.encode("utf-8"),
        required_key.encode("utf-8"),
    ):
        raise UnauthorizedApiError(
            message="Invalid or missing API key.",
            params={"message": "Invalid or missing API key."},
        )


def enforce_rate_limit(
    request: Request,
    limit: int = 120,
    window_seconds: float = 60.0,
    limiter_check_fn: Callable[..., bool] = check_rate_limit,
) -> None:
    """
    Enforce sliding-window rate limiting on a specific endpoint.

    Args:
    ----
        request: The incoming FastAPI HTTP request.
        limit: Maximum requests allowed in the window (default: 120).
        window_seconds: Window size in seconds (default: 60.0).
        limiter_check_fn: Callable checking rate-limit state.

    """
    _enforce_rate_limit(
        request,
        limit=limit,
        window_seconds=window_seconds,
        limiter_check_fn=limiter_check_fn,
    )


def parse_error(e: Exception) -> ApiError:
    """
    Parse an exception into a structured ApiError.

    Args:
    ----
        e: The exception to parse.

    Returns:
    -------
        ApiError: The resulting ApiError object.

    """
    detail = exception_to_error_detail(e)
    return ApiError(error_code=detail["error_code"], params=detail["params"])


def execute_sql(
    db_path: str,
    query: str,
    max_rows: int = 10000,
    timeout_seconds: float = 30.0,
    connect_fn: Callable[..., duckdb.DuckDBPyConnection] = duckdb.connect,
) -> List[Dict[str, SqlValue]]:
    """
    Execute a SQL query and return results as a list of dictionaries.

    Args:
    ----
        db_path: Path to the DuckDB database file.
        query: SQL query to execute.
        max_rows: Maximum rows to return to prevent memory exhaustion.
        timeout_seconds: Execution timeout in seconds.
        connect_fn: DuckDB connection factory callable.

    Returns:
    -------
        List[Dict[str, SqlValue]]: List of dictionaries representing the rows returned.

    """
    return execute_duckdb_query(
        db_path=db_path,
        query=query,
        max_rows=max_rows,
        timeout_seconds=timeout_seconds,
        connect_fn=connect_fn,
    )


@contextmanager
def get_sessions_conn(
    db_path: Optional[str] = None,
    conn_opener: Callable[..., duckdb.DuckDBPyConnection] = open_sessions_db_conn,
) -> Iterator[duckdb.DuckDBPyConnection]:
    """
    Context manager yielding a persistent chat sessions database connection and ensuring it closes.

    Args:
    ----
        db_path: Optional path to the sessions database.
        conn_opener: Callable opening a DuckDB connection to the sessions database.

    Yields:
    ------
        duckdb.DuckDBPyConnection: Database connection guaranteed to be closed upon exit.

    """
    conn = conn_opener(db_path) if db_path is not None else conn_opener()
    try:
        yield conn
    finally:
        conn.close()


def resolve_session_store(
    target_app: FastAPI, default_store: SessionStore
) -> SessionStore:
    """
    Resolve the active `SessionStore` from `target_app.state` or fall back to `default_store`.

    Args:
    ----
        target_app: FastAPI application instance.
        default_store: Fallback `SessionStore` instance.

    Returns:
    -------
        SessionStore: Active conversation session store.

    """
    store = getattr(target_app.state, "session_store", None)
    return store if isinstance(store, SessionStore) else default_store


def resolve_training_job_manager(target_app: FastAPI) -> TrainingJobManager:
    """
    Resolve the active `TrainingJobManager` from `target_app.state` or fall back to the global singleton.

    Args:
    ----
        target_app: FastAPI application instance.

    Returns:
    -------
        TrainingJobManager: Active background training job manager.

    """
    from t1d_analytics.training_runner import global_training_job_manager

    mgr = getattr(target_app.state, "training_job_manager", None)
    return mgr if isinstance(mgr, TrainingJobManager) else global_training_job_manager
