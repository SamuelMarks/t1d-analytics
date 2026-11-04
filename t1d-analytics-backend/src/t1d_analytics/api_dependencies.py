"""Shared FastAPI middleware, security dependencies, and store/job providers for T1D Analytics."""

import hmac
import logging
import os
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import (
    AsyncIterator,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
)

import duckdb
from fastapi import FastAPI, Request

from t1d_analytics.analytics import (
    TableProfileResult,
    get_database_schema,
    profile_table,
)
from t1d_analytics.api_schemas import (
    ApiError,
    ChatMessageModel,
    ChatRequest,
    DatabaseListResponse,
)
from t1d_analytics.chat_service import (
    generate_sse_chat_events,
    stream_llm_completion_tokens,
    translate_nl_to_sql,
)
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
from t1d_analytics.llm_client import CLOUD_MODEL_CATALOG
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
from t1d_analytics.sql_fallback_synthesizer import TranslationResult
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
    connect_fn: Optional[Callable[..., duckdb.DuckDBPyConnection]] = None,
) -> List[Dict[str, SqlValue]]:
    """
    Execute a SQL query and return results as a list of dictionaries.

    Args:
    ----
        db_path: Path to the DuckDB database file.
        query: SQL query to execute.
        max_rows: Maximum rows to return to prevent memory exhaustion.
        timeout_seconds: Execution timeout in seconds.
        connect_fn: Optional DuckDB connection factory callable (defaults to `duckdb.connect`).

    Returns:
    -------
        List[Dict[str, SqlValue]]: List of dictionaries representing the rows returned.

    """
    return execute_duckdb_query(
        db_path=db_path,
        query=query,
        max_rows=max_rows,
        timeout_seconds=timeout_seconds,
        connect_fn=connect_fn or duckdb.connect,
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


_default_session_store = SessionStore(
    conn_factory=get_sessions_conn, lock=_sessions_lock
)


def _default_stream_chat_events(
    request: ChatRequest,
    req: Optional[Request] = None,
) -> AsyncIterator[str]:
    """
    Generate Server-Sent Events for streaming chat queries and SQL results using active services.

    Args:
    ----
        request: The incoming ChatRequest.
        req: Optional FastAPI Request instance to detect client disconnection.

    Returns:
    -------
        AsyncIterator[str]: SSE data lines formatted as JSON strings.

    """
    svc = get_app_services()
    return generate_sse_chat_events(
        request=request,
        req=req,
        db_validator=svc.validate_db_path,
        execute_fn=svc.execute_sql,
        connect_fn=svc.duckdb_connect,
        schema_fn=svc.get_database_schema,
        stream_fn=svc.stream_llm_tokens,
        error_parser=svc.parse_error,
    )


@dataclass(frozen=True)
class AppServices:
    """
    Typed dependency bundle of backend service callables used across FastAPI routers.

    Eliminates function-level circular imports in route modules while supporting
    explicit dependency injection via FastAPI `dependency_overrides` or `app.state.services`.
    """

    validate_db_path: Callable[[Optional[str]], str]
    validate_workspace_artifact_path: Callable[[str], str]
    discover_workspace_databases: Callable[[], DatabaseListResponse]
    enforce_rate_limit: Callable[..., None]
    execute_sql: Callable[..., List[Dict[str, SqlValue]]]
    generate_sql_from_nl: Callable[..., Union[TranslationResult, Tuple[str, str]]]
    stream_llm_tokens: Callable[..., Iterator[str]]
    stream_chat_events: Callable[..., AsyncIterator[str]]
    parse_error: Callable[[Exception], ApiError]
    get_database_schema: Callable[[duckdb.DuckDBPyConnection], str]
    profile_table: Callable[[str, str], TableProfileResult]
    duckdb_connect: Callable[..., duckdb.DuckDBPyConnection]
    get_request_locale: Callable[[Request], str]
    cloud_provider_models: Dict[str, List[str]]
    get_session_store: Callable[[], SessionStore]
    get_training_job_manager: Callable[[], TrainingJobManager]


_active_app_services: Optional[AppServices] = None


def _invoke_validate_db_path(db_path: Optional[str]) -> str:
    """
    Delegate to active `validate_db_path` callable.

    Args:
    ----
        db_path: Optional path to the DuckDB database file.

    Returns:
    -------
        str: Validated absolute database path.

    """
    return validate_db_path(db_path)


def _invoke_validate_workspace_artifact_path(target_path: str) -> str:
    """
    Delegate to active `validate_workspace_artifact_path` callable.

    Args:
    ----
        target_path: Requested workspace artifact path.

    Returns:
    -------
        str: Validated absolute artifact path.

    """
    return validate_workspace_artifact_path(target_path)


def _invoke_discover_workspace_databases() -> DatabaseListResponse:
    """
    Delegate to active `discover_workspace_databases` callable.

    Returns
    -------
        DatabaseListResponse: Discovered workspace databases and default DB.

    """
    return discover_workspace_databases()


def _invoke_enforce_rate_limit(
    request: Request,
    limit: int = 120,
    window_seconds: float = 60.0,
) -> None:
    """
    Delegate to active `enforce_rate_limit` callable with `check_rate_limit`.

    Args:
    ----
        request: Incoming FastAPI HTTP request.
        limit: Maximum requests allowed in the window.
        window_seconds: Sliding window duration in seconds.

    """
    enforce_rate_limit(
        request,
        limit=limit,
        window_seconds=window_seconds,
        limiter_check_fn=check_rate_limit,
    )


def _invoke_execute_sql(
    db_path: str,
    query: str,
    max_rows: int = 10000,
    timeout_seconds: float = 30.0,
) -> List[Dict[str, SqlValue]]:
    """
    Delegate to active `execute_sql` callable with `duckdb.connect`.

    Args:
    ----
        db_path: Path to the DuckDB database file.
        query: SQL query to execute.
        max_rows: Maximum rows to return.
        timeout_seconds: Execution timeout in seconds.

    Returns:
    -------
        List[Dict[str, SqlValue]]: Executed query result rows.

    """
    return execute_sql(
        db_path=db_path,
        query=query,
        max_rows=max_rows,
        timeout_seconds=timeout_seconds,
        connect_fn=duckdb.connect,
    )


def _invoke_generate_sql_from_nl(
    db_path: str,
    user_query: str,
    model_name: str = "gemma4",
    history: Optional[List[ChatMessageModel]] = None,
    api_key: Optional[str] = None,
    provider: Optional[str] = None,
) -> Union[TranslationResult, Tuple[str, str]]:
    """
    Delegate to active `translate_nl_to_sql` callable.

    Args:
    ----
        db_path: Path to the DuckDB database file.
        user_query: Natural language query from the user.
        model_name: Target LLM model name.
        history: Optional conversation message history.
        api_key: Optional cloud provider API key.
        provider: Optional LLM provider override.

    Returns:
    -------
        Union[TranslationResult, Tuple[str, str]]: Translation result containing response and SQL.

    """
    return translate_nl_to_sql(
        db_path,
        user_query,
        model_name=model_name,
        history=history,
        api_key=api_key,
        provider=provider,
    )


def _invoke_stream_llm_tokens(
    prompt: str,
    model_name: str = "gemma4",
    api_key: Optional[str] = None,
    provider: Optional[str] = None,
) -> Iterator[str]:
    """
    Delegate to active `stream_llm_completion_tokens` callable.

    Args:
    ----
        prompt: Prompt string to send to the LLM.
        model_name: Target LLM model name.
        api_key: Optional cloud provider API key.
        provider: Optional LLM provider override.

    Returns:
    -------
        Iterator[str]: Streamed token strings.

    """
    return stream_llm_completion_tokens(
        prompt,
        model_name=model_name,
        api_key=api_key,
        provider=provider,
    )


def _invoke_parse_error(e: Exception) -> ApiError:
    """
    Delegate to active `parse_error` callable.

    Args:
    ----
        e: Exception instance to convert.

    Returns:
    -------
        ApiError: Structured API error model.

    """
    return parse_error(e)


def _invoke_get_database_schema(conn: duckdb.DuckDBPyConnection) -> str:
    """
    Delegate to active `get_database_schema` callable.

    Args:
    ----
        conn: Active DuckDB connection.

    Returns:
    -------
        str: Formatted database schema string.

    """
    return get_database_schema(conn)


def _invoke_profile_table(db_path: str, table_name: str) -> TableProfileResult:
    """
    Delegate to active `profile_table` callable.

    Args:
    ----
        db_path: Path to the DuckDB database file.
        table_name: Target table name to profile.

    Returns:
    -------
        TableProfileResult: Table profiling statistics.

    """
    return profile_table(db_path, table_name)


def _invoke_duckdb_connect(
    database: str = ":memory:",
    read_only: bool = False,
) -> duckdb.DuckDBPyConnection:
    """
    Delegate to active `duckdb.connect` callable.

    Args:
    ----
        database: Database file path or `:memory:`.
        read_only: Whether to open the connection in read-only mode.

    Returns:
    -------
        duckdb.DuckDBPyConnection: Opened DuckDB connection.

    """
    return duckdb.connect(database, read_only=read_only)


def _invoke_get_request_locale(request: Request) -> str:
    """
    Delegate to active `get_request_locale` callable.

    Args:
    ----
        request: Incoming FastAPI HTTP request.

    Returns:
    -------
        str: Resolved locale code.

    """
    return get_request_locale(request)


def create_default_app_services(
    target_app: Optional[FastAPI] = None,
    session_store_getter: Optional[Callable[[], SessionStore]] = None,
    job_manager_getter: Optional[Callable[[], TrainingJobManager]] = None,
) -> AppServices:
    """
    Construct a strongly typed `AppServices` bundle bound to canonical backend modules.

    Args:
    ----
        target_app: Optional FastAPI application instance.
        session_store_getter: Optional callable returning active `SessionStore`.
        job_manager_getter: Optional callable returning active `TrainingJobManager`.

    Returns:
    -------
        AppServices: Newly constructed service bundle.

    """
    from t1d_analytics.training_runner import global_training_job_manager

    _ = target_app
    return AppServices(
        validate_db_path=_invoke_validate_db_path,
        validate_workspace_artifact_path=_invoke_validate_workspace_artifact_path,
        discover_workspace_databases=_invoke_discover_workspace_databases,
        enforce_rate_limit=_invoke_enforce_rate_limit,
        execute_sql=_invoke_execute_sql,
        generate_sql_from_nl=_invoke_generate_sql_from_nl,
        stream_llm_tokens=_invoke_stream_llm_tokens,
        stream_chat_events=_default_stream_chat_events,
        parse_error=_invoke_parse_error,
        get_database_schema=_invoke_get_database_schema,
        profile_table=_invoke_profile_table,
        duckdb_connect=_invoke_duckdb_connect,
        get_request_locale=_invoke_get_request_locale,
        cloud_provider_models=CLOUD_MODEL_CATALOG,
        get_session_store=session_store_getter or (lambda: _default_session_store),
        get_training_job_manager=job_manager_getter
        or (lambda: global_training_job_manager),
    )


def set_default_app_services(services: Optional[AppServices]) -> None:
    """
    Register or clear the active `AppServices` container used by `get_app_services()`.

    Args:
    ----
        services: Optional `AppServices` instance to set as the active default.

    """
    global _active_app_services
    _active_app_services = services


def get_app_services() -> AppServices:
    """
    Resolve the active `AppServices` bundle without circular module imports or `sys.modules` reflection.

    Returns
    -------
        AppServices: Resolved service bundle.

    """
    return _active_app_services or create_default_app_services()
