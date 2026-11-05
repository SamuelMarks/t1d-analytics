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
    Protocol,
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


class RateLimitEnforcerCallable(Protocol):
    """Structural protocol for rate-limit enforcement callables."""

    def __call__(
        self,
        request: Request,
        limit: int = 120,
        window_seconds: float = 60.0,
    ) -> None:
        """Enforce sliding-window rate limiting on an incoming HTTP request."""


class SqlExecutorCallable(Protocol):
    """Structural protocol for read-only DuckDB query execution callables."""

    def __call__(
        self,
        db_path: str,
        query: str,
        max_rows: int = 10000,
        timeout_seconds: float = 30.0,
    ) -> List[Dict[str, SqlValue]]:
        """Execute a read-only SQL query and return sanitized row dictionaries."""


class NlToSqlTranslatorCallable(Protocol):
    """Structural protocol for natural-language-to-SQL translation callables."""

    def __call__(
        self,
        db_path: str,
        user_query: str,
        model_name: str = "gemma4",
        history: Optional[List[ChatMessageModel]] = None,
        api_key: Optional[str] = None,
        provider: Optional[str] = None,
    ) -> TranslationResult:
        """Translate a natural-language clinical question into a `TranslationResult`."""


class LlmTokenStreamerCallable(Protocol):
    """Structural protocol for streaming LLM token generators."""

    def __call__(
        self,
        prompt: str,
        model_name: str = "gemma4",
        api_key: Optional[str] = None,
        provider: Optional[str] = None,
    ) -> Iterator[str]:
        """Yield streamed token strings from the configured LLM provider."""


class ChatEventStreamerCallable(Protocol):
    """Structural protocol for SSE chat event stream generators."""

    def __call__(
        self,
        request: ChatRequest,
        req: Optional[Request] = None,
    ) -> AsyncIterator[str]:
        """Yield Server-Sent Event frames for a streaming chat request."""


class DuckDbConnectCallable(Protocol):
    """Structural protocol for DuckDB connection factories."""

    def __call__(
        self,
        database: str = ":memory:",
        read_only: bool = False,
    ) -> duckdb.DuckDBPyConnection:
        """Open and return a DuckDB connection."""


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
    enforce_rate_limit: RateLimitEnforcerCallable
    execute_sql: SqlExecutorCallable
    generate_sql_from_nl: NlToSqlTranslatorCallable
    stream_llm_tokens: LlmTokenStreamerCallable
    stream_chat_events: ChatEventStreamerCallable
    parse_error: Callable[[Exception], ApiError]
    get_database_schema: Callable[[duckdb.DuckDBPyConnection], str]
    profile_table: Callable[[str, str], TableProfileResult]
    duckdb_connect: DuckDbConnectCallable
    get_request_locale: Callable[[Request], str]
    cloud_provider_models: Dict[str, List[str]]
    get_session_store: Callable[[], SessionStore]
    get_training_job_manager: Callable[[], TrainingJobManager]


class AppServicesRegistry:
    """
    Thread-safe registry resolving `AppServices` from a bound FastAPI `app.state.services` or explicit override.

    Attributes
    ----------
        _lock: Mutex guarding reads and writes to `_bound_app` and `_default_services`.
        _bound_app: Optional bound FastAPI application whose `state.services` is inspected.
        _default_services: Optional explicit `AppServices` instance override.

    """

    def __init__(self) -> None:
        """Initialize an empty thread-safe `AppServicesRegistry`."""
        self._lock = threading.Lock()
        self._bound_app: Optional[FastAPI] = None
        self._default_services: Optional[AppServices] = None

    def set_default(
        self,
        services: Optional[AppServices],
        target_app: Optional[FastAPI] = None,
    ) -> None:
        """
        Store or clear the default `AppServices` instance and optional bound FastAPI application.

        Args:
        ----
            services: Optional `AppServices` instance.
            target_app: Optional FastAPI application instance.

        """
        with self._lock:
            if target_app is not None:
                self._bound_app = target_app
                target_app.state.services = services
            self._default_services = services

    def get_default(self) -> Optional[AppServices]:
        """
        Return the active `AppServices` from `_default_services` or `bound_app.state.services`.

        Returns
        -------
            Optional[AppServices]: Registered `AppServices` bundle or None.

        """
        with self._lock:
            if self._default_services is not None:
                return self._default_services
            app_svc = (
                getattr(self._bound_app.state, "services", None)
                if self._bound_app is not None
                else None
            )
            return app_svc if isinstance(app_svc, AppServices) else None


_app_services_registry = AppServicesRegistry()


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
        validate_db_path=validate_db_path,
        validate_workspace_artifact_path=validate_workspace_artifact_path,
        discover_workspace_databases=discover_workspace_databases,
        enforce_rate_limit=lambda request, limit=120, window_seconds=60.0: (
            enforce_rate_limit(
                request,
                limit=limit,
                window_seconds=window_seconds,
                limiter_check_fn=check_rate_limit,
            )
        ),
        execute_sql=lambda db_path, query, max_rows=10000, timeout_seconds=30.0: (
            execute_sql(
                db_path=db_path,
                query=query,
                max_rows=max_rows,
                timeout_seconds=timeout_seconds,
                connect_fn=duckdb.connect,
            )
        ),
        generate_sql_from_nl=lambda db_path, user_query, model_name="gemma4", history=None, api_key=None, provider=None: (
            translate_nl_to_sql(
                db_path,
                user_query,
                model_name=model_name,
                history=history,
                api_key=api_key,
                provider=provider,
            )
        ),
        stream_llm_tokens=lambda prompt, model_name="gemma4", api_key=None, provider=None: (
            stream_llm_completion_tokens(
                prompt,
                model_name=model_name,
                api_key=api_key,
                provider=provider,
            )
        ),
        stream_chat_events=_default_stream_chat_events,
        parse_error=parse_error,
        get_database_schema=get_database_schema,
        profile_table=lambda db_path, table_name: profile_table(db_path, table_name),
        duckdb_connect=lambda database=":memory:", read_only=False: duckdb.connect(
            database, read_only=read_only
        ),
        get_request_locale=get_request_locale,
        cloud_provider_models=CLOUD_MODEL_CATALOG,
        get_session_store=session_store_getter or (lambda: _default_session_store),
        get_training_job_manager=job_manager_getter
        or (lambda: global_training_job_manager),
    )


def set_default_app_services(
    services: Optional[AppServices],
    target_app: Optional[FastAPI] = None,
) -> None:
    """
    Register or clear the active `AppServices` container on `target_app.state.services` and the registry.

    Args:
    ----
        services: Optional `AppServices` instance to set as the active default.
        target_app: Optional FastAPI application instance whose `state.services` should be updated.

    """
    _app_services_registry.set_default(services, target_app=target_app)


def get_app_services() -> AppServices:
    """
    Resolve the active `AppServices` bundle from `app.state.services` without circular module imports.

    Returns
    -------
        AppServices: Resolved service bundle.

    """
    return _app_services_registry.get_default() or create_default_app_services()
