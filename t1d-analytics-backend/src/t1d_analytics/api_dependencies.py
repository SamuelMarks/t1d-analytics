"""Shared FastAPI middleware, security dependencies, and store/job providers for T1D Analytics."""

import hmac
import logging
import os
import threading
from collections.abc import AsyncIterator, Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import (
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
    SchemaResponse,
)
from t1d_analytics.chat_service import (
    stream_llm_completion_tokens,
    translate_nl_to_sql,
)
from t1d_analytics.db_security import (
    discover_workspace_databases,
    get_allowed_db_dirs,
    validate_db_path,
    validate_sessions_db_path,
    validate_workspace_artifact_path,
)
from t1d_analytics.errors import (
    UnauthorizedApiError,
    exception_to_error_detail,
)
from t1d_analytics.job_manager import (
    ChatTurnJobManager,
    TrainingJobManager,
    global_chat_job_manager,
)
from t1d_analytics.llm_client import CLOUD_MODEL_CATALOG
from t1d_analytics.rate_limiter import (
    BaseRateLimiter,
    DuckDbRateLimiter,
    MemoryRateLimiter,
    check_rate_limit,
    get_rate_limiter,
)
from t1d_analytics.rate_limiter import (
    enforce_rate_limit as _enforce_rate_limit,
)
from t1d_analytics.session_store import SessionStore, open_sessions_db_conn
from t1d_analytics.sql_executor import QueryTimerProtocol, execute_duckdb_query
from t1d_analytics.sql_fallback_synthesizer import TranslationResult
from t1d_analytics.sql_utils import SqlValue
from t1d_analytics.training_runner import global_training_job_manager


def configure_api_logging() -> None:
    """
    Configure default root logging formatting for the T1D Analytics API if not already configured.

    Returns:
        None

    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def get_cors_origins() -> list[str]:
    """
    Resolve allowed CORS origins from `T1D_CORS_ORIGINS` environment variable.

    Returns:
        List[str]: List of allowed origin URLs for cross-origin requests.

    """
    raw = os.environ.get(
        "T1D_CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000,http://127.0.0.1:3000,http://localhost:8000,http://127.0.0.1:8000",
    )
    return [o.strip() for o in raw.split(",") if o.strip()]


def get_request_locale(request: Request) -> str:
    """
    Extract the primary two-letter language code from an incoming HTTP request's `Accept-Language` header.

    Args:
        request: The incoming FastAPI HTTP request.

    Returns:
        str: Primary locale code (e.g. `'en'`, `'ja'`, `'ar'`, `'he'`).

    """
    lang_header = request.headers.get("accept-language", "en")
    return lang_header.split(",")[0].split("-")[0].split(";")[0]


def verify_api_auth(request: Request) -> None:
    """
    Validate optional API key or Bearer token authentication.

    If `T1D_API_KEY` or `T1D_API_BEARER_TOKEN` is configured in the environment,
    requests must provide either a matching `X-API-Key` header or
    `Authorization: Bearer <token>` header. Public health and OpenAPI documentation
    endpoints (`/api/health`, `/api/ready`, `/docs`, `/openapi.json`) are exempt.

    Args:
        request: The incoming FastAPI HTTP request.

    Raises:
        UnauthorizedApiError: 401 Unauthorized if auth fails or token is missing.

    """
    required_key = os.environ.get("T1D_API_KEY") or os.environ.get(
        "T1D_API_BEARER_TOKEN"
    )
    if not required_key:
        return

    exempt_paths = {"/api/health", "/api/ready", "/docs", "/openapi.json"}
    if request.url.path in exempt_paths or not request.url.path.startswith("/api/"):
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
    limiter_check_fn: Callable[[str, int, float], bool] | None = None,
    limiter: BaseRateLimiter | None = None,
) -> None:
    """
    Enforce sliding-window rate limiting on a specific endpoint.

    Args:
        request: The incoming FastAPI HTTP request.
        limit: Maximum requests allowed in the window (default: 120).
        window_seconds: Window size in seconds (default: 60.0).
        limiter_check_fn: Optional explicit callable checking rate-limit state.
        limiter: Optional isolated `BaseRateLimiter` instance.

    """
    _enforce_rate_limit(
        request,
        limit=limit,
        window_seconds=window_seconds,
        limiter_check_fn=limiter_check_fn,
        limiter=limiter,
    )


def parse_error(e: Exception) -> ApiError:
    """
    Parse an exception into a structured `ApiError`.

    Args:
        e: The exception to parse.

    Returns:
        ApiError: The resulting `ApiError` object.

    """
    detail = exception_to_error_detail(e)
    return ApiError(error_code=detail["error_code"], params=detail["params"])


def execute_sql(
    db_path: str,
    query: str,
    max_rows: int = 10000,
    timeout_seconds: float = 30.0,
    connect_fn: Callable[..., duckdb.DuckDBPyConnection] | None = None,
    timer_factory: Callable[
        [float, Callable[[], None]], QueryTimerProtocol
    ] = threading.Timer,
) -> list[dict[str, SqlValue]]:
    """
    Execute a read-only SQL query and return results as a list of dictionaries.

    Args:
        db_path: Path to the DuckDB database file.
        query: SQL query to execute.
        max_rows: Maximum rows to return to prevent memory exhaustion.
        timeout_seconds: Execution timeout in seconds.
        connect_fn: Optional DuckDB connection factory callable (defaults to `duckdb.connect`).
        timer_factory: Injectable timer constructor `(timeout_seconds, callback) -> QueryTimerProtocol`.

    Returns:
        List[Dict[str, SqlValue]]: List of dictionaries representing the rows returned.

    """
    return execute_duckdb_query(
        db_path=db_path,
        query=query,
        max_rows=max_rows,
        timeout_seconds=timeout_seconds,
        connect_fn=connect_fn or duckdb.connect,
        timer_factory=timer_factory,
    )


@contextmanager
def get_sessions_conn(
    db_path: str | None = None,
    conn_opener: Callable[..., duckdb.DuckDBPyConnection] = open_sessions_db_conn,
) -> Iterator[duckdb.DuckDBPyConnection]:
    """
    Context manager yielding a persistent chat sessions database connection and ensuring it closes.

    Args:
        db_path: Optional path to the sessions database.
        conn_opener: Callable opening a DuckDB connection to the sessions database.

    Yields:
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
        target_app: FastAPI application instance.
        default_store: Fallback `SessionStore` instance.

    Returns:
        SessionStore: Active conversation session store.

    """
    store = getattr(target_app.state, "session_store", None)
    return store if isinstance(store, SessionStore) else default_store


def resolve_training_job_manager(target_app: FastAPI) -> TrainingJobManager:
    """
    Resolve the active `TrainingJobManager` from `target_app.state` or fall back to the global singleton.

    Args:
        target_app: FastAPI application instance.

    Returns:
        TrainingJobManager: Active background training job manager.

    """
    mgr = getattr(target_app.state, "training_job_manager", None)
    return mgr if isinstance(mgr, TrainingJobManager) else global_training_job_manager


def resolve_chat_job_manager(target_app: FastAPI) -> ChatTurnJobManager:
    """
    Resolve the active `ChatTurnJobManager` from `target_app.state` or fall back to the global singleton.

    Args:
        target_app: FastAPI application instance.

    Returns:
        ChatTurnJobManager: Active background chat job manager.

    """
    mgr = getattr(target_app.state, "chat_job_manager", None)
    return mgr if isinstance(mgr, ChatTurnJobManager) else global_chat_job_manager


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
    ) -> list[dict[str, SqlValue]]:
        """Execute a read-only SQL query and return sanitized row dictionaries."""


class NlToSqlTranslatorCallable(Protocol):
    """Structural protocol for natural-language-to-SQL translation callables."""

    def __call__(
        self,
        db_path: str,
        user_query: str,
        model_name: str = "gemma4",
        history: list[ChatMessageModel] | None = None,
        api_key: str | None = None,
        provider: str | None = None,
    ) -> TranslationResult:
        """Translate a natural-language clinical question into a `TranslationResult`."""


class LlmTokenStreamerCallable(Protocol):
    """Structural protocol for streaming LLM token generators."""

    def __call__(
        self,
        prompt: str,
        model_name: str = "gemma4",
        api_key: str | None = None,
        provider: str | None = None,
    ) -> Iterator[str]:
        """Yield streamed token strings from the configured LLM provider."""


class ChatEventStreamerCallable(Protocol):
    """Structural protocol for SSE chat event stream generators."""

    def __call__(
        self,
        request: ChatRequest,
        req: Request | None = None,
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


def _default_stream_chat_events(
    request: ChatRequest,
    req: Request | None = None,
) -> AsyncIterator[str]:
    """
    Stream SSE chat events using the default `stream_chat_events` implementation.

    Args:
        request: Incoming `ChatRequest`.
        req: Optional FastAPI `Request` for disconnect checks.

    Returns:
        AsyncIterator[str]: SSE event frame stream.

    """
    from t1d_analytics.routers.chat_routes import stream_chat_events

    return stream_chat_events(request=request, req=req)


def _default_build_schema_response(target_db: str, lang: str = "en") -> SchemaResponse:
    """
    Build a `SchemaResponse` using the default table-route schema inspector.

    Args:
        target_db: Validated DuckDB database path.
        lang: Two-letter ISO locale code.

    Returns:
        SchemaResponse: Structured schema metadata and health dictionary.

    """
    from t1d_analytics.routers.table_routes import _build_schema_response

    return _build_schema_response(target_db=target_db, lang=lang)


def _default_translate_nl_to_sql(
    db_path: str,
    user_query: str,
    model_name: str = "gemma4",
    history: list[ChatMessageModel] | None = None,
    api_key: str | None = None,
    provider: str | None = None,
) -> TranslationResult:
    """
    Translate `user_query` into DuckDB SQL using canonical `chat_service.translate_nl_to_sql`.

    Args:
        db_path: Path to DuckDB database file.
        user_query: Natural-language clinical question.
        model_name: Target model name.
        history: Optional prior conversation messages.
        api_key: Optional provider API key.
        provider: Optional explicit provider name.

    Returns:
        TranslationResult: Validated or fallback translation result.

    """
    return translate_nl_to_sql(
        db_path,
        user_query,
        model_name=model_name,
        history=history,
        api_key=api_key,
        provider=provider,
    )


def _default_stream_llm_tokens(
    prompt: str,
    model_name: str = "gemma4",
    api_key: str | None = None,
    provider: str | None = None,
) -> Iterator[str]:
    """
    Stream LLM completion tokens using canonical `chat_service.stream_llm_completion_tokens`.

    Args:
        prompt: Prompt string.
        model_name: Target model name.
        api_key: Optional provider API key.
        provider: Optional explicit provider name.

    Returns:
        Iterator[str]: Token iterator.

    """
    return stream_llm_completion_tokens(
        prompt,
        model_name=model_name,
        api_key=api_key,
        provider=provider,
    )


def _default_duckdb_connect(
    database: str = ":memory:", read_only: bool = False
) -> duckdb.DuckDBPyConnection:
    """
    Open a DuckDB connection with standard `(database, read_only)` arguments.

    Args:
        database: Database file path or `':memory:'`.
        read_only: Whether to open in read-only mode.

    Returns:
        duckdb.DuckDBPyConnection: Opened DuckDB connection.

    """
    return duckdb.connect(database, read_only=read_only)


@dataclass
class AppServices:
    """
    Typed dependency bundle of backend service callables and per-app stateful stores.

    Supports explicit dependency injection via FastAPI `dependency_overrides` or
    per-application isolation via `create_app()`.
    """

    validate_db_path: Callable[[str | None], str] = validate_db_path
    validate_sessions_db_path: Callable[[str | None], str] = validate_sessions_db_path
    validate_workspace_artifact_path: Callable[[str], str] = (
        validate_workspace_artifact_path
    )
    discover_workspace_databases: Callable[[], DatabaseListResponse] = (
        discover_workspace_databases
    )
    enforce_rate_limit: RateLimitEnforcerCallable = enforce_rate_limit
    rate_limiter: BaseRateLimiter = field(default_factory=MemoryRateLimiter)
    execute_sql: SqlExecutorCallable = execute_sql
    translate_nl_to_sql: NlToSqlTranslatorCallable = _default_translate_nl_to_sql
    stream_llm_completion_tokens: LlmTokenStreamerCallable = _default_stream_llm_tokens
    stream_chat_events: ChatEventStreamerCallable = _default_stream_chat_events
    parse_error: Callable[[Exception], ApiError] = parse_error
    get_database_schema: Callable[[duckdb.DuckDBPyConnection], str] = (
        get_database_schema
    )
    profile_table: Callable[[str, str], TableProfileResult] = profile_table
    duckdb_connect: DuckDbConnectCallable = _default_duckdb_connect
    get_request_locale: Callable[[Request], str] = get_request_locale
    cloud_provider_models: Mapping[str, Sequence[str]] = field(
        default_factory=lambda: CLOUD_MODEL_CATALOG
    )
    session_store: SessionStore = field(
        default_factory=lambda: SessionStore(
            conn_factory=get_sessions_conn, lock=threading.Lock()
        )
    )
    get_session_store: Callable[[], SessionStore] = field(
        default_factory=lambda: (
            lambda: SessionStore(conn_factory=get_sessions_conn, lock=threading.Lock())
        )
    )
    training_job_manager: TrainingJobManager = global_training_job_manager
    get_training_job_manager: Callable[[], TrainingJobManager] = lambda: (
        global_training_job_manager
    )
    chat_job_manager: ChatTurnJobManager = global_chat_job_manager
    get_chat_job_manager: Callable[[], ChatTurnJobManager] = lambda: (
        global_chat_job_manager
    )
    build_schema_response: Callable[[str, str], SchemaResponse] = (
        _default_build_schema_response
    )


class AppServicesRegistry:
    """Thread-safe registry resolving fallback `AppServices` when invoked outside a FastAPI request."""

    def __init__(self) -> None:
        """Initialize an empty thread-safe `AppServicesRegistry`."""
        self._lock = threading.Lock()
        self._bound_app: FastAPI | None = None
        self._default_services: AppServices | None = None

    def set_default(
        self,
        services: AppServices | None,
        target_app: FastAPI | None = None,
    ) -> None:
        """
        Store or clear the default `AppServices` instance and optional bound FastAPI application.

        Args:
            services: Optional `AppServices` instance.
            target_app: Optional FastAPI application instance.

        """
        with self._lock:
            if target_app is not None:
                self._bound_app = target_app
                target_app.state.services = services
            self._default_services = services

    def get_default(self) -> AppServices | None:
        """
        Return the active `AppServices` from `_default_services` or `bound_app.state.services`.

        Returns:
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
    target_app: FastAPI | None = None,
    session_store: SessionStore | None = None,
    training_job_manager: TrainingJobManager | None = None,
    chat_job_manager: ChatTurnJobManager | None = None,
    rate_limiter: BaseRateLimiter | None = None,
    session_store_getter: Callable[[], SessionStore] | None = None,
    job_manager_getter: Callable[[], TrainingJobManager] | None = None,
    training_job_manager_getter: Callable[[], TrainingJobManager] | None = None,
    chat_job_manager_getter: Callable[[], ChatTurnJobManager] | None = None,
) -> AppServices:
    """
    Construct a strongly typed `AppServices` bundle bound to isolated or default service instances.

    Args:
        target_app: Optional FastAPI application instance.
        session_store: Optional isolated `SessionStore` instance.
        training_job_manager: Optional isolated `TrainingJobManager` instance.
        chat_job_manager: Optional isolated `ChatTurnJobManager` instance.
        rate_limiter: Optional isolated `BaseRateLimiter` instance.
        session_store_getter: Optional callable returning active `SessionStore`.
        job_manager_getter: Optional callable returning active `TrainingJobManager`.
        training_job_manager_getter: Optional callable returning active `TrainingJobManager`.
        chat_job_manager_getter: Optional callable returning active `ChatTurnJobManager`.

    Returns:
        AppServices: Newly constructed service bundle.

    """
    _ = target_app
    active_store = (
        session_store
        if session_store is not None
        else SessionStore(conn_factory=get_sessions_conn, lock=threading.Lock())
    )
    active_train_mgr = (
        training_job_manager
        if training_job_manager is not None
        else global_training_job_manager
    )
    active_chat_mgr = (
        chat_job_manager if chat_job_manager is not None else global_chat_job_manager
    )
    active_limiter = rate_limiter if rate_limiter is not None else MemoryRateLimiter()
    resolved_train_getter = (
        training_job_manager_getter or job_manager_getter or (lambda: active_train_mgr)
    )
    resolved_chat_getter = chat_job_manager_getter or (lambda: active_chat_mgr)

    svc_holder: list[AppServices] = []

    def _bound_stream_chat(
        request: ChatRequest, req: Request | None = None
    ) -> AsyncIterator[str]:
        """Delegate SSE streaming with the bound `AppServices` instance."""
        from t1d_analytics.routers.chat_routes import stream_chat_events

        return stream_chat_events(request=request, req=req, services=svc_holder[0])

    def _bound_build_schema(target_db: str, lang: str = "en") -> SchemaResponse:
        """Delegate schema inspection with the bound `AppServices` instance."""
        from t1d_analytics.routers.table_routes import _build_schema_response

        return _build_schema_response(
            target_db=target_db, lang=lang, services=svc_holder[0]
        )

    svc = AppServices(
        validate_db_path=validate_db_path,
        validate_sessions_db_path=validate_sessions_db_path,
        validate_workspace_artifact_path=validate_workspace_artifact_path,
        discover_workspace_databases=discover_workspace_databases,
        enforce_rate_limit=lambda request, limit=120, window_seconds=60.0: (
            enforce_rate_limit(
                request,
                limit=limit,
                window_seconds=window_seconds,
                limiter=active_limiter,
            )
        ),
        rate_limiter=active_limiter,
        execute_sql=lambda db_path, query, max_rows=10000, timeout_seconds=30.0: (
            execute_sql(
                db_path=db_path,
                query=query,
                max_rows=max_rows,
                timeout_seconds=timeout_seconds,
                connect_fn=duckdb.connect,
            )
        ),
        translate_nl_to_sql=_default_translate_nl_to_sql,
        stream_llm_completion_tokens=_default_stream_llm_tokens,
        stream_chat_events=_bound_stream_chat,
        parse_error=parse_error,
        get_database_schema=lambda conn: get_database_schema(conn),
        profile_table=lambda db_p, tbl: profile_table(db_p, tbl),
        duckdb_connect=_default_duckdb_connect,
        get_request_locale=get_request_locale,
        cloud_provider_models=CLOUD_MODEL_CATALOG,
        session_store=active_store,
        get_session_store=session_store_getter or (lambda: active_store),
        training_job_manager=active_train_mgr,
        get_training_job_manager=resolved_train_getter,
        chat_job_manager=active_chat_mgr,
        get_chat_job_manager=resolved_chat_getter,
        build_schema_response=_bound_build_schema,
    )
    svc_holder.append(svc)
    return svc


def set_default_app_services(
    services: AppServices | None,
    target_app: FastAPI | None = None,
) -> None:
    """
    Register or clear the active fallback `AppServices` container on `target_app.state.services` and the registry.

    Args:
        services: Optional `AppServices` instance to set as the active default.
        target_app: Optional FastAPI application instance whose `state.services` should be updated.

    """
    _app_services_registry.set_default(services, target_app=target_app)


def register_default_app_services(services: AppServices) -> None:
    """
    Register the fallback `AppServices` bundle for non-HTTP direct helper invocations.

    Args:
        services: Configured `AppServices` instance.

    """
    _app_services_registry.set_default(services)


def resolve_default_app_services() -> AppServices:
    """
    Resolve the fallback `AppServices` bundle when invoked outside a FastAPI HTTP request.

    Returns:
        AppServices: Resolved service bundle.

    """
    return _app_services_registry.get_default() or create_default_app_services()


def get_app_services(request: Request) -> AppServices:
    """
    FastAPI dependency resolving `AppServices` from `request.app.state.services` per application instance.

    Args:
        request: Active FastAPI HTTP request.

    Returns:
        AppServices: Per-application service bundle from `request.app.state.services` or fallback.

    """
    app_svc = getattr(request.app.state, "services", None)
    if isinstance(app_svc, AppServices):
        return app_svc
    return resolve_default_app_services()


__all__ = [
    "AppServices",
    "AppServicesRegistry",
    "BaseRateLimiter",
    "ChatEventStreamerCallable",
    "DuckDbConnectCallable",
    "DuckDbRateLimiter",
    "LlmTokenStreamerCallable",
    "MemoryRateLimiter",
    "NlToSqlTranslatorCallable",
    "RateLimitEnforcerCallable",
    "SqlExecutorCallable",
    "check_rate_limit",
    "configure_api_logging",
    "create_default_app_services",
    "discover_workspace_databases",
    "enforce_rate_limit",
    "execute_sql",
    "get_allowed_db_dirs",
    "get_app_services",
    "get_cors_origins",
    "get_database_schema",
    "get_rate_limiter",
    "get_request_locale",
    "get_sessions_conn",
    "parse_error",
    "register_default_app_services",
    "resolve_chat_job_manager",
    "resolve_default_app_services",
    "resolve_session_store",
    "resolve_training_job_manager",
    "set_default_app_services",
    "validate_db_path",
    "validate_sessions_db_path",
    "validate_workspace_artifact_path",
    "verify_api_auth",
]
