"""FastAPI application for the T1D Analytics web interface."""

import logging
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import (
    AsyncGenerator,
    Awaitable,
    Callable,
    Dict,
    List,
)

import duckdb as duckdb
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from t1d_analytics.analytics import get_database_schema, profile_table
from t1d_analytics.api_dependencies import (
    AppServices,
    BaseRateLimiter,
    DuckDbRateLimiter,
    MemoryRateLimiter,
    check_rate_limit,
    configure_api_logging,
    create_default_app_services,
    discover_workspace_databases,
    enforce_rate_limit,
    execute_sql,
    get_allowed_db_dirs,
    get_app_services,
    get_cors_origins,
    get_rate_limiter,
    get_request_locale,
    get_sessions_conn,
    parse_error,
    resolve_chat_job_manager,
    resolve_session_store,
    resolve_training_job_manager,
    set_default_app_services,
    validate_db_path,
    validate_workspace_artifact_path,
    verify_api_auth,
)
from t1d_analytics.api_schemas import (
    ApiError,
    ChatMessageModel,
    ChatRequest,
    ChatResponse,
    ColumnInfo,
    CreateSessionRequest,
    DatabaseItem,
    DatabaseListResponse,
    ExecuteSqlRequest,
    ExecuteSqlResponse,
    ModelInfo,
    ModelsResponse,
    ProvidersStatusResponse,
    ProviderStatus,
    SchemaResponse,
    SessionListResponse,
    SessionModel,
    StatusResponse,
    TableDataResponse,
    TableInfo,
    TrainingJobRequest,
)
from t1d_analytics.chat_service import (
    build_sql_prompt_with_history,
    stream_llm_completion_tokens,
    translate_nl_to_sql,
)
from t1d_analytics.diagnostics import (
    check_database_health,
    log_startup_diagnostics,
    run_system_diagnostics,
)
from t1d_analytics.errors import T1DError
from t1d_analytics.i18n import get_translator
from t1d_analytics.ingestion import load_data_to_duckdb
from t1d_analytics.job_manager import ChatJobManager, TrainingJobManager
from t1d_analytics.llm_client import (
    CLOUD_MODEL_CATALOG,
    extract_stream_delta,
    is_provider_configured,
    read_configured_ollama_model_list,
    resolve_provider_and_model,
    sort_ollama_models,
)
from t1d_analytics.rate_limiter import close_all_rate_limiters
from t1d_analytics.routers.chat_routes import (
    chat_endpoint,
    chat_router,
    chat_stream_endpoint,
    execute_sql_endpoint,
    stream_chat_events,
)
from t1d_analytics.routers.session_routes import (
    delete_session,
    get_session,
    list_sessions,
    save_session,
    sessions_router,
)
from t1d_analytics.routers.status_routes import (
    get_health,
    get_providers_status,
    get_ready,
    get_status,
    list_models,
    status_router,
)
from t1d_analytics.routers.table_routes import (
    _build_schema_response,
    export_csv,
    export_excel,
    export_json,
    export_parquet,
    get_schema,
    get_table_data,
    get_table_profile,
    list_databases,
    tables_router,
)
from t1d_analytics.routers.training_routes import (
    cancel_training_job,
    get_training_job,
    list_training_jobs,
    submit_training_job,
    training_router,
)
from t1d_analytics.seed import ensure_canonical_clinical_views, seed_database
from t1d_analytics.session_store import SessionStore, open_sessions_db_conn
from t1d_analytics.sql_utils import (
    close_read_only_duckdb_pool,
    extract_sql_from_response,
    sanitize_spreadsheet_cell,
    sanitize_sql_value,
)

generate_sql_from_nl = translate_nl_to_sql
stream_llm_tokens = stream_llm_completion_tokens
CLOUD_PROVIDER_MODELS: Dict[str, List[str]] = CLOUD_MODEL_CATALOG

configure_api_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Lifespan context manager that auto-populates an empty database and logs startup diagnostics.

    Args:
    ----
        app: The FastAPI application instance.

    Yields:
    ------
        None upon server startup.

    """
    db_path = os.environ.get("T1D_DB_PATH") or "data/t1d.duckdb"
    data_dir = os.environ.get("T1D_DATA_DIR") or "data"
    sessions_db_path = os.environ.get("T1D_SESSIONS_DB_PATH") or "data/sessions.duckdb"
    db_health = check_database_health(db_path=db_path)
    if (
        not db_health.exists
        or db_health.table_count == 0
        or db_health.total_rows == 0
        or not db_health.has_initial_data
    ):
        try:
            seed_database(
                data_dir=data_dir,
                db_path=db_path,
                sessions_db_path=sessions_db_path,
            )
            load_data_to_duckdb(data_dir, db_path)
            ensure_canonical_clinical_views(db_path)
        except (T1DError, duckdb.Error, OSError, RuntimeError, ValueError) as e:
            logger.warning(f"Startup auto-ingestion failed for {db_path}: {e}")
    run_system_diagnostics(db_path=db_path)
    log_startup_diagnostics(db_path=db_path)
    _setup_debug(app)
    try:
        yield
    finally:
        close_read_only_duckdb_pool()
        close_all_rate_limiters()


app = FastAPI(
    title="T1D Analytics API",
    description="API for querying T1D datasets.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def auth_and_rate_limit_middleware(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    """
    Global middleware enforcing API key verification when T1D_API_KEY is configured.

    Args:
    ----
        request: The incoming HTTP request.
        call_next: The next request handler.

    Returns:
    -------
        Response: The response produced by the application or an auth error JSON response.

    """
    try:
        verify_api_auth(request)
    except T1DError as exc:
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.to_dict()},
            headers=exc.headers,
        )
    return await call_next(request)


@app.exception_handler(T1DError)
async def t1d_error_exception_handler(request: Request, exc: T1DError) -> JSONResponse:
    """
    Handle structured T1DError domain exceptions and return a JSONResponse.

    Args:
    ----
        request: The incoming FastAPI HTTP request.
        exc: The raised T1DError instance.

    Returns:
    -------
        JSONResponse: Response with the domain error's status_code and structured detail dictionary.

    """
    _ = request
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.to_dict()},
        headers=exc.headers,
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Handle unhandled exceptions globally and return localized error message.

    Args:
    ----
        request: The incoming FastAPI HTTP request.
        exc: The unhandled exception.

    Returns:
    -------
        JSONResponse: Status 500 and localized error payload.

    """
    lang = get_request_locale(request)
    _ = get_translator(lang)

    logger.exception(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "detail": {
                "error_code": "backend.serverError",
                "params": {"error": _("Internal server error: {}", exc)},
            }
        },
    )


_sessions_lock = threading.Lock()
_session_store = SessionStore(
    conn_factory=get_sessions_conn,
    lock=_sessions_lock,
)
app.state.sessions_lock = _sessions_lock
app.state.session_store = _session_store


def get_session_store() -> SessionStore:
    """
    FastAPI dependency provider returning the active SessionStore instance from `app.state`.

    Returns
    -------
        SessionStore: Active conversation session store.

    """
    return resolve_session_store(app, _session_store)


def get_training_job_manager() -> TrainingJobManager:
    """
    FastAPI dependency provider returning the active TrainingJobManager instance from `app.state`.

    Returns
    -------
        TrainingJobManager: Active background training job manager.

    """
    return resolve_training_job_manager(app)


def get_chat_job_manager() -> ChatJobManager:
    """
    FastAPI dependency provider returning the active ChatJobManager instance from `app.state`.

    Returns
    -------
        ChatJobManager: Active background chat job manager.

    """
    return resolve_chat_job_manager(app)


app.state.services = create_default_app_services(
    target_app=app,
    session_store_getter=get_session_store,
    job_manager_getter=get_training_job_manager,
    chat_job_manager_getter=get_chat_job_manager,
)
set_default_app_services(app.state.services, target_app=app)


app.include_router(sessions_router)
app.include_router(chat_router)
app.include_router(tables_router)
app.include_router(status_router)
app.include_router(training_router)


_FRONTEND_DIST: Path = (
    Path(__file__).resolve().parents[3] / "t1d-analytics-web" / "dist"
)


@app.get("/chats/{chat_id}", include_in_schema=False, response_model=None)
async def serve_chat_spa_entrypoint(chat_id: str) -> Response:
    """
    Serve the compiled SPA `index.html` entrypoint for shareable `/chats/{chat_id}` deep-links.

    Args:
    ----
        chat_id: Target chat session identifier from the URL path.

    Returns:
    -------
        Response: `FileResponse` containing the frontend `index.html`.

    Raises:
    ------
        HTTPException: Status 404 when the compiled frontend `index.html` is not present on disk.

    """
    _ = chat_id
    env_dist = os.environ.get("T1D_WEB_DIST_DIR")
    dist_dir = Path(env_dist) if env_dist else _FRONTEND_DIST
    index_file = dist_dir / "index.html"
    if index_file.is_file():
        return FileResponse(str(index_file), media_type="text/html")
    raise HTTPException(status_code=404, detail="Frontend index.html not found")


def create_app() -> FastAPI:
    """
    Return the configured FastAPI application instance with mounted routers, lifespan, and middleware.

    Returns
    -------
        FastAPI: Configured T1D Analytics FastAPI application.

    """
    return app


def _setup_debug(target_app: FastAPI = app) -> None:
    """
    Configure debug static file serving if enabled.

    When the DEBUG environment variable is set, this function mounts
    the frontend static files (from the t1d-analytics-web/dist directory) at the root
    path. Useful for local development and testing.

    Args:
    ----
        target_app: FastAPI application instance to mount static files onto.

    Returns:
    -------
        None

    """
    if os.environ.get("DEBUG"):
        env_dist = os.environ.get("T1D_WEB_DIST_DIR")
        dist_path = Path(env_dist) if env_dist else _FRONTEND_DIST
        if dist_path.exists():
            logger.info(f"DEBUG mode enabled. Serving static frontend from {dist_path}")
            target_app.mount(
                "/", StaticFiles(directory=str(dist_path), html=True), name="static"
            )
        else:
            logger.warning(
                f"DEBUG mode enabled but static dist folder not found at {dist_path}"
            )


__all__ = [
    "CLOUD_PROVIDER_MODELS",
    "_FRONTEND_DIST",
    "ApiError",
    "AppServices",
    "BaseRateLimiter",
    "ChatJobManager",
    "ChatMessageModel",
    "ChatRequest",
    "ChatResponse",
    "ColumnInfo",
    "CreateSessionRequest",
    "DatabaseItem",
    "DatabaseListResponse",
    "DuckDbRateLimiter",
    "ExecuteSqlRequest",
    "ExecuteSqlResponse",
    "MemoryRateLimiter",
    "ModelInfo",
    "ModelsResponse",
    "ProviderStatus",
    "ProvidersStatusResponse",
    "SchemaResponse",
    "SessionListResponse",
    "SessionModel",
    "StatusResponse",
    "TableDataResponse",
    "TableInfo",
    "TrainingJobRequest",
    "_build_schema_response",
    "_setup_debug",
    "app",
    "auth_and_rate_limit_middleware",
    "build_sql_prompt_with_history",
    "cancel_training_job",
    "chat_endpoint",
    "chat_router",
    "chat_stream_endpoint",
    "check_rate_limit",
    "configure_api_logging",
    "create_app",
    "create_default_app_services",
    "delete_session",
    "discover_workspace_databases",
    "enforce_rate_limit",
    "execute_sql",
    "execute_sql_endpoint",
    "export_csv",
    "export_excel",
    "export_json",
    "export_parquet",
    "extract_sql_from_response",
    "extract_stream_delta",
    "generate_sql_from_nl",
    "get_allowed_db_dirs",
    "get_app_services",
    "get_chat_job_manager",
    "get_cors_origins",
    "get_database_schema",
    "get_health",
    "get_providers_status",
    "get_rate_limiter",
    "get_ready",
    "get_request_locale",
    "get_schema",
    "get_session",
    "get_session_store",
    "get_sessions_conn",
    "get_status",
    "get_table_data",
    "get_table_profile",
    "get_training_job",
    "get_training_job_manager",
    "global_exception_handler",
    "is_provider_configured",
    "lifespan",
    "list_databases",
    "list_models",
    "list_sessions",
    "list_training_jobs",
    "open_sessions_db_conn",
    "parse_error",
    "profile_table",
    "read_configured_ollama_model_list",
    "resolve_provider_and_model",
    "sanitize_spreadsheet_cell",
    "sanitize_sql_value",
    "save_session",
    "serve_chat_spa_entrypoint",
    "sessions_router",
    "set_default_app_services",
    "sort_ollama_models",
    "status_router",
    "stream_chat_events",
    "stream_llm_tokens",
    "submit_training_job",
    "t1d_error_exception_handler",
    "tables_router",
    "training_router",
    "validate_db_path",
    "validate_workspace_artifact_path",
    "verify_api_auth",
]
