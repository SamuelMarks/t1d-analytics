"""FastAPI application for the T1D Analytics web interface."""

import logging
import os
import threading
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import (
    AsyncGenerator,
    Awaitable,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
)

import duckdb as duckdb
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from t1d_analytics.analytics import (
    get_database_schema as get_database_schema,
)
from t1d_analytics.analytics import (
    profile_table as profile_table,
)
from t1d_analytics.api_dependencies import (
    BaseRateLimiter as BaseRateLimiter,
)
from t1d_analytics.api_dependencies import (
    DuckDbRateLimiter as DuckDbRateLimiter,
)
from t1d_analytics.api_dependencies import (
    MemoryRateLimiter as MemoryRateLimiter,
)
from t1d_analytics.api_dependencies import (
    check_rate_limit as check_rate_limit,
)
from t1d_analytics.api_dependencies import (
    configure_api_logging as configure_api_logging,
)
from t1d_analytics.api_dependencies import (
    discover_workspace_databases as discover_workspace_databases,
)
from t1d_analytics.api_dependencies import (
    enforce_rate_limit as _enforce_rate_limit_dep,
)
from t1d_analytics.api_dependencies import (
    execute_sql as _execute_sql_dep,
)
from t1d_analytics.api_dependencies import (
    get_allowed_db_dirs as get_allowed_db_dirs,
)
from t1d_analytics.api_dependencies import (
    get_cors_origins as get_cors_origins,
)
from t1d_analytics.api_dependencies import (
    get_rate_limiter as get_rate_limiter,
)
from t1d_analytics.api_dependencies import (
    get_request_locale as get_request_locale,
)
from t1d_analytics.api_dependencies import (
    get_sessions_conn as _get_sessions_conn_dep,
)
from t1d_analytics.api_dependencies import (
    parse_error as parse_error,
)
from t1d_analytics.api_dependencies import (
    resolve_session_store,
    resolve_training_job_manager,
)
from t1d_analytics.api_dependencies import (
    validate_db_path as validate_db_path,
)
from t1d_analytics.api_dependencies import (
    validate_workspace_artifact_path as validate_workspace_artifact_path,
)
from t1d_analytics.api_dependencies import (
    verify_api_auth as verify_api_auth,
)
from t1d_analytics.api_schemas import (
    ApiError as ApiError,
)
from t1d_analytics.api_schemas import (
    ChatMessageModel as ChatMessageModel,
)
from t1d_analytics.api_schemas import (
    ChatRequest as ChatRequest,
)
from t1d_analytics.api_schemas import (
    ChatResponse as ChatResponse,
)
from t1d_analytics.api_schemas import (
    ColumnInfo as ColumnInfo,
)
from t1d_analytics.api_schemas import (
    CreateSessionRequest as CreateSessionRequest,
)
from t1d_analytics.api_schemas import (
    DatabaseItem as DatabaseItem,
)
from t1d_analytics.api_schemas import (
    DatabaseListResponse as DatabaseListResponse,
)
from t1d_analytics.api_schemas import (
    ExecuteSqlRequest as ExecuteSqlRequest,
)
from t1d_analytics.api_schemas import (
    ExecuteSqlResponse as ExecuteSqlResponse,
)
from t1d_analytics.api_schemas import (
    ModelInfo as ModelInfo,
)
from t1d_analytics.api_schemas import (
    ModelsResponse as ModelsResponse,
)
from t1d_analytics.api_schemas import (
    ProvidersStatusResponse as ProvidersStatusResponse,
)
from t1d_analytics.api_schemas import (
    ProviderStatus as ProviderStatus,
)
from t1d_analytics.api_schemas import (
    SchemaResponse as SchemaResponse,
)
from t1d_analytics.api_schemas import (
    SessionListResponse as SessionListResponse,
)
from t1d_analytics.api_schemas import (
    SessionModel as SessionModel,
)
from t1d_analytics.api_schemas import (
    StatusResponse as StatusResponse,
)
from t1d_analytics.api_schemas import (
    TableDataResponse as TableDataResponse,
)
from t1d_analytics.api_schemas import (
    TableInfo as TableInfo,
)
from t1d_analytics.api_schemas import (
    TrainingJobRequest as TrainingJobRequest,
)
from t1d_analytics.chat_service import (
    build_sql_prompt_with_history,
    stream_llm_completion_tokens,
)
from t1d_analytics.chat_service import (
    generate_sql_from_nl as generate_sql_from_nl,
)
from t1d_analytics.diagnostics import (
    check_database_health,
    log_startup_diagnostics,
    run_system_diagnostics,
)
from t1d_analytics.errors import T1DError
from t1d_analytics.i18n import get_translator
from t1d_analytics.ingestion import load_data_to_duckdb
from t1d_analytics.job_manager import TrainingJobManager
from t1d_analytics.llm_client import (
    CLOUD_MODEL_CATALOG,
)
from t1d_analytics.llm_client import (
    extract_stream_delta as extract_stream_delta,
)
from t1d_analytics.llm_client import (
    is_provider_configured as is_provider_configured,
)
from t1d_analytics.llm_client import (
    read_configured_ollama_model_list as read_configured_ollama_model_list,
)
from t1d_analytics.llm_client import (
    resolve_provider_and_model as resolve_provider_and_model,
)
from t1d_analytics.llm_client import (
    sort_ollama_models as sort_ollama_models,
)
from t1d_analytics.rate_limiter import close_all_rate_limiters
from t1d_analytics.routers.chat_routes import (
    chat_endpoint as chat_endpoint,
)
from t1d_analytics.routers.chat_routes import (
    chat_router as chat_router,
)
from t1d_analytics.routers.chat_routes import (
    chat_stream_endpoint as chat_stream_endpoint,
)
from t1d_analytics.routers.chat_routes import (
    execute_sql_endpoint as execute_sql_endpoint,
)
from t1d_analytics.routers.chat_routes import (
    stream_chat_events as stream_chat_events,
)
from t1d_analytics.routers.session_routes import (
    delete_session as delete_session,
)
from t1d_analytics.routers.session_routes import (
    get_session as get_session,
)
from t1d_analytics.routers.session_routes import (
    list_sessions as list_sessions,
)
from t1d_analytics.routers.session_routes import (
    save_session as save_session,
)
from t1d_analytics.routers.session_routes import (
    sessions_router as sessions_router,
)
from t1d_analytics.routers.status_routes import (
    get_health as get_health,
)
from t1d_analytics.routers.status_routes import (
    get_providers_status as get_providers_status,
)
from t1d_analytics.routers.status_routes import (
    get_ready as get_ready,
)
from t1d_analytics.routers.status_routes import (
    get_status as get_status,
)
from t1d_analytics.routers.status_routes import (
    list_models as list_models,
)
from t1d_analytics.routers.status_routes import (
    status_router as status_router,
)
from t1d_analytics.routers.table_routes import (
    _build_schema_response as _build_schema_response,
)
from t1d_analytics.routers.table_routes import (
    export_csv as export_csv,
)
from t1d_analytics.routers.table_routes import (
    export_excel as export_excel,
)
from t1d_analytics.routers.table_routes import (
    export_json as export_json,
)
from t1d_analytics.routers.table_routes import (
    export_parquet as export_parquet,
)
from t1d_analytics.routers.table_routes import (
    get_schema as get_schema,
)
from t1d_analytics.routers.table_routes import (
    get_table_data as get_table_data,
)
from t1d_analytics.routers.table_routes import (
    get_table_profile as get_table_profile,
)
from t1d_analytics.routers.table_routes import (
    list_databases as list_databases,
)
from t1d_analytics.routers.table_routes import (
    tables_router as tables_router,
)
from t1d_analytics.routers.training_routes import (
    cancel_training_job as cancel_training_job,
)
from t1d_analytics.routers.training_routes import (
    get_training_job as get_training_job,
)
from t1d_analytics.routers.training_routes import (
    list_training_jobs as list_training_jobs,
)
from t1d_analytics.routers.training_routes import (
    submit_training_job as submit_training_job,
)
from t1d_analytics.routers.training_routes import (
    training_router as training_router,
)
from t1d_analytics.seed import ensure_canonical_clinical_views, seed_database
from t1d_analytics.session_store import SessionStore, open_sessions_db_conn
from t1d_analytics.sql_utils import (
    SqlValue,
    close_read_only_duckdb_pool,
    extract_sql_from_response,
)
from t1d_analytics.sql_utils import (
    sanitize_spreadsheet_cell as sanitize_spreadsheet_cell,
)
from t1d_analytics.sql_utils import (
    sanitize_sql_value as sanitize_sql_value,
)

_sessions_lock = threading.Lock()
_get_sessions_db_conn = open_sessions_db_conn
_build_sql_prompt = build_sql_prompt_with_history
stream_llm_tokens = stream_llm_completion_tokens
_extract_sql_from_response = extract_sql_from_response
CLOUD_PROVIDER_MODELS: Dict[str, List[str]] = CLOUD_MODEL_CATALOG

configure_api_logging()
logger = logging.getLogger(__name__)


def enforce_rate_limit(
    request: Request, limit: int = 120, window_seconds: float = 60.0
) -> None:
    """
    Enforce sliding-window rate limiting on a specific endpoint.

    Args:
    ----
        request: The incoming FastAPI HTTP request.
        limit: Maximum requests allowed in the window (default: 120).
        window_seconds: Window size in seconds (default: 60.0).

    """
    _enforce_rate_limit_dep(
        request,
        limit=limit,
        window_seconds=window_seconds,
        limiter_check_fn=check_rate_limit,
    )


def execute_sql(
    db_path: str,
    query: str,
    max_rows: int = 10000,
    timeout_seconds: float = 30.0,
) -> List[Dict[str, SqlValue]]:
    """
    Execute a SQL query and return results as a list of dictionaries.

    Args:
    ----
        db_path: Path to the DuckDB database file.
        query: SQL query to execute.
        max_rows: Maximum rows to return to prevent memory exhaustion.
        timeout_seconds: Execution timeout in seconds.

    Returns:
    -------
        List[Dict[str, SqlValue]]: List of dictionaries representing the rows returned.

    """
    return _execute_sql_dep(
        db_path=db_path,
        query=query,
        max_rows=max_rows,
        timeout_seconds=timeout_seconds,
        connect_fn=duckdb.connect,
    )


@contextmanager
def get_sessions_conn(
    db_path: Optional[str] = None,
) -> Iterator[duckdb.DuckDBPyConnection]:
    """
    Context manager yielding a persistent chat sessions database connection and ensuring it closes.

    Args:
    ----
        db_path: Optional path to the sessions database.

    Yields:
    ------
        duckdb.DuckDBPyConnection: Database connection guaranteed to be closed upon exit.

    """
    with _get_sessions_conn_dep(
        db_path=db_path,
        conn_opener=_get_sessions_db_conn,
    ) as conn:
        yield conn


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
        except Exception as e:
            logger.warning(f"Startup auto-ingestion failed for {db_path}: {e}")
    run_system_diagnostics(db_path=db_path)
    log_startup_diagnostics(db_path=db_path)
    _setup_debug(app)
    try:
        yield
    finally:
        close_read_only_duckdb_pool()
        close_all_rate_limiters()


_lifespan = lifespan

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


app.include_router(sessions_router)
app.include_router(chat_router)
app.include_router(tables_router)
app.include_router(status_router)
app.include_router(training_router)


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
        dist_path = (
            Path(env_dist)
            if env_dist
            else Path(__file__).resolve().parents[3] / "t1d-analytics-web" / "dist"
        )
        if os.path.exists(str(dist_path)):
            logger.info(f"DEBUG mode enabled. Serving static frontend from {dist_path}")
            target_app.mount(
                "/", StaticFiles(directory=str(dist_path), html=True), name="static"
            )
        else:
            logger.warning(
                f"DEBUG mode enabled but static dist folder not found at {dist_path}"
            )
