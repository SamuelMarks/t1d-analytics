"""FastAPI application for the T1D Analytics web interface."""

import logging
import os
import threading
import urllib.error
import urllib.request
from contextlib import asynccontextmanager, contextmanager
from typing import (
    AsyncGenerator,
    AsyncIterator,
    Awaitable,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Union,
)

import duckdb
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from t1d_analytics.analytics import (
    TableProfileResult,
    get_database_schema,
    load_data_to_duckdb,
    profile_table,
)
from t1d_analytics.analytics import (
    extract_sql_from_response as _extract_sql_from_response,
)
from t1d_analytics.api_exports import (
    fetch_paginated_table_data,
    fetch_table_profile,
    stream_csv_export,
    stream_excel_export,
    stream_json_export,
    stream_parquet_export,
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
    execute_chat_turn,
    generate_sse_chat_events,
    stream_llm_completion_tokens,
)
from t1d_analytics.chat_service import (
    generate_sql_from_nl as generate_sql_from_nl,
)
from t1d_analytics.chat_service import (
    stream_nl_to_sql_events as stream_nl_to_sql_events,
)
from t1d_analytics.db_security import (
    discover_workspace_databases,
    get_allowed_db_dirs,
    validate_db_path,
)
from t1d_analytics.diagnostics import (
    HealthStatus,
    SystemHealthDict,
    check_database_health,
    get_system_health,
    log_startup_diagnostics,
    run_system_diagnostics,
)
from t1d_analytics.errors import (
    InvalidDatabasePathError,
    InvalidTrainingBackendError,
    T1DError,
    TrainingJobNotFoundError,
)
from t1d_analytics.i18n import get_translator
from t1d_analytics.job_manager import TrainingJobDict, TrainingJobManager
from t1d_analytics.llm_client import (
    CLOUD_MODEL_CATALOG,
    extract_stream_delta,
    fetch_available_models,
    is_provider_configured,
    probe_providers_status,
    resolve_provider_and_model,
)
from t1d_analytics.rate_limiter import (
    RATE_LIMIT_LOCK as RATE_LIMIT_LOCK,
)
from t1d_analytics.rate_limiter import (
    RATE_LIMIT_STORE as RATE_LIMIT_STORE,
)
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
from t1d_analytics.seed import seed_database
from t1d_analytics.session_store import SessionStore, open_sessions_db_conn
from t1d_analytics.sql_executor import execute_duckdb_query
from t1d_analytics.sql_utils import (
    SqlValue,
    escape_sql_identifier,
    sanitize_spreadsheet_cell,
    sanitize_sql_value,
)

__all__ = [
    "ApiError",
    "BaseRateLimiter",
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
    "RATE_LIMIT_LOCK",
    "RATE_LIMIT_STORE",
    "SchemaResponse",
    "SessionListResponse",
    "SessionModel",
    "StatusResponse",
    "TableDataResponse",
    "TableInfo",
    "TrainingJobRequest",
    "_extract_sql_from_response",
    "app",
    "check_rate_limit",
    "enforce_rate_limit",
    "extract_stream_delta",
    "get_allowed_db_dirs",
    "get_rate_limiter",
    "is_provider_configured",
    "resolve_provider_and_model",
    "sanitize_spreadsheet_cell",
    "sanitize_sql_value",
    "validate_db_path",
]

_sessions_lock = threading.Lock()


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
    _enforce_rate_limit(
        request,
        limit=limit,
        window_seconds=window_seconds,
        limiter_check_fn=check_rate_limit,
    )


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
        HTTPException: 401 Unauthorized if auth fails or token is missing.

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
    if not provided_token or provided_token != required_key:
        raise HTTPException(
            status_code=401,
            detail={
                "error_code": "backend.unauthorized",
                "params": {"message": "Invalid or missing API key."},
            },
        )


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
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
        except Exception as e:
            logger.warning(f"Startup auto-ingestion failed for {db_path}: {e}")
    run_system_diagnostics(db_path=db_path)
    log_startup_diagnostics(db_path=db_path)
    yield


_lifespan = lifespan


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
        The response produced by the application or an auth error JSON response.

    """
    try:
        verify_api_auth(request)
    except HTTPException as exc:
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
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
        JSONResponse with the domain error's status_code and structured detail dictionary.

    """
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
        JSONResponse with status 500 and localized error payload.

    """
    lang_header = request.headers.get("accept-language", "en")
    lang = lang_header.split(",")[0].split("-")[0].split(";")[0]
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


CLOUD_PROVIDER_MODELS: Dict[str, List[str]] = CLOUD_MODEL_CATALOG


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
        List of dictionaries representing the rows returned.

    Raises:
    ------
        SqlExecutionError: If query execution fails or times out.
        DatabaseNotFoundError: If database file cannot be opened.

    """
    return execute_duckdb_query(
        db_path=db_path,
        query=query,
        max_rows=max_rows,
        timeout_seconds=timeout_seconds,
        connect_fn=duckdb.connect,
    )


def _get_sessions_db_conn(
    db_path: Optional[str] = None,
) -> duckdb.DuckDBPyConnection:
    """
    Open connection to persistent chat sessions database and ensure schema exists.

    Args:
    ----
        db_path: Optional path to the database file.

    Returns:
    -------
        duckdb.DuckDBPyConnection: Database connection.

    """
    return open_sessions_db_conn(db_path=db_path)


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
    conn = (
        _get_sessions_db_conn(db_path)
        if db_path is not None
        else _get_sessions_db_conn()
    )
    try:
        yield conn
    finally:
        conn.close()


_session_store = SessionStore(
    conn_factory=get_sessions_conn,
    lock=_sessions_lock,
)


def get_session_store() -> SessionStore:
    """
    FastAPI dependency provider returning the active SessionStore instance.

    Returns
    -------
        SessionStore: Active conversation session store.

    """
    return _session_store


def get_training_job_manager() -> TrainingJobManager:
    """
    FastAPI dependency provider returning the active TrainingJobManager instance.

    Returns
    -------
        TrainingJobManager: Active background training job manager.

    """
    from t1d_analytics.training_runner import global_training_job_manager

    return global_training_job_manager


_build_sql_prompt = build_sql_prompt_with_history
stream_llm_tokens = stream_llm_completion_tokens


@app.post("/api/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest, req: Request) -> ChatResponse:
    """
    Handle incoming chat messages and execute SQL or NLP queries.

    Args:
    ----
        request: The incoming ChatRequest.
        req: The incoming FastAPI HTTP request.

    Returns:
    -------
        A ChatResponse indicating success or error.

    Raises:
    ------
        HTTPException: If request parameters are invalid.

    """
    enforce_rate_limit(req)
    provider_hdr = req.headers.get("x-provider")
    api_key_hdr = req.headers.get("x-provider-api-key")
    if provider_hdr and not request.provider:
        request.provider = provider_hdr
    if api_key_hdr and not request.api_key:
        request.api_key = api_key_hdr

    if not request.message.strip():
        logger.warning("Received empty message.")
        raise HTTPException(
            status_code=400, detail={"error_code": "backend.emptyMessage", "params": {}}
        )

    return execute_chat_turn(
        request=request,
        db_validator=validate_db_path,
        execute_fn=execute_sql,
        translate_fn=generate_sql_from_nl,
        error_parser=parse_error,
    )


def stream_chat_events(
    request: ChatRequest,
    req: Optional[Request] = None,
) -> AsyncIterator[str]:
    """
    Generate Server-Sent Events for streaming chat queries and SQL results.

    Args:
    ----
        request: The incoming ChatRequest.
        req: Optional FastAPI Request object to detect client disconnection.

    Returns:
    -------
        AsyncIterator[str]: SSE data lines formatted as JSON strings.

    """
    return generate_sse_chat_events(
        request=request,
        req=req,
        db_validator=validate_db_path,
        execute_fn=execute_sql,
        connect_fn=duckdb.connect,
        schema_fn=get_database_schema,
        stream_fn=stream_llm_tokens,
        error_parser=parse_error,
    )


@app.post("/api/chat/stream")
async def chat_stream_endpoint(request: ChatRequest, req: Request) -> StreamingResponse:
    """
    Handle chat requests with Server-Sent Events (SSE) streaming.

    Args:
    ----
        request: The incoming ChatRequest.
        req: The incoming FastAPI HTTP request.

    Returns:
    -------
        StreamingResponse streaming SSE token and result events.

    """
    enforce_rate_limit(req)
    provider_hdr = req.headers.get("x-provider")
    api_key_hdr = req.headers.get("x-provider-api-key")
    if provider_hdr and not request.provider:
        request.provider = provider_hdr
    if api_key_hdr and not request.api_key:
        request.api_key = api_key_hdr

    return StreamingResponse(
        stream_chat_events(request, req=req),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


sessions_router = APIRouter()


@sessions_router.get("/api/sessions", response_model=SessionListResponse)
def list_sessions(
    db_path: Optional[str] = None,
    store: SessionStore = Depends(get_session_store),
) -> SessionListResponse:
    """
    List all persistent conversation sessions.

    Args:
    ----
        db_path: Optional path to the sessions database.
        store: Injected SessionStore dependency.

    Returns:
    -------
        SessionListResponse: List of stored conversation sessions sorted by updated_at.

    """
    return store.list_sessions(db_path=db_path)


@sessions_router.post("/api/sessions", response_model=SessionModel)
def save_session(
    request: CreateSessionRequest,
    db_path: Optional[str] = None,
    store: SessionStore = Depends(get_session_store),
) -> SessionModel:
    """
    Create or update a persistent conversation session.

    Args:
    ----
        request: The session creation or update payload.
        db_path: Optional path to the sessions database.
        store: Injected SessionStore dependency.

    Returns:
    -------
        SessionModel: The saved conversation session.

    """
    return store.upsert_session(request=request, db_path=db_path)


@sessions_router.get("/api/sessions/{session_id}", response_model=SessionModel)
def get_session(
    session_id: str,
    db_path: Optional[str] = None,
    store: SessionStore = Depends(get_session_store),
) -> SessionModel:
    """
    Retrieve a specific conversation session by ID.

    Args:
    ----
        session_id: The ID of the session to retrieve.
        db_path: Optional path to the sessions database.
        store: Injected SessionStore dependency.

    Returns:
    -------
        SessionModel: The found conversation session.

    Raises:
    ------
        HTTPException: 404 if session not found.

    """
    return store.get_session(session_id=session_id, db_path=db_path)


@sessions_router.delete("/api/sessions/{session_id}")
def delete_session(
    session_id: str,
    db_path: Optional[str] = None,
    store: SessionStore = Depends(get_session_store),
) -> Dict[str, str]:
    """
    Delete a conversation session.

    Args:
    ----
        session_id: The ID of the session to delete.
        db_path: Optional path to the sessions database.
        store: Injected SessionStore dependency.

    Returns:
    -------
        Dict[str, str]: Deletion confirmation message.

    Raises:
    ------
        HTTPException: 404 if session not found.

    """
    return store.delete_session(session_id=session_id, db_path=db_path)


app.include_router(sessions_router)


def parse_error(e: Exception) -> ApiError:
    """
    Parse an exception into a structured ApiError.

    Args:
    ----
        e (Exception): The exception to parse.

    Returns:
    -------
        ApiError: The resulting ApiError object.

    """
    if isinstance(e, T1DError):
        return ApiError(error_code=e.error_code, params=e.params)
    return ApiError(error_code="backend.errorUnexpected", params={"error": str(e)})


@app.post("/api/execute-sql", response_model=ExecuteSqlResponse)
def execute_sql_endpoint(
    request: ExecuteSqlRequest, req: Request
) -> ExecuteSqlResponse:
    """
    Execute an arbitrary SQL query against the database.

    Args:
    ----
        request: The request containing the query and optional db_path.
        req: The incoming FastAPI HTTP request.

    Returns:
    -------
        The execution response containing results or error.

    """
    enforce_rate_limit(req)
    if not request.query.strip():
        return ExecuteSqlResponse(
            error=ApiError(error_code="backend.emptyMessage", params={})
        )

    try:
        db_path = validate_db_path(request.db_path)
        results = execute_sql(db_path, request.query)
        return ExecuteSqlResponse(sqlResult=results, rows=results)
    except (HTTPException, InvalidDatabasePathError):
        raise
    except ValueError as ve:
        logger.error(f"Database error during sql execution request: {ve}")
        return ExecuteSqlResponse(error=parse_error(ve))
    except Exception as e:
        logger.exception("Unexpected error during sql execution request.")
        return ExecuteSqlResponse(error=parse_error(e))


@app.get("/api/table/{table_name}", response_model=TableDataResponse)
def get_table_data(
    table_name: str,
    limit: int = 25,
    offset: int = 0,
    sort_by: Optional[str] = None,
    order: Optional[str] = "asc",
    search: Optional[str] = None,
    db_path: Optional[str] = None,
) -> TableDataResponse:
    """
    Return paginated rows from a specific table with optional sorting and searching.

    Args:
    ----
        table_name: The table name.
        limit: The limit.
        offset: The offset.
        sort_by: Optional column name to sort by.
        order: Sort order, either 'asc' or 'desc' (default: 'asc').
        search: Optional string to filter across columns.
        db_path: Optional path to the DuckDB database.

    Returns:
    -------
        The data with pagination metadata.

    """
    return fetch_paginated_table_data(
        table_name=table_name,
        limit=limit,
        offset=offset,
        sort_by=sort_by,
        order=order,
        search=search,
        db_path=db_path,
        db_validator=validate_db_path,
    )


@app.get("/api/table/{table_name}/profile")
def get_table_profile(
    table_name: str, db_path: Optional[str] = None
) -> TableProfileResult:
    """
    Profile a table to return data quality, completeness, and column statistics.

    Args:
    ----
        table_name: The table name to profile.
        db_path: Optional path to the database file.

    Returns:
    -------
        TableProfileResult: Profiling summary dictionary.

    """
    return fetch_table_profile(
        table_name=table_name,
        db_path=db_path,
        db_validator=validate_db_path,
        profile_fn=profile_table,
    )


@app.get("/api/export/excel")
def export_excel(table_name: str, db_path: Optional[str] = None) -> StreamingResponse:
    """
    Export table data formatted as a native OpenXML Excel (.xlsx) workbook.

    Args:
    ----
        table_name: Table to export.
        db_path: Optional path to database file.

    Returns:
    -------
        StreamingResponse: Streamed XLSX file with spreadsheetml MIME type.

    """
    return stream_excel_export(
        table_name=table_name,
        db_path=db_path,
        db_validator=validate_db_path,
    )


@app.get("/api/export/parquet")
def export_parquet(
    table_name: str,
    db_path: Optional[str] = None,
) -> StreamingResponse:
    """
    Export table data in columnar Apache Parquet format.

    Args:
    ----
        table_name: Table to export.
        db_path: Optional path to database file.

    Returns:
    -------
        StreamingResponse: Streamed Parquet file.

    """
    return stream_parquet_export(
        table_name=table_name,
        db_path=db_path,
        db_validator=validate_db_path,
    )


@app.get("/api/export/csv")
def export_csv(
    table_name: str,
    db_path: Optional[str] = None,
) -> StreamingResponse:
    """
    Export table data formatted as a streaming CSV file with spreadsheet formula defense.

    Args:
    ----
        table_name: Table to export.
        db_path: Optional path to database file.

    Returns:
    -------
        StreamingResponse: Streamed CSV file with text/csv MIME type.

    """
    return stream_csv_export(
        table_name=table_name,
        db_path=db_path,
        db_validator=validate_db_path,
    )


@app.get("/api/export/json")
def export_json(
    table_name: str,
    db_path: Optional[str] = None,
) -> StreamingResponse:
    """
    Export table data formatted as a streaming JSON array of row objects.

    Args:
    ----
        table_name: Table to export.
        db_path: Optional path to database file.

    Returns:
    -------
        StreamingResponse: Streamed JSON file with application/json MIME type.

    """
    return stream_json_export(
        table_name=table_name,
        db_path=db_path,
        db_validator=validate_db_path,
    )


@app.get("/api/status", response_model=StatusResponse)
def get_status(
    request: Request,
    db_path: Optional[str] = None,
    require_ready: bool = False,
) -> Union[StatusResponse, JSONResponse]:
    """
    Return comprehensive system status, table counts, row counts, and LLM availability.

    Args:
    ----
        request: FastAPI request object.
        db_path: Optional custom database path.
        require_ready: If True, return HTTP 503 when database is unpopulated or not ready.

    Returns:
    -------
        StatusResponse or JSONResponse with status details.

    """
    target_db = validate_db_path(db_path) if db_path else None
    lang_header = request.headers.get("accept-language", "en")
    lang = lang_header.split(",")[0].split("-")[0].split(";")[0]
    health = get_system_health(db_path=target_db, lang=lang)
    db = health.database

    is_ready = (
        db.connected
        and db.table_count > 0
        and db.total_rows > 0
        and health.sessions_database_healthy
    )
    status_str = "ready" if is_ready else health.status
    resp = StatusResponse(
        status=status_str,
        tables_count=db.table_count,
        total_rows=db.total_rows,
        required_tables_present=db.has_initial_data,
        backend_accessible=health.backend_accessible,
        version=health.version,
        database=db.to_dict(),
        ollama=health.ollama.to_dict(),
        providers={k: v.to_dict() for k, v in health.providers.items()},
    )
    if require_ready and (
        status_str != "ready" or db.table_count == 0 or db.total_rows == 0
    ):
        return JSONResponse(status_code=503, content=resp.model_dump())
    return resp


@app.get("/api/ready", response_model=None)
def get_ready(
    request: Request, db_path: Optional[str] = None
) -> Union[StatusResponse, JSONResponse]:
    """
    Readiness probe endpoint returning HTTP 200 when database is populated or HTTP 503 otherwise.

    Args:
    ----
        request: FastAPI request object.
        db_path: Optional custom database path.

    Returns:
    -------
        StatusResponse or 503 JSONResponse.

    """
    return get_status(request=request, db_path=db_path, require_ready=True)


@app.get("/api/health", response_model=None)
def get_health(
    request: Request, db_path: Optional[str] = None
) -> Union[SystemHealthDict, JSONResponse]:
    """
    Deep health check endpoint returning HTTP 503 if the DuckDB database is missing or empty.

    Args:
    ----
        request: FastAPI request object.
        db_path: Optional custom database path.

    Returns:
    -------
        Dictionary or HTTP 503 JSONResponse containing system health details.

    """
    target_db = validate_db_path(db_path) if db_path else None
    lang_header = request.headers.get("accept-language", "en")
    lang = lang_header.split(",")[0].split("-")[0].split(";")[0]
    health = run_system_diagnostics(db_path=target_db, lang=lang)
    db = health.database
    payload = health.to_dict()
    if (
        health.status in ("error", HealthStatus.UNHEALTHY.value)
        or db.status == HealthStatus.UNHEALTHY
        or db.table_count == 0
        or db.total_rows == 0
    ):
        return JSONResponse(status_code=503, content=payload)
    return payload


def _build_schema_response(target_db: str, lang: str = "en") -> SchemaResponse:
    """
    Inspect a DuckDB database and build a SchemaResponse with table columns and health status.

    Args:
    ----
        target_db: Validated path to the DuckDB database file.
        lang: Localization language code for database health messages.

    Returns:
    -------
        SchemaResponse: Structured table and column metadata with database health dictionary.

    """
    db_health = check_database_health(target_db, lang=lang)

    conn = None
    try:
        conn = duckdb.connect(target_db, read_only=True)
        tables = [row[0] for row in conn.execute("SHOW TABLES").fetchall()]

        schema_data = []
        for table in tables:
            escaped_table = escape_sql_identifier(table)
            columns = conn.execute(f'DESCRIBE "{escaped_table}"').fetchall()
            col_info = [ColumnInfo(name=c[0], type=c[1]) for c in columns]
            schema_data.append(TableInfo(name=table, columns=col_info))

        return SchemaResponse(tables=schema_data, database=db_health.to_dict())
    except Exception as e:
        logger.error(f"Error fetching schema: {e}")
        return SchemaResponse(tables=[], database=db_health.to_dict())
    finally:
        if conn is not None:
            conn.close()


@app.get("/api/schema", response_model=SchemaResponse)
def get_schema(request: Request, db_path: Optional[str] = None) -> SchemaResponse:
    """
    Return structured schema for the frontend Schema Explorer.

    Args:
    ----
        request: FastAPI request.
        db_path: Optional target DuckDB database path.

    Returns:
    -------
        The structured schema response including database diagnostics.

    """
    target_db = validate_db_path(db_path)
    lang_header = request.headers.get("accept-language", "en")
    lang = lang_header.split(",")[0].split("-")[0].split(";")[0]
    return _build_schema_response(target_db=target_db, lang=lang)


@app.get("/api/providers/status", response_model=ProvidersStatusResponse)
def get_providers_status() -> ProvidersStatusResponse:
    """
    Report connection health and credential status for all supported LLM providers.

    Returns
    -------
        ProvidersStatusResponse containing status for ollama, openai, anthropic, google.

    """
    return probe_providers_status(urlopen_fn=urllib.request.urlopen)


@app.get("/api/models", response_model=ModelsResponse)
def list_models(
    include_cloud: bool = True,
    configured_models: Optional[str] = None,
) -> ModelsResponse:
    """
    List available local Ollama models and configured any-llm cloud providers.

    Args:
    ----
        include_cloud: Whether to include cloud and configured any-llm provider models.
        configured_models: Optional comma-separated list of custom configured provider/model IDs.

    Returns:
    -------
        ModelsResponse: List of available local and cloud models.

    """
    return fetch_available_models(
        include_cloud=include_cloud,
        configured_models=configured_models,
        cloud_catalog=CLOUD_PROVIDER_MODELS,
        urlopen_fn=urllib.request.urlopen,
    )


@app.get("/api/databases", response_model=DatabaseListResponse)
def list_databases() -> DatabaseListResponse:
    """
    Discover and list available DuckDB database files in the workspace.

    Returns
    -------
        DatabaseListResponse containing list of found databases and current default.

    """
    return discover_workspace_databases()


training_router = APIRouter()


@training_router.post("/api/training/jobs")
def submit_training_job(
    job_req: TrainingJobRequest,
    req: Request,
    job_manager: TrainingJobManager = Depends(get_training_job_manager),
) -> TrainingJobDict:
    """
    Submit a background training job using the configured accelerator backend.

    Args:
    ----
        job_req: Training job specification.
        req: Incoming FastAPI request.
        job_manager: Injected TrainingJobManager dependency.

    Returns:
    -------
        TrainingJobDict: Job ID and initial status.

    Raises:
    ------
        InvalidTrainingBackendError: If requested training backend is unsupported.

    """
    from t1d_analytics.models import TrainingBackend, TrainingJobConfig

    validated_dataset_path = validate_db_path(job_req.dataset_path)
    validated_output_dir = validate_db_path(job_req.output_dir)

    backend_val = job_req.backend.lower()
    if backend_val == "cpu":
        backend_val = "local-cpu"
    elif backend_val == "gpu":
        backend_val = "local-gpu"
    elif backend_val == "tpu":
        backend_val = "remote-tpu-maxtext"

    try:
        backend_enum = TrainingBackend(backend_val)
    except ValueError as exc:
        raise InvalidTrainingBackendError(
            params={"backend": job_req.backend},
        ) from exc

    config = TrainingJobConfig(
        dataset_path=validated_dataset_path,
        output_dir=validated_output_dir,
        backend=backend_enum,
        model_name=job_req.model_name,
        dry_run=job_req.dry_run,
    )
    job_id = job_manager.submit_job(config)
    job = job_manager.get_job(job_id)
    return job if job else {"job_id": job_id, "status": "pending"}


@training_router.get("/api/training/jobs")
def list_training_jobs(
    req: Request,
    job_manager: TrainingJobManager = Depends(get_training_job_manager),
) -> Dict[str, List[TrainingJobDict]]:
    """
    List all tracked background training jobs.

    Args:
    ----
        req: Incoming FastAPI request.
        job_manager: Injected TrainingJobManager dependency.

    Returns:
    -------
        Dict[str, List[TrainingJobDict]]: Dictionary containing list of training jobs.

    """
    return {"jobs": job_manager.list_jobs()}


@training_router.get("/api/training/jobs/{job_id}")
def get_training_job(
    job_id: str,
    req: Request,
    job_manager: TrainingJobManager = Depends(get_training_job_manager),
) -> TrainingJobDict:
    """
    Retrieve details and execution metrics for a specific background training job.

    Args:
    ----
        job_id: Unique job identifier string.
        req: Incoming FastAPI request.
        job_manager: Injected TrainingJobManager dependency.

    Returns:
    -------
        TrainingJobDict: Job status and execution results.

    Raises:
    ------
        TrainingJobNotFoundError: 404 if the job ID is not found.

    """
    job = job_manager.get_job(job_id)
    if not job:
        raise TrainingJobNotFoundError(params={"job_id": job_id})
    return job


@training_router.delete("/api/training/jobs/{job_id}")
def cancel_training_job(
    job_id: str,
    req: Request,
    job_manager: TrainingJobManager = Depends(get_training_job_manager),
) -> TrainingJobDict:
    """
    Request cancellation of a running background training job.

    Args:
    ----
        job_id: Unique job identifier string.
        req: Incoming FastAPI request.
        job_manager: Injected TrainingJobManager dependency.

    Returns:
    -------
        TrainingJobDict: Cancellation outcome.

    Raises:
    ------
        TrainingJobNotFoundError: 404 if the job ID is not found.

    """
    cancelled = job_manager.cancel_job(job_id)
    if not cancelled:
        raise TrainingJobNotFoundError(params={"job_id": job_id})
    job = job_manager.get_job(job_id)
    return job if job else {"job_id": job_id, "status": "cancelled"}


app.include_router(training_router)


def _setup_debug() -> None:
    """
    Configure debug static file serving if enabled.

    When the DEBUG environment variable is set, this function mounts
    the frontend static files (from the t1d-analytics-web/dist directory) at the root
    path. Useful for local development and testing.

    Returns
    -------
        None

    """
    if os.environ.get("DEBUG"):
        repo_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        )
        dist_path = os.path.join(repo_root, "t1d-analytics-web", "dist")
        if os.path.exists(dist_path):
            logger.info(f"DEBUG mode enabled. Serving static frontend from {dist_path}")
            app.mount("/", StaticFiles(directory=dist_path, html=True), name="static")
        else:
            logger.warning(
                f"DEBUG mode enabled but static dist folder not found at {dist_path}"
            )


_setup_debug()
