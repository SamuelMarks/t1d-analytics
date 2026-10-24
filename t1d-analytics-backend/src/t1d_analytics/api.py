"""FastAPI application for the T1D Analytics web interface."""

import json
import logging
import os
import threading
import urllib.error
import urllib.request
from contextlib import asynccontextmanager, contextmanager
from typing import (
    Any,
    AsyncGenerator,
    AsyncIterator,
    Dict,
    Iterator,
    List,
    Optional,
)

import duckdb
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from t1d_analytics.analytics import (
    extract_sql_from_response as _extract_sql_from_response,
)
from t1d_analytics.analytics import (
    get_database_schema,
    load_all_to_duckdb,
    profile_table,
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
from t1d_analytics.db_security import (
    discover_workspace_databases,
    get_allowed_db_dirs,
    validate_db_path,
)
from t1d_analytics.diagnostics import (
    HealthStatus,
    check_database_health,
    get_system_health,
    log_startup_diagnostics,
    run_system_diagnostics,
)
from t1d_analytics.errors import (
    DatabaseNotFoundError,
    InvalidDatabasePathError,
    LlmTranslationError,
    MissingSdkError,
    ProviderNotConfiguredError,
    SchemaReadError,
    SqlExecutionError,
    T1DError,
    TrainingJobNotFoundError,
)
from t1d_analytics.i18n import get_translator
from t1d_analytics.llm_client import (
    CLOUD_MODEL_CATALOG,
    PROVIDER_ENV_KEYS,
    build_clinical_sql_prompt,
    extract_stream_delta,
    get_effective_api_key,
    is_provider_configured,
    normalize_ollama_host,
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
from t1d_analytics.sql_utils import (
    SqlValue,
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
    db_path = os.environ.get("T1D_DB_PATH", "data/t1d.duckdb")
    data_dir = os.environ.get("T1D_DATA_DIR", "data")
    db_health = check_database_health(db_path=db_path)
    if not db_health.exists or db_health.table_count == 0:
        try:
            seed_database(data_dir=data_dir, db_path=db_path)
            load_all_to_duckdb(data_dir, db_path)
        except Exception as e:
            logger.warning(f"Startup auto-ingestion failed for {db_path}: {e}")
    run_system_diagnostics(db_path=db_path)
    log_startup_diagnostics(db_path=db_path)
    yield


_lifespan = lifespan


app = FastAPI(
    title="T1D Analytics API",
    description="API for querying T1D datasets.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def auth_and_rate_limit_middleware(request: Request, call_next: Any) -> Any:
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
    logger.info(f"Executing SQL query against database at {db_path}:\n{query}")
    conn = None
    is_closed = False
    lock = threading.Lock()

    def safe_interrupt() -> None:
        """Interrupt the DuckDB connection safely from another thread."""
        with lock:
            if not is_closed and conn is not None:
                conn.interrupt()

    try:
        try:
            conn = duckdb.connect(db_path, read_only=True)
        except duckdb.IOException as io_err:
            raise DatabaseNotFoundError(params={"error": str(io_err)}) from io_err
        conn.execute("SET enable_external_access = false")
        timer = threading.Timer(timeout_seconds, safe_interrupt)
        timer.start()
        try:
            result = conn.execute(query)
            columns = (
                [desc[0] for desc in result.description] if result.description else []
            )
            rows = result.fetchmany(max_rows)
        finally:
            timer.cancel()

        output = [
            {col: sanitize_sql_value(val) for col, val in zip(columns, row)}
            for row in rows
        ]
        logger.info(f"SQL query executed successfully. Returned {len(output)} rows.")
        return output
    except duckdb.InterruptException as ie:
        logger.error(f"SQL execution timed out: {ie}")
        raise SqlExecutionError(
            error_code="backend.queryTimeout",
            params={"error": "Query execution timed out."},
        ) from ie
    except DatabaseNotFoundError:
        raise
    except Exception as e:
        logger.error(f"SQL execution error: {e}")
        raise SqlExecutionError(params={"error": str(e)}) from e
    finally:
        with lock:
            is_closed = True
            if conn is not None:
                conn.close()


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


def _build_sql_prompt(
    schema: str,
    nl_query: str,
    history: Optional[List[ChatMessageModel]] = None,
) -> str:
    """
    Construct the clinical Text-to-SQL system prompt for an LLM.

    Args:
    ----
        schema: The database schema text.
        nl_query: The clinician or user query.
        history: Optional list of prior conversation messages for multi-turn context.

    Returns:
    -------
        str: Formatted system and user prompt.

    """
    history_context = ""
    if history:
        turns: List[str] = []
        for msg in history[-6:]:
            role = "User" if msg.role == "user" else "Assistant"
            content = msg.content.strip()
            sql_info = f" (SQL: {msg.sqlQuery})" if msg.sqlQuery else ""
            turns.append(f"{role}: {content}{sql_info}")
        history_context = "Prior Conversation History:\n" + "\n".join(turns) + "\n\n"

    return build_clinical_sql_prompt(
        schema=schema,
        nl_query=nl_query,
        history_context=history_context,
        require_markdown_fence=True,
    )


def stream_llm_tokens(
    prompt: str,
    model_name: str,
    api_key: Optional[str] = None,
    provider: Optional[str] = None,
) -> Iterator[str]:
    """
    Stream tokens from LLM using any-llm, falling back gracefully to single labeled completion block.

    Args:
    ----
        prompt: The full prompt string.
        model_name: The LLM model name or provider/model string.
        api_key: Optional API key for external cloud provider.
        provider: Optional provider name override.

    Yields:
    ------
        str: Streamed token chunks.

    """
    from any_llm import AnyLLM

    resolved_provider, actual_model = resolve_provider_and_model(model_name, provider)
    effective_api_key = get_effective_api_key(resolved_provider, api_key)

    if resolved_provider.lower() in PROVIDER_ENV_KEYS and not effective_api_key:
        env_var = PROVIDER_ENV_KEYS[resolved_provider.lower()]
        raise ProviderNotConfiguredError(
            params={"provider": resolved_provider, "env_var": env_var},
        )

    llm_kwargs: Dict[str, Any] = {}
    if effective_api_key:
        llm_kwargs["api_key"] = effective_api_key

    llm = AnyLLM.create(resolved_provider, **llm_kwargs)
    try:
        stream_resp = llm.completion(
            model=actual_model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )
        if hasattr(stream_resp, "__iter__"):
            for chunk in stream_resp:
                delta = extract_stream_delta(chunk, resolved_provider)
                if delta:
                    yield delta
            return
    except Exception as e:
        logger.warning(
            f"Native streaming failed for provider '{resolved_provider}': {e}. Falling back to non-streaming completion."
        )

    response = llm.completion(
        model=actual_model,
        messages=[{"role": "user", "content": prompt}],
    )
    full_text = extract_stream_delta(response, resolved_provider) or str(response)
    yield full_text


def generate_sql_from_nl(
    db_path: str,
    nl_query: str,
    model_name: str = "gemma4",
    history: Optional[List[ChatMessageModel]] = None,
    api_key: Optional[str] = None,
    provider: Optional[str] = None,
) -> tuple[str, str]:
    """
    Translate natural language to SQL using any-llm.

    Args:
    ----
        db_path: Path to the DuckDB database file.
        nl_query: Natural language question.
        model_name: The model to use or provider/model (e.g. 'gemma4', 'openai/gpt-4o').
        history: Optional list of prior conversation messages for multi-turn context.
        api_key: Optional API key for cloud provider.
        provider: Optional provider override name.

    Returns:
    -------
        The generated full response and extracted SQL query string.

    Raises:
    ------
        MissingSdkError: If any-llm-sdk is missing.
        ProviderNotConfiguredError: If API key is missing.
        LlmTranslationError: If LLM API fails.
        DatabaseNotFoundError: If database file is missing.
        SchemaReadError: If reading schema fails.

    """
    resolved_provider, actual_model = resolve_provider_and_model(model_name, provider)

    logger.info(
        f"Translating natural language to SQL: '{nl_query}' using provider '{resolved_provider}' and model '{actual_model}'"
    )
    try:
        from any_llm import AnyLLM
    except ImportError as e:
        logger.error(f"any-llm-sdk[{resolved_provider}] is not installed.")
        raise MissingSdkError(
            params={"error": f"any-llm-sdk[{resolved_provider}]"},
        ) from e

    effective_api_key = get_effective_api_key(resolved_provider, api_key)

    if resolved_provider.lower() in PROVIDER_ENV_KEYS and not effective_api_key:
        env_var = PROVIDER_ENV_KEYS[resolved_provider.lower()]
        raise ProviderNotConfiguredError(
            params={"provider": resolved_provider, "env_var": env_var},
        )

    try:
        try:
            conn = duckdb.connect(db_path, read_only=True)
        except duckdb.IOException as io_err:
            raise DatabaseNotFoundError(params={"error": str(io_err)}) from io_err
        schema = get_database_schema(conn)
        conn.close()
        logger.debug(f"Retrieved schema: {schema[:200]}... (truncated)")
    except DatabaseNotFoundError as dnf:
        logger.error(f"Failed to read schema: {dnf}")
        raise
    except Exception as e:
        logger.error(f"Failed to read schema: {e}")
        raise SchemaReadError(params={"error": str(e)}) from e

    prompt = _build_sql_prompt(schema, nl_query, history=history)

    try:
        logger.info(f"Sending prompt to LLM ({resolved_provider}/{actual_model})...")
        llm_kwargs: Dict[str, Any] = {}
        if effective_api_key:
            llm_kwargs["api_key"] = effective_api_key
        llm = AnyLLM.create(resolved_provider, **llm_kwargs)
        response = llm.completion(
            model=actual_model,
            messages=[{"role": "user", "content": prompt}],
        )
        raw_choice = response.choices[0].message.content
        full_response = (raw_choice or "").strip()
        logger.info(f"LLM response:\n{full_response}")
        sql_query = _extract_sql_from_response(full_response)
        return full_response, sql_query
    except Exception as e:
        logger.error(f"LLM translation error: {e}")
        raise LlmTranslationError(params={"error": str(e)}) from e


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

    try:
        logger.info(
            f"Received chat request: model='{request.model}', message='{request.message}'"
        )
        db_path = validate_db_path(request.db_path)
        if request.model == "sql":
            sql_query = request.message
            content_prefix = "backend.literalSql"
            results = execute_sql(db_path, sql_query)
            return ChatResponse(
                content=content_prefix, sqlResult=results, sqlQuery=sql_query
            )
        else:
            full_response, sql_query = generate_sql_from_nl(
                db_path,
                request.message,
                model_name=request.model,
                history=request.history,
                api_key=request.api_key,
                provider=request.provider,
            )
            results = None
            sql_error = None
            if sql_query:
                try:
                    results = execute_sql(db_path, sql_query)
                except (T1DError, ValueError, RuntimeError) as e:
                    sql_error = parse_error(e)

            return ChatResponse(
                content=full_response,
                sqlQuery=sql_query,
                sqlResult=results,
                error=sql_error,
            )
    except (HTTPException, InvalidDatabasePathError):
        raise
    except ValueError as ve:
        logger.error(f"Database error during chat request: {ve}")
        return ChatResponse(content="backend.errorDbExecution", error=parse_error(ve))
    except RuntimeError as re:
        logger.error(f"NLP error during chat request: {re}")
        return ChatResponse(
            content="backend.errorNlpTranslation", error=parse_error(re)
        )
    except Exception as e:
        logger.exception("Unexpected error during chat request.")
        return ChatResponse(content="backend.errorUnexpected", error=parse_error(e))


async def stream_chat_events(
    request: ChatRequest,
    req: Optional[Request] = None,
) -> AsyncIterator[str]:
    """
    Generate Server-Sent Events for streaming chat queries and SQL results.

    Args:
    ----
        request: The incoming ChatRequest.
        req: Optional FastAPI Request object to detect client disconnection.

    Yields:
    ------
        SSE data lines formatted as JSON strings.

    """
    if req is not None and await req.is_disconnected():
        logger.info("Client disconnected before stream generation.")
        return

    if not request.message.strip():
        payload = json.dumps({"event": "error", "error": "backend.emptyMessage"})
        yield f"data: {payload}\n\n"
        return

    try:
        db_path = validate_db_path(request.db_path)
    except (HTTPException, InvalidDatabasePathError):
        payload = json.dumps({"event": "error", "error": "backend.invalidDatabasePath"})
        yield f"data: {payload}\n\n"
        return

    if request.model == "sql":
        exec_payload = json.dumps(
            {"event": "token", "token": "Executing SQL query...\n"}
        )
        yield f"data: {exec_payload}\n\n"
        yield f"data: {json.dumps({'event': 'execution_progress', 'status': 'Executing SQL query...'})}\n\n"
        try:
            results = execute_sql(db_path, request.message)
            res_dict = {
                "content": "backend.literalSql",
                "sqlQuery": request.message,
                "sqlResult": results,
                "error": None,
            }
            yield f"data: {json.dumps({'event': 'done', **res_dict})}\n\n"
            yield f"data: {json.dumps({'event': 'result', **res_dict})}\n\n"
        except (T1DError, duckdb.Error, OSError, RuntimeError, ValueError) as e:
            err_dict = parse_error(e).model_dump()
            err_dict_payload = {
                "content": "backend.errorDbExecution",
                "sqlQuery": request.message,
                "sqlResult": None,
                "error": err_dict,
            }
            yield f"data: {json.dumps({'event': 'done', **err_dict_payload})}\n\n"
            yield f"data: {json.dumps({'event': 'result', **err_dict_payload})}\n\n"
        return

    try:
        conn = duckdb.connect(db_path, read_only=True)
        schema = get_database_schema(conn)
        conn.close()

        prompt = _build_sql_prompt(schema, request.message, history=request.history)
        full_parts: List[str] = []
        for token in stream_llm_tokens(
            prompt,
            request.model,
            api_key=request.api_key,
            provider=request.provider,
        ):
            if req is not None and await req.is_disconnected():
                logger.info("Client disconnected during SSE token streaming.")
                return
            full_parts.append(token)
            tok_payload = json.dumps({"event": "token", "token": token})
            yield f"data: {tok_payload}\n\n"
        full_response = "".join(full_parts)
        sql_query = _extract_sql_from_response(full_response)

        sql_results: Optional[List[Dict[str, SqlValue]]] = None
        sql_err_dict = None
        if sql_query:
            yield f"data: {json.dumps({'event': 'sql_generated', 'sqlQuery': sql_query})}\n\n"
            yield f"data: {json.dumps({'event': 'execution_progress', 'status': 'Executing SQL query...'})}\n\n"
            try:
                sql_results = execute_sql(db_path, sql_query)
            except (T1DError, duckdb.Error, OSError, RuntimeError, ValueError) as e:
                sql_err_dict = parse_error(e).model_dump()

        nl_res_dict: Dict[str, Any] = {
            "content": full_response,
            "sqlQuery": sql_query,
            "sqlResult": sql_results,
            "error": sql_err_dict,
        }
        yield f"data: {json.dumps({'event': 'done', **nl_res_dict})}\n\n"
        yield f"data: {json.dumps({'event': 'result', **nl_res_dict})}\n\n"
    except (T1DError, duckdb.Error, OSError, RuntimeError, ValueError, TypeError) as e:
        err_dict = parse_error(e).model_dump()
        nl_err_payload: Dict[str, Any] = {
            "content": "backend.errorNlpTranslation",
            "sqlQuery": None,
            "sqlResult": None,
            "error": err_dict,
        }
        yield f"data: {json.dumps({'event': 'done', **nl_err_payload})}\n\n"
        yield f"data: {json.dumps({'event': 'result', **nl_err_payload})}\n\n"


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


@app.get("/api/sessions", response_model=SessionListResponse)
def list_sessions(db_path: Optional[str] = None) -> SessionListResponse:
    """
    List all persistent conversation sessions.

    Args:
    ----
        db_path: Optional path to the sessions database.

    Returns:
    -------
        SessionListResponse: List of stored conversation sessions sorted by updated_at.

    """
    return _session_store.list_sessions(db_path=db_path)


@app.post("/api/sessions", response_model=SessionModel)
def save_session(
    request: CreateSessionRequest, db_path: Optional[str] = None
) -> SessionModel:
    """
    Create or update a persistent conversation session.

    Args:
    ----
        request: The session creation or update payload.
        db_path: Optional path to the sessions database.

    Returns:
    -------
        SessionModel: The saved conversation session.

    """
    return _session_store.upsert_session(request=request, db_path=db_path)


@app.get("/api/sessions/{session_id}", response_model=SessionModel)
def get_session(session_id: str, db_path: Optional[str] = None) -> SessionModel:
    """
    Retrieve a specific conversation session by ID.

    Args:
    ----
        session_id: The ID of the session to retrieve.
        db_path: Optional path to the sessions database.

    Returns:
    -------
        SessionModel: The found conversation session.

    Raises:
    ------
        HTTPException: 404 if session not found.

    """
    return _session_store.get_session(session_id=session_id, db_path=db_path)


@app.delete("/api/sessions/{session_id}")
def delete_session(session_id: str, db_path: Optional[str] = None) -> Dict[str, str]:
    """
    Delete a conversation session.

    Args:
    ----
        session_id: The ID of the session to delete.
        db_path: Optional path to the sessions database.

    Returns:
    -------
        Dict[str, str]: Deletion confirmation message.

    Raises:
    ------
        HTTPException: 404 if session not found.

    """
    return _session_store.delete_session(session_id=session_id, db_path=db_path)


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
def get_table_profile(table_name: str, db_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Profile a table to return data quality, completeness, and column statistics.

    Args:
    ----
        table_name: The table name to profile.
        db_path: Optional path to the database file.

    Returns:
    -------
        Dict[str, Any]: Profiling summary dictionary.

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
) -> Any:
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


@app.get("/api/ready")
def get_ready(request: Request, db_path: Optional[str] = None) -> Any:
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


@app.get("/api/health")
def get_health(request: Request, db_path: Optional[str] = None) -> Any:
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
    db_health = check_database_health(target_db, lang=lang)

    conn = None
    try:
        conn = duckdb.connect(target_db, read_only=True)
        tables = [row[0] for row in conn.execute("SHOW TABLES").fetchall()]

        schema_data = []
        for table in tables:
            escaped_table = table.replace('"', '""')
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


@app.get("/api/providers/status", response_model=ProvidersStatusResponse)
def get_providers_status() -> ProvidersStatusResponse:
    """
    Report connection health and credential status for all supported LLM providers.

    Returns
    -------
        ProvidersStatusResponse containing status for ollama, openai, anthropic, google.

    """
    statuses: List[ProviderStatus] = []

    ollama_healthy = False
    ollama_details = "Ollama service unavailable"
    ollama_base = normalize_ollama_host(default_host="http://localhost:11434")
    try:
        req = urllib.request.Request(f"{ollama_base}/api/tags")
        with urllib.request.urlopen(req, timeout=2) as resp:
            if resp.status == 200:
                ollama_healthy = True
                ollama_details = "Ollama service online"
    except Exception as e:
        ollama_details = f"Ollama unreachable: {e}"

    statuses.append(
        ProviderStatus(
            provider="ollama",
            configured=True,
            healthy=ollama_healthy,
            details=ollama_details,
            reachable=ollama_healthy,
            error_code="PROVIDER_ONLINE" if ollama_healthy else "PROVIDER_OFFLINE",
        )
    )

    openai_conf = is_provider_configured("openai")
    statuses.append(
        ProviderStatus(
            provider="openai",
            configured=openai_conf,
            healthy=openai_conf,
            details="Configured via OPENAI_API_KEY"
            if openai_conf
            else "Missing OPENAI_API_KEY",
            reachable=openai_conf,
            error_code="PROVIDER_ONLINE" if openai_conf else "MISSING_API_KEY",
        )
    )

    anthropic_conf = is_provider_configured("anthropic")
    statuses.append(
        ProviderStatus(
            provider="anthropic",
            configured=anthropic_conf,
            healthy=anthropic_conf,
            details="Configured via ANTHROPIC_API_KEY"
            if anthropic_conf
            else "Missing ANTHROPIC_API_KEY",
            reachable=anthropic_conf,
            error_code="PROVIDER_ONLINE" if anthropic_conf else "MISSING_API_KEY",
        )
    )

    google_conf = is_provider_configured("google")
    statuses.append(
        ProviderStatus(
            provider="google",
            configured=google_conf,
            healthy=google_conf,
            details="Configured via GEMINI_API_KEY/GOOGLE_API_KEY"
            if google_conf
            else "Missing GEMINI_API_KEY",
            reachable=google_conf,
            error_code="PROVIDER_ONLINE" if google_conf else "MISSING_API_KEY",
        )
    )

    return ProvidersStatusResponse(providers=statuses)


@app.get("/api/models", response_model=ModelsResponse)
def list_models(include_cloud: bool = True) -> ModelsResponse:
    """
    List available local Ollama models and configured cloud LLM providers.

    Args:
    ----
        include_cloud: Whether to include standard cloud provider models.

    Returns:
    -------
        ModelsResponse: List of available local and cloud models.

    """
    base_url = normalize_ollama_host()

    models: List[ModelInfo] = []
    try:
        req = urllib.request.Request(f"{base_url}/api/tags")
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode())

        for model in data.get("models", []):
            models.append(
                ModelInfo(
                    name=model.get("name"),
                    size=model.get("size"),
                    provider="ollama",
                    configured=True,
                    requires_key=False,
                    available=True,
                    reachable=True,
                    error_code=None,
                )
            )
    except Exception as e:
        logger.error(f"Error fetching models from local Ollama instance: {e}")

    if include_cloud:
        for prov, prov_models in CLOUD_PROVIDER_MODELS.items():
            conf = is_provider_configured(prov)
            for m in prov_models:
                models.append(
                    ModelInfo(
                        name=m,
                        provider=prov,
                        configured=conf,
                        requires_key=True,
                        available=conf,
                        reachable=conf,
                        error_code=None if conf else "MISSING_API_KEY",
                    )
                )

    return ModelsResponse(models=models)


@app.get("/api/databases", response_model=DatabaseListResponse)
def list_databases() -> DatabaseListResponse:
    """
    Discover and list available DuckDB database files in the workspace.

    Returns
    -------
        DatabaseListResponse containing list of found databases and current default.

    """
    return discover_workspace_databases()


@app.post("/api/training/jobs")
def submit_training_job(
    job_req: TrainingJobRequest,
    req: Request,
) -> Dict[str, Any]:
    """
    Submit a background training job using the configured accelerator backend.

    Args:
    ----
        job_req: Training job specification.
        req: Incoming FastAPI request.

    Returns:
    -------
        Dict[str, Any]: Job ID and initial status.

    """
    from t1d_analytics.models import TrainingBackend, TrainingJobConfig
    from t1d_analytics.training_runner import global_training_job_manager

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
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail={
                "error_code": "backend.invalidBackend",
                "params": {"backend": job_req.backend},
            },
        )

    config = TrainingJobConfig(
        dataset_path=validated_dataset_path,
        output_dir=validated_output_dir,
        backend=backend_enum,
        model_name=job_req.model_name,
        dry_run=job_req.dry_run,
    )
    job_id = global_training_job_manager.submit_job(config)
    job = global_training_job_manager.get_job(job_id)
    return job if job else {"job_id": job_id, "status": "pending"}


@app.get("/api/training/jobs")
def list_training_jobs(req: Request) -> Dict[str, Any]:
    """
    List all tracked background training jobs.

    Args:
    ----
        req: Incoming FastAPI request.

    Returns:
    -------
        Dict[str, Any]: Dictionary containing list of training jobs.

    """
    from t1d_analytics.training_runner import global_training_job_manager

    return {"jobs": global_training_job_manager.list_jobs()}


@app.get("/api/training/jobs/{job_id}")
def get_training_job(job_id: str, req: Request) -> Dict[str, Any]:
    """
    Retrieve details and execution metrics for a specific background training job.

    Args:
    ----
        job_id: Unique job identifier string.
        req: Incoming FastAPI request.

    Returns:
    -------
        Dict[str, Any]: Job status and execution results.

    Raises:
    ------
        TrainingJobNotFoundError: 404 if the job ID is not found.

    """
    from t1d_analytics.training_runner import global_training_job_manager

    job = global_training_job_manager.get_job(job_id)
    if not job:
        raise TrainingJobNotFoundError(params={"job_id": job_id})
    return job


@app.delete("/api/training/jobs/{job_id}")
def cancel_training_job(job_id: str, req: Request) -> Dict[str, Any]:
    """
    Request cancellation of a running background training job.

    Args:
    ----
        job_id: Unique job identifier string.
        req: Incoming FastAPI request.

    Returns:
    -------
        Dict[str, Any]: Cancellation outcome.

    Raises:
    ------
        TrainingJobNotFoundError: 404 if the job ID is not found.

    """
    from t1d_analytics.training_runner import global_training_job_manager

    cancelled = global_training_job_manager.cancel_job(job_id)
    if not cancelled:
        raise TrainingJobNotFoundError(params={"job_id": job_id})
    job = global_training_job_manager.get_job(job_id)
    return job if job else {"job_id": job_id, "status": "cancelled"}


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
