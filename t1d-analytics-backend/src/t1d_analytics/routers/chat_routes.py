"""Chat, NL-to-SQL streaming, asynchronous job polling, and read-only SQL execution endpoints."""

import logging
from collections.abc import AsyncIterator, Callable

import duckdb
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from t1d_analytics.api_dependencies import (
    AppServices,
    get_app_services,
    resolve_default_app_services,
)
from t1d_analytics.api_schemas import (
    ApiError,
    ChatRequest,
    ChatResponse,
    ExecuteSqlRequest,
    ExecuteSqlResponse,
    NlToSqlEvalRequest,
    NlToSqlEvalResponse,
)
from t1d_analytics.chat_service import (
    execute_chat_turn,
    execute_nl_to_sql_eval_turn,
    generate_sse_chat_events,
)
from t1d_analytics.errors import (
    ChatJobNotFoundError,
    InvalidDatabasePathError,
    T1DError,
)

logger = logging.getLogger(__name__)

chat_router = APIRouter()


def stream_chat_events(
    request: ChatRequest,
    req: Request | None = None,
    services: AppServices | None = None,
) -> AsyncIterator[str]:
    """
    Generate Server-Sent Events for streaming chat queries and SQL results.

    Args:
        request: The incoming ChatRequest.
        req: Optional FastAPI Request instance to detect client disconnection.
        services: Optional injected AppServices dependency bundle.

    Returns:
        AsyncIterator[str]: SSE data lines formatted as JSON strings.

    """
    svc = (
        services
        if services is not None
        else (
            get_app_services(req) if req is not None else resolve_default_app_services()
        )
    )
    return generate_sse_chat_events(
        request=request,
        req=req,
        db_validator=svc.validate_db_path,
        execute_fn=svc.execute_sql,
        connect_fn=svc.duckdb_connect,
        schema_fn=svc.get_database_schema,
        stream_fn=svc.stream_llm_completion_tokens,
        error_parser=svc.parse_error,
    )


def _apply_provider_headers(
    request: ChatRequest | NlToSqlEvalRequest, req: Request
) -> None:
    """
    Populate `request.provider` and `request.api_key` from HTTP headers when omitted in body.

    Args:
        request: Incoming `ChatRequest` or `NlToSqlEvalRequest` payload.
        req: Incoming FastAPI HTTP request.

    """
    provider_hdr = req.headers.get("x-provider")
    api_key_hdr = req.headers.get("x-provider-api-key")
    if provider_hdr and not request.provider:
        request.provider = provider_hdr
    if api_key_hdr and not request.api_key:
        request.api_key = api_key_hdr


def _dispatch_async_chat_job(
    request: ChatRequest,
    svc: AppServices,
) -> ChatResponse:
    """
    Validate `request.db_path` and enqueue an asynchronous chat job in `ChatJobManager`.

    Args:
        request: Validated `ChatRequest` payload.
        svc: Active `AppServices` bundle.

    Returns:
        ChatResponse: Immediate pending ticket (`status="pending"`, `job_id="chatjob-..."`).

    """
    resolved_db = svc.validate_db_path(request.db_path)
    normalized_req = request.model_copy(
        update={"db_path": resolved_db, "async_poll": False}
    )

    def _run_chat_worker(
        job_req: ChatRequest, on_progress: Callable[[str], None]
    ) -> ChatResponse:
        """Execute chat turn in background thread and emit final content to progress buffer."""
        resp = execute_chat_turn(
            request=job_req,
            db_validator=svc.validate_db_path,
            execute_fn=svc.execute_sql,
            translate_fn=svc.translate_nl_to_sql,
            error_parser=svc.parse_error,
        )
        on_progress(resp.content)
        return resp

    chat_mgr = svc.get_chat_job_manager()
    ticket = chat_mgr.submit_chat_job(normalized_req, _run_chat_worker)
    return ChatResponse(
        job_id=ticket["job_id"],
        status="pending",
        content="",
        partial_content="",
    )


@chat_router.post("/api/chat", response_model=ChatResponse)
def chat_endpoint(
    request: ChatRequest,
    req: Request,
    async_poll: bool = False,
    svc: AppServices = Depends(get_app_services),
) -> ChatResponse:
    """
    Handle incoming chat messages and execute SQL or NLP queries synchronously or asynchronously.

    Args:
        request: The incoming ChatRequest.
        req: The incoming FastAPI HTTP request.
        async_poll: Optional query parameter to dispatch asynchronously and return a pending ticket.
        svc: Injected AppServices dependency bundle.

    Returns:
        ChatResponse: A ChatResponse indicating pending job status, success, or error.

    Raises:
        HTTPException: If request parameters are invalid.

    """
    _apply_provider_headers(request, req)

    if not request.message.strip():
        logger.warning("Received empty message.")
        raise HTTPException(
            status_code=400, detail={"error_code": "backend.emptyMessage", "params": {}}
        )

    if request.async_poll or async_poll:
        return _dispatch_async_chat_job(request, svc)

    return execute_chat_turn(
        request=request,
        db_validator=svc.validate_db_path,
        execute_fn=svc.execute_sql,
        translate_fn=svc.translate_nl_to_sql,
        error_parser=svc.parse_error,
    )


@chat_router.post("/api/chat/jobs", response_model=ChatResponse)
def submit_chat_job_endpoint(
    request: ChatRequest,
    req: Request,
    svc: AppServices = Depends(get_app_services),
) -> ChatResponse:
    """
    Enqueue an asynchronous chat or NL-to-SQL request and immediately return a pending job ticket.

    Args:
        request: The incoming ChatRequest.
        req: The incoming FastAPI HTTP request.
        svc: Injected AppServices dependency bundle.

    Returns:
        ChatResponse: Immediate pending job response (`status="pending"`, `job_id="chatjob-..."`).

    Raises:
        HTTPException: If `request.message` is empty.

    """
    _apply_provider_headers(request, req)

    if not request.message.strip():
        logger.warning("Received empty message on /api/chat/jobs.")
        raise HTTPException(
            status_code=400, detail={"error_code": "backend.emptyMessage", "params": {}}
        )

    return _dispatch_async_chat_job(request, svc)


@chat_router.get("/api/chat/jobs/{job_id}", response_model=ChatResponse)
def get_chat_job_endpoint(
    job_id: str,
    svc: AppServices = Depends(get_app_services),
) -> ChatResponse:
    """
    Poll the status of an asynchronous chat job and return the completed `ChatResponse` once finished.

    Args:
        job_id: Target chat job identifier (`chatjob-...`).
        svc: Injected AppServices dependency bundle.

    Returns:
        ChatResponse: Pending status with `partial_content` or completed `ChatResponse` record.

    Raises:
        ChatJobNotFoundError: Status 404 when `job_id` does not exist.

    """
    chat_mgr = svc.get_chat_job_manager()
    job_resp = chat_mgr.get_chat_job_response(job_id)
    if job_resp is None:
        raise ChatJobNotFoundError(params={"job_id": job_id})
    return job_resp


@chat_router.post("/api/chat/stream")
async def chat_stream_endpoint(
    request: ChatRequest,
    req: Request,
    svc: AppServices = Depends(get_app_services),
) -> StreamingResponse:
    """
    Handle chat requests with Server-Sent Events (SSE) streaming.

    Args:
        request: The incoming ChatRequest.
        req: The incoming FastAPI HTTP request.
        svc: Injected AppServices dependency bundle.

    Returns:
        StreamingResponse: StreamingResponse streaming SSE token and result events.

    """
    _apply_provider_headers(request, req)

    return StreamingResponse(
        svc.stream_chat_events(request, req=req),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@chat_router.post("/api/execute-sql", response_model=ExecuteSqlResponse)
def execute_sql_endpoint(
    request: ExecuteSqlRequest,
    svc: AppServices = Depends(get_app_services),
) -> ExecuteSqlResponse:
    """
    Execute an arbitrary SQL query against the database.

    Args:
        request: The request containing the query and optional db_path.
        svc: Injected AppServices dependency bundle.

    Returns:
        ExecuteSqlResponse: The execution response containing results or error.

    """
    if not request.query.strip():
        return ExecuteSqlResponse(
            error=ApiError(error_code="backend.emptyMessage", params={})
        )

    try:
        db_path = svc.validate_db_path(request.db_path)
        results = svc.execute_sql(db_path, request.query)
        return ExecuteSqlResponse(sqlResult=results, rows=results)
    except (HTTPException, InvalidDatabasePathError):
        raise
    except ValueError as ve:
        logger.error(f"Database error during sql execution request: {ve}")
        return ExecuteSqlResponse(error=svc.parse_error(ve))
    except (T1DError, duckdb.Error, OSError, RuntimeError, TypeError, KeyError) as e:
        logger.exception("Unexpected error during sql execution request.")
        return ExecuteSqlResponse(error=svc.parse_error(e))


def _dispatch_async_eval_job(
    request: NlToSqlEvalRequest,
    svc: AppServices,
) -> NlToSqlEvalResponse:
    """
    Validate `request.db_path` and enqueue an asynchronous NL-to-SQL evaluation job in `ChatJobManager`.

    Args:
        request: Validated `NlToSqlEvalRequest` payload.
        svc: Active `AppServices` bundle.

    Returns:
        NlToSqlEvalResponse: Immediate pending ticket (`status="pending"`, `job_id="chatjob-..."`).

    """
    resolved_db = svc.validate_db_path(request.db_path)
    normalized_req = request.model_copy(
        update={"db_path": resolved_db, "async_poll": False}
    )

    def _run_eval_worker(
        job_req: NlToSqlEvalRequest, on_progress: Callable[[str], None]
    ) -> NlToSqlEvalResponse:
        """Execute NL-to-SQL evaluation turn in background thread."""
        resp = execute_nl_to_sql_eval_turn(
            request=job_req,
            db_validator=svc.validate_db_path,
            execute_fn=svc.execute_sql,
            translate_fn=svc.translate_nl_to_sql,
            error_parser=svc.parse_error,
        )
        on_progress(resp.content)
        return resp

    chat_mgr = svc.get_chat_job_manager()
    ticket = chat_mgr.submit_eval_job(normalized_req, _run_eval_worker)
    return NlToSqlEvalResponse(
        job_id=ticket["job_id"],
        status="pending",
        question=request.question,
        model=request.model,
        content="",
        partial_content="",
    )


@chat_router.post("/api/eval/nl-to-sql", response_model=NlToSqlEvalResponse)
def eval_nl_to_sql_endpoint(
    request: NlToSqlEvalRequest,
    req: Request,
    async_poll: bool = False,
    svc: AppServices = Depends(get_app_services),
) -> NlToSqlEvalResponse:
    """
    Evaluate a natural-language question against a gold-standard SQL query on the read-only DuckDB cursor.

    Args:
        request: Incoming `NlToSqlEvalRequest` payload.
        req: Incoming FastAPI HTTP request.
        async_poll: Optional query parameter to dispatch asynchronously and return a pending ticket.
        svc: Injected `AppServices` dependency bundle.

    Returns:
        NlToSqlEvalResponse: Evaluation metrics, generated SQL, and read-only DuckDB result rows.

    Raises:
        HTTPException: If `request.question` is empty.

    """
    _apply_provider_headers(request, req)

    if not request.question.strip():
        raise HTTPException(
            status_code=400, detail={"error_code": "backend.emptyMessage", "params": {}}
        )

    if request.async_poll or async_poll:
        return _dispatch_async_eval_job(request, svc)

    return execute_nl_to_sql_eval_turn(
        request=request,
        db_validator=svc.validate_db_path,
        execute_fn=svc.execute_sql,
        translate_fn=svc.translate_nl_to_sql,
        error_parser=svc.parse_error,
    )


@chat_router.post("/api/eval/nl-to-sql/jobs", response_model=NlToSqlEvalResponse)
def submit_eval_nl_to_sql_job_endpoint(
    request: NlToSqlEvalRequest,
    req: Request,
    svc: AppServices = Depends(get_app_services),
) -> NlToSqlEvalResponse:
    """
    Enqueue an asynchronous NL-to-SQL Gold-Standard evaluation job and immediately return a pending ticket.

    Args:
        request: Incoming `NlToSqlEvalRequest` payload.
        req: Incoming FastAPI HTTP request.
        svc: Injected `AppServices` dependency bundle.

    Returns:
        NlToSqlEvalResponse: Immediate pending evaluation ticket (`status="pending"`, `job_id="chatjob-..."`).

    Raises:
        HTTPException: If `request.question` is empty.

    """
    _apply_provider_headers(request, req)

    if not request.question.strip():
        raise HTTPException(
            status_code=400, detail={"error_code": "backend.emptyMessage", "params": {}}
        )

    return _dispatch_async_eval_job(request, svc)


@chat_router.get(
    "/api/eval/nl-to-sql/jobs/{job_id}", response_model=NlToSqlEvalResponse
)
def get_eval_nl_to_sql_job_endpoint(
    job_id: str,
    svc: AppServices = Depends(get_app_services),
) -> NlToSqlEvalResponse:
    """
    Poll the status of an asynchronous NL-to-SQL evaluation job and return the completed `NlToSqlEvalResponse`.

    Args:
        job_id: Target evaluation job identifier (`chatjob-...`).
        svc: Injected `AppServices` dependency bundle.

    Returns:
        NlToSqlEvalResponse: Pending status or completed `NlToSqlEvalResponse` record.

    Raises:
        ChatJobNotFoundError: Status 404 when `job_id` does not exist.

    """
    chat_mgr = svc.get_chat_job_manager()
    eval_resp = chat_mgr.get_eval_job_response(job_id)
    if eval_resp is None:
        raise ChatJobNotFoundError(params={"job_id": job_id})
    return eval_resp
