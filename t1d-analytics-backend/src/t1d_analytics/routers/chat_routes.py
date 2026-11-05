"""Chat, NL-to-SQL streaming, and read-only SQL execution endpoints."""

import logging
from typing import AsyncIterator, Optional

import duckdb
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from t1d_analytics.api_dependencies import AppServices, get_app_services
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
from t1d_analytics.errors import InvalidDatabasePathError, T1DError

logger = logging.getLogger(__name__)

chat_router = APIRouter()


def stream_chat_events(
    request: ChatRequest,
    req: Optional[Request] = None,
    services: Optional[AppServices] = None,
) -> AsyncIterator[str]:
    """
    Generate Server-Sent Events for streaming chat queries and SQL results.

    Args:
    ----
        request: The incoming ChatRequest.
        req: Optional FastAPI Request instance to detect client disconnection.
        services: Optional injected AppServices dependency bundle.

    Returns:
    -------
        AsyncIterator[str]: SSE data lines formatted as JSON strings.

    """
    svc = services or get_app_services()
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


@chat_router.post("/api/chat", response_model=ChatResponse)
def chat_endpoint(
    request: ChatRequest,
    req: Request,
    svc: AppServices = Depends(get_app_services),
) -> ChatResponse:
    """
    Handle incoming chat messages and execute SQL or NLP queries.

    Args:
    ----
        request: The incoming ChatRequest.
        req: The incoming FastAPI HTTP request.
        svc: Injected AppServices dependency bundle.

    Returns:
    -------
        ChatResponse: A ChatResponse indicating success or error.

    Raises:
    ------
        HTTPException: If request parameters are invalid.

    """
    svc.enforce_rate_limit(req)
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
        db_validator=svc.validate_db_path,
        execute_fn=svc.execute_sql,
        translate_fn=svc.generate_sql_from_nl,
        error_parser=svc.parse_error,
    )


@chat_router.post("/api/chat/stream")
async def chat_stream_endpoint(
    request: ChatRequest,
    req: Request,
    svc: AppServices = Depends(get_app_services),
) -> StreamingResponse:
    """
    Handle chat requests with Server-Sent Events (SSE) streaming.

    Args:
    ----
        request: The incoming ChatRequest.
        req: The incoming FastAPI HTTP request.
        svc: Injected AppServices dependency bundle.

    Returns:
    -------
        StreamingResponse: StreamingResponse streaming SSE token and result events.

    """
    svc.enforce_rate_limit(req)
    provider_hdr = req.headers.get("x-provider")
    api_key_hdr = req.headers.get("x-provider-api-key")
    if provider_hdr and not request.provider:
        request.provider = provider_hdr
    if api_key_hdr and not request.api_key:
        request.api_key = api_key_hdr

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
    req: Request,
    svc: AppServices = Depends(get_app_services),
) -> ExecuteSqlResponse:
    """
    Execute an arbitrary SQL query against the database.

    Args:
    ----
        request: The request containing the query and optional db_path.
        req: The incoming FastAPI HTTP request.
        svc: Injected AppServices dependency bundle.

    Returns:
    -------
        ExecuteSqlResponse: The execution response containing results or error.

    """
    svc.enforce_rate_limit(req)
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


@chat_router.post("/api/eval/nl-to-sql", response_model=NlToSqlEvalResponse)
def eval_nl_to_sql_endpoint(
    request: NlToSqlEvalRequest,
    req: Request,
    svc: AppServices = Depends(get_app_services),
) -> NlToSqlEvalResponse:
    """
    Evaluate a natural-language question against a gold-standard SQL query on the read-only DuckDB cursor.

    Args:
    ----
        request: Incoming `NlToSqlEvalRequest` payload.
        req: Incoming FastAPI HTTP request.
        svc: Injected `AppServices` dependency bundle.

    Returns:
    -------
        NlToSqlEvalResponse: Evaluation metrics, generated SQL, and read-only DuckDB result rows.

    Raises:
    ------
        HTTPException: If `request.question` is empty.

    """
    svc.enforce_rate_limit(req)
    provider_hdr = req.headers.get("x-provider")
    api_key_hdr = req.headers.get("x-provider-api-key")
    if provider_hdr and not request.provider:
        request.provider = provider_hdr
    if api_key_hdr and not request.api_key:
        request.api_key = api_key_hdr

    if not request.question.strip():
        raise HTTPException(
            status_code=400, detail={"error_code": "backend.emptyMessage", "params": {}}
        )

    return execute_nl_to_sql_eval_turn(
        request=request,
        db_validator=svc.validate_db_path,
        execute_fn=svc.execute_sql,
        translate_fn=svc.generate_sql_from_nl,
        error_parser=svc.parse_error,
    )
