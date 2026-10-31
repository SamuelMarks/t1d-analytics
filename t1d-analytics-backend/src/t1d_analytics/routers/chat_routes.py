"""Chat, NL-to-SQL streaming, and read-only SQL execution endpoints."""

import logging
from typing import AsyncIterator, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from t1d_analytics.api_schemas import (
    ApiError,
    ChatRequest,
    ChatResponse,
    ExecuteSqlRequest,
    ExecuteSqlResponse,
)
from t1d_analytics.chat_service import (
    execute_chat_turn,
    generate_sse_chat_events,
)
from t1d_analytics.errors import InvalidDatabasePathError

logger = logging.getLogger(__name__)

chat_router = APIRouter()


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
    import t1d_analytics.api as api_mod

    return generate_sse_chat_events(
        request=request,
        req=req,
        db_validator=api_mod.validate_db_path,
        execute_fn=api_mod.execute_sql,
        connect_fn=api_mod.duckdb.connect,
        schema_fn=api_mod.get_database_schema,
        stream_fn=api_mod.stream_llm_tokens,
        error_parser=api_mod.parse_error,
    )


@chat_router.post("/api/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest, req: Request) -> ChatResponse:
    """
    Handle incoming chat messages and execute SQL or NLP queries.

    Args:
    ----
        request: The incoming ChatRequest.
        req: The incoming FastAPI HTTP request.

    Returns:
    -------
        ChatResponse: A ChatResponse indicating success or error.

    Raises:
    ------
        HTTPException: If request parameters are invalid.

    """
    import t1d_analytics.api as api_mod

    api_mod.enforce_rate_limit(req)
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
        db_validator=api_mod.validate_db_path,
        execute_fn=api_mod.execute_sql,
        translate_fn=api_mod.generate_sql_from_nl,
        error_parser=api_mod.parse_error,
    )


@chat_router.post("/api/chat/stream")
async def chat_stream_endpoint(request: ChatRequest, req: Request) -> StreamingResponse:
    """
    Handle chat requests with Server-Sent Events (SSE) streaming.

    Args:
    ----
        request: The incoming ChatRequest.
        req: The incoming FastAPI HTTP request.

    Returns:
    -------
        StreamingResponse: StreamingResponse streaming SSE token and result events.

    """
    import t1d_analytics.api as api_mod

    api_mod.enforce_rate_limit(req)
    provider_hdr = req.headers.get("x-provider")
    api_key_hdr = req.headers.get("x-provider-api-key")
    if provider_hdr and not request.provider:
        request.provider = provider_hdr
    if api_key_hdr and not request.api_key:
        request.api_key = api_key_hdr

    return StreamingResponse(
        api_mod.stream_chat_events(request, req=req),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@chat_router.post("/api/execute-sql", response_model=ExecuteSqlResponse)
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
        ExecuteSqlResponse: The execution response containing results or error.

    """
    import t1d_analytics.api as api_mod

    api_mod.enforce_rate_limit(req)
    if not request.query.strip():
        return ExecuteSqlResponse(
            error=ApiError(error_code="backend.emptyMessage", params={})
        )

    try:
        db_path = api_mod.validate_db_path(request.db_path)
        results = api_mod.execute_sql(db_path, request.query)
        return ExecuteSqlResponse(sqlResult=results, rows=results)
    except (HTTPException, InvalidDatabasePathError):
        raise
    except ValueError as ve:
        logger.error(f"Database error during sql execution request: {ve}")
        return ExecuteSqlResponse(error=api_mod.parse_error(ve))
    except Exception as e:
        logger.exception("Unexpected error during sql execution request.")
        return ExecuteSqlResponse(error=api_mod.parse_error(e))
