"""Server-Sent Events (SSE) framing, client disconnect cancellation, and streaming chat orchestration."""

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Callable, Iterator
from typing import Literal, Protocol

import duckdb
from fastapi import HTTPException
from typing_extensions import TypedDict

from t1d_analytics.api_schemas import (
    ApiError,
    ChatRequest,
    SqlGoldComparisonModel,
)
from t1d_analytics.chat_turn_executor import (
    resolve_db_provenance_flags,
    resolve_sql_mode_query,
)
from t1d_analytics.duckdb_pool import connect_duckdb_read_only_pooled
from t1d_analytics.errors import (
    ErrorDetailDict,
    InvalidDatabasePathError,
    T1DError,
)
from t1d_analytics.llm_client import resolve_provider_and_model
from t1d_analytics.nl_to_sql_translator import (
    build_sql_prompt_with_history,
    compute_optional_gold_comparison,
)
from t1d_analytics.sql_fallback_synthesizer import (
    TranslationResult,
    extract_fallback_error_from_text,
    format_llm_failure_reason,
    is_fallback_response_text,
    is_recoverable_local_ollama_crash,
    lookup_curated_template_sql,
    synthesize_fallback_clinical_sql,
)
from t1d_analytics.sql_utils import (
    SqlValue,
    classify_sql_input,
    extract_sql_from_response,
    is_valid_read_only_sql,
)
from t1d_analytics.training_data import compare_predicted_and_gold_sql

logger = logging.getLogger(__name__)

NL = """
"""
DEFAULT_SSE_SEP = """

"""
STREAM_DONE_SENTINEL = "_T1D_SSE_STREAM_EOF_"


class SseTokenEvent(TypedDict):
    """Typed dictionary representing an SSE token chunk event."""

    event: Literal["token"]
    token: str


class SseProgressEvent(TypedDict):
    """Typed dictionary representing an SSE execution progress event."""

    event: Literal["execution_progress"]
    status: str


class SseSqlGeneratedEvent(TypedDict):
    """Typed dictionary representing an SSE generated SQL notification event."""

    event: Literal["sql_generated"]
    sqlQuery: str


class SseErrorEvent(TypedDict):
    """Typed dictionary representing a top-level SSE validation error event."""

    event: Literal["error"]
    error: str


class SqlGoldComparisonPayloadDict(TypedDict, total=False):
    """Typed dictionary representing serialized gold-standard SQL evaluation metrics in SSE frames."""

    gold_sql: str
    pred_sql: str
    exact_match: bool
    execution_accuracy: bool
    syntax_error: bool
    gold_row_count: int
    pred_row_count: int
    gold_columns: list[str]
    pred_columns: list[str]
    column_overlap_pct: float
    gold_rows_preview: list[dict[str, SqlValue]]
    error_message: str | None


class SseTerminalPayload(TypedDict, total=False):
    """Typed dictionary representing the terminal payload of a chat SSE stream."""

    content: str
    sqlQuery: str | None
    sqlResult: list[dict[str, SqlValue]] | None
    error: ErrorDetailDict | dict[str, SqlValue] | None
    is_synthetic_data: bool
    is_rule_based_sql: bool
    generation_mode: str
    llm_error: str | None
    llm_error_code: str | None
    data_provenance: str
    gold_comparison: SqlGoldComparisonPayloadDict | None


class SseDoneEvent(SseTerminalPayload, total=False):
    """Typed dictionary representing a single terminal SSE `'done'` event."""

    event: Literal["done"]


SseEventPayload = (
    SseTokenEvent
    | SseProgressEvent
    | SseSqlGeneratedEvent
    | SseErrorEvent
    | SseDoneEvent
)


class DisconnectCheckableProtocol(Protocol):
    """Structural protocol for ASGI/FastAPI requests supporting `await req.is_disconnected()`."""

    async def is_disconnected(self) -> bool:
        """Return True if the HTTP client has disconnected."""


def format_sse_frame(payload: SseEventPayload, sse_sep: str = DEFAULT_SSE_SEP) -> str:
    """
    Serialize a strongly-typed `SseEventPayload` into a wire-format SSE `data: ...` frame.

    Args:
        payload: Typed SSE event dictionary.
        sse_sep: Double-newline SSE frame separator.

    Returns:
        str: Formatted SSE `data: <json><sse_sep>` string.

    """
    return f"data: {json.dumps(payload)}{sse_sep}"


def emit_sse_terminal_payload(
    payload: SseTerminalPayload,
    sse_sep: str,
) -> Iterator[str]:
    """
    Format and yield a single terminal `'done'` SSE data frame without duplicate `'result'` doubling.

    Args:
        payload: Terminal chat response payload dictionary.
        sse_sep: Double-newline SSE frame separator.

    Yields:
        str: Single formatted `'done'` SSE frame.

    """
    done_event: SseDoneEvent = {"event": "done", **payload}
    yield format_sse_frame(done_event, sse_sep)


def next_stream_token_or_eof(iterator: Iterator[str]) -> str:
    """
    Pull the next token from a synchronous token iterator or return `STREAM_DONE_SENTINEL` on exhaustion.

    Args:
        iterator: Synchronous token iterator produced by `stream_fn`.

    Returns:
        str: Next token string or `STREAM_DONE_SENTINEL` when exhausted.

    """
    try:
        return next(iterator)
    except StopIteration:
        return STREAM_DONE_SENTINEL


def generate_sql_mode_sse_frames(
    db_path: str,
    message: str,
    sse_sep: str,
    execute_fn: Callable[[str, str], list[dict[str, SqlValue]]],
    connect_fn: Callable[..., duckdb.DuckDBPyConnection],
    schema_fn: Callable[[duckdb.DuckDBPyConnection], str],
    error_parser: Callable[[Exception], ApiError],
) -> Iterator[str]:
    """
    Generate SSE frames for direct SQL mode (`model in ('sql', '')`).

    Args:
        db_path: Validated DuckDB file path.
        message: User message or SQL query.
        sse_sep: Double-newline SSE frame separator.
        execute_fn: Read-only SQL execution callable.
        connect_fn: DuckDB connection factory callable.
        schema_fn: Database schema extraction callable.
        error_parser: Exception to ApiError converter callable.

    Yields:
        str: Formatted SSE event frames.

    """
    tok_event: SseTokenEvent = {
        "event": "token",
        "token": f"Executing SQL query...{NL}",
    }
    prog_event: SseProgressEvent = {
        "event": "execution_progress",
        "status": "Executing SQL query...",
    }
    yield format_sse_frame(tok_event, sse_sep)
    yield format_sse_frame(prog_event, sse_sep)
    sql_query = message
    provenance, is_synthetic = resolve_db_provenance_flags(
        db_path, connect_fn=connect_fn
    )
    is_raw_sql = classify_sql_input(message) != "natural_language"
    try:
        sql_query = resolve_sql_mode_query(
            db_path, message, connect_fn=connect_fn, schema_fn=schema_fn
        )
        results = execute_fn(db_path, sql_query)
        res_dict: SseTerminalPayload = {
            "content": "backend.literalSql",
            "sqlQuery": sql_query,
            "sqlResult": results,
            "error": None,
            "is_synthetic_data": is_synthetic,
            "is_rule_based_sql": (not is_raw_sql),
            "generation_mode": "literal_sql" if is_raw_sql else "rule_based_fallback",
            "data_provenance": provenance,
        }
        yield from emit_sse_terminal_payload(res_dict, sse_sep)
    except (T1DError, duckdb.Error, OSError, RuntimeError, ValueError) as e:
        err_dict = error_parser(e).model_dump()
        err_dict_payload: SseTerminalPayload = {
            "content": "backend.errorDbExecution",
            "sqlQuery": sql_query,
            "sqlResult": None,
            "error": err_dict,
            "is_synthetic_data": is_synthetic,
            "is_rule_based_sql": (not is_raw_sql),
            "generation_mode": "literal_sql" if is_raw_sql else "rule_based_fallback",
            "data_provenance": provenance,
        }
        yield from emit_sse_terminal_payload(err_dict_payload, sse_sep)


def serialize_gold_comparison(
    model: SqlGoldComparisonModel | None,
) -> SqlGoldComparisonPayloadDict | None:
    """
    Convert an optional `SqlGoldComparisonModel` into a `SqlGoldComparisonPayloadDict`.

    Args:
        model: Optional `SqlGoldComparisonModel` instance.

    Returns:
        SqlGoldComparisonPayloadDict | None: Typed dictionary representation or None.

    """
    if model is None:
        return None
    return {
        "gold_sql": model.gold_sql,
        "pred_sql": model.pred_sql,
        "exact_match": model.exact_match,
        "execution_accuracy": model.execution_accuracy,
        "syntax_error": model.syntax_error,
        "gold_row_count": model.gold_row_count,
        "pred_row_count": model.pred_row_count,
        "gold_columns": model.gold_columns,
        "pred_columns": model.pred_columns,
        "column_overlap_pct": model.column_overlap_pct,
        "gold_rows_preview": model.gold_rows_preview,
        "error_message": model.error_message,
    }


def finalize_nl_sse_frames(
    db_path: str,
    full_response: str,
    auto_execute: bool,
    sse_sep: str,
    execute_fn: Callable[[str, str], list[dict[str, SqlValue]]],
    error_parser: Callable[[Exception], ApiError],
    connect_fn: Callable[..., duckdb.DuckDBPyConnection] = duckdb.connect,
    used_fallback: bool | None = None,
    message: str = "",
    gold_sql: str | None = None,
    llm_error: str | None = None,
    llm_error_code: str | None = None,
) -> Iterator[str]:
    """
    Emit `sql_generated`, optional `execution_progress`, and terminal SSE `'done'` frame for an NL response.

    Args:
        db_path: Validated DuckDB file path.
        full_response: Complete assistant markdown text.
        auto_execute: Whether to automatically execute the extracted SQL.
        sse_sep: Double-newline SSE frame separator.
        execute_fn: Read-only SQL execution callable.
        error_parser: Exception to ApiError converter callable.
        connect_fn: DuckDB connection factory callable.
        used_fallback: Optional explicit flag indicating whether rule-based fallback synthesis was used.
        message: Original natural-language question for gold-standard lookup.
        gold_sql: Optional explicit gold-standard SQL query from the request.
        llm_error: Optional human-readable diagnostic reason why LLM inference failed.
        llm_error_code: Optional structured error code for the LLM failure.

    Yields:
        str: Formatted SSE event frames.

    """
    provenance, is_synthetic = resolve_db_provenance_flags(
        db_path, connect_fn=connect_fn
    )
    sql_query = extract_sql_from_response(full_response)
    effective_fallback = (
        used_fallback
        if used_fallback is not None
        else is_fallback_response_text(full_response)
    )
    if not effective_fallback and message:
        curated_tmpl_sql = lookup_curated_template_sql(message, db_path=db_path)
        if curated_tmpl_sql is not None:
            comp_check = compare_predicted_and_gold_sql(
                db_path=db_path,
                pred_sql=sql_query,
                gold_sql=curated_tmpl_sql,
                execute_fn=execute_fn,
            )
            if not comp_check.execution_accuracy or comp_check.syntax_error:
                sql_query = curated_tmpl_sql
                full_response = f"```sql{NL}{curated_tmpl_sql}{NL}```"
                effective_fallback = True
                llm_error = (
                    llm_error
                    or comp_check.error_message
                    or "LLM SQL did not match curated clinical reference execution."
                )
                llm_error_code = llm_error_code or "backend.llmTranslationError"
    resolved_llm_error = (
        llm_error
        if llm_error is not None
        else (
            extract_fallback_error_from_text(full_response)
            if effective_fallback
            else None
        )
    )
    resolved_llm_error_code = (
        llm_error_code
        if llm_error_code is not None
        else ("backend.llmTranslationError" if resolved_llm_error else None)
    )
    translation = TranslationResult(
        full_response=full_response,
        sql_query=sql_query,
        used_fallback=effective_fallback,
        generation_mode="rule_based_fallback" if effective_fallback else "llm",
        llm_error=resolved_llm_error,
        llm_error_code=resolved_llm_error_code,
    )
    sql_results: list[dict[str, SqlValue]] | None = None
    sql_err_dict: dict[str, SqlValue] | None = None
    pred_err_str: str | None = None
    if translation.sql_query:
        sql_gen_event: SseSqlGeneratedEvent = {
            "event": "sql_generated",
            "sqlQuery": translation.sql_query,
        }
        yield format_sse_frame(sql_gen_event, sse_sep)
        if auto_execute:
            prog_event: SseProgressEvent = {
                "event": "execution_progress",
                "status": "Executing SQL query...",
            }
            yield format_sse_frame(prog_event, sse_sep)
            try:
                sql_results = execute_fn(db_path, translation.sql_query)
            except (T1DError, duckdb.Error, OSError, RuntimeError, ValueError) as e:
                pred_err_str = str(e)
                sql_err_dict = error_parser(e).model_dump()

    gold_comp = compute_optional_gold_comparison(
        db_path=db_path,
        message=message,
        pred_sql=translation.sql_query,
        explicit_gold_sql=gold_sql,
        execute_fn=execute_fn,
        precomputed_pred_rows=sql_results,
        precomputed_pred_error=pred_err_str,
    )

    nl_res_dict: SseTerminalPayload = {
        "content": translation.full_response,
        "sqlQuery": translation.sql_query,
        "sqlResult": sql_results,
        "error": sql_err_dict,
        "is_synthetic_data": is_synthetic,
        "is_rule_based_sql": translation.used_fallback,
        "generation_mode": translation.generation_mode,
        "llm_error": translation.llm_error,
        "llm_error_code": translation.llm_error_code,
        "data_provenance": provenance,
        "gold_comparison": serialize_gold_comparison(gold_comp),
    }
    yield from emit_sse_terminal_payload(nl_res_dict, sse_sep)


async def is_request_disconnected(req: DisconnectCheckableProtocol | None) -> bool:
    """
    Check whether an optional request implementing `DisconnectCheckableProtocol` has disconnected.

    Args:
        req: Optional FastAPI Request or `DisconnectCheckableProtocol` instance.

    Returns:
        bool: True if `req` is present and disconnected.

    """
    return bool(req is not None and await req.is_disconnected())


def read_schema_from_db(
    db_path: str,
    connect_fn: Callable[..., duckdb.DuckDBPyConnection],
    schema_fn: Callable[[duckdb.DuckDBPyConnection], str],
) -> str:
    """
    Open a read-only DuckDB connection, extract the table schema text, and release the connection.

    Args:
        db_path: DuckDB file path.
        connect_fn: DuckDB connection factory callable.
        schema_fn: Database schema extraction callable.

    Returns:
        str: Extracted schema DDL/summary string.

    """
    with connect_duckdb_read_only_pooled(
        db_path=db_path,
        connect_fn=connect_fn,
        disable_external_access=True,
    ) as conn:
        return schema_fn(conn)


def validate_sse_request_or_error(
    request: ChatRequest,
    sse_sep: str,
    db_validator: Callable[[str | None], str],
) -> tuple[str, str | None]:
    """
    Validate `request.message` and `request.db_path`, returning `(db_path, error_sse_frame)`.

    Args:
        request: Incoming ChatRequest payload.
        sse_sep: Double-newline SSE frame separator.
        db_validator: Database path validator callable.

    Returns:
        tuple[str, str | None]: `(validated_db_path, optional_error_frame)`.

    """
    if not request.message.strip():
        empty_err: SseErrorEvent = {"event": "error", "error": "backend.emptyMessage"}
        return "", format_sse_frame(empty_err, sse_sep)

    try:
        return db_validator(request.db_path), None
    except (HTTPException, InvalidDatabasePathError):
        inv_err: SseErrorEvent = {
            "event": "error",
            "error": "backend.invalidDatabasePath",
        }
        return "", format_sse_frame(inv_err, sse_sep)


def recover_ollama_stream_fallback_token(
    resolved_prov: str,
    stream_exc: Exception,
    schema: str,
    message: str,
    sse_sep: str,
    model_name: str = "ollama",
) -> tuple[str, str]:
    """
    Recover from a local Ollama crash by synthesizing fallback SQL, or re-raise `stream_exc`.

    Args:
        resolved_prov: Resolved provider identifier.
        stream_exc: Exception raised during token streaming.
        schema: Extracted database schema string.
        message: User natural-language query.
        sse_sep: Double-newline SSE frame separator.
        model_name: Requested model identifier for diagnostic logging.

    Returns:
        tuple[str, str]: `(fallback_markdown_text, sse_token_frame)`.

    Raises:
        Exception: Re-raises `stream_exc` if not a recoverable local Ollama crash.

    """
    if not is_recoverable_local_ollama_crash(resolved_prov, stream_exc):
        raise stream_exc
    logger.warning(
        f"Local Ollama streaming for model '{model_name}' failed with {type(stream_exc).__name__} ({stream_exc}); using schema-aware clinical SQL fallback."
    )
    err_msg, err_code = format_llm_failure_reason(
        stream_exc, model_name=model_name, provider=resolved_prov
    )
    synthesized = synthesize_fallback_clinical_sql(
        schema,
        message,
        include_diagnostic_note=True,
        llm_error=err_msg,
        llm_error_code=err_code,
        model_name=model_name,
        provider=resolved_prov,
    )
    tok_event: SseTokenEvent = {"event": "token", "token": synthesized.full_response}
    return synthesized.full_response, format_sse_frame(tok_event, sse_sep)


async def stream_nl_mode_sse_frames(
    request: ChatRequest,
    req: DisconnectCheckableProtocol | None,
    db_path: str,
    sse_sep: str,
    execute_fn: Callable[[str, str], list[dict[str, SqlValue]]],
    connect_fn: Callable[..., duckdb.DuckDBPyConnection],
    schema_fn: Callable[[duckdb.DuckDBPyConnection], str],
    stream_fn: Callable[..., Iterator[str]],
    error_parser: Callable[[Exception], ApiError],
    heartbeat_interval_seconds: float = 3.0,
) -> AsyncIterator[str]:
    """
    Stream NL-to-SQL tokens with `: keep-alive` SSE heartbeats, cancelling `next_task` on client disconnect.

    Args:
        request: Incoming ChatRequest payload.
        req: Optional FastAPI Request instance to detect client disconnection.
        db_path: Validated DuckDB file path.
        sse_sep: Double-newline SSE frame separator.
        execute_fn: Read-only SQL execution callable.
        connect_fn: DuckDB connection factory callable.
        schema_fn: Database schema extraction callable.
        stream_fn: Token streaming callable.
        error_parser: Exception to ApiError converter callable.
        heartbeat_interval_seconds: Interval in seconds between `: keep-alive` SSE comments.

    Yields:
        str: Formatted SSE event frames and `: keep-alive` heartbeat comments.

    """
    schema = read_schema_from_db(db_path, connect_fn, schema_fn)
    resolved_prov, _ = resolve_provider_and_model(request.model, request.provider)
    prompt = build_sql_prompt_with_history(
        schema,
        request.message,
        history=request.history,
        compact_schema=(resolved_prov.lower() == "ollama"),
        model_name=request.model,
        provider=resolved_prov,
    )
    full_parts: list[str] = []
    used_fallback = False
    stream_llm_error: str | None = None
    stream_llm_error_code: str | None = None
    try:
        token_iter = iter(
            stream_fn(
                prompt,
                request.model,
                api_key=request.api_key,
                provider=request.provider,
            )
        )
        while True:
            if await is_request_disconnected(req):
                logger.info("Client disconnected during SSE token streaming.")
                return
            next_task: asyncio.Task[str] = asyncio.create_task(
                asyncio.to_thread(next_stream_token_or_eof, token_iter)
            )
            while True:
                done_set, _ = await asyncio.wait(
                    {next_task}, timeout=max(0.05, heartbeat_interval_seconds)
                )
                if done_set:
                    token = await next_task
                    break
                if await is_request_disconnected(req):
                    next_task.cancel()
                    logger.info("Client disconnected during SSE heartbeat wait.")
                    return
                yield f": keep-alive{sse_sep}"
            if token == STREAM_DONE_SENTINEL:
                break
            full_parts.append(token)
            tok_event: SseTokenEvent = {"event": "token", "token": token}
            yield format_sse_frame(tok_event, sse_sep)
    except Exception as stream_exc:
        stream_llm_error, stream_llm_error_code = format_llm_failure_reason(
            stream_exc, model_name=request.model, provider=resolved_prov
        )
        fb_resp, tok_frame = recover_ollama_stream_fallback_token(
            resolved_prov,
            stream_exc,
            schema,
            request.message,
            sse_sep,
            model_name=request.model,
        )
        full_parts.append(fb_resp)
        used_fallback = True
        yield tok_frame

    full_response = "".join(full_parts)
    for frame in finalize_nl_sse_frames(
        db_path,
        full_response,
        request.auto_execute,
        sse_sep,
        execute_fn=execute_fn,
        error_parser=error_parser,
        connect_fn=connect_fn,
        used_fallback=used_fallback,
        message=request.message,
        gold_sql=request.gold_sql,
        llm_error=stream_llm_error,
        llm_error_code=stream_llm_error_code,
    ):
        yield frame


async def generate_sse_chat_events(
    request: ChatRequest,
    req: DisconnectCheckableProtocol | None,
    db_validator: Callable[[str | None], str],
    execute_fn: Callable[[str, str], list[dict[str, SqlValue]]],
    connect_fn: Callable[..., duckdb.DuckDBPyConnection],
    schema_fn: Callable[[duckdb.DuckDBPyConnection], str],
    stream_fn: Callable[..., Iterator[str]],
    error_parser: Callable[[Exception], ApiError],
) -> AsyncIterator[str]:
    """
    Generate Server-Sent Events (SSE) for streaming chat queries and SQL execution results.

    Args:
        request: Incoming ChatRequest payload.
        req: Optional FastAPI Request instance to detect client disconnection.
        db_validator: Database path validator callable.
        execute_fn: SQL execution callable.
        connect_fn: DuckDB connection factory callable.
        schema_fn: Database schema extraction callable.
        stream_fn: Token streaming callable.
        error_parser: Exception to ApiError converter callable.

    Yields:
        str: SSE data lines formatted as JSON strings.

    """
    sse_sep = DEFAULT_SSE_SEP
    if await is_request_disconnected(req):
        logger.info("Client disconnected before stream generation.")
        return

    db_path, err_frame = validate_sse_request_or_error(request, sse_sep, db_validator)
    if err_frame is not None:
        yield err_frame
        return

    if request.model.strip().lower() in ("sql", "") or is_valid_read_only_sql(
        request.message
    ):
        for frame in generate_sql_mode_sse_frames(
            db_path,
            request.message,
            sse_sep,
            execute_fn=execute_fn,
            connect_fn=connect_fn,
            schema_fn=schema_fn,
            error_parser=error_parser,
        ):
            yield frame
        return

    try:
        async for frame in stream_nl_mode_sse_frames(
            request=request,
            req=req,
            db_path=db_path,
            sse_sep=sse_sep,
            execute_fn=execute_fn,
            connect_fn=connect_fn,
            schema_fn=schema_fn,
            stream_fn=stream_fn,
            error_parser=error_parser,
        ):
            yield frame
    except Exception as e:
        err_dict = error_parser(e).model_dump()
        nl_err_payload: SseTerminalPayload = {
            "content": "backend.errorNlpTranslation",
            "sqlQuery": None,
            "sqlResult": None,
            "error": err_dict,
        }
        for frame in emit_sse_terminal_payload(nl_err_payload, sse_sep):
            yield frame


__all__ = [
    "DEFAULT_SSE_SEP",
    "STREAM_DONE_SENTINEL",
    "DisconnectCheckableProtocol",
    "SqlGoldComparisonPayloadDict",
    "SseDoneEvent",
    "SseErrorEvent",
    "SseEventPayload",
    "SseProgressEvent",
    "SseSqlGeneratedEvent",
    "SseTerminalPayload",
    "SseTokenEvent",
    "emit_sse_terminal_payload",
    "finalize_nl_sse_frames",
    "format_sse_frame",
    "generate_sql_mode_sse_frames",
    "generate_sse_chat_events",
    "is_request_disconnected",
    "next_stream_token_or_eof",
    "read_schema_from_db",
    "recover_ollama_stream_fallback_token",
    "serialize_gold_comparison",
    "stream_nl_mode_sse_frames",
    "validate_sse_request_or_error",
]
