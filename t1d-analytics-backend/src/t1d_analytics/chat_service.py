"""Natural-language-to-SQL translation and Server-Sent Events (SSE) chat streaming services."""

import json
import logging
from typing import (
    AsyncIterator,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
)

import duckdb
from fastapi import HTTPException, Request
from typing_extensions import TypedDict

from t1d_analytics.analytics import (
    extract_sql_from_response as _extract_sql_from_response,
)
from t1d_analytics.analytics import (
    get_database_schema,
)
from t1d_analytics.api_schemas import (
    ApiError,
    ChatMessageModel,
    ChatRequest,
    ChatResponse,
)
from t1d_analytics.errors import (
    DatabaseNotFoundError,
    ErrorDetailDict,
    InvalidDatabasePathError,
    LlmTranslationError,
    MissingSdkError,
    ProviderNotConfiguredError,
    SchemaReadError,
    T1DError,
)
from t1d_analytics.gemma_bridge import prune_schema_with_gemma_rag
from t1d_analytics.llm_client import (
    PROVIDER_ENV_KEYS,
    AnyLlmFactoryProtocol,
    build_clinical_sql_prompt,
    complete_llm_prompt,
    create_any_llm_transport,
    extract_stream_delta,
    get_effective_api_key,
    resolve_provider_and_model,
)
from t1d_analytics.sql_agent import (
    AgenticSqlResult as AgenticSqlResult,
)
from t1d_analytics.sql_agent import (
    _classify_sql_error_hint as _classify_sql_error_hint,
)
from t1d_analytics.sql_agent import (
    execute_agentic_nl_to_sql as execute_agentic_nl_to_sql,
)
from t1d_analytics.sql_agent import (
    validate_sql_tokens as validate_sql_tokens,
)
from t1d_analytics.sql_utils import SqlValue

logger = logging.getLogger(__name__)


class SseTerminalPayload(TypedDict):
    """Typed dictionary representing the terminal payload of a chat SSE stream."""

    content: str
    sqlQuery: Optional[str]
    sqlResult: Optional[List[Dict[str, SqlValue]]]
    error: Optional[Union[ErrorDetailDict, Dict[str, SqlValue]]]


def _emit_sse_terminal_payload(
    payload: SseTerminalPayload,
    sse_sep: str,
) -> Iterator[str]:
    """
    Format and yield the terminal 'done' and 'result' SSE data frames.

    Args:
    ----
        payload: Terminal chat response payload dictionary.
        sse_sep: Double-newline SSE frame separator.

    Yields:
    ------
        str: Formatted 'done' and 'result' SSE frames.

    """
    yield f"data: {json.dumps({'event': 'done', **payload})}{sse_sep}"
    yield f"data: {json.dumps({'event': 'result', **payload})}{sse_sep}"


def build_sql_prompt_with_history(
    schema: str,
    nl_query: str,
    history: Optional[List[ChatMessageModel]] = None,
) -> str:
    """
    Construct the clinical Text-to-SQL system prompt for an LLM with conversation history and RAG pruning.

    Args:
    ----
        schema: The database schema text.
        nl_query: The clinician or user query.
        history: Optional list of prior conversation messages for multi-turn context.

    Returns:
    -------
        str: Formatted system and user prompt.

    """
    nl = chr(10)
    history_context = ""
    if history:
        turns: List[str] = []
        for msg in history[-6:]:
            role = "User" if msg.role == "user" else "Assistant"
            content = msg.content.strip()
            sql_info = f" (SQL: {msg.sqlQuery})" if msg.sqlQuery else ""
            turns.append(f"{role}: {content}{sql_info}")
        history_context = (
            f"Prior Conversation History:{nl}" + nl.join(turns) + f"{nl}{nl}"
        )

    enriched_schema = prune_schema_with_gemma_rag(nl_query, schema)

    return build_clinical_sql_prompt(
        schema=enriched_schema,
        nl_query=nl_query,
        history_context=history_context,
        require_markdown_fence=True,
    )


def stream_llm_completion_tokens(
    prompt: str,
    model_name: str,
    api_key: Optional[str] = None,
    provider: Optional[str] = None,
    llm_factory: Optional[AnyLlmFactoryProtocol] = None,
) -> Iterator[str]:
    """
    Stream tokens from LLM using any-llm, falling back gracefully to a single completion block.

    Args:
    ----
        prompt: The full prompt string.
        model_name: The LLM model name or provider/model string.
        api_key: Optional API key for external cloud provider.
        provider: Optional provider name override.
        llm_factory: Optional injectable factory creating an LLM transport client.

    Yields:
    ------
        str: Streamed token chunks.

    Raises:
    ------
        ProviderNotConfiguredError: If the requested cloud provider lacks an API key.

    """
    resolved_provider, actual_model = resolve_provider_and_model(model_name, provider)
    effective_api_key = get_effective_api_key(resolved_provider, api_key)

    if resolved_provider.lower() in PROVIDER_ENV_KEYS and not effective_api_key:
        env_var = PROVIDER_ENV_KEYS[resolved_provider.lower()]
        raise ProviderNotConfiguredError(
            params={"provider": resolved_provider, "env_var": env_var},
        )

    active_factory: AnyLlmFactoryProtocol = (
        llm_factory if llm_factory is not None else create_any_llm_transport
    )
    llm = (
        active_factory(resolved_provider, api_key=effective_api_key)
        if effective_api_key
        else active_factory(resolved_provider)
    )
    try:
        stream_resp = llm.completion(
            model=actual_model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )
        if isinstance(stream_resp, Iterable) and not isinstance(
            stream_resp, (str, bytes)
        ):
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


def translate_nl_to_sql(
    db_path: str,
    nl_query: str,
    model_name: str = "gemma4",
    history: Optional[List[ChatMessageModel]] = None,
    api_key: Optional[str] = None,
    provider: Optional[str] = None,
    connect_fn: Callable[..., duckdb.DuckDBPyConnection] = duckdb.connect,
    schema_fn: Callable[[duckdb.DuckDBPyConnection], str] = get_database_schema,
    llm_factory: Optional[AnyLlmFactoryProtocol] = None,
) -> tuple[str, str]:
    """
    Translate a natural language question into a DuckDB SQL query using `any-llm`.

    Args:
    ----
        db_path: Path to the DuckDB database file.
        nl_query: Natural language question.
        model_name: Model identifier or `provider/model` string.
        history: Optional list of prior conversation messages.
        api_key: Optional API key for cloud provider.
        provider: Optional provider override name.
        connect_fn: DuckDB connection factory callable.
        schema_fn: Schema extraction callable.
        llm_factory: Optional injectable factory creating an LLM transport client.

    Returns:
    -------
        tuple[str, str]: `(full_response, extracted_sql_query)`.

    Raises:
    ------
        MissingSdkError: If `any-llm-sdk` is missing.
        ProviderNotConfiguredError: If API key is missing.
        LlmTranslationError: If LLM API fails.
        DatabaseNotFoundError: If database file is missing.
        SchemaReadError: If reading schema fails.

    """
    resolved_provider, actual_model = resolve_provider_and_model(model_name, provider)

    logger.info(
        f"Translating natural language to SQL: '{nl_query}' using provider '{resolved_provider}' and model '{actual_model}'"
    )
    if llm_factory is None:
        try:
            import any_llm

            _ = any_llm
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
            conn = connect_fn(db_path, read_only=True)
        except duckdb.IOException as io_err:
            raise DatabaseNotFoundError(params={"error": str(io_err)}) from io_err
        schema = schema_fn(conn)
        conn.close()
        logger.debug(f"Retrieved schema: {schema[:200]}... (truncated)")
    except DatabaseNotFoundError as dnf:
        logger.error(f"Failed to read schema: {dnf}")
        raise
    except Exception as e:
        logger.error(f"Failed to read schema: {e}")
        raise SchemaReadError(params={"error": str(e)}) from e

    prompt = build_sql_prompt_with_history(schema, nl_query, history=history)
    nl = chr(10)

    try:
        logger.info(f"Sending prompt to LLM ({resolved_provider}/{actual_model})...")
        full_response = complete_llm_prompt(
            prompt=prompt,
            model_name=actual_model,
            api_key=effective_api_key,
            provider=resolved_provider,
            llm_factory=llm_factory,
        )
        logger.info(f"LLM response:{nl}{full_response}")
        sql_query = _extract_sql_from_response(full_response)
        return full_response, sql_query
    except Exception as e:
        logger.error(f"LLM translation error: {e}")
        raise LlmTranslationError(params={"error": str(e)}) from e


generate_sql_from_nl = translate_nl_to_sql


def execute_chat_turn(
    request: ChatRequest,
    db_validator: Callable[[Optional[str]], str],
    execute_fn: Callable[[str, str], List[Dict[str, SqlValue]]],
    translate_fn: Callable[..., Tuple[str, str]],
    error_parser: Callable[[Exception], ApiError],
) -> ChatResponse:
    """
    Execute a synchronous chat turn (literal SQL or natural-language-to-SQL with optional execution).

    Args:
    ----
        request: Validated `ChatRequest` payload.
        db_validator: Callable resolving and sandboxing the target database path.
        execute_fn: Callable executing a read-only SQL query against the database.
        translate_fn: Callable translating a natural-language question into `(full_response, sql_query)`.
        error_parser: Callable converting an `Exception` into a structured `ApiError`.

    Returns:
    -------
        ChatResponse: Structured chat response containing content, SQL query, result rows, or error details.

    Raises:
    ------
        HTTPException: If the database path validation raises an HTTPException.
        InvalidDatabasePathError: If the database path is outside the allowed workspace sandbox.

    """
    try:
        logger.info(
            f"Received chat request: model='{request.model}', message='{request.message}'"
        )
        db_path = db_validator(request.db_path)
        if request.model == "sql":
            sql_query = request.message
            results = execute_fn(db_path, sql_query)
            return ChatResponse(
                content="backend.literalSql",
                sqlResult=results,
                sqlQuery=sql_query,
            )

        full_response, sql_query = translate_fn(
            db_path,
            request.message,
            model_name=request.model,
            history=request.history,
            api_key=request.api_key,
            provider=request.provider,
        )
        nl_results: Optional[List[Dict[str, SqlValue]]] = None
        sql_error: Optional[ApiError] = None
        if sql_query and request.auto_execute:
            try:
                nl_results = execute_fn(db_path, sql_query)
            except (T1DError, ValueError, RuntimeError) as e:
                sql_error = error_parser(e)

        return ChatResponse(
            content=full_response,
            sqlQuery=sql_query,
            sqlResult=nl_results,
            error=sql_error,
        )
    except (HTTPException, InvalidDatabasePathError):
        raise
    except ValueError as ve:
        logger.error(f"Database error during chat request: {ve}")
        return ChatResponse(content="backend.errorDbExecution", error=error_parser(ve))
    except RuntimeError as re_err:
        logger.error(f"NLP error during chat request: {re_err}")
        return ChatResponse(
            content="backend.errorNlpTranslation", error=error_parser(re_err)
        )
    except Exception as e:
        logger.exception("Unexpected error during chat request.")
        return ChatResponse(content="backend.errorUnexpected", error=error_parser(e))


async def generate_sse_chat_events(
    request: ChatRequest,
    req: Optional[Request],
    db_validator: Callable[[Optional[str]], str],
    execute_fn: Callable[[str, str], List[Dict[str, SqlValue]]],
    connect_fn: Callable[..., duckdb.DuckDBPyConnection],
    schema_fn: Callable[[duckdb.DuckDBPyConnection], str],
    stream_fn: Callable[..., Iterator[str]],
    error_parser: Callable[[Exception], ApiError],
) -> AsyncIterator[str]:
    """
    Generate Server-Sent Events (SSE) for streaming chat queries and SQL execution results.

    Args:
    ----
        request: Incoming ChatRequest payload.
        req: Optional FastAPI Request object to detect client disconnection.
        db_validator: Database path validator callable.
        execute_fn: SQL execution callable.
        connect_fn: DuckDB connection factory callable.
        schema_fn: Database schema extraction callable.
        stream_fn: Token streaming callable.
        error_parser: Exception to ApiError converter callable.

    Yields:
    ------
        str: SSE data lines formatted as JSON strings.

    """
    nl = chr(10)
    sse_sep = f"{nl}{nl}"
    if req is not None and await req.is_disconnected():
        logger.info("Client disconnected before stream generation.")
        return

    if not request.message.strip():
        payload = json.dumps({"event": "error", "error": "backend.emptyMessage"})
        yield f"data: {payload}{sse_sep}"
        return

    try:
        db_path = db_validator(request.db_path)
    except (HTTPException, InvalidDatabasePathError):
        payload = json.dumps({"event": "error", "error": "backend.invalidDatabasePath"})
        yield f"data: {payload}{sse_sep}"
        return

    if request.model == "sql":
        exec_payload = json.dumps(
            {"event": "token", "token": f"Executing SQL query...{nl}"}
        )
        yield f"data: {exec_payload}{sse_sep}"
        yield f"data: {json.dumps({'event': 'execution_progress', 'status': 'Executing SQL query...'})}{sse_sep}"
        try:
            results = execute_fn(db_path, request.message)
            res_dict: SseTerminalPayload = {
                "content": "backend.literalSql",
                "sqlQuery": request.message,
                "sqlResult": results,
                "error": None,
            }
            for frame in _emit_sse_terminal_payload(res_dict, sse_sep):
                yield frame
        except (T1DError, duckdb.Error, OSError, RuntimeError, ValueError) as e:
            err_dict = error_parser(e).model_dump()
            err_dict_payload: SseTerminalPayload = {
                "content": "backend.errorDbExecution",
                "sqlQuery": request.message,
                "sqlResult": None,
                "error": err_dict,
            }
            for frame in _emit_sse_terminal_payload(err_dict_payload, sse_sep):
                yield frame
        return

    try:
        conn = connect_fn(db_path, read_only=True)
        schema = schema_fn(conn)
        conn.close()

        prompt = build_sql_prompt_with_history(
            schema, request.message, history=request.history
        )
        full_parts: List[str] = []
        for token in stream_fn(
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
            yield f"data: {tok_payload}{sse_sep}"
        full_response = "".join(full_parts)
        sql_query = _extract_sql_from_response(full_response)

        sql_results: Optional[List[Dict[str, SqlValue]]] = None
        sql_err_dict: Optional[Dict[str, SqlValue]] = None
        if sql_query:
            yield f"data: {json.dumps({'event': 'sql_generated', 'sqlQuery': sql_query})}{sse_sep}"
            if request.auto_execute:
                yield f"data: {json.dumps({'event': 'execution_progress', 'status': 'Executing SQL query...'})}{sse_sep}"
                try:
                    sql_results = execute_fn(db_path, sql_query)
                except (T1DError, duckdb.Error, OSError, RuntimeError, ValueError) as e:
                    sql_err_dict = error_parser(e).model_dump()

        nl_res_dict: SseTerminalPayload = {
            "content": full_response,
            "sqlQuery": sql_query,
            "sqlResult": sql_results,
            "error": sql_err_dict,
        }
        for frame in _emit_sse_terminal_payload(nl_res_dict, sse_sep):
            yield frame
    except (T1DError, duckdb.Error, OSError, RuntimeError, ValueError, TypeError) as e:
        err_dict = error_parser(e).model_dump()
        nl_err_payload: SseTerminalPayload = {
            "content": "backend.errorNlpTranslation",
            "sqlQuery": None,
            "sqlResult": None,
            "error": err_dict,
        }
        for frame in _emit_sse_terminal_payload(nl_err_payload, sse_sep):
            yield frame


stream_nl_to_sql_events = generate_sse_chat_events
