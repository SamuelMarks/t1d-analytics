"""Natural-language-to-SQL translation and Server-Sent Events (SSE) chat streaming services."""

import asyncio
import json
import logging
import urllib.error
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

from t1d_analytics.analytics import get_database_schema
from t1d_analytics.api_schemas import (
    ApiError,
    ChatMessageModel,
    ChatRequest,
    ChatResponse,
)
from t1d_analytics.diagnostics import detect_database_provenance
from t1d_analytics.errors import (
    DatabaseError,
    DatabaseNotFoundError,
    ErrorDetailDict,
    InvalidDatabasePathError,
    LlmProviderError,
    LlmTranslationError,
    MissingSdkError,
    ProviderNotConfiguredError,
    SchemaReadError,
    T1DError,
)
from t1d_analytics.llm_client import (
    PROVIDER_ENV_KEYS,
    AnyLlmFactoryProtocol,
    LlmTransportProtocol,
    build_clinical_sql_prompt,
    complete_llm_prompt,
    create_any_llm_transport,
    extract_stream_delta,
    get_effective_api_key,
    probe_ollama_tags_and_version,
    resolve_installed_ollama_model_tag,
    resolve_provider_and_model,
    sort_ollama_models,
)
from t1d_analytics.sql_agent import prune_schema_with_gemma_rag
from t1d_analytics.sql_fallback_synthesizer import (
    _FALLBACK_DIAGNOSTIC_PREFIX,
    _TABLE_1_FALLBACK_KEYWORDS,
    GenerationModeLiteral,
    TranslationResult,
    _build_fallback_table_1_sql,
    _find_matching_column,
    _is_recoverable_local_ollama_crash,
    _parse_schema_tables_and_columns,
    _score_table_for_query,
    _select_best_fallback_table,
    coerce_translation_result,
    is_fallback_response_text,
    synthesize_fallback_clinical_sql,
)
from t1d_analytics.sql_utils import (
    ALL_SQL_KEYWORDS,
    SqlValue,
    connect_duckdb_read_only_pooled,
    strip_sql_comments_and_parens,
)
from t1d_analytics.sql_utils import (
    extract_sql_from_response as _extract_sql_from_response,
)

logger = logging.getLogger(__name__)
_STREAM_DONE_SENTINEL = "_T1D_SSE_STREAM_EOF_"


def _next_stream_token_or_eof(iterator: Iterator[str]) -> str:
    """
    Pull the next token from a synchronous token iterator or return `_STREAM_DONE_SENTINEL` on exhaustion.

    Args:
    ----
        iterator: Synchronous token iterator produced by `stream_fn`.

    Returns:
    -------
        str: Next token string or `_STREAM_DONE_SENTINEL` when exhausted.

    """
    try:
        return next(iterator)
    except StopIteration:
        return _STREAM_DONE_SENTINEL


_VALID_SHOW_SECOND_TOKENS = {
    "tables",
    "databases",
    "schemas",
    "all",
    "variables",
}


def is_likely_sql_statement(query: str) -> bool:
    """
    Determine whether a user input string begins with a SQL keyword rather than natural language.

    Args:
    ----
        query: Raw user query string.

    Returns:
    -------
        bool: True if the input begins with a recognized SQL keyword (excluding natural language 'show ...' phrases), False otherwise.

    """
    unwrapped = strip_sql_comments_and_parens(query)
    if not unwrapped:
        return False
    tokens = unwrapped.split()
    first_word = tokens[0].lower().rstrip(";").strip()
    if first_word not in ALL_SQL_KEYWORDS:
        return False
    if first_word == "show" and len(tokens) > 1:
        second_word = tokens[1].lower().rstrip(";").strip()
        if second_word not in _VALID_SHOW_SECOND_TOKENS:
            return False
        if second_word == "all" and (
            len(tokens) < 3 or tokens[2].lower().rstrip(";").strip() != "tables"
        ):
            return False
    return True


def _probe_installed_ollama_names() -> List[str]:
    """
    Query the local Ollama `/api/tags` endpoint and return installed model names.

    Returns
    -------
        List[str]: Installed Ollama model names.

    """
    infos, _ = probe_ollama_tags_and_version(timeout=1.5, include_version=False)
    return [info.name for info in infos]


def _resolve_and_select_local_ollama_model(
    actual_model: str,
    probe_fn: Callable[[], List[str]] = _probe_installed_ollama_names,
) -> Tuple[str, Optional[str]]:
    """
    Normalize a requested local Ollama model against installed `/api/tags` models and return a lightweight fallback tag.

    Args:
    ----
        actual_model: Requested Ollama model name (without `ollama/` prefix).
        probe_fn: Callable returning the list of installed Ollama model names.

    Returns:
    -------
        Tuple[str, Optional[str]]: `(normalized_primary_model, fallback_lightweight_model)`.

    """
    try:
        installed = probe_fn()
    except (urllib.error.URLError, OSError, ValueError, RuntimeError):
        installed = []
    if not installed:
        return actual_model, None

    ranked = sort_ollama_models(installed)
    matched = resolve_installed_ollama_model_tag(actual_model, ranked)
    primary = matched if matched is not None else ranked[0]
    fallback: Optional[str] = None
    for cand in ranked:
        if cand != primary and not any(
            h in cand.lower() for h in ("26b", "27b", "70b")
        ):
            fallback = cand
            break
    return primary, fallback


class SseTerminalPayload(TypedDict, total=False):
    """Typed dictionary representing the terminal payload of a chat SSE stream."""

    content: str
    sqlQuery: Optional[str]
    sqlResult: Optional[List[Dict[str, SqlValue]]]
    error: Optional[Union[ErrorDetailDict, Dict[str, SqlValue]]]
    is_synthetic_data: bool
    is_rule_based_sql: bool
    generation_mode: str
    data_provenance: str


def _resolve_db_provenance_flags(
    db_path: str,
    connect_fn: Callable[..., duckdb.DuckDBPyConnection] = duckdb.connect,
) -> Tuple[str, bool]:
    """
    Determine `(data_provenance, is_synthetic_data)` for `db_path`.

    Args:
    ----
        db_path: Path to the target DuckDB file.
        connect_fn: DuckDB connection factory callable.

    Returns:
    -------
        Tuple[str, bool]: `(data_provenance, is_synthetic_data)`.

    """
    try:
        with connect_duckdb_read_only_pooled(
            db_path=db_path,
            connect_fn=connect_fn,
            disable_external_access=True,
        ) as conn:
            all_tables = [str(r[0]) for r in conn.execute("SHOW TABLES").fetchall()]
            provenance = detect_database_provenance(all_tables)
    except (duckdb.Error, OSError, RuntimeError, ValueError) as exc:
        logger.debug(f"Failed to resolve database provenance for '{db_path}': {exc}")
        provenance = "synthetic_seed"
    return provenance, (provenance != "jaeb_authentic")


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
    compact_schema: bool = False,
) -> str:
    """
    Construct the clinical Text-to-SQL system prompt for an LLM with conversation history and RAG pruning.

    Args:
    ----
        schema: The database schema text.
        nl_query: The clinician or user query.
        history: Optional list of prior conversation messages for multi-turn context.
        compact_schema: Whether to compact sample rows and table blocks for local CPU Ollama models.

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
        compact_schema=compact_schema,
    )


def _create_verified_llm_transport(
    resolved_provider: str,
    api_key: Optional[str],
    llm_factory: Optional[AnyLlmFactoryProtocol],
) -> LlmTransportProtocol:
    """
    Validate cloud provider API key readiness and instantiate an `LlmTransportProtocol`.

    Args:
    ----
        resolved_provider: Resolved LLM provider identifier.
        api_key: Optional explicit API key.
        llm_factory: Optional transport factory override.

    Returns:
    -------
        LlmTransportProtocol: Instantiated LLM transport client.

    Raises:
    ------
        ProviderNotConfiguredError: If the cloud provider requires an API key that is not set.

    """
    effective_api_key = get_effective_api_key(resolved_provider, api_key)
    if resolved_provider.lower() in PROVIDER_ENV_KEYS and not effective_api_key:
        env_var = PROVIDER_ENV_KEYS[resolved_provider.lower()]
        raise ProviderNotConfiguredError(
            params={"provider": resolved_provider, "env_var": env_var},
        )
    active_factory: AnyLlmFactoryProtocol = (
        llm_factory if llm_factory is not None else create_any_llm_transport
    )
    if effective_api_key:
        return active_factory(resolved_provider, api_key=effective_api_key)
    return active_factory(resolved_provider)


def _complete_non_streaming_with_fallback(
    llm: LlmTransportProtocol,
    resolved_provider: str,
    actual_model: str,
    fallback_model: Optional[str],
    prompt: str,
) -> str:
    """
    Execute a non-streaming completion call, retrying once with `fallback_model` on failure.

    Args:
    ----
        llm: Transport client instance.
        resolved_provider: Provider identifier.
        actual_model: Primary model name.
        fallback_model: Optional lightweight fallback model name.
        prompt: Prompt string.

    Returns:
    -------
        str: Extracted text from the completion response.

    """
    try:
        response = llm.completion(
            model=actual_model,
            messages=[{"role": "user", "content": prompt}],
        )
    except (T1DError, OSError, RuntimeError, ValueError, TypeError) as primary_exc:
        if fallback_model is not None:
            logger.warning(
                f"Local Ollama model '{actual_model}' failed ({type(primary_exc).__name__}: {primary_exc}); retrying with lightweight model '{fallback_model}'."
            )
            response = llm.completion(
                model=fallback_model,
                messages=[{"role": "user", "content": prompt}],
            )
        else:
            raise
    return extract_stream_delta(response, resolved_provider) or str(response)


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
    fallback_model: Optional[str] = None
    if resolved_provider.lower() == "ollama" and llm_factory is None:
        actual_model, fallback_model = _resolve_and_select_local_ollama_model(
            actual_model
        )
    llm = _create_verified_llm_transport(resolved_provider, api_key, llm_factory)
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
    except (T1DError, OSError, RuntimeError, ValueError, TypeError) as e:
        logger.warning(
            f"Native streaming failed for provider '{resolved_provider}' model '{actual_model}' ({type(e).__name__}: {e}). Falling back to non-streaming completion."
        )

    yield _complete_non_streaming_with_fallback(
        llm=llm,
        resolved_provider=resolved_provider,
        actual_model=actual_model,
        fallback_model=fallback_model,
        prompt=prompt,
    )


def _ensure_any_llm_installed_and_ready(
    resolved_provider: str,
    api_key: Optional[str],
    llm_factory: Optional[AnyLlmFactoryProtocol],
) -> Optional[str]:
    """
    Verify `any-llm-sdk` availability and API key configuration, returning the effective API key.

    Args:
    ----
        resolved_provider: Resolved LLM provider identifier.
        api_key: Optional explicit API key.
        llm_factory: Optional injectable LLM factory.

    Returns:
    -------
        Optional[str]: Resolved API key or None.

    Raises:
    ------
        MissingSdkError: If `any-llm-sdk` is not installed.
        ProviderNotConfiguredError: If required API key is missing.

    """
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
    return effective_api_key


def _extract_schema_for_translation(
    db_path: str,
    connect_fn: Callable[..., duckdb.DuckDBPyConnection],
    schema_fn: Callable[[duckdb.DuckDBPyConnection], str],
) -> str:
    """
    Open `db_path` read-only and extract its schema text for NL-to-SQL translation.

    Args:
    ----
        db_path: Path to the DuckDB database file.
        connect_fn: DuckDB connection factory callable.
        schema_fn: Schema extraction callable.

    Returns:
    -------
        str: Formatted database schema string.

    Raises:
    ------
        DatabaseNotFoundError: If `db_path` cannot be opened due to `duckdb.IOException`.
        SchemaReadError: If reading the database schema fails.

    """
    try:
        try:
            conn = connect_fn(db_path, read_only=True)
        except duckdb.IOException as io_err:
            raise DatabaseNotFoundError(params={"error": str(io_err)}) from io_err
        try:
            schema = schema_fn(conn)
        finally:
            conn.close()
        logger.debug(f"Retrieved schema: {schema[:200]}... (truncated)")
        return schema
    except DatabaseNotFoundError as dnf:
        logger.error(f"Failed to read schema: {dnf}")
        raise
    except (duckdb.Error, OSError, RuntimeError, ValueError, TypeError) as e:
        logger.error(f"Failed to read schema: {e}")
        raise SchemaReadError(params={"error": str(e)}) from e


def translate_nl_to_sql(
    db_path: str,
    nl_query: str,
    model_name: str = "gemma:2b",
    history: Optional[List[ChatMessageModel]] = None,
    api_key: Optional[str] = None,
    provider: Optional[str] = None,
    connect_fn: Callable[..., duckdb.DuckDBPyConnection] = duckdb.connect,
    schema_fn: Callable[[duckdb.DuckDBPyConnection], str] = get_database_schema,
    llm_factory: Optional[AnyLlmFactoryProtocol] = None,
) -> TranslationResult:
    """
    Translate a natural language question into a DuckDB SQL query using `any-llm`.

    Returns a `TranslationResult` which supports 2-tuple unpacking `(full_response, sql_query)`
    as well as structured `.used_fallback` and `.generation_mode` attributes.

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
        TranslationResult: Structured translation result `(full_response, extracted_sql_query)`.

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
    effective_api_key = _ensure_any_llm_installed_and_ready(
        resolved_provider, api_key, llm_factory
    )
    schema = _extract_schema_for_translation(db_path, connect_fn, schema_fn)

    is_local_ollama = resolved_provider.lower() == "ollama"
    fallback_model: Optional[str] = None
    if is_local_ollama and llm_factory is None:
        actual_model, fallback_model = _resolve_and_select_local_ollama_model(
            actual_model
        )
    prompt = build_sql_prompt_with_history(
        schema, nl_query, history=history, compact_schema=is_local_ollama
    )
    nl = chr(10)

    try:
        logger.info(f"Sending prompt to LLM ({resolved_provider}/{actual_model})...")
        try:
            full_response = complete_llm_prompt(
                prompt=prompt,
                model_name=actual_model,
                api_key=effective_api_key,
                provider=resolved_provider,
                llm_factory=llm_factory,
            )
        except (T1DError, OSError, RuntimeError, ValueError, TypeError) as primary_exc:
            if is_local_ollama and fallback_model is not None:
                logger.warning(
                    f"Local Ollama model '{actual_model}' failed ({type(primary_exc).__name__}: {primary_exc}); retrying once with lightweight model '{fallback_model}'."
                )
                full_response = complete_llm_prompt(
                    prompt=prompt,
                    model_name=fallback_model,
                    api_key=effective_api_key,
                    provider=resolved_provider,
                    llm_factory=llm_factory,
                )
            else:
                raise
        logger.info(f"LLM response:{nl}{full_response}")
        sql_query = _extract_sql_from_response(full_response)
        return TranslationResult(
            full_response=full_response,
            sql_query=sql_query,
            used_fallback=False,
            generation_mode="llm",
        )
    except Exception as e:
        if _is_recoverable_local_ollama_crash(resolved_provider, e):
            logger.warning(
                f"Local Ollama model '{actual_model}' failed with {type(e).__name__} ({e}); synthesizing schema-aware clinical SQL fallback."
            )
            return synthesize_fallback_clinical_sql(
                schema, nl_query, include_diagnostic_note=True
            )
        logger.error(f"LLM translation error ({type(e).__name__}): {e}")
        raise LlmTranslationError(params={"error": str(e)}) from e


def _resolve_sql_mode_query(
    db_path: str,
    message: str,
    connect_fn: Callable[..., duckdb.DuckDBPyConnection] = duckdb.connect,
    schema_fn: Callable[[duckdb.DuckDBPyConnection], str] = get_database_schema,
) -> str:
    """
    Resolve a query submitted in 'sql' mode, synthesizing schema-grounded SQL when input is natural language.

    Args:
    ----
        db_path: Validated path to the DuckDB database file.
        message: Raw user message string.
        connect_fn: DuckDB connection factory callable.
        schema_fn: Database schema extraction callable.

    Returns:
    -------
        str: Executable SQL query string.

    """
    if is_likely_sql_statement(message):
        return message
    with connect_duckdb_read_only_pooled(
        db_path=db_path,
        connect_fn=connect_fn,
        disable_external_access=True,
    ) as conn:
        schema = schema_fn(conn)
    synthesized = synthesize_fallback_clinical_sql(schema, message)
    return synthesized.sql_query


def execute_chat_turn(
    request: ChatRequest,
    db_validator: Callable[[Optional[str]], str],
    execute_fn: Callable[[str, str], List[Dict[str, SqlValue]]],
    translate_fn: Callable[..., Union[TranslationResult, Tuple[str, str]]],
    error_parser: Callable[[Exception], ApiError],
) -> ChatResponse:
    """
    Execute a synchronous chat turn (literal SQL or natural-language-to-SQL with optional execution).

    Args:
    ----
        request: Validated `ChatRequest` payload.
        db_validator: Callable resolving and sandboxing the target database path.
        execute_fn: Callable executing a read-only SQL query against the database.
        translate_fn: Callable translating a natural-language question into a `TranslationResult` or `(full_response, sql_query)` tuple.
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
        provenance, is_synthetic = _resolve_db_provenance_flags(db_path)
        is_raw_sql = is_likely_sql_statement(request.message)
        if request.model.strip().lower() in ("sql", "") or is_raw_sql:
            sql_query = _resolve_sql_mode_query(db_path, request.message)
            results = execute_fn(db_path, sql_query)
            return ChatResponse(
                content="backend.literalSql",
                sqlResult=results,
                sqlQuery=sql_query,
                is_synthetic_data=is_synthetic,
                is_rule_based_sql=(not is_raw_sql),
                generation_mode="literal_sql" if is_raw_sql else "rule_based_fallback",
                data_provenance=provenance,
            )

        raw_translation = translate_fn(
            db_path,
            request.message,
            model_name=request.model,
            history=request.history,
            api_key=request.api_key,
            provider=request.provider,
        )
        translation = coerce_translation_result(raw_translation)
        nl_results: Optional[List[Dict[str, SqlValue]]] = None
        sql_error: Optional[ApiError] = None
        if translation.sql_query and request.auto_execute:
            try:
                nl_results = execute_fn(db_path, translation.sql_query)
            except (T1DError, ValueError, RuntimeError) as e:
                sql_error = error_parser(e)

        return ChatResponse(
            content=translation.full_response,
            sqlQuery=translation.sql_query,
            sqlResult=nl_results,
            error=sql_error,
            is_synthetic_data=is_synthetic,
            is_rule_based_sql=translation.used_fallback,
            generation_mode=translation.generation_mode,
            data_provenance=provenance,
        )
    except (HTTPException, InvalidDatabasePathError):
        raise
    except (DatabaseError, ValueError) as ve:
        logger.error(f"Database error during chat request: {ve}")
        return ChatResponse(content="backend.errorDbExecution", error=error_parser(ve))
    except (LlmProviderError, T1DError, RuntimeError) as re_err:
        logger.error(f"NLP error during chat request: {re_err}")
        return ChatResponse(
            content="backend.errorNlpTranslation", error=error_parser(re_err)
        )
    except Exception as e:
        logger.exception("Unexpected error during chat request.")
        return ChatResponse(content="backend.errorUnexpected", error=error_parser(e))


def _generate_sql_mode_sse_frames(
    db_path: str,
    message: str,
    sse_sep: str,
    execute_fn: Callable[[str, str], List[Dict[str, SqlValue]]],
    connect_fn: Callable[..., duckdb.DuckDBPyConnection],
    schema_fn: Callable[[duckdb.DuckDBPyConnection], str],
    error_parser: Callable[[Exception], ApiError],
) -> Iterator[str]:
    """
    Generate SSE frames for direct SQL mode (`model in ('sql', '')`).

    Args:
    ----
        db_path: Validated DuckDB file path.
        message: User message or SQL query.
        sse_sep: Double-newline SSE frame separator.
        execute_fn: Read-only SQL execution callable.
        connect_fn: DuckDB connection factory callable.
        schema_fn: Database schema extraction callable.
        error_parser: Exception to ApiError converter callable.

    Yields:
    ------
        str: Formatted SSE event frames.

    """
    nl = chr(10)
    exec_payload = json.dumps(
        {"event": "token", "token": f"Executing SQL query...{nl}"}
    )
    yield f"data: {exec_payload}{sse_sep}"
    yield f"data: {json.dumps({'event': 'execution_progress', 'status': 'Executing SQL query...'})}{sse_sep}"
    sql_query = message
    provenance, is_synthetic = _resolve_db_provenance_flags(
        db_path, connect_fn=connect_fn
    )
    is_raw_sql = is_likely_sql_statement(message)
    try:
        sql_query = _resolve_sql_mode_query(
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
        yield from _emit_sse_terminal_payload(res_dict, sse_sep)
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
        yield from _emit_sse_terminal_payload(err_dict_payload, sse_sep)


def _finalize_nl_sse_frames(
    db_path: str,
    full_response: str,
    auto_execute: bool,
    sse_sep: str,
    execute_fn: Callable[[str, str], List[Dict[str, SqlValue]]],
    error_parser: Callable[[Exception], ApiError],
    connect_fn: Callable[..., duckdb.DuckDBPyConnection] = duckdb.connect,
    used_fallback: Optional[bool] = None,
) -> Iterator[str]:
    """
    Emit `sql_generated`, optional `execution_progress`, and terminal SSE frames for an NL response.

    Args:
    ----
        db_path: Validated DuckDB file path.
        full_response: Complete assistant markdown text.
        auto_execute: Whether to automatically execute the extracted SQL.
        sse_sep: Double-newline SSE frame separator.
        execute_fn: Read-only SQL execution callable.
        error_parser: Exception to ApiError converter callable.
        connect_fn: DuckDB connection factory callable.
        used_fallback: Optional explicit flag indicating whether rule-based fallback synthesis was used.

    Yields:
    ------
        str: Formatted SSE event frames.

    """
    provenance, is_synthetic = _resolve_db_provenance_flags(
        db_path, connect_fn=connect_fn
    )
    sql_query = _extract_sql_from_response(full_response)
    translation = TranslationResult(
        full_response=full_response,
        sql_query=sql_query,
        used_fallback=(
            used_fallback
            if used_fallback is not None
            else is_fallback_response_text(full_response)
        ),
        generation_mode=(
            "rule_based_fallback"
            if (
                used_fallback
                if used_fallback is not None
                else is_fallback_response_text(full_response)
            )
            else "llm"
        ),
    )
    sql_results: Optional[List[Dict[str, SqlValue]]] = None
    sql_err_dict: Optional[Dict[str, SqlValue]] = None
    if translation.sql_query:
        yield f"data: {json.dumps({'event': 'sql_generated', 'sqlQuery': translation.sql_query})}{sse_sep}"
        if auto_execute:
            yield f"data: {json.dumps({'event': 'execution_progress', 'status': 'Executing SQL query...'})}{sse_sep}"
            try:
                sql_results = execute_fn(db_path, translation.sql_query)
            except (T1DError, duckdb.Error, OSError, RuntimeError, ValueError) as e:
                sql_err_dict = error_parser(e).model_dump()

    nl_res_dict: SseTerminalPayload = {
        "content": translation.full_response,
        "sqlQuery": translation.sql_query,
        "sqlResult": sql_results,
        "error": sql_err_dict,
        "is_synthetic_data": is_synthetic,
        "is_rule_based_sql": translation.used_fallback,
        "generation_mode": translation.generation_mode,
        "data_provenance": provenance,
    }
    yield from _emit_sse_terminal_payload(nl_res_dict, sse_sep)


async def _is_request_disconnected(req: Optional[Request]) -> bool:
    """
    Check whether an optional FastAPI `Request` has disconnected.

    Args:
    ----
        req: Optional FastAPI Request instance.

    Returns:
    -------
        bool: True if `req` is present and disconnected.

    """
    return bool(req is not None and await req.is_disconnected())


def _read_schema_from_db(
    db_path: str,
    connect_fn: Callable[..., duckdb.DuckDBPyConnection],
    schema_fn: Callable[[duckdb.DuckDBPyConnection], str],
) -> str:
    """
    Open a read-only DuckDB connection, extract the table schema text, and release the connection.

    Args:
    ----
        db_path: DuckDB file path.
        connect_fn: DuckDB connection factory callable.
        schema_fn: Database schema extraction callable.

    Returns:
    -------
        str: Extracted schema DDL/summary string.

    """
    with connect_duckdb_read_only_pooled(
        db_path=db_path,
        connect_fn=connect_fn,
        disable_external_access=True,
    ) as conn:
        return schema_fn(conn)


def _validate_sse_request_or_error(
    request: ChatRequest,
    sse_sep: str,
    db_validator: Callable[[Optional[str]], str],
) -> Tuple[str, Optional[str]]:
    """
    Validate `request.message` and `request.db_path`, returning `(db_path, error_sse_frame)`.

    Args:
    ----
        request: Incoming ChatRequest payload.
        sse_sep: Double-newline SSE frame separator.
        db_validator: Database path validator callable.

    Returns:
    -------
        Tuple[str, Optional[str]]: `(validated_db_path, optional_error_frame)`.

    """
    if not request.message.strip():
        payload = json.dumps({"event": "error", "error": "backend.emptyMessage"})
        return "", f"data: {payload}{sse_sep}"

    try:
        return db_validator(request.db_path), None
    except (HTTPException, InvalidDatabasePathError):
        payload = json.dumps({"event": "error", "error": "backend.invalidDatabasePath"})
        return "", f"data: {payload}{sse_sep}"


def _recover_ollama_stream_fallback_token(
    resolved_prov: str,
    stream_exc: Exception,
    schema: str,
    message: str,
    sse_sep: str,
    model_name: str = "ollama",
) -> Tuple[str, str]:
    """
    Recover from a local Ollama crash by synthesizing fallback SQL, or re-raise `stream_exc`.

    Args:
    ----
        resolved_prov: Resolved provider identifier.
        stream_exc: Exception raised during token streaming.
        schema: Extracted database schema string.
        message: User natural-language query.
        sse_sep: Double-newline SSE frame separator.
        model_name: Requested model identifier for diagnostic logging.

    Returns:
    -------
        Tuple[str, str]: `(fallback_markdown_text, sse_token_frame)`.

    Raises:
    ------
        Exception: Re-raises `stream_exc` if not a recoverable local Ollama crash.

    """
    if not _is_recoverable_local_ollama_crash(resolved_prov, stream_exc):
        raise stream_exc
    logger.warning(
        f"Local Ollama streaming for model '{model_name}' failed with {type(stream_exc).__name__} ({stream_exc}); using schema-aware clinical SQL fallback."
    )
    synthesized = synthesize_fallback_clinical_sql(
        schema, message, include_diagnostic_note=True
    )
    tok_payload = json.dumps({"event": "token", "token": synthesized.full_response})
    return synthesized.full_response, f"data: {tok_payload}{sse_sep}"


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
        req: Optional FastAPI Request instance to detect client disconnection.
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
    if await _is_request_disconnected(req):
        logger.info("Client disconnected before stream generation.")
        return

    db_path, err_frame = _validate_sse_request_or_error(request, sse_sep, db_validator)
    if err_frame is not None:
        yield err_frame
        return

    if request.model.strip().lower() in ("sql", "") or is_likely_sql_statement(
        request.message
    ):
        for frame in _generate_sql_mode_sse_frames(
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
        schema = _read_schema_from_db(db_path, connect_fn, schema_fn)
        resolved_prov, _ = resolve_provider_and_model(request.model, request.provider)
        prompt = build_sql_prompt_with_history(
            schema,
            request.message,
            history=request.history,
            compact_schema=(resolved_prov.lower() == "ollama"),
        )
        full_parts: List[str] = []
        used_fallback = False
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
                if await _is_request_disconnected(req):
                    logger.info("Client disconnected during SSE token streaming.")
                    return
                token = await asyncio.to_thread(_next_stream_token_or_eof, token_iter)
                if token == _STREAM_DONE_SENTINEL:
                    break
                full_parts.append(token)
                tok_payload = json.dumps({"event": "token", "token": token})
                yield f"data: {tok_payload}{sse_sep}"
        except Exception as stream_exc:
            fb_resp, tok_frame = _recover_ollama_stream_fallback_token(
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
        for frame in _finalize_nl_sse_frames(
            db_path,
            full_response,
            request.auto_execute,
            sse_sep,
            execute_fn=execute_fn,
            error_parser=error_parser,
            connect_fn=connect_fn,
            used_fallback=used_fallback,
        ):
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


__all__ = [
    "_FALLBACK_DIAGNOSTIC_PREFIX",
    "_STREAM_DONE_SENTINEL",
    "_TABLE_1_FALLBACK_KEYWORDS",
    "_VALID_SHOW_SECOND_TOKENS",
    "GenerationModeLiteral",
    "SseTerminalPayload",
    "TranslationResult",
    "_build_fallback_table_1_sql",
    "_complete_non_streaming_with_fallback",
    "_create_verified_llm_transport",
    "_emit_sse_terminal_payload",
    "_ensure_any_llm_installed_and_ready",
    "_extract_schema_for_translation",
    "_finalize_nl_sse_frames",
    "_find_matching_column",
    "_generate_sql_mode_sse_frames",
    "_is_recoverable_local_ollama_crash",
    "_is_request_disconnected",
    "_next_stream_token_or_eof",
    "_parse_schema_tables_and_columns",
    "_probe_installed_ollama_names",
    "_read_schema_from_db",
    "_recover_ollama_stream_fallback_token",
    "_resolve_and_select_local_ollama_model",
    "_resolve_db_provenance_flags",
    "_resolve_sql_mode_query",
    "_score_table_for_query",
    "_select_best_fallback_table",
    "_validate_sse_request_or_error",
    "build_sql_prompt_with_history",
    "coerce_translation_result",
    "execute_chat_turn",
    "generate_sse_chat_events",
    "is_fallback_response_text",
    "is_likely_sql_statement",
    "stream_llm_completion_tokens",
    "synthesize_fallback_clinical_sql",
    "translate_nl_to_sql",
]
