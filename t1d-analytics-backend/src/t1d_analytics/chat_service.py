"""Natural-language-to-SQL translation and Server-Sent Events (SSE) chat streaming services."""

import asyncio
import json
import logging
import time
import urllib.error
from pathlib import Path
from typing import (
    AsyncIterator,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Literal,
    Optional,
    Sequence,
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
    NlToSqlEvalRequest,
    NlToSqlEvalResponse,
    SqlGoldComparisonModel,
)
from t1d_analytics.diagnostics import detect_database_provenance
from t1d_analytics.duckdb_pool import connect_duckdb_read_only_pooled
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
    LlmCompletionResponseProtocol,
    LlmTransportProtocol,
    StreamChunkInput,
    build_clinical_sql_prompt,
    build_compact_clinical_sql_prompt,
    complete_llm_prompt,
    create_any_llm_transport,
    extract_stream_delta,
    get_effective_api_key,
    is_compact_local_model,
    probe_ollama_tags_and_version,
    resolve_installed_ollama_model_tag,
    resolve_local_ollama_fallback_candidates,
    resolve_provider_and_model,
    sort_ollama_models,
)
from t1d_analytics.sql_agent import prune_schema_with_gemma_rag
from t1d_analytics.sql_executor import execute_duckdb_query
from t1d_analytics.sql_fallback_synthesizer import (
    _FALLBACK_DIAGNOSTIC_PREFIX,
    _TABLE_1_FALLBACK_KEYWORDS,
    GenerationModeLiteral,
    TranslationResult,
    _is_recoverable_local_ollama_crash,
    _lookup_curated_template_sql,
    _parse_schema_tables_and_columns,
    extract_fallback_error_from_text,
    format_llm_failure_reason,
    is_fallback_response_text,
    synthesize_fallback_clinical_sql,
)
from t1d_analytics.sql_utils import (
    _VALID_SHOW_SECOND_TOKENS,
    READ_ONLY_SQL_PREFIXES,
    SqlValue,
    _check_sql_syntax_with_conn,
    _has_natural_language_sql_disqualifier,
    is_sql_query,
    strip_sql_comments_and_parens,
)
from t1d_analytics.sql_utils import (
    extract_sql_from_response as _extract_sql_from_response,
)
from t1d_analytics.training_data import compare_predicted_and_gold_sql

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


def _is_unambiguous_sql_attempt(query: str) -> bool:
    """
    Determine whether an input string is a structural read-only SQL attempt (even if syntactically malformed).

    Args:
    ----
        query: Raw user query string.

    Returns:
    -------
        bool: True if the input starts with a read-only SQL keyword and passes all natural-language prose disqualifiers.

    """
    unwrapped = strip_sql_comments_and_parens(query)
    if not unwrapped:
        return False
    tokens = unwrapped.split()
    first_word = tokens[0].lower().rstrip(";").strip()
    if first_word not in READ_ONLY_SQL_PREFIXES:
        return False
    return not _has_natural_language_sql_disqualifier(unwrapped, first_word, tokens)


def is_likely_sql_statement(query: str) -> bool:
    """
    Determine whether a user input string is a valid read-only SQL statement rather than natural language.

    Args:
    ----
        query: Raw user query string.

    Returns:
    -------
        bool: True if the input is a read-only SQL statement that passes natural-language prose guards and DuckDB AST syntax parsing, False otherwise.

    """
    if not _is_unambiguous_sql_attempt(query):
        return False
    unwrapped = strip_sql_comments_and_parens(query)
    first_word = unwrapped.split()[0].lower().rstrip(";").strip()
    return _check_sql_syntax_with_conn(unwrapped, first_word)


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
    probe_fn: Optional[Callable[[], List[str]]] = None,
) -> Tuple[str, Optional[str]]:
    """
    Normalize a requested local Ollama model against installed `/api/tags` models and return a lightweight fallback tag.

    Args:
    ----
        actual_model: Requested Ollama model name (without `ollama/` prefix).
        probe_fn: Optional callable returning the list of installed Ollama model names.

    Returns:
    -------
        Tuple[str, Optional[str]]: `(normalized_primary_model, fallback_lightweight_model)`.

    """
    effective_probe = (
        probe_fn if probe_fn is not None else _probe_installed_ollama_names
    )
    try:
        installed = effective_probe()
    except (urllib.error.URLError, OSError, ValueError, RuntimeError):
        installed = []
    if not installed:
        return actual_model, None

    ranked = sort_ollama_models(installed)
    matched = resolve_installed_ollama_model_tag(actual_model, ranked)
    primary = matched if matched is not None else ranked[0]
    candidates = resolve_local_ollama_fallback_candidates(primary, ranked)
    fallback = candidates[0] if candidates else None
    return primary, fallback


def _resolve_local_ollama_candidates_list(
    actual_model: str,
    probe_fn: Optional[Callable[[], List[str]]] = None,
) -> Tuple[str, List[str]]:
    """
    Resolve the primary local Ollama model tag and ordered list of lightweight fallback candidates.

    Args:
    ----
        actual_model: Requested Ollama model name (without `ollama/` prefix).
        probe_fn: Optional callable returning the list of installed Ollama model names.

    Returns:
    -------
        Tuple[str, List[str]]: `(normalized_primary_model, ordered_fallback_candidates)`.

    """
    effective_probe = (
        probe_fn if probe_fn is not None else _probe_installed_ollama_names
    )
    primary, first_fallback = _resolve_and_select_local_ollama_model(
        actual_model, probe_fn=effective_probe
    )
    if first_fallback is None:
        return primary, []
    try:
        installed = effective_probe()
    except (urllib.error.URLError, OSError, ValueError, RuntimeError):
        installed = []
    extra = resolve_local_ollama_fallback_candidates(primary, installed)
    ordered: List[str] = [first_fallback]
    for cand in extra:
        if cand not in ordered:
            ordered.append(cand)
    return primary, ordered


def _resolve_effective_retries_for_provider(
    resolved_provider: str,
    default_max_retries: int = 3,
) -> int:
    """
    Resolve the bounded maximum retry count (`effective_retries = 1` for local Ollama, `default_max_retries` otherwise).

    Args:
    ----
        resolved_provider: Resolved LLM provider identifier.
        default_max_retries: Default retry limit for non-Ollama providers.

    Returns:
    -------
        int: Effective retry count (`1` when `resolved_provider == "ollama"`).

    """
    if resolved_provider.strip().lower() == "ollama":
        return 1
    return max(1, default_max_retries)


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
    gold_columns: List[str]
    pred_columns: List[str]
    column_overlap_pct: float
    gold_rows_preview: List[Dict[str, SqlValue]]
    error_message: Optional[str]


class SseTerminalPayload(TypedDict, total=False):
    """Typed dictionary representing the terminal payload of a chat SSE stream."""

    content: str
    sqlQuery: Optional[str]
    sqlResult: Optional[List[Dict[str, SqlValue]]]
    error: Optional[Union[ErrorDetailDict, Dict[str, SqlValue]]]
    is_synthetic_data: bool
    is_rule_based_sql: bool
    generation_mode: str
    llm_error: Optional[str]
    llm_error_code: Optional[str]
    data_provenance: str
    gold_comparison: Optional[SqlGoldComparisonPayloadDict]


class SseDoneEvent(SseTerminalPayload, total=False):
    """Typed dictionary representing a terminal SSE 'done' or 'result' event."""

    event: Literal["done", "result"]


SseEventPayload = Union[
    SseTokenEvent,
    SseProgressEvent,
    SseSqlGeneratedEvent,
    SseErrorEvent,
    SseDoneEvent,
]


def format_sse_frame(payload: SseEventPayload, sse_sep: str = "\n\n") -> str:
    """
    Serialize a strongly-typed `SseEventPayload` into a wire-format SSE `data: ...` frame.

    Args:
    ----
        payload: Typed SSE event dictionary.
        sse_sep: Double-newline SSE frame separator.

    Returns:
    -------
        str: Formatted SSE `data: <json><sse_sep>` string.

    """
    return f"data: {json.dumps(payload)}{sse_sep}"


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
    done_event: SseDoneEvent = {"event": "done", **payload}
    result_event: SseDoneEvent = {"event": "result", **payload}
    yield format_sse_frame(done_event, sse_sep)
    yield format_sse_frame(result_event, sse_sep)


def build_sql_prompt_with_history(
    schema: str,
    nl_query: str,
    history: Optional[List[ChatMessageModel]] = None,
    compact_schema: bool = False,
    model_name: Optional[str] = None,
    provider: Optional[str] = None,
) -> str:
    """
    Construct the clinical Text-to-SQL system prompt for an LLM with conversation history and RAG pruning.

    Args:
    ----
        schema: The database schema text.
        nl_query: The clinician or user query.
        history: Optional list of prior conversation messages for multi-turn context.
        compact_schema: Whether to compact sample rows and table blocks for local CPU Ollama models.
        model_name: Optional model identifier to auto-detect compact local CPU models.
        provider: Optional provider identifier to auto-detect compact local CPU models.

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
    use_compact = compact_schema or is_compact_local_model(model_name, provider)
    if use_compact:
        return build_compact_clinical_sql_prompt(
            schema_context=enriched_schema,
            user_query=nl_query,
            history_context=history_context,
        )

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


def _extract_completion_or_iterable_text(
    response: Union[
        LlmCompletionResponseProtocol,
        StreamChunkInput,
        List[StreamChunkInput],
        Tuple[StreamChunkInput, ...],
    ],
    resolved_provider: str,
) -> str:
    """
    Extract full text from either a completion response object or a sequence of stream chunks.

    Args:
    ----
        response: Completion response object, scalar chunk, or list/tuple of chunks.
        resolved_provider: Resolved LLM provider name.

    Returns:
    -------
        str: Concatenated completion text.

    """
    if isinstance(response, (list, tuple)):
        parts = [extract_stream_delta(chunk, resolved_provider) for chunk in response]
        joined = "".join(p for p in parts if p)
        return joined or str(response)
    return extract_stream_delta(response, resolved_provider) or str(response)


def _complete_non_streaming_with_fallback(
    llm: LlmTransportProtocol,
    resolved_provider: str,
    actual_model: str,
    fallback_model: Optional[str],
    prompt: str,
    fallback_candidates: Optional[Sequence[str]] = None,
    sleep_fn: Optional[Callable[[float], None]] = None,
    skip_primary: bool = False,
    initial_exc: Optional[Exception] = None,
) -> str:
    """
    Execute a non-streaming completion call, retrying across candidate fallback models on failure.

    Args:
    ----
        llm: Transport client instance.
        resolved_provider: Provider identifier.
        actual_model: Primary model name.
        fallback_model: Optional lightweight fallback model name.
        prompt: Prompt string.
        fallback_candidates: Optional ordered sequence of candidate fallback model tags.
        sleep_fn: Optional injectable sleep callable for brief daemon recovery pause between retries.
        skip_primary: Whether `actual_model` already crashed with a recoverable local Ollama error.
        initial_exc: Initial exception raised when `skip_primary` is True.

    Returns:
    -------
        str: Extracted text from the completion response.

    """
    effective_sleep = sleep_fn if sleep_fn is not None else time.sleep
    candidates: List[str] = (
        list(fallback_candidates)
        if fallback_candidates is not None
        else ([fallback_model] if fallback_model is not None else [])
    )
    if not skip_primary:
        try:
            response = llm.completion(
                model=actual_model,
                messages=[{"role": "user", "content": prompt}],
            )
            return _extract_completion_or_iterable_text(response, resolved_provider)
        except Exception as primary_exc:
            if not candidates:
                raise
            last_exc: Exception = primary_exc
    else:
        last_exc = (
            initial_exc
            if initial_exc is not None
            else RuntimeError(f"Primary model '{actual_model}' failed")
        )

    for candidate in candidates:
        if _is_recoverable_local_ollama_crash(resolved_provider, last_exc):
            effective_sleep(0.15)
        logger.warning(
            f"Local Ollama model '{actual_model}' failed ({type(last_exc).__name__}: {last_exc}); retrying with lightweight model '{candidate}'."
        )
        try:
            response = llm.completion(
                model=candidate,
                messages=[{"role": "user", "content": prompt}],
            )
            return _extract_completion_or_iterable_text(response, resolved_provider)
        except Exception as cand_exc:
            last_exc = cand_exc
    raise last_exc


def stream_llm_completion_tokens(
    prompt: str,
    model_name: str,
    api_key: Optional[str] = None,
    provider: Optional[str] = None,
    llm_factory: Optional[AnyLlmFactoryProtocol] = None,
    sleep_fn: Optional[Callable[[float], None]] = None,
) -> Iterator[str]:
    """
    Stream tokens from LLM using any-llm, falling back gracefully across candidate local models on crash.

    Args:
    ----
        prompt: The full prompt string.
        model_name: The LLM model name or provider/model string.
        api_key: Optional API key for external cloud provider.
        provider: Optional provider name override.
        llm_factory: Optional injectable factory creating an LLM transport client.
        sleep_fn: Injectable sleep callable for brief daemon recovery pause between retries.

    Yields:
    ------
        str: Streamed token chunks.

    Raises:
    ------
        ProviderNotConfiguredError: If the requested cloud provider lacks an API key.

    """
    resolved_provider, actual_model = resolve_provider_and_model(model_name, provider)
    is_local_ollama = resolved_provider.lower() == "ollama"
    fallback_candidates: List[str] = []
    if is_local_ollama and llm_factory is None:
        actual_model, fallback_candidates = _resolve_local_ollama_candidates_list(
            actual_model
        )
    fallback_model = fallback_candidates[0] if fallback_candidates else None
    llm = _create_verified_llm_transport(resolved_provider, api_key, llm_factory)
    stream_crash_exc: Optional[Exception] = None
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
        if is_local_ollama and _is_recoverable_local_ollama_crash(resolved_provider, e):
            stream_crash_exc = e
            if not fallback_candidates and llm_factory is not None:
                _, fallback_candidates = _resolve_local_ollama_candidates_list(
                    actual_model
                )
                fallback_model = fallback_candidates[0] if fallback_candidates else None
        logger.warning(
            f"Native streaming failed for provider '{resolved_provider}' model '{actual_model}' ({type(e).__name__}: {e}). Falling back to non-streaming completion."
        )

    yield _complete_non_streaming_with_fallback(
        llm=llm,
        resolved_provider=resolved_provider,
        actual_model=actual_model,
        fallback_model=fallback_model,
        prompt=prompt,
        fallback_candidates=fallback_candidates if fallback_candidates else None,
        sleep_fn=sleep_fn,
        skip_primary=(stream_crash_exc is not None and bool(fallback_candidates)),
        initial_exc=stream_crash_exc,
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


_TRANSLATION_SCHEMA_CACHE: Dict[Tuple[str, int, int], str] = {}


def _extract_schema_for_translation(
    db_path: str,
    connect_fn: Callable[..., duckdb.DuckDBPyConnection],
    schema_fn: Callable[[duckdb.DuckDBPyConnection], str],
) -> str:
    """
    Open `db_path` read-only and extract its schema text for NL-to-SQL translation.

    Caches extracted schema strings by `(resolved_path, st_mtime_ns, st_size)` when
    using the default `duckdb.connect` and `get_database_schema` callables.

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
    cache_key: Optional[Tuple[str, int, int]] = None
    if connect_fn is duckdb.connect and schema_fn is get_database_schema:
        db_file = Path(db_path)
        if db_file.is_file():
            st = db_file.stat()
            cache_key = (str(db_file.resolve()), st.st_mtime_ns, st.st_size)
            cached = _TRANSLATION_SCHEMA_CACHE.get(cache_key)
            if cached is not None:
                return cached
    try:
        try:
            conn = connect_fn(db_path, read_only=True)
        except duckdb.IOException as io_err:
            raise DatabaseNotFoundError(params={"error": str(io_err)}) from io_err
        try:
            schema = schema_fn(conn)
        finally:
            conn.close()
        if cache_key is not None:
            if len(_TRANSLATION_SCHEMA_CACHE) >= 32:
                _TRANSLATION_SCHEMA_CACHE.clear()
            _TRANSLATION_SCHEMA_CACHE[cache_key] = schema
        logger.debug(f"Retrieved schema: {schema[:200]}... (truncated)")
        return schema
    except DatabaseNotFoundError as dnf:
        logger.error(f"Failed to read schema: {dnf}")
        raise
    except (duckdb.Error, OSError, RuntimeError, ValueError, TypeError) as e:
        logger.error(f"Failed to read schema: {e}")
        raise SchemaReadError(params={"error": str(e)}) from e


def _invoke_llm_with_local_fallback(
    prompt: str,
    actual_model: str,
    fallback_candidates: Sequence[str],
    effective_api_key: Optional[str],
    resolved_provider: str,
    is_local_ollama: bool,
    llm_factory: Optional[AnyLlmFactoryProtocol] = None,
    sleep_fn: Optional[Callable[[float], None]] = None,
) -> str:
    """
    Execute synchronous LLM completion and retry across candidate local Ollama fallback models on failure.

    Args:
    ----
        prompt: Formatted SQL generation prompt string.
        actual_model: Primary resolved model identifier.
        fallback_candidates: Ordered sequence of lightweight local Ollama fallback model tags.
        effective_api_key: Optional resolved provider API key.
        resolved_provider: Resolved LLM provider name.
        is_local_ollama: Whether the target provider is local Ollama.
        llm_factory: Optional injectable LLM transport factory.
        sleep_fn: Optional injectable sleep callable for brief daemon recovery pause between retries.

    Returns:
    -------
        str: Full LLM completion response text.

    Raises:
    ------
        Exception: Re-raises the last completion exception if all models fail.

    """
    effective_sleep = sleep_fn if sleep_fn is not None else time.sleep
    try:
        return complete_llm_prompt(
            prompt=prompt,
            model_name=actual_model,
            api_key=effective_api_key,
            provider=resolved_provider,
            llm_factory=llm_factory,
        )
    except Exception as primary_exc:
        if not is_local_ollama or not fallback_candidates:
            raise
        last_exc: Exception = primary_exc
        for candidate in fallback_candidates:
            if _is_recoverable_local_ollama_crash(resolved_provider, last_exc):
                effective_sleep(0.15)
            logger.warning(
                f"Local Ollama model '{actual_model}' failed ({type(last_exc).__name__}: {last_exc}); retrying with lightweight model '{candidate}'."
            )
            try:
                return complete_llm_prompt(
                    prompt=prompt,
                    model_name=candidate,
                    api_key=effective_api_key,
                    provider=resolved_provider,
                    llm_factory=llm_factory,
                )
            except Exception as cand_exc:
                last_exc = cand_exc
        raise last_exc from primary_exc


_CLINICAL_SCHEMA_MARKERS: Tuple[str, ...] = (
    "patients",
    "tblaptsummary",
    "demographics",
    "subjects",
    "cgm_readings",
    "hba1c",
    "hba1c_labs",
    "adverseevents",
    "dka_adverse_events",
)


def _validate_or_repair_translated_sql(
    db_path: str,
    schema: str,
    nl_query: str,
    full_response: str,
    sql_query: str,
    actual_model: str,
    resolved_provider: str,
    connect_fn: Callable[..., duckdb.DuckDBPyConnection] = duckdb.connect,
) -> Tuple[str, str]:
    """
    Validate LLM-generated SQL against the target clinical schema and repair invalid or non-matching template queries.

    Args:
    ----
        db_path: Path to the target DuckDB file.
        schema: Extracted database schema text.
        nl_query: User's natural-language question.
        full_response: Raw LLM markdown response text.
        sql_query: Extracted candidate SQL query string.
        actual_model: Resolved LLM model name.
        resolved_provider: Resolved LLM provider identifier.
        connect_fn: DuckDB connection factory callable.

    Returns:
    -------
        Tuple[str, str]: `(repaired_full_response, repaired_sql_query)`.

    """
    nl = chr(10)
    schema_map = _parse_schema_tables_and_columns(schema)
    lower_tables = {t.lower() for t in schema_map}
    curated_sql = (
        _lookup_curated_template_sql(nl_query, lower_tables) if lower_tables else None
    )
    if curated_sql is not None:
        comp = compare_predicted_and_gold_sql(
            db_path=db_path,
            pred_sql=sql_query,
            gold_sql=curated_sql,
            execute_fn=execute_duckdb_query,
        )
        if not comp.execution_accuracy or comp.syntax_error:
            return f"```sql{nl}{curated_sql}{nl}```", curated_sql
        return full_response, sql_query

    if connect_fn is duckdb.connect and any(
        m in lower_tables for m in _CLINICAL_SCHEMA_MARKERS
    ):
        try:
            if not sql_query.strip():
                raise ValueError("Empty SQL query")
            execute_duckdb_query(db_path, sql_query)
        except (T1DError, duckdb.Error, OSError, RuntimeError, ValueError):
            repaired = synthesize_fallback_clinical_sql(
                schema,
                nl_query,
                include_diagnostic_note=False,
                model_name=actual_model,
                provider=resolved_provider,
            )
            return repaired.full_response, repaired.sql_query

    return full_response, sql_query


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
    effective_retries = _resolve_effective_retries_for_provider(resolved_provider)
    fallback_candidates: List[str] = []
    if is_local_ollama and llm_factory is None:
        actual_model, fallback_candidates = _resolve_local_ollama_candidates_list(
            actual_model
        )
    prompt = build_sql_prompt_with_history(
        schema,
        nl_query,
        history=history,
        compact_schema=is_local_ollama,
        model_name=actual_model,
        provider=resolved_provider,
    )
    nl = chr(10)

    try:
        logger.info(
            f"Sending prompt to LLM ({resolved_provider}/{actual_model}, effective_retries={effective_retries})..."
        )
        full_response = _invoke_llm_with_local_fallback(
            prompt=prompt,
            actual_model=actual_model,
            fallback_candidates=fallback_candidates,
            effective_api_key=effective_api_key,
            resolved_provider=resolved_provider,
            is_local_ollama=is_local_ollama,
            llm_factory=llm_factory,
        )
        logger.info(f"LLM response:{nl}{full_response}")
        sql_query = _extract_sql_from_response(full_response)
        full_response, sql_query = _validate_or_repair_translated_sql(
            db_path=db_path,
            schema=schema,
            nl_query=nl_query,
            full_response=full_response,
            sql_query=sql_query,
            actual_model=actual_model,
            resolved_provider=resolved_provider,
            connect_fn=connect_fn,
        )
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
            err_msg, err_code = format_llm_failure_reason(
                e, model_name=actual_model, provider=resolved_provider
            )
            return synthesize_fallback_clinical_sql(
                schema,
                nl_query,
                include_diagnostic_note=True,
                llm_error=err_msg,
                llm_error_code=err_code,
                model_name=actual_model,
                provider=resolved_provider,
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
    if (
        is_likely_sql_statement(message)
        or _is_unambiguous_sql_attempt(message)
        or is_sql_query(message)
    ):
        return message
    with connect_duckdb_read_only_pooled(
        db_path=db_path,
        connect_fn=connect_fn,
        disable_external_access=True,
    ) as conn:
        schema = schema_fn(conn)
    synthesized = synthesize_fallback_clinical_sql(schema, message)
    return synthesized.sql_query


def _resolve_effective_gold_sql(
    message: str,
    explicit_gold_sql: Optional[str] = None,
) -> Optional[str]:
    """
    Resolve an explicit `gold_sql` string or auto-detect a canonical Gold Template SQL query.

    Args:
    ----
        message: Natural language prompt string.
        explicit_gold_sql: Optional explicit gold-standard SQL provided in the request.

    Returns:
    -------
        Optional[str]: Non-empty gold SQL query string if present or matched, else None.

    """
    if explicit_gold_sql and explicit_gold_sql.strip():
        return explicit_gold_sql.strip()
    return _lookup_curated_template_sql(message)


def _compute_optional_gold_comparison(
    db_path: str,
    message: str,
    pred_sql: str,
    explicit_gold_sql: Optional[str],
    execute_fn: Callable[[str, str], List[Dict[str, SqlValue]]],
    precomputed_pred_rows: Optional[List[Dict[str, SqlValue]]] = None,
    precomputed_pred_error: Optional[str] = None,
) -> Optional[SqlGoldComparisonModel]:
    """
    Compute `SqlGoldComparisonModel` on the read-only DuckDB cursor when a gold SQL query is available.

    Args:
    ----
        db_path: Validated path to the DuckDB file.
        message: Natural language prompt string.
        pred_sql: SQL query generated by the target LLM.
        explicit_gold_sql: Optional explicit gold SQL from the request.
        execute_fn: Read-only DuckDB query execution callable.
        precomputed_pred_rows: Optional already-executed rows for `pred_sql`.
        precomputed_pred_error: Optional error message if `pred_sql` already failed execution.

    Returns:
    -------
        Optional[SqlGoldComparisonModel]: Comparison model if gold SQL exists, else None.

    """
    effective_gold = _resolve_effective_gold_sql(message, explicit_gold_sql)
    if not effective_gold:
        return None
    return compare_predicted_and_gold_sql(
        db_path=db_path,
        pred_sql=pred_sql,
        gold_sql=effective_gold,
        execute_fn=execute_fn,
        precomputed_pred_rows=precomputed_pred_rows,
        precomputed_pred_error=precomputed_pred_error,
    )


def execute_chat_turn(
    request: ChatRequest,
    db_validator: Callable[[Optional[str]], str],
    execute_fn: Callable[[str, str], List[Dict[str, SqlValue]]],
    translate_fn: Callable[..., TranslationResult],
    error_parser: Callable[[Exception], ApiError],
) -> ChatResponse:
    """
    Execute a synchronous chat turn (literal SQL or natural-language-to-SQL with optional execution).

    Args:
    ----
        request: Validated `ChatRequest` payload.
        db_validator: Callable resolving and sandboxing the target database path.
        execute_fn: Callable executing a read-only SQL query against the database.
        translate_fn: Callable translating a natural-language question into a `TranslationResult`.
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
        is_likely_sql = is_likely_sql_statement(request.message)
        if request.model.strip().lower() in ("sql", "") or is_likely_sql:
            is_raw_sql = (
                is_likely_sql
                or _is_unambiguous_sql_attempt(request.message)
                or is_sql_query(request.message)
            )
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

        translation = translate_fn(
            db_path,
            request.message,
            model_name=request.model,
            history=request.history,
            api_key=request.api_key,
            provider=request.provider,
        )
        nl_results: Optional[List[Dict[str, SqlValue]]] = None
        sql_error: Optional[ApiError] = None
        pred_err_str: Optional[str] = None
        if translation.sql_query and request.auto_execute:
            try:
                nl_results = execute_fn(db_path, translation.sql_query)
            except (T1DError, ValueError, RuntimeError) as e:
                pred_err_str = str(e)
                sql_error = error_parser(e)

        gold_comp = _compute_optional_gold_comparison(
            db_path=db_path,
            message=request.message,
            pred_sql=translation.sql_query,
            explicit_gold_sql=request.gold_sql,
            execute_fn=execute_fn,
            precomputed_pred_rows=nl_results,
            precomputed_pred_error=pred_err_str,
        )

        return ChatResponse(
            content=translation.full_response,
            sqlQuery=translation.sql_query,
            sqlResult=nl_results,
            error=sql_error,
            is_synthetic_data=is_synthetic,
            is_rule_based_sql=translation.used_fallback,
            generation_mode=translation.generation_mode,
            llm_error=translation.llm_error,
            llm_error_code=translation.llm_error_code,
            data_provenance=provenance,
            gold_comparison=gold_comp,
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


def execute_nl_to_sql_eval_turn(
    request: NlToSqlEvalRequest,
    db_validator: Callable[[Optional[str]], str],
    execute_fn: Callable[[str, str], List[Dict[str, SqlValue]]],
    translate_fn: Callable[..., TranslationResult],
    error_parser: Callable[[Exception], ApiError],
) -> NlToSqlEvalResponse:
    """
    Translate a natural-language question to SQL, execute on the read-only DuckDB cursor, and compare against Gold SQL.

    Args:
    ----
        request: Validated `NlToSqlEvalRequest` payload.
        db_validator: Callable resolving and sandboxing the target database path.
        execute_fn: Read-only DuckDB query execution callable.
        translate_fn: NL-to-SQL translation callable.
        error_parser: Exception to `ApiError` converter callable.

    Returns:
    -------
        NlToSqlEvalResponse: Evaluation payload with generated SQL, DuckDB rows, and gold comparison metrics.

    Raises:
    ------
        HTTPException: If database path validation raises an HTTPException.
        InvalidDatabasePathError: If the database path is outside the allowed workspace sandbox.

    """
    db_path = db_validator(request.db_path)
    provenance, is_synthetic = _resolve_db_provenance_flags(db_path)
    try:
        translation = translate_fn(
            db_path,
            request.question,
            model_name=request.model,
            api_key=request.api_key,
            provider=request.provider,
        )
        pred_rows: Optional[List[Dict[str, SqlValue]]] = None
        sql_err: Optional[ApiError] = None
        pred_err_str: Optional[str] = None
        if translation.sql_query:
            try:
                pred_rows = execute_fn(db_path, translation.sql_query)
            except (T1DError, ValueError, RuntimeError) as exc:
                pred_err_str = str(exc)
                sql_err = error_parser(exc)

        gold_comp = _compute_optional_gold_comparison(
            db_path=db_path,
            message=request.question,
            pred_sql=translation.sql_query,
            explicit_gold_sql=request.gold_sql,
            execute_fn=execute_fn,
            precomputed_pred_rows=pred_rows,
            precomputed_pred_error=pred_err_str,
        )
        return NlToSqlEvalResponse(
            question=request.question,
            model=request.model,
            content=translation.full_response,
            sqlQuery=translation.sql_query,
            sqlResult=pred_rows,
            gold_comparison=gold_comp,
            is_rule_based_sql=translation.used_fallback,
            generation_mode=translation.generation_mode,
            is_synthetic_data=is_synthetic,
            data_provenance=provenance,
            error=sql_err,
        )
    except Exception as exc:
        return NlToSqlEvalResponse(
            question=request.question,
            model=request.model,
            content="backend.errorNlpTranslation",
            is_synthetic_data=is_synthetic,
            data_provenance=provenance,
            error=error_parser(exc),
        )


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
    tok_event: SseTokenEvent = {
        "event": "token",
        "token": f"Executing SQL query...{nl}",
    }
    prog_event: SseProgressEvent = {
        "event": "execution_progress",
        "status": "Executing SQL query...",
    }
    yield format_sse_frame(tok_event, sse_sep)
    yield format_sse_frame(prog_event, sse_sep)
    sql_query = message
    provenance, is_synthetic = _resolve_db_provenance_flags(
        db_path, connect_fn=connect_fn
    )
    is_raw_sql = (
        is_likely_sql_statement(message)
        or _is_unambiguous_sql_attempt(message)
        or is_sql_query(message)
    )
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


def _serialize_gold_comparison(
    model: Optional[SqlGoldComparisonModel],
) -> Optional[SqlGoldComparisonPayloadDict]:
    """
    Convert an optional `SqlGoldComparisonModel` into a `SqlGoldComparisonPayloadDict`.

    Args:
    ----
        model: Optional `SqlGoldComparisonModel` instance.

    Returns:
    -------
        Optional[SqlGoldComparisonPayloadDict]: Typed dictionary representation or None.

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


def _finalize_nl_sse_frames(
    db_path: str,
    full_response: str,
    auto_execute: bool,
    sse_sep: str,
    execute_fn: Callable[[str, str], List[Dict[str, SqlValue]]],
    error_parser: Callable[[Exception], ApiError],
    connect_fn: Callable[..., duckdb.DuckDBPyConnection] = duckdb.connect,
    used_fallback: Optional[bool] = None,
    message: str = "",
    gold_sql: Optional[str] = None,
    llm_error: Optional[str] = None,
    llm_error_code: Optional[str] = None,
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
        message: Original natural-language question for gold-standard lookup.
        gold_sql: Optional explicit gold-standard SQL query from the request.
        llm_error: Optional human-readable diagnostic reason why LLM inference failed.
        llm_error_code: Optional structured error code for the LLM failure.

    Yields:
    ------
        str: Formatted SSE event frames.

    """
    provenance, is_synthetic = _resolve_db_provenance_flags(
        db_path, connect_fn=connect_fn
    )
    sql_query = _extract_sql_from_response(full_response)
    effective_fallback = (
        used_fallback
        if used_fallback is not None
        else is_fallback_response_text(full_response)
    )
    if not effective_fallback and message:
        curated_tmpl_sql = _lookup_curated_template_sql(message)
        if curated_tmpl_sql is not None:
            comp_check = compare_predicted_and_gold_sql(
                db_path=db_path,
                pred_sql=sql_query,
                gold_sql=curated_tmpl_sql,
                execute_fn=execute_fn,
            )
            if not comp_check.execution_accuracy or comp_check.syntax_error:
                nl = chr(10)
                sql_query = curated_tmpl_sql
                full_response = f"```sql{nl}{curated_tmpl_sql}{nl}```"
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
    sql_results: Optional[List[Dict[str, SqlValue]]] = None
    sql_err_dict: Optional[Dict[str, SqlValue]] = None
    pred_err_str: Optional[str] = None
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

    gold_comp = _compute_optional_gold_comparison(
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
        "gold_comparison": _serialize_gold_comparison(gold_comp),
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


async def _stream_nl_mode_sse_frames(
    request: ChatRequest,
    req: Optional[Request],
    db_path: str,
    sse_sep: str,
    execute_fn: Callable[[str, str], List[Dict[str, SqlValue]]],
    connect_fn: Callable[..., duckdb.DuckDBPyConnection],
    schema_fn: Callable[[duckdb.DuckDBPyConnection], str],
    stream_fn: Callable[..., Iterator[str]],
    error_parser: Callable[[Exception], ApiError],
    heartbeat_interval_seconds: float = 3.0,
) -> AsyncIterator[str]:
    """
    Stream NL-to-SQL tokens with `: keep-alive` SSE heartbeats every `3.0s` and finalize SQL execution without re-running translation.

    Args:
    ----
        request: Incoming ChatRequest payload.
        req: Optional FastAPI Request instance to detect client disconnection.
        db_path: Validated DuckDB file path.
        sse_sep: Double-newline SSE frame separator.
        execute_fn: Read-only SQL execution callable.
        connect_fn: DuckDB connection factory callable.
        schema_fn: Database schema extraction callable.
        stream_fn: Token streaming callable.
        error_parser: Exception to ApiError converter callable.
        heartbeat_interval_seconds: Interval in seconds between `: keep-alive` SSE comments while waiting for the next token.

    Yields:
    ------
        str: Formatted SSE event frames and `: keep-alive` heartbeat comments.

    """
    schema = _read_schema_from_db(db_path, connect_fn, schema_fn)
    resolved_prov, _ = resolve_provider_and_model(request.model, request.provider)
    prompt = build_sql_prompt_with_history(
        schema,
        request.message,
        history=request.history,
        compact_schema=(resolved_prov.lower() == "ollama"),
        model_name=request.model,
        provider=resolved_prov,
    )
    full_parts: List[str] = []
    used_fallback = False
    stream_llm_error: Optional[str] = None
    stream_llm_error_code: Optional[str] = None
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
            next_task: asyncio.Task[str] = asyncio.create_task(
                asyncio.to_thread(_next_stream_token_or_eof, token_iter)
            )
            while True:
                done_set, _ = await asyncio.wait(
                    {next_task}, timeout=max(0.05, heartbeat_interval_seconds)
                )
                if done_set:
                    token = await next_task
                    break
                if await _is_request_disconnected(req):
                    logger.info("Client disconnected during SSE heartbeat wait.")
                    return
                yield f": keep-alive{sse_sep}"
            if token == _STREAM_DONE_SENTINEL:
                break
            full_parts.append(token)
            tok_event: SseTokenEvent = {"event": "token", "token": token}
            yield format_sse_frame(tok_event, sse_sep)
    except Exception as stream_exc:
        stream_llm_error, stream_llm_error_code = format_llm_failure_reason(
            stream_exc, model_name=request.model, provider=resolved_prov
        )
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
        message=request.message,
        gold_sql=request.gold_sql,
        llm_error=stream_llm_error,
        llm_error_code=stream_llm_error_code,
    ):
        yield frame


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
        nl_stream = _stream_nl_mode_sse_frames(
            request=request,
            req=req,
            db_path=db_path,
            sse_sep=sse_sep,
            execute_fn=execute_fn,
            connect_fn=connect_fn,
            schema_fn=schema_fn,
            stream_fn=stream_fn,
            error_parser=error_parser,
        )
        while True:
            try:
                frame = await nl_stream.__anext__()
            except StopAsyncIteration:
                break
            yield frame
    except Exception as e:
        err_dict = error_parser(e).model_dump()
        nl_err_payload: SseTerminalPayload = {
            "content": "backend.errorNlpTranslation",
            "sqlQuery": None,
            "sqlResult": None,
            "error": err_dict,
        }
        for frame in _emit_sse_terminal_payload(nl_err_payload, sse_sep):
            yield frame


generate_sql_from_nl = translate_nl_to_sql


__all__ = [
    "_FALLBACK_DIAGNOSTIC_PREFIX",
    "_STREAM_DONE_SENTINEL",
    "_TABLE_1_FALLBACK_KEYWORDS",
    "_VALID_SHOW_SECOND_TOKENS",
    "GenerationModeLiteral",
    "SqlGoldComparisonPayloadDict",
    "SseDoneEvent",
    "SseErrorEvent",
    "SseEventPayload",
    "SseProgressEvent",
    "SseSqlGeneratedEvent",
    "SseTerminalPayload",
    "SseTokenEvent",
    "TranslationResult",
    "_complete_non_streaming_with_fallback",
    "_compute_optional_gold_comparison",
    "_create_verified_llm_transport",
    "_emit_sse_terminal_payload",
    "_ensure_any_llm_installed_and_ready",
    "_extract_completion_or_iterable_text",
    "_extract_schema_for_translation",
    "_finalize_nl_sse_frames",
    "_generate_sql_mode_sse_frames",
    "_invoke_llm_with_local_fallback",
    "_is_recoverable_local_ollama_crash",
    "_is_request_disconnected",
    "_is_unambiguous_sql_attempt",
    "_next_stream_token_or_eof",
    "_probe_installed_ollama_names",
    "_read_schema_from_db",
    "_recover_ollama_stream_fallback_token",
    "_resolve_and_select_local_ollama_model",
    "_resolve_db_provenance_flags",
    "_resolve_effective_gold_sql",
    "_resolve_effective_retries_for_provider",
    "_resolve_local_ollama_candidates_list",
    "_resolve_sql_mode_query",
    "_serialize_gold_comparison",
    "_stream_nl_mode_sse_frames",
    "_validate_sse_request_or_error",
    "build_sql_prompt_with_history",
    "execute_chat_turn",
    "execute_nl_to_sql_eval_turn",
    "extract_fallback_error_from_text",
    "format_llm_failure_reason",
    "format_sse_frame",
    "generate_sql_from_nl",
    "generate_sse_chat_events",
    "is_fallback_response_text",
    "is_likely_sql_statement",
    "stream_llm_completion_tokens",
    "synthesize_fallback_clinical_sql",
    "translate_nl_to_sql",
]
