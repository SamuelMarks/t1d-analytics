"""Thread-safe LRU schema caching, prompt construction, LLM fallback invocation, and Gold SQL comparison."""

import logging
import threading
import time
import urllib.error
from collections import OrderedDict
from collections.abc import Callable, Iterable, Iterator, Sequence
from pathlib import Path

import duckdb

from t1d_analytics.analytics import get_database_schema
from t1d_analytics.api_schemas import ChatMessageModel, SqlGoldComparisonModel
from t1d_analytics.errors import (
    DatabaseNotFoundError,
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
from t1d_analytics.sql_agent import (
    execute_agentic_nl_to_sql,
    prune_schema_with_gemma_rag,
)
from t1d_analytics.sql_executor import execute_duckdb_query
from t1d_analytics.sql_fallback_synthesizer import (
    TranslationResult,
    format_llm_failure_reason,
    is_recoverable_local_ollama_crash,
    lookup_curated_template_sql,
    parse_schema_tables_and_columns,
    synthesize_fallback_clinical_sql,
)
from t1d_analytics.sql_utils import SqlValue, extract_sql_from_response
from t1d_analytics.training_data import compare_predicted_and_gold_sql

logger = logging.getLogger(__name__)

NL = """
"""
MAX_TRANSLATION_SCHEMA_CACHE_ENTRIES = 32
_TRANSLATION_SCHEMA_CACHE_LOCK = threading.RLock()
TRANSLATION_SCHEMA_CACHE: OrderedDict[tuple[str, int, int], str] = OrderedDict()

CLINICAL_SCHEMA_MARKERS: tuple[str, ...] = (
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


def probe_installed_ollama_names() -> list[str]:
    """
    Query the local Ollama `/api/tags` endpoint and return installed model names.

    Returns:
        list[str]: Installed Ollama model names.

    """
    infos, _ = probe_ollama_tags_and_version(timeout=1.5, include_version=False)
    return [info.name for info in infos]


def resolve_and_select_local_ollama_model(
    actual_model: str,
    probe_fn: Callable[[], list[str]] | None = None,
) -> tuple[str, str | None]:
    """
    Normalize a requested local Ollama model against installed `/api/tags` models and return a lightweight fallback tag.

    Args:
        actual_model: Requested Ollama model name (without `ollama/` prefix).
        probe_fn: Optional callable returning the list of installed Ollama model names.

    Returns:
        tuple[str, str | None]: `(normalized_primary_model, fallback_lightweight_model)`.

    """
    effective_probe = probe_fn if probe_fn is not None else probe_installed_ollama_names
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


def resolve_local_ollama_candidates_list(
    actual_model: str,
    probe_fn: Callable[[], list[str]] | None = None,
) -> tuple[str, list[str]]:
    """
    Resolve the primary local Ollama model tag and ordered list of lightweight fallback candidates.

    Args:
        actual_model: Requested Ollama model name (without `ollama/` prefix).
        probe_fn: Optional callable returning the list of installed Ollama model names.

    Returns:
        tuple[str, list[str]]: `(normalized_primary_model, ordered_fallback_candidates)`.

    """
    effective_probe = probe_fn if probe_fn is not None else probe_installed_ollama_names
    primary, first_fallback = resolve_and_select_local_ollama_model(
        actual_model, probe_fn=effective_probe
    )
    if first_fallback is None:
        return primary, []
    try:
        installed = effective_probe()
    except (urllib.error.URLError, OSError, ValueError, RuntimeError):
        installed = []
    extra = resolve_local_ollama_fallback_candidates(primary, installed)
    ordered: list[str] = [first_fallback]
    for cand in extra:
        if cand not in ordered:
            ordered.append(cand)
    return primary, ordered


def resolve_effective_retries_for_provider(
    resolved_provider: str,
    default_max_retries: int = 3,
) -> int:
    """
    Resolve bounded maximum retry count (`1` for local Ollama, `default_max_retries` otherwise).

    Args:
        resolved_provider: Resolved LLM provider identifier.
        default_max_retries: Default retry limit for non-Ollama providers.

    Returns:
        int: Effective retry count.

    """
    if resolved_provider.strip().lower() == "ollama":
        return 1
    return max(1, default_max_retries)


def build_sql_prompt_with_history(
    schema: str,
    nl_query: str,
    history: list[ChatMessageModel] | None = None,
    compact_schema: bool = False,
    model_name: str | None = None,
    provider: str | None = None,
) -> str:
    """
    Construct the clinical Text-to-SQL system prompt for an LLM with conversation history and RAG pruning.

    Args:
        schema: The database schema text.
        nl_query: The clinician or user query.
        history: Optional list of prior conversation messages for multi-turn context.
        compact_schema: Whether to compact sample rows and table blocks for local CPU Ollama models.
        model_name: Optional model identifier to auto-detect compact local CPU models.
        provider: Optional provider identifier to auto-detect compact local CPU models.

    Returns:
        str: Formatted system and user prompt.

    """
    history_context = ""
    if history:
        turns: list[str] = []
        for msg in history[-6:]:
            role = "User" if msg.role == "user" else "Assistant"
            content = msg.content.strip()
            sql_info = f" (SQL: {msg.sql_query})" if msg.sql_query else ""
            turns.append(f"{role}: {content}{sql_info}")
        history_context = f"Prior Conversation History:{NL}{NL.join(turns)}{NL}{NL}"

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


def create_verified_llm_transport(
    resolved_provider: str,
    api_key: str | None,
    llm_factory: AnyLlmFactoryProtocol | None,
) -> LlmTransportProtocol:
    """
    Validate cloud provider API key readiness and instantiate an `LlmTransportProtocol`.

    Args:
        resolved_provider: Resolved LLM provider identifier.
        api_key: Optional explicit API key.
        llm_factory: Optional transport factory override.

    Returns:
        LlmTransportProtocol: Instantiated LLM transport client.

    Raises:
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


def extract_completion_or_iterable_text(
    response: LlmCompletionResponseProtocol
    | StreamChunkInput
    | list[StreamChunkInput]
    | tuple[StreamChunkInput, ...],
    resolved_provider: str,
) -> str:
    """
    Extract full text from either a completion response object or a sequence of stream chunks.

    Args:
        response: Completion response object, scalar chunk, or list/tuple of chunks.
        resolved_provider: Resolved LLM provider name.

    Returns:
        str: Concatenated completion text.

    """
    if isinstance(response, (list, tuple)):
        parts = [extract_stream_delta(chunk, resolved_provider) for chunk in response]
        joined = "".join(p for p in parts if p)
        return joined or str(response)
    return extract_stream_delta(response, resolved_provider) or str(response)


def complete_non_streaming_with_fallback(
    llm: LlmTransportProtocol,
    resolved_provider: str,
    actual_model: str,
    fallback_model: str | None,
    prompt: str,
    fallback_candidates: Sequence[str] | None = None,
    sleep_fn: Callable[[float], None] | None = None,
    skip_primary: bool = False,
    initial_exc: Exception | None = None,
) -> str:
    """
    Execute a non-streaming completion call, retrying across candidate fallback models on failure.

    Args:
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
        str: Extracted text from the completion response.

    """
    effective_sleep = sleep_fn if sleep_fn is not None else time.sleep
    candidates: list[str] = (
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
            return extract_completion_or_iterable_text(response, resolved_provider)
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
        if is_recoverable_local_ollama_crash(resolved_provider, last_exc):
            effective_sleep(0.15)
        logger.warning(
            f"Local Ollama model '{actual_model}' failed ({type(last_exc).__name__}: {last_exc}); retrying with lightweight model '{candidate}'."
        )
        try:
            response = llm.completion(
                model=candidate,
                messages=[{"role": "user", "content": prompt}],
            )
            return extract_completion_or_iterable_text(response, resolved_provider)
        except Exception as cand_exc:
            last_exc = cand_exc
    raise last_exc


def stream_llm_completion_tokens(
    prompt: str,
    model_name: str,
    api_key: str | None = None,
    provider: str | None = None,
    llm_factory: AnyLlmFactoryProtocol | None = None,
    sleep_fn: Callable[[float], None] | None = None,
) -> Iterator[str]:
    """
    Stream tokens from LLM using any-llm, falling back gracefully across candidate local models on crash.

    Args:
        prompt: The full prompt string.
        model_name: The LLM model name or provider/model string.
        api_key: Optional API key for external cloud provider.
        provider: Optional provider name override.
        llm_factory: Optional injectable factory creating an LLM transport client.
        sleep_fn: Injectable sleep callable for brief daemon recovery pause between retries.

    Yields:
        str: Streamed token chunks.

    Raises:
        ProviderNotConfiguredError: If the requested cloud provider lacks an API key.

    """
    resolved_provider, actual_model = resolve_provider_and_model(model_name, provider)
    is_local_ollama = resolved_provider.lower() == "ollama"
    fallback_candidates: list[str] = []
    if is_local_ollama and llm_factory is None:
        actual_model, fallback_candidates = resolve_local_ollama_candidates_list(
            actual_model
        )
    fallback_model = fallback_candidates[0] if fallback_candidates else None
    llm = create_verified_llm_transport(resolved_provider, api_key, llm_factory)
    stream_crash_exc: Exception | None = None
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
        if is_local_ollama and is_recoverable_local_ollama_crash(resolved_provider, e):
            stream_crash_exc = e
            if not fallback_candidates and llm_factory is not None:
                _, fallback_candidates = resolve_local_ollama_candidates_list(
                    actual_model
                )
                fallback_model = fallback_candidates[0] if fallback_candidates else None
        logger.warning(
            f"Native streaming failed for provider '{resolved_provider}' model '{actual_model}' ({type(e).__name__}: {e}). Falling back to non-streaming completion."
        )

    yield complete_non_streaming_with_fallback(
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


def ensure_any_llm_installed_and_ready(
    resolved_provider: str,
    api_key: str | None,
    llm_factory: AnyLlmFactoryProtocol | None,
) -> str | None:
    """
    Verify `any-llm-sdk` availability and API key configuration, returning the effective API key.

    Args:
        resolved_provider: Resolved LLM provider identifier.
        api_key: Optional explicit API key.
        llm_factory: Optional injectable LLM factory.

    Returns:
        str | None: Resolved API key or None.

    Raises:
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


def extract_schema_for_translation(
    db_path: str,
    connect_fn: Callable[..., duckdb.DuckDBPyConnection],
    schema_fn: Callable[[duckdb.DuckDBPyConnection], str],
    max_cache_entries: int = MAX_TRANSLATION_SCHEMA_CACHE_ENTRIES,
) -> str:
    """
    Open `db_path` read-only and extract its schema text with thread-safe `RLock` + `OrderedDict` LRU caching.

    Args:
        db_path: Path to the DuckDB database file.
        connect_fn: DuckDB connection factory callable.
        schema_fn: Schema extraction callable.
        max_cache_entries: Maximum LRU entries retained in `TRANSLATION_SCHEMA_CACHE`.

    Returns:
        str: Formatted database schema string.

    Raises:
        DatabaseNotFoundError: If `db_path` cannot be opened due to `duckdb.IOException`.
        SchemaReadError: If reading the database schema fails.

    """
    cache_key: tuple[str, int, int] | None = None
    if connect_fn is duckdb.connect and schema_fn is get_database_schema:
        db_file = Path(db_path)
        if db_file.is_file():
            st = db_file.stat()
            cache_key = (str(db_file.resolve()), st.st_mtime_ns, st.st_size)
            with _TRANSLATION_SCHEMA_CACHE_LOCK:
                cached = TRANSLATION_SCHEMA_CACHE.get(cache_key)
                if cached is not None:
                    TRANSLATION_SCHEMA_CACHE.move_to_end(cache_key)
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
            with _TRANSLATION_SCHEMA_CACHE_LOCK:
                TRANSLATION_SCHEMA_CACHE[cache_key] = schema
                TRANSLATION_SCHEMA_CACHE.move_to_end(cache_key)
                while len(TRANSLATION_SCHEMA_CACHE) > max(1, max_cache_entries):
                    TRANSLATION_SCHEMA_CACHE.popitem(last=False)
        logger.debug(f"Retrieved schema: {schema[:200]}... (truncated)")
        return schema
    except DatabaseNotFoundError as dnf:
        logger.error(f"Failed to read schema: {dnf}")
        raise
    except (duckdb.Error, OSError, RuntimeError, ValueError, TypeError) as e:
        logger.error(f"Failed to read schema: {e}")
        raise SchemaReadError(params={"error": str(e)}) from e


def invoke_llm_with_local_fallback(
    prompt: str,
    actual_model: str,
    fallback_candidates: Sequence[str],
    effective_api_key: str | None,
    resolved_provider: str,
    is_local_ollama: bool,
    llm_factory: AnyLlmFactoryProtocol | None = None,
    sleep_fn: Callable[[float], None] | None = None,
) -> str:
    """
    Execute synchronous LLM completion and retry across candidate local Ollama fallback models on failure.

    Args:
        prompt: Formatted SQL generation prompt string.
        actual_model: Primary resolved model identifier.
        fallback_candidates: Ordered sequence of lightweight local Ollama fallback model tags.
        effective_api_key: Optional resolved provider API key.
        resolved_provider: Resolved LLM provider name.
        is_local_ollama: Whether the target provider is local Ollama.
        llm_factory: Optional injectable LLM transport factory.
        sleep_fn: Optional injectable sleep callable for brief daemon recovery pause between retries.

    Returns:
        str: Full LLM completion response text.

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
            if is_recoverable_local_ollama_crash(resolved_provider, last_exc):
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


def attempt_agentic_sql_self_correction(
    db_path: str,
    base_prompt: str,
    initial_response: str,
    retry_llm_fn: Callable[[str], str] | None,
) -> tuple[bool, str, str | None]:
    """
    Attempt LLM self-correction via `execute_agentic_nl_to_sql` when initial SQL fails execution.

    Args:
        db_path: Path to the target DuckDB file.
        base_prompt: Base schema-enriched prompt text.
        initial_response: First raw LLM completion string.
        retry_llm_fn: Optional callable invoking the LLM with an error-feedback prompt.

    Returns:
        tuple[bool, str, str | None]: `(succeeded, final_sql, last_error)`.

    """
    if retry_llm_fn is None:
        return False, "", None
    call_idx = 0

    def _step_generate(feedback_prompt: str) -> str:
        """
        Yield initial LLM output on pass 1 and invoke `retry_llm_fn` on pass 2.

        Args:
            feedback_prompt: Self-correction prompt with DuckDB error hint.

        Returns:
            str: Raw LLM completion text.

        """
        nonlocal call_idx
        call_idx += 1
        if call_idx == 1:
            return initial_response
        return retry_llm_fn(feedback_prompt)

    agent_res = execute_agentic_nl_to_sql(
        db_path=db_path,
        nl_query=base_prompt,
        generate_fn=_step_generate,
        execute_fn=execute_duckdb_query,
        max_retries=2,
    )
    return agent_res["success"], agent_res["final_sql"], agent_res["error"]


def repair_curated_template_sql(
    db_path: str,
    prompt: str,
    full_response: str,
    sql_query: str,
    curated_sql: str,
    retry_llm_fn: Callable[[str], str] | None,
) -> TranslationResult:
    """
    Validate or self-correct LLM SQL against a curated benchmark reference, marking fallback honestly if substituted.

    Args:
        db_path: Path to the target DuckDB file.
        prompt: Full schema-enriched prompt sent to the LLM.
        full_response: Initial raw LLM completion string.
        sql_query: Extracted candidate SQL string.
        curated_sql: Gold-standard curated SQL query.
        retry_llm_fn: Optional callable for a self-correction LLM retry.

    Returns:
        TranslationResult: Honest `llm` result when LLM succeeds, or `rule_based_fallback` when substituted.

    """
    comp = compare_predicted_and_gold_sql(
        db_path=db_path,
        pred_sql=sql_query,
        gold_sql=curated_sql,
        execute_fn=execute_duckdb_query,
    )
    if comp.execution_accuracy and not comp.syntax_error:
        return TranslationResult(
            full_response=full_response,
            sql_query=sql_query,
            used_fallback=False,
            generation_mode="llm",
        )
    ok_retry, retried_sql, retry_err = attempt_agentic_sql_self_correction(
        db_path=db_path,
        base_prompt=prompt,
        initial_response=full_response,
        retry_llm_fn=retry_llm_fn,
    )
    if ok_retry:
        comp_retry = compare_predicted_and_gold_sql(
            db_path=db_path,
            pred_sql=retried_sql,
            gold_sql=curated_sql,
            execute_fn=execute_duckdb_query,
        )
        if comp_retry.execution_accuracy and not comp_retry.syntax_error:
            return TranslationResult(
                full_response=f"```sql{NL}{retried_sql}{NL}```",
                sql_query=retried_sql,
                used_fallback=False,
                generation_mode="llm",
            )
    diag_msg = (
        retry_err
        or comp.error_message
        or "LLM SQL did not match curated clinical reference execution."
    )
    return TranslationResult(
        full_response=f"```sql{NL}{curated_sql}{NL}```",
        sql_query=curated_sql,
        used_fallback=True,
        generation_mode="rule_based_fallback",
        llm_error=diag_msg,
        llm_error_code="backend.llmTranslationError",
    )


def validate_or_repair_translated_sql(
    db_path: str,
    schema: str,
    nl_query: str,
    full_response: str,
    sql_query: str,
    actual_model: str,
    resolved_provider: str,
    connect_fn: Callable[..., duckdb.DuckDBPyConnection] = duckdb.connect,
    prompt: str = "",
    retry_llm_fn: Callable[[str], str] | None = None,
) -> TranslationResult:
    """
    Validate LLM-generated SQL via agentic self-correction and report rule-based fallback honestly when repaired.

    Args:
        db_path: Path to the target DuckDB file.
        schema: Extracted database schema text.
        nl_query: User's natural-language question.
        full_response: Raw LLM markdown response text.
        sql_query: Extracted candidate SQL query string.
        actual_model: Resolved LLM model name.
        resolved_provider: Resolved LLM provider identifier.
        connect_fn: DuckDB connection factory callable.
        prompt: Base schema-enriched LLM prompt used for self-correction retries.
        retry_llm_fn: Optional callable invoking a self-correction LLM pass.

    Returns:
        TranslationResult: Validated LLM result or honest `rule_based_fallback` result.

    """
    schema_map = parse_schema_tables_and_columns(schema)
    lower_tables = {t.lower() for t in schema_map}
    curated_sql = (
        lookup_curated_template_sql(nl_query, lower_tables, db_path=db_path)
        if lower_tables
        else None
    )
    if curated_sql is not None:
        return repair_curated_template_sql(
            db_path=db_path,
            prompt=prompt or nl_query,
            full_response=full_response,
            sql_query=sql_query,
            curated_sql=curated_sql,
            retry_llm_fn=retry_llm_fn,
        )

    if connect_fn is duckdb.connect and any(
        m in lower_tables for m in CLINICAL_SCHEMA_MARKERS
    ):
        ok_agent, agent_sql, agent_err = attempt_agentic_sql_self_correction(
            db_path=db_path,
            base_prompt=prompt or nl_query,
            initial_response=full_response,
            retry_llm_fn=retry_llm_fn,
        )
        if ok_agent:
            resolved_full = (
                full_response
                if agent_sql == sql_query
                else f"```sql{NL}{agent_sql}{NL}```"
            )
            return TranslationResult(
                full_response=resolved_full,
                sql_query=agent_sql,
                used_fallback=False,
                generation_mode="llm",
            )
        try:
            if not sql_query.strip():
                raise ValueError("Empty SQL query")
            execute_duckdb_query(db_path, sql_query)
        except (T1DError, duckdb.Error, OSError, RuntimeError, ValueError) as exec_err:
            diag_err = agent_err or str(exec_err)
            return synthesize_fallback_clinical_sql(
                schema,
                nl_query,
                include_diagnostic_note=False,
                llm_error=diag_err,
                llm_error_code="backend.llmTranslationError",
                model_name=actual_model,
                provider=resolved_provider,
                db_path=db_path,
            )

    return TranslationResult(
        full_response=full_response,
        sql_query=sql_query,
        used_fallback=False,
        generation_mode="llm",
    )


def translate_nl_to_sql(
    db_path: str,
    nl_query: str,
    model_name: str = "gemma:2b",
    history: list[ChatMessageModel] | None = None,
    api_key: str | None = None,
    provider: str | None = None,
    connect_fn: Callable[..., duckdb.DuckDBPyConnection] = duckdb.connect,
    schema_fn: Callable[[duckdb.DuckDBPyConnection], str] = get_database_schema,
    llm_factory: AnyLlmFactoryProtocol | None = None,
) -> TranslationResult:
    """
    Translate a natural language question into a DuckDB SQL query using `any-llm`.

    Args:
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
        TranslationResult: Structured translation result `(full_response, extracted_sql_query)`.

    Raises:
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
    effective_api_key = ensure_any_llm_installed_and_ready(
        resolved_provider, api_key, llm_factory
    )
    schema = extract_schema_for_translation(db_path, connect_fn, schema_fn)

    is_local_ollama = resolved_provider.lower() == "ollama"
    effective_retries = resolve_effective_retries_for_provider(resolved_provider)
    fallback_candidates: list[str] = []
    if is_local_ollama and llm_factory is None:
        actual_model, fallback_candidates = resolve_local_ollama_candidates_list(
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

    try:
        logger.info(
            f"Sending prompt to LLM ({resolved_provider}/{actual_model}, effective_retries={effective_retries})..."
        )

        def _retry_llm_once(feedback_prompt: str) -> str:
            """
            Invoke the configured LLM once with DuckDB error feedback for self-correction.

            Args:
                feedback_prompt: Error-annotated prompt text.

            Returns:
                str: LLM completion text.

            """
            return invoke_llm_with_local_fallback(
                prompt=feedback_prompt,
                actual_model=actual_model,
                fallback_candidates=fallback_candidates,
                effective_api_key=effective_api_key,
                resolved_provider=resolved_provider,
                is_local_ollama=is_local_ollama,
                llm_factory=llm_factory,
            )

        full_response = _retry_llm_once(prompt)
        logger.info("LLM response:%s%s", NL, full_response)
        sql_query = extract_sql_from_response(full_response)
        return validate_or_repair_translated_sql(
            db_path=db_path,
            schema=schema,
            nl_query=nl_query,
            full_response=full_response,
            sql_query=sql_query,
            actual_model=actual_model,
            resolved_provider=resolved_provider,
            connect_fn=connect_fn,
            prompt=prompt,
            retry_llm_fn=_retry_llm_once,
        )
    except Exception as e:
        if is_recoverable_local_ollama_crash(resolved_provider, e):
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
                db_path=db_path,
            )
        logger.error(f"LLM translation error ({type(e).__name__}): {e}")
        raise LlmTranslationError(params={"error": str(e)}) from e


def resolve_effective_gold_sql(
    message: str,
    explicit_gold_sql: str | None = None,
) -> str | None:
    """
    Resolve an explicit `gold_sql` string or auto-detect a canonical Gold Template SQL query.

    Args:
        message: Natural language prompt string.
        explicit_gold_sql: Optional explicit gold-standard SQL provided in the request.

    Returns:
        str | None: Non-empty gold SQL query string if present or matched, else None.

    """
    if explicit_gold_sql and explicit_gold_sql.strip():
        return explicit_gold_sql.strip()
    return lookup_curated_template_sql(message)


def compute_optional_gold_comparison(
    db_path: str,
    message: str,
    pred_sql: str,
    explicit_gold_sql: str | None,
    execute_fn: Callable[[str, str], list[dict[str, SqlValue]]],
    precomputed_pred_rows: list[dict[str, SqlValue]] | None = None,
    precomputed_pred_error: str | None = None,
) -> SqlGoldComparisonModel | None:
    """
    Compute `SqlGoldComparisonModel` on DuckDB when a gold SQL query is available.

    Args:
        db_path: Validated path to the DuckDB file.
        message: Natural language prompt string.
        pred_sql: SQL query generated by the target LLM.
        explicit_gold_sql: Optional explicit gold SQL from the request.
        execute_fn: Read-only DuckDB query execution callable.
        precomputed_pred_rows: Optional already-executed rows for `pred_sql`.
        precomputed_pred_error: Optional error message if `pred_sql` already failed execution.

    Returns:
        SqlGoldComparisonModel | None: Comparison model if gold SQL exists, else None.

    """
    effective_gold = resolve_effective_gold_sql(message, explicit_gold_sql)
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


__all__ = [
    "CLINICAL_SCHEMA_MARKERS",
    "MAX_TRANSLATION_SCHEMA_CACHE_ENTRIES",
    "TRANSLATION_SCHEMA_CACHE",
    "attempt_agentic_sql_self_correction",
    "build_sql_prompt_with_history",
    "complete_non_streaming_with_fallback",
    "compute_optional_gold_comparison",
    "create_verified_llm_transport",
    "ensure_any_llm_installed_and_ready",
    "extract_completion_or_iterable_text",
    "extract_schema_for_translation",
    "invoke_llm_with_local_fallback",
    "probe_installed_ollama_names",
    "repair_curated_template_sql",
    "resolve_and_select_local_ollama_model",
    "resolve_effective_gold_sql",
    "resolve_effective_retries_for_provider",
    "resolve_local_ollama_candidates_list",
    "stream_llm_completion_tokens",
    "translate_nl_to_sql",
    "validate_or_repair_translated_sql",
]
