"""Natural-language-to-SQL translation and Server-Sent Events (SSE) chat streaming services."""

import asyncio
import json
import logging
import re
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
    get_database_schema,
)
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
from t1d_analytics.sql_agent import (
    _extract_schema_map_from_text,
    prune_schema_with_gemma_rag,
)
from t1d_analytics.sql_utils import (
    ALL_SQL_KEYWORDS,
    SqlValue,
    connect_duckdb_read_only_pooled,
    escape_sql_identifier,
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


_TABLE_1_FALLBACK_KEYWORDS = {
    "table 1",
    "table1",
    "baseline",
    "characteristics",
    "stratified",
    "arm",
    "cohort",
    "summary",
    "demographics",
    "outcomes",
    "quartile",
    "iqr",
    "stddev",
}


def _score_table_for_query(
    tbl_name: str,
    cols: List[str],
    lower_q: str,
    query_words: set[str],
) -> int:
    """
    Compute a relevance score for a schema table against a natural-language query.

    Args:
    ----
        tbl_name: Candidate table name.
        cols: Column names belonging to the table.
        lower_q: Lowercased natural-language query.
        query_words: Tokenized lowercase query words.

    Returns:
    -------
        int: Non-negative relevance score.

    """
    tbl_lower = tbl_name.lower()
    score = 10 if tbl_lower in lower_q else 0
    if tbl_lower in ("patients", "demographics", "tblaptsummary") and (
        "table 1" in lower_q
        or "baseline" in lower_q
        or len(query_words.intersection(_TABLE_1_FALLBACK_KEYWORDS)) >= 2
    ):
        score += 6
    for w in query_words:
        if len(w) >= 3 and w in tbl_lower:
            score += 4
        score += sum(2 for c in cols if len(w) >= 3 and w == c.lower())
    return score


def _build_fallback_table_1_sql(safe_tbl: str, cols: List[str]) -> str:
    """
    Build a deterministic Table 1 baseline summary DuckDB SQL query for `safe_tbl`.

    Args:
    ----
        safe_tbl: Escaped DuckDB table identifier.
        cols: Column names present in `safe_tbl`.

    Returns:
    -------
        str: Valid DuckDB SELECT statement summarizing baseline cohort characteristics.

    """
    col_lookup = {c.lower(): c for c in cols}
    group_col = (
        col_lookup.get("txgroup")
        or col_lookup.get("treatment_group")
        or col_lookup.get("pt_pumpmanuf")
    )
    select_parts: List[str] = []
    if group_col:
        safe_grp = escape_sql_identifier(group_col)
        select_parts.append(f'"{safe_grp}" AS treatment_arm')
    select_parts.append("COUNT(*) AS n_participants")
    for metric_key, alias_label in (
        ("age", "mean_age_yr"),
        ("ageasofranddt", "mean_age_yr"),
        ("hba1c", "mean_hba1c_pct"),
        ("tir", "mean_tir_pct"),
        ("tbr", "mean_tbr_pct"),
        ("tar", "mean_tar_pct"),
        ("mean_glucose", "mean_glucose_mgdl"),
    ):
        actual_col = col_lookup.get(metric_key)
        if actual_col and all(alias_label not in p for p in select_parts):
            safe_col = escape_sql_identifier(actual_col)
            select_parts.append(
                f'ROUND(AVG(TRY_CAST("{safe_col}" AS DOUBLE)), 1) AS {alias_label}'
            )
    proj = ", ".join(select_parts)
    if group_col:
        safe_grp = escape_sql_identifier(group_col)
        return (
            f'SELECT {proj} FROM "{safe_tbl}" '
            f'GROUP BY "{safe_grp}" ORDER BY "{safe_grp}";'
        )
    return f'SELECT {proj} FROM "{safe_tbl}";'


def _select_best_fallback_table(
    schema_map: Dict[str, List[str]],
    lower_q: str,
) -> str:
    """
    Choose the best target table from `schema_map` for fallback SQL synthesis.

    Args:
    ----
        schema_map: Mapping of table names to column names.
        lower_q: Lowercased natural-language query.

    Returns:
    -------
        str: Selected table identifier.

    """
    if not schema_map:
        return "patients"

    query_words = set(re.findall(r"[a-z0-9_]+", lower_q))
    best_table: Optional[str] = None
    best_score = 0
    for tbl_name, cols in schema_map.items():
        score = _score_table_for_query(tbl_name, cols, lower_q, query_words)
        if score > best_score:
            best_score = score
            best_table = tbl_name

    if best_table is not None:
        return best_table

    lower_to_actual = {tbl_name.lower(): tbl_name for tbl_name in schema_map}
    for preferred in ("demographics", "patients", "tblaptsummary", "subjects"):
        if preferred in lower_to_actual:
            return lower_to_actual[preferred]

    return next(iter(schema_map.keys()))


def synthesize_fallback_clinical_sql(
    schema: str,
    nl_query: str,
    include_diagnostic_note: bool = False,
) -> Tuple[str, str]:
    """
    Synthesize a deterministic, schema-validated DuckDB SQL query when local LLM inference crashes.

    Used as a resilient fallback when local Ollama (`llama-server`) terminates due to
    memory allocation failure (`ggml_backend_amx_buffer_type_alloc_buffer` / OOM) or crash.

    Args:
    ----
        schema: Extracted database schema text.
        nl_query: User's natural language question.
        include_diagnostic_note: Whether to prepend a diagnostic notice explaining that local inference fell back to schema-guided SQL synthesis.

    Returns:
    -------
        Tuple[str, str]: `(markdown_response_with_sql_fence, raw_sql_query)`.

    """
    schema_map = _extract_schema_map_from_text(schema)
    lower_q = nl_query.lower()

    limit_match = re.search(r"\b(?:first|top|limit)\s+(\d+)\b", lower_q)
    limit_val = int(limit_match.group(1)) if limit_match else 10
    target_table = _select_best_fallback_table(schema_map, lower_q)

    safe_tbl = escape_sql_identifier(target_table)
    if re.search(
        r"\b(?:table\s*1|baseline\s+characteristics|cohort\s+summary)\b", lower_q
    ):
        sql = _build_fallback_table_1_sql(safe_tbl, schema_map.get(target_table, []))
    elif re.search(r"\b(?:how\s+many|count|total\s+number)\b", lower_q):
        sql = f'SELECT COUNT(*) AS count FROM "{safe_tbl}";'
    else:
        sql = f'SELECT * FROM "{safe_tbl}" LIMIT {limit_val};'

    nl = chr(10)
    note_prefix = (
        f"_Note: Local LLM inference was unavailable; synthesized schema-guided fallback SQL for `{safe_tbl}`._{nl}{nl}"
        if include_diagnostic_note
        else ""
    )
    full_resp = f"{note_prefix}```sql{nl}{sql}{nl}```"
    return full_resp, sql


def _is_recoverable_local_ollama_crash(provider: str, exc: BaseException) -> bool:
    """
    Return True if `exc` is a local Ollama server crash, timeout, missing model tag, or OOM allocation failure.

    Args:
    ----
        provider: Resolved LLM provider name.
        exc: Exception raised by the underlying transport.

    Returns:
    -------
        bool: True when `provider == 'ollama'` and `exc` indicates a server crash/OOM/disconnect/timeout.

    """
    if provider.lower() != "ollama":
        return False
    msg = str(exc).lower()
    crash_markers = (
        "llama-server process has terminated",
        "insufficient memory",
        "unable to allocate",
        "alloc_tensor_range",
        "segmentation fault",
        "server disconnected without sending a response",
        "cannot use the `sync` api in an `async` context",
        "timed out",
        "timeout",
        "not found, try pulling it first",
        "model_not_pulled",
    )
    return any(marker in msg for marker in crash_markers)


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
    except Exception:
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
    except Exception:
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
            f"Native streaming failed for provider '{resolved_provider}' model '{actual_model}' ({type(e).__name__}: {e}). Falling back to non-streaming completion."
        )

    try:
        response = llm.completion(
            model=actual_model,
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception as primary_exc:
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
    full_text = extract_stream_delta(response, resolved_provider) or str(response)
    yield full_text


_stream_llm_with_ollama_fallback = stream_llm_completion_tokens


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
        try:
            schema = schema_fn(conn)
        finally:
            conn.close()
        logger.debug(f"Retrieved schema: {schema[:200]}... (truncated)")
    except DatabaseNotFoundError as dnf:
        logger.error(f"Failed to read schema: {dnf}")
        raise
    except Exception as e:
        logger.error(f"Failed to read schema: {e}")
        raise SchemaReadError(params={"error": str(e)}) from e

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
        except Exception as primary_exc:
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
        return full_response, sql_query
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


generate_sql_from_nl = translate_nl_to_sql


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
    _, synthesized_sql = synthesize_fallback_clinical_sql(schema, message)
    return synthesized_sql


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

        full_response, sql_query = translate_fn(
            db_path,
            request.message,
            model_name=request.model,
            history=request.history,
            api_key=request.api_key,
            provider=request.provider,
        )
        used_fallback = (
            "Local LLM inference encountered a memory error" in full_response
            or "Local LLM inference was unavailable" in full_response
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
            is_synthetic_data=is_synthetic,
            is_rule_based_sql=used_fallback,
            generation_mode="rule_based_fallback" if used_fallback else "llm",
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

    Yields:
    ------
        str: Formatted SSE event frames.

    """
    provenance, is_synthetic = _resolve_db_provenance_flags(
        db_path, connect_fn=connect_fn
    )
    used_fallback = (
        "Local LLM inference encountered a memory error" in full_response
        or "Local LLM inference was unavailable" in full_response
    )
    sql_query = _extract_sql_from_response(full_response)
    sql_results: Optional[List[Dict[str, SqlValue]]] = None
    sql_err_dict: Optional[Dict[str, SqlValue]] = None
    if sql_query:
        yield f"data: {json.dumps({'event': 'sql_generated', 'sqlQuery': sql_query})}{sse_sep}"
        if auto_execute:
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
        "is_synthetic_data": is_synthetic,
        "is_rule_based_sql": used_fallback,
        "generation_mode": "rule_based_fallback" if used_fallback else "llm",
        "data_provenance": provenance,
    }
    yield from _emit_sse_terminal_payload(nl_res_dict, sse_sep)


async def _is_request_disconnected(req: Optional[Request]) -> bool:
    """
    Check whether an optional FastAPI `Request` has disconnected.

    Args:
    ----
        req: Optional FastAPI Request object.

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
    fb_resp, _ = synthesize_fallback_clinical_sql(
        schema, message, include_diagnostic_note=True
    )
    tok_payload = json.dumps({"event": "token", "token": fb_resp})
    return fb_resp, f"data: {tok_payload}{sse_sep}"


_recover_or_reraise_local_ollama_failure = _recover_ollama_stream_fallback_token


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
