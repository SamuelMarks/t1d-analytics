"""Rule-based clinical SQL fallback synthesis and translation result structures."""

import re
import urllib.error
from typing import Dict, List, Literal, Optional, Set, Tuple, Union

from pydantic.dataclasses import dataclass

from t1d_analytics.errors import T1DError
from t1d_analytics.sql_utils import escape_sql_identifier, parse_schema_text_to_map

GenerationModeLiteral = Literal["llm", "rule_based_fallback", "literal_sql"]

_FALLBACK_DIAGNOSTIC_PREFIX = "_Note: Local LLM inference"
_FALLBACK_NOTE_ERROR_RE = re.compile(
    r"Local LLM inference was unavailable\s*\(\s*([^)]+?)\s*\)"
)

_TABLE_1_FALLBACK_KEYWORDS: Set[str] = {
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


@dataclass(frozen=True)
class TranslationResult:
    """
    Structured result of translating a natural-language clinical query into DuckDB SQL.

    Carries explicit typed metadata about whether rule-based fallback synthesis
    was used and any underlying LLM error.

    Attributes
    ----------
        full_response: Complete assistant markdown text (including SQL fence).
        sql_query: Extracted executable DuckDB SQL statement.
        used_fallback: True if rule-based SQL synthesis was used instead of an LLM.
        generation_mode: One of `'llm'`, `'rule_based_fallback'`, or `'literal_sql'`.
        llm_error: Optional human-readable diagnostic reason why LLM inference failed.
        llm_error_code: Optional structured error code for the LLM failure.

    """

    full_response: str
    sql_query: str
    used_fallback: bool = False
    generation_mode: GenerationModeLiteral = "llm"
    llm_error: Optional[str] = None
    llm_error_code: Optional[str] = None

    @property
    def is_rule_based_sql(self) -> bool:
        """
        Return True when SQL was synthesized via rule-based fallback rather than an LLM.

        Returns
        -------
            bool: True if `used_fallback` is True or `generation_mode == 'rule_based_fallback'`.

        """
        return self.used_fallback or self.generation_mode == "rule_based_fallback"


def extract_fallback_error_from_text(response_text: str) -> Optional[str]:
    """
    Extract an embedded parenthetical LLM error reason from a fallback response text string.

    Args:
    ----
        response_text: Assistant response markdown text.

    Returns:
    -------
        Optional[str]: Extracted parenthetical error message if present, else None.

    """
    match = _FALLBACK_NOTE_ERROR_RE.search(response_text)
    return match.group(1) if match else None


def _deduplicate_internal_server_error_prefix(message: str) -> str:
    """
    Collapse repeated `'Internal server error: '` prefixes in upstream LLM error messages.

    Args:
    ----
        message: Raw single-line error message.

    Returns:
    -------
        str: Error message with duplicate `'Internal server error: '` prefixes collapsed.

    """
    cleaned = message
    prefix = "Internal server error: "
    double_prefix = f"{prefix}{prefix}"
    while cleaned.lower().startswith(double_prefix.lower()):
        cleaned = f"{prefix}{cleaned[len(double_prefix) :].lstrip()}"
    return cleaned


def format_llm_failure_reason(
    exc_or_reason: Optional[Union[BaseException, str]],
    model_name: Optional[str] = None,
    provider: Optional[str] = None,
    *,
    model: Optional[str] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Format an LLM exception or failure reason into a sanitized `(human_readable_reason, error_code)` tuple.

    Args:
    ----
        exc_or_reason: Caught exception or descriptive failure string.
        model_name: Optional requested model identifier.
        provider: Optional resolved LLM provider name.
        model: Optional keyword alias for `model_name`.

    Returns:
    -------
        Tuple[Optional[str], Optional[str]]: `(sanitized_error_message, structured_error_code)`.

    """
    if exc_or_reason is None:
        return None, None

    effective_model = model_name if model_name is not None else model
    model_label = f" for model '{effective_model}'" if effective_model else ""
    prov_label = f" ({provider})" if provider else ""

    if isinstance(exc_or_reason, str):
        cleaned_str = _deduplicate_internal_server_error_prefix(
            " ".join(exc_or_reason.strip().split())
        )
        if not cleaned_str:
            return None, None
        return cleaned_str, "backend.llmInvalidSqlOutput"

    if isinstance(exc_or_reason, T1DError):
        detail = exc_or_reason.params.get("error")
        msg = (
            str(detail).strip()
            if detail is not None and str(detail).strip()
            else str(exc_or_reason).strip()
        )
        single_line = _deduplicate_internal_server_error_prefix(" ".join(msg.split()))
        return single_line, exc_or_reason.error_code

    if isinstance(exc_or_reason, urllib.error.HTTPError):
        reason_str = str(exc_or_reason.reason or "HTTP error").strip()
        return (
            f"HTTP {exc_or_reason.code} {reason_str}{prov_label}{model_label}",
            "backend.llmHttpError",
        )

    if isinstance(exc_or_reason, urllib.error.URLError):
        reason_text = " ".join(str(exc_or_reason.reason).strip().split())
        return (
            f"Connection failed{prov_label}{model_label}: {reason_text}",
            "backend.llmConnectionError",
        )

    raw_msg = _deduplicate_internal_server_error_prefix(
        " ".join(str(exc_or_reason).strip().split())
    )
    sanitized_msg = re.sub(
        r"(sk-[A-Za-z0-9_-]{6,}|Bearer\s+[A-Za-z0-9._-]+)",
        "[REDACTED]",
        raw_msg,
    )
    lower_msg = sanitized_msg.lower()

    if (
        isinstance(exc_or_reason, TimeoutError)
        or "timed out" in lower_msg
        or "timeout" in lower_msg
    ):
        return (
            f"Inference timed out{prov_label}{model_label}: {sanitized_msg}",
            "backend.llmTimeoutError",
        )

    if (
        isinstance(exc_or_reason, ConnectionError)
        or "connection refused" in lower_msg
        or "failed to connect to ollama" in lower_msg
    ):
        return (
            f"Connection refused{prov_label}{model_label}: {sanitized_msg}",
            "backend.llmConnectionError",
        )

    if (
        "insufficient memory" in lower_msg
        or "unable to allocate" in lower_msg
        or "alloc_tensor_range" in lower_msg
        or "llama-server process has terminated" in lower_msg
        or "segmentation fault" in lower_msg
        or "core dumped" in lower_msg
    ):
        clean_model = (effective_model or "ollama").strip() or "ollama"
        return (
            f"Local model '{clean_model}' was unavailable on CPU runner",
            "backend.llmMemoryCrash",
        )

    return (
        f"{type(exc_or_reason).__name__}{prov_label}{model_label}: {sanitized_msg}",
        "backend.llmTranslationError",
    )


def is_fallback_response_text(response_text: str) -> bool:
    """
    Determine whether an assistant response text contains a local LLM fallback diagnostic marker.

    Args:
    ----
        response_text: Assistant response markdown text.

    Returns:
    -------
        bool: True if the response text indicates rule-based fallback SQL synthesis.

    """
    return (
        _FALLBACK_DIAGNOSTIC_PREFIX in response_text
        or "Local LLM inference encountered a memory error" in response_text
        or "Local LLM inference was unavailable" in response_text
    )


def _find_matching_column(
    cols: List[str], candidates: Tuple[str, ...]
) -> Optional[str]:
    """
    Find the first column in `cols` whose lowercase name matches one of `candidates`.

    Args:
    ----
        cols: List of column names from a table schema.
        candidates: Candidate lowercase column identifiers in priority order.

    Returns:
    -------
        Optional[str]: Original-case column name if matched, else None.

    """
    col_lookup = {c.lower(): c for c in cols}
    for cand in candidates:
        matched = col_lookup.get(cand.lower())
        if matched is not None:
            return matched
    return None


def _parse_schema_tables_and_columns(schema: str) -> Dict[str, List[str]]:
    """
    Parse a formatted database schema string into a mapping of table names to column names.

    Args:
    ----
        schema: Formatted database schema string.

    Returns:
    -------
        Dict[str, List[str]]: Mapping from table name to list of column names.

    """
    return parse_schema_text_to_map(schema)


def _score_table_for_query(
    tbl_name: str,
    cols: List[str],
    lower_q: str,
    query_words: Set[str],
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
    group_col = _find_matching_column(
        cols, ("txgroup", "treatment_group", "pt_pumpmanuf")
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
        actual_col = _find_matching_column(cols, (metric_key,))
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


def _build_treatment_group_hypo_summary_sql(
    schema_map: Dict[str, List[str]],
    fallback_table: str,
) -> Optional[str]:
    """
    Build a grouped DuckDB SQL query for treatment group, patient count, average age, and severe hypoglycemia events.

    Args:
    ----
        schema_map: Mapping of table names to column names.
        fallback_table: Default selected fallback table.

    Returns:
    -------
        Optional[str]: Valid DuckDB SELECT...GROUP BY statement if matching columns exist, else None.

    """
    lower_to_actual = {t.lower(): t for t in schema_map}
    target_tbl = fallback_table
    for preferred in ("patients", "tblaptsummary", "demographics"):
        if preferred in lower_to_actual:
            cand_cols = schema_map.get(lower_to_actual[preferred], [])
            if _find_matching_column(cand_cols, ("txgroup", "treatment_group")):
                target_tbl = lower_to_actual[preferred]
                break

    cols = schema_map.get(target_tbl, [])
    grp_col = _find_matching_column(
        cols, ("txgroup", "treatment_group", "arm", "group")
    )
    age_col = _find_matching_column(cols, ("age", "ageasofranddt"))
    hypo_col = _find_matching_column(
        cols,
        (
            "numsevhypo",
            "severe_hypo_events",
            "severe_hypoglycemia_events",
            "sevhypo",
            "numhospdka",
        ),
    )
    id_col = _find_matching_column(cols, ("subject_id", "patient_id", "ptid", "id"))
    if not grp_col:
        return None

    safe_tbl = escape_sql_identifier(target_tbl)
    safe_grp = escape_sql_identifier(grp_col)
    select_exprs: List[str] = [f'"{safe_grp}" AS treatment_group']
    if id_col:
        safe_id = escape_sql_identifier(id_col)
        select_exprs.append(f'COUNT(DISTINCT "{safe_id}") AS patient_count')
    else:
        select_exprs.append("COUNT(*) AS patient_count")
    if age_col:
        safe_age = escape_sql_identifier(age_col)
        select_exprs.append(
            f'ROUND(AVG(TRY_CAST("{safe_age}" AS DOUBLE)), 1) AS avg_age'
        )
    if hypo_col:
        safe_hypo = escape_sql_identifier(hypo_col)
        select_exprs.append(
            f'COALESCE(SUM(TRY_CAST("{safe_hypo}" AS DOUBLE)), 0) AS total_severe_hypo_events'
        )
    proj = ", ".join(select_exprs)
    return (
        f'SELECT {proj} FROM "{safe_tbl}" GROUP BY "{safe_grp}" ORDER BY "{safe_grp}";'
    )


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
    llm_error: Optional[str] = None,
    llm_error_code: Optional[str] = None,
    model_name: Optional[str] = None,
    provider: Optional[str] = None,
    *,
    model: Optional[str] = None,
) -> TranslationResult:
    """
    Synthesize a deterministic, schema-validated DuckDB SQL query when local LLM inference crashes.

    Used as a resilient fallback when local Ollama (`llama-server`) terminates due to
    memory allocation failure (`ggml_backend_amx_buffer_type_alloc_buffer` / OOM) or crash.

    Args:
    ----
        schema: Extracted database schema text.
        nl_query: User's natural language question.
        include_diagnostic_note: Whether to prepend a diagnostic notice explaining that local inference fell back to schema-guided SQL synthesis.
        llm_error: Optional human-readable diagnostic reason why LLM inference failed.
        llm_error_code: Optional structured error code for the LLM failure.
        model_name: Optional requested model identifier.
        provider: Optional resolved LLM provider name.
        model: Optional keyword alias for `model_name`.

    Returns:
    -------
        TranslationResult: Structured translation result (unpackable as `(markdown_response_with_sql_fence, raw_sql_query)`).

    """
    _ = provider
    effective_model = model_name if model_name is not None else model
    clean_model = (effective_model or "ollama").strip() or "ollama"
    schema_map = _parse_schema_tables_and_columns(schema)
    lower_q = nl_query.lower()

    limit_match = re.search(r"\b(?:first|top|limit)\s+(\d+)\b", lower_q)
    limit_val = int(limit_match.group(1)) if limit_match else 10
    target_table = _select_best_fallback_table(schema_map, lower_q)

    safe_tbl = escape_sql_identifier(target_table)
    lower_schema_tables = {tbl.lower() for tbl in schema_map}
    curated_sql = _lookup_curated_template_sql(nl_query, lower_schema_tables)
    grouped_hypo_sql = (
        _build_treatment_group_hypo_summary_sql(schema_map, target_table)
        if (
            "grouped by" in lower_q
            or ("treatment" in lower_q and ("hypo" in lower_q or "age" in lower_q))
        )
        else None
    )
    if curated_sql is not None:
        sql = curated_sql
    elif grouped_hypo_sql is not None:
        sql = grouped_hypo_sql
    elif re.search(
        r"\b(?:table\s*1|baseline\s+characteristics|cohort\s+summary)\b", lower_q
    ):
        sql = _build_fallback_table_1_sql(safe_tbl, schema_map.get(target_table, []))
    elif re.search(r"\b(?:how\s+many|count|total\s+number)\b", lower_q):
        sql = f'SELECT COUNT(*) AS count FROM "{safe_tbl}";'
    else:
        sql = f'SELECT * FROM "{safe_tbl}" LIMIT {limit_val};'

    cleaned_err = llm_error.strip() if llm_error and llm_error.strip() else None
    effective_code = (
        llm_error_code
        if llm_error_code
        else ("backend.llmTranslationError" if cleaned_err else None)
    )

    nl = chr(10)
    if include_diagnostic_note:
        if cleaned_err:
            note_prefix = (
                f"_Note: Local LLM inference was unavailable ({cleaned_err}); "
                f"synthesized schema-guided fallback SQL for `{safe_tbl}`._{nl}{nl}"
            )
        else:
            note_prefix = (
                f"_Note: Local LLM inference was unavailable; "
                f"synthesized schema-guided fallback SQL for `{safe_tbl}`._{nl}{nl}"
            )
    else:
        note_prefix = ""

    sql_comment = (
        f"-- Synthesized schema-guided fallback SQL for {safe_tbl} ({clean_model})"
    )
    full_resp = f"{note_prefix}```sql{nl}{sql_comment}{nl}{sql}{nl}```"
    return TranslationResult(
        full_response=full_resp,
        sql_query=sql,
        used_fallback=True,
        generation_mode="rule_based_fallback",
        llm_error=cleaned_err,
        llm_error_code=effective_code,
    )


_LOCALIZED_STARTER_5_PROMPTS: Tuple[str, ...] = (
    "show me the first 5 patients in the demographics table.",
    "人口統計テーブルの最初の5人の患者を表示してください。",
    "أظهر لي أول 5 مرضى في جدول التركيبة السكانية.",
    "הצג לי את 5 המטופלים הראשונים בטבלת הדמוגרפיה.",
)


def _lookup_table_1_curated_sql(prompt: str) -> Optional[str]:
    """
    Match a prompt against the canonical Academic Table 1 benchmark suite.

    Args:
    ----
        prompt: Natural language user question.

    Returns:
    -------
        Optional[str]: Gold DuckDB SQL if the prompt matches a Table 1 benchmark query, else None.

    """
    from t1d_analytics.training_data import TABLE_1_BENCHMARK_SUITE

    norm_prompt = " ".join(prompt.strip().lower().split())
    for item in TABLE_1_BENCHMARK_SUITE:
        norm_q = " ".join(item["question"].strip().lower().split())
        if norm_prompt == norm_q:
            return item["gold_sql"]
    return None


def _lookup_curated_template_sql(
    prompt: str,
    lower_schema_tables: Optional[Set[str]] = None,
) -> Optional[str]:
    """
    Match a prompt against all 12 Gold Templates (5 Starter Templates + 7 Academic Table 1 Templates).

    Verifies that the required target table(s) for the matched template exist in
    `lower_schema_tables` when a non-empty schema set is supplied.

    Args:
    ----
        prompt: Natural language user question.
        lower_schema_tables: Optional set of lowercase table names present in the active database schema.

    Returns:
    -------
        Optional[str]: Gold DuckDB SQL if matched and schema-compatible, else None.

    """
    from t1d_analytics.training_data import STARTER_TEMPLATE_SUITE

    t1_sql = _lookup_table_1_curated_sql(prompt)
    if t1_sql is not None:
        if not lower_schema_tables or "patients" in lower_schema_tables:
            return t1_sql
        return None

    norm_prompt = " ".join(prompt.strip().lower().split())
    if norm_prompt in _LOCALIZED_STARTER_5_PROMPTS and (
        not lower_schema_tables or "demographics" in lower_schema_tables
    ):
        return "SELECT * FROM demographics LIMIT 5;"

    starter_req_tables: Dict[str, Tuple[str, ...]] = {
        "starter-1": ("tblaptsummary",),
        "starter-2": ("adverseevents",),
        "starter-3": ("subjects",),
        "starter-4": ("tblaptsummary", "hba1c"),
        "starter-5": ("demographics",),
    }
    for item in STARTER_TEMPLATE_SUITE:
        norm_q = " ".join(item["question"].strip().lower().split())
        if norm_prompt == norm_q:
            req_tbls = starter_req_tables.get(item["id"], ())
            if not lower_schema_tables or all(
                t in lower_schema_tables for t in req_tbls
            ):
                return item["gold_sql"]
            return None

    return None


def _is_recoverable_local_ollama_crash(provider: str, exc: Exception) -> bool:
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
        "core dumped",
        "signal: ",
        "exit status ",
        "status code: 500",
        "server disconnected without sending a response",
        "cannot use the `sync` api in an `async` context",
        "timed out",
        "timeout",
        "not found, try pulling it first",
        "model_not_pulled",
        "failed to connect to ollama",
        "check that ollama is downloaded",
        "connection refused",
        "connecterror",
        "connection error",
        "connection reset",
        "remoteprotocolerror",
        "broken pipe",
    )
    return any(marker in msg for marker in crash_markers)


__all__ = [
    "_FALLBACK_DIAGNOSTIC_PREFIX",
    "_TABLE_1_FALLBACK_KEYWORDS",
    "GenerationModeLiteral",
    "TranslationResult",
    "_build_fallback_table_1_sql",
    "_build_treatment_group_hypo_summary_sql",
    "_deduplicate_internal_server_error_prefix",
    "_find_matching_column",
    "_is_recoverable_local_ollama_crash",
    "_lookup_curated_template_sql",
    "_lookup_table_1_curated_sql",
    "_parse_schema_tables_and_columns",
    "_score_table_for_query",
    "_select_best_fallback_table",
    "extract_fallback_error_from_text",
    "format_llm_failure_reason",
    "is_fallback_response_text",
    "synthesize_fallback_clinical_sql",
]
