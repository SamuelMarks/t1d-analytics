"""Rule-based clinical SQL fallback synthesis and translation result structures."""

import re
import urllib.error
from typing import Literal

from pydantic.dataclasses import dataclass

from t1d_analytics.errors import (
    LlmAuthenticationError,
    LlmConnectionError,
    LlmQuotaExceededError,
    LlmRateLimitError,
    T1DError,
)
from t1d_analytics.sql_utils import escape_sql_identifier, parse_schema_text_to_map
from t1d_analytics.table1_builder import (
    build_dynamic_table1_sql,
    has_custom_table1_modifiers,
)

GenerationModeLiteral = Literal["llm", "rule_based_fallback", "literal_sql"]

FALLBACK_DIAGNOSTIC_PREFIX = "_Note: Local LLM inference"
_FALLBACK_NOTE_ERROR_RE = re.compile(
    r"Local LLM inference was unavailable\s*\(\s*([^)]+?)\s*\)"
)

TABLE_1_FALLBACK_KEYWORDS: set[str] = {
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

    Attributes:
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
    llm_error: str | None = None
    llm_error_code: str | None = None

    @property
    def is_rule_based_sql(self) -> bool:
        """
        Return True when SQL was synthesized via rule-based fallback rather than an LLM.

        Returns:
            bool: True if `used_fallback` is True or `generation_mode == 'rule_based_fallback'`.

        """
        return self.used_fallback or self.generation_mode == "rule_based_fallback"


def extract_fallback_error_from_text(response_text: str) -> str | None:
    """
    Extract an embedded parenthetical LLM error reason from a fallback response text string.

    Args:
        response_text: Assistant response markdown text.

    Returns:
        Optional[str]: Extracted parenthetical error message if present, else None.

    """
    match = _FALLBACK_NOTE_ERROR_RE.search(response_text)
    return match.group(1) if match else None


def _deduplicate_internal_server_error_prefix(message: str) -> str:
    """
    Collapse repeated `'Internal server error: '` prefixes in upstream LLM error messages.

    Args:
        message: Raw single-line error message.

    Returns:
        str: Error message with duplicate `'Internal server error: '` prefixes collapsed.

    """
    cleaned = message
    prefix = "Internal server error: "
    double_prefix = f"{prefix}{prefix}"
    while cleaned.lower().startswith(double_prefix.lower()):
        cleaned = f"{prefix}{cleaned[len(double_prefix) :].lstrip()}"
    return cleaned


def _format_typed_t1d_llm_error(
    exc: T1DError,
    effective_model: str | None,
    prov_label: str,
    model_label: str,
) -> tuple[str, str]:
    """
    Format a typed `T1DError` subclass into `(human_readable_reason, error_code)`.

    Args:
        exc: Typed `T1DError` subclass instance.
        effective_model: Optional target model identifier.
        prov_label: Formatted provider suffix string.
        model_label: Formatted model suffix string.

    Returns:
        Tuple[str, str]: `(sanitized_error_message, structured_error_code)`.

    """
    if isinstance(exc, LlmRateLimitError):
        return f"Rate limit exceeded{prov_label}{model_label}", exc.error_code
    if isinstance(exc, LlmAuthenticationError):
        return f"Authentication failed{prov_label}{model_label}", exc.error_code
    if isinstance(exc, LlmQuotaExceededError):
        return f"Quota exceeded{prov_label}{model_label}", exc.error_code
    if isinstance(exc, LlmConnectionError):
        raw_conn = " ".join(str(exc).strip().split())
        if any(
            m in raw_conn.lower()
            for m in (
                "insufficient memory",
                "unable to allocate",
                "alloc_tensor_range",
                "llama-server process has terminated",
                "segmentation fault",
                "core dumped",
            )
        ):
            clean_model = (effective_model or "ollama").strip() or "ollama"
            return (
                f"Local model '{clean_model}' was unavailable on CPU runner",
                "backend.llmMemoryCrash",
            )
        return (
            f"Connection failed{prov_label}{model_label}: {raw_conn}",
            exc.error_code,
        )
    detail = exc.params.get("error")
    msg = (
        str(detail).strip()
        if detail is not None and str(detail).strip()
        else str(exc).strip()
    )
    single_line = _deduplicate_internal_server_error_prefix(" ".join(msg.split()))
    return single_line, exc.error_code


def format_llm_failure_reason(
    exc_or_reason: BaseException | str | None,
    model_name: str | None = None,
    provider: str | None = None,
    *,
    model: str | None = None,
) -> tuple[str | None, str | None]:
    """
    Format an LLM exception or failure reason into a sanitized `(human_readable_reason, error_code)` tuple.

    Args:
        exc_or_reason: Caught exception or descriptive failure string.
        model_name: Optional requested model identifier.
        provider: Optional resolved LLM provider name.
        model: Optional keyword alias for `model_name`.

    Returns:
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
        return _format_typed_t1d_llm_error(
            exc_or_reason, effective_model, prov_label, model_label
        )

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
        response_text: Assistant response markdown text.

    Returns:
        bool: True if the response text indicates rule-based fallback SQL synthesis.

    """
    return (
        FALLBACK_DIAGNOSTIC_PREFIX in response_text
        or "Local LLM inference encountered a memory error" in response_text
        or "Local LLM inference was unavailable" in response_text
    )


def _find_matching_column(cols: list[str], candidates: tuple[str, ...]) -> str | None:
    """
    Find the first column in `cols` whose lowercase name matches one of `candidates`.

    Args:
        cols: List of column names from a table schema.
        candidates: Candidate lowercase column identifiers in priority order.

    Returns:
        Optional[str]: Original-case column name if matched, else None.

    """
    col_lookup = {c.lower(): c for c in cols}
    for cand in candidates:
        matched = col_lookup.get(cand.lower())
        if matched is not None:
            return matched
    return None


def parse_schema_tables_and_columns(schema: str) -> dict[str, list[str]]:
    """
    Parse a formatted database schema string into a mapping of table names to column names.

    Args:
        schema: Formatted database schema string.

    Returns:
        Dict[str, List[str]]: Mapping from table name to list of column names.

    """
    return parse_schema_text_to_map(schema)


def _score_table_for_query(
    tbl_name: str,
    cols: list[str],
    lower_q: str,
    query_words: set[str],
) -> int:
    """
    Compute a relevance score for a schema table against a natural-language query.

    Args:
        tbl_name: Candidate table name.
        cols: Column names belonging to the table.
        lower_q: Lowercased natural-language query.
        query_words: Tokenized lowercase query words.

    Returns:
        int: Non-negative relevance score.

    """
    tbl_lower = tbl_name.lower()
    score = 10 if tbl_lower in lower_q else 0
    if tbl_lower in ("patients", "demographics", "tblaptsummary") and (
        "table 1" in lower_q
        or "baseline" in lower_q
        or len(query_words.intersection(TABLE_1_FALLBACK_KEYWORDS)) >= 2
    ):
        score += 6
    for w in query_words:
        if len(w) >= 3 and w in tbl_lower:
            score += 4
        score += sum(2 for c in cols if len(w) >= 3 and w == c.lower())
    return score


def _build_fallback_table_1_sql(safe_tbl: str, cols: list[str]) -> str:
    """
    Build a deterministic Table 1 baseline summary DuckDB SQL query for `safe_tbl`.

    Args:
        safe_tbl: Escaped DuckDB table identifier.
        cols: Column names present in `safe_tbl`.

    Returns:
        str: Valid DuckDB SELECT statement summarizing baseline cohort characteristics.

    """
    group_col = _find_matching_column(
        cols, ("txgroup", "treatment_group", "pt_pumpmanuf")
    )
    select_parts: list[str] = []
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


def build_treatment_group_hypo_summary_sql(
    schema_map: dict[str, list[str]],
    fallback_table: str,
) -> str | None:
    """
    Build a grouped DuckDB SQL query for treatment group, patient count, average age, and severe hypoglycemia events.

    Args:
        schema_map: Mapping of table names to column names.
        fallback_table: Default selected fallback table.

    Returns:
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
    select_exprs: list[str] = [f'"{safe_grp}" AS treatment_group']
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
    schema_map: dict[str, list[str]],
    lower_q: str,
) -> str:
    """
    Choose the best target table from `schema_map` for fallback SQL synthesis.

    Args:
        schema_map: Mapping of table names to column names.
        lower_q: Lowercased natural-language query.

    Returns:
        str: Selected table identifier.

    """
    if not schema_map:
        return "patients"

    query_words = set(re.findall(r"[a-z0-9_]+", lower_q))
    best_table: str | None = None
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
    llm_error: str | None = None,
    llm_error_code: str | None = None,
    model_name: str | None = None,
    provider: str | None = None,
    *,
    model: str | None = None,
    db_path: str | None = None,
) -> TranslationResult:
    """
    Synthesize a deterministic, schema-validated DuckDB SQL query when local LLM inference crashes.

    Used as a resilient fallback when local Ollama (`llama-server`) terminates due to
    memory allocation failure (`ggml_backend_amx_buffer_type_alloc_buffer` / OOM) or crash.

    Args:
        schema: Extracted database schema text.
        nl_query: User's natural language question.
        include_diagnostic_note: Whether to prepend a diagnostic notice explaining that local inference fell back to schema-guided SQL synthesis.
        llm_error: Optional human-readable diagnostic reason why LLM inference failed.
        llm_error_code: Optional structured error code for the LLM failure.
        model_name: Optional requested model identifier.
        provider: Optional resolved LLM provider name.
        model: Optional keyword alias for `model_name`.
        db_path: Optional path to the active DuckDB database file for live stratum `N` resolution.

    Returns:
        TranslationResult: Structured translation result (unpackable as `(markdown_response_with_sql_fence, raw_sql_query)`).

    """
    _ = provider
    effective_model = model_name if model_name is not None else model
    clean_model = (effective_model or "ollama").strip() or "ollama"
    schema_map = parse_schema_tables_and_columns(schema)
    lower_q = nl_query.lower()

    limit_match = re.search(r"\b(?:first|top|limit)\s+(\d+)\b", lower_q)
    limit_val = int(limit_match.group(1)) if limit_match else 10
    target_table = _select_best_fallback_table(schema_map, lower_q)

    safe_tbl = escape_sql_identifier(target_table)
    lower_schema_tables = {tbl.lower() for tbl in schema_map}
    curated_sql = lookup_curated_template_sql(
        nl_query, lower_schema_tables, db_path=db_path
    )
    is_t1_request = bool(
        re.search(
            r"\b(?:table\s*1|baseline\s+characteristics|cohort\s+summary|participant\s+characteristics)\b",
            lower_q,
        )
    )
    has_rich_patients = bool(
        {"patients", "jaeb_beck2023_cohort", "dclp3_ptroster"}.intersection(
            lower_schema_tables
        )
        and (
            has_custom_table1_modifiers(nl_query, schema_map=schema_map)
            or "dclp3_ptroster" in lower_schema_tables
            or "race_ethnicity"
            in {
                c.lower()
                for c in schema_map.get(target_table, schema_map.get("patients", []))
            }
        )
    )
    grouped_hypo_sql = (
        build_treatment_group_hypo_summary_sql(schema_map, target_table)
        if (
            not is_t1_request
            and (
                "grouped by" in lower_q
                or ("treatment" in lower_q and ("hypo" in lower_q or "age" in lower_q))
            )
        )
        else None
    )
    if curated_sql is not None:
        sql = curated_sql
    elif is_t1_request and has_rich_patients:
        sql = build_dynamic_table1_sql(nl_query, schema_map=schema_map, db_path=db_path)
    elif grouped_hypo_sql is not None:
        sql = grouped_hypo_sql
    elif is_t1_request:
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

    if include_diagnostic_note:
        if cleaned_err:
            note_prefix = (
                f"_Note: Local LLM inference was unavailable ({cleaned_err}); "
                f"synthesized schema-guided fallback SQL for `{safe_tbl}`._\n\n"
            )
        else:
            note_prefix = (
                f"_Note: Local LLM inference was unavailable; "
                f"synthesized schema-guided fallback SQL for `{safe_tbl}`._\n\n"
            )
    else:
        note_prefix = ""

    sql_comment = (
        f"-- Synthesized schema-guided fallback SQL for {safe_tbl} ({clean_model})"
    )
    full_resp = f"{note_prefix}```sql\n{sql_comment}\n{sql}\n```"
    return TranslationResult(
        full_response=full_resp,
        sql_query=sql,
        used_fallback=True,
        generation_mode="rule_based_fallback",
        llm_error=cleaned_err,
        llm_error_code=effective_code,
    )


_LOCALIZED_STARTER_5_PROMPTS: tuple[str, ...] = (
    "show me the first 5 patients in the demographics table.",
    "人口統計テーブルの最初の5人の患者を表示してください。",
    "أظهر لي أول 5 مرضى في جدول التركيبة السكانية.",
    "הצג לי את 5 המטופלים הראשונים בטבלת הדמוגרפיה.",
)


_BECK_2023_TABLE_1_MARKERS: tuple[str, ...] = (
    "beck",
    "control-iq",
    "control iq",
    "dclp3",
    "dclp5",
    "pedap",
    "jaeb",
    "369",
    "meta-analysis",
    "meta analysis",
)


def _is_beck_2023_table_1_query(norm_prompt: str) -> bool:
    """
    Return True when a normalized natural-language prompt requests the full-cohort Beck et al. (2023) Table 1.

    Returns False if the prompt specifies custom subset filters (e.g., single study, age range,
    HbA1c threshold), custom stratification columns (e.g., by study or sex), or custom row subsets.

    Args:
        norm_prompt: Lowercased, whitespace-normalized user query.

    Returns:
        bool: True if the query asks for the unmodified Beck/JAEB/Control-IQ pooled Table 1.

    """
    has_table1_intent = (
        "table 1" in norm_prompt
        or "table1" in norm_prompt
        or "participant characteristics at baseline" in norm_prompt
    )
    if not has_table1_intent or not any(
        marker in norm_prompt for marker in _BECK_2023_TABLE_1_MARKERS
    ):
        return False
    return not has_custom_table1_modifiers(norm_prompt)


_LEGACY_TABLE_1_PROMPT_ALIASES: dict[str, str] = {
    "summarize baseline age, sex, hba1c, and time in range for all patients.": "table1-overall",
    "create a table 1 comparing closed-loop vs control treatment arms.": "table1-by-arm",
    "make a table 1 stratified by age group (<18, 18-49, and 50+ years).": "table1-by-age",
    "build an age-stratified table 1 across pediatric (<18 yr), adult (18-49 yr), and older adult (>=50 yr) cohorts.": "table1-by-age",
    "produce a table 1 by insulin pump manufacturer (insulet, medtronic, tandem).": "table1-by-pump",
    "produce a table 1 stratified by insulin pump manufacturer (insulet, medtronic, tandem).": "table1-by-pump",
    "compare baseline and endpoint hba1c and c-peptide by treatment arm.": "table1-longitudinal-lab",
    "build a table 1 combining demographics, raw cgm sensor metrics, and adverse events by arm.": "table1-multi-domain-rollup",
    "build a table 1 combining demographics, raw cgm sensor metrics, and adverse events by treatment arm.": "table1-multi-domain-rollup",
    "generate a transposed nejm-style table 1 comparing closed-loop and control arms.": "table1-transposed-nejm",
    "make table 1 for the jaeb control-iq meta-analysis (beck et al. 2023).": "table1-beck-2023",
}


def lookup_table_1_curated_sql(
    prompt: str,
    db_path: str | None = None,
) -> str | None:
    """
    Match a prompt against the canonical Academic Table 1 benchmark suite.

    When `db_path` contains the raw JAEB study tables (`dclp3_*`, `dclp5_*`, `pedap_*`)
    and the prompt matches the Beck et al. (2023) Table 1, compiles the self-contained
    raw JAEB CTE SQL (`build_dynamic_table1_sql`) directly against the raw study tables.

    Args:
        prompt: Natural language user question.
        db_path: Optional path to the active DuckDB database file.

    Returns:
        Optional[str]: Gold DuckDB SQL if the prompt matches a Table 1 benchmark query, else None.

    """
    from t1d_analytics.clinical_query_catalog import (
        BECK_2023_TABLE_1_SQL,
        TABLE_1_BENCHMARK_SUITE,
    )
    from t1d_analytics.table1_specs import db_has_raw_jaeb_tables

    norm_prompt = " ".join(prompt.strip().lower().split())
    legacy_id = _LEGACY_TABLE_1_PROMPT_ALIASES.get(norm_prompt)
    for item in TABLE_1_BENCHMARK_SUITE:
        norm_q = " ".join(item["question"].strip().lower().split())
        if norm_prompt == norm_q or item["id"] == legacy_id:
            if item["id"] == "table1-beck-2023" and db_has_raw_jaeb_tables(db_path):
                return build_dynamic_table1_sql(prompt, db_path=db_path)
            return item["gold_sql"]
    if _is_beck_2023_table_1_query(norm_prompt):
        if db_has_raw_jaeb_tables(db_path):
            return build_dynamic_table1_sql(prompt, db_path=db_path)
        return BECK_2023_TABLE_1_SQL
    return None


def lookup_curated_template_sql(
    prompt: str,
    lower_schema_tables: set[str] | None = None,
    db_path: str | None = None,
) -> str | None:
    """
    Match a prompt against all 12 Gold Templates (5 Starter Templates + 7 Academic Table 1 Templates).

    Verifies that the required target table(s) for the matched template exist in
    `lower_schema_tables` when a non-empty schema set is supplied.

    Args:
        prompt: Natural language user question.
        lower_schema_tables: Optional set of lowercase table names present in the active database schema.
        db_path: Optional path to the active DuckDB database file.

    Returns:
        Optional[str]: Gold DuckDB SQL if matched and schema-compatible, else None.

    """
    from t1d_analytics.clinical_query_catalog import STARTER_TEMPLATE_SUITE

    t1_sql = lookup_table_1_curated_sql(prompt, db_path=db_path)
    if t1_sql is not None:
        if (
            not lower_schema_tables
            or "patients" in lower_schema_tables
            or "dclp3_ptroster" in lower_schema_tables
        ):
            return t1_sql
        return None

    norm_prompt = " ".join(prompt.strip().lower().split())
    if norm_prompt in _LOCALIZED_STARTER_5_PROMPTS and (
        not lower_schema_tables or "demographics" in lower_schema_tables
    ):
        return "SELECT * FROM demographics LIMIT 5;"

    starter_req_tables: dict[str, tuple[str, ...]] = {
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


def is_recoverable_local_ollama_crash(provider: str, exc: Exception) -> bool:
    """
    Return True if `exc` is a local Ollama server crash, timeout, missing model tag, or OOM allocation failure.

    Args:
        provider: Resolved LLM provider name.
        exc: Exception raised by the underlying transport.

    Returns:
        bool: True when `provider == 'ollama'` and `exc` indicates a server crash/OOM/disconnect/timeout.

    """
    if provider.lower() != "ollama":
        return False
    if isinstance(
        exc,
        (
            LlmConnectionError,
            urllib.error.URLError,
            TimeoutError,
            ConnectionError,
            OSError,
        ),
    ):
        return True
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
    "FALLBACK_DIAGNOSTIC_PREFIX",
    "TABLE_1_FALLBACK_KEYWORDS",
    "GenerationModeLiteral",
    "TranslationResult",
    "extract_fallback_error_from_text",
    "format_llm_failure_reason",
    "is_fallback_response_text",
    "is_recoverable_local_ollama_crash",
    "lookup_curated_template_sql",
    "lookup_table_1_curated_sql",
    "parse_schema_tables_and_columns",
    "synthesize_fallback_clinical_sql",
]
