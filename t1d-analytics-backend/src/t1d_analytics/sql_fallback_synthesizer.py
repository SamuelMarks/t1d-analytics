"""Rule-based clinical SQL fallback synthesis and translation result structures."""

import re
from typing import Dict, Iterator, List, Literal, Optional, Set, Tuple, Union

from pydantic.dataclasses import dataclass

from t1d_analytics.sql_utils import escape_sql_identifier, parse_schema_text_to_map

GenerationModeLiteral = Literal["llm", "rule_based_fallback", "literal_sql"]

_FALLBACK_DIAGNOSTIC_PREFIX = "_Note: Local LLM inference"

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

    Supports 2-tuple unpacking `(full_response, sql_query)` for backward compatibility
    with callers expecting `Tuple[str, str]`, while carrying explicit typed metadata
    about whether rule-based fallback synthesis was used.

    Attributes
    ----------
        full_response: Complete assistant markdown text (including SQL fence).
        sql_query: Extracted executable DuckDB SQL statement.
        used_fallback: True if rule-based SQL synthesis was used instead of an LLM.
        generation_mode: One of `'llm'`, `'rule_based_fallback'`, or `'literal_sql'`.

    """

    full_response: str
    sql_query: str
    used_fallback: bool = False
    generation_mode: GenerationModeLiteral = "llm"

    @property
    def is_rule_based_sql(self) -> bool:
        """
        Return True when SQL was synthesized via rule-based fallback rather than an LLM.

        Returns
        -------
            bool: True if `used_fallback` is True or `generation_mode == 'rule_based_fallback'`.

        """
        return self.used_fallback or self.generation_mode == "rule_based_fallback"

    def __iter__(self) -> Iterator[str]:
        """
        Yield `full_response` and `sql_query` to support 2-tuple unpacking.

        Yields
        ------
            str: First `full_response`, then `sql_query`.

        """
        yield self.full_response
        yield self.sql_query


def coerce_translation_result(
    raw: Union[TranslationResult, Tuple[str, str]],
) -> TranslationResult:
    """
    Normalize a `TranslationResult` or legacy `(full_response, sql_query)` tuple into a `TranslationResult`.

    Args:
    ----
        raw: Either a `TranslationResult` instance or a 2-tuple `(full_response, sql_query)`.

    Returns:
    -------
        TranslationResult: Normalized `TranslationResult` with `used_fallback` and `generation_mode` populated.

    """
    if isinstance(raw, TranslationResult):
        return raw
    full_response, sql_query = raw
    used_fallback = is_fallback_response_text(full_response)
    return TranslationResult(
        full_response=full_response,
        sql_query=sql_query,
        used_fallback=used_fallback,
        generation_mode="rule_based_fallback" if used_fallback else "llm",
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

    Returns:
    -------
        TranslationResult: Structured translation result (unpackable as `(markdown_response_with_sql_fence, raw_sql_query)`).

    """
    schema_map = _parse_schema_tables_and_columns(schema)
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
    return TranslationResult(
        full_response=full_resp,
        sql_query=sql,
        used_fallback=True,
        generation_mode="rule_based_fallback",
    )


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
    "_find_matching_column",
    "_is_recoverable_local_ollama_crash",
    "_parse_schema_tables_and_columns",
    "_score_table_for_query",
    "_select_best_fallback_table",
    "coerce_translation_result",
    "is_fallback_response_text",
    "synthesize_fallback_clinical_sql",
]
