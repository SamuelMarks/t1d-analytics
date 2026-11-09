"""Clinical Text-to-SQL prompt templates and database schema compaction utilities."""

import re
from typing import List, Optional, Set, Tuple

_COMPACT_LOCAL_MODEL_MARKERS: Tuple[str, ...] = (
    "ollama",
    "gemma:2b",
    "gemma2:2b",
    "gemma3:1b",
    "gemma3:4b",
    "gemma4",
    "llama3.2:1b",
    "llama3.2:3b",
    "qwen2.5-coder:1.5b",
    "qwen2.5-coder:3b",
    "phi4-mini",
    "phi3:mini",
)

_CANONICAL_CLINICAL_VIEW_NAMES: Set[str] = {
    "patients",
    "demographics",
    "tblaptsummary",
    "subjects",
    "cgm_readings",
    "hba1c",
    "hba1c_labs",
    "adverseevents",
    "dka_adverse_events",
}

_PRIORITY_CLINICAL_COLUMNS: Set[str] = {
    "ptid",
    "subject_id",
    "patient_id",
    "txgroup",
    "treatment_group",
    "gender",
    "sex",
    "ageasofranddt",
    "age",
    "numsevhypo",
    "severe_hypo_events",
    "severe_hypoglycemia_events",
    "pt_pumpmanuf",
    "hba1c",
    "c_peptide",
    "visitdate",
    "visit",
    "tir",
    "tbr",
    "tar",
    "mean_glucose",
    "glucose_value",
    "numhospdka",
    "dka_events",
    "event",
    "eventtype",
    "severity",
    "hospitalized",
    "onset_date",
}

TERMINOLOGY_RULES_SECTION: str = """1. Terminology & Aliases:
   - "Lows" / "Hypoglycemia": Look for Time Below Range (TBR, <70 or <54 mg/dL), `pct_time_under_70`, `hypo`.
   - "Highs" / "Hyperglycemia": Look for Time Above Range (TAR, >180 or >250 mg/dL), `pct_time_over_180`, `hyper`.
   - "In-range" / "TIR": Look for Time in Range (70-180 mg/dL), `pct_time_in_range_70_180`, `tir`, `cgm_in_range`.
   - "A1C" / "HbA1c": Look for glycated hemoglobin, `a1c`, `hba1c`.
   - "Variability": Look for Coefficient of Variation (`cv`), Standard Deviation (`sd`).
   - "Average sugar": Look for Mean Glucose, `mean_cgm`, Glucose Management Indicator (`gmi`).
   - "Adverse Events": "DKA" (Diabetic Ketoacidosis), "SH" (Severe Hypoglycemia).
   - Demographics: "Duration" (years with diabetes), "Age", "BMI", "Gender/Sex", "Weight".
   - Treatment: "Pump" (CSII), "Injections/Shots" (MDI), "Loop" (AID/Closed-Loop), "Insulin" (TDD, Basal, Bolus).
   - Study Timepoints: "Baseline" (visit 0 or baseline flag), "End of trial/Follow-up" (max visit month)."""

MULTI_TRIAL_UNION_SECTION: str = """2. SQL Generation Rules for Multi-Trial Data:
   - Trial Names: Acronyms (DCLP3, DCLP5, Pedap, etc.) usually correspond directly to table names.
   - Aggregating Combined Trials: To calculate metrics for multiple trials "together", "combined", or "overall", you MUST use a CTE with `UNION ALL` to combine raw rows from all requested tables *before* applying aggregate functions like `AVG()`.
   - Comparing Individual Trials: To calculate metrics for multiple trials "individually" or "each", use `UNION ALL` with an identifying literal string column (e.g., `SELECT 'DCLP3' as trial, AVG(...) ... UNION ALL SELECT 'DCLP5'...`).
   - Aligning UNION Columns: Clinical trial tables often have different schemas. When using `UNION ALL`, you MUST explicitly `SELECT` and alias only the specific columns needed (e.g. `age`, `tir`) to ensure they align perfectly in number, order, and type across all blocks. NEVER use `SELECT *` with `UNION ALL`.
   - Linking Data: If questions span patient demographics and CGM outcomes, use `JOIN` on the patient/subject ID columns (e.g., `pt_id`, `subject_id`)."""

STRICT_SCHEMA_SECTION: str = """3. Formatting & Logic:
   - Handle colloquial/informal language intuitively (e.g., "betwixt" -> `BETWEEN`, "kids/pediatric" -> `age < 18`, "adults" -> `age >= 18`).
   - Account for string vs. numeric data types; use `CAST(col AS FLOAT)` if computing averages on textual columns that contain numbers.
   - Ignore NULL values appropriately when calculating averages or sums.
4. Strict Schema Enforcement:
   - CRITICAL: You must ONLY use the exact table names and column names provided in the schema above.
   - Do NOT invent, guess, or hallucinate table names (e.g., do not use "Visit1" or "demographics" if they are not in the schema).
   - If a requested concept (like "demographics") is not a table, find the relevant columns within the existing tables (e.g., `visits` or `patients`)."""

TABLE_1_RULES_SECTION: str = """5. Academic "Table 1" (Baseline Participant Characteristics & Stratified Outcomes) Rules:
   - Trigger Phrases: When asked for `"Table 1"`, `"baseline characteristics"`, `"cohort summary table"`, `"CONSORT baseline table"`, or `"stratified demographic and glycemic table"`, produce a publication-ready DuckDB summary query.
   - Parametric Continuous Metrics (Mean ± SD): Use `ROUND(AVG(col), 1) || ' ± ' || ROUND(STDDEV_SAMP(col), 1)`.
   - Non-Parametric Skewed Metrics (Median (IQR)): Use `ROUND(MEDIAN(col), 2) || ' (' || ROUND(QUANTILE_CONT(col, 0.25), 2) || '-' || ROUND(QUANTILE_CONT(col, 0.75), 2) || ')'`.
   - Categorical Proportions (n (%)): Use `SUM(CASE WHEN cond THEN 1 ELSE 0 END) || ' (' || ROUND(100.0 * AVG(CASE WHEN cond THEN 1.0 ELSE 0.0 END), 1) || '%)'`.
   - Stratified Arms + Overall Rollup: Use `GROUP BY ROLLUP(TxGroup)` and `COALESCE(TxGroup, 'Overall Cohort') AS "Cohort Arm"` to include both treatment arms and the overall cohort.
   - Transposed Publication Table 1: When asked for a transposed NEJM/Lancet-style Table 1 where rows are clinical characteristics and columns are study arms (`"Characteristic"`, `"Closed-Loop Arm"`, `"Control Arm"`, `"Overall Cohort"`, `"Arm Difference"`), use a `UNION ALL` of characteristic rows with explicit `CAST(... AS VARCHAR)` alignment and `ORDER BY sort_ord`."""


def compact_schema_for_prompt(
    schema: str,
    max_tables: int = 8,
    max_sample_chars: int = 120,
) -> str:
    """
    Compact a verbose database schema string for fast CPU prompt evaluation on small local Ollama models.

    Truncates long `Sample row:` lines to `max_sample_chars` and limits the number of full table
    blocks to `max_tables` while preserving the `-- Relevant Schema Context:` header.

    Args:
    ----
        schema: Raw or RAG-enriched database schema string.
        max_tables: Maximum number of detailed `Table:` blocks to include.
        max_sample_chars: Maximum character length per `Sample row:` line.

    Returns:
    -------
        str: Compacted schema string.

    """
    if not schema.strip():
        return schema
    nl = chr(10)
    blocks = schema.split(f"{nl}{nl}")
    out_blocks: List[str] = []
    table_count = 0
    for block in blocks:
        stripped = block.strip()
        if not stripped:
            continue
        if stripped.startswith("Table:"):
            if table_count >= max_tables:
                continue
            table_count += 1
        lines: List[str] = []
        for line in stripped.splitlines():
            if line.strip().startswith("Sample row:") and len(line) > max_sample_chars:
                lines.append(line[:max_sample_chars] + "...")
            else:
                lines.append(line)
        out_blocks.append(nl.join(lines))
    return f"{nl}{nl}".join(out_blocks)


def build_clinical_sql_prompt(
    schema: str,
    nl_query: str,
    history_context: str = "",
    require_markdown_fence: bool = True,
    compact_schema: bool = False,
) -> str:
    """
    Construct the clinical Text-to-SQL system prompt for an LLM.

    Args:
    ----
        schema: The database schema text.
        nl_query: The clinician or user query.
        history_context: Optional formatted prior conversation history block.
        require_markdown_fence: Whether to instruct the LLM to wrap SQL in markdown fences.
        compact_schema: Whether to compact sample rows and table count for fast local CPU inference.

    Returns:
    -------
        str: Formatted system and user prompt.

    """
    effective_schema = compact_schema_for_prompt(schema) if compact_schema else schema
    formatting_instruction = (
        """Write a SQL query that answers the user's request.
You may provide an explanation of your thought process before or after the query.
Ensure the SQL query is enclosed in a markdown code block (e.g. ```sql ... ```).
The query should be a valid DuckDB SQL SELECT statement."""
        if require_markdown_fence
        else """Write a SQL query that answers the user's request. 
You may provide an explanation of your thought process as SQL comments (`--`) before the query.
Return ONLY the raw SQL query, with no markdown formatting and no code blocks.
The query should be a valid DuckDB SQL SELECT statement."""
    )

    rules_block = chr(10).join(
        (
            TERMINOLOGY_RULES_SECTION,
            MULTI_TRIAL_UNION_SECTION,
            STRICT_SCHEMA_SECTION,
            TABLE_1_RULES_SECTION,
        )
    )

    return f"""You are a DuckDB SQL expert. Given the following database schema for Type 1 Diabetes (T1D) clinical trial datasets:

{effective_schema}

Context & Rules for T1D Analytics:
{rules_block}

{formatting_instruction}

{history_context}User request: {nl_query}"""


def is_compact_local_model(
    model_name: Optional[str],
    provider: Optional[str] = None,
) -> bool:
    """
    Detect whether the target model or provider is a lightweight local CPU model.

    Args:
    ----
        model_name: Optional model identifier string (e.g. `'gemma:2b'`, `'ollama:gemma:2b'`).
        provider: Optional explicit provider name (e.g. `'ollama'`).

    Returns:
    -------
        bool: True if the model or provider targets local CPU Ollama or a small (<4B) local model.

    """
    prov_lower = (provider or "").strip().lower()
    if prov_lower == "ollama":
        return True
    if not model_name:
        return False
    norm = model_name.strip().lower()
    if norm in ("sql", ""):
        return False
    if norm.startswith(("ollama:", "ollama/")):
        return True
    if any(
        norm.startswith((f"{cloud_prov}/", f"{cloud_prov}:"))
        for cloud_prov in (
            "openai",
            "anthropic",
            "google",
            "gemini",
            "groq",
            "mistral",
            "deepseek",
            "cohere",
            "together",
            "fireworks",
            "openrouter",
        )
    ):
        return False
    return any(marker in norm for marker in _COMPACT_LOCAL_MODEL_MARKERS)


def _prune_table_columns_line(
    line: str,
    max_cols_per_table: int,
    query_words: Optional[Set[str]] = None,
) -> str:
    """
    Limit the number of columns in a `'Columns: ...'` schema line to `max_cols_per_table`.

    Prioritizes canonical T1D clinical columns (`PtID`, `Pt_PumpManuf`, `TxGroup`, `NumSevHypo`,
    `HbA1c`, `tir`, `severity`, `hospitalized`, etc.) and columns matching `query_words` so wide
    Jaeb tables (such as the 334-column `subjects` table) retain their key analytical columns.

    Args:
    ----
        line: Schema line beginning with `'Columns:'`.
        max_cols_per_table: Maximum number of column definitions to retain.
        query_words: Optional set of lowercase query keywords to boost column retention.

    Returns:
    -------
        str: Pruned `'Columns: ...'` line.

    """
    prefix = "Columns:"
    stripped = line.strip()
    if not stripped.startswith(prefix) or max_cols_per_table <= 0:
        return line
    raw_cols = [c.strip() for c in stripped[len(prefix) :].split(",") if c.strip()]
    if len(raw_cols) <= max_cols_per_table:
        return line
    ranked_indices: List[Tuple[int, int]] = []
    for idx, col_spec in enumerate(raw_cols):
        col_ident = col_spec.split("(", 1)[0].strip().lower()
        is_prio = col_ident in _PRIORITY_CLINICAL_COLUMNS or bool(
            query_words and any(w in col_ident for w in query_words)
        )
        ranked_indices.append((0 if is_prio else 1, idx))
    ranked_indices.sort()
    chosen_indices = sorted(idx for _, idx in ranked_indices[:max_cols_per_table])
    kept = ", ".join(raw_cols[i] for i in chosen_indices)
    return f"Columns: {kept}"


def _score_canonical_domain_boost(tbl_name: str, q_lower: str) -> int:
    """
    Compute domain-specific relevance boost for canonical T1D tables given `q_lower`.

    Args:
    ----
        tbl_name: Lowercase table or view name.
        q_lower: Lowercase user question.

    Returns:
    -------
        int: Additional relevance score boost.

    """
    boost = 0
    if ("pump" in q_lower or "manufacturer" in q_lower) and tbl_name == "subjects":
        boost += 60
    if (
        "adverse" in q_lower or "occurrence" in q_lower or "severity" in q_lower
    ) and tbl_name in ("adverseevents", "dka_adverse_events"):
        boost += 60
    if (
        "hypo" in q_lower
        or "treatment group" in q_lower
        or "subject summaries" in q_lower
    ) and tbl_name in ("tblaptsummary", "patients", "demographics"):
        boost += 60
    if (
        "hba1c" in q_lower or "lab" in q_lower or "c-peptide" in q_lower
    ) and tbl_name in ("hba1c", "hba1c_labs"):
        boost += 60
    if "demographics" in q_lower and tbl_name == "demographics":
        boost += 80
    return boost


def build_compact_clinical_sql_prompt(
    schema_context: str,
    user_query: str,
    history_context: str = "",
    max_tables: int = 6,
    max_cols_per_table: int = 14,
) -> str:
    """
    Build a concise clinical Text-to-SQL prompt optimized for fast local CPU Ollama prefill (<5s).

    Prunes `schema_context` to canonical clinical views (`patients`, `demographics`,
    `tblaptsummary`, `subjects`, `cgm_readings`, `hba1c`, `hba1c_labs`, `adverseevents`,
    `dka_adverse_events`) plus top query-matched tables (`max_tables=6`,
    `max_cols_per_table=14`) and uses a compact DuckDB SQL instruction block.

    Args:
    ----
        schema_context: Raw or RAG-enriched database schema string.
        user_query: Natural language clinical question.
        history_context: Optional formatted prior conversation history block.
        max_tables: Maximum number of table blocks to include.
        max_cols_per_table: Maximum number of columns to keep per table block.

    Returns:
    -------
        str: Compact clinical SQL prompt string.

    """
    nl = chr(10)
    q_lower = user_query.lower()
    query_words = {w for w in re.findall(r"[a-z0-9_]+", q_lower) if len(w) >= 3}
    raw_blocks = [b.strip() for b in schema_context.split(f"{nl}{nl}") if b.strip()]
    header_blocks: List[str] = []
    scored_tables: List[Tuple[int, int, str]] = []

    for idx, block in enumerate(raw_blocks):
        if not block.startswith("Table:"):
            header_blocks.append(block)
            continue
        first_line = block.splitlines()[0]
        tbl_name = first_line[len("Table:") :].strip().lower()
        score = 0
        if tbl_name in _CANONICAL_CLINICAL_VIEW_NAMES:
            score += 50
        if tbl_name in q_lower:
            score += 40
        score += _score_canonical_domain_boost(tbl_name, q_lower)
        block_lower = block.lower()
        score += sum(5 for w in query_words if w in block_lower)
        scored_tables.append((-score, idx, block))

    scored_tables.sort()
    selected_blocks = [b for _, _, b in scored_tables[: max(1, max_tables)]]
    compacted_blocks: List[str] = list(header_blocks)
    for block in selected_blocks:
        lines: List[str] = []
        for line in block.splitlines():
            s_line = line.strip()
            if s_line.startswith("Columns:"):
                lines.append(
                    _prune_table_columns_line(
                        s_line, max_cols_per_table, query_words=query_words
                    )
                )
            elif s_line.startswith("Sample row:") and len(s_line) > 90:
                lines.append(s_line[:90] + "...")
            else:
                lines.append(line)
        compacted_blocks.append(nl.join(lines))

    pruned_schema = (
        f"{nl}{nl}".join(compacted_blocks)
        if compacted_blocks
        else compact_schema_for_prompt(schema_context, max_tables=max_tables)
    )

    return (
        f"You are a DuckDB SQL expert for Type 1 Diabetes (T1D) clinical data.{nl}{nl}"
        f"Schema:{nl}{pruned_schema}{nl}{nl}"
        f"Rules:{nl}"
        f"- Output ONLY a single valid read-only DuckDB SELECT (or WITH ... SELECT) statement.{nl}"
        f"- NEVER output CREATE TABLE, INSERT, UPDATE, DELETE, or DDL statements; a Table 1 is a SELECT query.{nl}"
        f"- Use ONLY exact table and column names from the Schema above.{nl}"
        f"- Prefer canonical tables (`patients`, `tblaptsummary`, `demographics`, `subjects`, `cgm_readings`, `hba1c`, `hba1c_labs`, `adverseevents`, `dka_adverse_events`) when present.{nl}"
        f"- Wrap VARCHAR numeric columns in TRY_CAST(col AS DOUBLE) for AVG/SUM.{nl}"
        f"- Return only the SQL query inside ```sql and ``` fences without leading 'sql' or '...' text.{nl}{nl}"
        f"{history_context}User request: {user_query}"
    )


build_system_prompt = build_clinical_sql_prompt

__all__ = [
    "MULTI_TRIAL_UNION_SECTION",
    "STRICT_SCHEMA_SECTION",
    "TABLE_1_RULES_SECTION",
    "TERMINOLOGY_RULES_SECTION",
    "build_clinical_sql_prompt",
    "build_compact_clinical_sql_prompt",
    "build_system_prompt",
    "compact_schema_for_prompt",
    "is_compact_local_model",
]
