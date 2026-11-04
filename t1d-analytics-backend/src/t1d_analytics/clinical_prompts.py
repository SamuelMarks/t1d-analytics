"""Clinical Text-to-SQL prompt templates and database schema compaction utilities."""

from typing import List

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


build_system_prompt = build_clinical_sql_prompt

__all__ = [
    "MULTI_TRIAL_UNION_SECTION",
    "STRICT_SCHEMA_SECTION",
    "TABLE_1_RULES_SECTION",
    "TERMINOLOGY_RULES_SECTION",
    "build_clinical_sql_prompt",
    "build_system_prompt",
    "compact_schema_for_prompt",
]
