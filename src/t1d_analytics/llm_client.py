"""Centralized LLM provider configuration, prompt construction, and streaming helpers."""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

PROVIDER_ENV_KEYS: Dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GEMINI_API_KEY",
    "gemini": "GEMINI_API_KEY",
}

CLOUD_MODEL_CATALOG: Dict[str, List[str]] = {
    "openai": ["openai/gpt-4o", "openai/gpt-4o-mini"],
    "anthropic": ["anthropic/claude-3-5-sonnet", "anthropic/claude-3-5-haiku"],
    "google": ["google/gemini-1.5-pro", "google/gemini-1.5-flash"],
}


def normalize_ollama_host(default_host: str = "http://127.0.0.1:11434") -> str:
    """
    Resolve and normalize the Ollama host URL from the OLLAMA_HOST environment variable.

    Args:
    ----
        default_host: Fallback host URL if OLLAMA_HOST is not set.

    Returns:
    -------
        str: Normalized base URL starting with http:// or https:// and without trailing slash.

    """
    host = os.environ.get("OLLAMA_HOST", default_host).rstrip("/")
    if not host.startswith(("http://", "https://")):
        host = f"http://{host}"
    return host


def resolve_provider_and_model(
    model_name: str, provider: Optional[str] = None
) -> Tuple[str, str]:
    """
    Resolve the LLM provider identifier and model name from a composite model string.

    Args:
    ----
        model_name: Model identifier or 'provider/model' string (e.g., 'openai/gpt-4o').
        provider: Optional explicit provider override.

    Returns:
    -------
        Tuple[str, str]: A tuple of (resolved_provider, actual_model).

    """
    resolved_provider = provider or "ollama"
    actual_model = model_name
    if "/" in model_name and not provider:
        resolved_provider, actual_model = model_name.split("/", 1)
    return resolved_provider, actual_model


def is_provider_configured(provider: str) -> bool:
    """
    Check if the specified LLM provider has credentials configured in the environment.

    Args:
    ----
        provider: Provider identifier string ('ollama', 'openai', 'anthropic', 'google').

    Returns:
    -------
        bool: True if configured, False otherwise.

    """
    prov = provider.lower()
    if prov == "ollama":
        return True
    if prov == "openai":
        return bool(os.environ.get("OPENAI_API_KEY"))
    if prov == "anthropic":
        return bool(os.environ.get("ANTHROPIC_API_KEY"))
    if prov in ("google", "gemini"):
        return bool(
            os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        )
    return False


def get_effective_api_key(
    provider: str, api_key: Optional[str] = None
) -> Optional[str]:
    """
    Resolve the effective API key for a cloud provider from explicit argument or environment.

    Args:
    ----
        provider: Provider identifier string.
        api_key: Optional explicit API key override.

    Returns:
    -------
        Optional[str]: Resolved API key or None.

    """
    if api_key:
        return api_key
    prov = provider.lower()
    if prov in PROVIDER_ENV_KEYS:
        return os.environ.get(PROVIDER_ENV_KEYS[prov])
    return None


def extract_stream_delta(chunk: Any, provider: str = "ollama") -> str:
    """
    Extract token delta string from provider-specific streaming response chunk.

    Accurately inspects delta objects across ollama, openai, anthropic, and google.

    Args:
    ----
        chunk: Raw response chunk from any-llm or provider SDK.
        provider: Provider identifier ('ollama', 'openai', 'anthropic', 'google').

    Returns:
    -------
        str: Extracted token content string, or empty string.

    """
    if isinstance(chunk, str):
        return chunk
    if isinstance(chunk, dict):
        for choice in chunk.get("choices") or []:
            if isinstance(choice, dict):
                delta = choice.get("delta")
                if isinstance(delta, dict) and delta.get("content"):
                    return str(delta["content"])
                if choice.get("text"):
                    return str(choice["text"])
        msg = chunk.get("message")
        if isinstance(msg, dict) and msg.get("content"):
            return str(msg["content"])
        if chunk.get("response"):
            return str(chunk["response"])
        if chunk.get("text"):
            return str(chunk["text"])
        return ""

    prov = provider.lower()
    if prov in ("google", "gemini"):
        for cand in getattr(chunk, "candidates", None) or []:
            content = getattr(cand, "content", None)
            for part in getattr(content, "parts", None) or []:
                text_val = getattr(part, "text", None)
                if isinstance(text_val, str) and text_val:
                    return text_val
        text_attr = getattr(chunk, "text", None)
        if isinstance(text_attr, str) and text_attr:
            return text_attr
        return ""

    if prov == "anthropic":
        delta_obj = getattr(chunk, "delta", None)
        if delta_obj is not None:
            text_val = getattr(delta_obj, "text", None) or getattr(
                delta_obj, "content", None
            )
            if isinstance(text_val, str) and text_val:
                return text_val
        for c in getattr(chunk, "choices", None) or []:
            c_delta = getattr(c, "delta", None)
            text_val = getattr(c_delta, "text", None)
            if isinstance(text_val, str) and text_val:
                return text_val
        return ""

    for c in getattr(chunk, "choices", None) or []:
        c_delta = getattr(c, "delta", None)
        if c_delta is not None:
            text_val = getattr(c_delta, "content", None) or getattr(
                c_delta, "text", None
            )
            if isinstance(text_val, str) and text_val:
                return text_val
        c_msg = getattr(c, "message", None)
        if c_msg is not None:
            text_val = getattr(c_msg, "content", None)
            if isinstance(text_val, str) and text_val:
                return text_val
        text_val = getattr(c, "text", None)
        if isinstance(text_val, str) and text_val:
            return text_val

    direct_text = getattr(chunk, "text", None)
    if isinstance(direct_text, str) and direct_text:
        return direct_text

    return ""


def build_clinical_sql_prompt(
    schema: str,
    nl_query: str,
    history_context: str = "",
    require_markdown_fence: bool = True,
) -> str:
    """
    Construct the clinical Text-to-SQL system prompt for an LLM.

    Args:
    ----
        schema: The database schema text.
        nl_query: The clinician or user query.
        history_context: Optional formatted prior conversation history block.
        require_markdown_fence: Whether to instruct the LLM to wrap SQL in markdown fences.

    Returns:
    -------
        str: Formatted system and user prompt.

    """
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

    return f"""You are a DuckDB SQL expert. Given the following database schema for Type 1 Diabetes (T1D) clinical trial datasets:

{schema}

Context & Rules for T1D Analytics:
1. Terminology & Aliases:
   - "Lows" / "Hypoglycemia": Look for Time Below Range (TBR, <70 or <54 mg/dL), `pct_time_under_70`, `hypo`.
   - "Highs" / "Hyperglycemia": Look for Time Above Range (TAR, >180 or >250 mg/dL), `pct_time_over_180`, `hyper`.
   - "In-range" / "TIR": Look for Time in Range (70-180 mg/dL), `pct_time_in_range_70_180`, `tir`, `cgm_in_range`.
   - "A1C" / "HbA1c": Look for glycated hemoglobin, `a1c`, `hba1c`.
   - "Variability": Look for Coefficient of Variation (`cv`), Standard Deviation (`sd`).
   - "Average sugar": Look for Mean Glucose, `mean_cgm`, Glucose Management Indicator (`gmi`).
   - "Adverse Events": "DKA" (Diabetic Ketoacidosis), "SH" (Severe Hypoglycemia).
   - Demographics: "Duration" (years with diabetes), "Age", "BMI", "Gender/Sex", "Weight".
   - Treatment: "Pump" (CSII), "Injections/Shots" (MDI), "Loop" (AID/Closed-Loop), "Insulin" (TDD, Basal, Bolus).
   - Study Timepoints: "Baseline" (visit 0 or baseline flag), "End of trial/Follow-up" (max visit month).
2. SQL Generation Rules for Multi-Trial Data:
   - Trial Names: Acronyms (DCLP3, DCLP5, Pedap, etc.) usually correspond directly to table names.
   - Aggregating Combined Trials: To calculate metrics for multiple trials "together", "combined", or "overall", you MUST use a CTE with `UNION ALL` to combine raw rows from all requested tables *before* applying aggregate functions like `AVG()`.
   - Comparing Individual Trials: To calculate metrics for multiple trials "individually" or "each", use `UNION ALL` with an identifying literal string column (e.g., `SELECT 'DCLP3' as trial, AVG(...) ... UNION ALL SELECT 'DCLP5'...`).
   - Aligning UNION Columns: Clinical trial tables often have different schemas. When using `UNION ALL`, you MUST explicitly `SELECT` and alias only the specific columns needed (e.g. `age`, `tir`) to ensure they align perfectly in number, order, and type across all blocks. NEVER use `SELECT *` with `UNION ALL`.
   - Linking Data: If questions span patient demographics and CGM outcomes, use `JOIN` on the patient/subject ID columns (e.g., `pt_id`, `subject_id`).
3. Formatting & Logic:
   - Handle colloquial/informal language intuitively (e.g., "betwixt" -> `BETWEEN`, "kids/pediatric" -> `age < 18`, "adults" -> `age >= 18`).
   - Account for string vs. numeric data types; use `CAST(col AS FLOAT)` if computing averages on textual columns that contain numbers.
   - Ignore NULL values appropriately when calculating averages or sums.
4. Strict Schema Enforcement:
   - CRITICAL: You must ONLY use the exact table names and column names provided in the schema above.
   - Do NOT invent, guess, or hallucinate table names (e.g., do not use "Visit1" or "demographics" if they are not in the schema).
   - If a requested concept (like "demographics") is not a table, find the relevant columns within the existing tables (e.g., `visits` or `patients`).

{formatting_instruction}

{history_context}User request: {nl_query}"""
