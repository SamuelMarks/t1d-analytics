"""Unit tests for sanitized LLM failure diagnostics and schema-guided SQL fallback synthesis."""

from t1d_analytics.sql_fallback_synthesizer import (
    format_llm_failure_reason,
    synthesize_fallback_clinical_sql,
)
from t1d_analytics.sql_utils import extract_sql_from_response


def test_format_llm_failure_reason_sanitizes_segfault_messages() -> None:
    """Verify format_llm_failure_reason sanitizes raw llama-server segmentation fault messages into a concise CPU runner diagnostic."""
    raw_exc = RuntimeError(
        "llama-server process has terminated: signal: segmentation fault (core dumped) (status code: 500)"
    )
    msg, code = format_llm_failure_reason(raw_exc, model="gemma:2b")
    assert msg == "Local model 'gemma:2b' was unavailable on CPU runner"
    assert code == "backend.llmMemoryCrash"
    assert "segmentation fault" not in (msg or "")
    assert "core dumped" not in (msg or "")

    msg_default, code_default = format_llm_failure_reason(raw_exc)
    assert msg_default == "Local model 'ollama' was unavailable on CPU runner"
    assert code_default == "backend.llmMemoryCrash"


def test_synthesize_fallback_clinical_sql_comment_header_and_extraction() -> None:
    """Verify synthesize_fallback_clinical_sql includes the clean SQL comment header inside the sql fence and extract_sql_from_response strips it cleanly."""
    schema = """Table: patients
Columns: subject_id (INT), age (DOUBLE), TxGroup (VARCHAR), hba1c (DOUBLE), tir (DOUBLE), tbr (DOUBLE), tar (DOUBLE), mean_glucose (DOUBLE), numhospdka (INT)
"""
    res = synthesize_fallback_clinical_sql(
        schema=schema,
        nl_query="How many patients are in the dataset?",
        include_diagnostic_note=True,
        llm_error="Local model 'gemma:2b' was unavailable on CPU runner",
        llm_error_code="backend.llmMemoryCrash",
        model="gemma:2b",
    )
    assert res.used_fallback is True
    assert res.generation_mode == "rule_based_fallback"
    assert (
        "-- Synthesized schema-guided fallback SQL for patients (gemma:2b)"
        in res.full_response
    )
    assert "segmentation fault" not in res.full_response
    assert res.sql_query == 'SELECT COUNT(*) AS count FROM "patients";'
    assert (
        extract_sql_from_response(res.full_response)
        == 'SELECT COUNT(*) AS count FROM "patients";'
    )
