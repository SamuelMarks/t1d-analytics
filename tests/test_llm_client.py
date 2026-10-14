"""Unit tests for centralized LLM client utilities in t1d_analytics.llm_client."""

import os
from types import SimpleNamespace
from unittest.mock import patch

from t1d_analytics.llm_client import (
    build_clinical_sql_prompt,
    extract_stream_delta,
    get_effective_api_key,
    is_provider_configured,
    normalize_ollama_host,
    resolve_provider_and_model,
)


def test_normalize_ollama_host() -> None:
    """Test normalize_ollama_host with and without scheme."""
    with patch.dict(os.environ, {"OLLAMA_HOST": "127.0.0.1:11434/"}):
        assert normalize_ollama_host() == "http://127.0.0.1:11434"
    with patch.dict(os.environ, {"OLLAMA_HOST": "https://ollama.example.com/"}):
        assert normalize_ollama_host() == "https://ollama.example.com"


def test_resolve_provider_and_model_and_keys() -> None:
    """Test provider/model resolution and API key lookup."""
    assert resolve_provider_and_model("gemma4") == ("ollama", "gemma4")
    assert resolve_provider_and_model("openai/gpt-4o") == ("openai", "gpt-4o")
    assert resolve_provider_and_model("openai/gpt-4o", provider="custom") == (
        "custom",
        "openai/gpt-4o",
    )

    assert is_provider_configured("ollama") is True
    assert is_provider_configured("unknown_provider") is False
    with patch.dict(
        os.environ,
        {
            "OPENAI_API_KEY": "sk-1",
            "ANTHROPIC_API_KEY": "sk-2",
            "GOOGLE_API_KEY": "sk-3",
        },
        clear=True,
    ):
        assert is_provider_configured("openai") is True
        assert is_provider_configured("anthropic") is True
        assert is_provider_configured("google") is True
        assert get_effective_api_key("openai") == "sk-1"
        assert get_effective_api_key("openai", api_key="explicit") == "explicit"
        assert get_effective_api_key("ollama") is None


def test_extract_stream_delta_all_providers() -> None:
    """Test extract_stream_delta across dict and object shapes for all providers."""
    assert extract_stream_delta("plain") == "plain"
    assert extract_stream_delta({"choices": [{"delta": {"content": "d1"}}]}) == "d1"
    assert extract_stream_delta({"choices": [{"text": "t1"}]}) == "t1"
    assert extract_stream_delta({"choices": ["non-dict"]}) == ""
    assert extract_stream_delta({"message": {"content": "m1"}}) == "m1"
    assert extract_stream_delta({"response": "r1"}) == "r1"
    assert extract_stream_delta({"text": "tx1"}) == "tx1"
    assert extract_stream_delta({}) == ""

    # Google object chunks
    g_part = SimpleNamespace(
        candidates=[
            SimpleNamespace(content=SimpleNamespace(parts=[SimpleNamespace(text="g1")]))
        ]
    )
    assert extract_stream_delta(g_part, "google") == "g1"
    g_text = SimpleNamespace(candidates=[], text="g2")
    assert extract_stream_delta(g_text, "gemini") == "g2"
    g_empty = SimpleNamespace(
        candidates=[
            SimpleNamespace(content=SimpleNamespace(parts=[SimpleNamespace(text="")]))
        ],
        text="",
    )
    assert extract_stream_delta(g_empty, "google") == ""

    # Anthropic object chunks
    a_delta = SimpleNamespace(delta=SimpleNamespace(text="a1"))
    assert extract_stream_delta(a_delta, "anthropic") == "a1"
    a_choice = SimpleNamespace(
        delta=SimpleNamespace(text=None, content=None),
        choices=[SimpleNamespace(delta=SimpleNamespace(text="a2"))],
    )
    assert extract_stream_delta(a_choice, "anthropic") == "a2"
    a_empty = SimpleNamespace(
        delta=None,
        choices=[SimpleNamespace(delta=SimpleNamespace(text=""))],
    )
    assert extract_stream_delta(a_empty, "anthropic") == ""

    # OpenAI / Ollama object chunks
    o_delta = SimpleNamespace(
        choices=[SimpleNamespace(delta=SimpleNamespace(content="o1"))]
    )
    assert extract_stream_delta(o_delta, "openai") == "o1"
    o_msg = SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(content="", text=""),
                message=SimpleNamespace(content="o2"),
            )
        ]
    )
    assert extract_stream_delta(o_msg, "openai") == "o2"
    o_choice_text = SimpleNamespace(
        choices=[SimpleNamespace(delta=None, message=None, text="o3")]
    )
    assert extract_stream_delta(o_choice_text, "openai") == "o3"
    o_direct = SimpleNamespace(
        choices=[
            SimpleNamespace(delta=None, message=SimpleNamespace(content=""), text="")
        ],
        text="o4",
    )
    assert extract_stream_delta(o_direct, "openai") == "o4"
    assert extract_stream_delta(SimpleNamespace(choices=[], text=""), "openai") == ""


def test_build_clinical_sql_prompt() -> None:
    """Test clinical prompt builder with and without markdown fence instructions."""
    p_md = build_clinical_sql_prompt(
        "Table: p", "How many?", require_markdown_fence=True
    )
    assert "markdown code block" in p_md
    p_raw = build_clinical_sql_prompt(
        "Table: p", "How many?", require_markdown_fence=False
    )
    assert "Return ONLY the raw SQL query" in p_raw
