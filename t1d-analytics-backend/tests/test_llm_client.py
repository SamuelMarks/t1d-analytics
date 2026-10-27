"""Unit tests for centralized LLM client utilities in t1d_analytics.llm_client."""

import os
from types import SimpleNamespace
from typing import Optional, Sequence
from unittest.mock import patch

from t1d_analytics.llm_client import (
    _AnyLlmModelProtocol,
    _CandidateProtocol,
    _extract_delta_or_message_str,
    build_clinical_sql_prompt,
    discover_configured_any_llm_models,
    extract_stream_delta,
    get_effective_api_key,
    is_mapping_chunk,
    is_provider_configured,
    is_sequence_of_choices,
    normalize_ollama_host,
    parse_configured_models_string,
    resolve_provider_and_model,
)


class _EmptyChunk:
    """Chunk object implementing GeminiStreamChunkProtocol with no candidates, delta, choices, or text."""

    candidates: Optional[Sequence[_CandidateProtocol]] = None


def test_normalize_ollama_host() -> None:
    """Test normalize_ollama_host with and without scheme."""
    with patch.dict(os.environ, {"OLLAMA_HOST": "127.0.0.1:11434/"}):
        assert normalize_ollama_host() == "http://127.0.0.1:11434"
    with patch.dict(os.environ, {"OLLAMA_HOST": "https://ollama.example.com/"}):
        assert normalize_ollama_host() == "https://ollama.example.com"


def test_resolve_provider_and_model_and_keys() -> None:
    """Test provider/model resolution and API key lookup across any-llm formats."""
    assert resolve_provider_and_model("gemma4") == ("ollama", "gemma4")
    assert resolve_provider_and_model("gemma4:26b") == ("ollama", "gemma4:26b")
    assert resolve_provider_and_model("openai/gpt-4o") == ("openai", "gpt-4o")
    assert resolve_provider_and_model("groq:llama-3.3-70b-versatile") == (
        "groq",
        "llama-3.3-70b-versatile",
    )
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
            "GROQ_API_KEY": "gsk-4",
        },
        clear=True,
    ):
        assert is_provider_configured("openai") is True
        assert is_provider_configured("anthropic") is True
        assert is_provider_configured("google") is True
        assert is_provider_configured("groq") is True
        assert get_effective_api_key("openai") == "sk-1"
        assert get_effective_api_key("google") == "sk-3"
        assert get_effective_api_key("groq") == "gsk-4"
        assert get_effective_api_key("openai", api_key="explicit") == "explicit"
        assert get_effective_api_key("ollama") is None
        assert get_effective_api_key("mistral") is None
        assert get_effective_api_key("huggingface") is None


def test_parse_and_discover_configured_any_llm_models() -> None:
    """Test parsing and discovering configured Mozilla any-llm models."""
    parsed = parse_configured_models_string(
        "groq/llama-3.3-70b, mistral:mistral-large-latest, , groq/llama-3.3-70b, gemma4"
    )
    assert parsed == [
        ("groq", "groq/llama-3.3-70b"),
        ("mistral", "mistral/mistral-large-latest"),
        ("ollama", "gemma4"),
    ]

    with patch.dict(
        os.environ,
        {
            "GROQ_API_KEY": "gsk-test",
            "T1D_ANY_LLM_MODELS": "deepseek/deepseek-chat, openai/gpt-4o, groq/llama-3.3-70b-versatile",
        },
        clear=True,
    ):
        models = discover_configured_any_llm_models(
            extra_configured_models="together:meta-llama/Llama-3-70b",
        )
        names = [m[1] for m in models]
        assert "openai/gpt-4o" in names
        assert "groq/llama-3.3-70b-versatile" in names
        assert "deepseek/deepseek-chat" in names
        assert "together/meta-llama/Llama-3-70b" in names

        def _fake_list_models(
            provider: str, api_key: Optional[str] = None
        ) -> Sequence[_AnyLlmModelProtocol]:
            """Return fake SDK models for testing."""
            assert provider == "groq"
            assert api_key == "gsk-test"
            return [
                SimpleNamespace(id="llama3-groq-tool-use"),
                SimpleNamespace(id="groq/llama3-groq-tool-use"),
                SimpleNamespace(id="groq/llama-3.3-70b-versatile"),
            ]

        sdk_discovered = discover_configured_any_llm_models(
            list_models_fn=_fake_list_models
        )
        sdk_names = [m[1] for m in sdk_discovered]
        assert "groq/llama3-groq-tool-use" in sdk_names

        def _failing_list_models(
            provider: str, api_key: Optional[str] = None
        ) -> Sequence[_AnyLlmModelProtocol]:
            """Raise RuntimeError to test fallback."""
            raise RuntimeError(f"sdk err for {provider}")

        fallback_discovered = discover_configured_any_llm_models(
            list_models_fn=_failing_list_models
        )
        assert any(m[0] == "groq" for m in fallback_discovered)


def test_extract_stream_delta_all_providers() -> None:
    """Test extract_stream_delta across dict and object shapes for all providers."""
    assert extract_stream_delta("plain") == "plain"
    assert extract_stream_delta({"choices": [{"delta": {"content": "d1"}}]}) == "d1"
    assert extract_stream_delta({"choices": [{"text": "t1"}]}) == "t1"
    assert extract_stream_delta({"choices": [{}, "non-dict"]}) == ""
    assert extract_stream_delta({"message": {"content": "m1"}}) == "m1"
    assert extract_stream_delta({"response": "r1"}) == "r1"
    assert extract_stream_delta({"text": "tx1"}) == "tx1"
    assert extract_stream_delta({}) == ""

    # Google object chunks
    assert is_mapping_chunk({"a": 1}) is True
    assert is_mapping_chunk("not-a-map") is False
    assert is_sequence_of_choices([{"text": "x"}]) is True
    assert is_sequence_of_choices("not-a-choice-seq") is False
    assert _extract_delta_or_message_str(None) == ""

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
            SimpleNamespace(content=None),
            SimpleNamespace(content=SimpleNamespace(parts=[SimpleNamespace(text="")])),
        ],
        text="",
    )
    assert extract_stream_delta(g_empty, "google") == ""

    # Empty / non-matching protocol chunks for Google, Anthropic, OpenAI
    empty_obj = _EmptyChunk()
    assert extract_stream_delta(empty_obj, "google") == ""
    assert extract_stream_delta(SimpleNamespace(text=None), "google") == ""
    assert extract_stream_delta(empty_obj, "anthropic") == ""
    assert extract_stream_delta(empty_obj, "openai") == ""

    g_non_matching_nested = SimpleNamespace(
        candidates=[
            empty_obj,
            SimpleNamespace(content=empty_obj),
            SimpleNamespace(content=SimpleNamespace(parts=[empty_obj])),
        ],
        text=None,
    )
    assert extract_stream_delta(g_non_matching_nested, "google") == ""

    # Anthropic object chunks
    a_delta = SimpleNamespace(delta=SimpleNamespace(text="a1"))
    assert extract_stream_delta(a_delta, "anthropic") == "a1"
    a_choice = SimpleNamespace(
        delta=SimpleNamespace(text=None, content=None),
        choices=[SimpleNamespace(delta=SimpleNamespace(text="a2"))],
    )
    assert extract_stream_delta(a_choice, "anthropic") == "a2"
    a_empty = SimpleNamespace(
        delta=empty_obj,
        choices=[
            empty_obj,
            SimpleNamespace(delta=None),
            SimpleNamespace(delta=empty_obj),
            SimpleNamespace(delta=SimpleNamespace(text="")),
        ],
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
            empty_obj,
            SimpleNamespace(
                delta=empty_obj,
                message=empty_obj,
                text="",
            ),
            SimpleNamespace(delta=None, message=SimpleNamespace(content=""), text=""),
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


def test_complete_llm_prompt_and_stream_llm_factory() -> None:
    """Test complete_llm_prompt and stream_llm_completion_tokens with custom llm_factory and non-choice responses."""
    from typing import Mapping

    from t1d_analytics.chat_service import stream_llm_completion_tokens
    from t1d_analytics.llm_client import (
        LlmTransportProtocol,
        complete_llm_prompt,
    )

    class _DirectTextTransport:
        """Transport returning a direct string response rather than choices."""

        def completion(
            self,
            *,
            model: str,
            messages: Sequence[Mapping[str, str]],
            stream: bool = False,
        ) -> str:
            """Return direct SQL string."""
            _ = (model, messages, stream)
            return "  SELECT 1;  "

    def _make_transport(
        provider: str, api_key: Optional[str] = None
    ) -> LlmTransportProtocol:
        """Create a _DirectTextTransport instance."""
        _ = (provider, api_key)
        return _DirectTextTransport()

    res_direct = complete_llm_prompt(
        prompt="Count",
        model_name="openai/gpt-4o",
        api_key="sk-test",
        llm_factory=_make_transport,
    )
    assert res_direct == "SELECT 1;"

    tokens_with_key = list(
        stream_llm_completion_tokens(
            prompt="Count",
            model_name="openai/gpt-4o",
            api_key="sk-test",
            llm_factory=_make_transport,
        )
    )
    assert tokens_with_key == ["  SELECT 1;  "]

    tokens_without_key = list(
        stream_llm_completion_tokens(
            prompt="Count",
            model_name="ollama/gemma4",
            llm_factory=_make_transport,
        )
    )
    assert tokens_without_key == ["  SELECT 1;  "]
