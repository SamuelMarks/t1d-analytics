"""Unit tests for Ollama discovery, CPU-safe runner options, and multi-candidate fallback ranking."""

import pytest

from t1d_analytics.ollama_discovery import (
    build_safe_ollama_options,
    resolve_local_ollama_fallback_candidates,
)


def test_resolve_local_ollama_fallback_candidates() -> None:
    """Verify resolve_local_ollama_fallback_candidates returns distinct lightweight fallback models in priority order excluding requested_model."""
    assert resolve_local_ollama_fallback_candidates("gemma:2b", []) == []

    installed = [
        "gemma2:27b",
        "phi4-mini:latest",
        "gemma:2b",
        "llama3.2:1b",
        "phi4-mini:latest",
        "   ",
        "ollama/llama3.2:1b",
    ]
    cands = resolve_local_ollama_fallback_candidates("ollama/gemma:2b", installed)
    assert cands == ["llama3.2:1b", "phi4-mini:latest"]

    cands_unmatched = resolve_local_ollama_fallback_candidates(
        "uninstalled:model",
        ["gemma:2b", "llama3.2:1b", "phi4-mini", "gemma4:26b"],
    )
    assert cands_unmatched == ["gemma:2b", "llama3.2:1b", "phi4-mini"]


def test_build_safe_ollama_options_use_mmap_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify build_safe_ollama_options includes use_mmap=False by default and respects explicit overrides."""
    monkeypatch.delenv("OLLAMA_NUM_CTX", raising=False)
    monkeypatch.delenv("OLLAMA_NUM_GPU", raising=False)
    monkeypatch.delenv("OLLAMA_NUM_THREAD", raising=False)

    opts = build_safe_ollama_options()
    assert opts["use_mmap"] is False
    assert opts["num_ctx"] == 2048
    assert opts["num_gpu"] == 0
    assert opts["num_thread"] == 8

    opts_mmap_true = build_safe_ollama_options(
        num_ctx=1024, num_gpu=0, num_thread=4, use_mmap=True
    )
    assert opts_mmap_true["use_mmap"] is True
    assert opts_mmap_true["num_ctx"] == 1024
    assert opts_mmap_true["num_thread"] == 4
