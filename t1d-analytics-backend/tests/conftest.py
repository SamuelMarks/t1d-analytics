"""Pytest fixtures and configuration."""

import sys
import types
from typing import Generator
from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def _mock_any_llm(
    monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory
) -> Generator[None, None, None]:
    """Ensure a default mock any_llm module and isolated sessions DB are registered during tests."""
    mock_module = types.ModuleType("any_llm")
    setattr(mock_module, "AnyLLM", MagicMock())
    monkeypatch.setitem(sys.modules, "any_llm", mock_module)
    monkeypatch.delenv("T1D_DB_PATH", raising=False)
    monkeypatch.delenv("T1D_SESSIONS_DB", raising=False)
    default_sess_dir = tmp_path_factory.getbasetemp() / "default_sessions"
    default_sess_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv(
        "T1D_SESSIONS_DB_PATH", str(default_sess_dir / "sessions.duckdb")
    )
    yield
