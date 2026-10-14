"""Pytest fixtures and configuration."""

import sys
import types
from typing import Generator
from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def _mock_any_llm(monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
    """Ensure a default mock any_llm module is registered during tests."""
    mock_module = types.ModuleType("any_llm")
    setattr(mock_module, "AnyLLM", MagicMock())
    monkeypatch.setitem(sys.modules, "any_llm", mock_module)
    yield
