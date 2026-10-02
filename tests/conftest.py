"""Pytest fixtures and configuration."""

import sys
import types
from typing import Generator
from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def _mock_any_llm() -> Generator[None, None, None]:
    if "any_llm" not in sys.modules:
        mock_module = types.ModuleType("any_llm")
        setattr(mock_module, "AnyLLM", MagicMock())
        sys.modules["any_llm"] = mock_module
        yield
        del sys.modules["any_llm"]
    else:
        yield
