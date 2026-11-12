"""Pytest fixtures and configuration."""

import sys
import types
from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock

import duckdb
import pytest


@pytest.fixture
def mock_db(tmp_path: Path) -> str:
    """Fixture to provide a test DuckDB database path populated with sample users."""
    db_path = str(tmp_path / "test.duckdb")
    conn = duckdb.connect(db_path)
    conn.execute("CREATE TABLE users (id INTEGER, name VARCHAR)")
    conn.execute("INSERT INTO users VALUES (1, 'Alice'), (2, 'Bob')")
    conn.close()
    return db_path


class MockAnyLLMModule(types.ModuleType):
    """Typed mock module representing the any_llm package."""

    AnyLLM: MagicMock


@pytest.fixture(autouse=True)
def _mock_any_llm(
    monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory
) -> Generator[None, None, None]:
    """Ensure a default mock any_llm module and isolated sessions DB are registered during tests."""
    mock_module = MockAnyLLMModule("any_llm")
    mock_module.AnyLLM = MagicMock()
    monkeypatch.setitem(sys.modules, "any_llm", mock_module)
    monkeypatch.delenv("T1D_DB_PATH", raising=False)
    monkeypatch.delenv("T1D_DB_SEARCH_DIR", raising=False)
    monkeypatch.delenv("T1D_DB_SEARCH_PATH", raising=False)
    monkeypatch.delenv("T1D_JAEB_DIR", raising=False)
    monkeypatch.delenv("T1D_DB_DIR", raising=False)
    monkeypatch.delenv("T1D_SESSIONS_DB", raising=False)
    basetemp = tmp_path_factory.getbasetemp()
    monkeypatch.setenv("T1D_EXTRA_DB_ROOTS", str(basetemp))
    default_sess_dir = basetemp / "default_sessions"
    default_sess_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv(
        "T1D_SESSIONS_DB_PATH", str(default_sess_dir / "sessions.duckdb")
    )
    from t1d_analytics.api import app
    from t1d_analytics.rate_limiter import (
        RATE_LIMIT_STORE,
        BaseRateLimiter,
        get_rate_limiter,
    )

    RATE_LIMIT_STORE.clear()
    get_rate_limiter().reset()
    app_limiter = getattr(app.state, "rate_limiter", None)
    if isinstance(app_limiter, BaseRateLimiter):
        app_limiter.reset()
    yield
