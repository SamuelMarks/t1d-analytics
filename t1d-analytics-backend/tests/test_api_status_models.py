"""Tests for FastAPI status, readiness, models, providers, database discovery, and lifespan endpoints."""

import json
import os
import types
import typing
import urllib.error
import urllib.request
from pathlib import Path
from unittest.mock import MagicMock, patch

import duckdb
import pytest
from fastapi.testclient import TestClient

from t1d_analytics.api import app

client = TestClient(app)


def test_list_models_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test successful model listing from local Ollama."""

    class MockResponse:
        """Mock HTTP response object."""

        def read(self) -> bytes:
            """Return mock JSON byte payload."""
            return json.dumps(
                {
                    "models": [
                        {"name": "gemma4", "size": 12345},
                        {"name": "llama3", "size": 67890},
                    ]
                }
            ).encode()

    class MockUrlopen:
        """Mock context manager for urllib urlopen."""

        def __init__(
            self,
            req: typing.Union[str, urllib.request.Request],
            timeout: typing.Optional[float] = None,
        ) -> None:
            """Initialize MockUrlopen."""
            self.req = req

        def __enter__(self) -> MockResponse:
            """Enter the context manager."""
            return MockResponse()

        def __exit__(
            self,
            exc_type: typing.Optional[type],
            exc_val: typing.Optional[BaseException],
            exc_tb: typing.Optional[types.TracebackType],
        ) -> None:
            """Exit the context manager."""
            pass

    with patch("urllib.request.urlopen", MockUrlopen):
        response = client.get("/api/models")
        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        assert len(data["models"]) == 8
        assert data["models"][0]["name"] == "gemma4"
        assert data["models"][0]["provider"] == "ollama"
        assert data["models"][1]["name"] == "llama3"
        assert any(
            m["name"] == "openai/gpt-4o" and m["provider"] == "openai"
            for m in data["models"]
        )

        resp_local_only = client.get("/api/models?include_cloud=false")
        assert resp_local_only.status_code == 200
        assert len(resp_local_only.json()["models"]) == 2


def test_list_models_url_error() -> None:
    """Test when Ollama is unreachable via URLError."""

    def mock_urlopen(
        req: typing.Union[str, urllib.request.Request],
        timeout: typing.Optional[float] = None,
    ) -> None:
        """Simulate urlopen throwing URLError."""
        raise urllib.error.URLError("Connection refused")

    with patch("urllib.request.urlopen", mock_urlopen):
        response = client.get("/api/models?include_cloud=false")
        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        assert len(data["models"]) == 0

        resp_cloud = client.get("/api/models?include_cloud=true")
        assert resp_cloud.status_code == 200
        data_cloud = resp_cloud.json()
        assert len(data_cloud["models"]) > 0
        for m in data_cloud["models"]:
            assert "available" in m
            assert "reachable" in m


def test_list_models_general_error() -> None:
    """Test when an unexpected exception occurs during Ollama fetch."""

    def mock_urlopen(
        req: typing.Union[str, urllib.request.Request],
        timeout: typing.Optional[float] = None,
    ) -> None:
        """Simulate urlopen throwing a generic exception."""
        raise Exception("Unexpected boom")

    with patch("urllib.request.urlopen", mock_urlopen):
        response = client.get("/api/models?include_cloud=false")
        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        assert len(data["models"]) == 0


@patch("urllib.request.urlopen")
def test_list_models_with_ollama_host(mock_urlopen: MagicMock) -> None:
    """Test list_models respects OLLAMA_HOST env variable."""
    mock_response = MagicMock()
    mock_response.read.return_value = json.dumps(
        {"models": [{"name": "gemma4-custom", "size": 12345}]}
    ).encode("utf-8")
    mock_urlopen.return_value.__enter__.return_value = mock_response

    with patch.dict(os.environ, {"OLLAMA_HOST": "remote-ollama:11434"}):
        response = client.get("/api/models")
        assert response.status_code == 200
        data = response.json()
        assert any(m["name"] == "gemma4-custom" for m in data["models"])


def test_global_exception_handler() -> None:
    """Test that unhandled exceptions are caught by the global exception handler."""
    from fastapi import APIRouter

    router = APIRouter()

    @router.get("/error_endpoint_for_test")
    def error_endpoint() -> None:
        """Endpoint designed to raise an unhandled exception."""
        raise Exception("Trigger unhandled exception")

    app.include_router(router)

    test_client = TestClient(app, raise_server_exceptions=False)
    response = test_client.get(
        "/error_endpoint_for_test", headers={"Accept-Language": "ja-JP,ja;q=0.9"}
    )
    assert response.status_code == 500
    data = response.json()
    assert "detail" in data
    assert data["detail"]["error_code"] == "backend.serverError"
    assert "Trigger unhandled exception" in data["detail"]["params"]["error"]


@pytest.mark.anyio
async def test_lifespan() -> None:
    """Test lifespan startup diagnostics execution."""
    from t1d_analytics.api import lifespan

    with patch("t1d_analytics.api.log_startup_diagnostics") as mock_diag:
        async with lifespan(app):
            pass
        mock_diag.assert_called_once()


def test_get_status_endpoint(mock_db: str, tmp_path: Path) -> None:
    """Test /api/status, /api/ready, and /api/health endpoints with populated and empty databases."""
    response = client.get("/api/status", headers={"Accept-Language": "ja"})
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "total_rows" in data
    assert "required_tables_present" in data
    assert "database" in data
    assert "ollama" in data

    empty_db = str(tmp_path / "empty_health.duckdb")
    conn_empty = duckdb.connect(empty_db)
    conn_empty.close()
    resp_health_empty = client.get(f"/api/health?db_path={empty_db}")
    assert resp_health_empty.status_code == 503
    resp_ready_empty = client.get(f"/api/ready?db_path={empty_db}")
    assert resp_ready_empty.status_code == 503

    resp_health_ok = client.get(f"/api/health?db_path={mock_db}")
    assert resp_health_ok.status_code == 200
    assert resp_health_ok.json()["database"]["details"]["total_rows"] == 2
    resp_ready_ok = client.get(f"/api/ready?db_path={mock_db}")
    assert resp_ready_ok.status_code == 200
    assert resp_ready_ok.json()["status"] == "ready"
    assert resp_ready_ok.json()["total_rows"] == 2


def test_get_status_with_db_path(mock_db: str) -> None:
    """Test get_status with explicit valid db_path."""
    resp = client.get(f"/api/status?db_path={mock_db}")
    assert resp.status_code == 200
    data = resp.json()
    assert "status" in data
    assert data["database"]["connected"] is True


def test_list_databases_endpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test /api/databases endpoint returns database list and current DB."""
    db1 = tmp_path / "custom.duckdb"
    conn = duckdb.connect(str(db1))
    conn.execute("CREATE TABLE t (id INT)")
    conn.close()

    monkeypatch.chdir(tmp_path)
    resp1 = client.get("/api/databases")
    assert resp1.status_code == 200

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "custom.duckdb").symlink_to(db1)

    resp2 = client.get("/api/databases")
    assert resp2.status_code == 200
    data = resp2.json()
    assert "databases" in data
    assert "current_db" in data
    assert any(d["name"] == "custom.duckdb" for d in data["databases"])

    monkeypatch.setenv("T1D_DB_PATH", str(db1))
    resp3 = client.get("/api/databases")
    assert resp3.status_code == 200
    assert any(d["name"] == "custom.duckdb" for d in resp3.json()["databases"])

    monkeypatch.setenv("T1D_DB_PATH", str(tmp_path / "absent.duckdb"))
    resp4 = client.get("/api/databases")
    assert resp4.status_code == 200

    monkeypatch.setenv("T1D_DB_PATH", "")
    resp5 = client.get("/api/databases")
    assert resp5.status_code == 200


def test_validate_db_path_null_byte() -> None:
    """Test validate_db_path rejects null byte injection."""
    from t1d_analytics.api import validate_db_path
    from t1d_analytics.errors import InvalidDatabasePathError

    with pytest.raises(InvalidDatabasePathError) as exc_info:
        validate_db_path("test" + chr(0) + ".duckdb")
    assert exc_info.value.status_code == 400
    assert exc_info.value.error_code == "backend.invalidDatabasePath"


def test_validate_db_path_traversal() -> None:
    """Test validate_db_path rejects paths outside allowed directory containment."""
    from t1d_analytics.api import validate_db_path
    from t1d_analytics.errors import InvalidDatabasePathError

    allowed = [Path("/mock/allowed/dir")]
    with pytest.raises(InvalidDatabasePathError) as exc_info:
        validate_db_path("/etc/passwd", allowed_dirs=allowed)
    assert exc_info.value.status_code == 400
    assert exc_info.value.error_code == "backend.invalidDatabasePath"


def test_validate_db_path_env_var(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test validate_db_path respects T1D_DB_PATH environment variable."""
    from t1d_analytics.api import validate_db_path
    from t1d_analytics.diagnostics import check_database_health

    env_db = tmp_path / "env_test.duckdb"
    env_db.touch()
    monkeypatch.setenv("T1D_DB_PATH", str(env_db))
    resolved = validate_db_path()
    assert resolved == str(env_db.resolve())

    resolved_fallback = validate_db_path("t1d.duckdb")
    assert resolved_fallback == str(env_db.resolve())

    monkeypatch.setenv("T1D_DB_PATH", str(tmp_path / "nonexistent.duckdb"))
    resolved_no_fallback = validate_db_path("t1d.duckdb")
    assert resolved_no_fallback == str(Path("t1d.duckdb").resolve())

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("T1D_DB_PATH", raising=False)
    data_sub = tmp_path / "data"
    data_sub.mkdir(exist_ok=True)
    (data_sub / "t1d.duckdb").write_text("dummy")
    resolved_data_fb = validate_db_path("t1d.duckdb")
    assert resolved_data_fb == str((data_sub / "t1d.duckdb").resolve())
    disc_res = client.get("/api/databases").json()
    assert disc_res["current_db"] == "data/t1d.duckdb"

    with patch("duckdb.connect", side_effect=RuntimeError("mock")):
        hlth = check_database_health(db_path=None)
        assert hlth.configured_path == "data/t1d.duckdb"


def test_get_providers_status_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test /api/providers/status reports health and credential readiness for all providers."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test12345")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test12345")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-test-key")

    class MockResp:
        """Mock HTTP response."""

        status = 200

        def __enter__(self) -> "MockResp":
            """Enter context."""
            return self

        def __exit__(
            self,
            exc_type: typing.Optional[type] = None,
            exc_val: typing.Optional[BaseException] = None,
            exc_tb: typing.Optional[types.TracebackType] = None,
        ) -> None:
            """Exit context."""
            pass

    with patch("urllib.request.urlopen", return_value=MockResp()):
        resp = client.get("/api/providers/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "providers" in data
        provs = {p["provider"]: p for p in data["providers"]}
        assert provs["ollama"]["healthy"] is True
        assert provs["ollama"]["reachable"] is True
        assert provs["ollama"]["error_code"] == "PROVIDER_ONLINE"
        assert provs["openai"]["configured"] is True
        assert provs["openai"]["error_code"] == "PROVIDER_ONLINE"
        assert provs["anthropic"]["configured"] is True
        assert provs["google"]["configured"] is True

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    with patch("urllib.request.urlopen", side_effect=Exception("Connection refused")):
        resp = client.get("/api/providers/status")
        assert resp.status_code == 200
        data = resp.json()
        provs = {p["provider"]: p for p in data["providers"]}
        assert provs["ollama"]["healthy"] is False
        assert provs["ollama"]["reachable"] is False
        assert provs["ollama"]["error_code"] == "PROVIDER_OFFLINE"
        assert provs["openai"]["configured"] is False
        assert provs["openai"]["error_code"] == "MISSING_API_KEY"
        assert provs["anthropic"]["configured"] is False
        assert provs["google"]["configured"] is False


def test_is_provider_configured_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test is_provider_configured handles known and unknown providers and environment keys."""
    from t1d_analytics.api import is_provider_configured

    assert is_provider_configured("ollama") is True

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert is_provider_configured("openai") is False
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    assert is_provider_configured("openai") is True

    assert is_provider_configured("unknown_provider_xyz") is False


def test_get_providers_status_unprefixed_host_and_non_200(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test get_providers_status with unprefixed OLLAMA_HOST and non-200 HTTP status."""
    monkeypatch.setenv("OLLAMA_HOST", "localhost:11434")

    class MockStatus500:
        """Mock HTTP 500 response."""

        status = 500

        def __enter__(self) -> "MockStatus500":
            """Enter context."""
            return self

        def __exit__(
            self,
            exc_type: typing.Optional[type] = None,
            exc_val: typing.Optional[BaseException] = None,
            exc_tb: typing.Optional[types.TracebackType] = None,
        ) -> None:
            """Exit context."""
            pass

    with patch("urllib.request.urlopen", return_value=MockStatus500()):
        resp = client.get("/api/providers/status")
        assert resp.status_code == 200
        data = resp.json()
        provs = {p["provider"]: p for p in data["providers"]}
        assert provs["ollama"]["healthy"] is False


def test_debug_static_mount() -> None:
    """Test static mount behavior in debug mode."""
    import t1d_analytics.api

    with (
        patch.dict(os.environ, {"DEBUG": "1"}),
        patch("t1d_analytics.api.app.mount") as mock_mount,
        patch("t1d_analytics.api.logger.info") as mock_info,
        patch("t1d_analytics.api.StaticFiles"),
        patch("os.path.exists", return_value=True),
    ):
        t1d_analytics.api._setup_debug()
        mock_mount.assert_called_once()
        mock_info.assert_called_once()

    with (
        patch.dict(os.environ, {"DEBUG": "1"}),
        patch("t1d_analytics.api.logger.warning") as mock_warn,
        patch("os.path.exists", return_value=False),
    ):
        t1d_analytics.api._setup_debug()
        mock_warn.assert_called_once()


def test_lifespan_auto_ingestion_branches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test lifespan context manager auto-seeding when DB is missing/empty and when already populated."""
    import asyncio

    from t1d_analytics.api import lifespan

    data_dir = tmp_path / "lifespan_data"
    data_dir.mkdir(parents=True)
    db_file = tmp_path / "lifespan.duckdb"
    sess_file = tmp_path / "lifespan_sessions.duckdb"

    monkeypatch.setenv("T1D_DB_PATH", str(db_file))
    monkeypatch.setenv("T1D_DATA_DIR", str(data_dir))
    monkeypatch.setenv("T1D_SESSIONS_DB_PATH", str(sess_file))

    async def _run_lifespan() -> None:
        """Execute async lifespan context."""
        async with lifespan(app):
            pass

    asyncio.run(_run_lifespan())
    assert db_file.exists()

    asyncio.run(_run_lifespan())

    bad_db = tmp_path / "bad_lifespan.duckdb"
    monkeypatch.setenv("T1D_DB_PATH", str(bad_db))
    with patch(
        "t1d_analytics.api.seed_database", side_effect=RuntimeError("Seed error")
    ):
        asyncio.run(_run_lifespan())


def test_providers_status_and_models_with_any_llm_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test /api/providers/status and /api/models when additional any-llm providers and custom models are configured."""
    monkeypatch.setenv("GROQ_API_KEY", "gsk-configured")
    monkeypatch.setenv("T1D_ANY_LLM_MODELS", "deepseek/deepseek-chat")

    resp_status = client.get("/api/providers/status")
    assert resp_status.status_code == 200
    provs = {p["provider"]: p for p in resp_status.json()["providers"]}
    assert "groq" in provs
    assert provs["groq"]["configured"] is True

    resp_models = client.get(
        "/api/models?configured_models=mistral/mistral-large-latest"
    )
    assert resp_models.status_code == 200
    model_names = [m["name"] for m in resp_models.json()["models"]]
    assert "groq/llama-3.3-70b-versatile" in model_names
    assert "deepseek/deepseek-chat" in model_names
    assert "mistral/mistral-large-latest" in model_names

    with patch(
        "t1d_analytics.api.CLOUD_PROVIDER_MODELS",
        {"groq": ["groq/custom-llama"]},
    ):
        resp_override = client.get("/api/models")
        assert resp_override.status_code == 200
        override_names = [m["name"] for m in resp_override.json()["models"]]
        assert "groq/custom-llama" in override_names
