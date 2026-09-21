"""Tests for the diagnostics and health validation module."""

import json
import logging
import typing
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, patch

import duckdb
import pytest

from t1d_analytics.diagnostics import (
    DatabaseStatusCode,
    check_database_health,
    check_ollama_health,
    get_system_health,
    log_startup_diagnostics,
)


def test_check_database_health_missing_file(tmp_path: Path) -> None:
    """Test check_database_health when file does not exist."""
    missing_file = str(tmp_path / "nonexistent.duckdb")
    status = check_database_health(missing_file)
    assert not status.exists
    assert not status.connected
    assert status.status_code == DatabaseStatusCode.MISSING_FILE
    assert "does not exist" in status.message
    assert status.remediation is not None
    assert status.to_dict()["status_code"] == "missing_file"


def test_check_database_health_empty_file(tmp_path: Path) -> None:
    """Test check_database_health when file exists but is 0 bytes."""
    empty_file = tmp_path / "empty.duckdb"
    empty_file.touch()
    status = check_database_health(str(empty_file))
    assert status.exists
    assert not status.connected
    assert status.status_code == DatabaseStatusCode.EMPTY_FILE
    assert "0 bytes" in status.message
    assert status.remediation is not None


def test_check_database_health_os_error(tmp_path: Path) -> None:
    """Test check_database_health when stat raises OSError."""
    target_file = tmp_path / "locked.duckdb"
    target_file.touch()
    with patch.object(Path, "stat", side_effect=OSError("Permission denied")):
        status = check_database_health(str(target_file))
        assert status.status_code == DatabaseStatusCode.UNREADABLE
        assert "Cannot access" in status.message


def test_check_database_health_connect_exception(tmp_path: Path) -> None:
    """Test check_database_health when duckdb.connect fails."""
    corrupt_file = tmp_path / "corrupt.duckdb"
    corrupt_file.write_text("corrupt content")
    with patch("duckdb.connect", side_effect=Exception("Corrupt database")):
        status = check_database_health(str(corrupt_file))
        assert status.status_code == DatabaseStatusCode.UNREADABLE
        assert "Failed to open DuckDB database" in status.message


def test_check_database_health_empty_db(tmp_path: Path) -> None:
    """Test check_database_health when database connects but has no tables."""
    db_file = str(tmp_path / "empty_db.duckdb")
    conn = duckdb.connect(db_file)
    conn.close()

    status = check_database_health(db_file)
    assert status.exists
    assert status.connected
    assert status.status_code == DatabaseStatusCode.EMPTY_DB
    assert status.table_count == 0
    assert not status.has_initial_data
    assert "0 tables" in status.message


def test_check_database_health_missing_initial_data(tmp_path: Path) -> None:
    """Test check_database_health when database has tables but lacks T1D trial tables."""
    db_file = str(tmp_path / "custom.duckdb")
    conn = duckdb.connect(db_file)
    conn.execute("CREATE TABLE my_custom_table (id INT)")
    conn.close()

    status = check_database_health(db_file)
    assert status.exists
    assert status.connected
    assert status.status_code == DatabaseStatusCode.MISSING_INITIAL_DATA
    assert status.table_count == 1
    assert "my_custom_table" in status.tables
    assert not status.has_initial_data


def test_check_database_health_healthy(tmp_path: Path) -> None:
    """Test check_database_health when database has standard clinical tables."""
    db_file = str(tmp_path / "healthy.duckdb")
    conn = duckdb.connect(db_file)
    conn.execute("CREATE TABLE patients (id INT, age INT)")
    conn.execute("CREATE TABLE cgms (id INT, glucose FLOAT)")
    conn.close()

    status = check_database_health(db_file)
    assert status.exists
    assert status.connected
    assert status.status_code == DatabaseStatusCode.HEALTHY
    assert status.table_count == 2
    assert status.has_initial_data
    assert status.remediation is None


def test_check_database_health_japanese(tmp_path: Path) -> None:
    """Test check_database_health with Japanese localization."""
    missing_file = str(tmp_path / "nonexistent.duckdb")
    status = check_database_health(missing_file, lang="ja")
    assert "存在しません" in status.message


@patch("urllib.request.urlopen")
def test_check_ollama_health_success(mock_urlopen: MagicMock) -> None:
    """Test check_ollama_health when service is online with recommended model."""
    mock_response = MagicMock()
    mock_response.read.return_value = json.dumps(
        {"models": [{"name": "gemma4:latest"}]}
    ).encode()
    mock_response.__enter__.return_value = mock_response
    mock_urlopen.return_value = mock_response

    status = check_ollama_health(recommended_model="gemma4")
    assert status.accessible
    assert "gemma4:latest" in status.available_models
    assert status.remediation is None
    assert status.to_dict()["accessible"] is True


@patch("urllib.request.urlopen")
def test_check_ollama_health_missing_model(mock_urlopen: MagicMock) -> None:
    """Test check_ollama_health when service is online but recommended model missing."""
    mock_response = MagicMock()
    mock_response.read.return_value = json.dumps(
        {"models": [{"name": "llama3:latest"}]}
    ).encode()
    mock_response.__enter__.return_value = mock_response
    mock_urlopen.return_value = mock_response

    status = check_ollama_health(recommended_model="gemma4")
    assert status.accessible
    assert "gemma4" not in status.available_models[0]
    assert status.remediation is not None
    assert "ollama pull" in status.remediation


@patch("urllib.request.urlopen")
def test_check_ollama_health_unreachable(mock_urlopen: MagicMock) -> None:
    """Test check_ollama_health when Ollama endpoint is down."""
    mock_urlopen.side_effect = urllib.error.URLError("Connection refused")

    status = check_ollama_health()
    assert not status.accessible
    assert status.available_models == []
    assert status.remediation is not None


def test_get_system_health_healthy(tmp_path: Path) -> None:
    """Test get_system_health returns healthy when all services are ok."""
    db_file = str(tmp_path / "healthy.duckdb")
    conn = duckdb.connect(db_file)
    conn.execute("CREATE TABLE patients (id INT)")
    conn.close()

    with patch("t1d_analytics.diagnostics.check_ollama_health") as mock_ollama:
        from t1d_analytics.diagnostics import OllamaStatus

        mock_ollama.return_value = OllamaStatus(
            accessible=True,
            available_models=["gemma4"],
            message="Online",
        )
        health = get_system_health(db_path=db_file)
        assert health.status == "healthy"
        assert health.database.connected
        assert health.ollama.accessible
        dict_rep = health.to_dict()
        assert dict_rep["status"] == "healthy"
        assert dict_rep["version"] == "0.1.0"


def test_get_system_health_degraded(tmp_path: Path) -> None:
    """Test get_system_health returns degraded when initial data is missing."""
    db_file = str(tmp_path / "custom.duckdb")
    conn = duckdb.connect(db_file)
    conn.execute("CREATE TABLE other (id INT)")
    conn.close()

    with patch("t1d_analytics.diagnostics.check_ollama_health") as mock_ollama:
        from t1d_analytics.diagnostics import OllamaStatus

        mock_ollama.return_value = OllamaStatus(
            accessible=True,
            available_models=["gemma4"],
            message="Online",
        )
        health = get_system_health(db_path=db_file)
        assert health.status == "degraded"


def test_get_system_health_error(tmp_path: Path) -> None:
    """Test get_system_health returns error when database is missing."""
    missing_file = str(tmp_path / "missing.duckdb")
    health = get_system_health(db_path=missing_file)
    assert health.status == "error"


def test_log_startup_diagnostics_all_branches(tmp_path: Path) -> None:
    """Test log_startup_diagnostics logs appropriately for different states."""
    # 1. Healthy state
    db_file = str(tmp_path / "healthy.duckdb")
    conn = duckdb.connect(db_file)
    conn.execute("CREATE TABLE patients (id INT)")
    conn.close()

    with patch("t1d_analytics.diagnostics.check_ollama_health") as mock_ollama:
        from t1d_analytics.diagnostics import OllamaStatus

        mock_ollama.return_value = OllamaStatus(
            accessible=True,
            available_models=["gemma4"],
            message="Online",
        )
        with patch.object(
            logging.getLogger("t1d_analytics.diagnostics"), "info"
        ) as mock_info:
            log_startup_diagnostics(db_path=db_file)
            assert mock_info.called

    # 2. Degraded state with missing initial data and ollama model warning
    db_file2 = str(tmp_path / "empty.duckdb")
    conn2 = duckdb.connect(db_file2)
    conn2.close()

    with patch("t1d_analytics.diagnostics.check_ollama_health") as mock_ollama:
        from t1d_analytics.diagnostics import OllamaStatus

        mock_ollama.return_value = OllamaStatus(
            accessible=True,
            available_models=["other"],
            message="Model missing",
            remediation="Run ollama pull",
        )
        with patch.object(
            logging.getLogger("t1d_analytics.diagnostics"), "warning"
        ) as mock_warn:
            log_startup_diagnostics(db_path=db_file2)
            assert mock_warn.called

    # 3. Error state with missing db and offline ollama
    missing_file = str(tmp_path / "missing.duckdb")
    with patch("t1d_analytics.diagnostics.check_ollama_health") as mock_ollama:
        from t1d_analytics.diagnostics import OllamaStatus

        mock_ollama.return_value = OllamaStatus(
            accessible=False,
            available_models=[],
            message="Offline",
            remediation="Start Ollama",
        )
        with patch.object(
            logging.getLogger("t1d_analytics.diagnostics"), "error"
        ) as mock_err:
            log_startup_diagnostics(db_path=missing_file)
            assert mock_err.called


def test_log_startup_diagnostics_no_remediations() -> None:
    """Test log_startup_diagnostics when remediation strings are None."""
    from t1d_analytics.diagnostics import DatabaseStatus, OllamaStatus, SystemHealth

    # Test warning branch with remediation=None
    mock_health_warn = SystemHealth(
        status="degraded",
        backend_accessible=True,
        version="0.1.0",
        database=DatabaseStatus(
            configured_path="test.duckdb",
            exists=True,
            connected=True,
            status_code=DatabaseStatusCode.EMPTY_DB,
            message="Empty",
            remediation=None,
        ),
        ollama=OllamaStatus(
            accessible=True,
            available_models=["other"],
            message="No gemma",
            remediation=None,
        ),
    )
    with patch(
        "t1d_analytics.diagnostics.get_system_health", return_value=mock_health_warn
    ):
        log_startup_diagnostics()

    # Test error branch with remediation=None
    mock_health_err = SystemHealth(
        status="error",
        backend_accessible=True,
        version="0.1.0",
        database=DatabaseStatus(
            configured_path="test.duckdb",
            exists=False,
            connected=False,
            status_code=DatabaseStatusCode.MISSING_FILE,
            message="Missing",
            remediation=None,
        ),
        ollama=OllamaStatus(
            accessible=False,
            available_models=[],
            message="Offline",
            remediation=None,
        ),
    )
    with patch(
        "t1d_analytics.diagnostics.get_system_health", return_value=mock_health_err
    ):
        log_startup_diagnostics()


def test_check_ollama_health_host_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test check_ollama_health honors OLLAMA_HOST environment variable."""
    import io
    from urllib.request import Request

    # Test bare hostname without http:// prefix
    monkeypatch.setenv("OLLAMA_HOST", "remote-gpu:11434")
    requested_urls = []

    def mock_urlopen(req: Request, timeout: float = 2.0) -> typing.Any:
        requested_urls.append(req.full_url)
        return io.BytesIO(b'{"models": [{"name": "gemma4:latest"}]}')

    with patch("urllib.request.urlopen", side_effect=mock_urlopen):
        status = check_ollama_health()
        assert status.accessible is True
        assert "http://remote-gpu:11434/api/tags" in requested_urls

    # Test hostname with http:// prefix
    monkeypatch.setenv("OLLAMA_HOST", "http://remote-gpu:11434")
    with patch("urllib.request.urlopen", side_effect=mock_urlopen):
        status2 = check_ollama_health()
        assert status2.accessible is True

    # Test explicit ollama_url parameter
    with patch("urllib.request.urlopen", side_effect=mock_urlopen):
        status3 = check_ollama_health(ollama_url="http://explicit-host:11434/api/tags")
        assert status3.accessible is True


def test_check_disk_space(tmp_path: Path) -> None:
    """Test check_disk_space on directory and nonexistent path."""
    from t1d_analytics.diagnostics import check_disk_space

    usage = check_disk_space(tmp_path)
    assert usage["total_bytes"] > 0
    assert usage["free_bytes"] > 0

    nonexistent = tmp_path / "does_not_exist" / "sub"
    usage_fallback = check_disk_space(nonexistent)
    assert usage_fallback["total_bytes"] > 0


def test_check_database_health_metrics(tmp_path: Path) -> None:
    """Test check_database_health populates file size, writable, integrity, and disk space."""
    db_file = tmp_path / "metrics.duckdb"
    conn = duckdb.connect(str(db_file))
    conn.execute("CREATE TABLE patients (id INT)")
    conn.close()

    status = check_database_health(str(db_file))
    assert status.exists is True
    assert status.file_size_bytes > 0
    assert status.writable is True
    assert status.integrity_ok is True
    assert status.disk_free_bytes > 0
    d = status.to_dict()
    assert d["file_size_bytes"] == status.file_size_bytes
    assert d["writable"] is True
    assert d["integrity_ok"] is True


def test_check_database_health_integrity_fail(tmp_path: Path) -> None:
    """Test check_database_health when PRAGMA integrity_check fails."""
    db_file = tmp_path / "bad_integrity.duckdb"
    conn = duckdb.connect(str(db_file))
    conn.execute("CREATE TABLE patients (id INT)")
    conn.close()

    class FakeConn:
        def execute(self, sql: str) -> typing.Any:
            if "duckdb_tables" in sql:
                raise Exception("Corrupt block detected")
            res = MagicMock()
            res.fetchall.return_value = [("patients",)]
            return res

        def close(self) -> None:
            pass

    with patch("duckdb.connect", return_value=FakeConn()):
        status = check_database_health(str(db_file))
        assert status.integrity_ok is False


def test_check_ollama_health_with_version(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test check_ollama_health populates version string when /api/version responds."""
    import io
    from urllib.request import Request

    monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:11434")

    def mock_urlopen(req: Request, timeout: float = 2.0) -> typing.Any:
        if "/api/version" in req.full_url:
            return io.BytesIO(b'{"version": "0.1.28"}')
        return io.BytesIO(b'{"models": [{"name": "gemma4:latest"}]}')

    with patch("urllib.request.urlopen", side_effect=mock_urlopen):
        status = check_ollama_health()
        assert status.accessible is True
        assert status.version == "0.1.28"
        d = status.to_dict()
        assert d["version"] == "0.1.28"


def test_check_ollama_health_version_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test check_ollama_health gracefully handles error when querying /api/version."""
    import io
    from urllib.request import Request

    monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:11434")

    def mock_urlopen(req: Request, timeout: float = 2.0) -> typing.Any:
        if "/api/version" in req.full_url:
            raise urllib.error.URLError("Not found")
        return io.BytesIO(b'{"models": [{"name": "gemma4:latest"}]}')

    with patch("urllib.request.urlopen", side_effect=mock_urlopen):
        status = check_ollama_health()
        assert status.accessible is True
        assert status.version is None


def test_check_disk_space_exception() -> None:
    """Test check_disk_space fallback when resolving path raises Exception."""
    from t1d_analytics.diagnostics import check_disk_space

    with patch("pathlib.Path.resolve", side_effect=Exception("Disk error")):
        usage = check_disk_space("any_path")
        assert usage["total_bytes"] > 0


def test_check_provider_health_all_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test check_provider_health across configured, missing SDK, and missing key branches."""
    import sys
    from unittest.mock import MagicMock

    from t1d_analytics.diagnostics import check_provider_health

    # Branch 1: SDK installed and API key set
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    with patch.dict(sys.modules, {"any_llm": MagicMock()}):
        status = check_provider_health("openai")
        assert status.configured is True
        assert status.sdk_installed is True
        assert status.api_key_set is True
        assert "ready" in status.message
        assert status.remediation is None
        d = status.to_dict()
        assert d["configured"] is True

    # Branch 2: SDK missing
    with patch.dict(sys.modules, {"any_llm": None}):
        status_no_sdk = check_provider_health("anthropic")
        assert status_no_sdk.configured is False
        assert status_no_sdk.sdk_installed is False
        assert status_no_sdk.remediation is not None
        assert "pip install" in status_no_sdk.remediation

    # Branch 3: SDK installed, API key missing
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with patch.dict(sys.modules, {"any_llm": MagicMock()}):
        status_no_key = check_provider_health("anthropic")
        assert status_no_key.configured is False
        assert status_no_key.sdk_installed is True
        assert status_no_key.api_key_set is False
        assert "API key" in status_no_key.message
        assert status_no_key.remediation is not None

    # Branch 4: Google GEMINI_API_KEY / GOOGLE_API_KEY
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini")
    with patch.dict(sys.modules, {"any_llm": MagicMock()}):
        status_google = check_provider_health("google")
        assert status_google.api_key_set is True
        assert status_google.configured is True


def test_get_system_health_providers_dict(tmp_path: Path) -> None:
    """Test get_system_health populates provider status in object and dictionary."""
    from t1d_analytics.diagnostics import get_system_health

    health = get_system_health()
    assert "openai" in health.providers
    assert "anthropic" in health.providers
    assert "google" in health.providers
    d = health.to_dict()
    assert "providers" in d
    providers_dict = d["providers"]
    assert isinstance(providers_dict, dict)
    assert "openai" in providers_dict
