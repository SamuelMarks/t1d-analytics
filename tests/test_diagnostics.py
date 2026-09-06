"""Tests for the diagnostics and health validation module."""

import json
import logging
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, patch

import duckdb
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
