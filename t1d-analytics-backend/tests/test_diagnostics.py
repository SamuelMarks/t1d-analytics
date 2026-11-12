"""Tests for the diagnostics and health validation module."""

import json
import logging
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
    with patch("pathlib.Path.stat", side_effect=OSError("Permission denied")):
        status = check_database_health(str(target_file))
        assert status.status_code == DatabaseStatusCode.UNREADABLE
        assert "Cannot access" in status.message


def test_check_database_health_connect_exception(tmp_path: Path) -> None:
    """Test check_database_health when duckdb.connect fails."""
    corrupt_file = tmp_path / "corrupt.duckdb"
    corrupt_file.write_text("corrupt content")
    with patch("duckdb.connect", side_effect=duckdb.Error("Corrupt database")):
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
    conn.execute("INSERT INTO patients VALUES (1)")
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
        assert health.database.total_rows == 1
        assert health.ollama.accessible
        dict_rep = health.to_dict()
        assert dict_rep["status"] == "healthy"
        assert dict_rep["version"] == "0.1.0"


def test_get_system_health_degraded(tmp_path: Path) -> None:
    """Test get_system_health returns degraded when initial data is missing or empty."""
    db_file = str(tmp_path / "custom.duckdb")
    conn = duckdb.connect(db_file)
    conn.execute("CREATE TABLE other (id INT)")
    conn.execute("INSERT INTO other VALUES (1)")
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
    conn.execute("INSERT INTO patients VALUES (1)")
    conn.close()

    with patch("t1d_analytics.diagnostics.check_ollama_health") as mock_ollama:
        from t1d_analytics.diagnostics import OllamaStatus

        mock_ollama.return_value = OllamaStatus(
            accessible=True,
            available_models=["gemma4"],
            message="Online",
        )
        with patch("t1d_analytics.diagnostics.logger.info") as mock_info:
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
        with patch("t1d_analytics.diagnostics.logger.warning") as mock_warn:
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
        with patch("t1d_analytics.diagnostics.logger.error") as mock_err:
            log_startup_diagnostics(db_path=missing_file)
            assert mock_err.called


def test_log_startup_diagnostics_no_remediations(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test log_startup_diagnostics when remediation strings are None."""
    from t1d_analytics.diagnostics import DatabaseStatus, OllamaStatus, SystemHealth

    caplog.set_level(logging.INFO, logger="t1d_analytics.diagnostics")

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
    assert "Empty" in caplog.text
    assert "No gemma" in caplog.text
    assert "-> Action:" not in caplog.text
    caplog.clear()

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
    assert "Missing" in caplog.text
    assert "-> Action:" not in caplog.text


def test_check_ollama_health_host_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test check_ollama_health honors OLLAMA_HOST environment variable."""
    import io
    from urllib.request import Request

    # Test bare hostname without http:// prefix
    monkeypatch.setenv("OLLAMA_HOST", "remote-gpu:11434")
    requested_urls = []

    def mock_urlopen(req: Request, timeout: float = 2.0) -> io.BytesIO:
        """Record requested URL and return mock models JSON."""
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
        """Fake DuckDB connection that fails on duckdb_tables inspection."""

        def execute(self, sql: str) -> MagicMock:
            """Raise corruption error on duckdb_tables query."""
            if "duckdb_tables" in sql:
                raise duckdb.Error("Corrupt block detected")
            res = MagicMock()
            res.fetchall.return_value = [("patients",)]
            res.fetchone.return_value = (1,)
            return res

        def close(self) -> None:
            """Close fake connection."""

    with patch("duckdb.connect", return_value=FakeConn()):
        status = check_database_health(str(db_file))
        assert status.integrity_ok is False


def test_check_sessions_database_health(tmp_path: Path) -> None:
    """Test check_sessions_database_health read-only inspection, initialization, and failure branches."""
    from t1d_analytics.diagnostics import (
        check_sessions_database_health,
        run_system_diagnostics,
    )

    sess_db = str(tmp_path / "sessions.duckdb")
    # 1. Initializes non-existent sessions.duckdb cleanly
    assert check_sessions_database_health(sess_db) is True

    # 2. Existing sessions.duckdb is inspected in read-only mode without calling open_sessions_db_conn
    with patch(
        "t1d_analytics.session_store.open_sessions_db_conn",
        side_effect=AssertionError("Should not open write connection when existing"),
    ):
        assert check_sessions_database_health(sess_db) is True

    # 3. InvalidInputException during read-only open (e.g. active write connection) falls back to open_sessions_db_conn
    with patch(
        "t1d_analytics.diagnostics.connect_duckdb_with_retry",
        side_effect=duckdb.InvalidInputException("Mixed config"),
    ):
        assert check_sessions_database_health(sess_db) is True

    # 4. Non-existent sessions.duckdb failure returns False
    missing_sess_db = str(tmp_path / "missing_sessions.duckdb")
    with patch(
        "t1d_analytics.session_store.open_sessions_db_conn",
        side_effect=RuntimeError("Locked"),
    ):
        assert check_sessions_database_health(missing_sess_db) is False
        diag = run_system_diagnostics(sessions_db_path=missing_sess_db)
        assert diag.status == "error"
        assert diag.sessions_database_healthy is False

    from t1d_analytics.diagnostics import (
        _DIAGNOSTICS_STATS_CACHE,
        _lookup_cached_diagnostics_stats,
        _store_cached_diagnostics_stats,
    )

    assert _lookup_cached_diagnostics_stats(
        str(tmp_path / "does_not_exist.duckdb")
    ) == (
        None,
        None,
    )
    _DIAGNOSTICS_STATS_CACHE.clear()
    _store_cached_diagnostics_stats(None, ([], 0, True, "unknown"))
    for idx in range(65):
        _store_cached_diagnostics_stats(
            (f"/fake/{idx}.duckdb", idx, idx), (["patients"], 10, True, "seeded_demo")
        )
    assert len(_DIAGNOSTICS_STATS_CACHE) == 1
    _DIAGNOSTICS_STATS_CACHE.clear()


def test_check_ollama_health_with_version(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test check_ollama_health populates version string when /api/version responds."""
    import io
    from urllib.request import Request

    monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:11434")

    def mock_urlopen(req: Request, timeout: float = 2.0) -> io.BytesIO:
        """Return version JSON or models JSON based on endpoint."""
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

    def mock_urlopen(req: Request, timeout: float = 2.0) -> io.BytesIO:
        """Raise URLError on /api/version and return models on /api/tags."""
        if "/api/version" in req.full_url:
            raise urllib.error.URLError("Not found")
        return io.BytesIO(b'{"models": [{"name": "gemma4:latest"}]}')

    with patch("urllib.request.urlopen", side_effect=mock_urlopen):
        status = check_ollama_health()
        assert status.accessible is True
        assert status.version is None


def test_check_disk_space_exception() -> None:
    """Test check_disk_space fallback when resolving path raises OSError."""
    from t1d_analytics.diagnostics import check_disk_space

    with patch("pathlib.Path.resolve", side_effect=OSError("Disk error")):
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


def test_detect_database_provenance_and_jaeb_views(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test detect_database_provenance, ensure_canonical_clinical_views, and seed_database across jaeb_authentic, synthetic_seed, and custom DBs."""
    from t1d_analytics.diagnostics import (
        check_database_health,
        detect_database_provenance,
    )
    from t1d_analytics.seed import ensure_canonical_clinical_views, seed_database

    assert detect_database_provenance([]) == "custom"
    assert detect_database_provenance(["custom_only"]) == "custom"
    assert detect_database_provenance(["patients", "cgm_readings"]) == "synthetic_seed"
    assert (
        detect_database_provenance(["acpepptroster", "tblaptsummary"])
        == "jaeb_authentic"
    )
    assert (
        detect_database_provenance(["tblaptsummary"], base_tables={"tblaptsummary"})
        == "jaeb_authentic"
    )
    assert (
        detect_database_provenance(
            ["tblaptsummary", "patients"], base_tables={"tblaptsummary", "patients"}
        )
        == "synthetic_seed"
    )

    # Test check_database_health on an authentic Jaeb DB that also had synthetic base tables and synthetic_seed manifest
    jaeb_db = tmp_path / "jaeb_large.duckdb"
    conn = duckdb.connect(str(jaeb_db))
    conn.execute(
        "CREATE TABLE tblaptsummary (RecID BIGINT, ptid BIGINT, gender VARCHAR, AgeAsOfRandDt BIGINT, InsulinModality VARCHAR)"
    )
    conn.execute("INSERT INTO tblaptsummary VALUES (1, 101, 'F', 28, 'Pump')")
    conn.execute(
        "CREATE TABLE patients (subject_id INT, age INT, gender VARCHAR, hba1c DOUBLE, tir DOUBLE, numhospdka INT)"
    )
    conn.execute("INSERT INTO patients VALUES (999, 40, 'M', 8.0, 55.0, 0)")
    conn.execute(
        "CREATE TABLE _t1d_seed_manifest (key VARCHAR PRIMARY KEY, value VARCHAR)"
    )
    conn.execute(
        "INSERT INTO _t1d_seed_manifest VALUES ('provenance', 'synthetic_seed')"
    )
    for idx in range(28):
        conn.execute(f"CREATE TABLE study_tbl_{idx} (id INT)")
    conn.close()

    # Even before ensure_canonical_clinical_views, synthetic_seed manifest does NOT override jaeb_authentic
    status_before = check_database_health(str(jaeb_db))
    assert status_before.data_provenance == "jaeb_authentic"
    assert status_before.is_synthetic_faked_data is False

    # ensure_canonical_clinical_views drops synthetic 'patients' base table, replaces it with VIEW over tblaptsummary, and updates manifest
    ensure_canonical_clinical_views(str(jaeb_db))
    status_jaeb = check_database_health(str(jaeb_db))
    assert status_jaeb.data_provenance == "jaeb_authentic"
    assert status_jaeb.is_synthetic_faked_data is False
    assert status_jaeb.to_dict()["data_provenance"] == "jaeb_authentic"
    assert status_jaeb.to_dict()["is_synthetic_faked_data"] is False
    assert "recommended_jaeb_db_path" in status_jaeb.to_dict()

    conn_verify = duckdb.connect(str(jaeb_db), read_only=True)
    assert conn_verify.execute("SELECT subject_id FROM patients").fetchone() == (101,)
    assert conn_verify.execute(
        "SELECT value FROM _t1d_seed_manifest WHERE key = 'provenance'"
    ).fetchone() == ("jaeb_authentic",)
    conn_verify.close()

    # seed_database on an already-authentic Jaeb DB preserves jaeb_authentic provenance and does not overwrite with synthetic CSV rows
    jaeb_seed_counts = seed_database(
        data_dir=str(tmp_path / "seed_data"),
        db_path=str(jaeb_db),
        sessions_db_path=str(tmp_path / "seed_sessions.duckdb"),
    )
    assert "tblaptsummary" in jaeb_seed_counts
    assert check_database_health(str(jaeb_db)).is_synthetic_faked_data is False

    # Also test _t1d_seed_manifest upgrading a custom DB to jaeb_authentic or synthetic_seed
    manifest_db = tmp_path / "manifest_only.duckdb"
    conn_m = duckdb.connect(str(manifest_db))
    conn_m.execute("CREATE TABLE custom_tbl (id INT)")
    conn_m.execute("INSERT INTO custom_tbl VALUES (1)")
    conn_m.execute(
        "CREATE TABLE _t1d_seed_manifest (key VARCHAR PRIMARY KEY, value VARCHAR)"
    )
    conn_m.execute(
        "INSERT INTO _t1d_seed_manifest VALUES ('provenance', 'jaeb_authentic')"
    )
    conn_m.close()
    assert check_database_health(str(manifest_db)).data_provenance == "jaeb_authentic"

    empty_views_db = tmp_path / "empty_views.duckdb"
    conn_ev = duckdb.connect(str(empty_views_db))
    conn_ev.execute("CREATE TABLE unrelated_only (id INT)")
    conn_ev.close()
    ensure_canonical_clinical_views(str(empty_views_db))

    # Full Jaeb schema with tblaptsummary (TxGroup, NumSevHypo), tblalabhba1c, bdatacgm, adverseevents, and _t1d_ingestion_manifest
    from t1d_analytics.ingestion import load_data_to_duckdb
    from t1d_analytics.seed import write_baseline_csvs

    jaeb_store = tmp_path / "jaeb_store" / "jaeb"
    jaeb_store.mkdir(parents=True)
    full_jaeb_src = jaeb_store / "t1d.duckdb"
    c_fj = duckdb.connect(str(full_jaeb_src))
    c_fj.execute(
        "CREATE TABLE tblaptsummary (RecID BIGINT, ptid BIGINT, gender VARCHAR, AgeAsOfRandDt BIGINT, InsulinModality VARCHAR, NumSevHypo VARCHAR, TxGroup VARCHAR)"
    )
    c_fj.execute(
        "INSERT INTO tblaptsummary VALUES (1, 307, 'M', 46, 'Pump', '1', 'RT-CGM'), (2, 258, 'F', 18, 'Injections', '0', 'Control')"
    )
    c_fj.execute(
        "CREATE TABLE tblalabhba1c (RecID BIGINT, visit VARCHAR, ptid BIGINT, LabHbA1cDt TIMESTAMP, LabA1cResult DOUBLE)"
    )
    c_fj.execute(
        "INSERT INTO tblalabhba1c VALUES (1, 'Baseline', 307, TIMESTAMP '2024-01-09 00:00:00', 7.3), (2, 'Baseline', 258, TIMESTAMP '2024-01-14 00:00:00', 6.1)"
    )
    c_fj.execute(
        "CREATE TABLE bdatacgm (RecID BIGINT, ptid BIGINT, DeviceDaysFromEnroll BIGINT, devicetm TIME, glucose BIGINT)"
    )
    c_fj.execute(
        "INSERT INTO bdatacgm VALUES (1, 307, 0, TIME '08:00:00', 142), (2, 258, 1, TIME '09:30:00', 118)"
    )
    c_fj.execute(
        "CREATE TABLE adverseevents (subjectid BIGINT, event VARCHAR, eventDt VARCHAR)"
    )
    c_fj.execute(
        "INSERT INTO adverseevents VALUES (307, 'DKA', '08DEC2024'), (258, 'Severe Hypo', '02OCT2024')"
    )
    c_fj.execute(
        "CREATE TABLE _t1d_ingestion_manifest (source_file VARCHAR, table_name VARCHAR, file_hash VARCHAR, row_count BIGINT, loaded_at TIMESTAMP, file_size BIGINT)"
    )
    c_fj.execute(
        "INSERT INTO _t1d_ingestion_manifest VALUES ('data/patients.csv', 'patients', 'h', 100, CURRENT_TIMESTAMP, 1000)"
    )
    c_fj.execute("CREATE TABLE patients (subject_id INT)")
    c_fj.execute("INSERT INTO patients VALUES (1)")
    for idx in range(26):
        c_fj.execute(f"CREATE TABLE jaeb_extra_{idx} (id INT)")
    c_fj.close()

    # Promote full_jaeb_src via T1D_DB_SEARCH_DIR into a fresh target_promoted_db
    monkeypatch.setenv(
        "T1D_DB_SEARCH_DIR",
        f"{tmp_path / 'empty_search'}:{tmp_path / 'jaeb_store'}",
    )
    promoted_db = tmp_path / "promoted_jaeb.duckdb"
    syn_data_dir = tmp_path / "syn_csv_dir"
    write_baseline_csvs(str(syn_data_dir))
    promoted_counts = seed_database(
        data_dir=str(syn_data_dir),
        db_path=str(promoted_db),
        sessions_db_path=str(tmp_path / "prom_sessions.duckdb"),
    )
    assert promoted_counts["patients"] == 2
    assert promoted_counts["cgm_readings"] == 2
    assert promoted_counts["hba1c_labs"] == 2
    assert promoted_counts["dka_adverse_events"] == 2

    # Calling load_data_to_duckdb with synthetic CSVs on >25-table Jaeb DB skips overwriting canonical views
    load_data_to_duckdb(str(syn_data_dir), str(promoted_db))
    c_pv = duckdb.connect(str(promoted_db), read_only=True)
    assert c_pv.execute("SELECT COUNT(*) FROM patients").fetchone() == (2,)
    c_pv.close()
    monkeypatch.setenv("T1D_DB_SEARCH_DIR", str(tmp_path / "empty_search"))
    from t1d_analytics.seed_verifier import (
        find_authentic_jaeb_source_db,
        resolve_non_empty_db_path,
    )

    assert find_authentic_jaeb_source_db(tmp_path / "none_found.duckdb") is None
    monkeypatch.delenv("T1D_DB_SEARCH_DIR", raising=False)
    monkeypatch.delenv("T1D_DB_PATH", raising=False)
    assert resolve_non_empty_db_path("", default_path="fallback.duckdb") == (
        "fallback.duckdb"
    )
    monkeypatch.setenv("T1D_DB_PATH", str(promoted_db))
    assert resolve_non_empty_db_path("") == str(promoted_db)
    ensure_canonical_clinical_views("")

    # Test resolve_subdir_prefix and resolve_target_table_stem for JAEB multi-study folders
    from t1d_analytics.ingestion_study_resolver import (
        resolve_subdir_prefix,
        resolve_target_table_stem,
    )

    assert (
        resolve_subdir_prefix(
            Path("The_International_Diabetes_Closed_Loop_Trial_DCLP3/Data_Files")
        )
        == "dclp3"
    )
    assert (
        resolve_subdir_prefix(
            Path("Some_Very_Long_Custom_Study_Directory_Without_Known_Code/Sub")
        )
        == "some_very_long_custom_study_directory_without_known_code_sub"
    )
    root_ing = tmp_path / "jaeb_ing"
    d3_sub = root_ing / "The_International_Diabetes_Closed_Loop_Trial_DCLP3"
    d3_sub.mkdir(parents=True)
    f_unprefixed = d3_sub / "PtRoster.txt"
    f_prefixed = d3_sub / "dclp3_PtRoster.txt"
    assert (
        resolve_target_table_stem(f_unprefixed, root_ing, True, {"ptroster": 2})
        == "dclp3_ptroster"
    )
    assert (
        resolve_target_table_stem(f_prefixed, root_ing, True, {"dclp3_ptroster": 1})
        == "dclp3_ptroster"
    )

    # Test build_jaeb_beck2023_cohort_table and ensure_canonical_clinical_views with DCLP3/DCLP5/PEDAP tables
    beck_db = tmp_path / "beck2023_test.duckdb"
    c_beck = duckdb.connect(str(beck_db))
    c_beck.execute(
        "CREATE TABLE dclp3_ptroster (PtID INT, EnrollDt VARCHAR, RandDt VARCHAR, trtGroup VARCHAR)"
    )
    c_beck.execute(
        "INSERT INTO dclp3_ptroster VALUES (1, '2018-01-10', '2018-01-24', 'CLC')"
    )
    c_beck.execute(
        "CREATE TABLE dclp3_diabscreening (PtID INT, AgeAtEnrollment INT, Gender VARCHAR, Ethnicity VARCHAR, Race VARCHAR, DiagDt VARCHAR, DiagAge INT, InsModPump INT, CGMUseStat VARCHAR)"
    )
    c_beck.execute(
        "INSERT INTO dclp3_diabscreening VALUES (1, 20, 'F', 'Not Hispanic or Latino', 'White', '2010-01-10', 12, 1, 'Current')"
    )
    c_beck.execute("CREATE TABLE dclp3_diabsocioecon (PtID INT, AnnualIncome VARCHAR)")
    c_beck.execute(
        "INSERT INTO dclp3_diabsocioecon VALUES (1, '$100,000 to less than $200,000')"
    )
    c_beck.execute(
        "CREATE TABLE dclp3_sampleresults (PtID INT, Visit VARCHAR, ResultName VARCHAR, Value VARCHAR)"
    )
    c_beck.execute(
        "INSERT INTO dclp3_sampleresults VALUES (1, 'Randomization', 'GLYHB', '7.2')"
    )
    c_beck.execute("CREATE TABLE dclp3_cgm (PtID INT, Period VARCHAR, CGM INT)")
    c_beck.execute(
        "INSERT INTO dclp3_cgm SELECT 1, '1. Baseline', 120 FROM range(2100)"
    )

    c_beck.execute(
        "CREATE TABLE dclp5_ptroster (PtID INT, EnrollDt VARCHAR, RandDt VARCHAR, trtGroup VARCHAR, AgeAtEnrollment INT)"
    )
    c_beck.execute(
        "INSERT INTO dclp5_ptroster VALUES (2, '2019-01-10', '2019-01-24', 'SC', 10)"
    )
    c_beck.execute(
        "CREATE TABLE dclp5_diabscreening (PtID INT, Gender VARCHAR, Ethnicity VARCHAR, Race VARCHAR, DiagDt VARCHAR, DiagAge INT, InsModPump INT, CGMUseStat VARCHAR)"
    )
    c_beck.execute(
        "INSERT INTO dclp5_diabscreening VALUES (2, 'M', 'Hispanic or Latino', 'White', '2015-01-10', 6, 0, 'Never')"
    )
    c_beck.execute("CREATE TABLE dclp5_diabsocioecon (PtID INT, AnnualIncome VARCHAR)")
    c_beck.execute(
        "INSERT INTO dclp5_diabsocioecon VALUES (2, '$50,000 to less than $75,000')"
    )
    c_beck.execute(
        "CREATE TABLE dclp5_sampleresults (PtID INT, Visit VARCHAR, ResultName VARCHAR, Value VARCHAR)"
    )
    c_beck.execute(
        "INSERT INTO dclp5_sampleresults VALUES (2, 'Baseline', 'GLYHB', '7.8')"
    )
    c_beck.execute(
        "CREATE TABLE dclp5_dexcomclaritycgm (PtID INT, DataDtTm VARCHAR, DataDtTm_adj VARCHAR, CGM INT)"
    )
    c_beck.execute(
        "INSERT INTO dclp5_dexcomclaritycgm SELECT 2, strftime(TIMESTAMP '2019-01-10 00:00:00' + to_minutes(CAST(i * 5 AS INT)), '%Y-%m-%d %H:%M:%S'), NULL, 140 FROM range(2100) t(i)"
    )
    c_beck.execute(
        "CREATE TABLE dclp5_othercgm (PtID INT, DataDtTm VARCHAR, CGM INT, IsCalibration BOOLEAN)"
    )
    c_beck.execute(
        "CREATE TABLE dclp5_tandemcgmdatagxb_b (PtID INT, DataDtTm VARCHAR, DataDtTm_adjusted VARCHAR, CGMValue INT, HighLowIndicator INT)"
    )

    c_beck.execute(
        "CREATE TABLE pedap_ptroster (PtID INT, EnrollDt VARCHAR, RandDt VARCHAR, TrtGroup VARCHAR, AgeAsofEnrollDt INT)"
    )
    c_beck.execute(
        "INSERT INTO pedap_ptroster VALUES (3, '2020-01-10', '2020-01-24 12:00:00', 'CLC', 4)"
    )
    c_beck.execute(
        "CREATE TABLE pedap_diabscreening (PtID INT, Sex VARCHAR, Ethnicity VARCHAR, Race VARCHAR, DiagDt VARCHAR, DiagAge INT, InsModPump INT, CGMUseStat VARCHAR)"
    )
    c_beck.execute(
        "INSERT INTO pedap_diabscreening VALUES (3, 'F', 'Not Hispanic or Latino', 'Asian', '2018-01-10', 2, 1, 'Current')"
    )
    c_beck.execute("CREATE TABLE pedap_diabsocioecon (PtID INT, AnnualIncome VARCHAR)")
    c_beck.execute("INSERT INTO pedap_diabsocioecon VALUES (3, 'Unknown')")
    c_beck.execute(
        "CREATE TABLE pedap_stasampleresults (PtID INT, Visit VARCHAR, ResultName VARCHAR, Value VARCHAR, STAResultStatus VARCHAR)"
    )
    c_beck.execute(
        "INSERT INTO pedap_stasampleresults VALUES (3, 'Randomization', 'GLYHB', '7.5', 'Complete')"
    )
    c_beck.execute(
        "CREATE TABLE pedap_dexcomclaritycgm (PtID INT, DeviceDtTm VARCHAR, CGM INT)"
    )
    c_beck.execute(
        "INSERT INTO pedap_dexcomclaritycgm SELECT 3, strftime(TIMESTAMP '2020-01-10 00:00:00' + to_minutes(CAST(i * 5 AS INT)), '%Y-%m-%d %H:%M:%S'), 130 FROM range(2100) t(i)"
    )
    c_beck.execute(
        "CREATE TABLE pedap_othercgm (PtID INT, DeviceDtTm VARCHAR, CGM INT, IsCalibration BOOLEAN)"
    )
    c_beck.execute(
        "CREATE TABLE pedap_tandemcgmdatagxb (PtID INT, DeviceDtTm VARCHAR, CGMValue INT, HighLowIndicator INT)"
    )
    c_beck.execute("CREATE TABLE dclp3_adv_event (PtID INT, Event VARCHAR)")
    c_beck.execute(
        "INSERT INTO dclp3_adv_event VALUES (1, 'DKA'), (1, 'Severe Hypoglycemia')"
    )
    c_beck.execute("CREATE TABLE dclp5_adverseevents (OtherCol INT)")
    c_beck.close()

    ensure_canonical_clinical_views(str(beck_db))
    c_bchk = duckdb.connect(str(beck_db), read_only=True)
    assert c_bchk.execute("SELECT COUNT(*) FROM jaeb_beck2023_cohort").fetchone() == (
        3,
    )
    assert c_bchk.execute("SELECT COUNT(*) FROM patients").fetchone() == (3,)
    assert c_bchk.execute(
        "SELECT dka_events, severe_hypo_events FROM patients WHERE PtID = 1"
    ).fetchone() == (
        1,
        1,
    )
    c_bchk.close()

    c_beck_no_ae = duckdb.connect(str(beck_db))
    c_beck_no_ae.execute("DROP TABLE dclp3_adv_event")
    c_beck_no_ae.execute("DROP TABLE dclp5_adverseevents")
    c_beck_no_ae.close()
    ensure_canonical_clinical_views(str(beck_db))
