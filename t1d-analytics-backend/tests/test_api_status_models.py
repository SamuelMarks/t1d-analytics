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
            req: str | urllib.request.Request,
            timeout: float | None = None,
        ) -> None:
            """Initialize MockUrlopen."""
            self.req = req

        def __enter__(self) -> MockResponse:
            """Enter the context manager."""
            return MockResponse()

        def __exit__(
            self,
            exc_type: type | None,
            exc_val: BaseException | None,
            exc_tb: types.TracebackType | None,
        ) -> None:
            """Exit the context manager."""

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
        req: str | urllib.request.Request,
        timeout: float | None = None,
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
        req: str | urllib.request.Request,
        timeout: float | None = None,
    ) -> None:
        """Simulate urlopen throwing an unexpected OSError."""
        raise OSError("Unexpected boom")

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

    from t1d_analytics import db_security

    remote_cand_db = tmp_path / "mnt_data_t1d_analytics.duckdb"
    remote_cand_db.write_text("remote-db")
    monkeypatch.setattr(
        db_security, "REMOTE_DISCOVERY_DB_CANDIDATES", (remote_cand_db,)
    )
    resp6 = client.get("/api/databases")
    assert resp6.status_code == 200
    assert any(
        Path(d["path"]).resolve() == remote_cand_db.resolve()
        for d in resp6.json()["databases"]
    )


def test_validate_db_path_null_byte() -> None:
    """Test validate_db_path rejects null byte injection."""
    from t1d_analytics.db_security import validate_db_path
    from t1d_analytics.errors import InvalidDatabasePathError

    with pytest.raises(InvalidDatabasePathError) as exc_info:
        validate_db_path("test" + chr(0) + ".duckdb")
    assert exc_info.value.status_code == 400
    assert exc_info.value.error_code == "backend.invalidDatabasePath"


def test_validate_db_path_traversal(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test validate_db_path and validate_workspace_artifact_path reject paths outside allowed directories or invalid extensions."""
    from t1d_analytics.db_security import (
        get_allowed_db_roots,
        validate_db_path,
        validate_workspace_artifact_path,
    )
    from t1d_analytics.errors import InvalidDatabasePathError

    allowed = [Path("/mock/allowed/dir")]
    with pytest.raises(InvalidDatabasePathError) as exc_info:
        validate_db_path("/etc/passwd", allowed_dirs=allowed)
    assert exc_info.value.status_code == 400
    assert exc_info.value.error_code == "backend.invalidDatabasePath"

    # Reject non-.duckdb / non-.db files even inside an allowed directory
    bad_ext = tmp_path / "secret.txt"
    bad_ext.write_text("secret", encoding="utf-8")
    with pytest.raises(InvalidDatabasePathError):
        validate_db_path(str(bad_ext), allowed_dirs=[tmp_path])

    # validate_workspace_artifact_path allows JSONL/directories inside allowed roots, rejects empty or traversal
    valid_art = tmp_path / "train.jsonl"
    assert validate_workspace_artifact_path(
        str(valid_art), allowed_dirs=[tmp_path]
    ) == str(valid_art.resolve())
    with pytest.raises(InvalidDatabasePathError):
        validate_workspace_artifact_path("   ", allowed_dirs=[tmp_path])

    # T1D_ALLOWED_DB_DIR overrides default allowed roots
    monkeypatch.setenv("T1D_ALLOWED_DB_DIR", str(tmp_path))
    assert get_allowed_db_roots() == [tmp_path.resolve()]
    monkeypatch.delenv("T1D_ALLOWED_DB_DIR", raising=False)

    # T1D_JAEB_DIR and T1D_DB_SEARCH_DIR add existing external roots and ignore non-existent roots
    from t1d_analytics import db_security
    from t1d_analytics.db_security import (
        _split_search_dir_spec,
        get_configured_db_search_dirs,
        get_jaeb_external_root,
        get_recommended_jaeb_db_path,
        get_t1d_db_search_dirs,
    )

    assert _split_search_dir_spec("   ") == []
    assert _split_search_dir_spec("/mnt/data/t1d:/opt/t1d") == [
        Path("/mnt/data/t1d"),
        Path("/opt/t1d"),
    ]
    assert _split_search_dir_spec("/mnt/data/t1d;/opt/t1d") == [
        Path("/mnt/data/t1d"),
        Path("/opt/t1d"),
    ]
    assert _split_search_dir_spec("/mnt/data/t1d,/opt/t1d") == [
        Path("/mnt/data/t1d"),
        Path("/opt/t1d"),
    ]
    assert _split_search_dir_spec("C:\\t1d") == [Path("C:\\t1d")]

    # When no search env vars are set, default roots (/mnt/data/t1d and ./data) and /jaeb subdirs are searched
    monkeypatch.delenv("T1D_EXTRA_DB_ROOTS", raising=False)
    monkeypatch.delenv("T1D_DB_SEARCH_DIR", raising=False)
    monkeypatch.delenv("T1D_JAEB_DIR", raising=False)
    default_cfg = get_configured_db_search_dirs()
    assert Path("/mnt/data/t1d") in default_cfg
    assert Path("/mnt/data/t1d/jaeb") in default_cfg
    assert Path("./data") in default_cfg
    assert Path("./data/jaeb") in default_cfg
    default_roots = get_allowed_db_roots()
    assert Path("/tmp").resolve() not in default_roots
    assert Path("/var/tmp").resolve() not in default_roots
    monkeypatch.setenv("T1D_EXTRA_DB_ROOTS", str(tmp_path.parent))

    jaeb_dir = tmp_path / "jaeb_ext"
    jaeb_dir.mkdir()
    monkeypatch.setenv("T1D_JAEB_DIR", str(jaeb_dir))
    assert get_jaeb_external_root() == jaeb_dir
    assert jaeb_dir.resolve() in get_allowed_db_roots()
    assert get_recommended_jaeb_db_path() == str(jaeb_dir / "jaeb" / "t1d.duckdb")

    explicit_jaeb_file = jaeb_dir / "custom_jaeb.duckdb"
    monkeypatch.setenv("T1D_JAEB_DB_PATH", str(explicit_jaeb_file))
    assert get_recommended_jaeb_db_path() == str(explicit_jaeb_file)
    monkeypatch.delenv("T1D_JAEB_DB_PATH", raising=False)

    jaeb_named = tmp_path / "jaeb"
    jaeb_named.mkdir()
    monkeypatch.setenv("T1D_JAEB_DIR", str(jaeb_named))
    assert get_recommended_jaeb_db_path() == str(jaeb_named / "t1d.duckdb")

    monkeypatch.setenv("T1D_JAEB_DIR", str(tmp_path / "missing_jaeb_ext"))
    assert (tmp_path / "missing_jaeb_ext").resolve() not in get_allowed_db_roots()
    monkeypatch.delenv("T1D_JAEB_DIR", raising=False)

    # REMOTE_MNT_DATA_ROOT when present on disk and T1D_DB_PATH inside an already-allowed dir
    mnt_sim = tmp_path / "mnt_data_root"
    mnt_sim.mkdir()
    monkeypatch.setattr(db_security, "REMOTE_MNT_DATA_ROOT", mnt_sim)
    monkeypatch.setenv("T1D_DB_PATH", str(Path.cwd() / "in_cwd.duckdb"))
    roots_with_mnt = get_allowed_db_roots()
    assert mnt_sim.resolve() in roots_with_mnt
    assert Path("/mnt/data/t1d").resolve() in roots_with_mnt
    assert Path("/mnt/data/t1d/jaeb").resolve() in roots_with_mnt
    monkeypatch.setattr(db_security, "REMOTE_MNT_DATA_ROOT", mnt_sim)
    monkeypatch.setattr(db_security, "REMOTE_T1D_DB_SEARCH_ROOT", mnt_sim)
    assert mnt_sim.resolve() in get_allowed_db_roots()
    monkeypatch.setattr(db_security, "REMOTE_T1D_DB_SEARCH_ROOT", Path("/mnt/data/t1d"))
    monkeypatch.delenv("T1D_DB_PATH", raising=False)

    # T1D_DB_SEARCH_DIR with multiple colon-separated paths (remote + local host) and duplicates
    remote_sim = tmp_path / "mnt_data_t1d"
    local_sim = tmp_path / "local_ext_t1d"
    local_sim.mkdir()
    monkeypatch.setenv(
        "T1D_DB_SEARCH_DIR",
        f"{remote_sim}:{local_sim}:{local_sim}:/opt/nonexistent/t1d",
    )
    monkeypatch.setenv("T1D_DB_PATH", str(local_sim / "test.duckdb"))
    assert get_jaeb_external_root() == local_sim
    assert local_sim.resolve() in get_allowed_db_roots()
    monkeypatch.delenv("T1D_DB_PATH", raising=False)
    cfg_dirs = get_configured_db_search_dirs()
    assert remote_sim in cfg_dirs
    assert (remote_sim / "jaeb") in cfg_dirs
    assert local_sim in cfg_dirs
    assert local_sim in get_t1d_db_search_dirs()

    # When T1D_DB_SEARCH_DIR points to a non-existent path from another host, falls back to existing host default
    remote_existing = tmp_path / "remote_mnt_data_t1d"
    remote_existing.mkdir()
    (remote_existing / "jaeb").mkdir()
    remote_jaeb_db = remote_existing / "jaeb" / "t1d.duckdb"
    remote_jaeb_db.write_text("jaeb")
    monkeypatch.setattr(
        db_security,
        "DEFAULT_DB_SEARCH_DIRS",
        (remote_existing, remote_sim),
    )
    monkeypatch.setattr(
        db_security,
        "DEFAULT_T1D_DB_SEARCH_DIRS",
        (
            remote_existing,
            remote_existing / "jaeb",
            remote_sim,
            remote_sim / "jaeb",
        ),
    )
    monkeypatch.setenv("T1D_DB_SEARCH_DIR", str(remote_sim))
    fallback_cfg = get_configured_db_search_dirs()
    assert remote_sim in fallback_cfg
    assert remote_existing in fallback_cfg
    assert (remote_existing / "jaeb") in fallback_cfg
    assert get_jaeb_external_root() == remote_existing
    assert get_recommended_jaeb_db_path() == str(remote_jaeb_db.resolve())
    remote_jaeb_db.unlink()
    assert get_recommended_jaeb_db_path() == str(
        remote_existing / "jaeb" / "t1d.duckdb"
    )

    # When neither configured nor default roots exist on disk
    monkeypatch.delenv("T1D_EXTRA_DB_ROOTS", raising=False)
    missing_def = tmp_path / "missing_default_root"
    monkeypatch.setattr(
        db_security,
        "DEFAULT_DB_SEARCH_DIRS",
        (missing_def,),
    )
    monkeypatch.setattr(
        db_security,
        "DEFAULT_T1D_DB_SEARCH_DIRS",
        (missing_def, missing_def / "jaeb"),
    )
    monkeypatch.setattr(db_security, "DEFAULT_JAEB_EXTERNAL_ROOT", missing_def)
    assert get_jaeb_external_root() == remote_sim
    monkeypatch.delenv("T1D_DB_SEARCH_DIR", raising=False)
    assert get_jaeb_external_root() == missing_def
    assert get_recommended_jaeb_db_path() == str(missing_def / "jaeb" / "t1d.duckdb")


def test_validate_db_path_env_var(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test validate_db_path respects T1D_DB_PATH and prioritizes T1D_DB_SEARCH_DIR over synthetic data/t1d.duckdb."""
    from t1d_analytics import db_security
    from t1d_analytics.db_security import validate_db_path
    from t1d_analytics.diagnostics import check_database_health

    missing_root = tmp_path / "no_default_jaeb"
    monkeypatch.setattr(
        db_security,
        "DEFAULT_DB_SEARCH_DIRS",
        (missing_root,),
    )
    monkeypatch.setattr(
        db_security,
        "DEFAULT_T1D_DB_SEARCH_DIRS",
        (missing_root, missing_root / "jaeb"),
    )
    monkeypatch.delenv("T1D_EXTRA_DB_ROOTS", raising=False)
    monkeypatch.delenv("T1D_DB_SEARCH_DIR", raising=False)
    monkeypatch.delenv("T1D_JAEB_DIR", raising=False)

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
    monkeypatch.setenv("T1D_EXTRA_DB_ROOTS", str(tmp_path))
    monkeypatch.delenv("T1D_DB_PATH", raising=False)
    data_sub = tmp_path / "data"
    data_sub.mkdir(exist_ok=True)
    (data_sub / "t1d.duckdb").write_text("dummy")
    resolved_data_fb = validate_db_path("t1d.duckdb")
    assert resolved_data_fb == str((data_sub / "t1d.duckdb").resolve())
    monkeypatch.delenv("T1D_EXTRA_DB_ROOTS", raising=False)
    disc_res = client.get("/api/databases").json()
    assert disc_res["current_db"] == "data/t1d.duckdb"
    monkeypatch.setenv("T1D_EXTRA_DB_ROOTS", str(tmp_path))

    with patch("duckdb.connect", side_effect=RuntimeError("mock")):
        hlth = check_database_health(db_path=None)
        assert hlth.configured_path == "data/t1d.duckdb"

    # When both data/t1d.duckdb and an authentic DB in T1D_DB_SEARCH_DIR exist, T1D_DB_SEARCH_DIR wins
    monkeypatch.delenv("T1D_EXTRA_DB_ROOTS", raising=False)
    search_root = tmp_path / "ext_search"
    jaeb_sub = search_root / "jaeb"
    jaeb_sub.mkdir(parents=True)
    ext_t1d_db = jaeb_sub / "t1d.duckdb"
    ext_t1d_db.write_text("dummy_jaeb")
    monkeypatch.setenv("T1D_DB_SEARCH_DIR", str(search_root))
    assert validate_db_path("t1d.duckdb") == str(ext_t1d_db.resolve())
    assert validate_db_path("jaeb/t1d.duckdb") == str(ext_t1d_db.resolve())
    assert validate_db_path(str(search_root)) == str(ext_t1d_db.resolve())
    assert validate_db_path(str(jaeb_sub)) == str(ext_t1d_db.resolve())
    empty_dir = tmp_path / "empty_dir_no_db"
    empty_dir.mkdir(exist_ok=True)
    assert db_security.normalize_db_shorthand_arg(str(empty_dir)) == str(empty_dir)

    # If t1d.duckdb is absent in search_root, falls back to first *.duckdb in search_root
    ext_t1d_db.unlink()
    alt_ext_db = jaeb_sub / "cohort_jaeb.duckdb"
    alt_ext_db.write_text("dummy_alt")
    assert validate_db_path("t1d.duckdb") == str(alt_ext_db.resolve())

    # Bare filename resolution against T1D_DB_PATH and T1D_DB_SEARCH_DIR
    assert validate_db_path("cohort_jaeb.duckdb") == str(alt_ext_db.resolve())
    mnt_sub = tmp_path / "mnt_data"
    mnt_sub.mkdir(exist_ok=True)
    mnt_db = mnt_sub / "t1d_analytics.duckdb"
    mnt_db.write_text("dummy_mnt")
    monkeypatch.setenv("T1D_DB_PATH", str(mnt_db))
    assert validate_db_path("t1d_analytics.duckdb") == str(mnt_db.resolve())
    assert validate_db_path("unmatched_bare.duckdb") == str(
        (tmp_path / "unmatched_bare.duckdb").resolve()
    )

    # discover_workspace_databases retains valid data/t1d.duckdb even when default_db > 50 MB (deduplicates only by canonical Path.resolve)
    with open(mnt_db, "wb") as f_sparse:
        f_sparse.truncate(55_000_000)
    disc_large = db_security.discover_workspace_databases()
    assert any(d.path.endswith("data/t1d.duckdb") for d in disc_large.databases)
    mnt_db.unlink()
    db_security.discover_workspace_databases()
    monkeypatch.delenv("T1D_DB_PATH", raising=False)

    # If no .duckdb files exist in search_root or data/t1d.duckdb, returns raw t1d.duckdb
    alt_ext_db.unlink()
    (data_sub / "t1d.duckdb").unlink()
    assert validate_db_path("t1d.duckdb") == str(Path("t1d.duckdb").resolve())
    monkeypatch.delenv("T1D_DB_SEARCH_DIR", raising=False)


def test_get_providers_status_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test /api/providers/status reports health and credential readiness for all providers."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test12345")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test12345")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-test-key")

    class MockResp:
        """Mock HTTP response."""

        status = 200

        def read(self) -> bytes:
            """Return empty Ollama models JSON payload."""
            return b'{"models": []}'

        def __enter__(self) -> "MockResp":
            """Enter context."""
            return self

        def __exit__(
            self,
            exc_type: type | None = None,
            exc_val: BaseException | None = None,
            exc_tb: types.TracebackType | None = None,
        ) -> None:
            """Exit context."""

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

    with patch(
        "urllib.request.urlopen",
        side_effect=ConnectionRefusedError("Connection refused"),
    ):
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
    from t1d_analytics.llm_client import is_provider_configured

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
            exc_type: type | None = None,
            exc_val: BaseException | None = None,
            exc_tb: types.TracebackType | None = None,
        ) -> None:
            """Exit context."""

    with patch("urllib.request.urlopen", return_value=MockStatus500()):
        resp = client.get("/api/providers/status")
        assert resp.status_code == 200
        data = resp.json()
        provs = {p["provider"]: p for p in data["providers"]}
        assert provs["ollama"]["healthy"] is False


def test_debug_static_mount(tmp_path: Path) -> None:
    """Test static mount behavior in debug mode via public mount_frontend_static_assets."""
    from t1d_analytics.api import create_app, mount_frontend_static_assets

    isolated_app = create_app()
    with patch.dict(os.environ, {}, clear=True):
        mount_frontend_static_assets(isolated_app)

    existing_dist = tmp_path / "dist"
    existing_dist.mkdir()
    with (
        patch.dict(os.environ, {"DEBUG": "1", "T1D_WEB_DIST_DIR": str(existing_dist)}),
        patch.object(isolated_app, "mount") as mock_mount,
    ):
        mount_frontend_static_assets(isolated_app)
        mock_mount.assert_called_once()

    missing_dist = tmp_path / "missing_dist"
    with patch.dict(os.environ, {"DEBUG": "1"}):
        mount_frontend_static_assets(isolated_app, frontend_dist=missing_dist)


def test_serve_chat_spa_entrypoint_branches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test GET /chats/{chat_id} SPA fallback route when dist/index.html exists and when missing."""
    from t1d_analytics.api import create_app

    dist_with_index = tmp_path / "dist_ok"
    dist_with_index.mkdir()
    index_html = dist_with_index / "index.html"
    index_html.write_text("<!doctype html><html><body>t1d-spa</body></html>")

    monkeypatch.setenv("T1D_WEB_DIST_DIR", str(dist_with_index))
    resp_ok = client.get("/chats/chat-1793964989501-1-476")
    assert resp_ok.status_code == 200
    assert "text/html" in resp_ok.headers["content-type"]
    assert "t1d-spa" in resp_ok.text

    empty_dist = tmp_path / "dist_empty"
    empty_dist.mkdir()
    monkeypatch.delenv("T1D_WEB_DIST_DIR", raising=False)
    isolated_client = TestClient(create_app(frontend_dist=empty_dist))
    resp_404 = isolated_client.get("/chats/chat-1793964989501-1-476")
    assert resp_404.status_code == 404
    assert resp_404.json()["detail"]["error_code"] == "backend.frontendBuildNotFound"


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
    tmp_path: Path,
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

    from dataclasses import replace

    from t1d_analytics.api import create_app
    from t1d_analytics.api_dependencies import (
        _app_services_registry,
        create_default_app_services,
        get_app_services,
        register_default_app_services,
        resolve_default_app_services,
        set_default_app_services,
    )

    isolated_app = create_app(frontend_dist=tmp_path / "custom_dist")
    assert isolated_app is not app
    assert isolated_app.state.services is not app.state.services
    assert isolated_app.state.session_store is not app.state.session_store
    assert isolated_app.state.training_job_manager is not app.state.training_job_manager
    assert isolated_app.state.chat_job_manager is not app.state.chat_job_manager
    assert isolated_app.state.rate_limiter is not app.state.rate_limiter
    assert isolated_app.state.frontend_dist == tmp_path / "custom_dist"

    monkeypatch.setenv("T1D_RATE_LIMITER_BACKEND", "duckdb")
    monkeypatch.setenv("T1D_RATE_LIMIT_DB_PATH", str(tmp_path / "isolated_rl.duckdb"))
    duck_rl_app = create_app()
    duck_rl_app.state.rate_limiter.close()
    monkeypatch.setenv("T1D_RATE_LIMITER_BACKEND", "memory")

    import asyncio

    from fastapi import FastAPI

    from t1d_analytics.api import lifespan
    from t1d_analytics.api_dependencies import AppServices
    from t1d_analytics.api_schemas import ChatRequest

    bare_svc = AppServices()
    bare_schema = bare_svc.build_schema_response(
        str(tmp_path / "missing_bare.duckdb"), "en"
    )
    assert bare_schema.tables == []
    bare_stream = bare_svc.stream_chat_events(
        ChatRequest(message="SELECT 1", model="sql")
    )
    assert hasattr(bare_stream, "__aiter__")

    async def _run_bare_lifespan() -> None:
        """Run lifespan on a bare FastAPI instance with no state.rate_limiter."""
        with (
            patch("t1d_analytics.api.check_database_health") as mock_hlth,
            patch("t1d_analytics.api.run_system_diagnostics"),
            patch("t1d_analytics.api.log_startup_diagnostics"),
        ):
            mock_hlth.return_value = MagicMock(
                exists=True, table_count=1, total_rows=1, has_initial_data=True
            )
            async with lifespan(FastAPI()):
                pass

    asyncio.run(_run_bare_lifespan())

    orig_svc = app.state.services
    set_default_app_services(None)
    assert resolve_default_app_services() is not orig_svc
    register_default_app_services(orig_svc)
    assert resolve_default_app_services() is orig_svc
    app.state.services = None
    dummy_req = MagicMock()
    dummy_req.app = app
    assert get_app_services(dummy_req) is orig_svc
    set_default_app_services(orig_svc, target_app=app)
    assert _app_services_registry.get_default() is orig_svc
    assert get_app_services(dummy_req) is orig_svc

    custom_svc = replace(
        create_default_app_services(target_app=app),
        cloud_provider_models={"groq": ["groq/custom-llama"]},
    )
    app.dependency_overrides[get_app_services] = lambda: custom_svc
    try:
        resp_override = client.get("/api/models")
        assert resp_override.status_code == 200
        override_names = [m["name"] for m in resp_override.json()["models"]]
        assert "groq/custom-llama" in override_names
    finally:
        app.dependency_overrides.pop(get_app_services, None)


def test_libscript_manifest_caching_and_build_hooks() -> None:
    """
    Verify libscript.json caching and build hooks for backend and frontend.

    Ensures:
    - install-t1d-backend-deps reads .venv/.libscript_lock_hash before invoking
      libscript_python_venv, hashes uv.lock, pyproject.toml, and requirements.txt,
      and supports pip install -r requirements.txt.
    - install-t1d-frontend-deps hashes both package.json and package-lock.json
      and writes both node_modules/.libscript_pkg_hash and .libscript_deps.sha256.
    - build-vite-web, frontends.web, and infrastructure.frontends.web compute
      _src_hash using the canonical 16-file + src/public loop and stamp both
      dist/.libscript_src_hash and .libscript_build.sha256.
    """

    class HookSpec(typing.TypedDict, total=False):
        """Schema for a lifecycle hook entry in libscript.json."""

        name: str
        command: str
        cmd_windows: str
        working_dir: str

    class HooksGroup(typing.TypedDict, total=False):
        """Schema for the hooks section in libscript.json."""

        install: list[HookSpec]
        build: list[HookSpec]
        post_start: list[HookSpec]

    class FrontendSpec(typing.TypedDict):
        """Schema for a frontend target in libscript.json."""

        framework: str
        working_dir: str
        source_dir: str
        output_dir: str
        artifact_check: str
        build_command: str

    class InfrastructureSpec(typing.TypedDict):
        """Schema for the infrastructure section in libscript.json."""

        frontends: dict[str, FrontendSpec]
        hooks: HooksGroup
        service_overrides: dict[str, dict[str, str]]

    class ManifestSpec(typing.TypedDict):
        """Schema for relevant sections of libscript.json."""

        hooks: HooksGroup
        frontends: dict[str, FrontendSpec]
        infrastructure: InfrastructureSpec
        service_overrides: dict[str, dict[str, str]]

    manifest_path = Path(__file__).resolve().parent.parent.parent / "libscript.json"
    manifest: ManifestSpec = json.loads(manifest_path.read_text(encoding="utf-8"))

    for ollama_ovr in (
        manifest["service_overrides"]["ollama"],
        manifest["infrastructure"]["service_overrides"]["ollama"],
    ):
        assert ollama_ovr["OLLAMA_LLM_LIBRARY"] == "cpu_avx2"
        assert ollama_ovr["OLLAMA_NUM_GPU"] == "0"
        assert ollama_ovr["OLLAMA_FLASH_ATTENTION"] == "0"
        assert ollama_ovr["OLLAMA_KV_CACHE_TYPE"] == "f16"
        assert ollama_ovr["OLLAMA_CONTEXT_LENGTH"] == "2048"
        assert ollama_ovr["OLLAMA_MAX_QUEUE"] == "16"

    for post_hooks in (
        manifest["hooks"]["post_start"],
        manifest["infrastructure"]["hooks"]["post_start"],
    ):
        smoke_hook = next(
            h for h in post_hooks if h["name"] == "verify-ollama-gemma-2b"
        )
        assert "http://127.0.0.1:11434/api/generate" in smoke_hook["command"]
        assert "ollama pull gemma:2b" in smoke_hook["command"]
        assert "if id ollama >/dev/null 2>&1; then" in smoke_hook["command"]
        assert "${SERVICE_USER:-$(id -un)}" in smoke_hook["command"]
        assert "http://127.0.0.1:11434/api/generate" in smoke_hook["cmd_windows"]

    install_hooks = manifest["hooks"]["install"]
    backend_hook = next(
        h for h in install_hooks if h["name"] == "install-t1d-backend-deps"
    )
    backend_cmd = backend_hook["command"]
    backend_win = backend_hook["cmd_windows"]

    assert "_lock_hash=$(cksum uv.lock pyproject.toml requirements.txt" in backend_cmd
    assert "_old_lock_hash=$(cat .venv/.libscript_lock_hash" in backend_cmd
    assert backend_cmd.index("_old_lock_hash=") < backend_cmd.index(
        "libscript_python_venv"
    )
    assert "pip install -r requirements.txt" in backend_cmd
    assert ".venv\\.libscript_lock_hash" in backend_win
    assert backend_win.index(".venv\\.libscript_lock_hash") < backend_win.index(
        ":libscript_python_venv"
    )
    assert "pip.exe install -r requirements.txt" in backend_win

    frontend_hook = next(
        h for h in install_hooks if h["name"] == "install-t1d-frontend-deps"
    )
    frontend_cmd = frontend_hook["command"]
    frontend_win = frontend_hook["cmd_windows"]

    assert "_pkg_hash=$(cksum package.json package-lock.json" in frontend_cmd
    assert "node_modules/.libscript_pkg_hash" in frontend_cmd
    assert ".libscript_deps.sha256" in frontend_cmd
    assert "npm i --no-audit --no-fund" in frontend_cmd
    assert "package.json package-lock.json" in frontend_win
    assert "node_modules\\.libscript_pkg_hash" in frontend_win
    assert ".libscript_deps.sha256" in frontend_win
    assert "npm i --no-audit --no-fund" in frontend_win

    canonical_16_files = (
        "for _cff_f in package.json package-lock.json pnpm-lock.yaml yarn.lock "
        "angular.json vite.config.ts vite.config.js vite.config.mts tsconfig.json "
        "tsconfig.app.json tsconfig.spec.json index.html postcss.config.js "
        "postcss.config.cjs tailwind.config.js tailwind.config.ts .env .env.production"
    )
    canonical_dir_loop = "for _cff_sub in src public; do"

    build_hook = manifest["hooks"]["build"][0]
    assert build_hook["name"] == "build-vite-web"
    build_cmd = build_hook["command"]
    build_win = build_hook["cmd_windows"]

    assert canonical_16_files in build_cmd
    assert canonical_dir_loop in build_cmd
    assert "dist/.libscript_src_hash" in build_cmd
    assert ".libscript_build.sha256" in build_cmd
    assert ".libscript_build.sha256" in build_win
    assert "dist\\.libscript_src_hash" in build_win

    for fe_cfg in (
        manifest["frontends"]["web"],
        manifest["infrastructure"]["frontends"]["web"],
    ):
        assert fe_cfg["output_dir"] == "dist"
        assert fe_cfg["artifact_check"] == "dist/index.html"
        fe_build_cmd = fe_cfg["build_command"]
        assert canonical_16_files in fe_build_cmd
        assert canonical_dir_loop in fe_build_cmd
        assert "dist/.libscript_src_hash" in fe_build_cmd
        assert ".libscript_build.sha256" in fe_build_cmd
