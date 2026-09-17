"""System diagnostics and health validation module for T1D Analytics."""

import json
import logging
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union

import duckdb

from t1d_analytics.i18n import get_translator

logger = logging.getLogger(__name__)

# Clinical trial tables commonly expected in a populated T1D dataset
EXPECTED_T1D_TABLES = {
    "patients",
    "cgms",
    "cgm",
    "visits",
    "dclp3",
    "dclp5",
    "pedap",
    "loop",
    "flair",
}


class DatabaseStatusCode(str, Enum):
    """Enumeration of database status codes."""

    HEALTHY = "healthy"
    MISSING_FILE = "missing_file"
    EMPTY_FILE = "empty_file"
    UNREADABLE = "unreadable"
    EMPTY_DB = "empty_db"
    MISSING_INITIAL_DATA = "missing_initial_data"


def check_disk_space(path: Union[str, Path]) -> dict[str, int]:
    """
    Check total, used, and free disk space for a path.

    Args:
    ----
        path: Directory or file path to check.

    Returns:
    -------
        Dictionary with total_bytes, used_bytes, and free_bytes.

    """
    import shutil

    try:
        p = Path(path).resolve()
        target = p.parent
        if not target.exists():
            target = Path.cwd()
    except Exception:
        target = Path.cwd()
    usage = shutil.disk_usage(str(target))
    return {
        "total_bytes": usage.total,
        "used_bytes": usage.used,
        "free_bytes": usage.free,
    }


@dataclass
class DatabaseStatus:
    """Detailed health status for the DuckDB database."""

    configured_path: str
    exists: bool
    connected: bool
    status_code: DatabaseStatusCode
    message: str
    remediation: Optional[str] = None
    table_count: int = 0
    tables: List[str] = field(default_factory=list)
    has_initial_data: bool = False
    file_size_bytes: int = 0
    writable: bool = False
    integrity_ok: bool = True
    disk_free_bytes: int = 0

    def to_dict(self) -> dict[str, object]:
        """Convert database status to dictionary representation."""
        return {
            "configured_path": self.configured_path,
            "exists": self.exists,
            "connected": self.connected,
            "status_code": self.status_code.value,
            "message": self.message,
            "remediation": self.remediation,
            "table_count": self.table_count,
            "tables": self.tables,
            "has_initial_data": self.has_initial_data,
            "file_size_bytes": self.file_size_bytes,
            "writable": self.writable,
            "integrity_ok": self.integrity_ok,
            "disk_free_bytes": self.disk_free_bytes,
        }


@dataclass
class OllamaStatus:
    """Health and model availability status for Ollama LLM service."""

    accessible: bool
    available_models: List[str] = field(default_factory=list)
    message: str = ""
    remediation: Optional[str] = None
    version: Optional[str] = None

    def to_dict(self) -> dict[str, object]:
        """Convert Ollama status to dictionary representation."""
        return {
            "accessible": self.accessible,
            "available_models": self.available_models,
            "message": self.message,
            "remediation": self.remediation,
            "version": self.version,
        }


@dataclass
class CloudProviderStatus:
    """Health and readiness status for external cloud LLM provider."""

    provider: str
    configured: bool
    sdk_installed: bool
    api_key_set: bool
    message: str = ""
    remediation: Optional[str] = None

    def to_dict(self) -> dict[str, object]:
        """Convert cloud provider status to dictionary representation."""
        return {
            "provider": self.provider,
            "configured": self.configured,
            "sdk_installed": self.sdk_installed,
            "api_key_set": self.api_key_set,
            "message": self.message,
            "remediation": self.remediation,
        }


@dataclass
class SystemHealth:
    """Aggregated system health summary."""

    status: str
    backend_accessible: bool
    version: str
    database: DatabaseStatus
    ollama: OllamaStatus
    providers: Dict[str, CloudProviderStatus] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        """Convert system health to dictionary representation."""
        return {
            "status": self.status,
            "backend_accessible": self.backend_accessible,
            "version": self.version,
            "database": self.database.to_dict(),
            "ollama": self.ollama.to_dict(),
            "providers": {k: v.to_dict() for k, v in self.providers.items()},
        }


def check_database_health(
    db_path: Optional[str] = None, lang: Optional[str] = None
) -> DatabaseStatus:
    """
    Validate the configuration, presence, connectivity, and data contents of the database.

    Args:
    ----
        db_path: Path to the DuckDB database file. Defaults to T1D_DB_PATH or 't1d.duckdb'.
        lang: Optional language code for localized messages.

    Returns:
    -------
        DatabaseStatus containing diagnostic results and actionable remediation advice.

    """
    _ = get_translator(lang)
    target_path_str = db_path or os.environ.get("T1D_DB_PATH", "t1d.duckdb")
    target_path = Path(target_path_str)

    # 1. Check if database file exists on disk
    disk_free = check_disk_space(target_path)["free_bytes"]
    file_size = 0
    writable = False
    try:
        if target_path.exists():
            file_size = target_path.stat().st_size
            writable = os.access(target_path_str, os.W_OK)
    except Exception:
        pass

    try:
        if not target_path.exists():
            msg = _("Database file '{}' does not exist.", target_path_str)
            remediation = _(
                "Run 't1d-analytics load --data-dir <data_dir> --db {}' to create and populate the database.",
                target_path_str,
            )
            return DatabaseStatus(
                configured_path=target_path_str,
                exists=False,
                connected=False,
                status_code=DatabaseStatusCode.MISSING_FILE,
                message=msg,
                remediation=remediation,
                file_size_bytes=0,
                writable=False,
                integrity_ok=True,
                disk_free_bytes=disk_free,
            )

        # 2. Check if file is non-empty
        if target_path.stat().st_size == 0:
            msg = _("Database file '{}' is empty (0 bytes).", target_path_str)
            remediation = _(
                "Populate the database with 't1d-analytics load --data-dir <data_dir> --db {}'.",
                target_path_str,
            )
            return DatabaseStatus(
                configured_path=target_path_str,
                exists=True,
                connected=False,
                status_code=DatabaseStatusCode.EMPTY_FILE,
                message=msg,
                remediation=remediation,
                file_size_bytes=file_size,
                writable=writable,
                integrity_ok=True,
                disk_free_bytes=disk_free,
            )
    except OSError as e:
        msg = _(
            "Cannot access database file '{}': {}",
            target_path_str,
            e,
        )
        return DatabaseStatus(
            configured_path=target_path_str,
            exists=True,
            connected=False,
            status_code=DatabaseStatusCode.UNREADABLE,
            message=msg,
            remediation=_("Verify filesystem read permissions for the database file."),
            file_size_bytes=file_size,
            writable=writable,
            integrity_ok=False,
            disk_free_bytes=disk_free,
        )

    # 3. Attempt DuckDB connection and run integrity check
    integrity_ok = True
    try:
        conn = duckdb.connect(target_path_str, read_only=True)
        try:
            conn.execute("SELECT * FROM duckdb_tables() LIMIT 1")
        except Exception:
            integrity_ok = False
        raw_tables = conn.execute("SHOW TABLES").fetchall()
        tables = [row[0] for row in raw_tables]
        conn.close()
    except Exception as e:
        msg = _(
            "Failed to open DuckDB database '{}': {}",
            target_path_str,
            e,
        )
        remediation = _(
            "Check that the database is not corrupted or locked by another process."
        )
        return DatabaseStatus(
            configured_path=target_path_str,
            exists=True,
            connected=False,
            status_code=DatabaseStatusCode.UNREADABLE,
            message=msg,
            remediation=remediation,
            file_size_bytes=file_size,
            writable=writable,
            integrity_ok=False,
            disk_free_bytes=disk_free,
        )

    # 4. Check table count
    table_count = len(tables)
    if table_count == 0:
        msg = _(
            "Database '{}' connected successfully but contains 0 tables.",
            target_path_str,
        )
        remediation = _(
            "Run 't1d-analytics load --data-dir <data_dir> --db {}' to load clinical trial data.",
            target_path_str,
        )
        return DatabaseStatus(
            configured_path=target_path_str,
            exists=True,
            connected=True,
            status_code=DatabaseStatusCode.EMPTY_DB,
            message=msg,
            remediation=remediation,
            table_count=0,
            tables=[],
            has_initial_data=False,
            file_size_bytes=file_size,
            writable=writable,
            integrity_ok=integrity_ok,
            disk_free_bytes=disk_free,
        )

    # 5. Check for initial/default clinical trial datasets
    found_expected = [
        t
        for t in tables
        if any(
            t.lower() == exp
            or t.lower().endswith(f"_{exp}")
            or t.lower().startswith(f"{exp}_")
            for exp in EXPECTED_T1D_TABLES
        )
    ]

    # Specifically detect if lacking default standard T1D clinical trial datasets
    if not found_expected:
        msg = _(
            "Database '{}' contains {} custom table(s), but lacks standard T1D clinical trial datasets (e.g. patients, cgms, dclp3).",
            target_path_str,
            table_count,
        )
        remediation = _(
            "Download and load public clinical trial datasets using 't1d-analytics download' and 't1d-analytics load'."
        )
        return DatabaseStatus(
            configured_path=target_path_str,
            exists=True,
            connected=True,
            status_code=DatabaseStatusCode.MISSING_INITIAL_DATA,
            message=msg,
            remediation=remediation,
            table_count=table_count,
            tables=tables,
            has_initial_data=False,
            file_size_bytes=file_size,
            writable=writable,
            integrity_ok=integrity_ok,
            disk_free_bytes=disk_free,
        )

    msg = _(
        "Database '{}' is healthy with {} table(s) ready.",
        target_path_str,
        table_count,
    )
    return DatabaseStatus(
        configured_path=target_path_str,
        exists=True,
        connected=True,
        status_code=DatabaseStatusCode.HEALTHY,
        message=msg,
        remediation=None,
        table_count=table_count,
        tables=tables,
        has_initial_data=True,
        file_size_bytes=file_size,
        writable=writable,
        integrity_ok=integrity_ok,
        disk_free_bytes=disk_free,
    )


def check_ollama_health(
    ollama_url: Optional[str] = None,
    recommended_model: str = "gemma4",
    timeout: float = 2.0,
    lang: Optional[str] = None,
) -> OllamaStatus:
    """
    Check availability of the local Ollama LLM service and recommended models.

    Args:
    ----
        ollama_url: Optional URL to Ollama tags endpoint (defaults to OLLAMA_HOST or http://127.0.0.1:11434).
        recommended_model: Model name to check.
        timeout: Request timeout in seconds.
        lang: Optional language code for localized messages.

    Returns:
    -------
        OllamaStatus describing service availability and model list.

    """
    _ = get_translator(lang)
    host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
    if not host.startswith(("http://", "https://")):
        host = f"http://{host}"

    if ollama_url is None:
        ollama_url = f"{host}/api/tags"

    version: Optional[str] = None
    try:
        req = urllib.request.Request(ollama_url)
        with urllib.request.urlopen(req, timeout=timeout) as response:
            data = json.loads(response.read().decode())

        try:
            v_url = f"{host}/api/version"
            v_req = urllib.request.Request(v_url)
            with urllib.request.urlopen(v_req, timeout=timeout) as v_resp:
                v_data = json.loads(v_resp.read().decode())
                version = str(v_data.get("version", "")) or None
        except Exception:
            version = None

        models = [m.get("name") for m in data.get("models", []) if m.get("name")]
        has_recommended = any(recommended_model in m for m in models)

        if not has_recommended:
            msg = _(
                "Ollama is accessible, but recommended model '{}' is not pulled.",
                recommended_model,
            )
            remediation = _(
                "Run 'ollama pull {}' to enable full Natural Language query translation.",
                recommended_model,
            )
            return OllamaStatus(
                accessible=True,
                available_models=models,
                message=msg,
                remediation=remediation,
                version=version,
            )

        return OllamaStatus(
            accessible=True,
            available_models=models,
            message=_("Ollama LLM service is online and ready."),
            remediation=None,
            version=version,
        )
    except (urllib.error.URLError, TimeoutError, OSError, Exception) as e:
        logger.debug(f"Ollama health check failed: {e}")
        msg = _("Cannot connect to local Ollama LLM service at {}.", ollama_url)
        remediation = _(
            "Start Ollama or select 'Literal SQL' mode in the web interface to query directly."
        )
        return OllamaStatus(
            accessible=False,
            available_models=[],
            message=msg,
            remediation=remediation,
        )


def check_provider_health(
    provider: str, lang: Optional[str] = None
) -> CloudProviderStatus:
    """
    Validate SDK installation and API key configuration for a cloud LLM provider.

    Args:
    ----
        provider: Provider identifier ('openai', 'anthropic', 'google').
        lang: Optional language code for localized messages.

    Returns:
    -------
        CloudProviderStatus: Diagnostic details for the specified provider.

    """
    _ = get_translator(lang)
    prov_key = provider.lower().strip()

    try:
        import any_llm  # type: ignore[import-not-found]  # noqa: F401

        sdk_installed = True
    except ImportError:
        sdk_installed = False

    key_env_vars: dict[str, list[str]] = {
        "openai": ["OPENAI_API_KEY"],
        "anthropic": ["ANTHROPIC_API_KEY"],
        "google": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
    }

    env_vars = key_env_vars.get(prov_key, [f"{prov_key.upper()}_API_KEY"])
    api_key_set = any(bool(os.environ.get(var)) for var in env_vars)

    configured = sdk_installed and api_key_set
    if configured:
        msg = f"Provider '{provider}' is configured and ready."
        remediation = None
    elif not sdk_installed:
        msg = f"SDK 'any-llm-sdk[{provider}]' is not installed."
        remediation = (
            f"Run 'pip install any-llm-sdk[{provider}]' to enable {provider} support."
        )
    else:
        msg = f"API key environment variable ({', '.join(env_vars)}) is not set."
        remediation = f"Set {env_vars[0]} to enable {provider} queries."

    return CloudProviderStatus(
        provider=provider,
        configured=configured,
        sdk_installed=sdk_installed,
        api_key_set=api_key_set,
        message=msg,
        remediation=remediation,
    )


def get_system_health(
    db_path: Optional[str] = None, lang: Optional[str] = None
) -> SystemHealth:
    """
    Compute comprehensive system health for backend, database, and LLM services.

    Args:
    ----
        db_path: Optional custom database path.
        lang: Optional language code.

    Returns:
    -------
        SystemHealth object containing individual component statuses and overall status.

    """
    db_status = check_database_health(db_path=db_path, lang=lang)
    ollama_status = check_ollama_health(lang=lang)
    providers_status: Dict[str, CloudProviderStatus] = {
        p: check_provider_health(p, lang=lang)
        for p in ("openai", "anthropic", "google")
    }

    overall_status = "healthy"
    if not db_status.connected or db_status.status_code in (
        DatabaseStatusCode.MISSING_FILE,
        DatabaseStatusCode.UNREADABLE,
    ):
        overall_status = "error"
    elif not db_status.has_initial_data or not ollama_status.accessible:
        overall_status = "degraded"

    return SystemHealth(
        status=overall_status,
        backend_accessible=True,
        version="0.1.0",
        database=db_status,
        ollama=ollama_status,
        providers=providers_status,
    )


def log_startup_diagnostics(db_path: Optional[str] = None) -> None:
    """
    Print prominent startup diagnostics to console logs when the backend launches.

    Args:
    ----
        db_path: Optional custom database path.

    """
    health = get_system_health(db_path=db_path)
    db = health.database
    ollama = health.ollama

    logger.info("=" * 60)
    logger.info("T1D Analytics Backend Service - Startup Diagnostics")
    logger.info("=" * 60)

    # Database report
    if db.status_code == DatabaseStatusCode.HEALTHY:
        logger.info(f"[OK] Database ({db.configured_path}): {db.message}")
    elif db.status_code in (
        DatabaseStatusCode.EMPTY_DB,
        DatabaseStatusCode.MISSING_INITIAL_DATA,
    ):
        logger.warning(f"[WARNING] Database ({db.configured_path}): {db.message}")
        if db.remediation:
            logger.warning(f"  -> Action: {db.remediation}")
    else:
        logger.error(f"[ERROR] Database ({db.configured_path}): {db.message}")
        if db.remediation:
            logger.error(f"  -> Action: {db.remediation}")

    # Ollama report
    if ollama.accessible and not ollama.remediation:
        logger.info(f"[OK] Ollama LLM Service: {ollama.message}")
    else:
        logger.warning(f"[WARNING] Ollama LLM Service: {ollama.message}")
        if ollama.remediation:
            logger.warning(f"  -> Action: {ollama.remediation}")

    logger.info(f"System Status: {health.status.upper()}")
    logger.info("=" * 60)
