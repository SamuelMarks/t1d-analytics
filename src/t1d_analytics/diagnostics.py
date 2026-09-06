"""System diagnostics and health validation module for T1D Analytics."""

import json
import logging
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional

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
        }


@dataclass
class OllamaStatus:
    """Health and model availability status for Ollama LLM service."""

    accessible: bool
    available_models: List[str] = field(default_factory=list)
    message: str = ""
    remediation: Optional[str] = None

    def to_dict(self) -> dict[str, object]:
        """Convert Ollama status to dictionary representation."""
        return {
            "accessible": self.accessible,
            "available_models": self.available_models,
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

    def to_dict(self) -> dict[str, object]:
        """Convert system health to dictionary representation."""
        return {
            "status": self.status,
            "backend_accessible": self.backend_accessible,
            "version": self.version,
            "database": self.database.to_dict(),
            "ollama": self.ollama.to_dict(),
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
        )

    # 3. Attempt DuckDB connection
    try:
        conn = duckdb.connect(target_path_str, read_only=True)
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
        )

    # 5. Check for initial/default clinical trial datasets
    found_expected = [t for t in tables if t.lower() in EXPECTED_T1D_TABLES]

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
    )


def check_ollama_health(
    ollama_url: str = "http://127.0.0.1:11434/api/tags",
    recommended_model: str = "gemma4",
    timeout: float = 2.0,
    lang: Optional[str] = None,
) -> OllamaStatus:
    """
    Check availability of the local Ollama LLM service and recommended models.

    Args:
    ----
        ollama_url: URL to Ollama tags endpoint.
        recommended_model: Model name to check.
        timeout: Request timeout in seconds.
        lang: Optional language code for localized messages.

    Returns:
    -------
        OllamaStatus describing service availability and model list.

    """
    _ = get_translator(lang)
    try:
        req = urllib.request.Request(ollama_url)
        with urllib.request.urlopen(req, timeout=timeout) as response:
            data = json.loads(response.read().decode())

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
            )

        return OllamaStatus(
            accessible=True,
            available_models=models,
            message=_("Ollama LLM service is online and ready."),
            remediation=None,
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
    elif ollama.accessible:
        logger.warning(f"[WARNING] Ollama LLM Service: {ollama.message}")
        if ollama.remediation:
            logger.warning(f"  -> Action: {ollama.remediation}")
    else:
        logger.warning(f"[WARNING] Ollama LLM Service: {ollama.message}")
        if ollama.remediation:
            logger.warning(f"  -> Action: {ollama.remediation}")

    logger.info(f"System Status: {health.status.upper()}")
    logger.info("=" * 60)
