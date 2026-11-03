"""Database path sandboxing and workspace DuckDB discovery utilities."""

import os
import tempfile
from pathlib import Path
from typing import List, Optional

from t1d_analytics.api_schemas import DatabaseItem, DatabaseListResponse
from t1d_analytics.errors import InvalidDatabasePathError

ALLOWED_DB_EXTENSIONS = (".duckdb", ".db")
DEFAULT_JAEB_EXTERNAL_ROOT = Path("/Volumes/TOSHIBA_EXT/stanford/t1d/jaeb")


def get_jaeb_external_root() -> Path:
    """
    Resolve the external Jaeb clinical dataset directory from `T1D_JAEB_DIR` or `DEFAULT_JAEB_EXTERNAL_ROOT`.

    Returns
    -------
        Path: Candidate path to the external Jaeb dataset root directory.

    """
    env_jaeb = os.environ.get("T1D_JAEB_DIR")
    if env_jaeb and env_jaeb.strip():
        return Path(env_jaeb.strip())
    return DEFAULT_JAEB_EXTERNAL_ROOT


def get_allowed_db_roots() -> List[Path]:
    """
    Return the default list of allowed filesystem directory roots for DuckDB access.

    Returns
    -------
        List[Path]: Resolved allowed directory Paths.

    """
    allowed_env = os.environ.get("T1D_ALLOWED_DB_DIR")
    if allowed_env:
        return [Path(allowed_env).resolve()]

    cwd = Path.cwd().resolve()
    sys_tmp = Path(tempfile.gettempdir()).resolve()
    allowed_dirs = [
        cwd,
        sys_tmp,
        Path("/tmp").resolve(),
        Path("/private/tmp").resolve(),
        Path("/var/tmp").resolve(),
    ]
    env_db = os.environ.get("T1D_DB_PATH")
    if env_db:
        allowed_dirs.append(Path(env_db).resolve().parent)
    jaeb_root = get_jaeb_external_root()
    if jaeb_root.exists():
        allowed_dirs.append(jaeb_root.resolve())
    return allowed_dirs


get_allowed_db_dirs = get_allowed_db_roots


def resolve_default_db_path(db_path: Optional[str] = None) -> str:
    """
    Resolve the target DuckDB database path from an explicit argument, `T1D_DB_PATH`, or `data/t1d.duckdb`.

    Args:
    ----
        db_path: Optional explicit database file path.

    Returns:
    -------
        str: Resolved candidate path string prior to directory sandboxing.

    """
    raw_path = db_path or os.environ.get("T1D_DB_PATH", "t1d.duckdb")
    if raw_path == "t1d.duckdb" and not Path(raw_path).exists():
        env_db_fallback = os.environ.get("T1D_DB_PATH")
        if env_db_fallback and Path(env_db_fallback).exists():
            raw_path = env_db_fallback
        elif not env_db_fallback and Path("data/t1d.duckdb").exists():
            raw_path = "data/t1d.duckdb"
    return raw_path


def _resolve_contained_path(
    raw_path: str,
    allowed_dirs: Optional[List[Path]] = None,
) -> Path:
    """
    Validate that `raw_path` contains no null bytes and resolves inside an allowed root directory.

    Args:
    ----
        raw_path: Unvalidated path string.
        allowed_dirs: Optional list of allowed directory roots.

    Returns:
    -------
        Path: Resolved filesystem Path within an allowed root.

    Raises:
    ------
        InvalidDatabasePathError: If the path contains null bytes or traverses outside allowed roots.

    """
    if "\0" in raw_path:
        raise InvalidDatabasePathError(
            params={"path": raw_path},
        )

    resolved_path = Path(raw_path).resolve()

    if allowed_dirs is None:
        allowed_dirs = get_allowed_db_roots()

    is_allowed = False
    for allowed_dir in allowed_dirs:
        resolved_allowed = allowed_dir.resolve()
        if resolved_path == resolved_allowed or resolved_path.is_relative_to(
            resolved_allowed
        ):
            is_allowed = True
            break

    if not is_allowed:
        raise InvalidDatabasePathError(
            params={"path": str(raw_path)},
        )

    return resolved_path


def validate_db_path(
    db_path: Optional[str] = None,
    allowed_dirs: Optional[List[Path]] = None,
) -> str:
    """
    Validate and resolve a DuckDB database path against allowed directory containment and `.duckdb`/`.db` extensions.

    Args:
    ----
        db_path: The requested database file path, or None for default.
        allowed_dirs: Optional list of allowed directory roots.

    Returns:
    -------
        str: Validated string path to the database file.

    Raises:
    ------
        InvalidDatabasePathError: If the path traverses outside allowed boundaries, contains null bytes,
            or does not end with `.duckdb` or `.db`.

    """
    raw_path = resolve_default_db_path(db_path)
    resolved_path = _resolve_contained_path(raw_path, allowed_dirs=allowed_dirs)
    if resolved_path.suffix.lower() not in ALLOWED_DB_EXTENSIONS:
        raise InvalidDatabasePathError(
            params={"path": str(raw_path)},
        )
    return str(resolved_path)


def validate_workspace_artifact_path(
    artifact_path: str,
    allowed_dirs: Optional[List[Path]] = None,
) -> str:
    """
    Validate and resolve a training dataset or checkpoint directory path against allowed directory containment.

    Args:
    ----
        artifact_path: The requested dataset file or output directory path.
        allowed_dirs: Optional list of allowed directory roots.

    Returns:
    -------
        str: Validated string path to the workspace artifact.

    Raises:
    ------
        InvalidDatabasePathError: If the path is empty, contains null bytes, or traverses outside allowed roots.

    """
    cleaned = artifact_path.strip()
    if not cleaned:
        raise InvalidDatabasePathError(
            params={"path": artifact_path},
        )
    resolved_path = _resolve_contained_path(cleaned, allowed_dirs=allowed_dirs)
    return str(resolved_path)


def discover_workspace_databases() -> DatabaseListResponse:
    """
    Discover and list available DuckDB database files in the workspace.

    Returns
    -------
        DatabaseListResponse: Found database items and current default path.

    """
    default_db = resolve_default_db_path()
    all_files: List[Path] = []
    if default_db:
        p_def = Path(default_db)
        if p_def.exists():
            all_files.append(p_def)
    search_dirs: List[Path] = [Path(), Path("./data"), get_jaeb_external_root()]
    for s_dir in search_dirs:
        if s_dir.exists():
            all_files.extend(
                p
                for p in s_dir.glob("*.duckdb")
                if p.name not in ("sessions.duckdb", "t1d_sessions.duckdb")
            )

    found_dbs: List[DatabaseItem] = []
    seen_paths = set()
    for p in all_files:
        resolved = str(p.resolve())
        if resolved not in seen_paths:
            seen_paths.add(resolved)
            found_dbs.append(
                DatabaseItem(
                    name=p.name,
                    path=str(p),
                    size_bytes=p.stat().st_size if p.exists() else 0,
                )
            )

    return DatabaseListResponse(databases=found_dbs, current_db=default_db)
