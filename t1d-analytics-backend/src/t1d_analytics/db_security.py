"""Database path sandboxing and workspace DuckDB discovery utilities."""

import os
import tempfile
from pathlib import Path
from typing import List, Optional

from t1d_analytics.api_schemas import DatabaseItem, DatabaseListResponse
from t1d_analytics.errors import InvalidDatabasePathError


def get_allowed_db_roots() -> List[Path]:
    """
    Return the default list of allowed filesystem directory roots for DuckDB access.

    Returns
    -------
        List[Path]: Resolved allowed directory Paths.

    """
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
    return allowed_dirs


get_allowed_db_dirs = get_allowed_db_roots


def validate_db_path(
    db_path: Optional[str] = None,
    allowed_dirs: Optional[List[Path]] = None,
) -> str:
    """
    Validate and resolve a database path against allowed directory containment.

    Args:
    ----
        db_path: The requested database file path, or None for default.
        allowed_dirs: Optional list of allowed directory roots.

    Returns:
    -------
        str: Validated string path to the database file.

    Raises:
    ------
        InvalidDatabasePathError: If the path traverses outside allowed boundaries or contains null bytes.

    """
    raw_path = db_path or os.environ.get("T1D_DB_PATH", "t1d.duckdb")
    if raw_path == "t1d.duckdb" and not Path(raw_path).exists():
        env_db_fallback = os.environ.get("T1D_DB_PATH")
        if env_db_fallback and Path(env_db_fallback).exists():
            raw_path = env_db_fallback
        elif not env_db_fallback and Path("data/t1d.duckdb").exists():
            raw_path = "data/t1d.duckdb"

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

    return str(resolved_path)


def discover_workspace_databases() -> DatabaseListResponse:
    """
    Discover and list available DuckDB database files in the workspace.

    Returns
    -------
        DatabaseListResponse: Found database items and current default path.

    """
    default_db = os.environ.get("T1D_DB_PATH", "t1d.duckdb")
    if (
        default_db == "t1d.duckdb"
        and not Path(default_db).exists()
        and Path("data/t1d.duckdb").exists()
    ):
        default_db = "data/t1d.duckdb"
    all_files: List[Path] = []
    if default_db:
        p_def = Path(default_db)
        if p_def.exists():
            all_files.append(p_def)
    for s_dir in (Path("."), Path("./data")):
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
