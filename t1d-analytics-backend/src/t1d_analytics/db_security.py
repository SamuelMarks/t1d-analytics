"""Database path sandboxing and workspace DuckDB discovery utilities."""

import os
import tempfile
from pathlib import Path

from t1d_analytics.api_schemas import DatabaseItem, DatabaseListResponse
from t1d_analytics.errors import InvalidDatabasePathError

ALLOWED_DB_EXTENSIONS = (".duckdb", ".db")
DEFAULT_WORKSPACE_DATA_ROOT = Path("./data")
REMOTE_T1D_DB_SEARCH_ROOT = Path("/mnt/data/t1d")
REMOTE_MNT_DATA_ROOT = Path("/mnt/data")
REMOTE_DISCOVERY_DB_CANDIDATES: tuple[Path, ...] = (
    REMOTE_T1D_DB_SEARCH_ROOT / "jaeb" / "t1d.duckdb",
    REMOTE_T1D_DB_SEARCH_ROOT / "t1d.duckdb",
    REMOTE_MNT_DATA_ROOT / "t1d_analytics.duckdb",
)
DEFAULT_JAEB_EXTERNAL_ROOT = DEFAULT_WORKSPACE_DATA_ROOT
DEFAULT_DB_SEARCH_DIRS: tuple[Path, ...] = (
    REMOTE_T1D_DB_SEARCH_ROOT,
    DEFAULT_WORKSPACE_DATA_ROOT,
)
DEFAULT_T1D_DB_SEARCH_DIRS: tuple[Path, ...] = (
    REMOTE_T1D_DB_SEARCH_ROOT,
    REMOTE_T1D_DB_SEARCH_ROOT / "jaeb",
    DEFAULT_WORKSPACE_DATA_ROOT,
    DEFAULT_WORKSPACE_DATA_ROOT / "jaeb",
)
DB_SEARCH_ENV_VARS: tuple[str, ...] = (
    "T1D_DB_SEARCH_DIR",
    "T1D_DB_SEARCH_PATH",
    "T1D_JAEB_DIR",
    "T1D_DB_DIR",
)


def _split_search_dir_spec(raw_spec: str) -> list[Path]:
    r"""
    Parse a single path or delimiter-separated list of search directories into `Path` objects.

    Supports `os.pathsep` (`:` on POSIX, `;` on Windows), `:`, and `,` delimiters while
    preserving Windows drive-letter prefixes (e.g., `C:\...`).

    Args:
        raw_spec: Raw environment variable value specifying one or more directory paths.

    Returns:
        List[Path]: Non-empty parsed directory `Path` objects in order.

    """
    cleaned = raw_spec.strip()
    if not cleaned:
        return []
    if ";" in cleaned:
        parts = cleaned.split(";")
    elif "," in cleaned:
        parts = cleaned.split(",")
    elif ":" in cleaned and not (len(cleaned) >= 2 and cleaned[1] == ":"):
        parts = cleaned.split(":")
    else:
        parts = [cleaned]
    return [Path(part.strip()) for part in parts if part.strip()]


def _safe_path_exists(candidate: Path) -> bool:
    """
    Check whether `candidate` exists on disk, returning `False` on `OSError`.

    Args:
        candidate: Filesystem `Path` to probe.

    Returns:
        bool: `True` when `candidate.exists()` succeeds and returns `True`.

    """
    try:
        return candidate.exists()
    except OSError:
        return False


def _expand_jaeb_candidates(roots: list[Path]) -> list[Path]:
    """
    Expand each root directory `Path` to include both the root and its `jaeb` subdirectory.

    Args:
        roots: Base directory `Path` objects.

    Returns:
        List[Path]: Expanded list of directory `Path` objects.

    """
    expanded: list[Path] = []
    for candidate in roots:
        if candidate.name != "jaeb":
            expanded.extend((candidate, candidate / "jaeb"))
        else:
            expanded.append(candidate)
    return expanded


def get_configured_db_search_dirs() -> list[Path]:
    """
    Return the ordered, deduplicated list of database search directories from env vars and defaults.

    Inspects `T1D_DB_SEARCH_DIR`, `T1D_DB_SEARCH_PATH`, `T1D_JAEB_DIR`, and `T1D_DB_DIR`.
    When no env var is set or when none of the explicitly configured paths exist on the
    current host, automatically appends `DEFAULT_DB_SEARCH_DIRS` and their `/jaeb`
    subdirectories.

    Returns:
        List[Path]: Ordered, deduplicated list of search directory `Path` objects.

    """
    explicit_roots: list[Path] = []
    for env_name in DB_SEARCH_ENV_VARS:
        explicit_roots.extend(_split_search_dir_spec(os.environ.get(env_name) or ""))

    explicit_expanded = _expand_jaeb_candidates(explicit_roots)
    include_defaults = not explicit_expanded or not any(
        _safe_path_exists(p) for p in explicit_expanded
    )

    ordered_defaults: list[Path] = []
    if include_defaults:
        existing_defaults = [d for d in DEFAULT_DB_SEARCH_DIRS if _safe_path_exists(d)]
        missing_defaults = [
            d for d in DEFAULT_DB_SEARCH_DIRS if not _safe_path_exists(d)
        ]
        ordered_defaults = _expand_jaeb_candidates(existing_defaults + missing_defaults)

    configured: list[Path] = []
    seen: set[str] = set()
    for candidate in (*explicit_expanded, *ordered_defaults):
        key = str(candidate)
        if key not in seen:
            seen.add(key)
            configured.append(candidate)
    return configured


def get_t1d_db_search_dirs() -> list[Path]:
    """
    Return all candidate T1D database search directories (configured env vars plus workspace/remote defaults).

    Includes any directories from `get_configured_db_search_dirs()` followed by
    `DEFAULT_T1D_DB_SEARCH_DIRS`.

    Returns:
        List[Path]: Ordered, deduplicated list of search directory `Path` objects.

    """
    dirs: list[Path] = list(get_configured_db_search_dirs())
    seen: set[str] = {str(d) for d in dirs}
    for candidate in DEFAULT_T1D_DB_SEARCH_DIRS:
        key = str(candidate)
        if key not in seen:
            seen.add(key)
            dirs.append(candidate)
    return dirs


def get_jaeb_external_root() -> Path:
    """
    Resolve the external T1D / Jaeb clinical dataset directory from env vars or workspace defaults.

    Checks `DB_SEARCH_ENV_VARS` first; returns the first path that exists on disk among
    `[configured_paths..., *DEFAULT_DB_SEARCH_DIRS]`, falling back to the first configured
    path or `DEFAULT_JAEB_EXTERNAL_ROOT` when none exist on disk.

    Returns:
        Path: Candidate path to the external T1D / Jaeb dataset root directory.

    """
    configured_paths = [
        p
        for env_name in DB_SEARCH_ENV_VARS
        for p in _split_search_dir_spec(os.environ.get(env_name) or "")
    ]
    for candidate in (
        *configured_paths,
        *DEFAULT_DB_SEARCH_DIRS,
    ):
        if _safe_path_exists(candidate):
            return candidate
    if configured_paths:
        return configured_paths[0]
    return DEFAULT_JAEB_EXTERNAL_ROOT


def get_recommended_jaeb_db_path() -> str:
    """
    Resolve the recommended authentic Jaeb DuckDB file path for the current host.

    Honors `T1D_JAEB_DB_PATH` when explicitly set; otherwise searches
    `get_configured_db_search_dirs()` for an existing `t1d.duckdb` file on disk,
    falling back to `<external_root>/jaeb/t1d.duckdb`.

    Returns:
        str: Recommended filesystem path to the authentic Jaeb `t1d.duckdb` file.

    """
    explicit_jaeb_db = (os.environ.get("T1D_JAEB_DB_PATH") or "").strip()
    if explicit_jaeb_db:
        return explicit_jaeb_db
    for search_dir in get_configured_db_search_dirs():
        candidate = search_dir / "t1d.duckdb"
        if _safe_path_exists(candidate):
            return str(candidate.resolve())
    ext_root = get_jaeb_external_root()
    if ext_root.name == "jaeb":
        return str(ext_root / "t1d.duckdb")
    return str(ext_root / "jaeb" / "t1d.duckdb")


def get_allowed_db_roots() -> list[Path]:
    """
    Return the default list of allowed filesystem directory roots for DuckDB access.

    Restricts access to the workspace directory, an application-scoped temp subdirectory
    (`<tempdir>/t1d-analytics`), `/mnt/data/t1d`, and explicitly configured environment
    roots (`T1D_EXTRA_DB_ROOTS`, `T1D_JAEB_DB_PATH`, `T1D_DB_PATH`, `T1D_SESSIONS_DB*`,
    and existing `get_t1d_db_search_dirs()`). Excludes world-writable `/tmp` and `/var/tmp`.

    Returns:
        List[Path]: Resolved allowed directory Paths.

    """
    allowed_env = os.environ.get("T1D_ALLOWED_DB_DIR")
    if allowed_env:
        return [Path(allowed_env).resolve()]

    cwd = Path.cwd().resolve()
    app_tmp = (Path(tempfile.gettempdir()) / "t1d-analytics").resolve()
    allowed_dirs: list[Path] = [
        cwd,
        (cwd / "data").resolve(),
        app_tmp,
        REMOTE_T1D_DB_SEARCH_ROOT.resolve(),
        (REMOTE_T1D_DB_SEARCH_ROOT / "jaeb").resolve(),
    ]
    if _safe_path_exists(REMOTE_MNT_DATA_ROOT):
        resolved_mnt = REMOTE_MNT_DATA_ROOT.resolve()
        if resolved_mnt not in allowed_dirs:
            allowed_dirs.append(resolved_mnt)
    for extra_root in _split_search_dir_spec(
        os.environ.get("T1D_EXTRA_DB_ROOTS") or ""
    ):
        resolved_extra = extra_root.resolve()
        if resolved_extra not in allowed_dirs:
            allowed_dirs.append(resolved_extra)
    for file_env_name in (
        "T1D_DB_PATH",
        "T1D_JAEB_DB_PATH",
        "T1D_SESSIONS_DB",
        "T1D_SESSIONS_DB_PATH",
    ):
        env_file_val = (os.environ.get(file_env_name) or "").strip()
        if env_file_val:
            env_parent = Path(env_file_val).resolve().parent
            if env_parent not in allowed_dirs:
                allowed_dirs.append(env_parent)
    for search_dir in get_t1d_db_search_dirs():
        if _safe_path_exists(search_dir):
            resolved_search = search_dir.resolve()
            if resolved_search not in allowed_dirs:
                allowed_dirs.append(resolved_search)
    return allowed_dirs


get_allowed_db_dirs = get_allowed_db_roots


def _iter_duckdb_files_in_dir(
    search_dir: Path, include_subdirs: bool = False
) -> list[Path]:
    """
    List valid DuckDB database files in `search_dir` (and optionally one subdirectory level deep).

    Excludes session store databases (`sessions.duckdb`, `t1d_sessions.duckdb`) and macOS
    AppleDouble resource fork files (`._*`).

    Args:
        search_dir: Directory to scan for `.duckdb` files.
        include_subdirs: Whether to also scan immediate child directories (`*/*.duckdb`).

    Returns:
        List[Path]: Discovered `.duckdb` file paths.

    """
    if not search_dir.exists():
        return []
    excluded_names = {"sessions.duckdb", "t1d_sessions.duckdb", "sessions.db"}
    patterns = (
        ("*.duckdb", "*.db", "*/*.duckdb", "*/*.db")
        if include_subdirs
        else ("*.duckdb", "*.db")
    )
    matches: list[Path] = []
    for pattern in patterns:
        matches.extend(
            p
            for p in search_dir.glob(pattern)
            if p.name not in excluded_names and not p.name.startswith("._")
        )
    return matches


def _resolve_bare_filename_db_path(raw_path: str) -> str:
    """
    Resolve a bare `.duckdb` filename against `T1D_DB_PATH` and `get_t1d_db_search_dirs()`.

    Args:
        raw_path: Bare database filename without directory separators.

    Returns:
        str: Matched existing path string, or `raw_path` if no match exists.

    """
    env_db_cand = os.environ.get("T1D_DB_PATH")
    if (
        env_db_cand
        and Path(env_db_cand).name == raw_path
        and Path(env_db_cand).exists()
    ):
        return env_db_cand
    for s_dir in get_t1d_db_search_dirs():
        for cand in (s_dir / raw_path, s_dir / "jaeb" / raw_path):
            if cand.exists():
                return str(cand)
    return raw_path


JAEB_SHORTHAND_IDENTIFIERS: tuple[str, ...] = (
    "jaeb",
    "jaeb.duckdb",
    "jaeb/t1d.duckdb",
    "/mnt/data/t1d",
    "/mnt/data/t1d/jaeb",
)


def normalize_db_shorthand_arg(db_path: str | None) -> str | None:
    """
    Normalize a shorthand database identifier or external directory root into a concrete `.duckdb` target.

    Args:
        db_path: Optional explicit database file or directory path.

    Returns:
        Optional[str]: Normalized path string, or `"t1d.duckdb"` for known JAEB shorthands.

    """
    if not db_path:
        return db_path
    cleaned = db_path.strip().rstrip("/\\")
    if cleaned.lower().replace("\\", "/") in JAEB_SHORTHAND_IDENTIFIERS:
        return "t1d.duckdb"
    cand = Path(cleaned)
    if "\0" not in cleaned and _safe_path_exists(cand) and cand.is_dir():
        for rel in (cand / "jaeb" / "t1d.duckdb", cand / "t1d.duckdb"):
            if _safe_path_exists(rel):
                return str(rel)
    return db_path


def resolve_default_db_path(db_path: str | None = None) -> str:
    """
    Resolve the target DuckDB database path from an explicit argument, `T1D_DB_PATH`, `T1D_DB_SEARCH_DIR`, or `data/t1d.duckdb`.

    Args:
        db_path: Optional explicit database file path.

    Returns:
        str: Resolved candidate path string prior to directory sandboxing.

    """
    env_db_fallback = os.environ.get("T1D_DB_PATH")
    if (
        env_db_fallback
        and env_db_fallback.startswith("/mnt/data")
        and not Path(env_db_fallback).exists()
    ):
        env_db_fallback = None
    normalized_arg = normalize_db_shorthand_arg(db_path)
    raw_path = normalized_arg or (
        env_db_fallback if env_db_fallback is not None else "t1d.duckdb"
    )
    if raw_path == "t1d.duckdb" and not Path(raw_path).exists():
        if env_db_fallback and Path(env_db_fallback).exists():
            raw_path = env_db_fallback
        elif not env_db_fallback:
            search_dirs = get_configured_db_search_dirs()
            for s_dir in search_dirs:
                candidate = s_dir / "t1d.duckdb"
                if candidate.exists():
                    return str(candidate)
            found = [
                f
                for s_dir in search_dirs
                for f in _iter_duckdb_files_in_dir(s_dir, include_subdirs=True)
            ]
            if found:
                return str(found[0])
            if Path("data/t1d.duckdb").exists():
                raw_path = "data/t1d.duckdb"
    elif (
        normalized_arg
        and "/" not in raw_path
        and "\\" not in raw_path
        and "\0" not in raw_path
        and not Path(raw_path).exists()
    ):
        return _resolve_bare_filename_db_path(raw_path)
    return raw_path


def _resolve_contained_path(
    raw_path: str,
    allowed_dirs: list[Path] | None = None,
) -> Path:
    """
    Validate that `raw_path` contains no null bytes and resolves inside an allowed root directory.

    Args:
        raw_path: Unvalidated path string.
        allowed_dirs: Optional list of allowed directory roots.

    Returns:
        Path: Resolved filesystem Path within an allowed root.

    Raises:
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
    db_path: str | None = None,
    allowed_dirs: list[Path] | None = None,
) -> str:
    """
    Validate and resolve a DuckDB database path against allowed directory containment and `.duckdb`/`.db` extensions.

    Args:
        db_path: The requested database file path, or None for default.
        allowed_dirs: Optional list of allowed directory roots.

    Returns:
        str: Validated string path to the database file.

    Raises:
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


def resolve_default_sessions_db_path(db_path: str | None = None) -> str:
    """
    Resolve the target persistent sessions DuckDB file path from an argument, env var, or `./data/sessions.duckdb`.

    Args:
        db_path: Optional explicit session database file path.

    Returns:
        str: Candidate session database file path prior to containment checks.

    Raises:
        InvalidDatabasePathError: If an explicit `db_path` string is blank/whitespace-only.

    """
    if db_path is not None:
        cleaned = db_path.strip()
        if not cleaned:
            raise InvalidDatabasePathError(params={"path": db_path})
        return cleaned
    env_db = os.environ.get("T1D_SESSIONS_DB") or os.environ.get("T1D_SESSIONS_DB_PATH")
    if env_db and env_db.strip():
        return env_db.strip()
    return str(Path("./data") / "sessions.duckdb")


def validate_sessions_db_path(
    db_path: str | None = None,
    allowed_dirs: list[Path] | None = None,
) -> str:
    """
    Validate and resolve a session DuckDB path against allowed roots, traversal rules, and `.duckdb`/`.db` extensions.

    Rejects null bytes, parent traversal (`..`) segments, symlinks escaping allowed roots,
    non-DuckDB extensions, and arbitrary paths outside configured workspace/session roots.

    Args:
        db_path: Optional requested session database file path, or None for default.
        allowed_dirs: Optional list of allowed directory roots.

    Returns:
        str: Canonical validated string path to the session database file.

    Raises:
        InvalidDatabasePathError: If the path contains `..` or null bytes, escapes allowed roots,
            or lacks a `.duckdb` / `.db` extension.

    """
    raw_path = resolve_default_sessions_db_path(db_path)
    normalized_slashes = raw_path.replace("\\", "/")
    if "\0" in raw_path or ".." in Path(normalized_slashes).parts:
        raise InvalidDatabasePathError(params={"path": str(raw_path)})
    resolved_path = _resolve_contained_path(raw_path, allowed_dirs=allowed_dirs)
    if resolved_path.suffix.lower() not in ALLOWED_DB_EXTENSIONS:
        raise InvalidDatabasePathError(params={"path": str(raw_path)})
    return str(resolved_path)


def validate_workspace_artifact_path(
    artifact_path: str,
    allowed_dirs: list[Path] | None = None,
) -> str:
    """
    Validate and resolve a training dataset or checkpoint directory path against allowed directory containment.

    Args:
        artifact_path: The requested dataset file or output directory path.
        allowed_dirs: Optional list of allowed directory roots.

    Returns:
        str: Validated string path to the workspace artifact.

    Raises:
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
    Discover and list available DuckDB database files in the workspace and external T1D search roots.

    Deduplicates candidate files strictly by canonical `Path.resolve()`.

    Returns:
        DatabaseListResponse: Found database items and current default path.

    """
    default_db = resolve_default_db_path()
    all_files: list[Path] = []
    default_resolved: str | None = None
    if default_db:
        p_def = Path(default_db)
        default_resolved = str(p_def.resolve())
        if p_def.exists():
            all_files.append(p_def)

    for local_dir in (Path(), Path("./data")):
        all_files.extend(_iter_duckdb_files_in_dir(local_dir, include_subdirs=False))
    for remote_cand in REMOTE_DISCOVERY_DB_CANDIDATES:
        if _safe_path_exists(remote_cand):
            all_files.append(remote_cand)
    for ext_dir in get_t1d_db_search_dirs():
        all_files.extend(_iter_duckdb_files_in_dir(ext_dir, include_subdirs=True))

    cwd_resolved = Path.cwd().resolve()
    found_dbs: list[DatabaseItem] = []
    seen_paths: set[str] = set()
    for p in all_files:
        resolved_path = p.resolve()
        resolved = str(resolved_path)
        p_size = p.stat().st_size if p.exists() else 0
        if resolved not in seen_paths:
            seen_paths.add(resolved)
            is_in_cwd = resolved_path.is_relative_to(cwd_resolved)
            name_collision = any(existing.name == p.name for existing in found_dbs)
            item_name = (
                str(p)
                if (name_collision or (not is_in_cwd and resolved != default_resolved))
                else p.name
            )
            found_dbs.append(
                DatabaseItem(
                    name=item_name,
                    path=str(p),
                    size_bytes=p_size,
                )
            )

    return DatabaseListResponse(databases=found_dbs, current_db=default_db)
