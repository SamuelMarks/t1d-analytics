"""Safe archive extraction, ZIP/TAR path-traversal defense, and decompression quota utilities."""

import logging
import os
import tarfile
import zipfile
from pathlib import Path
from typing import Callable, List, Optional

from t1d_analytics.i18n import get_translator

logger = logging.getLogger(__name__)

DEFAULT_MAX_DECOMPRESSED_BYTES = 200 * 1024 * 1024 * 1024  # 200 GiB quota
MAX_MEMBER_COMPRESSION_RATIO = 1000  # Zip-bomb ratio threshold for >100 MiB members
MIN_RATIO_CHECK_BYTES = 100 * 1024 * 1024  # 100 MiB threshold for ratio check

_TABULAR_AND_ARCHIVE_SUFFIXES = (
    ".csv",
    ".txt",
    ".parquet",
    ".csv.gz",
    ".sas7bdat",
    ".xpt",
    ".sav",
    ".zip",
    ".tar.gz",
    ".tgz",
    ".tar.bz2",
    ".tar",
)

_ARCHIVE_GLOB_PATTERNS = (
    "*.zip",
    "*.ZIP",
    "*.tar.gz",
    "*.TAR.GZ",
    "*.tgz",
    "*.TGZ",
    "*.tar.bz2",
    "*.TAR.BZ2",
    "*.tar",
    "*.TAR",
)


def resolve_effective_decompressed_quota(
    max_decompressed_bytes: Optional[int] = None,
) -> int:
    """
    Resolve the effective maximum decompressed byte quota from explicit arguments or environment variables.

    Supports `T1D_MAX_DECOMPRESSED_BYTES` and `T1D_MAX_DECOMPRESSED_GB` when `max_decompressed_bytes`
    is None or equals `DEFAULT_MAX_DECOMPRESSED_BYTES`. A value `<= 0` indicates an unlimited quota.

    Args:
    ----
        max_decompressed_bytes: Explicit maximum decompressed bytes (or None to use default/env).

    Returns:
    -------
        int: Effective quota in bytes (`<= 0` means unlimited).

    """
    if (
        max_decompressed_bytes is not None
        and max_decompressed_bytes != DEFAULT_MAX_DECOMPRESSED_BYTES
    ):
        return max_decompressed_bytes

    env_bytes = os.environ.get("T1D_MAX_DECOMPRESSED_BYTES", "").strip()
    if env_bytes:
        try:
            return int(env_bytes)
        except ValueError:
            pass

    env_gb = os.environ.get("T1D_MAX_DECOMPRESSED_GB", "").strip()
    if env_gb:
        try:
            gb_val = float(env_gb)
            return int(gb_val * 1024 * 1024 * 1024) if gb_val > 0 else 0
        except ValueError:
            pass

    return (
        DEFAULT_MAX_DECOMPRESSED_BYTES
        if max_decompressed_bytes is None
        else max_decompressed_bytes
    )


def _should_skip_archive_member_name(
    member_name: str,
    tabular_only: bool = False,
) -> bool:
    """
    Determine whether an archive member should be skipped as OS metadata or non-tabular documentation.

    Args:
    ----
        member_name: Internal relative path of the archive entry.
        tabular_only: If True, skip entries that are neither tabular data files nor nested archives.

    Returns:
    -------
        bool: True if the member should be skipped without warning.

    """
    bs = chr(92)
    norm_parts = Path(member_name.replace(bs, "/")).parts
    if not norm_parts:
        return True
    for part in norm_parts:
        if part == "__MACOSX" or part.startswith("._"):
            return True
    leaf_lower = norm_parts[-1].lower()
    if leaf_lower in ("thumbs.db", ".ds_store"):
        return True
    if tabular_only:
        return not leaf_lower.endswith(_TABULAR_AND_ARCHIVE_SUFFIXES)
    return False


def _is_safe_member_target(target_path: Path, resolved_extract_dir: Path) -> bool:
    """
    Verify that `target_path` resolves inside `resolved_extract_dir` (ZIP/TAR slip prevention).

    Args:
    ----
        target_path: Resolved destination path for an archive member.
        resolved_extract_dir: Resolved root extraction directory.

    Returns:
    -------
        bool: True if `target_path` is equal to or inside `resolved_extract_dir`.

    """
    return target_path == resolved_extract_dir or target_path.is_relative_to(
        resolved_extract_dir
    )


def _is_already_extracted(
    target_path: Path,
    expected_size: int,
    overwrite: bool,
) -> bool:
    """
    Check if `target_path` already exists on disk with `expected_size` when `overwrite` is False.

    Args:
    ----
        target_path: Destination file path.
        expected_size: Expected uncompressed byte size.
        overwrite: Whether existing files should be overwritten.

    Returns:
    -------
        bool: True if extraction can be skipped as idempotent.

    """
    return (
        not overwrite
        and target_path.exists()
        and target_path.is_file()
        and target_path.stat().st_size == expected_size
    )


def _extract_tar_archive(
    path: Path,
    extract_dir: Path,
    resolved_extract_dir: Path,
    total_extracted_bytes: int,
    max_decompressed_bytes: int,
    report_fn: Callable[[str], None],
    overwrite: bool = False,
    tabular_only: bool = False,
) -> int:
    """
    Safely extract a tar archive into `extract_dir` with symlink, path-traversal, idempotency, and size-quota checks.

    Args:
    ----
        path: Source tar archive path.
        extract_dir: Target extraction directory.
        resolved_extract_dir: Resolved target extraction directory.
        total_extracted_bytes: Running cumulative extracted byte count.
        max_decompressed_bytes: Maximum cumulative decompressed size quota in bytes (`<= 0` for unlimited).
        report_fn: Progress/warning reporter callback.
        overwrite: Whether to overwrite already-extracted member files that match expected size.
        tabular_only: Whether to skip non-tabular documentation files.

    Returns:
    -------
        int: Updated cumulative extracted byte count.

    """
    _ = get_translator()
    announced = False
    try:
        with tarfile.open(path, "r:*") as tar_ref:
            for tar_member in tar_ref.getmembers():
                if tar_member.issym() or tar_member.islnk():
                    warn_sym = _(
                        "Skipping unsafe entry {} in {} (symlink or hardlink detected).",
                        tar_member.name,
                        path.name,
                    )
                    logger.debug(warn_sym)
                    report_fn(warn_sym)
                    continue
                tar_target = (resolved_extract_dir / tar_member.name).resolve()
                if not _is_safe_member_target(tar_target, resolved_extract_dir):
                    warn_trav = _(
                        "Skipping unsafe entry {} in {} (path traversal detected).",
                        tar_member.name,
                        path.name,
                    )
                    logger.debug(warn_trav)
                    report_fn(warn_trav)
                    continue
                if tar_member.isdir() or _should_skip_archive_member_name(
                    tar_member.name, tabular_only=tabular_only
                ):
                    continue
                expected_size = max(0, tar_member.size)
                if _is_already_extracted(tar_target, expected_size, overwrite):
                    continue
                if (
                    max_decompressed_bytes > 0
                    and total_extracted_bytes + expected_size > max_decompressed_bytes
                ):
                    warn_quota = _(
                        "Skipping entry {} in {} (decompressed size quota exceeded).",
                        tar_member.name,
                        path.name,
                    )
                    logger.debug(warn_quota)
                    report_fn(warn_quota)
                    continue
                if not announced:
                    ext_msg = _("Extracting {}...", path.name)
                    logger.info(ext_msg)
                    report_fn(ext_msg)
                    extract_dir.mkdir(parents=True, exist_ok=True)
                    announced = True
                total_extracted_bytes += expected_size
                tar_ref.extract(tar_member, extract_dir)
    except (tarfile.TarError, EOFError, OSError):
        err_tar = _("Failed to extract {} (Bad Archive File).", path.name)
        logger.debug(err_tar)
        report_fn(err_tar)
    return total_extracted_bytes


def _extract_zip_archive(
    path: Path,
    extract_dir: Path,
    resolved_extract_dir: Path,
    total_extracted_bytes: int,
    max_decompressed_bytes: int,
    report_fn: Callable[[str], None],
    overwrite: bool = False,
    tabular_only: bool = False,
    min_ratio_check_bytes: int = MIN_RATIO_CHECK_BYTES,
    max_member_compression_ratio: int = MAX_MEMBER_COMPRESSION_RATIO,
) -> int:
    """
    Safely extract a zip archive into `extract_dir` with symlink, path-traversal, idempotency, and size-quota checks.

    Args:
    ----
        path: Source zip archive path.
        extract_dir: Target extraction directory.
        resolved_extract_dir: Resolved target extraction directory.
        total_extracted_bytes: Running cumulative extracted byte count.
        max_decompressed_bytes: Maximum cumulative decompressed size quota in bytes (`<= 0` for unlimited).
        report_fn: Progress/warning reporter callback.
        overwrite: Whether to overwrite already-extracted member files that match expected size.
        tabular_only: Whether to skip non-tabular documentation files.
        min_ratio_check_bytes: Minimum uncompressed member byte size before applying ratio checks.
        max_member_compression_ratio: Maximum allowed uncompressed/compressed size ratio.

    Returns:
    -------
        int: Updated cumulative extracted byte count.

    """
    _ = get_translator()
    announced = False
    try:
        with zipfile.ZipFile(path, "r") as zip_ref:
            for zip_info in zip_ref.infolist():
                if (zip_info.external_attr >> 16) & 0o170000 == 0o120000:
                    warn_zsym = _(
                        "Skipping unsafe entry {} in {} (symlink detected).",
                        zip_info.filename,
                        path.name,
                    )
                    logger.debug(warn_zsym)
                    report_fn(warn_zsym)
                    continue
                zip_target = (resolved_extract_dir / zip_info.filename).resolve()
                if not _is_safe_member_target(zip_target, resolved_extract_dir):
                    warn_ztrav = _(
                        "Skipping unsafe entry {} in {} (path traversal detected).",
                        zip_info.filename,
                        path.name,
                    )
                    logger.debug(warn_ztrav)
                    report_fn(warn_ztrav)
                    continue
                if zip_info.is_dir() or _should_skip_archive_member_name(
                    zip_info.filename, tabular_only=tabular_only
                ):
                    continue
                expected_size = max(0, zip_info.file_size)
                compressed_size = max(1, zip_info.compress_size)
                if (
                    expected_size >= min_ratio_check_bytes
                    and (expected_size / compressed_size) > max_member_compression_ratio
                ):
                    warn_bomb = _(
                        "Skipping unsafe entry {} in {} (suspicious compression ratio).",
                        zip_info.filename,
                        path.name,
                    )
                    logger.debug(warn_bomb)
                    report_fn(warn_bomb)
                    continue
                if _is_already_extracted(zip_target, expected_size, overwrite):
                    continue
                if (
                    max_decompressed_bytes > 0
                    and total_extracted_bytes + expected_size > max_decompressed_bytes
                ):
                    warn_zquota = _(
                        "Skipping entry {} in {} (decompressed size quota exceeded).",
                        zip_info.filename,
                        path.name,
                    )
                    logger.debug(warn_zquota)
                    report_fn(warn_zquota)
                    continue
                if not announced:
                    ext_msg = _("Extracting {}...", path.name)
                    logger.info(ext_msg)
                    report_fn(ext_msg)
                    extract_dir.mkdir(parents=True, exist_ok=True)
                    announced = True
                total_extracted_bytes += expected_size
                zip_ref.extract(zip_info, extract_dir)
    except zipfile.BadZipFile:
        err_zip = _("Failed to extract {} (Bad Zip File).", path.name)
        logger.debug(err_zip)
        report_fn(err_zip)
    return total_extracted_bytes


_extract_single_zip = _extract_zip_archive


def _resolve_archive_extract_dir(path: Path) -> Path:
    """
    Determine the target extraction directory for an archive path.

    Args:
    ----
        path: Archive file path (`.zip`, `.tar.gz`, `.tgz`, `.tar.bz2`, `.tar`).

    Returns:
    -------
        Path: Target directory path with the archive suffix stripped.

    """
    p_name = path.name.lower()
    if p_name.endswith((".tar.gz", ".tar.bz2")):
        return path.parent / path.name[:-7]
    if p_name.endswith(".tgz"):
        return path.parent / path.name[:-4]
    return path.with_suffix("")


def _discover_unprocessed_archives(
    data_dir: Path,
    processed_archives: set[Path],
) -> List[Path]:
    """
    Discover archive files in `data_dir` that have not yet been processed.

    Args:
    ----
        data_dir: Root directory to scan.
        processed_archives: Set of resolved archive paths already processed.

    Returns:
    -------
        List[Path]: Ordered list of unprocessed archive paths.

    """
    candidate_archives: List[Path] = []
    for pat in _ARCHIVE_GLOB_PATTERNS:
        candidate_archives.extend(data_dir.rglob(pat))
    deduped = [
        p for p in dict.fromkeys(candidate_archives) if not p.name.startswith("._")
    ]
    return [p for p in deduped if p.resolve() not in processed_archives]


def _extract_single_archive_file(
    path: Path,
    total_extracted_bytes: int,
    effective_quota: int,
    report_fn: Callable[[str], None],
    overwrite: bool,
    tabular_only: bool,
    min_ratio_check_bytes: int = MIN_RATIO_CHECK_BYTES,
    max_member_compression_ratio: int = MAX_MEMBER_COMPRESSION_RATIO,
) -> int:
    """
    Dispatch a single archive file (`.tar*` or `.zip`) to the appropriate extractor.

    Args:
    ----
        path: Archive file path.
        total_extracted_bytes: Running cumulative extracted byte count.
        effective_quota: Maximum decompressed byte quota (`<= 0` for unlimited).
        report_fn: Progress reporter callback.
        overwrite: Whether to overwrite existing files.
        tabular_only: Whether to only extract tabular files and nested archives.
        min_ratio_check_bytes: Minimum uncompressed member byte size before applying ratio checks.
        max_member_compression_ratio: Maximum allowed uncompressed/compressed size ratio.

    Returns:
    -------
        int: Updated cumulative extracted byte count.

    """
    extract_dir = _resolve_archive_extract_dir(path)
    resolved_extract_dir = extract_dir.resolve()
    if path.name.lower().endswith((".tar.gz", ".tgz", ".tar.bz2", ".tar")):
        return _extract_tar_archive(
            path,
            extract_dir,
            resolved_extract_dir,
            total_extracted_bytes,
            effective_quota,
            report_fn,
            overwrite=overwrite,
            tabular_only=tabular_only,
        )
    return _extract_zip_archive(
        path,
        extract_dir,
        resolved_extract_dir,
        total_extracted_bytes,
        effective_quota,
        report_fn,
        overwrite=overwrite,
        tabular_only=tabular_only,
        min_ratio_check_bytes=min_ratio_check_bytes,
        max_member_compression_ratio=max_member_compression_ratio,
    )


def extract_zips(
    data_dir_str: str,
    max_depth: int = 3,
    report_fn: Callable[[str], None] = print,
    max_decompressed_bytes: int = DEFAULT_MAX_DECOMPRESSED_BYTES,
    overwrite: bool = False,
    tabular_only: bool = False,
    min_ratio_check_bytes: Optional[int] = None,
    max_member_compression_ratio: Optional[int] = None,
) -> None:
    """
    Extract zip and tar archives found in the data directory safely without path traversal or zip bombs.

    Recursively decompresses nested archives up to max_depth with per-entry size-verified idempotency.

    Args:
    ----
        data_dir_str: Directory path containing archive files to extract.
        max_depth: Maximum recursion depth for decompressing nested archives.
        report_fn: Progress reporter callback (defaults to `print`).
        max_decompressed_bytes: Maximum cumulative decompressed size in bytes allowed across archives (`<= 0` for unlimited).
        overwrite: Whether to force re-extraction of existing member files even if sizes match.
        tabular_only: Whether to only extract tabular data files and nested archives (skipping PDFs/docs).
        min_ratio_check_bytes: Optional minimum byte threshold for zip-bomb ratio check.
        max_member_compression_ratio: Optional maximum compression ratio threshold for zip-bomb check.

    """
    _ = get_translator()
    data_dir = Path(data_dir_str)
    if not data_dir.exists():
        msg = _("Data directory {} does not exist.", data_dir)
        logger.debug(msg)
        report_fn(msg)
        return

    effective_quota = resolve_effective_decompressed_quota(max_decompressed_bytes)
    effective_min_ratio_bytes = (
        min_ratio_check_bytes
        if min_ratio_check_bytes is not None
        else MIN_RATIO_CHECK_BYTES
    )
    effective_max_ratio = (
        max_member_compression_ratio
        if max_member_compression_ratio is not None
        else MAX_MEMBER_COMPRESSION_RATIO
    )
    check_msg = _("Checking for zip files to extract...")
    logger.info(check_msg)
    report_fn(check_msg)

    processed_archives: set[Path] = set()
    found_any = False
    total_extracted_bytes = 0

    for _depth in range(max_depth):
        unprocessed = _discover_unprocessed_archives(data_dir, processed_archives)
        if not unprocessed:
            break
        for path in unprocessed:
            found_any = True
            processed_archives.add(path.resolve())
            total_extracted_bytes = _extract_single_archive_file(
                path=path,
                total_extracted_bytes=total_extracted_bytes,
                effective_quota=effective_quota,
                report_fn=report_fn,
                overwrite=overwrite,
                tabular_only=tabular_only,
                min_ratio_check_bytes=effective_min_ratio_bytes,
                max_member_compression_ratio=effective_max_ratio,
            )

    done_msg = (
        _("Extraction complete.") if found_any else _("No zip files found to extract.")
    )
    logger.info(done_msg)
    report_fn(done_msg)


__all__ = [
    "DEFAULT_MAX_DECOMPRESSED_BYTES",
    "MAX_MEMBER_COMPRESSION_RATIO",
    "MIN_RATIO_CHECK_BYTES",
    "_ARCHIVE_GLOB_PATTERNS",
    "_TABULAR_AND_ARCHIVE_SUFFIXES",
    "_discover_unprocessed_archives",
    "_extract_single_archive_file",
    "_extract_single_zip",
    "_extract_tar_archive",
    "_extract_zip_archive",
    "_is_already_extracted",
    "_is_safe_member_target",
    "_resolve_archive_extract_dir",
    "_should_skip_archive_member_name",
    "extract_zips",
    "resolve_effective_decompressed_quota",
]
