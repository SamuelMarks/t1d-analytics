"""Archive extraction, clinical dataset parsing, and DuckDB ingestion pipeline."""

import csv
import hashlib
import json
import logging
import os
import re
import tarfile
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple

import duckdb
from typing_extensions import TypedDict

if TYPE_CHECKING:
    import pandas as pd

from t1d_analytics.errors import DatabaseError
from t1d_analytics.i18n import get_translator
from t1d_analytics.sql_utils import (
    escape_sql_identifier,
    escape_sql_literal,
    global_read_only_duckdb_pool,
)

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


class IngestionManifestRecord(TypedDict):
    """Audit record for an ingested clinical dataset file."""

    source_file: str
    table_name: str
    file_hash: str
    row_count: int
    loaded_at: str
    file_size: int


def compute_file_hash(path: Path) -> str:
    """
    Compute SHA-256 hash of a file.

    Args:
    ----
        path: Path to the file.

    Returns:
    -------
        str: Hexadecimal SHA-256 digest of the file contents.

    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


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
    norm_parts = Path(member_name.replace("\\", "/")).parts
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
                is_safe_tar = (
                    tar_target == resolved_extract_dir
                    or tar_target.is_relative_to(resolved_extract_dir)
                )
                if not is_safe_tar:
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
                if (
                    not overwrite
                    and tar_target.exists()
                    and tar_target.is_file()
                    and tar_target.stat().st_size == expected_size
                ):
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
                is_safe_zip = (
                    zip_target == resolved_extract_dir
                    or zip_target.is_relative_to(resolved_extract_dir)
                )
                if not is_safe_zip:
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
                    expected_size >= MIN_RATIO_CHECK_BYTES
                    and (expected_size / compressed_size) > MAX_MEMBER_COMPRESSION_RATIO
                ):
                    warn_bomb = _(
                        "Skipping unsafe entry {} in {} (suspicious compression ratio).",
                        zip_info.filename,
                        path.name,
                    )
                    logger.debug(warn_bomb)
                    report_fn(warn_bomb)
                    continue
                if (
                    not overwrite
                    and zip_target.exists()
                    and zip_target.is_file()
                    and zip_target.stat().st_size == expected_size
                ):
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


def extract_zips(
    data_dir_str: str,
    max_depth: int = 3,
    report_fn: Callable[[str], None] = print,
    max_decompressed_bytes: int = DEFAULT_MAX_DECOMPRESSED_BYTES,
    overwrite: bool = False,
    tabular_only: bool = False,
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

    """
    _ = get_translator()
    data_dir = Path(data_dir_str)
    if not data_dir.exists():
        msg = _("Data directory {} does not exist.", data_dir)
        logger.debug(msg)
        report_fn(msg)
        return

    effective_quota = resolve_effective_decompressed_quota(max_decompressed_bytes)

    check_msg = _("Checking for zip files to extract...")
    logger.info(check_msg)
    report_fn(check_msg)
    processed_archives: set[Path] = set()
    found_any = False
    total_extracted_bytes = 0
    archive_patterns = (
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

    for _depth in range(max_depth):
        candidate_archives: list[Path] = []
        for pat in archive_patterns:
            candidate_archives.extend(data_dir.rglob(pat))
        candidate_archives = [
            p for p in dict.fromkeys(candidate_archives) if not p.name.startswith("._")
        ]

        unprocessed = [
            p for p in candidate_archives if p.resolve() not in processed_archives
        ]
        if not unprocessed:
            break

        for path in unprocessed:
            found_any = True
            processed_archives.add(path.resolve())

            p_name = path.name.lower()
            if p_name.endswith((".tar.gz", ".tar.bz2")):
                extract_dir = path.parent / path.name[:-7]
            elif p_name.endswith(".tgz"):
                extract_dir = path.parent / path.name[:-4]
            else:
                extract_dir = path.with_suffix("")

            resolved_extract_dir = extract_dir.resolve()

            if p_name.endswith((".tar.gz", ".tgz", ".tar.bz2", ".tar")):
                total_extracted_bytes = _extract_tar_archive(
                    path,
                    extract_dir,
                    resolved_extract_dir,
                    total_extracted_bytes,
                    effective_quota,
                    report_fn,
                    overwrite=overwrite,
                    tabular_only=tabular_only,
                )
            else:
                total_extracted_bytes = _extract_zip_archive(
                    path,
                    extract_dir,
                    resolved_extract_dir,
                    total_extracted_bytes,
                    effective_quota,
                    report_fn,
                    overwrite=overwrite,
                    tabular_only=tabular_only,
                )

    if not found_any:
        no_zips_msg = _("No zip files found to extract.")
        logger.info(no_zips_msg)
        report_fn(no_zips_msg)
    else:
        done_msg = _("Extraction complete.")
        logger.info(done_msg)
        report_fn(done_msg)


def _read_clinical_dataframe(data_file: Path) -> "pd.DataFrame":
    """
    Read SAS (.sas7bdat, .xpt) or SPSS (.sav) data file into a pandas DataFrame.

    Translates SAS missing values and dates to standard nullable types.

    Args:
    ----
        data_file: Path to the clinical data file.

    Returns:
    -------
        pd.DataFrame: Parsed DataFrame with converted column data.

    """
    import pandas as pd

    suffix = data_file.suffix.lower()
    df: Optional[pd.DataFrame] = None
    if suffix in (".sas7bdat", ".xpt"):
        try:
            import pyreadstat

            if suffix == ".sas7bdat":
                df, _ = pyreadstat.read_sas7bdat(str(data_file))
            else:
                df, _ = pyreadstat.read_xport(str(data_file))
        except (ImportError, Exception):
            fmt = "sas7bdat" if suffix == ".sas7bdat" else "xport"
            df = pd.read_sas(str(data_file), format=fmt)
    elif suffix == ".sav":
        try:
            import pyreadstat

            df, _ = pyreadstat.read_sav(str(data_file))
        except (ImportError, Exception):
            df = pd.read_spss(str(data_file))
    else:
        raise DatabaseError(f"Unsupported clinical file format: {suffix}")

    if df is None:
        raise DatabaseError(f"Could not read clinical file {data_file}")

    # Ensure column names are standard strings
    df.columns = [
        col.decode("utf-8", errors="ignore") if isinstance(col, bytes) else str(col)
        for col in df.columns
    ]
    # Decode byte string elements if any
    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col].dtype):
            df[col] = df[col].apply(
                lambda x: (
                    x.decode("utf-8", errors="ignore") if isinstance(x, bytes) else x
                )
            )
    return df


def get_ingestion_manifest(db_path: str) -> list[IngestionManifestRecord]:
    """
    Retrieve the ingestion manifest audit records from DuckDB.

    Args:
    ----
        db_path: Path to the DuckDB database file.

    Returns:
    -------
        list[IngestionManifestRecord]: List of manifest records containing source_file,
            table_name, file_hash, row_count, loaded_at, and file_size.

    """
    if not Path(db_path).exists():
        return []

    conn = duckdb.connect(db_path, read_only=True)
    has_manifest = conn.execute(
        "SELECT 1 FROM information_schema.tables WHERE table_name = '_t1d_ingestion_manifest'"
    ).fetchone()
    if not has_manifest:
        conn.close()
        return []

    rows = conn.execute(
        "SELECT source_file, table_name, file_hash, row_count, loaded_at, file_size FROM _t1d_ingestion_manifest ORDER BY loaded_at"
    ).fetchall()
    conn.close()

    return [
        {
            "source_file": str(r[0]),
            "table_name": str(r[1]),
            "file_hash": str(r[2]),
            "row_count": int(r[3]),
            "loaded_at": str(r[4]),
            "file_size": int(r[5]),
        }
        for r in rows
    ]


_NUMERIC_CLINICAL_CONCEPTS = {
    "age",
    "ageasofranddt",
    "agebaseline",
    "hba1c",
    "a1c",
    "a1cval",
    "c_peptide",
    "glucose",
    "glucose_value",
    "sensorglucose",
    "historicglucose",
    "meancgm",
    "meanglucose",
    "mean_glucose",
    "tir",
    "tbr",
    "tar",
    "bmi",
    "bmibaseline",
    "bodymassindex",
    "bmi_kg_m2",
    "t1d_duration_years",
    "duration",
    "diagduration",
    "yrsdiag",
    "t1dduration",
    "pt_duration",
    "total_daily_insulin",
    "tdd",
    "totaldailydose",
    "unitsperkg",
    "basalbolustotal",
    "cgm_active_wear_pct",
    "pctwear",
    "activewear",
    "sensorwearpct",
    "validcgmdays",
    "weight_kg",
    "height_cm",
    "insulin_total_daily",
    "insulin_total_daily_units",
    "insulin_basal",
    "insulin_bolus",
    "numsevhypo",
    "pt_numsevhypo",
    "numhospdka",
    "pt_numhospdka",
}
_CLINICAL_NUMERIC_COLUMNS = _NUMERIC_CLINICAL_CONCEPTS


def coerce_known_clinical_numeric_columns(
    conn: duckdb.DuckDBPyConnection,
    table_name: str,
) -> None:
    """
    Alter known clinical numeric columns stored as `VARCHAR` in an ingested DuckDB table to `DOUBLE`.

    Args:
    ----
        conn: Active writable DuckDB connection.
        table_name: Target DuckDB table identifier.

    """
    safe_tbl = escape_sql_identifier(table_name)
    cols = conn.execute(f'DESCRIBE "{safe_tbl}"').fetchall()
    for col_row in cols:
        col_name = str(col_row[0])
        col_type = str(col_row[1]).upper()
        if col_name.lower() in _NUMERIC_CLINICAL_CONCEPTS and "CHAR" in col_type:
            safe_col = escape_sql_identifier(col_name)
            conn.execute(
                f'ALTER TABLE "{safe_tbl}" ALTER COLUMN "{safe_col}" '
                f"TYPE DOUBLE USING TRY_CAST(NULLIF(TRIM(\"{safe_col}\"), '') AS DOUBLE)"
            )


def get_supported_tabular_globs(
    include_sas: bool = False,
    include_spss: bool = False,
) -> List[str]:
    """
    Return the ordered list of supported tabular file glob patterns for dataset discovery.

    Args:
    ----
        include_sas: Whether to include SAS dataset patterns (`*.sas7bdat`, `*.xpt`).
        include_spss: Whether to include SPSS dataset patterns (`*.sav`).

    Returns:
    -------
        List[str]: Glob pattern strings suitable for `Path.rglob`.

    """
    base_exts = ["csv", "txt", "parquet", "csv.gz"]
    if include_sas:
        base_exts.extend(["sas7bdat", "xpt"])
    if include_spss:
        base_exts.append("sav")

    patterns: List[str] = []
    for ext in base_exts:
        patterns.append(f"*.{ext}")
        patterns.append(f"*.{ext.upper()}")
    return patterns


def get_clinical_variable_mappings() -> Dict[str, List[str]]:
    """
    Load the clinical variable ontology mapping standard concepts to trial-specific column names.

    Returns
    -------
        Dict[str, List[str]]: Mapping from standard concept names to list of original column names.

    """
    matches_file = Path(__file__).parent / "likely_matches.json"
    if matches_file.exists():
        try:
            with open(matches_file, "r") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return {
                        str(k): [str(x) for x in v]
                        for k, v in data.items()
                        if isinstance(v, list)
                    }
        except Exception:
            pass
    return {}


def _ensure_manifest_table(conn: duckdb.DuckDBPyConnection) -> None:
    """
    Ensure the `_t1d_ingestion_manifest` tracking table exists in DuckDB.

    Args:
    ----
        conn: Active read-write DuckDB connection.

    """
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS _t1d_ingestion_manifest (
            source_file VARCHAR,
            table_name VARCHAR,
            file_hash VARCHAR,
            row_count BIGINT,
            loaded_at TIMESTAMP,
            file_size BIGINT
        )
        """
    )


def _detect_csv_encoding_and_delimiter(csv_path: Path) -> Tuple[str, str, str]:
    """
    Detect encoding and column delimiter for a tabular CSV or TXT file.

    Args:
    ----
        csv_path: Path to the CSV or TXT file on disk.

    Returns:
    -------
        Tuple[str, str, str]: `(py_encoding, duckdb_encoding, duckdb_sep)`.

    """
    with open(csv_path, "rb") as f_rb:
        raw_bytes = f_rb.read(4)
    if raw_bytes[:2] in (b"\xff\xfe", b"\xfe\xff"):
        py_encoding = "utf-16"
        duckdb_encoding = "utf-16"
    elif raw_bytes[:3] == b"\xef\xbb\xbf":
        py_encoding = "utf-8-sig"
        duckdb_encoding = "utf-8"
    else:
        try:
            with open(csv_path, "r", encoding="utf-8") as f_chk:
                f_chk.read(8192)
            py_encoding = "utf-8"
            duckdb_encoding = "utf-8"
        except UnicodeDecodeError:
            py_encoding = "latin1"
            duckdb_encoding = "latin-1"

    sep = ","
    tab_char = chr(9)
    try:
        with open(csv_path, "r", encoding=py_encoding, errors="ignore") as f:
            sample = "".join([f.readline() for _ in range(5)])
        if sample.strip():
            dialect = csv.Sniffer().sniff(sample, delimiters=f",{tab_char}|;")
            sep = dialect.delimiter
    except Exception:
        with open(csv_path, "r", encoding=py_encoding, errors="ignore") as f:
            first_line = f.readline()
        if "|" in first_line:
            sep = "|"
        elif tab_char in first_line:
            sep = tab_char
        elif ";" in first_line:
            sep = ";"

    duckdb_sep = chr(92) + "t" if sep == tab_char else sep
    return py_encoding, duckdb_encoding, duckdb_sep


def _load_csv_or_parquet(
    conn: duckdb.DuckDBPyConnection,
    data_file: Path,
    table_name: str,
    report_fn: Callable[[str], None],
) -> None:
    """
    Create or replace `temp_view` in DuckDB for a Parquet, CSV.GZ, or CSV/TXT dataset file.

    Args:
    ----
        conn: Active DuckDB connection.
        data_file: Path to the tabular dataset file.
        table_name: Target DuckDB table name for progress logging.
        report_fn: Progress reporter callback.

    """
    _ = get_translator()
    escaped_path = escape_sql_literal(str(data_file))
    if data_file.suffix.lower() == ".parquet":
        load_pq_msg = _(
            "Loading {} (parquet) into table {}...",
            data_file.name,
            table_name,
        )
        logger.info(load_pq_msg)
        report_fn(load_pq_msg)
        conn.execute(
            f"CREATE OR REPLACE TEMP VIEW temp_view AS SELECT * FROM read_parquet('{escaped_path}')"
        )
    elif data_file.name.endswith(".csv.gz"):
        load_gz_msg = _(
            "Loading {} (csv.gz) into table {}...",
            data_file.name,
            table_name,
        )
        logger.info(load_gz_msg)
        report_fn(load_gz_msg)
        conn.execute(
            f"CREATE OR REPLACE TEMP VIEW temp_view AS SELECT * FROM read_csv('{escaped_path}', auto_detect=true, sample_size=-1, nullstr=['', 'NA', 'N/A', '.', 'NULL', 'null', 'None', 'Unknown', 'Missing', '-'])"
        )
    else:
        _py_enc, duckdb_encoding, duckdb_sep = _detect_csv_encoding_and_delimiter(
            data_file
        )
        load_csv_msg = _(
            "Loading {} (encoding={}, sep='{}') into table {}...",
            data_file.name,
            duckdb_encoding,
            duckdb_sep,
            table_name,
        )
        logger.info(load_csv_msg)
        report_fn(load_csv_msg)
        escaped_sep = escape_sql_literal(duckdb_sep)
        escaped_enc = escape_sql_literal(duckdb_encoding)
        conn.execute(
            f"CREATE OR REPLACE TEMP VIEW temp_view AS SELECT * FROM read_csv('{escaped_path}', auto_detect=true, sample_size=-1, nullstr=['', 'NA', 'N/A', '.', 'NULL', 'null', 'None', 'Unknown', 'Missing', '-'], sep='{escaped_sep}', encoding='{escaped_enc}')"
        )


def _resolve_target_table_stem(
    data_file: Path,
    data_dir: Path,
    prefix_subdirs: bool,
    stem_counts: dict[str, int],
) -> str:
    """
    Derive a sanitized SQL table name stem for `data_file`.

    Args:
    ----
        data_file: Source tabular file path.
        data_dir: Root ingestion directory path.
        prefix_subdirs: Whether to prefix table names with relative subdirectory names.
        stem_counts: Frequency map of normalized file stems.

    Returns:
    -------
        str: Valid SQL table identifier starting with an alphabetic character.

    """
    s_name = (
        Path(data_file.stem).stem
        if data_file.name.endswith(".csv.gz")
        else data_file.stem
    )
    raw_stem = re.sub(r"[^a-zA-Z0-9_]", "_", s_name).lower().strip("_") or "dataset"
    safe_stem = raw_stem
    if prefix_subdirs or stem_counts.get(raw_stem, 0) > 1:
        rel_parent = data_file.parent.relative_to(data_dir)
        if rel_parent != Path():
            prefix = re.sub(r"[^a-zA-Z0-9_]", "_", str(rel_parent)).strip("_").lower()
            safe_stem = f"{prefix}_{raw_stem}" if prefix else raw_stem

    table_name = safe_stem.strip("_") or "dataset"
    return table_name if table_name[0].isalpha() else f"t_{table_name}"


def _build_column_select_expressions(
    cols: List[Tuple[str, ...]],
    reverse_map: Dict[str, str],
) -> List[str]:
    """
    Build standardized DuckDB `SELECT` column expressions for `temp_view`.

    Args:
    ----
        cols: Output rows from `DESCRIBE temp_view`.
        reverse_map: Lowercase original column name to standardized clinical column name mapping.

    Returns:
    -------
        List[str]: SQL column projection expressions.

    """
    select_exprs: List[str] = []
    seen_std_names: set[str] = set()
    for c in cols:
        orig_col = str(c[0])
        col_type = str(c[1]).upper() if len(c) > 1 else ""
        std_name = reverse_map.get(orig_col.lower(), orig_col)

        if std_name.lower() in seen_std_names:
            std_name = orig_col

        counter = 1
        base_std_name = std_name
        while std_name.lower() in seen_std_names:
            counter += 1
            std_name = f"{base_std_name}_{counter}"

        seen_std_names.add(std_name.lower())

        escaped_orig = escape_sql_identifier(orig_col)
        escaped_std = escape_sql_identifier(std_name)
        col_expr = f'"{escaped_orig}"'
        if std_name.lower() in _NUMERIC_CLINICAL_CONCEPTS and "CHAR" in col_type:
            col_expr = f"TRY_CAST(NULLIF(TRIM(\"{escaped_orig}\"), '') AS DOUBLE)"
        if std_name != orig_col or col_expr != f'"{escaped_orig}"':
            select_exprs.append(f'{col_expr} AS "{escaped_std}"')
        else:
            select_exprs.append(f'"{escaped_orig}"')
    return select_exprs


def _ingest_single_tabular_file(
    conn: duckdb.DuckDBPyConnection,
    data_file: Path,
    data_dir: Path,
    prefix_subdirs: bool,
    stem_counts: dict[str, int],
    reverse_map: Dict[str, str],
    existing_tables: list[str],
    existing_views: set[str],
    loaded_tables: list[str],
    report_fn: Callable[[str], None],
) -> None:
    """
    Ingest a single tabular file into `conn` and record its manifest entry.

    Args:
    ----
        conn: Open writable DuckDB connection.
        data_file: Source tabular file path.
        data_dir: Root ingestion directory path.
        prefix_subdirs: Whether to prefix table names with relative subdirectory names.
        stem_counts: Frequency map of normalized file stems.
        reverse_map: Lowercase original column name to standardized clinical column name mapping.
        existing_tables: Existing table names before ingestion.
        existing_views: Existing view names in `conn`.
        loaded_tables: Mutable list of loaded table names.
        report_fn: Progress reporter callback.

    """
    _ = get_translator()
    current_source = str(data_file.resolve())
    table_name = _resolve_target_table_stem(
        data_file, data_dir, prefix_subdirs, stem_counts
    )
    counter = 1
    base_tbl = table_name
    is_already_loaded = False

    try:
        current_hash = compute_file_hash(data_file)
        manifest_row = conn.execute(
            "SELECT table_name FROM _t1d_ingestion_manifest WHERE source_file = ? AND file_hash = ?",
            [current_source, current_hash],
        ).fetchone()

        while table_name in existing_tables or table_name in loaded_tables:
            row = conn.execute(
                "SELECT comment FROM duckdb_tables() WHERE table_name = ?",
                [table_name],
            ).fetchone()
            existing_comment = row[0] if row else None
            if existing_comment == current_source or (
                manifest_row and manifest_row[0] == table_name
            ):
                is_already_loaded = True
                break
            counter += 1
            table_name = f"{base_tbl}_{counter}"

        if is_already_loaded:
            skip_msg = _("Table {} already exists, skipping...", table_name)
            logger.info(skip_msg)
            report_fn(skip_msg)
            return

        if table_name in existing_views:
            safe_view_name = escape_sql_identifier(table_name)
            conn.execute(f'DROP VIEW IF EXISTS "{safe_view_name}"')
            existing_views.discard(table_name)

        is_clinical = data_file.suffix.lower() in (".sas7bdat", ".xpt", ".sav")
        if is_clinical:
            load_clin_msg = _(
                "Loading {} (clinical) into table {}...",
                data_file.name,
                table_name,
            )
            logger.info(load_clin_msg)
            report_fn(load_clin_msg)
            clinical_df = _read_clinical_dataframe(data_file)
            conn.register("_clinical_df", clinical_df)
            conn.execute(
                "CREATE OR REPLACE TEMP VIEW temp_view AS SELECT * FROM _clinical_df"
            )
        else:
            _load_csv_or_parquet(
                conn=conn,
                data_file=data_file,
                table_name=table_name,
                report_fn=report_fn,
            )
        cols = conn.execute("DESCRIBE temp_view").fetchall()
        select_exprs = _build_column_select_expressions(cols, reverse_map)

        nl = chr(10)
        select_sql = f",{nl}                ".join(select_exprs)
        escaped_table = escape_sql_identifier(table_name)

        query = (
            f'CREATE TABLE "{escaped_table}" AS SELECT {nl}'
            f"                {select_sql} {nl}"
            f"            FROM temp_view"
        )
        conn.execute(query)
        conn.execute("DROP VIEW temp_view")
        if is_clinical:
            conn.unregister("_clinical_df")
        coerce_known_clinical_numeric_columns(conn, table_name)

        escaped_source = escape_sql_literal(current_source)
        conn.execute(f"COMMENT ON TABLE \"{escaped_table}\" IS '{escaped_source}'")

        row_count_res = conn.execute(
            f'SELECT COUNT(*) FROM "{escaped_table}"'
        ).fetchone()
        row_count = row_count_res[0] if row_count_res else 0
        file_size = data_file.stat().st_size
        conn.execute(
            "DELETE FROM _t1d_ingestion_manifest WHERE source_file = ? OR table_name = ?",
            [current_source, table_name],
        )
        conn.execute(
            "INSERT INTO _t1d_ingestion_manifest VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP, ?)",
            [current_source, table_name, current_hash, row_count, file_size],
        )

        loaded_tables.append(table_name)
    except Exception as e:
        err_load = _("Failed to load {}: {}", data_file.name, e)
        logger.error(err_load)
        report_fn(err_load)


def load_data_to_duckdb(
    data_dir_str: str,
    db_path: str,
    prefix_subdirs: bool = False,
    include_sas: bool = False,
    include_spss: bool = False,
    report_fn: Callable[[str], None] = print,
) -> None:
    """
    Load tabular files from the data directory into DuckDB tables.

    Supports CSV, TXT, Parquet, GZ, and optionally SAS and SPSS datasets.
    Tracks ingested file hashes in `_t1d_ingestion_manifest` to prevent redundant ingestion.

    Args:
    ----
        data_dir_str: Directory containing extracted CSV/TXT datasets.
        db_path: Path to the target DuckDB database file.
        prefix_subdirs: Whether to prefix table names with parent subdirectories to disambiguate.
        include_sas: Whether to include SAS clinical files (.sas7bdat, .xpt).
        include_spss: Whether to include SPSS clinical files (.sav).
        report_fn: Progress reporter callback (defaults to `print`).

    """
    _ = get_translator()
    data_dir = Path(data_dir_str)
    if not data_dir.exists():
        missing_msg = _("Data directory {} does not exist.", data_dir)
        logger.warning(missing_msg)
        report_fn(missing_msg)
        return

    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    global_read_only_duckdb_pool.close_path(db_path)
    conn_msg = _("Connecting to DuckDB at {}...", db_path)
    logger.info(conn_msg)
    report_fn(conn_msg)
    conn = duckdb.connect(db_path, read_only=False)
    try:
        _ensure_manifest_table(conn)

        scan_msg = _("Scanning for tabular files (CSV, TXT, Parquet, GZ)...")
        logger.info(scan_msg)
        report_fn(scan_msg)
        patterns = get_supported_tabular_globs(
            include_sas=include_sas,
            include_spss=include_spss,
        )

        raw_files: list[Path] = []
        for pat in patterns:
            raw_files.extend(data_dir.rglob(pat))
        data_files = [
            f
            for f in dict.fromkeys(raw_files)
            if not f.name.startswith("._")
            and "Glossary" not in f.name
            and "Protocol" not in f.name
        ]

        if not data_files:
            no_files_msg = _("No tabular files found.")
            logger.info(no_files_msg)
            report_fn(no_files_msg)
            return

        reverse_map: Dict[str, str] = {
            orig.lower(): std_name
            for std_name, orig_names in get_clinical_variable_mappings().items()
            for orig in orig_names
        }

        stem_counts: dict[str, int] = {}
        for df in data_files:
            s_name = Path(df.stem).stem if df.name.endswith(".csv.gz") else df.stem
            raw_stem = (
                re.sub(r"[^a-zA-Z0-9_]", "_", s_name).lower().strip("_") or "dataset"
            )
            stem_counts[raw_stem] = stem_counts.get(raw_stem, 0) + 1

        existing_views = {
            v[0]
            for v in conn.execute("SELECT view_name FROM duckdb_views()").fetchall()
        }
        existing_tables = [
            t[0]
            for t in conn.execute("SHOW TABLES").fetchall()
            if not t[0].startswith("_t1d_") and t[0] not in existing_views
        ]
        loaded_tables = list(existing_tables)
        for data_file in data_files:
            _ingest_single_tabular_file(
                conn=conn,
                data_file=data_file,
                data_dir=data_dir,
                prefix_subdirs=prefix_subdirs,
                stem_counts=stem_counts,
                reverse_map=reverse_map,
                existing_tables=existing_tables,
                existing_views=existing_views,
                loaded_tables=loaded_tables,
                report_fn=report_fn,
            )

        if loaded_tables:
            pop_msg = _(
                "Successfully populated {} tables in {}.", len(loaded_tables), db_path
            )
            logger.info(pop_msg)
            report_fn(pop_msg)
    finally:
        conn.close()
