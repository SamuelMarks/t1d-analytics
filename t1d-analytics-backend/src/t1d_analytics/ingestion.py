"""Archive extraction, clinical dataset parsing, and DuckDB ingestion pipeline."""

import csv
import hashlib
import json
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Sequence, Tuple

import duckdb
from typing_extensions import TypedDict

if TYPE_CHECKING:
    import pandas as pd

from t1d_analytics.archive_extractor import (
    _ARCHIVE_GLOB_PATTERNS,
    _TABULAR_AND_ARCHIVE_SUFFIXES,
    DEFAULT_MAX_DECOMPRESSED_BYTES,
    MAX_MEMBER_COMPRESSION_RATIO,
    MIN_RATIO_CHECK_BYTES,
    _discover_unprocessed_archives,
    _extract_single_archive_file,
    _extract_single_zip,
    _extract_tar_archive,
    _extract_zip_archive,
    _is_already_extracted,
    _is_safe_member_target,
    _resolve_archive_extract_dir,
    _should_skip_archive_member_name,
    resolve_effective_decompressed_quota,
)
from t1d_analytics.archive_extractor import (
    extract_zips as _archive_extract_zips,
)
from t1d_analytics.errors import DatabaseError
from t1d_analytics.i18n import get_translator
from t1d_analytics.sql_utils import (
    escape_sql_identifier,
    escape_sql_literal,
    global_read_only_duckdb_pool,
)

logger = logging.getLogger(__name__)


class IngestionManifestRecord(TypedDict):
    """Audit record for an ingested clinical dataset file."""

    source_file: str
    table_name: str
    file_hash: str
    row_count: int
    loaded_at: str
    file_size: int


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

    Args:
    ----
        data_dir_str: Directory path containing archive files to extract.
        max_depth: Maximum recursion depth for decompressing nested archives.
        report_fn: Progress reporter callback (defaults to `print`).
        max_decompressed_bytes: Maximum cumulative decompressed size in bytes allowed across archives.
        overwrite: Whether to force re-extraction of existing member files even if sizes match.
        tabular_only: Whether to only extract tabular data files and nested archives.

    """
    _archive_extract_zips(
        data_dir_str=data_dir_str,
        max_depth=max_depth,
        report_fn=report_fn,
        max_decompressed_bytes=max_decompressed_bytes,
        overwrite=overwrite,
        tabular_only=tabular_only,
        min_ratio_check_bytes=MIN_RATIO_CHECK_BYTES,
        max_member_compression_ratio=MAX_MEMBER_COMPRESSION_RATIO,
    )


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

    Raises:
    ------
        DatabaseError: If the file format is unsupported or reading fails.

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
        except (ImportError, OSError, RuntimeError, ValueError):
            fmt = "sas7bdat" if suffix == ".sas7bdat" else "xport"
            df = pd.read_sas(str(data_file), format=fmt)
    elif suffix == ".sav":
        try:
            import pyreadstat

            df, _ = pyreadstat.read_sav(str(data_file))
        except (ImportError, OSError, RuntimeError, ValueError):
            df = pd.read_spss(str(data_file))
    else:
        raise DatabaseError(f"Unsupported clinical file format: {suffix}")

    if df is None:
        raise DatabaseError(f"Could not read clinical file {data_file}")

    df.columns = [
        col.decode("utf-8", errors="ignore") if isinstance(col, bytes) else str(col)
        for col in df.columns
    ]
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
    dq = chr(34)
    safe_tbl = escape_sql_identifier(table_name)
    cols = conn.execute(f'DESCRIBE "{safe_tbl}"').fetchall()
    for col_row in cols:
        col_name = str(col_row[0])
        col_type = str(col_row[1]).upper()
        if col_name.lower() in _NUMERIC_CLINICAL_CONCEPTS and "CHAR" in col_type:
            safe_col = escape_sql_identifier(col_name)
            conn.execute(
                f'ALTER TABLE "{safe_tbl}" ALTER COLUMN "{safe_col}" '
                f"TYPE DOUBLE USING TRY_CAST(NULLIF(TRIM({dq}{safe_col}{dq}), '') AS DOUBLE)"
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
            with open(matches_file) as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return {
                        str(k): [str(x) for x in v]
                        for k, v in data.items()
                        if isinstance(v, list)
                    }
        except (OSError, json.JSONDecodeError, ValueError, TypeError):
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
            with open(csv_path, encoding="utf-8") as f_chk:
                f_chk.read(8192)
            py_encoding = "utf-8"
            duckdb_encoding = "utf-8"
        except UnicodeDecodeError:
            py_encoding = "latin1"
            duckdb_encoding = "latin-1"

    sep = ","
    tab_char = chr(9)
    try:
        with open(csv_path, encoding=py_encoding, errors="ignore") as f:
            sample = "".join([f.readline() for _ in range(5)])
        if sample.strip():
            dialect = csv.Sniffer().sniff(sample, delimiters=f",{tab_char}|;")
            sep = dialect.delimiter
    except (csv.Error, OSError, ValueError):
        with open(csv_path, encoding=py_encoding, errors="ignore") as f:
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

        dq = chr(34)
        escaped_orig = escape_sql_identifier(orig_col)
        escaped_std = escape_sql_identifier(std_name)
        col_expr = f'"{escaped_orig}"'
        if std_name.lower() in _NUMERIC_CLINICAL_CONCEPTS and "CHAR" in col_type:
            col_expr = f"TRY_CAST(NULLIF(TRIM({dq}{escaped_orig}{dq}), '') AS DOUBLE)"
        if std_name != orig_col or col_expr != f'"{escaped_orig}"':
            select_exprs.append(f'{col_expr} AS "{escaped_std}"')
        else:
            select_exprs.append(f'"{escaped_orig}"')
    return select_exprs


def _resolve_unique_table_name_or_skip(
    conn: duckdb.DuckDBPyConnection,
    base_tbl: str,
    current_source: str,
    current_hash: str,
    existing_tables: Sequence[str],
    loaded_tables: Sequence[str],
) -> Tuple[str, bool]:
    """
    Resolve a non-colliding `table_name` or detect if `current_source` was already loaded.

    Args:
    ----
        conn: Active DuckDB connection.
        base_tbl: Candidate table stem.
        current_source: Resolved source file path string.
        current_hash: SHA-256 digest of the source file.
        existing_tables: Table names existing before this ingestion run.
        loaded_tables: Table names loaded so far in this ingestion run.

    Returns:
    -------
        Tuple[str, bool]: `(resolved_table_name, is_already_loaded)`.

    """
    manifest_row = conn.execute(
        "SELECT table_name FROM _t1d_ingestion_manifest WHERE source_file = ? AND file_hash = ?",
        [current_source, current_hash],
    ).fetchone()
    table_name = base_tbl
    counter = 1
    while table_name in existing_tables or table_name in loaded_tables:
        row = conn.execute(
            "SELECT comment FROM duckdb_tables() WHERE table_name = ?",
            [table_name],
        ).fetchone()
        existing_comment = row[0] if row else None
        if existing_comment == current_source or (
            manifest_row and manifest_row[0] == table_name
        ):
            return table_name, True
        counter += 1
        table_name = f"{base_tbl}_{counter}"
    return table_name, False


def _materialize_temp_view_into_table(
    conn: duckdb.DuckDBPyConnection,
    table_name: str,
    current_source: str,
    current_hash: str,
    file_size: int,
    reverse_map: Dict[str, str],
    is_clinical: bool,
) -> None:
    """
    Materialize `temp_view` into `table_name`, coerce numeric columns, and record manifest metadata.

    Args:
    ----
        conn: Active writable DuckDB connection.
        table_name: Target DuckDB table name.
        current_source: Resolved source file path string.
        current_hash: SHA-256 hash of the source file.
        file_size: File size in bytes.
        reverse_map: Clinical column mapping dictionary.
        is_clinical: Whether `_clinical_df` was registered for SAS/SPSS ingestion.

    """
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

    dq = chr(34)
    escaped_source = escape_sql_literal(current_source)
    conn.execute(f"COMMENT ON TABLE {dq}{escaped_table}{dq} IS '{escaped_source}'")

    row_count_res = conn.execute(f'SELECT COUNT(*) FROM "{escaped_table}"').fetchone()
    row_count = row_count_res[0] if row_count_res else 0
    conn.execute(
        "DELETE FROM _t1d_ingestion_manifest WHERE source_file = ? OR table_name = ?",
        [current_source, table_name],
    )
    conn.execute(
        "INSERT INTO _t1d_ingestion_manifest VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP, ?)",
        [current_source, table_name, current_hash, row_count, file_size],
    )


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
    base_tbl = _resolve_target_table_stem(
        data_file, data_dir, prefix_subdirs, stem_counts
    )

    try:
        current_hash = compute_file_hash(data_file)
        table_name, is_already_loaded = _resolve_unique_table_name_or_skip(
            conn=conn,
            base_tbl=base_tbl,
            current_source=current_source,
            current_hash=current_hash,
            existing_tables=existing_tables,
            loaded_tables=loaded_tables,
        )
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

        _materialize_temp_view_into_table(
            conn=conn,
            table_name=table_name,
            current_source=current_source,
            current_hash=current_hash,
            file_size=data_file.stat().st_size,
            reverse_map=reverse_map,
            is_clinical=is_clinical,
        )
        loaded_tables.append(table_name)
    except (
        DatabaseError,
        duckdb.Error,
        OSError,
        UnicodeError,
        ValueError,
        RuntimeError,
    ) as e:
        err_load = _("Failed to load {}: {}", data_file.name, e)
        logger.error(err_load)
        report_fn(err_load)


_ingest_single_file = _ingest_single_tabular_file


def _discover_tabular_files(
    data_dir: Path,
    include_sas: bool,
    include_spss: bool,
) -> Tuple[List[Path], dict[str, int]]:
    """
    Scan `data_dir` for supported tabular files and compute stem collision counts.

    Args:
    ----
        data_dir: Root directory containing tabular datasets.
        include_sas: Whether to include SAS files.
        include_spss: Whether to include SPSS files.

    Returns:
    -------
        Tuple[List[Path], dict[str, int]]: `(discovered_files, stem_frequency_map)`.

    """
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
    stem_counts: dict[str, int] = {}
    for df in data_files:
        s_name = Path(df.stem).stem if df.name.endswith(".csv.gz") else df.stem
        raw_stem = re.sub(r"[^a-zA-Z0-9_]", "_", s_name).lower().strip("_") or "dataset"
        stem_counts[raw_stem] = stem_counts.get(raw_stem, 0) + 1
    return data_files, stem_counts


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
        data_files, stem_counts = _discover_tabular_files(
            data_dir=data_dir,
            include_sas=include_sas,
            include_spss=include_spss,
        )
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


__all__ = [
    "DEFAULT_MAX_DECOMPRESSED_BYTES",
    "MAX_MEMBER_COMPRESSION_RATIO",
    "MIN_RATIO_CHECK_BYTES",
    "_ARCHIVE_GLOB_PATTERNS",
    "_CLINICAL_NUMERIC_COLUMNS",
    "_NUMERIC_CLINICAL_CONCEPTS",
    "_TABULAR_AND_ARCHIVE_SUFFIXES",
    "IngestionManifestRecord",
    "_build_column_select_expressions",
    "_detect_csv_encoding_and_delimiter",
    "_discover_tabular_files",
    "_discover_unprocessed_archives",
    "_ensure_manifest_table",
    "_extract_single_archive_file",
    "_extract_single_zip",
    "_extract_tar_archive",
    "_extract_zip_archive",
    "_ingest_single_file",
    "_ingest_single_tabular_file",
    "_is_already_extracted",
    "_is_safe_member_target",
    "_load_csv_or_parquet",
    "_materialize_temp_view_into_table",
    "_read_clinical_dataframe",
    "_resolve_archive_extract_dir",
    "_resolve_target_table_stem",
    "_resolve_unique_table_name_or_skip",
    "_should_skip_archive_member_name",
    "coerce_known_clinical_numeric_columns",
    "compute_file_hash",
    "extract_zips",
    "get_clinical_variable_mappings",
    "get_ingestion_manifest",
    "get_supported_tabular_globs",
    "load_data_to_duckdb",
    "resolve_effective_decompressed_quota",
]
