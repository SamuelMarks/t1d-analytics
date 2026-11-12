"""DuckDB transactional clinical table loading and archive ingestion pipeline."""

import logging
from collections.abc import Callable, Sequence
from pathlib import Path

import duckdb
from typing_extensions import TypedDict

from t1d_analytics.archive_extractor import (
    DEFAULT_MAX_DECOMPRESSED_BYTES,
    MAX_MEMBER_COMPRESSION_RATIO,
    MIN_RATIO_CHECK_BYTES,
    resolve_effective_decompressed_quota,
)
from t1d_analytics.archive_extractor import (
    extract_zips as _archive_extract_zips,
)
from t1d_analytics.clinical_file_readers import (
    compute_file_hash,
    detect_csv_encoding_and_delimiter,
    load_csv_or_parquet_temp_view,
    load_tabular_file_into_temp_view,
    read_clinical_dataframe,
)
from t1d_analytics.duckdb_pool import global_read_only_duckdb_pool
from t1d_analytics.errors import DatabaseError, UnsupportedFileFormatError
from t1d_analytics.i18n import get_translator
from t1d_analytics.ingestion_study_resolver import (
    CLINICAL_NUMERIC_COLUMNS,
    JAEB_STUDY_ACRONYMS,
    build_column_select_expressions,
    discover_tabular_files,
    get_clinical_variable_mappings,
    get_supported_tabular_globs,
    resolve_subdir_prefix,
    resolve_target_table_stem,
)
from t1d_analytics.sql_utils import (
    escape_sql_identifier,
    escape_sql_literal,
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


def get_ingestion_manifest(db_path: str) -> list[IngestionManifestRecord]:
    """
    Retrieve the ingestion manifest audit records from DuckDB.

    Args:
        db_path: Path to the DuckDB database file.

    Returns:
        list[IngestionManifestRecord]: List of manifest records containing source_file,
            table_name, file_hash, row_count, loaded_at, and file_size.

    """
    if not Path(db_path).exists():
        return []

    with duckdb.connect(db_path, read_only=True) as conn:
        has_manifest = conn.execute(
            "SELECT 1 FROM information_schema.tables WHERE table_name = '_t1d_ingestion_manifest'"
        ).fetchone()
        if not has_manifest:
            return []

        rows = conn.execute(
            "SELECT source_file, table_name, file_hash, row_count, loaded_at, file_size FROM _t1d_ingestion_manifest ORDER BY loaded_at"
        ).fetchall()

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


def coerce_known_clinical_numeric_columns(
    conn: duckdb.DuckDBPyConnection,
    table_name: str,
) -> None:
    """
    Alter known clinical numeric columns stored as `VARCHAR` in an ingested DuckDB table to `DOUBLE`.

    Args:
        conn: Active writable DuckDB connection.
        table_name: Target DuckDB table identifier.

    """
    safe_tbl = escape_sql_identifier(table_name)
    cols = conn.execute(f'DESCRIBE "{safe_tbl}"').fetchall()
    for col_row in cols:
        col_name = str(col_row[0])
        col_type = str(col_row[1]).upper()
        if col_name.lower() in CLINICAL_NUMERIC_COLUMNS and "CHAR" in col_type:
            safe_col = escape_sql_identifier(col_name)
            quoted_col = f'"{safe_col}"'
            conn.execute(
                f'ALTER TABLE "{safe_tbl}" ALTER COLUMN {quoted_col} '
                f"TYPE DOUBLE USING TRY_CAST(NULLIF(TRIM({quoted_col}), '') AS DOUBLE)"
            )


def ensure_manifest_table(conn: duckdb.DuckDBPyConnection) -> None:
    """
    Ensure the `_t1d_ingestion_manifest` tracking table exists in DuckDB.

    Args:
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


def resolve_unique_table_name_or_skip(
    conn: duckdb.DuckDBPyConnection,
    base_tbl: str,
    current_source: str,
    current_hash: str,
    existing_tables: Sequence[str],
    loaded_tables: Sequence[str],
) -> tuple[str, bool]:
    """
    Resolve a non-colliding `table_name` or detect if `current_source` was already loaded.

    Args:
        conn: Active DuckDB connection.
        base_tbl: Candidate table stem.
        current_source: Resolved source file path string.
        current_hash: SHA-256 digest of the source file.
        existing_tables: Table names existing before this ingestion run.
        loaded_tables: Table names loaded so far in this ingestion run.

    Returns:
        tuple[str, bool]: `(resolved_table_name, is_already_loaded)`.

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


def materialize_temp_view_into_table(
    conn: duckdb.DuckDBPyConnection,
    table_name: str,
    current_source: str,
    current_hash: str,
    file_size: int,
    reverse_map: dict[str, str],
    is_clinical: bool,
) -> None:
    """
    Materialize `temp_view` into `table_name` transactionally, coerce numeric columns, and record manifest metadata.

    Args:
        conn: Active writable DuckDB connection.
        table_name: Target DuckDB table name.
        current_source: Resolved source file path string.
        current_hash: SHA-256 hash of the source file.
        file_size: File size in bytes.
        reverse_map: Clinical column mapping dictionary.
        is_clinical: Whether `_clinical_df` was registered for SAS/SPSS ingestion.

    """
    cols = conn.execute("DESCRIBE temp_view").fetchall()
    select_exprs = build_column_select_expressions(cols, reverse_map)

    select_sql = ",\n                ".join(select_exprs)
    escaped_table = escape_sql_identifier(table_name)
    quoted_table = f'"{escaped_table}"'
    escaped_source = escape_sql_literal(current_source)

    conn.execute("BEGIN TRANSACTION")
    try:
        query = (
            f"CREATE TABLE {quoted_table} AS SELECT \n"
            f"                {select_sql} \n"
            f"            FROM temp_view"
        )
        conn.execute(query)
        coerce_known_clinical_numeric_columns(conn, table_name)
        conn.execute(f"COMMENT ON TABLE {quoted_table} IS '{escaped_source}'")

        row_count_res = conn.execute(f"SELECT COUNT(*) FROM {quoted_table}").fetchone()
        row_count = row_count_res[0] if row_count_res else 0
        conn.execute(
            "DELETE FROM _t1d_ingestion_manifest WHERE source_file = ? OR table_name = ?",
            [current_source, table_name],
        )
        conn.execute(
            "INSERT INTO _t1d_ingestion_manifest VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP, ?)",
            [current_source, table_name, current_hash, row_count, file_size],
        )
        conn.execute("COMMIT")
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    finally:
        conn.execute("DROP VIEW IF EXISTS temp_view")
        if is_clinical:
            conn.unregister("_clinical_df")


def ingest_single_tabular_file(
    conn: duckdb.DuckDBPyConnection,
    data_file: Path,
    data_dir: Path,
    prefix_subdirs: bool,
    stem_counts: dict[str, int],
    reverse_map: dict[str, str],
    existing_tables: list[str],
    existing_views: set[str],
    loaded_tables: list[str],
    report_fn: Callable[[str], None],
) -> None:
    """
    Ingest a single tabular file into `conn` and record its manifest entry.

    Args:
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
    base_tbl = resolve_target_table_stem(
        data_file, data_dir, prefix_subdirs, stem_counts
    )
    if len(existing_tables) > 25 and base_tbl in {
        "patients",
        "cgm_readings",
        "hba1c_labs",
        "dka_adverse_events",
    }:
        skip_syn_msg = _(
            "Table {} already exists, skipping...",
            base_tbl,
        )
        logger.info(skip_syn_msg)
        report_fn(skip_syn_msg)
        return

    try:
        current_hash = compute_file_hash(data_file)
        table_name, is_already_loaded = resolve_unique_table_name_or_skip(
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

        is_clinical = load_tabular_file_into_temp_view(
            conn=conn,
            data_file=data_file,
            table_name=table_name,
            report_fn=report_fn,
        )

        materialize_temp_view_into_table(
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
        UnsupportedFileFormatError,
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
        ensure_manifest_table(conn)

        scan_msg = _("Scanning for tabular files (CSV, TXT, Parquet, GZ)...")
        logger.info(scan_msg)
        report_fn(scan_msg)
        data_files, stem_counts = discover_tabular_files(
            data_dir=data_dir,
            include_sas=include_sas,
            include_spss=include_spss,
        )
        if not data_files:
            no_files_msg = _("No tabular files found.")
            logger.info(no_files_msg)
            report_fn(no_files_msg)
            return

        reverse_map: dict[str, str] = {
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
            ingest_single_tabular_file(
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
    "CLINICAL_NUMERIC_COLUMNS",
    "DEFAULT_MAX_DECOMPRESSED_BYTES",
    "JAEB_STUDY_ACRONYMS",
    "MAX_MEMBER_COMPRESSION_RATIO",
    "MIN_RATIO_CHECK_BYTES",
    "IngestionManifestRecord",
    "build_column_select_expressions",
    "coerce_known_clinical_numeric_columns",
    "compute_file_hash",
    "detect_csv_encoding_and_delimiter",
    "discover_tabular_files",
    "ensure_manifest_table",
    "extract_zips",
    "get_clinical_variable_mappings",
    "get_ingestion_manifest",
    "get_supported_tabular_globs",
    "ingest_single_tabular_file",
    "load_csv_or_parquet_temp_view",
    "load_data_to_duckdb",
    "load_tabular_file_into_temp_view",
    "materialize_temp_view_into_table",
    "read_clinical_dataframe",
    "resolve_effective_decompressed_quota",
    "resolve_subdir_prefix",
    "resolve_target_table_stem",
    "resolve_unique_table_name_or_skip",
]
