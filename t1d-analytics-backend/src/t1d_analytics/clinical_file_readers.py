"""Clinical file reader adapters for SAS (.sas7bdat, .xpt), SPSS (.sav), CSV/TXT, CSV.GZ, and Parquet."""

import csv
import hashlib
import logging
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import duckdb

if TYPE_CHECKING:
    import pandas as pd

from t1d_analytics.errors import DatabaseError, UnsupportedFileFormatError
from t1d_analytics.i18n import get_translator
from t1d_analytics.sql_utils import escape_sql_literal

logger = logging.getLogger(__name__)


def compute_file_hash(path: Path) -> str:
    """
    Compute SHA-256 hash of a file.

    Args:
        path: Path to the file.

    Returns:
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


def read_clinical_dataframe(data_file: Path) -> "pd.DataFrame":
    """
    Read SAS (.sas7bdat, .xpt) or SPSS (.sav) data file into a pandas DataFrame.

    Translates SAS missing values and byte-encoded columns to standard UTF-8 nullable types.

    Args:
        data_file: Path to the clinical data file.

    Returns:
        pd.DataFrame: Parsed DataFrame with converted column data.

    Raises:
        UnsupportedFileFormatError: If the file format is unsupported.
        DatabaseError: If reading the supported file returns no DataFrame.

    """
    import pandas as pd

    suffix = data_file.suffix.lower()
    df: pd.DataFrame | None = None
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
        raise UnsupportedFileFormatError(
            f"Unsupported clinical file format: {suffix}",
            params={"format": suffix, "path": str(data_file)},
        )

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


def detect_csv_encoding_and_delimiter(csv_path: Path) -> tuple[str, str, str]:
    """
    Detect encoding and column delimiter for a tabular CSV or TXT file.

    Args:
        csv_path: Path to the CSV or TXT file on disk.

    Returns:
        tuple[str, str, str]: `(py_encoding, duckdb_encoding, duckdb_sep)`.

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
                f_chk.read(65536)
            py_encoding = "utf-8"
            duckdb_encoding = "utf-8"
        except UnicodeDecodeError:
            py_encoding = "latin1"
            duckdb_encoding = "latin-1"

    sep = ","
    tab_char = "\t"
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

    duckdb_sep = "\\t" if sep == tab_char else sep
    return py_encoding, duckdb_encoding, duckdb_sep


def load_csv_or_parquet_temp_view(
    conn: duckdb.DuckDBPyConnection,
    data_file: Path,
    table_name: str,
    report_fn: Callable[[str], None],
) -> None:
    """
    Create or replace `temp_view` in DuckDB for a Parquet, CSV.GZ, or CSV/TXT dataset file.

    Args:
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
            f"CREATE OR REPLACE TEMP VIEW temp_view AS SELECT * FROM read_csv('{escaped_path}', auto_detect=true, sample_size=-1, null_padding=true, strict_mode=false, nullstr=['', 'NA', 'N/A', '.', 'NULL', 'null', 'None', 'Unknown', 'Missing', '-'])"
        )
    else:
        _py_enc, duckdb_encoding, duckdb_sep = detect_csv_encoding_and_delimiter(
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
            f"CREATE OR REPLACE TEMP VIEW temp_view AS SELECT * FROM read_csv('{escaped_path}', auto_detect=true, sample_size=-1, null_padding=true, strict_mode=false, nullstr=['', 'NA', 'N/A', '.', 'NULL', 'null', 'None', 'Unknown', 'Missing', '-'], sep='{escaped_sep}', encoding='{escaped_enc}')"
        )


def load_tabular_file_into_temp_view(
    conn: duckdb.DuckDBPyConnection,
    data_file: Path,
    table_name: str,
    report_fn: Callable[[str], None],
) -> bool:
    """
    Stage a SAS, SPSS, Parquet, CSV.GZ, or CSV/TXT file as `temp_view` inside `conn`.

    Args:
        conn: Active writable DuckDB connection.
        data_file: Source dataset file path.
        table_name: Destination DuckDB table identifier.
        report_fn: Progress reporter callback.

    Returns:
        bool: True when `_clinical_df` was registered on `conn` (SAS/SPSS), False otherwise.

    """
    _ = get_translator()
    is_clinical = data_file.suffix.lower() in (".sas7bdat", ".xpt", ".sav")
    if is_clinical:
        load_clin_msg = _(
            "Loading {} (clinical) into table {}...",
            data_file.name,
            table_name,
        )
        logger.info(load_clin_msg)
        report_fn(load_clin_msg)
        clinical_df = read_clinical_dataframe(data_file)
        conn.register("_clinical_df", clinical_df)
        conn.execute(
            "CREATE OR REPLACE TEMP VIEW temp_view AS SELECT * FROM _clinical_df"
        )
    else:
        load_csv_or_parquet_temp_view(
            conn=conn,
            data_file=data_file,
            table_name=table_name,
            report_fn=report_fn,
        )
    return is_clinical


__all__ = [
    "compute_file_hash",
    "detect_csv_encoding_and_delimiter",
    "load_csv_or_parquet_temp_view",
    "load_tabular_file_into_temp_view",
    "read_clinical_dataframe",
]
