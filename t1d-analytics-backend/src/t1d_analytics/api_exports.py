"""Table data pagination, profiling, and streaming export services for the T1D Analytics REST API."""

import csv
import json
import logging
import math
import tempfile
from pathlib import Path
from typing import Callable, Iterator, List, Optional, Sequence, Tuple, Union

import duckdb
from fastapi.responses import StreamingResponse

from t1d_analytics.analytics import (
    TableProfileResult,
    profile_table,
)
from t1d_analytics.api_schemas import TableDataResponse
from t1d_analytics.db_security import validate_db_path
from t1d_analytics.errors import (
    DatabaseNotFoundError,
    ExportError,
    InvalidPaginationError,
    InvalidSortColumnError,
    InvalidSortOrderError,
    InvalidTableNameError,
    T1DError,
    TableNotFoundError,
)
from t1d_analytics.sql_utils import (
    SqlRawInputValue,
    SqlValue,
    escape_sql_identifier,
    sanitize_spreadsheet_cell,
    sanitize_sql_value,
)

logger = logging.getLogger(__name__)


def iter_sanitized_table_batches(
    result: duckdb.DuckDBPyConnection,
    batch_size: int = 5000,
) -> Iterator[List[List[SqlValue]]]:
    """
    Yield batches of spreadsheet-sanitized cell rows from an executed DuckDB query cursor.

    Args:
    ----
        result: Executed DuckDB query relation/cursor.
        batch_size: Number of rows to fetch per batch.

    Yields:
    ------
        List[List[SqlValue]]: Sanitized row batches.

    """
    while True:
        batch = result.fetchmany(batch_size)
        if not batch:
            break
        yield [[sanitize_spreadsheet_cell(val) for val in row] for row in batch]


def write_sanitized_excel_workbook(
    result: duckdb.DuckDBPyConnection,
    columns: Sequence[str],
    sheet_title: str,
    target_path: Union[str, Path],
    batch_size: int = 5000,
) -> None:
    """
    Write an executed DuckDB cursor's columns and sanitized rows into an OpenXML (.xlsx) workbook.

    Args:
    ----
        result: Executed DuckDB query relation/cursor.
        columns: Sequence of column header names.
        sheet_title: Title of the Excel worksheet (truncated to 31 characters).
        target_path: Destination filesystem path for the saved `.xlsx` file.
        batch_size: Number of rows fetched per batch.

    """
    from openpyxl import Workbook

    wb = Workbook(write_only=True)
    ws = wb.create_sheet(title=sheet_title[:31])
    ws.append(list(columns))

    for sanitized_batch in iter_sanitized_table_batches(result, batch_size=batch_size):
        for sanitized_row in sanitized_batch:
            ws.append(sanitized_row)

    wb.save(str(target_path))


def export_table(
    db_path: str,
    table_name: str,
    output_path: Union[str, Path],
    export_format: str = "excel",
) -> None:
    """
    Export a DuckDB table to Excel (.xlsx), Parquet, or CSV format with security sanitization.

    Args:
    ----
        db_path: Path to the DuckDB database file.
        table_name: Table identifier to export.
        output_path: Target filesystem path for the exported file.
        export_format: Format string, one of 'excel', 'parquet', or 'csv'.

    Raises:
    ------
        FileNotFoundError: If the database file does not exist.
        ValueError: If the table identifier is invalid, the table does not exist, or format is unsupported.

    """
    if not table_name.isidentifier():
        raise InvalidTableNameError(f"Invalid table identifier: {table_name}")

    if not Path(db_path).exists():
        raise FileNotFoundError(f"Database file not found: {db_path}")

    fmt = export_format.lower().strip()
    if fmt not in ("excel", "parquet", "csv"):
        raise ExportError(
            f"Unsupported export format: {export_format}. Choose 'excel', 'parquet', or 'csv'."
        )

    target_path = Path(output_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    conn = duckdb.connect(db_path, read_only=True)
    try:
        tables = [r[0] for r in conn.execute("SHOW TABLES").fetchall()]
        if table_name not in tables:
            raise TableNotFoundError(
                f"Table '{table_name}' does not exist in database."
            )

        escaped_tbl = table_name.replace('"', '""')

        if fmt == "parquet":
            escaped_out = str(target_path.resolve()).replace("'", "''")
            conn.execute(f"COPY \"{escaped_tbl}\" TO '{escaped_out}' (FORMAT PARQUET)")
        elif fmt == "csv":
            result = conn.execute(f'SELECT * FROM "{escaped_tbl}"')
            columns = (
                [desc[0] for desc in result.description] if result.description else []
            )
            with open(target_path, "w", encoding="utf-8", newline="") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(columns)
                for sanitized_batch in iter_sanitized_table_batches(result):
                    for sanitized_row in sanitized_batch:
                        writer.writerow(sanitized_row)
        else:
            result = conn.execute(f'SELECT * FROM "{escaped_tbl}"')
            columns = (
                [desc[0] for desc in result.description] if result.description else []
            )
            write_sanitized_excel_workbook(
                result=result,
                columns=columns,
                sheet_title=table_name,
                target_path=target_path,
            )
    finally:
        conn.close()


def _validate_table_and_db_path(
    table_name: str,
    db_path: Optional[str],
    db_validator: Callable[[Optional[str]], str],
) -> Tuple[str, str]:
    """
    Validate that the target DuckDB file exists and the table name is a valid identifier.

    Args:
    ----
        table_name: Target table name to validate.
        db_path: Optional user-supplied database path.
        db_validator: Security validator callable resolving `db_path`.

    Returns:
    -------
        Tuple[str, str]: `(target_db, escaped_table)` tuple.

    Raises:
    ------
        DatabaseNotFoundError: 404 if the database file does not exist.
        InvalidTableNameError: 400 if `table_name` is invalid.

    """
    target_db = db_validator(db_path)
    if not Path(target_db).exists():
        raise DatabaseNotFoundError(params={"path": target_db})

    if not table_name.isidentifier():
        raise InvalidTableNameError(params={})

    return target_db, escape_sql_identifier(table_name)


def _open_validated_table_connection(
    target_db: str,
    table_name: str,
) -> duckdb.DuckDBPyConnection:
    """
    Open a read-only DuckDB connection and verify that `table_name` exists in `SHOW TABLES`.

    Args:
    ----
        target_db: Validated filesystem path to the DuckDB database.
        table_name: Validated table identifier.

    Returns:
    -------
        duckdb.DuckDBPyConnection: Open read-only connection when the table exists.

    Raises:
    ------
        TableNotFoundError: 404 if `table_name` is not present in the database.

    """
    conn = duckdb.connect(target_db, read_only=True)
    try:
        tables = [row[0] for row in conn.execute("SHOW TABLES").fetchall()]
    except Exception:
        conn.close()
        raise
    if table_name not in tables:
        conn.close()
        raise TableNotFoundError(params={})
    return conn


def fetch_paginated_table_data(
    table_name: str,
    limit: int = 25,
    offset: int = 0,
    sort_by: Optional[str] = None,
    order: Optional[str] = "asc",
    search: Optional[str] = None,
    db_path: Optional[str] = None,
    db_validator: Callable[[Optional[str]], str] = validate_db_path,
) -> TableDataResponse:
    """
    Return paginated rows from a specific table with optional sorting and searching.

    Args:
    ----
        table_name: The table name.
        limit: Maximum rows per page.
        offset: Row offset for pagination.
        sort_by: Optional column name to sort by.
        order: Sort order, either 'asc' or 'desc' (default: 'asc').
        search: Optional string to filter across columns.
        db_path: Optional path to the DuckDB database.
        db_validator: Validator callable resolving db_path.

    Returns:
    -------
        TableDataResponse: Paginated rows and metadata.

    Raises:
    ------
        T1DError: On invalid table, pagination, sort parameters, or missing DB/table.

    """
    target_db, escaped_table = _validate_table_and_db_path(
        table_name, db_path, db_validator
    )

    if limit <= 0 or offset < 0:
        raise InvalidPaginationError(params={"limit": limit, "offset": offset})

    sort_order_clean = (order or "asc").strip().lower()
    if sort_order_clean not in ("asc", "desc"):
        raise InvalidSortOrderError(params={"order": order})

    conn = None
    try:
        conn = _open_validated_table_connection(target_db, table_name)

        cols_info = conn.execute(f'DESCRIBE "{escaped_table}"').fetchall()
        column_names = [c[0] for c in cols_info]

        if sort_by is not None and sort_by != "":
            if sort_by not in column_names:
                raise InvalidSortColumnError(params={"sort_by": sort_by})

        where_clause = ""
        params: List[str] = []
        if search is not None and search.strip():
            search_pattern = f"%{search.strip()}%"
            or_conditions = []
            for col in column_names:
                escaped_col = escape_sql_identifier(col)
                or_conditions.append(f'CAST("{escaped_col}" AS VARCHAR) ILIKE ?')
                params.append(search_pattern)
            where_clause = f"WHERE ({' OR '.join(or_conditions)})"

        limit = min(limit, 1000)

        count_query = f'SELECT COUNT(*) FROM "{escaped_table}" {where_clause}'.strip()
        count_result = conn.execute(count_query, params).fetchone()
        total_count = count_result[0] if count_result else 0
        total_pages = max(1, math.ceil(total_count / limit)) if limit > 0 else 1
        page = (offset // limit) + 1 if limit > 0 else 1

        if not column_names:
            order_by_clause = ""
        elif sort_by is not None and sort_by != "":
            escaped_sort = escape_sql_identifier(sort_by)
            order_by_clause = f'ORDER BY "{escaped_sort}" {sort_order_clean.upper()}'
        else:
            escaped_first = escape_sql_identifier(column_names[0])
            order_by_clause = f'ORDER BY "{escaped_first}" ASC'

        query = f'SELECT * FROM "{escaped_table}" {where_clause} {order_by_clause} LIMIT {limit} OFFSET {offset}'.strip()
        result = conn.execute(query, params)
        columns = [desc[0] for desc in result.description] if result.description else []
        rows = result.fetchall()
        output = [
            {col: sanitize_sql_value(val) for col, val in zip(columns, row)}
            for row in rows
        ]
        return TableDataResponse(
            rows=output, total_count=total_count, page=page, total_pages=total_pages
        )
    except T1DError:
        raise
    except Exception as e:
        err_msg = str(e).lower()
        if "cannot open file" in err_msg or "does not exist" in err_msg:
            raise DatabaseNotFoundError(params={"error": str(e)}) from e
        logger.error(f"Error fetching table data: {e}")
        raise ExportError(params={"error": str(e)}) from e
    finally:
        if conn is not None:
            conn.close()


def fetch_table_profile(
    table_name: str,
    db_path: Optional[str] = None,
    db_validator: Callable[[Optional[str]], str] = validate_db_path,
    profile_fn: Callable[[str, str], TableProfileResult] = profile_table,
) -> TableProfileResult:
    """
    Profile a table to return data quality, completeness, and column statistics.

    Args:
    ----
        table_name: The table name to profile.
        db_path: Optional path to the database file.
        db_validator: Validator callable resolving db_path.
        profile_fn: Profiling callable to invoke.

    Returns:
    -------
        TableProfileResult: Profiling summary dictionary.

    Raises:
    ------
        T1DError: 400 for invalid table, 404 if db or table not found, 500 otherwise.

    """
    target_db, _ = _validate_table_and_db_path(table_name, db_path, db_validator)

    try:
        return profile_fn(target_db, table_name)
    except (TableNotFoundError, ValueError) as ve:
        raise TableNotFoundError(params={"error": str(ve)}) from ve
    except (DatabaseNotFoundError, FileNotFoundError) as fnf:
        raise DatabaseNotFoundError(params={"error": str(fnf)}) from fnf
    except Exception as e:
        logger.error(f"Error profiling table {table_name}: {e}")
        raise ExportError(params={"error": str(e)}) from e


def stream_excel_export(
    table_name: str,
    db_path: Optional[str] = None,
    db_validator: Callable[[Optional[str]], str] = validate_db_path,
) -> StreamingResponse:
    """
    Export table data formatted as a native OpenXML Excel (.xlsx) workbook.

    Args:
    ----
        table_name: Table to export.
        db_path: Optional path to database file.
        db_validator: Validator callable resolving db_path.

    Returns:
    -------
        StreamingResponse: Streamed XLSX file with spreadsheetml MIME type.

    Raises:
    ------
        T1DError: If table or DB is invalid or export fails.

    """
    target_db, escaped_table = _validate_table_and_db_path(
        table_name, db_path, db_validator
    )

    conn = None
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
        temp_xlsx_file = tmp.name
    try:
        conn = _open_validated_table_connection(target_db, table_name)
        result = conn.execute(f'SELECT * FROM "{escaped_table}"')
        columns = [desc[0] for desc in result.description] if result.description else []

        def _generate_excel_chunks(
            file_path: str,
            db_conn: duckdb.DuckDBPyConnection,
            chunk_size: int = 65536,
        ) -> Iterator[bytes]:
            """
            Stream table data formatted as native OpenXML workbook in chunks via temporary file.

            Args:
            ----
                file_path: Path to the temporary XLSX file.
                db_conn: Open DuckDB connection closed when streaming completes.
                chunk_size: Byte size of chunks to stream.

            Yields:
            ------
                bytes: Binary XLSX chunk bytes.

            """
            try:
                write_sanitized_excel_workbook(
                    result=result,
                    columns=columns,
                    sheet_title=table_name,
                    target_path=file_path,
                )

                with open(file_path, "rb") as f:
                    while True:
                        data = f.read(chunk_size)
                        if not data:
                            break
                        yield data
            finally:
                db_conn.close()
                Path(file_path).unlink(missing_ok=True)

        return StreamingResponse(
            _generate_excel_chunks(temp_xlsx_file, conn),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={
                "Content-Disposition": f'attachment; filename="{table_name}_export.xlsx"'
            },
        )
    except T1DError:
        Path(temp_xlsx_file).unlink(missing_ok=True)
        raise
    except Exception as e:
        if conn is not None:
            conn.close()
        Path(temp_xlsx_file).unlink(missing_ok=True)
        logger.error(f"Error exporting table to excel: {e}")
        raise ExportError(params={"error": str(e)}) from e


def stream_parquet_export(
    table_name: str,
    db_path: Optional[str] = None,
    db_validator: Callable[[Optional[str]], str] = validate_db_path,
) -> StreamingResponse:
    """
    Export table data in columnar Apache Parquet format.

    Args:
    ----
        table_name: Table to export.
        db_path: Optional path to database file.
        db_validator: Validator callable resolving db_path.

    Returns:
    -------
        StreamingResponse: Streamed Parquet file.

    Raises:
    ------
        T1DError: If table or DB is invalid or export fails.

    """
    target_db, escaped_table = _validate_table_and_db_path(
        table_name, db_path, db_validator
    )

    conn = None
    temp_parquet_file = None
    try:
        conn = _open_validated_table_connection(target_db, table_name)
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
            temp_parquet_file = tmp.name

        escaped_out_path = temp_parquet_file.replace("'", "''")
        conn.execute(
            f"COPY \"{escaped_table}\" TO '{escaped_out_path}' (FORMAT PARQUET, COMPRESSION ZSTD)"
        )

        def _iter_parquet_chunks(
            file_path: str,
            db_conn: duckdb.DuckDBPyConnection,
            chunk_size: int = 65536,
        ) -> Iterator[bytes]:
            """
            Iterate over generated Parquet file in chunks and clean up database and filesystem upon exit.

            Args:
            ----
                file_path: Path to the temporary Parquet file.
                db_conn: Open DuckDB connection to close after streaming.
                chunk_size: Byte size of chunks to stream.

            Yields:
            ------
                bytes: Binary Parquet chunk.

            """
            try:
                with open(file_path, "rb") as f:
                    while True:
                        chunk = f.read(chunk_size)
                        if not chunk:
                            break
                        yield chunk
            finally:
                db_conn.close()
                Path(file_path).unlink(missing_ok=True)

        return StreamingResponse(
            _iter_parquet_chunks(temp_parquet_file, conn),
            media_type="application/vnd.apache.parquet",
            headers={
                "Content-Disposition": f'attachment; filename="{table_name}_export.parquet"'
            },
        )
    except T1DError:
        raise
    except Exception as e:
        if conn is not None:
            conn.close()
        if temp_parquet_file:
            Path(temp_parquet_file).unlink(missing_ok=True)
        logger.error(f"Error exporting table to parquet: {e}")
        raise ExportError(params={"error": str(e)}) from e


def _csv_quote(s: str) -> str:
    """
    Escape double quotes for standard CSV field representation.

    Args:
    ----
        s: Raw cell string.

    Returns:
    -------
        str: Quoted CSV string.

    """
    return '"' + s.replace('"', '""') + '"'


def stream_csv_export(
    table_name: str,
    db_path: Optional[str] = None,
    db_validator: Callable[[Optional[str]], str] = validate_db_path,
) -> StreamingResponse:
    """
    Export table data formatted as a streaming CSV file with spreadsheet formula defense.

    Args:
    ----
        table_name: Table to export.
        db_path: Optional path to database file.
        db_validator: Validator callable resolving db_path.

    Returns:
    -------
        StreamingResponse: Streamed CSV file with text/csv MIME type.

    Raises:
    ------
        T1DError: If table or DB is invalid or export fails.

    """
    target_db, escaped_table = _validate_table_and_db_path(
        table_name, db_path, db_validator
    )

    conn = None
    try:
        conn = _open_validated_table_connection(target_db, table_name)
        result = conn.execute(f'SELECT * FROM "{escaped_table}"')
        columns = [desc[0] for desc in result.description] if result.description else []
        first_batch: Sequence[Sequence[SqlRawInputValue]] = result.fetchmany(500)

        def _generate_csv_chunks(
            db_conn: duckdb.DuckDBPyConnection,
            initial_batch: Sequence[Sequence[SqlRawInputValue]],
        ) -> Iterator[bytes]:
            """
            Stream CSV header and sanitized row batches lazily without full-table buffering.

            Args:
            ----
                db_conn: Open DuckDB connection closed when streaming completes.
                initial_batch: Prefetched first batch of rows.

            Yields:
            ------
                bytes: UTF-8 encoded CSV chunks starting with UTF-8 BOM.

            """
            try:
                crlf = chr(13) + chr(10)
                header_line = ",".join(_csv_quote(col) for col in columns) + crlf
                yield bytes([0xEF, 0xBB, 0xBF]) + header_line.encode("utf-8")

                batch = initial_batch
                while batch:
                    lines: List[str] = []
                    for row in batch:
                        formatted_cells: List[str] = []
                        for val in row:
                            if val is None:
                                formatted_cells.append('""')
                            elif isinstance(val, (int, float, bool)):
                                formatted_cells.append(str(val))
                            else:
                                s_val = str(sanitize_spreadsheet_cell(str(val)))
                                formatted_cells.append(_csv_quote(s_val))
                        lines.append(",".join(formatted_cells) + crlf)
                    yield "".join(lines).encode("utf-8")
                    batch = result.fetchmany(500)
            finally:
                db_conn.close()

        return StreamingResponse(
            _generate_csv_chunks(conn, first_batch),
            media_type="text/csv",
            headers={
                "Content-Disposition": f'attachment; filename="{table_name}_export.csv"',
                "Cache-Control": "no-cache",
            },
        )
    except T1DError:
        raise
    except Exception as e:
        if conn is not None:
            conn.close()
        logger.error(f"Error exporting table to csv: {e}")
        raise ExportError(params={"error": str(e)}) from e


def stream_json_export(
    table_name: str,
    db_path: Optional[str] = None,
    db_validator: Callable[[Optional[str]], str] = validate_db_path,
) -> StreamingResponse:
    """
    Export table data formatted as a streaming JSON array of row objects.

    Args:
    ----
        table_name: Table to export.
        db_path: Optional path to database file.
        db_validator: Validator callable resolving db_path.

    Returns:
    -------
        StreamingResponse: Streamed JSON file with application/json MIME type.

    Raises:
    ------
        T1DError: If table or DB is invalid or export fails.

    """
    target_db, escaped_table = _validate_table_and_db_path(
        table_name, db_path, db_validator
    )

    conn = None
    try:
        conn = _open_validated_table_connection(target_db, table_name)
        result = conn.execute(f'SELECT * FROM "{escaped_table}"')
        columns = [desc[0] for desc in result.description] if result.description else []
        first_batch: Sequence[Sequence[SqlRawInputValue]] = result.fetchmany(500)

        def _generate_json_chunks(
            db_conn: duckdb.DuckDBPyConnection,
            initial_batch: Sequence[Sequence[SqlRawInputValue]],
        ) -> Iterator[bytes]:
            """
            Stream JSON array brackets and serialized row objects in batches.

            Args:
            ----
                db_conn: Open DuckDB connection closed when streaming completes.
                initial_batch: Prefetched first batch of rows.

            Yields:
            ------
                bytes: UTF-8 encoded JSON array chunks.

            """
            try:
                nl = chr(10)
                yield f"[{nl}".encode("utf-8")
                first_row = True
                batch = initial_batch
                while batch:
                    chunk_parts: List[str] = []
                    for row in batch:
                        row_dict = dict(zip(columns, row))
                        prefix = "  " if first_row else f",{nl}  "
                        first_row = False
                        chunk_parts.append(prefix + json.dumps(row_dict, default=str))
                    yield "".join(chunk_parts).encode("utf-8")
                    batch = result.fetchmany(500)
                yield f"{nl}]{nl}".encode("utf-8")
            finally:
                db_conn.close()

        return StreamingResponse(
            _generate_json_chunks(conn, first_batch),
            media_type="application/json",
            headers={
                "Content-Disposition": f'attachment; filename="{table_name}_export.json"',
                "Cache-Control": "no-cache",
            },
        )
    except T1DError:
        raise
    except Exception as e:
        if conn is not None:
            conn.close()
        logger.error(f"Error exporting table to json: {e}")
        raise ExportError(params={"error": str(e)}) from e
