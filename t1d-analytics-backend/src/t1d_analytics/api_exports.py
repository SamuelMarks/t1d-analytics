"""Table data pagination, profiling, and streaming export services for the T1D Analytics REST API."""

from __future__ import annotations

import csv
import json
import logging
import math
import re
import tempfile
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path

import duckdb
from fastapi.responses import StreamingResponse

from t1d_analytics.analytics import (
    TableProfileResult,
    profile_table,
)
from t1d_analytics.api_schemas import TableDataResponse
from t1d_analytics.db_security import validate_db_path
from t1d_analytics.duckdb_pool import (
    ReadOnlyDuckDbCursorProxy,
    attach_jaeb_read_only_if_present,
    connect_duckdb_read_only_cursor,
    create_read_only_cursor_proxy,
)
from t1d_analytics.errors import (
    DatabaseNotFoundError,
    ExportError,
    InvalidPaginationError,
    InvalidSortColumnError,
    InvalidSortOrderError,
    InvalidTableNameError,
    T1DError,
    TableNotFoundError,
    UnsupportedExportFormatError,
)
from t1d_analytics.sql_utils import (
    SqlRawInputValue,
    SqlValue,
    escape_sql_identifier,
    sanitize_spreadsheet_cell,
    sanitize_sql_value,
)

logger = logging.getLogger(__name__)


def _copy_table_to_parquet_file(
    conn: duckdb.DuckDBPyConnection
    | ReadOnlyDuckDbCursorProxy
    | _ClosingReadOnlyTableCursor,
    escaped_table: str,
    target_path: str | Path,
    source_db_path: str | Path,
    use_zstd_compression: bool = False,
) -> None:
    """
    Stream a DuckDB table to a Parquet file via `COPY (SELECT ...) TO '...' (FORMAT PARQUET)` without loading rows into Pandas RAM.

    When a pooled read-only connection on the same DuckDB database file has executed
    ``SET enable_external_access = false``, DuckDB blocks direct ``COPY TO`` on that
    ``DatabaseInstance``. In that case, an isolated ``:memory:`` DuckDB engine attaches
    `source_db_path` in ``READ_ONLY`` mode and streams ``COPY (SELECT * FROM ...)``
    directly to `target_path` in C++ with bounded memory.

    Args:
        conn: Open DuckDB connection or read-only cursor proxy to the source database.
        escaped_table: Double-quote-escaped table identifier.
        target_path: Destination filesystem path for the Parquet file.
        source_db_path: Path to the source DuckDB file used when attaching from ``:memory:``.
        use_zstd_compression: Whether to include ``COMPRESSION ZSTD`` in the ``COPY`` options.

    """
    dq = chr(34)
    escaped_out = str(target_path).replace("'", "''")
    format_opts = (
        "FORMAT PARQUET, COMPRESSION ZSTD" if use_zstd_compression else "FORMAT PARQUET"
    )
    copy_sql = f"COPY (SELECT * FROM {dq}{escaped_table}{dq}) TO '{escaped_out}' ({format_opts})"
    try:
        conn.execute(copy_sql)
    except duckdb.PermissionException:
        escaped_src = str(Path(source_db_path).resolve()).replace("'", "''")
        mem_conn = duckdb.connect(":memory:")
        try:
            mem_conn.execute(f"ATTACH '{escaped_src}' AS _t1d_export_src (READ_ONLY)")
            attach_jaeb_read_only_if_present(mem_conn, source_db_path)
            mem_conn.execute(
                f"COPY (SELECT * FROM _t1d_export_src.{dq}{escaped_table}{dq}) TO '{escaped_out}' ({format_opts})"
            )
        finally:
            mem_conn.close()


def iter_sanitized_table_batches(
    result: duckdb.DuckDBPyConnection,
    batch_size: int = 5000,
) -> Iterator[list[list[SqlValue]]]:
    """
    Yield batches of spreadsheet-sanitized cell rows from an executed DuckDB query cursor.

    Args:
        result: Executed DuckDB query relation/cursor.
        batch_size: Number of rows to fetch per batch.

    Yields:
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
    target_path: str | Path,
    batch_size: int = 5000,
) -> None:
    """
    Write an executed DuckDB cursor's columns and sanitized rows into an OpenXML (.xlsx) workbook.

    Args:
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
    output_path: str | Path,
    export_format: str = "excel",
) -> None:
    """
    Export a DuckDB table to Excel (.xlsx), Parquet, or CSV format with security sanitization.

    Args:
        db_path: Path to the DuckDB database file.
        table_name: Table identifier to export.
        output_path: Target filesystem path for the exported file.
        export_format: Format string, one of 'excel', 'parquet', or 'csv'.

    Raises:
        InvalidTableNameError: If the table identifier is invalid.
        DatabaseNotFoundError: If the database file does not exist.
        UnsupportedExportFormatError: If `export_format` is not one of `'excel'`, `'parquet'`, or `'csv'`.
        TableNotFoundError: If the table does not exist in the database.

    """
    if not table_name.isidentifier():
        raise InvalidTableNameError(f"Invalid table identifier: {table_name}")

    if not Path(db_path).exists():
        raise DatabaseNotFoundError(f"Database file not found: {db_path}")

    fmt = export_format.lower().strip()
    if fmt not in ("excel", "parquet", "csv"):
        raise UnsupportedExportFormatError(
            f"Unsupported export format: {export_format}. Choose 'excel', 'parquet', or 'csv'."
        )

    target_path = Path(output_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    conn = duckdb.connect(db_path, read_only=True)
    attach_jaeb_read_only_if_present(conn, db_path)
    cursor_proxy = create_read_only_cursor_proxy(conn)
    try:
        tables = [r[0] for r in cursor_proxy.execute("SHOW TABLES").fetchall()]
        if table_name not in tables:
            raise TableNotFoundError(
                f"Table '{table_name}' does not exist in database."
            )

        escaped_tbl = table_name.replace('"', '""')

        if fmt == "parquet":
            _copy_table_to_parquet_file(
                conn=cursor_proxy,
                escaped_table=escaped_tbl,
                target_path=target_path.resolve(),
                source_db_path=db_path,
                use_zstd_compression=False,
            )
        elif fmt == "csv":
            result = cursor_proxy.execute(f'SELECT * FROM "{escaped_tbl}"')
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
            result = cursor_proxy.execute(f'SELECT * FROM "{escaped_tbl}"')
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
        cursor_proxy.close()
        conn.close()


def _build_content_disposition_header(table_name: str, extension: str) -> str:
    """
    Build a sanitized `Content-Disposition` attachment header value for a table export.

    Args:
        table_name: Validated table name identifier.
        extension: File extension without leading dot (e.g. `'xlsx'`, `'parquet'`, `'csv'`, `'json'`).

    Returns:
        str: Formatted `attachment; filename="..."` header string.

    """
    safe_stem = re.sub(r"[^A-Za-z0-9_.-]", "_", table_name)
    safe_ext = re.sub(r"[^A-Za-z0-9]", "", extension)
    return f'attachment; filename="{safe_stem}_export.{safe_ext}"'


def _validate_table_and_db_path(
    table_name: str,
    db_path: str | None,
    db_validator: Callable[[str | None], str],
) -> tuple[str, str]:
    """
    Validate that the target DuckDB file exists and the table name is a valid identifier.

    Args:
        table_name: Target table name to validate.
        db_path: Optional user-supplied database path.
        db_validator: Security validator callable resolving `db_path`.

    Returns:
        Tuple[str, str]: `(target_db, escaped_table)` tuple.

    Raises:
        DatabaseNotFoundError: 404 if the database file does not exist.
        InvalidTableNameError: 400 if `table_name` is invalid.

    """
    target_db = db_validator(db_path)
    if not Path(target_db).exists():
        raise DatabaseNotFoundError(params={"path": target_db})

    if not table_name.isidentifier():
        raise InvalidTableNameError(params={})

    return target_db, escape_sql_identifier(table_name)


class _ClosingReadOnlyTableCursor:
    """Wrapper closing both the dedicated read-only cursor and its parent DuckDB connection on `.close()`."""

    def __init__(
        self,
        conn: duckdb.DuckDBPyConnection,
        cursor_proxy: ReadOnlyDuckDbCursorProxy,
    ) -> None:
        """
        Initialize the closing table cursor wrapper.

        Args:
            conn: Parent DuckDB connection.
            cursor_proxy: Dedicated read-only cursor proxy.

        """
        self._conn = conn
        self._cursor_proxy = cursor_proxy

    def execute(
        self,
        query: str,
        parameters: Sequence[SqlRawInputValue] | None = None,
    ) -> duckdb.DuckDBPyConnection:
        """
        Execute a SQL query on the underlying read-only cursor proxy.

        Args:
            query: SQL statement to execute.
            parameters: Optional positional bind parameters.

        Returns:
            duckdb.DuckDBPyConnection: Executed DuckDB cursor relation.

        """
        return self._cursor_proxy.execute(query, parameters)

    def close(self) -> None:
        """Close the read-only cursor proxy and then close the parent DuckDB connection."""
        try:
            self._cursor_proxy.close()
        finally:
            self._conn.close()


def _open_validated_table_connection(
    target_db: str,
    table_name: str,
) -> _ClosingReadOnlyTableCursor:
    """
    Open a read-only DuckDB connection, attach Jaeb data if present, create a cursor, and verify `table_name` exists.

    Args:
        target_db: Validated filesystem path to the DuckDB database.
        table_name: Validated table identifier.

    Returns:
        _ClosingReadOnlyTableCursor: Open read-only cursor wrapper when the table exists.

    Raises:
        TableNotFoundError: 404 if `table_name` is not present in the database.

    """
    conn = duckdb.connect(target_db, read_only=True)
    attach_jaeb_read_only_if_present(conn, target_db)
    cursor_proxy = create_read_only_cursor_proxy(conn)
    try:
        tables = [row[0] for row in cursor_proxy.execute("SHOW TABLES").fetchall()]
    except (duckdb.Error, OSError, RuntimeError, ValueError):
        cursor_proxy.close()
        conn.close()
        raise
    if table_name not in tables:
        cursor_proxy.close()
        conn.close()
        raise TableNotFoundError(params={})
    return _ClosingReadOnlyTableCursor(conn=conn, cursor_proxy=cursor_proxy)


def _build_table_search_filter(
    column_names: Sequence[str],
    search: str | None,
) -> tuple[str, list[SqlValue]]:
    """
    Build a parameterized DuckDB `WHERE` filter and bind parameter list across `column_names`.

    Args:
        column_names: Ordered table column names.
        search: Optional search term string.

    Returns:
        Tuple[str, List[SqlValue]]: `(where_clause, bind_params)`.

    """
    if search is None or not search.strip() or not column_names:
        return "", []
    search_pattern = f"%{search.strip()}%"
    or_conditions: list[str] = []
    params: list[SqlValue] = []
    for col in column_names:
        escaped_col = escape_sql_identifier(col)
        or_conditions.append(f'CAST("{escaped_col}" AS VARCHAR) ILIKE ?')
        params.append(search_pattern)
    return f"WHERE ({' OR '.join(or_conditions)})", params


def _build_table_order_by_clause(
    column_names: Sequence[str],
    sort_by: str | None,
    sort_order: str,
) -> str:
    """
    Validate `sort_by` against `column_names` and construct a safe `ORDER BY` SQL clause.

    Args:
        column_names: Ordered table column names.
        sort_by: Optional requested sort column name.
        sort_order: Validated lowercase sort order (`'asc'` or `'desc'`).

    Returns:
        str: Formatted `ORDER BY` clause (or empty string if `column_names` is empty).

    Raises:
        InvalidSortColumnError: If `sort_by` is non-empty and not in `column_names`.

    """
    if sort_by is not None and sort_by != "" and sort_by not in column_names:
        raise InvalidSortColumnError(params={"sort_by": sort_by})
    if not column_names:
        return ""
    if sort_by is not None and sort_by != "":
        escaped_sort = escape_sql_identifier(sort_by)
        return f'ORDER BY "{escaped_sort}" {sort_order.upper()}'
    escaped_first = escape_sql_identifier(column_names[0])
    return f'ORDER BY "{escaped_first}" ASC'


def _compute_pagination_bounds(
    total_count: int,
    limit: int,
    offset: int,
) -> tuple[int, int, int]:
    """
    Clamp `limit` and compute `(clamped_limit, page, total_pages)` for table pagination.

    Args:
        total_count: Total matching row count.
        limit: Requested page size (> 0).
        offset: Requested row offset (>= 0).

    Returns:
        Tuple[int, int, int]: `(clamped_limit, page_number, total_pages)`.

    """
    clamped_limit = min(limit, 1000)
    total_pages = max(1, math.ceil(total_count / clamped_limit))
    page = (offset // clamped_limit) + 1
    return clamped_limit, page, total_pages


def fetch_paginated_table_data(
    table_name: str,
    limit: int = 25,
    offset: int = 0,
    sort_by: str | None = None,
    order: str | None = "asc",
    search: str | None = None,
    db_path: str | None = None,
    db_validator: Callable[[str | None], str] = validate_db_path,
) -> TableDataResponse:
    """
    Return paginated rows from a specific table with optional sorting and searching.

    Args:
        table_name: The table name.
        limit: Maximum rows per page.
        offset: Row offset for pagination.
        sort_by: Optional column name to sort by.
        order: Sort order, either 'asc' or 'desc' (default: 'asc').
        search: Optional string to filter across columns.
        db_path: Optional path to the DuckDB database.
        db_validator: Validator callable resolving db_path.

    Returns:
        TableDataResponse: Paginated rows and metadata.

    Raises:
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

    try:
        with connect_duckdb_read_only_cursor(
            db_path=target_db,
            connect_fn=duckdb.connect,
            disable_external_access=True,
        ) as db_cursor:
            tables = [row[0] for row in db_cursor.execute("SHOW TABLES").fetchall()]
            if table_name not in tables:
                raise TableNotFoundError(params={})

            cols_info = db_cursor.execute(f'DESCRIBE "{escaped_table}"').fetchall()
            column_names = [str(c[0]) for c in cols_info]

            order_by_clause = _build_table_order_by_clause(
                column_names, sort_by, sort_order_clean
            )
            where_clause, params = _build_table_search_filter(column_names, search)

            count_query = (
                f'SELECT COUNT(*) FROM "{escaped_table}" {where_clause}'.strip()
            )
            count_result = db_cursor.execute(count_query, params).fetchone()
            total_count = int(count_result[0]) if count_result else 0
            clamped_limit, page, total_pages = _compute_pagination_bounds(
                total_count, limit, offset
            )

            query = f'SELECT * FROM "{escaped_table}" {where_clause} {order_by_clause} LIMIT {clamped_limit} OFFSET {offset}'.strip()
            result = db_cursor.execute(query, params)
            columns = (
                [desc[0] for desc in result.description] if result.description else []
            )
            rows = result.fetchall()
            output = [
                {
                    col: sanitize_sql_value(val)
                    for col, val in zip(columns, row, strict=False)
                }
                for row in rows
            ]
            return TableDataResponse(
                rows=output, total_count=total_count, page=page, total_pages=total_pages
            )
    except T1DError:
        raise
    except (duckdb.Error, OSError, RuntimeError, ValueError) as e:
        err_msg = str(e).lower()
        if "cannot open file" in err_msg or "does not exist" in err_msg:
            raise DatabaseNotFoundError(params={"error": str(e)}) from e
        logger.error(f"Error fetching table data: {e}")
        raise ExportError(params={"error": str(e)}) from e


def fetch_table_profile(
    table_name: str,
    db_path: str | None = None,
    db_validator: Callable[[str | None], str] = validate_db_path,
    profile_fn: Callable[[str, str], TableProfileResult] = profile_table,
) -> TableProfileResult:
    """
    Profile a table to return data quality, completeness, and column statistics.

    Args:
        table_name: The table name to profile.
        db_path: Optional path to the database file.
        db_validator: Validator callable resolving db_path.
        profile_fn: Profiling callable to invoke.

    Returns:
        TableProfileResult: Profiling summary dictionary.

    Raises:
        T1DError: 400 for invalid table, 404 if db or table not found, 500 otherwise.

    """
    target_db, _ = _validate_table_and_db_path(table_name, db_path, db_validator)

    try:
        return profile_fn(target_db, table_name)
    except TableNotFoundError as tnf:
        raise TableNotFoundError(params={"error": str(tnf)}) from tnf
    except (DatabaseNotFoundError, FileNotFoundError) as fnf:
        raise DatabaseNotFoundError(params={"error": str(fnf)}) from fnf
    except (T1DError, duckdb.Error, OSError, RuntimeError, TypeError, ValueError) as e:
        logger.error(f"Error profiling table {table_name}: {e}")
        raise ExportError(params={"error": str(e)}) from e


def _stream_temp_file_and_cleanup(
    file_path: str,
    db_conn: duckdb.DuckDBPyConnection | _ClosingReadOnlyTableCursor | None = None,
    prepare_fn: Callable[[], None] | None = None,
    chunk_size: int = 65536,
) -> Iterator[bytes]:
    """
    Stream a temporary file in binary chunks and always close `db_conn` and unlink `file_path`.

    Args:
        file_path: Path to the temporary file on disk.
        db_conn: Optional open DuckDB connection to close on completion.
        prepare_fn: Optional callback executed before reading `file_path`.
        chunk_size: Byte size of streamed chunks.

    Yields:
        bytes: Binary file chunks.

    """
    try:
        if prepare_fn is not None:
            prepare_fn()
        with open(file_path, "rb") as f:
            while True:
                data = f.read(chunk_size)
                if not data:
                    break
                yield data
    finally:
        if db_conn is not None:
            db_conn.close()
        Path(file_path).unlink(missing_ok=True)


def stream_excel_export(
    table_name: str,
    db_path: str | None = None,
    db_validator: Callable[[str | None], str] = validate_db_path,
) -> StreamingResponse:
    """
    Export table data formatted as a native OpenXML Excel (.xlsx) workbook.

    Args:
        table_name: Table to export.
        db_path: Optional path to database file.
        db_validator: Validator callable resolving db_path.

    Returns:
        StreamingResponse: Streamed XLSX file with spreadsheetml MIME type.

    Raises:
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

        def _write_xlsx() -> None:
            """Materialize the sanitized Excel workbook into `temp_xlsx_file`."""
            write_sanitized_excel_workbook(
                result=result,
                columns=columns,
                sheet_title=table_name,
                target_path=temp_xlsx_file,
            )

        return StreamingResponse(
            _stream_temp_file_and_cleanup(
                file_path=temp_xlsx_file,
                db_conn=conn,
                prepare_fn=_write_xlsx,
            ),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={
                "Content-Disposition": _build_content_disposition_header(
                    table_name, "xlsx"
                )
            },
        )
    except T1DError:
        Path(temp_xlsx_file).unlink(missing_ok=True)
        raise
    except (duckdb.Error, OSError, RuntimeError, ValueError) as e:
        if conn is not None:
            conn.close()
        Path(temp_xlsx_file).unlink(missing_ok=True)
        logger.error(f"Error exporting table to excel: {e}")
        raise ExportError(params={"error": str(e)}) from e


def stream_parquet_export(
    table_name: str,
    db_path: str | None = None,
    db_validator: Callable[[str | None], str] = validate_db_path,
) -> StreamingResponse:
    """
    Export table data in columnar Apache Parquet format.

    Args:
        table_name: Table to export.
        db_path: Optional path to database file.
        db_validator: Validator callable resolving db_path.

    Returns:
        StreamingResponse: Streamed Parquet file.

    Raises:
        T1DError: If table or DB is invalid or export fails.

    """
    target_db, escaped_table = _validate_table_and_db_path(
        table_name, db_path, db_validator
    )

    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
        temp_parquet_file = tmp.name
    try:
        with connect_duckdb_read_only_cursor(
            db_path=target_db,
            connect_fn=duckdb.connect,
            disable_external_access=False,
        ) as pooled_cursor:
            tables = [row[0] for row in pooled_cursor.execute("SHOW TABLES").fetchall()]
            if table_name not in tables:
                raise TableNotFoundError(params={})
            _copy_table_to_parquet_file(
                conn=pooled_cursor,
                escaped_table=escaped_table,
                target_path=temp_parquet_file,
                source_db_path=target_db,
                use_zstd_compression=True,
            )

        return StreamingResponse(
            _stream_temp_file_and_cleanup(
                file_path=temp_parquet_file,
                db_conn=None,
            ),
            media_type="application/vnd.apache.parquet",
            headers={
                "Content-Disposition": _build_content_disposition_header(
                    table_name, "parquet"
                )
            },
        )
    except T1DError:
        Path(temp_parquet_file).unlink(missing_ok=True)
        raise
    except (duckdb.Error, OSError, RuntimeError, ValueError) as e:
        Path(temp_parquet_file).unlink(missing_ok=True)
        logger.error(f"Error exporting table to parquet: {e}")
        raise ExportError(params={"error": str(e)}) from e


def _csv_quote(s: str) -> str:
    """
    Escape double quotes for standard CSV field representation.

    Args:
        s: Raw cell string.

    Returns:
        str: Quoted CSV string.

    """
    return '"' + s.replace('"', '""') + '"'


def _format_csv_cell(val: SqlRawInputValue) -> str:
    """
    Format a single database cell value for safe CSV streaming output.

    Args:
        val: Raw database column value.

    Returns:
        str: Quoted or numeric CSV cell string.

    """
    if val is None:
        return '""'
    if isinstance(val, (int, float, bool)):
        return str(val)
    s_val = str(sanitize_spreadsheet_cell(str(val)))
    return _csv_quote(s_val)


def stream_csv_export(
    table_name: str,
    db_path: str | None = None,
    db_validator: Callable[[str | None], str] = validate_db_path,
) -> StreamingResponse:
    """
    Export table data formatted as a streaming CSV file with spreadsheet formula defense.

    Args:
        table_name: Table to export.
        db_path: Optional path to database file.
        db_validator: Validator callable resolving db_path.

    Returns:
        StreamingResponse: Streamed CSV file with text/csv MIME type.

    Raises:
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
            db_conn: duckdb.DuckDBPyConnection | _ClosingReadOnlyTableCursor,
            initial_batch: Sequence[Sequence[SqlRawInputValue]],
        ) -> Iterator[bytes]:
            """
            Stream CSV header and sanitized row batches lazily without full-table buffering.

            Args:
                db_conn: Open DuckDB connection closed when streaming completes.
                initial_batch: Prefetched first batch of rows.

            Yields:
                bytes: UTF-8 encoded CSV chunks starting with UTF-8 BOM.

            """
            try:
                crlf = "\r\n"
                header_line = ",".join(_csv_quote(col) for col in columns) + crlf
                yield bytes([0xEF, 0xBB, 0xBF]) + header_line.encode("utf-8")

                batch = initial_batch
                while batch:
                    lines = [
                        ",".join(_format_csv_cell(val) for val in row) + crlf
                        for row in batch
                    ]
                    yield "".join(lines).encode("utf-8")
                    batch = result.fetchmany(500)
            finally:
                db_conn.close()

        return StreamingResponse(
            _generate_csv_chunks(conn, first_batch),
            media_type="text/csv",
            headers={
                "Content-Disposition": _build_content_disposition_header(
                    table_name, "csv"
                ),
                "Cache-Control": "no-cache",
            },
        )
    except T1DError:
        raise
    except (duckdb.Error, OSError, RuntimeError, ValueError) as e:
        if conn is not None:
            conn.close()
        logger.error(f"Error exporting table to csv: {e}")
        raise ExportError(params={"error": str(e)}) from e


def stream_json_export(
    table_name: str,
    db_path: str | None = None,
    db_validator: Callable[[str | None], str] = validate_db_path,
) -> StreamingResponse:
    """
    Export table data formatted as a streaming JSON array of row objects.

    Args:
        table_name: Table to export.
        db_path: Optional path to database file.
        db_validator: Validator callable resolving db_path.

    Returns:
        StreamingResponse: Streamed JSON file with application/json MIME type.

    Raises:
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
            db_conn: duckdb.DuckDBPyConnection | _ClosingReadOnlyTableCursor,
            initial_batch: Sequence[Sequence[SqlRawInputValue]],
        ) -> Iterator[bytes]:
            """
            Stream JSON array brackets and serialized row objects in batches.

            Args:
                db_conn: Open DuckDB connection closed when streaming completes.
                initial_batch: Prefetched first batch of rows.

            Yields:
                bytes: UTF-8 encoded JSON array chunks.

            """
            try:
                nl = "\n"
                yield f"[{nl}".encode()
                first_row = True
                batch = initial_batch
                while batch:
                    chunk_parts: list[str] = []
                    for row in batch:
                        row_dict = dict(zip(columns, row, strict=False))
                        prefix = "  " if first_row else f",{nl}  "
                        first_row = False
                        chunk_parts.append(prefix + json.dumps(row_dict, default=str))
                    yield "".join(chunk_parts).encode("utf-8")
                    batch = result.fetchmany(500)
                yield f"{nl}]{nl}".encode()
            finally:
                db_conn.close()

        return StreamingResponse(
            _generate_json_chunks(conn, first_batch),
            media_type="application/json",
            headers={
                "Content-Disposition": _build_content_disposition_header(
                    table_name, "json"
                ),
                "Cache-Control": "no-cache",
            },
        )
    except T1DError:
        raise
    except (duckdb.Error, OSError, RuntimeError, ValueError) as e:
        if conn is not None:
            conn.close()
        logger.error(f"Error exporting table to json: {e}")
        raise ExportError(params={"error": str(e)}) from e
