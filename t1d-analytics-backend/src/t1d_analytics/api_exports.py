"""Table data pagination, profiling, and streaming export services for the T1D Analytics REST API."""

import io
import json
import logging
import math
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional

import duckdb
from fastapi import HTTPException
from fastapi.responses import StreamingResponse

from t1d_analytics.analytics import profile_table
from t1d_analytics.api_schemas import TableDataResponse
from t1d_analytics.db_security import validate_db_path
from t1d_analytics.errors import DatabaseNotFoundError, TableNotFoundError
from t1d_analytics.sql_utils import (
    sanitize_spreadsheet_cell,
    sanitize_sql_value,
)

logger = logging.getLogger(__name__)


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
        HTTPException: On invalid table, pagination, sort parameters, or missing DB/table.

    """
    target_db = db_validator(db_path)
    if not Path(target_db).exists():
        raise HTTPException(
            status_code=404,
            detail={"error_code": "backend.dbNotFound", "params": {"path": target_db}},
        )

    if not table_name.isidentifier():
        raise HTTPException(
            status_code=400, detail={"error_code": "backend.invalidTable", "params": {}}
        )

    if limit <= 0 or offset < 0:
        raise HTTPException(
            status_code=400,
            detail={
                "error_code": "backend.invalidPagination",
                "params": {"limit": limit, "offset": offset},
            },
        )

    sort_order_clean = (order or "asc").strip().lower()
    if sort_order_clean not in ("asc", "desc"):
        raise HTTPException(
            status_code=400,
            detail={
                "error_code": "backend.invalidSortOrder",
                "params": {"order": order},
            },
        )

    conn = None
    try:
        conn = duckdb.connect(target_db, read_only=True)

        tables = [row[0] for row in conn.execute("SHOW TABLES").fetchall()]
        if table_name not in tables:
            raise TableNotFoundError(params={"table": table_name})

        escaped_table = table_name.replace('"', '""')
        cols_info = conn.execute(f'DESCRIBE "{escaped_table}"').fetchall()
        column_names = [c[0] for c in cols_info]

        if sort_by is not None and sort_by != "":
            if sort_by not in column_names:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error_code": "backend.invalidSortColumn",
                        "params": {"sort_by": sort_by},
                    },
                )

        where_clause = ""
        params: List[Any] = []
        if search is not None and search.strip():
            search_pattern = f"%{search.strip()}%"
            or_conditions = []
            for col in column_names:
                escaped_col = col.replace('"', '""')
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
            escaped_sort = sort_by.replace('"', '""')
            order_by_clause = f'ORDER BY "{escaped_sort}" {sort_order_clean.upper()}'
        else:
            escaped_first = column_names[0].replace('"', '""')
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
    except HTTPException:
        raise
    except TableNotFoundError as tnf:
        raise HTTPException(
            status_code=404,
            detail={"error_code": tnf.error_code, "params": {}},
        ) from tnf
    except Exception as e:
        err_msg = str(e).lower()
        if "cannot open file" in err_msg or "does not exist" in err_msg:
            raise HTTPException(
                status_code=404,
                detail={
                    "error_code": "backend.dbNotFound",
                    "params": {"error": str(e)},
                },
            ) from e
        logger.error(f"Error fetching table data: {e}")
        raise HTTPException(
            status_code=500,
            detail={"error_code": "backend.serverError", "params": {"error": str(e)}},
        ) from e
    finally:
        if conn is not None:
            conn.close()


def fetch_table_profile(
    table_name: str,
    db_path: Optional[str] = None,
    db_validator: Callable[[Optional[str]], str] = validate_db_path,
    profile_fn: Callable[[str, str], Dict[str, Any]] = profile_table,
) -> Dict[str, Any]:
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
        Dict[str, Any]: Profiling summary dictionary.

    Raises:
    ------
        HTTPException: 400 for invalid table, 404 if db or table not found, 500 otherwise.

    """
    target_db = db_validator(db_path)
    if not Path(target_db).exists():
        raise HTTPException(
            status_code=404,
            detail={"error_code": "backend.dbNotFound", "params": {"path": target_db}},
        )

    if not table_name.isidentifier():
        raise HTTPException(
            status_code=400,
            detail={"error_code": "backend.invalidTable", "params": {}},
        )

    try:
        return profile_fn(target_db, table_name)
    except (TableNotFoundError, ValueError) as ve:
        raise HTTPException(
            status_code=404,
            detail={
                "error_code": "backend.tableNotFound",
                "params": {"error": str(ve)},
            },
        ) from ve
    except (DatabaseNotFoundError, FileNotFoundError) as fnf:
        raise HTTPException(
            status_code=404,
            detail={"error_code": "backend.dbNotFound", "params": {"error": str(fnf)}},
        ) from fnf
    except Exception as e:
        logger.error(f"Error profiling table {table_name}: {e}")
        raise HTTPException(
            status_code=500,
            detail={"error_code": "backend.serverError", "params": {"error": str(e)}},
        ) from e


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
        HTTPException: If table or DB is invalid.

    """
    target_db = db_validator(db_path)
    if not Path(target_db).exists():
        raise HTTPException(
            status_code=404,
            detail={"error_code": "backend.dbNotFound", "params": {"path": target_db}},
        )

    if not table_name.isidentifier():
        raise HTTPException(
            status_code=400,
            detail={"error_code": "backend.invalidTable", "params": {}},
        )

    conn = None
    try:
        conn = duckdb.connect(target_db, read_only=True)
        tables = [row[0] for row in conn.execute("SHOW TABLES").fetchall()]
        if table_name not in tables:
            conn.close()
            raise HTTPException(
                status_code=404,
                detail={"error_code": "backend.tableNotFound", "params": {}},
            )

        escaped_table = table_name.replace('"', '""')
        result = conn.execute(f'SELECT * FROM "{escaped_table}"')
        columns = [desc[0] for desc in result.description] if result.description else []

        def _generate_excel_chunks() -> Iterator[bytes]:
            """
            Stream table data formatted as native OpenXML workbook in chunks.

            Yields
            ------
                bytes: Binary XLSX chunk bytes.

            """
            try:
                from openpyxl import Workbook

                wb = Workbook(write_only=True)
                ws = wb.create_sheet(title=table_name[:31])
                ws.append(columns)

                while True:
                    batch = result.fetchmany(5000)
                    if not batch:
                        break
                    for r in batch:
                        ws.append([sanitize_spreadsheet_cell(val) for val in r])

                output = io.BytesIO()
                wb.save(output)
                output.seek(0)

                chunk_size = 65536
                while True:
                    data = output.read(chunk_size)
                    if not data:
                        break
                    yield data
            finally:
                conn.close()

        return StreamingResponse(
            _generate_excel_chunks(),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={
                "Content-Disposition": f'attachment; filename="{table_name}_export.xlsx"'
            },
        )
    except HTTPException:
        raise
    except Exception as e:
        if conn is not None:
            conn.close()
        logger.error(f"Error exporting table to excel: {e}")
        raise HTTPException(
            status_code=500,
            detail={"error_code": "backend.serverError", "params": {"error": str(e)}},
        ) from e


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
        HTTPException: If table or DB is invalid.

    """
    target_db = db_validator(db_path)
    if not Path(target_db).exists():
        raise HTTPException(
            status_code=404,
            detail={"error_code": "backend.dbNotFound", "params": {"path": target_db}},
        )

    if not table_name.isidentifier():
        raise HTTPException(
            status_code=400,
            detail={"error_code": "backend.invalidTable", "params": {}},
        )

    conn = None
    temp_parquet_file = None
    try:
        conn = duckdb.connect(target_db, read_only=True)
        tables = [row[0] for row in conn.execute("SHOW TABLES").fetchall()]
        if table_name not in tables:
            conn.close()
            raise HTTPException(
                status_code=404,
                detail={"error_code": "backend.tableNotFound", "params": {}},
            )

        escaped_table = table_name.replace('"', '""')
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
    except HTTPException:
        raise
    except Exception as e:
        if conn is not None:
            conn.close()
        if temp_parquet_file:
            Path(temp_parquet_file).unlink(missing_ok=True)
        logger.error(f"Error exporting table to parquet: {e}")
        raise HTTPException(
            status_code=500,
            detail={"error_code": "backend.serverError", "params": {"error": str(e)}},
        ) from e


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
        HTTPException: If table or DB is invalid.

    """
    target_db = db_validator(db_path)
    if not Path(target_db).exists():
        raise HTTPException(
            status_code=404,
            detail={"error_code": "backend.dbNotFound", "params": {"path": target_db}},
        )

    if not table_name.isidentifier():
        raise HTTPException(
            status_code=400,
            detail={"error_code": "backend.invalidTable", "params": {}},
        )

    conn = None
    try:
        conn = duckdb.connect(target_db, read_only=True)
        try:
            tables = [row[0] for row in conn.execute("SHOW TABLES").fetchall()]
            if table_name not in tables:
                raise HTTPException(
                    status_code=404,
                    detail={"error_code": "backend.tableNotFound", "params": {}},
                )

            escaped_table = table_name.replace('"', '""')
            result = conn.execute(f'SELECT * FROM "{escaped_table}"')
            columns = (
                [desc[0] for desc in result.description] if result.description else []
            )

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

            output = io.BytesIO()
            output.write(b"\xef\xbb\xbf")
            header_line = ",".join(_csv_quote(col) for col in columns) + "\r\n"
            output.write(header_line.encode("utf-8"))

            while True:
                batch = result.fetchmany(5000)
                if not batch:
                    break
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
                    lines.append(",".join(formatted_cells) + "\r\n")
                output.write("".join(lines).encode("utf-8"))
        finally:
            conn.close()

        output.seek(0)
        return StreamingResponse(
            output,
            media_type="text/csv",
            headers={
                "Content-Disposition": f'attachment; filename="{table_name}_export.csv"',
                "Cache-Control": "no-cache",
            },
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error exporting table to csv: {e}")
        raise HTTPException(
            status_code=500,
            detail={"error_code": "backend.serverError", "params": {"error": str(e)}},
        ) from e


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
        HTTPException: If table or DB is invalid.

    """
    target_db = db_validator(db_path)
    if not Path(target_db).exists():
        raise HTTPException(
            status_code=404,
            detail={"error_code": "backend.dbNotFound", "params": {"path": target_db}},
        )

    if not table_name.isidentifier():
        raise HTTPException(
            status_code=400,
            detail={"error_code": "backend.invalidTable", "params": {}},
        )

    conn = None
    try:
        conn = duckdb.connect(target_db, read_only=True)
        try:
            tables = [row[0] for row in conn.execute("SHOW TABLES").fetchall()]
            if table_name not in tables:
                raise HTTPException(
                    status_code=404,
                    detail={"error_code": "backend.tableNotFound", "params": {}},
                )

            escaped_table = table_name.replace('"', '""')
            result = conn.execute(f'SELECT * FROM "{escaped_table}"')
            columns = (
                [desc[0] for desc in result.description] if result.description else []
            )

            output = io.BytesIO()
            output.write("[\n".encode("utf-8"))
            first_row = True
            while True:
                batch = result.fetchmany(5000)
                if not batch:
                    break
                chunk_parts: List[str] = []
                for row in batch:
                    row_dict = dict(zip(columns, row))
                    prefix = "  " if first_row else ",\n  "
                    first_row = False
                    chunk_parts.append(prefix + json.dumps(row_dict, default=str))
                output.write("".join(chunk_parts).encode("utf-8"))
            output.write("\n]\n".encode("utf-8"))
        finally:
            conn.close()

        output.seek(0)
        return StreamingResponse(
            output,
            media_type="application/json",
            headers={
                "Content-Disposition": f'attachment; filename="{table_name}_export.json"',
                "Cache-Control": "no-cache",
            },
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error exporting table to json: {e}")
        raise HTTPException(
            status_code=500,
            detail={"error_code": "backend.serverError", "params": {"error": str(e)}},
        ) from e
