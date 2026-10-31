"""Table preview, profiling, schema inspection, database discovery, and streaming export endpoints."""

import logging
from typing import Optional

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from t1d_analytics.analytics import TableProfileResult
from t1d_analytics.api_exports import (
    fetch_paginated_table_data,
    fetch_table_profile,
    stream_csv_export,
    stream_excel_export,
    stream_json_export,
    stream_parquet_export,
)
from t1d_analytics.api_schemas import (
    ColumnInfo,
    DatabaseListResponse,
    SchemaResponse,
    TableDataResponse,
    TableInfo,
)
from t1d_analytics.diagnostics import check_database_health
from t1d_analytics.sql_utils import extract_table_columns_map

logger = logging.getLogger(__name__)

tables_router = APIRouter()


@tables_router.get("/api/table/{table_name}", response_model=TableDataResponse)
def get_table_data(
    table_name: str,
    limit: int = 25,
    offset: int = 0,
    sort_by: Optional[str] = None,
    order: Optional[str] = "asc",
    search: Optional[str] = None,
    db_path: Optional[str] = None,
) -> TableDataResponse:
    """
    Return paginated rows from a specific table with optional sorting and searching.

    Args:
    ----
        table_name: The table name.
        limit: The limit.
        offset: The offset.
        sort_by: Optional column name to sort by.
        order: Sort order, either 'asc' or 'desc' (default: 'asc').
        search: Optional string to filter across columns.
        db_path: Optional path to the DuckDB database.

    Returns:
    -------
        TableDataResponse: The data with pagination metadata.

    """
    import t1d_analytics.api as api_mod

    return fetch_paginated_table_data(
        table_name=table_name,
        limit=limit,
        offset=offset,
        sort_by=sort_by,
        order=order,
        search=search,
        db_path=db_path,
        db_validator=api_mod.validate_db_path,
    )


@tables_router.get("/api/table/{table_name}/profile")
def get_table_profile(
    table_name: str, db_path: Optional[str] = None
) -> TableProfileResult:
    """
    Profile a table to return data quality, completeness, and column statistics.

    Args:
    ----
        table_name: The table name to profile.
        db_path: Optional path to the database file.

    Returns:
    -------
        TableProfileResult: Profiling summary dictionary.

    """
    import t1d_analytics.api as api_mod

    return fetch_table_profile(
        table_name=table_name,
        db_path=db_path,
        db_validator=api_mod.validate_db_path,
        profile_fn=api_mod.profile_table,
    )


@tables_router.get("/api/export/excel")
def export_excel(table_name: str, db_path: Optional[str] = None) -> StreamingResponse:
    """
    Export table data formatted as a native OpenXML Excel (.xlsx) workbook.

    Args:
    ----
        table_name: Table to export.
        db_path: Optional path to database file.

    Returns:
    -------
        StreamingResponse: Streamed XLSX file with spreadsheetml MIME type.

    """
    import t1d_analytics.api as api_mod

    return stream_excel_export(
        table_name=table_name,
        db_path=db_path,
        db_validator=api_mod.validate_db_path,
    )


@tables_router.get("/api/export/parquet")
def export_parquet(
    table_name: str,
    db_path: Optional[str] = None,
) -> StreamingResponse:
    """
    Export table data in columnar Apache Parquet format.

    Args:
    ----
        table_name: Table to export.
        db_path: Optional path to database file.

    Returns:
    -------
        StreamingResponse: Streamed Parquet file.

    """
    import t1d_analytics.api as api_mod

    return stream_parquet_export(
        table_name=table_name,
        db_path=db_path,
        db_validator=api_mod.validate_db_path,
    )


@tables_router.get("/api/export/csv")
def export_csv(
    table_name: str,
    db_path: Optional[str] = None,
) -> StreamingResponse:
    """
    Export table data formatted as a streaming CSV file with spreadsheet formula defense.

    Args:
    ----
        table_name: Table to export.
        db_path: Optional path to database file.

    Returns:
    -------
        StreamingResponse: Streamed CSV file with text/csv MIME type.

    """
    import t1d_analytics.api as api_mod

    return stream_csv_export(
        table_name=table_name,
        db_path=db_path,
        db_validator=api_mod.validate_db_path,
    )


@tables_router.get("/api/export/json")
def export_json(
    table_name: str,
    db_path: Optional[str] = None,
) -> StreamingResponse:
    """
    Export table data formatted as a streaming JSON array of row objects.

    Args:
    ----
        table_name: Table to export.
        db_path: Optional path to database file.

    Returns:
    -------
        StreamingResponse: Streamed JSON file with application/json MIME type.

    """
    import t1d_analytics.api as api_mod

    return stream_json_export(
        table_name=table_name,
        db_path=db_path,
        db_validator=api_mod.validate_db_path,
    )


def _build_schema_response(target_db: str, lang: str = "en") -> SchemaResponse:
    """
    Inspect a DuckDB database and build a SchemaResponse with table columns and health status.

    Args:
    ----
        target_db: Validated path to the DuckDB database file.
        lang: Localization language code for database health messages.

    Returns:
    -------
        SchemaResponse: Structured table and column metadata with database health dictionary.

    """
    import t1d_analytics.api as api_mod

    db_health = check_database_health(target_db, lang=lang)

    conn = None
    try:
        conn = api_mod.duckdb.connect(target_db, read_only=True)
        cols_map = extract_table_columns_map(
            conn, exclude_internal=False, exclude_training_tables=False
        )
        schema_data = [
            TableInfo(
                name=table,
                columns=[
                    ColumnInfo(name=c_name, type=c_type) for c_name, c_type in columns
                ],
            )
            for table, columns in cols_map.items()
        ]

        return SchemaResponse(tables=schema_data, database=db_health.to_dict())
    except Exception as e:
        logger.error(f"Error fetching schema: {e}")
        return SchemaResponse(tables=[], database=db_health.to_dict())
    finally:
        if conn is not None:
            conn.close()


@tables_router.get("/api/schema", response_model=SchemaResponse)
def get_schema(request: Request, db_path: Optional[str] = None) -> SchemaResponse:
    """
    Return structured schema for the frontend Schema Explorer.

    Args:
    ----
        request: FastAPI request.
        db_path: Optional target DuckDB database path.

    Returns:
    -------
        SchemaResponse: The structured schema response including database diagnostics.

    """
    import t1d_analytics.api as api_mod

    target_db = api_mod.validate_db_path(db_path)
    lang = api_mod.get_request_locale(request)
    return _build_schema_response(target_db=target_db, lang=lang)


@tables_router.get("/api/databases", response_model=DatabaseListResponse)
def list_databases() -> DatabaseListResponse:
    """
    Discover and list available DuckDB database files in the workspace.

    Returns
    -------
        DatabaseListResponse: Found databases and current default.

    """
    import t1d_analytics.api as api_mod

    return api_mod.discover_workspace_databases()
