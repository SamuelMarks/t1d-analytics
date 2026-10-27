"""Analytics module for querying downloaded T1D datasets."""

from pathlib import Path
from typing import (
    List,
    Optional,
)

import duckdb
from typing_extensions import TypedDict

from t1d_analytics.api_schemas import ColumnInfo, TableSchemaModel
from t1d_analytics.diagnostics import DatabaseStatusCode, check_database_health
from t1d_analytics.errors import (
    InvalidTableNameError,
    ReadOnlyViolationError,
    TableNotFoundError,
)
from t1d_analytics.i18n import get_translator
from t1d_analytics.ingestion import (
    _NUMERIC_CLINICAL_CONCEPTS as _NUMERIC_CLINICAL_CONCEPTS,
)
from t1d_analytics.ingestion import (
    IngestionManifestRecord as IngestionManifestRecord,
)
from t1d_analytics.ingestion import (
    _detect_csv_encoding_and_delimiter as _detect_csv_encoding_and_delimiter,
)
from t1d_analytics.ingestion import (
    _ensure_manifest_table as _ensure_manifest_table,
)
from t1d_analytics.ingestion import (
    _read_clinical_dataframe as _read_clinical_dataframe,
)
from t1d_analytics.ingestion import (
    compute_file_hash as compute_file_hash,
)
from t1d_analytics.ingestion import (
    extract_zips as extract_zips,
)
from t1d_analytics.ingestion import (
    get_clinical_variable_mappings as get_clinical_variable_mappings,
)
from t1d_analytics.ingestion import (
    get_ingestion_manifest as get_ingestion_manifest,
)
from t1d_analytics.ingestion import (
    load_data_to_duckdb as load_data_to_duckdb,
)
from t1d_analytics.llm_client import (
    build_clinical_sql_prompt,
    complete_llm_prompt,
    resolve_provider_and_model,
)
from t1d_analytics.sql_utils import (
    SqlValue,
    coerce_numeric_aggregates_in_sql,
    escape_sql_identifier,
    is_read_only_sql_query,
)
from t1d_analytics.sql_utils import (
    extract_sql_from_response as extract_sql_from_response,
)
from t1d_analytics.sql_utils import (
    is_sql_query as is_sql_query,
)


class _ColumnMetaRecord(TypedDict):
    """Internal metadata for a profiled table column."""

    column_name: str
    data_type: str
    is_numeric: bool


class ColumnProfileRecord(TypedDict, total=False):
    """Statistics and completeness metrics for a single database column."""

    column_name: str
    data_type: str
    null_count: int
    null_percentage: float
    distinct_count: int
    min: SqlValue
    max: SqlValue
    mean: Optional[float]


class TableProfileResult(TypedDict):
    """Complete profiling summary for a database table."""

    table_name: str
    total_rows: int
    column_count: int
    columns: List[ColumnProfileRecord]


def get_database_schema_models(
    conn: duckdb.DuckDBPyConnection,
) -> List[TableSchemaModel]:
    """
    Extract structured table schema models from the DuckDB database.

    Args:
    ----
        conn: The DuckDB connection.

    Returns:
    -------
        List[TableSchemaModel]: Ordered list of structured table schema models.

    """
    tables = [
        t
        for t in conn.execute("SHOW TABLES").fetchall()
        if not t[0].startswith("_t1d_")
    ]
    models: List[TableSchemaModel] = []
    for (table_name,) in tables:
        escaped_table = escape_sql_identifier(str(table_name))
        columns = conn.execute(f'DESCRIBE "{escaped_table}"').fetchall()
        col_infos = [ColumnInfo(name=str(c[0]), type=str(c[1])) for c in columns]

        try:
            sample = conn.execute(f'SELECT * FROM "{escaped_table}" LIMIT 1').fetchone()
            sample_desc = (
                f"Sample row: {sample}" if sample else "Sample row: (empty table)"
            )
        except Exception:
            sample_desc = "Sample row: (could not fetch)"

        models.append(
            TableSchemaModel(
                table_name=str(table_name),
                columns=col_infos,
                sample_row=sample_desc,
            )
        )
    return models


def get_database_schema(conn: duckdb.DuckDBPyConnection) -> str:
    """
    Extract the database schema as a text description, including sample data.

    Args:
    ----
        conn: The DuckDB connection.

    Returns:
    -------
        The database schema string.

    """
    return "\n\n".join(m.to_prompt_block() for m in get_database_schema_models(conn))


def handle_natural_language(
    conn: duckdb.DuckDBPyConnection,
    query: str,
    model: str = "gemma4",
    provider: Optional[str] = None,
) -> None:
    """
    Translate natural language to SQL using an LLM via any-llm and execute it.

    Args:
    ----
        conn: DB connection.
        query: The natural language query.
        model: Model name for translation (or 'provider/model').
        provider: Optional provider name ('ollama', 'openai', 'anthropic', 'google').

    """
    _ = get_translator()
    try:
        from any_llm import AnyLLM

        del AnyLLM
    except ImportError:
        print(_("Error: any-llm-sdk[ollama] is not installed. Please install it."))
        return

    prov, actual_model = resolve_provider_and_model(model, provider)

    from t1d_analytics.training_data import validate_provider_readiness

    try:
        validate_provider_readiness(prov)
    except RuntimeError as err:
        print(_("Failed to generate or execute query: {}", err))
        return

    print(_("Thinking..."))
    schema = get_database_schema(conn)
    prompt = build_clinical_sql_prompt(
        schema=schema,
        nl_query=query,
        require_markdown_fence=False,
    )
    try:
        full_response = complete_llm_prompt(
            prompt=prompt,
            model_name=actual_model,
            provider=prov,
        )
        sql_query = extract_sql_from_response(full_response)

        print(_("Generated SQL: \n{}\n", sql_query))
        if not is_read_only_sql_query(sql_query):
            raise ReadOnlyViolationError(
                f"Generated query is not read-only: {sql_query}"
            )
        print(_("Executing...\n"))

        result = conn.sql(coerce_numeric_aggregates_in_sql(sql_query))
        if result:
            result.show()

    except Exception as e:
        print(_("Failed to generate or execute query: {}", e))


def run_query_repl(
    db_path: str, model: str = "gemma4", provider: Optional[str] = None
) -> None:
    """
    Run the interactive query interface.

    Args:
    ----
        db_path: Path to DB.
        model: LLM model name for query translation (or 'provider/model').
        provider: Optional LLM provider name ('ollama', 'openai', 'anthropic', 'google').

    """
    _ = get_translator()
    if not Path(db_path).exists():
        print(_("Database {} does not exist.", db_path))
        print(_("Please run the 'load' command first to populate the database."))
        return

    db_health = check_database_health(db_path)
    if db_health.status_code == DatabaseStatusCode.EMPTY_DB:
        print(_("Warning: Database {} contains 0 tables.", db_path))
        print(_("Please run the 'load' command first to populate the database."))
    elif db_health.status_code == DatabaseStatusCode.MISSING_INITIAL_DATA:
        print(
            _(
                "Warning: Database {} lacks standard initial clinical trial datasets.",
                db_path,
            )
        )

    print(_("Connecting to DuckDB at {}...", db_path))
    conn = duckdb.connect(db_path, read_only=True)

    print("\n" + "=" * 50)
    print(_("Welcome to T1D Analytics Interface!"))
    print(_("You can enter:"))
    print(
        _("  - Standard SQL queries (starting with SELECT, WITH, SHOW, DESCRIBE, etc.)")
    )
    print(_("  - Natural language queries (will be translated to SQL via LLM)"))
    print(_("  - 'exit' or 'quit' to close."))
    print("=" * 50 + "\n")

    while True:
        try:
            user_input = input("query> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n" + _("Exiting."))
            break

        if not user_input:
            continue

        lower_input = user_input.lower()
        if lower_input in ("exit", "quit"):
            print(_("Exiting."))
            break

        if is_sql_query(user_input, conn=conn):
            try:
                result = conn.sql(coerce_numeric_aggregates_in_sql(user_input))
                # Show results nicely
                if result:
                    result.show()
            except Exception as e:
                print(_("SQL Error: {}", e))
        else:
            handle_natural_language(conn, user_input, model=model, provider=provider)

    conn.close()


def profile_table(db_path: str, table_name: str) -> TableProfileResult:
    """
    Profile a table to calculate data quality, completeness, and column statistics.

    Args:
    ----
        db_path: Path to DuckDB file.
        table_name: Table name to profile.

    Returns:
    -------
        TableProfileResult: Profiling summary containing total_rows, columns stats, null percentages.

    Raises:
    ------
        ValueError: If table name is invalid or table does not exist.
        FileNotFoundError: If database file does not exist.

    """
    if not table_name.isidentifier():
        raise InvalidTableNameError(f"Invalid table identifier: {table_name}")

    if not Path(db_path).exists():
        raise FileNotFoundError(f"Database file not found: {db_path}")

    conn = duckdb.connect(db_path, read_only=True)
    try:
        tables = [r[0] for r in conn.execute("SHOW TABLES").fetchall()]
        if table_name not in tables:
            raise TableNotFoundError(
                f"Table '{table_name}' does not exist in database."
            )

        cols = conn.execute(f'DESCRIBE "{table_name}"').fetchall()
        column_profiles: List[ColumnProfileRecord] = []

        if not cols:
            count_row = conn.execute(f'SELECT COUNT(*) FROM "{table_name}"').fetchone()
            empty_total_rows = int(count_row[0]) if count_row else 0
            return {
                "table_name": table_name,
                "total_rows": empty_total_rows,
                "column_count": 0,
                "columns": [],
            }

        select_exprs: List[str] = ['COUNT(*) AS "_total_rows"']
        col_meta: List[_ColumnMetaRecord] = []

        for idx, col_info in enumerate(cols):
            col_name = str(col_info[0])
            col_type = str(col_info[1])
            escaped_col = col_name.replace('"', '""')

            select_exprs.append(
                f'SUM(CASE WHEN "{escaped_col}" IS NULL THEN 1 ELSE 0 END) AS "null_{idx}"'
            )
            select_exprs.append(f'COUNT(DISTINCT "{escaped_col}") AS "dist_{idx}"')

            type_upper = col_type.upper()
            is_numeric = any(
                t in type_upper
                for t in (
                    "INT",
                    "FLOAT",
                    "DOUBLE",
                    "DECIMAL",
                    "NUMERIC",
                    "HUGEINT",
                )
            )
            if is_numeric:
                select_exprs.append(f'MIN("{escaped_col}") AS "min_{idx}"')
                select_exprs.append(f'MAX("{escaped_col}") AS "max_{idx}"')
                select_exprs.append(f'AVG("{escaped_col}") AS "avg_{idx}"')

            col_meta.append(
                {
                    "column_name": col_name,
                    "data_type": col_type,
                    "is_numeric": is_numeric,
                }
            )

        escaped_tbl = table_name.replace('"', '""')
        agg_sql = f'SELECT {", ".join(select_exprs)} FROM "{escaped_tbl}"'
        agg_res = conn.execute(agg_sql).fetchone()

        total_rows: int = int(agg_res[0]) if (agg_res and agg_res[0] is not None) else 0

        cursor = 1
        for meta in col_meta:
            null_val = agg_res[cursor] if agg_res else 0
            null_count: int = int(null_val) if null_val is not None else 0
            cursor += 1

            dist_val = agg_res[cursor] if agg_res else 0
            distinct_count: int = int(dist_val) if dist_val is not None else 0
            cursor += 1

            null_pct = (null_count / total_rows * 100.0) if total_rows > 0 else 0.0

            col_stat: ColumnProfileRecord = {
                "column_name": meta["column_name"],
                "data_type": meta["data_type"],
                "null_count": null_count,
                "null_percentage": round(null_pct, 2),
                "distinct_count": distinct_count,
            }

            if meta["is_numeric"]:
                min_val = agg_res[cursor] if agg_res else None
                cursor += 1
                max_val = agg_res[cursor] if agg_res else None
                cursor += 1
                avg_val = agg_res[cursor] if agg_res else None
                cursor += 1

                if min_val is not None and max_val is not None:
                    col_stat["min"] = (
                        min_val
                        if isinstance(min_val, (int, float, str, bool))
                        else float(min_val)
                    )
                    col_stat["max"] = (
                        max_val
                        if isinstance(max_val, (int, float, str, bool))
                        else float(max_val)
                    )
                    col_stat["mean"] = (
                        round(float(avg_val), 2) if avg_val is not None else None
                    )
                else:
                    col_stat["min"] = None
                    col_stat["max"] = None
                    col_stat["mean"] = None

            column_profiles.append(col_stat)

        return {
            "table_name": table_name,
            "total_rows": total_rows,
            "column_count": len(column_profiles),
            "columns": column_profiles,
        }
    finally:
        conn.close()


from t1d_analytics.api_exports import (  # noqa: E402
    export_table as export_table,
)
from t1d_analytics.api_exports import (  # noqa: E402
    iter_sanitized_table_batches as iter_sanitized_table_batches,
)
from t1d_analytics.api_exports import (  # noqa: E402
    write_sanitized_excel_workbook as write_sanitized_excel_workbook,
)
