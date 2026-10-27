"""Read-only DuckDB query execution service with timeout interruption and numeric coercion."""

import logging
import threading
from typing import Callable, Dict, List

import duckdb

from t1d_analytics.errors import DatabaseNotFoundError, SqlExecutionError
from t1d_analytics.sql_utils import (
    SqlValue,
    coerce_numeric_aggregates_in_sql,
    sanitize_sql_value,
)

logger = logging.getLogger(__name__)


def execute_duckdb_query(
    db_path: str,
    query: str,
    max_rows: int = 10000,
    timeout_seconds: float = 30.0,
    connect_fn: Callable[..., duckdb.DuckDBPyConnection] = duckdb.connect,
) -> List[Dict[str, SqlValue]]:
    """
    Execute a read-only SQL query against DuckDB with timeout protection and numeric aggregate coercion.

    Args:
    ----
        db_path: Path to the DuckDB database file.
        query: SQL query to execute.
        max_rows: Maximum rows to return to prevent memory exhaustion.
        timeout_seconds: Execution timeout in seconds.
        connect_fn: DuckDB connection factory callable.

    Returns:
    -------
        List[Dict[str, SqlValue]]: List of sanitized row dictionaries.

    Raises:
    ------
        SqlExecutionError: If query execution fails or times out.
        DatabaseNotFoundError: If database file cannot be opened.

    """
    nl = chr(10)
    logger.info(f"Executing SQL query against database at {db_path}:{nl}{query}")
    conn = None
    is_closed = False
    lock = threading.Lock()

    def safe_interrupt() -> None:
        """Interrupt the DuckDB connection safely from another thread."""
        with lock:
            if not is_closed and conn is not None:
                conn.interrupt()

    try:
        try:
            conn = connect_fn(db_path, read_only=True)
        except duckdb.IOException as io_err:
            raise DatabaseNotFoundError(params={"error": str(io_err)}) from io_err
        conn.execute("SET enable_external_access = false")
        coerced_query = coerce_numeric_aggregates_in_sql(query)
        timer = threading.Timer(timeout_seconds, safe_interrupt)
        timer.daemon = True
        timer.start()
        try:
            result = conn.execute(coerced_query)
            columns = (
                [desc[0] for desc in result.description] if result.description else []
            )
            rows = result.fetchmany(max_rows)
        finally:
            timer.cancel()
            timer.join(timeout=1.0)

        output = [
            {col: sanitize_sql_value(val) for col, val in zip(columns, row)}
            for row in rows
        ]
        logger.info(f"SQL query executed successfully. Returned {len(output)} rows.")
        return output
    except duckdb.InterruptException as ie:
        logger.error(f"SQL execution timed out: {ie}")
        raise SqlExecutionError(
            error_code="backend.queryTimeout",
            params={"error": "Query execution timed out."},
        ) from ie
    except DatabaseNotFoundError:
        raise
    except Exception as e:
        logger.error(f"SQL execution error: {e}")
        raise SqlExecutionError(params={"error": str(e)}) from e
    finally:
        with lock:
            is_closed = True
            if conn is not None:
                conn.close()
