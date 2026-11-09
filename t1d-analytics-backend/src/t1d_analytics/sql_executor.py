"""Read-only DuckDB query execution service with timeout interruption and numeric coercion."""

import logging
import threading
from typing import Callable, Dict, List

import duckdb

from t1d_analytics.errors import (
    DatabaseNotFoundError,
    SqlExecutionError,
    SqlTimeoutError,
)
from t1d_analytics.sql_utils import (
    SqlValue,
    coerce_numeric_aggregates_in_sql,
    connect_duckdb_read_only_cursor,
    sanitize_sql_value,
)

logger = logging.getLogger(__name__)


def _noop_interrupt() -> None:
    """No-op interrupt placeholder when no DuckDB query is actively executing."""


def execute_duckdb_query(
    db_path: str,
    query: str,
    max_rows: int = 10000,
    timeout_seconds: float = 30.0,
    connect_fn: Callable[..., duckdb.DuckDBPyConnection] = duckdb.connect,
) -> List[Dict[str, SqlValue]]:
    """
    Execute a read-only SQL query against connection-pooled DuckDB via an isolated cursor with timeout protection and numeric aggregate coercion.

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
    interrupt_fn: Callable[[], None] = _noop_interrupt
    lock = threading.Lock()

    def safe_interrupt() -> None:
        """Interrupt the active DuckDB connection safely from another thread."""
        with lock:
            interrupt_fn()

    try:
        with connect_duckdb_read_only_cursor(
            db_path=db_path,
            connect_fn=connect_fn,
            disable_external_access=True,
        ) as cursor:
            with lock:
                interrupt_fn = cursor.interrupt
            coerced_query = coerce_numeric_aggregates_in_sql(query)
            timer = threading.Timer(timeout_seconds, safe_interrupt)
            timer.daemon = True
            timer.start()
            try:
                result = cursor.execute(coerced_query)
                columns = (
                    [desc[0] for desc in result.description]
                    if result.description
                    else []
                )
                rows = result.fetchmany(max_rows)
            finally:
                timer.cancel()
                timer.join(timeout=1.0)
                with lock:
                    interrupt_fn = _noop_interrupt

        output = [
            {col: sanitize_sql_value(val) for col, val in zip(columns, row)}
            for row in rows
        ]
        logger.info(f"SQL query executed successfully. Returned {len(output)} rows.")
        return output
    except duckdb.IOException as io_err:
        raise DatabaseNotFoundError(params={"error": str(io_err)}) from io_err
    except duckdb.InterruptException as ie:
        logger.error(f"SQL execution timed out: {ie}")
        raise SqlTimeoutError(
            params={"error": "Query execution timed out."},
        ) from ie
    except (duckdb.Error, OSError, RuntimeError, ValueError, TypeError) as e:
        logger.error(f"SQL execution error: {e}")
        raise SqlExecutionError(params={"error": str(e)}) from e
