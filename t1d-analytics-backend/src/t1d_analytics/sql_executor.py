"""Read-only DuckDB query execution service with timeout interruption and numeric coercion."""

import logging
import threading
from collections.abc import Callable
from typing import Protocol

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
    validate_read_only_sql,
)

logger = logging.getLogger(__name__)


class QueryTimerProtocol(Protocol):
    """Structural protocol for query execution timeout timers."""

    daemon: bool

    def start(self) -> None:
        """Start the timeout timer."""
        ...

    def cancel(self) -> None:
        """Cancel the timeout timer."""
        ...

    def join(self, timeout: float | None = None) -> None:
        """Wait for timer thread completion."""
        ...


def _noop_interrupt() -> None:
    """No-op interrupt placeholder when no DuckDB query is actively executing."""


def execute_duckdb_query(
    db_path: str,
    query: str,
    max_rows: int = 10000,
    timeout_seconds: float = 30.0,
    connect_fn: Callable[..., duckdb.DuckDBPyConnection] = duckdb.connect,
    timer_factory: Callable[
        [float, Callable[[], None]], QueryTimerProtocol
    ] = threading.Timer,
) -> list[dict[str, SqlValue]]:
    """
    Execute a read-only SQL query against connection-pooled DuckDB via an isolated cursor with timeout protection and numeric aggregate coercion.

    Args:
        db_path: Path to the DuckDB database file.
        query: SQL query to execute.
        max_rows: Maximum rows to return to prevent memory exhaustion.
        timeout_seconds: Execution timeout in seconds.
        connect_fn: DuckDB connection factory callable.
        timer_factory: Injectable timer constructor `(timeout_seconds, callback) -> QueryTimerProtocol`.

    Returns:
        List[Dict[str, SqlValue]]: List of sanitized row dictionaries.

    Raises:
        EmptyQueryError: If the SQL query is empty or comment-only.
        MultipleStatementsError: If multiple SQL statements are supplied.
        ReadOnlyViolationError: If the SQL statement is not read-only.
        SqlExecutionError: If query execution fails or times out.
        DatabaseNotFoundError: If database file cannot be opened.

    """
    validated_query = validate_read_only_sql(query)
    logger.info(
        "Executing SQL query against database at %s:\n%s", db_path, validated_query
    )
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
            coerced_query = coerce_numeric_aggregates_in_sql(validated_query)
            timer = timer_factory(timeout_seconds, safe_interrupt)
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
            {
                col: sanitize_sql_value(val)
                for col, val in zip(columns, row, strict=False)
            }
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
