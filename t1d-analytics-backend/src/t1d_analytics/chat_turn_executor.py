"""Synchronous SQL and natural-language chat turn and benchmark evaluation orchestration."""

import logging
from collections.abc import Callable

import duckdb
from fastapi import HTTPException

from t1d_analytics.analytics import get_database_schema
from t1d_analytics.api_schemas import (
    ApiError,
    ChatRequest,
    ChatResponse,
    NlToSqlEvalRequest,
    NlToSqlEvalResponse,
)
from t1d_analytics.diagnostics import detect_database_provenance
from t1d_analytics.duckdb_pool import connect_duckdb_read_only_pooled
from t1d_analytics.errors import (
    DatabaseError,
    InvalidDatabasePathError,
    LlmProviderError,
    T1DError,
)
from t1d_analytics.nl_to_sql_translator import compute_optional_gold_comparison
from t1d_analytics.sql_fallback_synthesizer import (
    TranslationResult,
    synthesize_fallback_clinical_sql,
)
from t1d_analytics.sql_utils import SqlValue, classify_sql_input

logger = logging.getLogger(__name__)


def resolve_db_provenance_flags(
    db_path: str,
    connect_fn: Callable[..., duckdb.DuckDBPyConnection] = duckdb.connect,
) -> tuple[str, bool]:
    """
    Determine `(data_provenance, is_synthetic_data)` for `db_path`.

    Args:
        db_path: Path to the target DuckDB file.
        connect_fn: DuckDB connection factory callable.

    Returns:
        tuple[str, bool]: `(data_provenance, is_synthetic_data)`.

    """
    try:
        with connect_duckdb_read_only_pooled(
            db_path=db_path,
            connect_fn=connect_fn,
            disable_external_access=True,
        ) as conn:
            all_tables = [str(r[0]) for r in conn.execute("SHOW TABLES").fetchall()]
            provenance = detect_database_provenance(all_tables)
    except (duckdb.Error, OSError, RuntimeError, ValueError) as exc:
        logger.debug(f"Failed to resolve database provenance for '{db_path}': {exc}")
        provenance = "synthetic_seed"
    return provenance, (provenance != "jaeb_authentic")


def resolve_sql_mode_query(
    db_path: str,
    message: str,
    connect_fn: Callable[..., duckdb.DuckDBPyConnection] = duckdb.connect,
    schema_fn: Callable[[duckdb.DuckDBPyConnection], str] = get_database_schema,
) -> str:
    """
    Resolve a query submitted in `'sql'` mode, synthesizing schema-grounded SQL when input is natural language.

    Args:
        db_path: Validated path to the DuckDB database file.
        message: Raw user message string.
        connect_fn: DuckDB connection factory callable.
        schema_fn: Database schema extraction callable.

    Returns:
        str: Executable SQL query string.

    """
    if classify_sql_input(message) != "natural_language":
        return message
    with connect_duckdb_read_only_pooled(
        db_path=db_path,
        connect_fn=connect_fn,
        disable_external_access=True,
    ) as conn:
        schema = schema_fn(conn)
    synthesized = synthesize_fallback_clinical_sql(schema, message, db_path=db_path)
    return synthesized.sql_query


def execute_chat_turn(
    request: ChatRequest,
    db_validator: Callable[[str | None], str],
    execute_fn: Callable[[str, str], list[dict[str, SqlValue]]],
    translate_fn: Callable[..., TranslationResult],
    error_parser: Callable[[Exception], ApiError],
) -> ChatResponse:
    """
    Execute a synchronous chat turn (literal SQL or natural-language-to-SQL with optional execution).

    Args:
        request: Validated `ChatRequest` payload.
        db_validator: Callable resolving and sandboxing the target database path.
        execute_fn: Callable executing a read-only SQL query against the database.
        translate_fn: Callable translating a natural-language question into a `TranslationResult`.
        error_parser: Callable converting an `Exception` into a structured `ApiError`.

    Returns:
        ChatResponse: Structured chat response containing content, SQL query, result rows, or error details.

    Raises:
        HTTPException: If the database path validation raises an HTTPException.
        InvalidDatabasePathError: If the database path is outside the allowed workspace sandbox.

    """
    try:
        logger.info(
            f"Received chat request: model='{request.model}', message='{request.message}'"
        )
        db_path = db_validator(request.db_path)
        provenance, is_synthetic = resolve_db_provenance_flags(db_path)
        sql_kind = classify_sql_input(request.message)
        if (
            request.model.strip().lower() in ("sql", "")
            or sql_kind == "valid_read_only_sql"
        ):
            is_raw_sql = sql_kind != "natural_language"
            sql_query = resolve_sql_mode_query(db_path, request.message)
            results = execute_fn(db_path, sql_query)
            return ChatResponse(
                content="backend.literalSql",
                sqlResult=results,
                sqlQuery=sql_query,
                is_synthetic_data=is_synthetic,
                is_rule_based_sql=(not is_raw_sql),
                generation_mode="literal_sql" if is_raw_sql else "rule_based_fallback",
                data_provenance=provenance,
            )

        translation = translate_fn(
            db_path,
            request.message,
            model_name=request.model,
            history=request.history,
            api_key=request.api_key,
            provider=request.provider,
        )
        nl_results: list[dict[str, SqlValue]] | None = None
        sql_error: ApiError | None = None
        pred_err_str: str | None = None
        if translation.sql_query and request.auto_execute:
            try:
                nl_results = execute_fn(db_path, translation.sql_query)
            except (T1DError, ValueError, RuntimeError) as e:
                pred_err_str = str(e)
                sql_error = error_parser(e)

        gold_comp = compute_optional_gold_comparison(
            db_path=db_path,
            message=request.message,
            pred_sql=translation.sql_query,
            explicit_gold_sql=request.gold_sql,
            execute_fn=execute_fn,
            precomputed_pred_rows=nl_results,
            precomputed_pred_error=pred_err_str,
        )

        return ChatResponse(
            content=translation.full_response,
            sqlQuery=translation.sql_query,
            sqlResult=nl_results,
            error=sql_error,
            is_synthetic_data=is_synthetic,
            is_rule_based_sql=translation.used_fallback,
            generation_mode=translation.generation_mode,
            llm_error=translation.llm_error,
            llm_error_code=translation.llm_error_code,
            data_provenance=provenance,
            gold_comparison=gold_comp,
        )
    except (HTTPException, InvalidDatabasePathError):
        raise
    except (DatabaseError, ValueError) as ve:
        logger.error(f"Database error during chat request: {ve}")
        return ChatResponse(content="backend.errorDbExecution", error=error_parser(ve))
    except (LlmProviderError, T1DError, RuntimeError) as re_err:
        logger.error(f"NLP error during chat request: {re_err}")
        return ChatResponse(
            content="backend.errorNlpTranslation", error=error_parser(re_err)
        )
    except Exception as e:
        logger.exception("Unexpected error during chat request.")
        return ChatResponse(content="backend.errorUnexpected", error=error_parser(e))


def execute_nl_to_sql_eval_turn(
    request: NlToSqlEvalRequest,
    db_validator: Callable[[str | None], str],
    execute_fn: Callable[[str, str], list[dict[str, SqlValue]]],
    translate_fn: Callable[..., TranslationResult],
    error_parser: Callable[[Exception], ApiError],
) -> NlToSqlEvalResponse:
    """
    Translate a natural-language question to SQL, execute on DuckDB, and compare against Gold SQL.

    Args:
        request: Validated `NlToSqlEvalRequest` payload.
        db_validator: Callable resolving and sandboxing the target database path.
        execute_fn: Read-only DuckDB query execution callable.
        translate_fn: NL-to-SQL translation callable.
        error_parser: Exception to `ApiError` converter callable.

    Returns:
        NlToSqlEvalResponse: Evaluation payload with generated SQL, DuckDB rows, and gold comparison metrics.

    Raises:
        HTTPException: If database path validation raises an HTTPException.
        InvalidDatabasePathError: If the database path is outside the allowed workspace sandbox.

    """
    db_path = db_validator(request.db_path)
    provenance, is_synthetic = resolve_db_provenance_flags(db_path)
    try:
        translation = translate_fn(
            db_path,
            request.question,
            model_name=request.model,
            api_key=request.api_key,
            provider=request.provider,
        )
        pred_rows: list[dict[str, SqlValue]] | None = None
        sql_err: ApiError | None = None
        pred_err_str: str | None = None
        if translation.sql_query:
            try:
                pred_rows = execute_fn(db_path, translation.sql_query)
            except (T1DError, ValueError, RuntimeError) as exc:
                pred_err_str = str(exc)
                sql_err = error_parser(exc)

        gold_comp = compute_optional_gold_comparison(
            db_path=db_path,
            message=request.question,
            pred_sql=translation.sql_query,
            explicit_gold_sql=request.gold_sql,
            execute_fn=execute_fn,
            precomputed_pred_rows=pred_rows,
            precomputed_pred_error=pred_err_str,
        )
        return NlToSqlEvalResponse(
            question=request.question,
            model=request.model,
            content=translation.full_response,
            sqlQuery=translation.sql_query,
            sqlResult=pred_rows,
            gold_comparison=gold_comp,
            is_rule_based_sql=translation.used_fallback,
            generation_mode=translation.generation_mode,
            is_synthetic_data=is_synthetic,
            data_provenance=provenance,
            error=sql_err,
        )
    except Exception as exc:
        return NlToSqlEvalResponse(
            question=request.question,
            model=request.model,
            content="backend.errorNlpTranslation",
            is_synthetic_data=is_synthetic,
            data_provenance=provenance,
            error=error_parser(exc),
        )


__all__ = [
    "execute_chat_turn",
    "execute_nl_to_sql_eval_turn",
    "resolve_db_provenance_flags",
    "resolve_sql_mode_query",
]
