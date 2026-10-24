"""Tests for the T1DError domain exception hierarchy."""

from fastapi import Request

from t1d_analytics.api import parse_error, t1d_error_exception_handler
from t1d_analytics.errors import (
    DatabaseError,
    DatabaseLockedError,
    DatabaseNotFoundError,
    DownloadError,
    EmptyQueryError,
    HardwareUnavailableError,
    InsufficientDataError,
    InvalidColumnNameError,
    InvalidDatabasePathError,
    InvalidIdentifierError,
    InvalidTableNameError,
    LlmAuthenticationError,
    LlmConnectionError,
    LlmProviderError,
    LlmQuotaExceededError,
    LlmRateLimitError,
    LlmTranslationError,
    MissingApiKeyError,
    MissingSdkError,
    MultipleStatementsError,
    PathAccessDeniedError,
    ProviderNotConfiguredError,
    ReadOnlyViolationError,
    SchemaReadError,
    SessionNotFoundError,
    SqlExecutionError,
    T1DError,
    TableNotFoundError,
    TrainingError,
    TrainingExecutionError,
    TrainingJobNotFoundError,
)


def test_t1d_error_defaults_and_overrides() -> None:
    """Test T1DError base initialization with default and custom arguments."""
    err_default = T1DError()
    assert err_default.error_code == "backend.errorUnexpected"
    assert err_default.status_code == 500
    assert err_default.params == {}
    assert err_default.to_dict() == {
        "error_code": "backend.errorUnexpected",
        "params": {},
    }

    err_with_msg = T1DError("Something failed")
    assert err_with_msg.error_code == "backend.errorUnexpected"
    assert err_with_msg.params == {"error": "Something failed"}
    assert "Something failed" in str(err_with_msg)

    err_custom = T1DError(
        "Custom error",
        error_code="backend.customCode",
        params={"key": "val"},
        status_code=422,
    )
    assert err_custom.error_code == "backend.customCode"
    assert err_custom.status_code == 422
    assert err_custom.params == {"key": "val"}
    assert err_custom.to_dict() == {
        "error_code": "backend.customCode",
        "params": {"key": "val"},
    }


def test_domain_error_subclasses() -> None:
    """Test all domain error subclasses preserve expected status codes, error codes, and base types."""
    db_err = DatabaseNotFoundError(params={"path": "missing.duckdb"})
    assert isinstance(db_err, (T1DError, ValueError, FileNotFoundError))
    assert db_err.error_code == "backend.dbNotFound"
    assert db_err.status_code == 404

    tbl_err = TableNotFoundError(params={"table": "patients"})
    assert isinstance(tbl_err, (T1DError, ValueError))
    assert tbl_err.error_code == "backend.tableNotFound"
    assert tbl_err.status_code == 404

    id_err = InvalidIdentifierError(params={"identifier": "bad;table"})
    assert isinstance(id_err, (T1DError, ValueError))
    assert id_err.error_code == "backend.invalidTableName"
    assert id_err.status_code == 400

    sql_err = SqlExecutionError(params={"error": "syntax error"})
    assert isinstance(sql_err, (T1DError, ValueError))
    assert sql_err.error_code == "backend.sqlExecution"
    assert sql_err.status_code == 400

    ro_err = ReadOnlyViolationError("DROP is not allowed")
    assert isinstance(ro_err, (SqlExecutionError, T1DError, ValueError))
    assert ro_err.error_code == "backend.sqlReadOnlyViolation"
    assert ro_err.status_code == 400

    schema_err = SchemaReadError(params={"error": "corrupt"})
    assert isinstance(schema_err, (T1DError, ValueError))
    assert schema_err.error_code == "backend.readSchemaFailed"
    assert schema_err.status_code == 500

    prov_err = ProviderNotConfiguredError(params={"provider": "openai"})
    assert isinstance(prov_err, (T1DError, RuntimeError))
    assert prov_err.error_code == "backend.missingApiKey"
    assert prov_err.status_code == 400

    sdk_err = MissingSdkError(params={"error": "any-llm-sdk[openai]"})
    assert isinstance(sdk_err, (T1DError, RuntimeError))
    assert sdk_err.error_code == "backend.missingSdk"
    assert sdk_err.status_code == 500

    llm_err = LlmTranslationError(params={"error": "rate limit"})
    assert isinstance(llm_err, (T1DError, RuntimeError))
    assert llm_err.error_code == "backend.llmTranslationError"
    assert llm_err.status_code == 502

    path_err = PathAccessDeniedError(params={"path": "/etc/passwd"})
    assert isinstance(path_err, (T1DError, PermissionError))
    assert path_err.error_code == "backend.accessDenied"
    assert path_err.status_code == 403

    db_path_err = InvalidDatabasePathError(params={"path": "../../etc/passwd"})
    assert isinstance(db_path_err, (T1DError, ValueError))
    assert db_path_err.error_code == "backend.invalidDatabasePath"
    assert db_path_err.status_code == 400

    sess_err = SessionNotFoundError(
        "Session not found", params={"session_id": "sess-1"}
    )
    assert isinstance(sess_err, (T1DError, KeyError))
    assert sess_err.error_code == "backend.sessionNotFound"
    assert sess_err.status_code == 404

    job_err = TrainingJobNotFoundError(params={"job_id": "job-1"})
    assert isinstance(job_err, (T1DError, KeyError))
    assert job_err.error_code == "backend.jobNotFound"
    assert job_err.status_code == 404

    train_err = TrainingError("Training failed")
    assert isinstance(train_err, (T1DError, RuntimeError))
    assert train_err.error_code == "backend.trainingFailed"
    assert train_err.status_code == 500

    db_base_err = DatabaseError("DB error")
    assert db_base_err.status_code == 500
    assert DatabaseLockedError("Locked").status_code == 503
    assert InvalidTableNameError("Bad table").status_code == 400
    assert InvalidColumnNameError("Bad col").status_code == 400
    assert EmptyQueryError("Empty").status_code == 400
    assert MultipleStatementsError("Multi").status_code == 400
    assert LlmProviderError("Provider err").status_code == 502
    assert MissingApiKeyError("Missing key").status_code == 400
    assert LlmRateLimitError("Rate limit").status_code == 429
    assert LlmAuthenticationError("Auth failed").status_code == 401
    assert LlmQuotaExceededError("Quota").status_code == 402
    assert LlmConnectionError("Conn").status_code == 502
    assert InsufficientDataError("Too few rows").status_code == 400
    assert HardwareUnavailableError("No TPU").status_code == 503
    assert TrainingExecutionError("Exec failed").status_code == 500

    dl_err = DownloadError("SHA-256 mismatch")
    assert isinstance(dl_err, (T1DError, RuntimeError))
    assert dl_err.error_code == "backend.downloadFailed"
    assert dl_err.status_code == 502


def test_t1d_error_fastapi_exception_handler() -> None:
    """Test FastAPI exception handler for T1DError."""
    import asyncio

    scope = {"type": "http", "method": "GET", "path": "/api/test", "headers": []}
    req = Request(scope)
    exc = DatabaseNotFoundError(params={"path": "missing.duckdb"})
    resp = asyncio.run(t1d_error_exception_handler(req, exc))
    assert resp.status_code == 404
    assert b"backend.dbNotFound" in resp.body

    api_err = parse_error(exc)
    assert api_err.error_code == "backend.dbNotFound"
    assert api_err.params == {"path": "missing.duckdb"}
