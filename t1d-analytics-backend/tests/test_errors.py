"""Tests for the T1DError domain exception hierarchy."""

from fastapi import Request

from t1d_analytics.api import parse_error, t1d_error_exception_handler
from t1d_analytics.errors import (
    AuthPluginError,
    ChecksumMismatchError,
    CliValidationError,
    ConfigFileNotFoundError,
    DatabaseError,
    DatabaseLockedError,
    DatabaseNotFoundError,
    DownloadError,
    EmptyQueryError,
    ExportError,
    GcsStorageError,
    GcsSyncError,
    GemmaToolchainError,
    HardwareUnavailableError,
    InsufficientDataError,
    InvalidColumnNameError,
    InvalidDatabasePathError,
    InvalidGcsUriError,
    InvalidIdentifierError,
    InvalidPaginationError,
    InvalidSortColumnError,
    InvalidSortOrderError,
    InvalidStageError,
    InvalidTableNameError,
    InvalidTrainingBackendError,
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
    RateLimitExceededError,
    ReadOnlyViolationError,
    SchemaExtractionError,
    SchemaReadError,
    SeedVerificationError,
    SessionNotFoundError,
    SqlExecutionError,
    SubprocessTimeoutError,
    T1DError,
    TableNotFoundError,
    ToolchainNotFoundError,
    TpuOrchestrationError,
    TpuPollingTimeoutError,
    TpuPreemptionError,
    TpuWorkloadError,
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

    cfg_err = ConfigFileNotFoundError("Missing config.yml")
    assert isinstance(cfg_err, (TrainingError, FileNotFoundError, ValueError))
    assert cfg_err.error_code == "backend.configNotFound"
    assert cfg_err.status_code == 404

    gcs_uri_err = InvalidGcsUriError("Invalid gs:// URI")
    assert isinstance(gcs_uri_err, (GcsStorageError, ValueError))
    assert not isinstance(gcs_uri_err, ChecksumMismatchError)
    assert gcs_uri_err.error_code == "backend.invalidGcsUri"
    assert gcs_uri_err.status_code == 400

    subproc_err = SubprocessTimeoutError("Subprocess timed out")
    assert isinstance(subproc_err, (TrainingError, TimeoutError, RuntimeError))
    assert not isinstance(subproc_err, TpuPollingTimeoutError)
    assert subproc_err.error_code == "backend.subprocessTimeout"
    assert subproc_err.status_code == 504

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

    pag_err = InvalidPaginationError(params={"limit": 0, "offset": -1})
    assert isinstance(pag_err, (T1DError, ValueError))
    assert pag_err.error_code == "backend.invalidPagination"
    assert pag_err.status_code == 400

    sort_ord_err = InvalidSortOrderError(params={"order": "invalid"})
    assert isinstance(sort_ord_err, (T1DError, ValueError))
    assert sort_ord_err.error_code == "backend.invalidSortOrder"
    assert sort_ord_err.status_code == 400

    sort_col_err = InvalidSortColumnError(params={"sort_by": "missing"})
    assert isinstance(sort_col_err, (T1DError, ValueError))
    assert sort_col_err.error_code == "backend.invalidSortColumn"
    assert sort_col_err.status_code == 400

    exp_err = ExportError(params={"error": "I/O failure"})
    assert isinstance(exp_err, (T1DError, ValueError))
    assert exp_err.error_code == "backend.serverError"
    assert exp_err.status_code == 500

    rl_err = RateLimitExceededError(retry_after=15, params={"limit": 10, "window": 60})
    assert isinstance(rl_err, T1DError)
    assert rl_err.error_code == "backend.rateLimitExceeded"
    assert rl_err.status_code == 429
    assert rl_err.retry_after == 15
    assert rl_err.headers == {"Retry-After": "15"}

    tc_err = ToolchainNotFoundError("gemma-4-sql missing")
    assert isinstance(tc_err, (T1DError, RuntimeError))
    assert tc_err.error_code == "backend.toolchainMissing"
    assert tc_err.status_code == 503

    gcs_err = GcsSyncError("GCS upload failed")
    assert isinstance(gcs_err, (T1DError, RuntimeError))
    assert gcs_err.error_code == "backend.gcsSyncFailed"
    assert gcs_err.status_code == 502

    tpu_err = TpuWorkloadError("TPU preempted")
    assert isinstance(tpu_err, (T1DError, RuntimeError))
    assert tpu_err.error_code == "backend.tpuWorkloadFailed"
    assert tpu_err.status_code == 502

    seed_err = SeedVerificationError("Missing clinical tables")
    assert isinstance(seed_err, (T1DError, ValueError, RuntimeError))
    assert seed_err.error_code == "backend.seedVerificationFailed"
    assert seed_err.status_code == 500

    cli_err = CliValidationError("Invalid learning rate")
    assert isinstance(cli_err, (T1DError, ValueError))
    assert cli_err.error_code == "backend.cliValidationFailed"
    assert cli_err.status_code == 400

    auth_err = AuthPluginError("Invalid plugin spec")
    assert isinstance(auth_err, (T1DError, ValueError))
    assert auth_err.error_code == "backend.authPluginFailed"
    assert auth_err.status_code == 400

    backend_err = InvalidTrainingBackendError(params={"backend": "bad"})
    assert isinstance(backend_err, (TrainingError, T1DError, ValueError))
    assert backend_err.error_code == "backend.invalidBackend"
    assert backend_err.status_code == 400

    gemma_err = GemmaToolchainError("gemma-4-sql failed")
    assert isinstance(gemma_err, (ToolchainNotFoundError, T1DError, RuntimeError))
    assert gemma_err.error_code == "backend.gemmaToolchainFailed"
    assert gemma_err.status_code == 503

    stage_err = InvalidStageError("Unsupported stage")
    assert isinstance(stage_err, (TrainingError, T1DError, ValueError))
    assert stage_err.error_code == "backend.invalidStage"
    assert stage_err.status_code == 400

    orch_err = TpuOrchestrationError("TPU dispatch error")
    assert isinstance(orch_err, (TpuWorkloadError, T1DError, RuntimeError))
    assert orch_err.error_code == "backend.tpuOrchestrationFailed"
    assert orch_err.status_code == 502

    preempt_err = TpuPreemptionError("Spot TPU preempted")
    assert isinstance(preempt_err, (TpuOrchestrationError, T1DError, RuntimeError))
    assert preempt_err.error_code == "backend.tpuPreempted"
    assert preempt_err.status_code == 503

    timeout_err = TpuPollingTimeoutError("TPU poll timed out")
    assert isinstance(timeout_err, (TpuOrchestrationError, T1DError, TimeoutError))
    assert timeout_err.error_code == "backend.tpuPollingTimeout"
    assert timeout_err.status_code == 504

    gcs_store_err = GcsStorageError("Bucket unreachable")
    assert isinstance(gcs_store_err, (GcsSyncError, T1DError, RuntimeError))
    assert gcs_store_err.error_code == "backend.gcsStorageFailed"
    assert gcs_store_err.status_code == 502

    checksum_err = ChecksumMismatchError("MD5 mismatch")
    assert isinstance(checksum_err, (GcsStorageError, T1DError, ValueError))
    assert checksum_err.error_code == "backend.checksumMismatch"
    assert checksum_err.status_code == 502

    schema_ext_err = SchemaExtractionError("Failed to extract schema")
    assert isinstance(schema_ext_err, (DatabaseError, T1DError, RuntimeError))
    assert schema_ext_err.error_code == "backend.schemaExtractionFailed"
    assert schema_ext_err.status_code == 500


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
