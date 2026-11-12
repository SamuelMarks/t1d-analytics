"""Tests for the T1DError domain exception hierarchy and production usage AST audit."""

import ast
import asyncio
import inspect
from pathlib import Path

from fastapi import Request

from t1d_analytics import errors as errors_module
from t1d_analytics.api import t1d_error_exception_handler
from t1d_analytics.api_dependencies import parse_error
from t1d_analytics.errors import (
    AuthPluginError,
    ChatJobNotFoundError,
    ChecksumMismatchError,
    CliValidationError,
    ConfigFileNotFoundError,
    DatabaseError,
    DatabaseLockedError,
    DatabaseNotFoundError,
    DatasetNotFoundError,
    DownloadError,
    EmptyQueryError,
    ExportError,
    FrontendBuildNotFoundError,
    GcsStorageError,
    GemmaToolchainError,
    HardwareUnavailableError,
    InsufficientDataError,
    InvalidDatabasePathError,
    InvalidGcsUriError,
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
    MissingSdkError,
    MultipleStatementsError,
    PathAccessDeniedError,
    ProviderNotConfiguredError,
    RateLimitExceededError,
    ReadOnlyViolationError,
    SchemaReadError,
    SeedVerificationError,
    SessionNotFoundError,
    SqlExecutionError,
    SqlTimeoutError,
    SubprocessTimeoutError,
    T1DError,
    TableNotFoundError,
    TpuOrchestrationError,
    TpuPollingTimeoutError,
    TpuPreemptionError,
    TrainingError,
    TrainingExecutionError,
    TrainingJobNotFoundError,
    UnauthorizedApiError,
    UnsupportedExportFormatError,
    UnsupportedFileFormatError,
    exception_to_error_detail,
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


def test_domain_error_subclasses_serialization_and_http_mapping() -> None:
    """Verify every T1DError subclass serializes cleanly and maps through FastAPI t1d_error_exception_handler."""
    scope = {"type": "http", "method": "GET", "path": "/api/test", "headers": []}
    req = Request(scope)

    expected_specs: list[tuple[type[T1DError], str, int]] = [
        (DatabaseError, "backend.errorDbExecution", 500),
        (DatabaseLockedError, "backend.databaseLocked", 503),
        (DatabaseNotFoundError, "backend.dbNotFound", 404),
        (TableNotFoundError, "backend.tableNotFound", 404),
        (InvalidTableNameError, "backend.invalidTable", 400),
        (InvalidSortColumnError, "backend.invalidSortColumn", 400),
        (InvalidPaginationError, "backend.invalidPagination", 400),
        (InvalidSortOrderError, "backend.invalidSortOrder", 400),
        (ExportError, "backend.serverError", 500),
        (RateLimitExceededError, "backend.rateLimitExceeded", 429),
        (SqlExecutionError, "backend.sqlExecution", 400),
        (ReadOnlyViolationError, "backend.sqlReadOnlyViolation", 400),
        (EmptyQueryError, "backend.emptyMessage", 400),
        (MultipleStatementsError, "backend.sqlMultipleStatements", 400),
        (SchemaReadError, "backend.readSchemaFailed", 500),
        (LlmProviderError, "backend.llmTranslationError", 502),
        (ProviderNotConfiguredError, "backend.missingApiKey", 400),
        (LlmRateLimitError, "backend.llmRateLimit", 429),
        (LlmAuthenticationError, "backend.llmAuthFailed", 401),
        (LlmQuotaExceededError, "backend.llmQuotaExceeded", 402),
        (LlmConnectionError, "backend.llmConnectionFailed", 502),
        (MissingSdkError, "backend.missingSdk", 500),
        (LlmTranslationError, "backend.llmTranslationError", 502),
        (PathAccessDeniedError, "backend.accessDenied", 403),
        (FrontendBuildNotFoundError, "backend.frontendBuildNotFound", 404),
        (UnauthorizedApiError, "backend.unauthorized", 401),
        (InvalidDatabasePathError, "backend.invalidDatabasePath", 400),
        (SessionNotFoundError, "backend.sessionNotFound", 404),
        (TrainingJobNotFoundError, "backend.jobNotFound", 404),
        (ChatJobNotFoundError, "backend.chatJobNotFound", 404),
        (TrainingError, "backend.trainingFailed", 500),
        (InsufficientDataError, "backend.insufficientData", 400),
        (HardwareUnavailableError, "backend.hardwareUnavailable", 503),
        (TrainingExecutionError, "backend.trainingExecutionFailed", 500),
        (SeedVerificationError, "backend.seedVerificationFailed", 500),
        (DownloadError, "backend.downloadFailed", 502),
        (CliValidationError, "backend.cliValidationFailed", 400),
        (AuthPluginError, "backend.authPluginFailed", 400),
        (InvalidTrainingBackendError, "backend.invalidBackend", 400),
        (GemmaToolchainError, "backend.gemmaToolchainFailed", 503),
        (InvalidStageError, "backend.invalidStage", 400),
        (TpuOrchestrationError, "backend.tpuOrchestrationFailed", 502),
        (TpuPreemptionError, "backend.tpuPreempted", 503),
        (TpuPollingTimeoutError, "backend.tpuPollingTimeout", 504),
        (GcsStorageError, "backend.gcsStorageFailed", 502),
        (ChecksumMismatchError, "backend.checksumMismatch", 502),
        (InvalidGcsUriError, "backend.invalidGcsUri", 400),
        (ConfigFileNotFoundError, "backend.configNotFound", 404),
        (DatasetNotFoundError, "backend.datasetNotFound", 404),
        (SqlTimeoutError, "backend.queryTimeout", 504),
        (UnsupportedExportFormatError, "backend.unsupportedExportFormat", 400),
        (UnsupportedFileFormatError, "backend.unsupportedFileFormat", 400),
        (SubprocessTimeoutError, "backend.subprocessTimeout", 504),
    ]

    discovered_subclasses = {
        cls
        for _, cls in inspect.getmembers(errors_module, inspect.isclass)
        if issubclass(cls, T1DError) and cls is not T1DError
    }
    assert {cls for cls, _, _ in expected_specs} == discovered_subclasses

    fallback_resp = asyncio.run(
        t1d_error_exception_handler(req, RuntimeError("unexpected plain error"))
    )
    assert fallback_resp.status_code == 500
    assert b"backend.errorUnexpected" in fallback_resp.body

    for err_cls, expected_code, expected_status in expected_specs:
        instance = (
            RateLimitExceededError(
                "Rate limited", retry_after=15, params={"limit": 10, "window": 60}
            )
            if err_cls is RateLimitExceededError
            else err_cls("Diagnostic message")
        )
        assert instance.error_code == expected_code
        assert instance.status_code == expected_status
        serialized = exception_to_error_detail(instance)
        assert serialized["error_code"] == expected_code
        api_err = parse_error(instance)
        assert api_err.error_code == expected_code

        http_resp = asyncio.run(t1d_error_exception_handler(req, instance))
        assert http_resp.status_code == expected_status
        assert expected_code.encode("utf-8") in http_resp.body
        if isinstance(instance, RateLimitExceededError):
            assert instance.retry_after == 15
            assert http_resp.headers.get("retry-after") == "15"

    resp_fallback = asyncio.run(t1d_error_exception_handler(req, RuntimeError("raw")))
    assert resp_fallback.status_code == 500
    assert b"backend.errorUnexpected" in resp_fallback.body
    assert exception_to_error_detail(RuntimeError("boom")) == {
        "error_code": "backend.errorUnexpected",
        "params": {"error": "boom"},
    }


def test_every_defined_domain_error_is_raised_in_production() -> None:
    """AST-verify every concrete T1DError subclass in errors.py is raised or constructed in src/t1d_analytics/."""
    src_dir = Path(errors_module.__file__).resolve().parent
    category_base_classes = {
        "T1DError",
        "DatabaseError",
        "LlmProviderError",
        "TrainingError",
    }
    defined_concrete_errors = {
        name
        for name, cls in inspect.getmembers(errors_module, inspect.isclass)
        if issubclass(cls, T1DError) and name not in category_base_classes
    }

    production_instantiated: set[str] = set()
    for py_file in src_dir.rglob("*.py"):
        if py_file.name == "errors.py":
            continue
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.Raise) and node.exc is not None:
                target = node.exc.func if isinstance(node.exc, ast.Call) else node.exc
                if isinstance(target, ast.Name):
                    production_instantiated.add(target.id)
                elif isinstance(target, ast.Attribute):
                    production_instantiated.add(target.attr)
            elif isinstance(node, ast.Call):
                fn = node.func
                if isinstance(fn, ast.Name) and fn.id in defined_concrete_errors:
                    production_instantiated.add(fn.id)
                elif (
                    isinstance(fn, ast.Attribute) and fn.attr in defined_concrete_errors
                ):
                    production_instantiated.add(fn.attr)

    missing_in_production = sorted(defined_concrete_errors - production_instantiated)
    assert missing_in_production == []
