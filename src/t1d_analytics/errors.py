"""Domain error hierarchy for T1D Analytics."""

import json
from typing import Any, Dict, Optional, TypedDict


class ErrorDetailDict(TypedDict):
    """
    Structured dictionary representation of a domain error for API responses.

    Attributes
    ----------
        error_code: Internationalization translation key for the error.
        params: Structured interpolation parameters for the error message.

    """

    error_code: str
    params: Dict[str, Any]


class T1DError(Exception):
    """
    Base exception class for all T1D Analytics domain errors.

    Attributes
    ----------
        error_code: Internationalization translation key (e.g., 'backend.sqlExecution').
        message: Human-readable fallback error description.
        params: Structured interpolation parameters for localized error rendering.
        status_code: Default HTTP status code corresponding to this error.

    """

    default_error_code: str = "backend.errorUnexpected"
    default_status_code: int = 500

    def __init__(
        self,
        message: str = "",
        *,
        error_code: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
        status_code: Optional[int] = None,
    ) -> None:
        """
        Initialize a T1DError instance.

        Args:
        ----
            message: Human-readable error message.
            error_code: Optional override for the i18n error code.
            params: Optional dictionary of error interpolation parameters.
            status_code: Optional override for the HTTP status code.

        """
        self.error_code: str = error_code or self.default_error_code
        self.params: Dict[str, Any] = (
            dict(params)
            if params is not None
            else ({"error": message} if message else {})
        )
        self.message: str = message or json.dumps(
            {"error_code": self.error_code, "params": self.params}
        )
        self.status_code: int = (
            status_code if status_code is not None else self.default_status_code
        )
        super().__init__(
            json.dumps({"error_code": self.error_code, "params": self.params})
            if not message
            else f"{message} ({json.dumps({'error_code': self.error_code, 'params': self.params})})"
        )

    def to_dict(self) -> ErrorDetailDict:
        """
        Serialize the error into a structured dictionary suitable for API responses.

        Returns
        -------
            ErrorDetailDict: Dictionary containing 'error_code' and 'params'.

        """
        return {
            "error_code": self.error_code,
            "params": self.params,
        }


class DatabaseError(T1DError, ValueError):
    """Base exception for DuckDB database access, connection, or schema errors."""

    default_error_code = "backend.errorDbExecution"
    default_status_code = 500


class DatabaseNotFoundError(DatabaseError, FileNotFoundError):
    """Raised when a target DuckDB database file does not exist or cannot be opened."""

    default_error_code = "backend.dbNotFound"
    default_status_code = 404


class DatabaseLockedError(DatabaseError, RuntimeError):
    """Raised when a DuckDB database file remains locked after retry attempts."""

    default_error_code = "backend.dbLocked"
    default_status_code = 503


class TableNotFoundError(DatabaseError):
    """Raised when a requested table does not exist in the target DuckDB database."""

    default_error_code = "backend.tableNotFound"
    default_status_code = 404


class InvalidIdentifierError(DatabaseError):
    """Raised when a SQL table or column identifier fails validation."""

    default_error_code = "backend.invalidTableName"
    default_status_code = 400


class InvalidTableNameError(InvalidIdentifierError):
    """Raised when a SQL table name fails identifier validation."""

    default_error_code = "backend.invalidTable"
    default_status_code = 400


class InvalidColumnNameError(InvalidIdentifierError):
    """Raised when a SQL column name fails identifier validation."""

    default_error_code = "backend.invalidSortColumn"
    default_status_code = 400


class SqlExecutionError(DatabaseError):
    """Raised when a DuckDB SQL query fails during execution or times out."""

    default_error_code = "backend.sqlExecution"
    default_status_code = 400


class ReadOnlyViolationError(SqlExecutionError):
    """Raised when a non-read-only SQL statement is submitted to a read-only execution context."""

    default_error_code = "backend.sqlReadOnlyViolation"
    default_status_code = 400


class EmptyQueryError(SqlExecutionError):
    """Raised when an empty or whitespace-only SQL statement is submitted."""

    default_error_code = "backend.emptyMessage"
    default_status_code = 400


class MultipleStatementsError(SqlExecutionError):
    """Raised when multiple semicolon-delimited SQL statements are submitted in a single query."""

    default_error_code = "backend.multipleStatements"
    default_status_code = 400


class SchemaReadError(DatabaseError):
    """Raised when introspecting the database schema fails."""

    default_error_code = "backend.readSchemaFailed"
    default_status_code = 500


class LlmProviderError(T1DError, RuntimeError):
    """Base exception for LLM provider configuration, connectivity, and translation failures."""

    default_error_code = "backend.llmTranslationError"
    default_status_code = 502


class ProviderNotConfiguredError(LlmProviderError):
    """Raised when an external cloud LLM provider lacks a required API key."""

    default_error_code = "backend.missingApiKey"
    default_status_code = 400


MissingApiKeyError = ProviderNotConfiguredError


class LlmRateLimitError(LlmProviderError):
    """Raised when an external LLM provider rejects a request due to rate limiting."""

    default_error_code = "backend.llmRateLimit"
    default_status_code = 429


class LlmAuthenticationError(LlmProviderError):
    """Raised when an external LLM provider rejects the supplied API key."""

    default_error_code = "backend.llmAuthFailed"
    default_status_code = 401


class LlmQuotaExceededError(LlmProviderError):
    """Raised when an external LLM provider reports insufficient quota or billing limits."""

    default_error_code = "backend.llmQuotaExceeded"
    default_status_code = 402


class LlmConnectionError(LlmProviderError):
    """Raised when connecting to a local or cloud LLM service fails."""

    default_error_code = "backend.llmConnectionFailed"
    default_status_code = 502


class MissingSdkError(LlmProviderError):
    """Raised when an optional LLM or cloud SDK dependency is not installed."""

    default_error_code = "backend.missingSdk"
    default_status_code = 500


class LlmTranslationError(LlmProviderError):
    """Raised when natural-language-to-SQL translation via an LLM fails."""

    default_error_code = "backend.llmTranslationError"
    default_status_code = 502


class PathAccessDeniedError(T1DError, PermissionError):
    """Raised when a file or directory path violates security sandboxing rules."""

    default_error_code = "backend.accessDenied"
    default_status_code = 403


class InvalidDatabasePathError(DatabaseError):
    """Raised when a database path traverses outside allowed directories or contains invalid characters."""

    default_error_code = "backend.invalidDatabasePath"
    default_status_code = 400


class SessionNotFoundError(T1DError, KeyError):
    """Raised when a requested chat conversation session does not exist."""

    default_error_code = "backend.sessionNotFound"
    default_status_code = 404


class TrainingJobNotFoundError(T1DError, KeyError):
    """Raised when a requested background training job ID does not exist."""

    default_error_code = "backend.jobNotFound"
    default_status_code = 404


class TrainingError(T1DError, RuntimeError):
    """Base exception for synthetic data generation, training runner, and toolchain bridge errors."""

    default_error_code = "backend.trainingFailed"
    default_status_code = 500


class InsufficientDataError(TrainingError, ValueError):
    """Raised when a training dataset is missing or has fewer samples than required."""

    default_error_code = "backend.insufficientData"
    default_status_code = 400


class HardwareUnavailableError(TrainingError):
    """Raised when the requested hardware accelerator (GPU/TPU) is unavailable."""

    default_error_code = "backend.hardwareUnavailable"
    default_status_code = 503


class TrainingExecutionError(TrainingError):
    """Raised when a training step, checkpoint save, or external trainer command fails."""

    default_error_code = "backend.trainingExecutionFailed"
    default_status_code = 500


class DownloadError(T1DError, RuntimeError):
    """Raised when dataset downloading, checksum verification, or ASP.NET postback fails."""

    default_error_code = "backend.downloadFailed"
    default_status_code = 502
