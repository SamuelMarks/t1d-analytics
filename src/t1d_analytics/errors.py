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


class DatabaseNotFoundError(T1DError, ValueError, FileNotFoundError):
    """Raised when a target DuckDB database file does not exist or cannot be opened."""

    default_error_code = "backend.dbNotFound"
    default_status_code = 404


class TableNotFoundError(T1DError, ValueError):
    """Raised when a requested table does not exist in the target DuckDB database."""

    default_error_code = "backend.tableNotFound"
    default_status_code = 404


class InvalidIdentifierError(T1DError, ValueError):
    """Raised when a SQL table or column identifier fails validation."""

    default_error_code = "backend.invalidTableName"
    default_status_code = 400


class SqlExecutionError(T1DError, ValueError):
    """Raised when a DuckDB SQL query fails during execution or times out."""

    default_error_code = "backend.sqlExecution"
    default_status_code = 400


class ReadOnlyViolationError(SqlExecutionError):
    """Raised when a non-read-only SQL statement is submitted to a read-only execution context."""

    default_error_code = "backend.sqlReadOnlyViolation"
    default_status_code = 400


class SchemaReadError(T1DError, ValueError):
    """Raised when introspecting the database schema fails."""

    default_error_code = "backend.readSchemaFailed"
    default_status_code = 500


class ProviderNotConfiguredError(T1DError, RuntimeError):
    """Raised when an external cloud LLM provider lacks a required API key."""

    default_error_code = "backend.missingApiKey"
    default_status_code = 400


class MissingSdkError(T1DError, RuntimeError):
    """Raised when an optional LLM or cloud SDK dependency is not installed."""

    default_error_code = "backend.missingSdk"
    default_status_code = 500


class LlmTranslationError(T1DError, RuntimeError):
    """Raised when natural-language-to-SQL translation via an LLM fails."""

    default_error_code = "backend.llmTranslationError"
    default_status_code = 502


class PathAccessDeniedError(T1DError, PermissionError):
    """Raised when a file or directory path violates security sandboxing rules."""

    default_error_code = "backend.accessDenied"
    default_status_code = 403


class TrainingError(T1DError, RuntimeError):
    """Base exception for synthetic data generation, training runner, and toolchain bridge errors."""

    default_error_code = "backend.trainingFailed"
    default_status_code = 500


class DownloadError(T1DError, RuntimeError):
    """Raised when dataset downloading, checksum verification, or ASP.NET postback fails."""

    default_error_code = "backend.downloadFailed"
    default_status_code = 502
