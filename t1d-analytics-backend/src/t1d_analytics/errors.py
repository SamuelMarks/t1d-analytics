"""Domain error hierarchy for T1D Analytics."""

from typing import Dict, Mapping, Optional, Union

from typing_extensions import TypedDict

ErrorParamValue = Union[str, int, float, bool, None]


class ErrorDetailDict(TypedDict):
    """
    Structured dictionary representation of a domain error for API responses.

    Attributes
    ----------
        error_code: Internationalization translation key for the error.
        params: Structured interpolation parameters for the error message.

    """

    error_code: str
    params: Dict[str, ErrorParamValue]


class T1DError(Exception):
    """
    Base exception class for all T1D Analytics domain errors.

    Attributes
    ----------
        error_code: Internationalization translation key (e.g., 'backend.sqlExecution').
        message: Human-readable fallback error description.
        params: Structured interpolation parameters for localized error rendering.
        status_code: Default HTTP status code corresponding to this error.
        headers: Optional HTTP response headers associated with the error.

    """

    default_error_code: str = "backend.errorUnexpected"
    default_status_code: int = 500

    def __init__(
        self,
        message: str = "",
        *,
        error_code: Optional[str] = None,
        params: Optional[Mapping[str, ErrorParamValue]] = None,
        status_code: Optional[int] = None,
        headers: Optional[Mapping[str, str]] = None,
    ) -> None:
        """
        Initialize a T1DError instance.

        Args:
        ----
            message: Human-readable error message.
            error_code: Optional override for the i18n error code.
            params: Optional dictionary of error interpolation parameters.
            status_code: Optional override for the HTTP status code.
            headers: Optional HTTP headers to attach to error responses.

        """
        self.error_code: str = error_code or self.default_error_code
        self.params: Dict[str, ErrorParamValue] = (
            dict(params)
            if params is not None
            else ({"error": message} if message else {})
        )
        fallback_msg = str(self.params.get("error") or self.error_code)
        self.message: str = message or fallback_msg
        self.status_code: int = (
            status_code if status_code is not None else self.default_status_code
        )
        self.headers: Optional[Dict[str, str]] = (
            dict(headers) if headers is not None else None
        )
        display_msg = (
            f"{self.message} [{self.error_code}]"
            if self.error_code not in self.message
            else self.message
        )
        super().__init__(display_msg)

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


InvalidSortColumnError = InvalidColumnNameError


class InvalidPaginationError(DatabaseError):
    """Raised when table pagination parameters (limit or offset) are out of valid bounds."""

    default_error_code = "backend.invalidPagination"
    default_status_code = 400


class InvalidSortOrderError(DatabaseError):
    """Raised when table sort order is neither 'asc' nor 'desc'."""

    default_error_code = "backend.invalidSortOrder"
    default_status_code = 400


class ExportError(DatabaseError):
    """Raised when streaming a table export (CSV, JSON, XLSX, Parquet) fails unexpectedly."""

    default_error_code = "backend.serverError"
    default_status_code = 500


class RateLimitExceededError(T1DError):
    """Raised when a client exceeds the sliding-window API request rate limit."""

    default_error_code = "backend.rateLimitExceeded"
    default_status_code = 429

    def __init__(
        self,
        message: str = "",
        *,
        retry_after: int = 1,
        params: Optional[Mapping[str, ErrorParamValue]] = None,
    ) -> None:
        """
        Initialize a RateLimitExceededError with a Retry-After header.

        Args:
        ----
            message: Optional human-readable error message.
            retry_after: Seconds until the client may retry.
            params: Optional dictionary of error interpolation parameters.

        """
        self.retry_after: int = max(1, int(retry_after))
        super().__init__(
            message,
            error_code=self.default_error_code,
            params=params,
            status_code=self.default_status_code,
            headers={"Retry-After": str(self.retry_after)},
        )


class SqlExecutionError(DatabaseError):
    """Raised when a DuckDB SQL query fails during execution or times out."""

    default_error_code = "backend.sqlExecution"
    default_status_code = 400


class ReadOnlyViolationError(SqlExecutionError):
    """Raised when a non-read-only SQL statement is submitted to a read-only execution context."""

    default_error_code = "backend.sqlReadOnlyViolation"
    default_status_code = 400


class EmptyQueryError(ReadOnlyViolationError):
    """Raised when an empty or whitespace-only SQL statement is submitted."""

    default_error_code = "backend.emptyMessage"
    default_status_code = 400


class MultipleStatementsError(ReadOnlyViolationError):
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


class ToolchainNotFoundError(TrainingError):
    """Raised when an external training or cloud CLI toolchain (gemma-4-sql, xpk, gcloud) is not installed."""

    default_error_code = "backend.toolchainMissing"
    default_status_code = 503


class GcsSyncError(TrainingError):
    """Raised when syncing training datasets or checkpoints with Google Cloud Storage fails."""

    default_error_code = "backend.gcsSyncFailed"
    default_status_code = 502


class TpuWorkloadError(TrainingError):
    """Raised when dispatching, polling, or reclaiming a remote TPU workload fails."""

    default_error_code = "backend.tpuWorkloadFailed"
    default_status_code = 502


class SeedVerificationError(DatabaseError, RuntimeError):
    """Raised when database seeding verification fails due to missing or empty clinical tables."""

    default_error_code = "backend.seedVerificationFailed"
    default_status_code = 500


class DownloadError(T1DError, RuntimeError, ValueError):
    """Raised when dataset downloading, checksum verification, or ASP.NET postback fails."""

    default_error_code = "backend.downloadFailed"
    default_status_code = 502


class CliValidationError(T1DError, ValueError):
    """Raised when CLI arguments or training hyperparameters fail validation."""

    default_error_code = "backend.cliValidationFailed"
    default_status_code = 400


class AuthPluginError(T1DError, ValueError):
    """Raised when dynamically loading or invoking an authentication plugin fails."""

    default_error_code = "backend.authPluginFailed"
    default_status_code = 400


class InvalidTrainingBackendError(TrainingError, ValueError):
    """Raised when an unsupported training backend identifier is requested."""

    default_error_code = "backend.invalidBackend"
    default_status_code = 400


class GemmaToolchainError(ToolchainNotFoundError):
    """Raised when the external gemma-4-sql binary or Python module fails or is missing."""

    default_error_code = "backend.gemmaToolchainFailed"
    default_status_code = 503


class InvalidStageError(TrainingError, ValueError):
    """Raised when an unsupported training or ETL pipeline stage is requested."""

    default_error_code = "backend.invalidStage"
    default_status_code = 400


class TpuOrchestrationError(TpuWorkloadError):
    """Raised when Cloud TPU workload dispatch, status polling, or reclamation fails."""

    default_error_code = "backend.tpuOrchestrationFailed"
    default_status_code = 502


class TpuPreemptionError(TpuOrchestrationError):
    """Raised when a spot Cloud TPU workload is preempted during execution."""

    default_error_code = "backend.tpuPreempted"
    default_status_code = 503


class TpuPollingTimeoutError(TpuOrchestrationError, TimeoutError):
    """Raised when polling a remote TPU workload or external trainer exceeds its timeout."""

    default_error_code = "backend.tpuPollingTimeout"
    default_status_code = 504


class GcsStorageError(GcsSyncError):
    """Raised when Google Cloud Storage upload, download, or bucket validation fails."""

    default_error_code = "backend.gcsStorageFailed"
    default_status_code = 502


class ChecksumMismatchError(GcsStorageError, ValueError):
    """Raised when a downloaded checkpoint or dataset fails checksum verification."""

    default_error_code = "backend.checksumMismatch"
    default_status_code = 502


class InvalidGcsUriError(GcsStorageError, ValueError):
    """Raised when a Google Cloud Storage URI does not begin with the required 'gs://' scheme."""

    default_error_code = "backend.invalidGcsUri"
    default_status_code = 400


class ConfigFileNotFoundError(TrainingError, FileNotFoundError, ValueError):
    """Raised when a required training or ETL configuration file is missing on disk."""

    default_error_code = "backend.configNotFound"
    default_status_code = 404


class SubprocessTimeoutError(TrainingError, TimeoutError):
    """Raised when a local training or ETL CLI subprocess exceeds its execution timeout."""

    default_error_code = "backend.subprocessTimeout"
    default_status_code = 504


class SchemaExtractionError(DatabaseError, RuntimeError):
    """Raised when extracting or formatting database schemas for LLM prompts fails."""

    default_error_code = "backend.schemaExtractionFailed"
    default_status_code = 500
