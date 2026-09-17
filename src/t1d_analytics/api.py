"""FastAPI application for the T1D Analytics web interface."""

import datetime
import decimal
import io
import json
import logging
import math
import os
import threading
import time
import urllib.error
import urllib.request
import uuid
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import (
    Any,
    AsyncGenerator,
    AsyncIterator,
    Dict,
    Iterator,
    List,
    Optional,
    Union,
)

import duckdb
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from t1d_analytics.analytics import (
    extract_sql_from_response,
    get_database_schema,
    profile_table,
)
from t1d_analytics.diagnostics import (
    check_database_health,
    get_system_health,
    log_startup_diagnostics,
)
from t1d_analytics.i18n import get_translator

SqlValue = Union[str, int, float, bool, None]

_sessions_lock = threading.Lock()


def sanitize_sql_value(val: Any) -> SqlValue:
    """
    Sanitize an arbitrary database column value into a JSON-compliant SqlValue.

    Args:
    ----
        val: The raw value returned from the database driver.

    Returns:
    -------
        A JSON-compliant and Pydantic-compatible SqlValue.

    """
    if val is None:
        return None
    if isinstance(val, (datetime.date, datetime.datetime, datetime.time)):
        return val.isoformat()
    if isinstance(val, decimal.Decimal):
        return float(val)
    if isinstance(val, uuid.UUID):
        return str(val)
    if isinstance(val, float):
        if math.isnan(val) or math.isinf(val):
            return None
        return val
    if isinstance(val, (int, bool, str)):
        return val
    return str(val)


# Rate Limiting & Auth State
RATE_LIMIT_STORE: Dict[str, List[float]] = {}
RATE_LIMIT_LOCK = threading.Lock()


class BaseRateLimiter(ABC):
    """Abstract base class defining the contract for rate limiters."""

    @abstractmethod
    def is_allowed(
        self, client_ip: str, limit: int = 120, window_seconds: float = 60.0
    ) -> bool:
        """
        Check if a client IP has exceeded the sliding-window rate limit.

        Args:
        ----
            client_ip: The client IP address string.
            limit: Maximum allowed requests within the window.
            window_seconds: Duration of the sliding window in seconds.

        Returns:
        -------
            bool: True if allowed, False if rate-limited.

        """

    @abstractmethod
    def get_retry_after(self, client_ip: str, window_seconds: float = 60.0) -> int:
        """
        Calculate remaining seconds before the client can retry.

        Args:
        ----
            client_ip: The client IP address string.
            window_seconds: Duration of the sliding window in seconds.

        Returns:
        -------
            int: Seconds until the rate limit expires.

        """

    @abstractmethod
    def reset(self) -> None:
        """Reset rate limiter state."""


class MemoryRateLimiter(BaseRateLimiter):
    """In-memory sliding-window rate limiter using thread synchronization."""

    def __init__(self) -> None:
        """Initialize in-memory rate limiter."""
        super().__init__()

    def is_allowed(
        self, client_ip: str, limit: int = 120, window_seconds: float = 60.0
    ) -> bool:
        """
        Check and record request timestamp in memory.

        Args:
        ----
            client_ip: The client IP address string.
            limit: Maximum allowed requests within the window.
            window_seconds: Duration of the sliding window in seconds.

        Returns:
        -------
            bool: True if allowed, False if rate-limited.

        """
        now = time.time()
        with RATE_LIMIT_LOCK:
            timestamps = RATE_LIMIT_STORE.get(client_ip, [])
            valid_timestamps = [t for t in timestamps if now - t < window_seconds]
            if len(valid_timestamps) >= limit:
                RATE_LIMIT_STORE[client_ip] = valid_timestamps
                return False
            valid_timestamps.append(now)
            RATE_LIMIT_STORE[client_ip] = valid_timestamps
            return True

    def get_retry_after(self, client_ip: str, window_seconds: float = 60.0) -> int:
        """
        Calculate retry-after delay based on oldest valid timestamp.

        Args:
        ----
            client_ip: The client IP address string.
            window_seconds: Duration of the sliding window in seconds.

        Returns:
        -------
            int: Seconds remaining until window expiry.

        """
        now = time.time()
        with RATE_LIMIT_LOCK:
            timestamps = RATE_LIMIT_STORE.get(client_ip, [])
            valid = [t for t in timestamps if now - t < window_seconds]
            if not valid:
                return 1
            oldest = min(valid)
            return max(1, math.ceil(window_seconds - (now - oldest)))

    def reset(self) -> None:
        """Clear all stored timestamps."""
        with RATE_LIMIT_LOCK:
            RATE_LIMIT_STORE.clear()


class DuckDbRateLimiter(BaseRateLimiter):
    """Shared persistent rate limiter backed by a DuckDB table."""

    def __init__(self, db_path: Optional[str] = None) -> None:
        """
        Initialize DuckDB-backed rate limiter.

        Args:
        ----
            db_path: Path to DuckDB file, or ':memory:' if None.

        """
        super().__init__()
        self.db_path = db_path or os.environ.get("T1D_RATE_LIMITER_DB", ":memory:")
        self._lock = threading.Lock()
        self._conn = duckdb.connect(self.db_path)
        with self._lock:
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS api_rate_limits (client_ip VARCHAR, ts DOUBLE)"
            )

    def is_allowed(
        self, client_ip: str, limit: int = 120, window_seconds: float = 60.0
    ) -> bool:
        """
        Check and record request in DuckDB.

        Args:
        ----
            client_ip: The client IP address string.
            limit: Maximum allowed requests within the window.
            window_seconds: Duration of the sliding window in seconds.

        Returns:
        -------
            bool: True if allowed, False if rate-limited.

        """
        now = time.time()
        cutoff = now - window_seconds
        with self._lock:
            self._conn.execute("DELETE FROM api_rate_limits WHERE ts < ?", [cutoff])
            count_res = self._conn.execute(
                "SELECT COUNT(*) FROM api_rate_limits WHERE client_ip = ? AND ts >= ?",
                [client_ip, cutoff],
            ).fetchone()
            count = count_res[0] if count_res else 0
            if count >= limit:
                return False
            self._conn.execute(
                "INSERT INTO api_rate_limits VALUES (?, ?)", [client_ip, now]
            )
            return True

    def get_retry_after(self, client_ip: str, window_seconds: float = 60.0) -> int:
        """
        Calculate retry-after delay based on oldest valid record in DuckDB.

        Args:
        ----
            client_ip: The client IP address string.
            window_seconds: Duration of the sliding window in seconds.

        Returns:
        -------
            int: Seconds remaining until window expiry.

        """
        now = time.time()
        cutoff = now - window_seconds
        with self._lock:
            res = self._conn.execute(
                "SELECT MIN(ts) FROM api_rate_limits WHERE client_ip = ? AND ts >= ?",
                [client_ip, cutoff],
            ).fetchone()
            if not res or res[0] is None:
                return 1
            min_ts = float(res[0])
            return max(1, math.ceil(window_seconds - (now - min_ts)))

    def reset(self) -> None:
        """Drop or clear all records from rate limits table."""
        with self._lock:
            self._conn.execute("DELETE FROM api_rate_limits")


_RATE_LIMITER_INSTANCES: Dict[str, BaseRateLimiter] = {}
_RATE_LIMITER_INSTANCES_LOCK = threading.Lock()


def get_rate_limiter() -> BaseRateLimiter:
    """
    Resolve the configured rate limiter instance based on environment configuration.

    Returns
    -------
        BaseRateLimiter: MemoryRateLimiter or DuckDbRateLimiter instance.

    """
    backend = os.environ.get("T1D_RATE_LIMITER_BACKEND", "memory").lower().strip()
    with _RATE_LIMITER_INSTANCES_LOCK:
        if backend not in _RATE_LIMITER_INSTANCES:
            if backend == "duckdb":
                _RATE_LIMITER_INSTANCES[backend] = DuckDbRateLimiter()
            else:
                _RATE_LIMITER_INSTANCES[backend] = MemoryRateLimiter()
        return _RATE_LIMITER_INSTANCES[backend]


def check_rate_limit(
    client_ip: str, limit: int = 120, window_seconds: float = 60.0
) -> bool:
    """
    Check if a client IP has exceeded the sliding-window rate limit.

    Args:
    ----
        client_ip: The client IP address string.
        limit: Maximum number of allowed requests in the window.
        window_seconds: The duration of the sliding window in seconds.

    Returns:
    -------
        bool: True if the request is allowed, False if rate limited.

    """
    return get_rate_limiter().is_allowed(
        client_ip, limit=limit, window_seconds=window_seconds
    )


def enforce_rate_limit(
    request: Request, limit: int = 120, window_seconds: float = 60.0
) -> None:
    """
    Enforce sliding-window rate limiting on a specific endpoint.

    Args:
    ----
        request: The incoming FastAPI HTTP request.
        limit: Maximum requests allowed in the window (default: 120).
        window_seconds: Window size in seconds (default: 60.0).

    Raises:
    ------
        HTTPException: 429 Too Many Requests if the limit is exceeded.

    """
    client_ip = request.client.host if request.client else "unknown"
    if not check_rate_limit(client_ip, limit=limit, window_seconds=window_seconds):
        retry_after = get_rate_limiter().get_retry_after(
            client_ip, window_seconds=window_seconds
        )
        raise HTTPException(
            status_code=429,
            detail={
                "error_code": "backend.rateLimitExceeded",
                "params": {"limit": limit, "window": int(window_seconds)},
            },
            headers={"Retry-After": str(retry_after)},
        )


def verify_api_auth(request: Request) -> None:
    """
    Validate optional API key or Bearer token authentication.

    If T1D_API_KEY is configured in the environment, requests must provide either
    a matching 'X-API-Key' header or 'Authorization: Bearer <token>' header.
    Public health check endpoints (/api/health, /api/ready) are exempt.

    Args:
    ----
        request: The incoming FastAPI HTTP request.

    Raises:
    ------
        HTTPException: 401 Unauthorized if auth fails or token is missing.

    """
    required_key = os.environ.get("T1D_API_KEY")
    if not required_key:
        return

    # Exempt public health & doc routes
    exempt_paths = {"/api/health", "/api/ready", "/docs", "/openapi.json"}
    if request.url.path in exempt_paths:
        return

    api_key_header = request.headers.get("x-api-key")
    auth_header = request.headers.get("authorization")

    bearer_token = None
    if auth_header and auth_header.lower().startswith("bearer "):
        bearer_token = auth_header[7:].strip()

    provided_token = api_key_header or bearer_token
    if not provided_token or provided_token != required_key:
        raise HTTPException(
            status_code=401,
            detail={
                "error_code": "backend.unauthorized",
                "params": {"message": "Invalid or missing API key."},
            },
        )


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Lifespan context manager that logs system startup diagnostics.

    Args:
    ----
        app: The FastAPI application instance.

    Yields:
    ------
        None upon server startup.

    """
    log_startup_diagnostics()
    yield


app = FastAPI(
    title="T1D Analytics API",
    description="API for querying T1D datasets.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def auth_and_rate_limit_middleware(request: Request, call_next: Any) -> Any:
    """
    Global middleware enforcing API key verification when T1D_API_KEY is configured.

    Args:
    ----
        request: The incoming HTTP request.
        call_next: The next request handler.

    Returns:
    -------
        The response produced by the application or an auth error JSON response.

    """
    try:
        verify_api_auth(request)
    except HTTPException as exc:
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
    return await call_next(request)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle unhandled exceptions globally and return localized error message."""
    lang_header = request.headers.get("accept-language", "en")
    lang = lang_header.split(",")[0].split("-")[0].split(";")[0]
    _ = get_translator(lang)

    logger.exception(f"Unhandled exception: {exc}")
    # Return structured error for frontend
    return JSONResponse(
        status_code=500,
        content={
            "detail": {
                "error_code": "backend.serverError",
                "params": {"error": _("Internal server error: {}", exc)},
            }
        },
    )


def validate_db_path(
    db_path: Optional[str] = None,
    allowed_dirs: Optional[List[Path]] = None,
) -> str:
    """
    Validate and resolve a database path against allowed directory containment.

    Args:
    ----
        db_path: The requested database file path, or None for default.
        allowed_dirs: Optional list of allowed directory roots. Defaults to
            the workspace root, its subdirectories, configured T1D_DB_PATH
            parent directory, and system temp directories.

    Returns:
    -------
        Validated string path to the database file.

    Raises:
    ------
        HTTPException: If the path traverses outside allowed boundaries or contains null bytes.

    """
    raw_path = db_path or os.environ.get("T1D_DB_PATH", "t1d.duckdb")
    if "\0" in raw_path:
        raise HTTPException(
            status_code=400,
            detail={
                "error_code": "backend.invalidDatabasePath",
                "params": {"path": raw_path},
            },
        )

    resolved_path = Path(raw_path).resolve()

    if allowed_dirs is None:
        import tempfile

        cwd = Path.cwd().resolve()
        sys_tmp = Path(tempfile.gettempdir()).resolve()
        allowed_dirs = [
            cwd,
            sys_tmp,
            Path("/tmp").resolve(),
            Path("/private/tmp").resolve(),
            Path("/var/tmp").resolve(),
        ]
        env_db = os.environ.get("T1D_DB_PATH")
        if env_db:
            allowed_dirs.append(Path(env_db).resolve().parent)

    is_allowed = False
    for allowed_dir in allowed_dirs:
        resolved_allowed = allowed_dir.resolve()
        try:
            if resolved_path == resolved_allowed or resolved_path.is_relative_to(
                resolved_allowed
            ):
                is_allowed = True
                break
        except AttributeError:  # pragma: no cover
            if str(resolved_path).startswith(str(resolved_allowed)):
                is_allowed = True
                break

    if not is_allowed:
        raise HTTPException(
            status_code=400,
            detail={
                "error_code": "backend.invalidDatabasePath",
                "params": {"path": str(raw_path)},
            },
        )

    return str(resolved_path)


class ApiError(BaseModel):
    """Structured API error format."""

    error_code: str
    params: Dict[str, Any] = {}


class ChatMessageModel(BaseModel):
    """A single message in a conversation session."""

    role: str
    content: str
    sqlQuery: Optional[str] = None
    sqlResult: Optional[List[Dict[str, SqlValue]]] = None
    error: Optional[ApiError] = None
    model: Optional[str] = None


class ChatRequest(BaseModel):
    """
    Request model for the chat and streaming chat endpoints.

    Attributes
    ----------
        message: User prompt or query string.
        model: Model identifier name.
        provider: Optional provider name override ('ollama', 'openai', 'anthropic', 'google').
        api_key: Optional client-supplied API key for external cloud providers.
        db_path: Optional path to the DuckDB database.
        session_id: Optional ID of the active chat session.
        history: Optional list of prior conversation messages.

    """

    message: str
    model: str = "gemma4"
    provider: Optional[str] = None
    api_key: Optional[str] = None
    db_path: Optional[str] = None
    session_id: Optional[str] = None
    history: Optional[List[ChatMessageModel]] = None


class ChatResponse(BaseModel):
    """Response model for the chat endpoint."""

    content: str
    sqlResult: Optional[List[Dict[str, SqlValue]]] = None
    sqlQuery: Optional[str] = None
    error: Optional[ApiError] = None


class SessionModel(BaseModel):
    """A chat conversation session."""

    session_id: str
    title: str
    created_at: str
    updated_at: str
    messages: List[ChatMessageModel] = []


class CreateSessionRequest(BaseModel):
    """Payload to create or update a chat conversation session."""

    session_id: Optional[str] = None
    title: Optional[str] = None
    messages: List[ChatMessageModel] = []


class SessionListResponse(BaseModel):
    """Response model containing a list of chat sessions."""

    sessions: List[SessionModel]


class ModelInfo(BaseModel):
    """
    Model information.

    Attributes
    ----------
        name: Model identifier name.
        size: Optional size in bytes for local models.
        provider: Provider identifier ('ollama', 'openai', 'anthropic', 'google').
        configured: Whether credentials or host configurations exist in the environment.
        requires_key: Whether external authentication is required.
        available: Whether the model is currently active and usable.
        reachable: Whether the upstream provider host is responsive.
        error_code: Optional diagnostic error code if unavailable ('PROVIDER_OFFLINE', 'MISSING_API_KEY').

    """

    name: str
    size: Optional[int] = None
    provider: str = "ollama"
    configured: bool = True
    requires_key: bool = False
    available: bool = True
    reachable: bool = True
    error_code: Optional[str] = None


class ModelsResponse(BaseModel):
    """Response model for the models endpoint."""

    models: List[ModelInfo]


class ProviderStatus(BaseModel):
    """
    Connection health and credential status for an LLM provider.

    Attributes
    ----------
        provider: Provider identifier name.
        configured: Whether API credentials are present in the environment.
        healthy: Whether provider connection is reachable/healthy.
        details: Optional diagnostic details or error message.
        reachable: Whether provider endpoint responded successfully to health check.
        error_code: Diagnostic status code ('PROVIDER_ONLINE', 'PROVIDER_OFFLINE', 'MISSING_API_KEY').

    """

    provider: str
    configured: bool
    healthy: bool
    details: Optional[str] = None
    reachable: bool = True
    error_code: Optional[str] = None


class ProvidersStatusResponse(BaseModel):
    """
    Response model for provider statuses.

    Attributes
    ----------
        providers: List of provider status objects.

    """

    providers: List[ProviderStatus]


CLOUD_PROVIDER_MODELS: Dict[str, List[str]] = {
    "openai": ["openai/gpt-4o", "openai/gpt-4o-mini"],
    "anthropic": ["anthropic/claude-3-5-sonnet", "anthropic/claude-3-5-haiku"],
    "google": ["google/gemini-1.5-pro", "google/gemini-1.5-flash"],
}


def is_provider_configured(provider: str) -> bool:
    """
    Check if the specified LLM provider has credentials configured in the environment.

    Args:
    ----
        provider: Provider identifier string.

    Returns:
    -------
        bool: True if configured, False otherwise.

    """
    prov = provider.lower()
    if prov == "ollama":
        return True
    if prov == "openai":
        return bool(os.environ.get("OPENAI_API_KEY"))
    if prov == "anthropic":
        return bool(os.environ.get("ANTHROPIC_API_KEY"))
    if prov in ("google", "gemini"):
        return bool(
            os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        )
    return False


def execute_sql(
    db_path: str,
    query: str,
    max_rows: int = 10000,
    timeout_seconds: float = 30.0,
) -> List[Dict[str, SqlValue]]:
    """
    Execute a SQL query and return results as a list of dictionaries.

    Args:
    ----
        db_path: Path to the DuckDB database file.
        query: SQL query to execute.
        max_rows: Maximum rows to return to prevent memory exhaustion.
        timeout_seconds: Execution timeout in seconds.

    Returns:
    -------
        List of dictionaries representing the rows returned.

    Raises:
    ------
        ValueError: If there is an error executing the query.

    """
    import threading

    logger.info(f"Executing SQL query against database at {db_path}:\n{query}")
    conn = None
    try:
        conn = duckdb.connect(db_path, read_only=True)
        conn.execute("SET enable_external_access = false")
        timer = threading.Timer(timeout_seconds, conn.interrupt)
        timer.start()
        try:
            result = conn.execute(query)
            columns = (
                [desc[0] for desc in result.description] if result.description else []
            )
            rows = result.fetchmany(max_rows)
        finally:
            timer.cancel()

        output = [
            {col: sanitize_sql_value(val) for col, val in zip(columns, row)}
            for row in rows
        ]
        logger.info(f"SQL query executed successfully. Returned {len(output)} rows.")
        return output
    except duckdb.InterruptException as ie:
        logger.error(f"SQL execution timed out: {ie}")
        raise ValueError(
            json.dumps(
                {
                    "error_code": "backend.queryTimeout",
                    "params": {"error": "Query execution timed out."},
                }
            )
        )
    except Exception as e:
        logger.error(f"SQL execution error: {e}")
        err_msg = str(e)
        if (
            "database does not exist" in err_msg.lower()
            or "cannot open" in err_msg.lower()
            or "no such file" in err_msg.lower()
        ):
            err_code = "backend.dbNotFound"
        else:
            err_code = "backend.sqlExecution"
        raise ValueError(
            json.dumps({"error_code": err_code, "params": {"error": err_msg}})
        )
    finally:
        if conn is not None:
            conn.close()


def _get_sessions_db_conn(
    db_path: Optional[str] = None,
) -> duckdb.DuckDBPyConnection:
    """
    Open connection to persistent chat sessions database and ensure schema exists.

    Args:
    ----
        db_path: Optional path to the database file.

    Returns:
    -------
        duckdb.DuckDBPyConnection: Database connection.

    """
    if db_path is None:
        env_db = os.environ.get("T1D_SESSIONS_DB")
        if env_db:
            db_path = env_db
        else:
            sessions_dir = Path("./data")
            sessions_dir.mkdir(parents=True, exist_ok=True)
            db_path = str((sessions_dir / "sessions.duckdb").resolve())
    max_retries = 5
    conn: Optional[duckdb.DuckDBPyConnection] = None
    for attempt in range(max_retries - 1):
        try:
            conn = duckdb.connect(db_path, read_only=False)
            break
        except duckdb.IOException as e:
            if "lock" in str(e).lower():
                time.sleep(0.05 * (attempt + 1))
            else:
                raise
    else:
        conn = duckdb.connect(db_path, read_only=False)

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS chat_sessions (
            session_id VARCHAR PRIMARY KEY,
            title VARCHAR,
            created_at TIMESTAMP,
            updated_at TIMESTAMP,
            messages_json VARCHAR
        )
        """
    )
    return conn


@contextmanager
def get_sessions_conn(
    db_path: Optional[str] = None,
) -> Iterator[duckdb.DuckDBPyConnection]:
    """
    Context manager yielding a persistent chat sessions database connection and ensuring it closes.

    Args:
    ----
        db_path: Optional path to the sessions database.

    Yields:
    ------
        duckdb.DuckDBPyConnection: Database connection guaranteed to be closed upon exit.

    """
    conn = (
        _get_sessions_db_conn(db_path)
        if db_path is not None
        else _get_sessions_db_conn()
    )
    try:
        yield conn
    finally:
        conn.close()


def _build_sql_prompt(
    schema: str,
    nl_query: str,
    history: Optional[List[ChatMessageModel]] = None,
) -> str:
    """
    Construct the clinical Text-to-SQL system prompt for an LLM.

    Args:
    ----
        schema: The database schema text.
        nl_query: The clinician or user query.
        history: Optional list of prior conversation messages for multi-turn context.

    Returns:
    -------
        str: Formatted system and user prompt.

    """
    history_context = ""
    if history:
        turns: List[str] = []
        for msg in history[-6:]:
            role = "User" if msg.role == "user" else "Assistant"
            content = msg.content.strip()
            sql_info = f" (SQL: {msg.sqlQuery})" if msg.sqlQuery else ""
            turns.append(f"{role}: {content}{sql_info}")
        history_context = "Prior Conversation History:\n" + "\n".join(turns) + "\n\n"

    return f"""You are a DuckDB SQL expert. Given the following database schema for Type 1 Diabetes (T1D) clinical trial datasets:

{schema}

Context & Rules for T1D Analytics:
1. Terminology & Aliases:
   - "Lows" / "Hypoglycemia": Look for Time Below Range (TBR, <70 or <54 mg/dL), `pct_time_under_70`, `hypo`.
   - "Highs" / "Hyperglycemia": Look for Time Above Range (TAR, >180 or >250 mg/dL), `pct_time_over_180`, `hyper`.
   - "In-range" / "TIR": Look for Time in Range (70-180 mg/dL), `pct_time_in_range_70_180`, `tir`, `cgm_in_range`.
   - "A1C" / "HbA1c": Look for glycated hemoglobin, `a1c`, `hba1c`.
   - "Variability": Look for Coefficient of Variation (`cv`), Standard Deviation (`sd`).
   - "Average sugar": Look for Mean Glucose, `mean_cgm`, Glucose Management Indicator (`gmi`).
   - "Adverse Events": "DKA" (Diabetic Ketoacidosis), "SH" (Severe Hypoglycemia).
   - Demographics: "Duration" (years with diabetes), "Age", "BMI", "Gender/Sex", "Weight".
   - Treatment: "Pump" (CSII), "Injections/Shots" (MDI), "Loop" (AID/Closed-Loop), "Insulin" (TDD, Basal, Bolus).
   - Study Timepoints: "Baseline" (visit 0 or baseline flag), "End of trial/Follow-up" (max visit month).
2. SQL Generation Rules for Multi-Trial Data:
   - Trial Names: Acronyms (DCLP3, DCLP5, Pedap, etc.) usually correspond directly to table names.
   - Aggregating Combined Trials: To calculate metrics for multiple trials "together", "combined", or "overall", you MUST use a CTE with `UNION ALL` to combine raw rows from all requested tables *before* applying aggregate functions like `AVG()`.
   - Comparing Individual Trials: To calculate metrics for multiple trials "individually" or "each", use `UNION ALL` with an identifying literal string column (e.g., `SELECT 'DCLP3' as trial, AVG(...) ... UNION ALL SELECT 'DCLP5'...`).
   - Aligning UNION Columns: Clinical trial tables often have different schemas. When using `UNION ALL`, you MUST explicitly `SELECT` and alias only the specific columns needed (e.g. `age`, `tir`) to ensure they align perfectly in number, order, and type across all blocks. NEVER use `SELECT *` with `UNION ALL`.
   - Linking Data: If questions span patient demographics and CGM outcomes, use `JOIN` on the patient/subject ID columns (e.g., `pt_id`, `subject_id`).
3. Formatting & Logic:
   - Handle colloquial/informal language intuitively (e.g., "betwixt" -> `BETWEEN`, "kids/pediatric" -> `age < 18`, "adults" -> `age >= 18`).
   - Account for string vs. numeric data types; use `CAST(col AS FLOAT)` if computing averages on textual columns that contain numbers.
   - Ignore NULL values appropriately when calculating averages or sums.
4. Strict Schema Enforcement:
   - CRITICAL: You must ONLY use the exact table names and column names provided in the schema above.
   - Do NOT invent, guess, or hallucinate table names (e.g., do not use "Visit1" or "demographics" if they are not in the schema).
   - If a requested concept (like "demographics") is not a table, find the relevant columns within the existing tables (e.g., `visits` or `patients`).

Write a SQL query that answers the user's request.
You may provide an explanation of your thought process before or after the query.
Ensure the SQL query is enclosed in a markdown code block (e.g. ```sql ... ```).
The query should be a valid DuckDB SQL SELECT statement.

{history_context}User request: {nl_query}"""


def _extract_sql_from_response(full_response: str) -> str:
    """
    Extract SQL query from markdown code blocks or plain text in an LLM completion.

    Args:
    ----
        full_response: Raw text response from the LLM.

    Returns:
    -------
        str: Extracted SQL query string.

    """
    return extract_sql_from_response(full_response)


def extract_stream_delta(chunk: Any, provider: str = "ollama") -> str:
    """
    Extract token delta string from provider-specific streaming response chunk.

    Accurately inspects delta objects across ollama, openai, anthropic, and google.

    Args:
    ----
        chunk: Raw response chunk from any-llm or provider SDK.
        provider: Provider identifier ('ollama', 'openai', 'anthropic', 'google').

    Returns:
    -------
        str: Extracted token content string, or empty string.

    """
    if isinstance(chunk, str):
        return chunk
    if isinstance(chunk, dict):
        for choice in chunk.get("choices") or []:
            if isinstance(choice, dict):
                delta = choice.get("delta")
                if isinstance(delta, dict) and delta.get("content"):
                    return str(delta["content"])
                if choice.get("text"):
                    return str(choice["text"])
        msg = chunk.get("message")
        if isinstance(msg, dict) and msg.get("content"):
            return str(msg["content"])
        if chunk.get("response"):
            return str(chunk["response"])
        if chunk.get("text"):
            return str(chunk["text"])
        return ""

    prov = provider.lower()
    if prov in ("google", "gemini"):
        for cand in getattr(chunk, "candidates", None) or []:
            content = getattr(cand, "content", None)
            for part in getattr(content, "parts", None) or []:
                text_val = getattr(part, "text", None)
                if isinstance(text_val, str) and text_val:
                    return text_val
        text_attr = getattr(chunk, "text", None)
        if isinstance(text_attr, str) and text_attr:
            return text_attr
        return ""

    if prov == "anthropic":
        delta_obj = getattr(chunk, "delta", None)
        if delta_obj is not None:
            text_val = getattr(delta_obj, "text", None) or getattr(
                delta_obj, "content", None
            )
            if isinstance(text_val, str) and text_val:
                return text_val
        for c in getattr(chunk, "choices", None) or []:
            c_delta = getattr(c, "delta", None)
            text_val = getattr(c_delta, "text", None)
            if isinstance(text_val, str) and text_val:
                return text_val
        return ""

    for c in getattr(chunk, "choices", None) or []:
        c_delta = getattr(c, "delta", None)
        if c_delta is not None:
            text_val = getattr(c_delta, "content", None) or getattr(
                c_delta, "text", None
            )
            if isinstance(text_val, str) and text_val:
                return text_val
        c_msg = getattr(c, "message", None)
        if c_msg is not None:
            text_val = getattr(c_msg, "content", None)
            if isinstance(text_val, str) and text_val:
                return text_val
        text_val = getattr(c, "text", None)
        if isinstance(text_val, str) and text_val:
            return text_val

    direct_text = getattr(chunk, "text", None)
    if isinstance(direct_text, str) and direct_text:
        return direct_text

    return ""


def stream_llm_tokens(
    prompt: str,
    model_name: str,
    api_key: Optional[str] = None,
    provider: Optional[str] = None,
) -> Iterator[str]:
    """
    Stream tokens from LLM using any-llm, falling back gracefully to single labeled completion block.

    Args:
    ----
        prompt: The full prompt string.
        model_name: The LLM model name or provider/model string.
        api_key: Optional API key for external cloud provider.
        provider: Optional provider name override.

    Yields:
    ------
        str: Streamed token chunks.

    """
    from any_llm import AnyLLM  # type: ignore

    resolved_provider = provider or "ollama"
    actual_model = model_name
    if "/" in model_name and not provider:
        resolved_provider, actual_model = model_name.split("/", 1)

    llm_kwargs: Dict[str, Any] = {}
    if api_key:
        llm_kwargs["api_key"] = api_key

    llm = AnyLLM.create(resolved_provider, **llm_kwargs)
    try:
        stream_resp = llm.completion(
            model=actual_model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )
        if hasattr(stream_resp, "__iter__"):
            for chunk in stream_resp:
                delta = extract_stream_delta(chunk, resolved_provider)
                if delta:
                    yield delta
            return
    except Exception as e:
        logger.warning(
            f"Native streaming failed for provider '{resolved_provider}': {e}. Falling back to non-streaming completion."
        )

    # Single labeled completion block fallback (no synthetic space splitting)
    response = llm.completion(
        model=actual_model,
        messages=[{"role": "user", "content": prompt}],
    )
    full_text = extract_stream_delta(response, resolved_provider) or str(response)
    yield full_text


def generate_sql_from_nl(
    db_path: str,
    nl_query: str,
    model_name: str = "gemma4",
    history: Optional[List[ChatMessageModel]] = None,
    api_key: Optional[str] = None,
    provider: Optional[str] = None,
) -> tuple[str, str]:
    """
    Translate natural language to SQL using any-llm.

    Args:
    ----
        db_path: Path to the DuckDB database file.
        nl_query: Natural language question.
        model_name: The model to use or provider/model (e.g. 'gemma4', 'openai/gpt-4o').
        history: Optional list of prior conversation messages for multi-turn context.
        api_key: Optional API key for cloud provider.
        provider: Optional provider override name.

    Returns:
    -------
        The generated SQL query string.

    Raises:
    ------
        RuntimeError: If any-llm-sdk is missing, API key is missing, or LLM API fails.
        ValueError: If reading schema fails.

    """
    resolved_provider = provider or "ollama"
    actual_model = model_name
    if "/" in model_name and not provider:
        resolved_provider, actual_model = model_name.split("/", 1)

    logger.info(
        f"Translating natural language to SQL: '{nl_query}' using provider '{resolved_provider}' and model '{actual_model}'"
    )
    try:
        from any_llm import AnyLLM
    except ImportError:
        logger.error(f"any-llm-sdk[{resolved_provider}] is not installed.")
        raise RuntimeError(
            json.dumps(
                {
                    "error_code": "backend.missingSdk",
                    "params": {"error": f"any-llm-sdk[{resolved_provider}]"},
                }
            )
        )

    env_keys = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "google": "GEMINI_API_KEY",
        "gemini": "GEMINI_API_KEY",
    }
    if api_key and resolved_provider.lower() in env_keys:
        os.environ.setdefault(env_keys[resolved_provider.lower()], api_key)

    if resolved_provider.lower() in env_keys:
        env_var = env_keys[resolved_provider.lower()]
        if not os.environ.get(env_var):
            raise RuntimeError(
                json.dumps(
                    {
                        "error_code": "backend.missingApiKey",
                        "params": {"provider": resolved_provider, "env_var": env_var},
                    }
                )
            )

    try:
        conn = duckdb.connect(db_path, read_only=True)
        schema = get_database_schema(conn)
        conn.close()
        logger.debug(f"Retrieved schema: {schema[:200]}... (truncated)")
    except Exception as e:
        logger.error(f"Failed to read schema: {e}")
        err_msg = str(e)
        if (
            "database does not exist" in err_msg.lower()
            or "cannot open" in err_msg.lower()
            or "no such file" in err_msg.lower()
        ):
            err_code = "backend.dbNotFound"
        else:
            err_code = "backend.readSchemaFailed"
        raise ValueError(
            json.dumps({"error_code": err_code, "params": {"error": err_msg}})
        )

    prompt = _build_sql_prompt(schema, nl_query, history=history)

    try:
        logger.info(f"Sending prompt to LLM ({resolved_provider}/{actual_model})...")
        llm_kwargs: Dict[str, Any] = {}
        if api_key:
            llm_kwargs["api_key"] = api_key
        llm = AnyLLM.create(resolved_provider, **llm_kwargs)
        response = llm.completion(
            model=actual_model,
            messages=[{"role": "user", "content": prompt}],
        )
        full_response = response.choices[0].message.content.strip()
        logger.info(f"LLM response:\n{full_response}")
        sql_query = _extract_sql_from_response(full_response)
        return full_response, sql_query
    except Exception as e:
        logger.error(f"LLM translation error: {e}")
        raise RuntimeError(
            json.dumps(
                {
                    "error_code": "backend.llmTranslationError",
                    "params": {"error": str(e)},
                }
            )
        )


@app.post("/api/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest, req: Request) -> ChatResponse:
    """
    Handle incoming chat messages and execute SQL or NLP queries.

    Args:
    ----
        request: The incoming ChatRequest.
        req: The incoming FastAPI HTTP request.

    Returns:
    -------
        A ChatResponse indicating success or error.

    Raises:
    ------
        HTTPException: If request parameters are invalid.

    """
    enforce_rate_limit(req)
    provider_hdr = req.headers.get("x-provider")
    api_key_hdr = req.headers.get("x-provider-api-key") or req.headers.get("x-api-key")
    if provider_hdr and not request.provider:
        request.provider = provider_hdr
    if api_key_hdr and not request.api_key:
        request.api_key = api_key_hdr

    if not request.message.strip():
        logger.warning("Received empty message.")
        raise HTTPException(
            status_code=400, detail={"error_code": "backend.emptyMessage", "params": {}}
        )

    try:
        logger.info(
            f"Received chat request: model='{request.model}', message='{request.message}'"
        )
        db_path = validate_db_path(request.db_path)
        if request.model == "sql":
            sql_query = request.message
            content_prefix = "backend.literalSql"
            results = execute_sql(db_path, sql_query)
            return ChatResponse(
                content=content_prefix, sqlResult=results, sqlQuery=sql_query
            )
        else:
            full_response, sql_query = generate_sql_from_nl(
                db_path,
                request.message,
                model_name=request.model,
                history=request.history,
                api_key=request.api_key,
                provider=request.provider,
            )
            results = None
            sql_error = None
            if sql_query:
                try:
                    results = execute_sql(db_path, sql_query)
                except Exception as e:
                    sql_error = parse_error(e)

            return ChatResponse(
                content=full_response,
                sqlQuery=sql_query,
                sqlResult=results,
                error=sql_error,
            )
    except ValueError as ve:
        logger.error(f"Database error during chat request: {ve}")
        return ChatResponse(content="backend.errorDbExecution", error=parse_error(ve))
    except RuntimeError as re:
        logger.error(f"NLP error during chat request: {re}")
        return ChatResponse(
            content="backend.errorNlpTranslation", error=parse_error(re)
        )
    except Exception as e:
        logger.exception("Unexpected error during chat request.")
        return ChatResponse(content="backend.errorUnexpected", error=parse_error(e))


async def stream_chat_events(
    request: ChatRequest,
    req: Optional[Request] = None,
) -> AsyncIterator[str]:
    """
    Generate Server-Sent Events for streaming chat queries and SQL results.

    Args:
    ----
        request: The incoming ChatRequest.
        req: Optional FastAPI Request object to detect client disconnection.

    Yields:
    ------
        SSE data lines formatted as JSON strings.

    """
    if req is not None and await req.is_disconnected():
        logger.info("Client disconnected before stream generation.")
        return

    if not request.message.strip():
        payload = json.dumps({"event": "error", "error": "backend.emptyMessage"})
        yield f"data: {payload}\n\n"
        return

    try:
        db_path = validate_db_path(request.db_path)
    except HTTPException:
        payload = json.dumps({"event": "error", "error": "backend.invalidDatabasePath"})
        yield f"data: {payload}\n\n"
        return

    if request.model == "sql":
        exec_payload = json.dumps(
            {"event": "token", "token": "Executing SQL query...\n"}
        )
        yield f"data: {exec_payload}\n\n"
        yield f"data: {json.dumps({'event': 'execution_progress', 'status': 'Executing SQL query...'})}\n\n"
        try:
            results = execute_sql(db_path, request.message)
            res_dict = {
                "content": "backend.literalSql",
                "sqlQuery": request.message,
                "sqlResult": results,
                "error": None,
            }
            yield f"data: {json.dumps({'event': 'done', **res_dict})}\n\n"
            yield f"data: {json.dumps({'event': 'result', **res_dict})}\n\n"
        except Exception as e:
            err_dict = parse_error(e).model_dump()
            err_dict_payload = {
                "content": "backend.errorDbExecution",
                "sqlQuery": request.message,
                "sqlResult": None,
                "error": err_dict,
            }
            yield f"data: {json.dumps({'event': 'done', **err_dict_payload})}\n\n"
            yield f"data: {json.dumps({'event': 'result', **err_dict_payload})}\n\n"
        return

    try:
        conn = duckdb.connect(db_path, read_only=True)
        schema = get_database_schema(conn)
        conn.close()

        prompt = _build_sql_prompt(schema, request.message, history=request.history)
        full_parts: List[str] = []
        for token in stream_llm_tokens(
            prompt,
            request.model,
            api_key=request.api_key,
            provider=request.provider,
        ):
            if req is not None and await req.is_disconnected():
                logger.info("Client disconnected during SSE token streaming.")
                return
            full_parts.append(token)
            tok_payload = json.dumps({"event": "token", "token": token})
            yield f"data: {tok_payload}\n\n"
        full_response = "".join(full_parts)
        sql_query = _extract_sql_from_response(full_response)

        sql_results: Optional[List[Dict[str, SqlValue]]] = None
        sql_err_dict = None
        if sql_query:
            yield f"data: {json.dumps({'event': 'sql_generated', 'sqlQuery': sql_query})}\n\n"
            yield f"data: {json.dumps({'event': 'execution_progress', 'status': 'Executing SQL query...'})}\n\n"
            try:
                sql_results = execute_sql(db_path, sql_query)
            except Exception as e:
                sql_err_dict = parse_error(e).model_dump()

        nl_res_dict: Dict[str, Any] = {
            "content": full_response,
            "sqlQuery": sql_query,
            "sqlResult": sql_results,
            "error": sql_err_dict,
        }
        yield f"data: {json.dumps({'event': 'done', **nl_res_dict})}\n\n"
        yield f"data: {json.dumps({'event': 'result', **nl_res_dict})}\n\n"
    except Exception as e:
        err_dict = parse_error(e).model_dump()
        nl_err_payload: Dict[str, Any] = {
            "content": "backend.errorNlpTranslation",
            "sqlQuery": None,
            "sqlResult": None,
            "error": err_dict,
        }
        yield f"data: {json.dumps({'event': 'done', **nl_err_payload})}\n\n"
        yield f"data: {json.dumps({'event': 'result', **nl_err_payload})}\n\n"


@app.post("/api/chat/stream")
async def chat_stream_endpoint(request: ChatRequest, req: Request) -> StreamingResponse:
    """
    Handle chat requests with Server-Sent Events (SSE) streaming.

    Args:
    ----
        request: The incoming ChatRequest.
        req: The incoming FastAPI HTTP request.

    Returns:
    -------
        StreamingResponse streaming SSE token and result events.

    """
    enforce_rate_limit(req)
    provider_hdr = req.headers.get("x-provider")
    api_key_hdr = req.headers.get("x-provider-api-key") or req.headers.get("x-api-key")
    if provider_hdr and not request.provider:
        request.provider = provider_hdr
    if api_key_hdr and not request.api_key:
        request.api_key = api_key_hdr

    return StreamingResponse(
        stream_chat_events(request, req=req),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/sessions", response_model=SessionListResponse)
def list_sessions(db_path: Optional[str] = None) -> SessionListResponse:
    """
    List all persistent conversation sessions.

    Args:
    ----
        db_path: Optional path to the sessions database.

    Returns:
    -------
        SessionListResponse: List of stored conversation sessions sorted by updated_at.

    """
    with _sessions_lock:
        with get_sessions_conn(db_path) as conn:
            rows = conn.execute(
                "SELECT session_id, title, strftime(created_at, '%Y-%m-%d %H:%M:%S'), strftime(updated_at, '%Y-%m-%d %H:%M:%S'), messages_json FROM chat_sessions ORDER BY updated_at DESC"
            ).fetchall()

    sessions: List[SessionModel] = []
    for r in rows:
        try:
            msgs_raw = json.loads(r[4])
            msgs = [ChatMessageModel(**m) for m in msgs_raw]
        except Exception:
            msgs = []
        sessions.append(
            SessionModel(
                session_id=r[0],
                title=r[1],
                created_at=str(r[2]),
                updated_at=str(r[3]),
                messages=msgs,
            )
        )
    return SessionListResponse(sessions=sessions)


@app.post("/api/sessions", response_model=SessionModel)
def save_session(
    request: CreateSessionRequest, db_path: Optional[str] = None
) -> SessionModel:
    """
    Create or update a persistent conversation session.

    Args:
    ----
        request: The session creation or update payload.
        db_path: Optional path to the sessions database.

    Returns:
    -------
        SessionModel: The saved conversation session.

    """
    session_id = request.session_id or f"session-{uuid.uuid4()}"
    title = request.title
    if not title:
        if request.messages:
            title = request.messages[0].content[:40]
        else:
            title = "New Chat"

    messages_json = json.dumps([m.model_dump() for m in request.messages])
    with _sessions_lock:
        with get_sessions_conn(db_path) as conn:
            exists = conn.execute(
                "SELECT 1 FROM chat_sessions WHERE session_id = ?", [session_id]
            ).fetchone()
            if exists:
                conn.execute(
                    "UPDATE chat_sessions SET title = ?, updated_at = now(), messages_json = ? WHERE session_id = ?",
                    [title, messages_json, session_id],
                )
            else:
                conn.execute(
                    "INSERT INTO chat_sessions VALUES (?, ?, now(), now(), ?)",
                    [session_id, title, messages_json],
                )
            row = conn.execute(
                "SELECT strftime(created_at, '%Y-%m-%d %H:%M:%S'), strftime(updated_at, '%Y-%m-%d %H:%M:%S') FROM chat_sessions WHERE session_id = ?",
                [session_id],
            ).fetchone()

    created_at = str(row[0]) if row else ""
    updated_at = str(row[1]) if row else ""

    return SessionModel(
        session_id=session_id,
        title=title,
        created_at=created_at,
        updated_at=updated_at,
        messages=request.messages,
    )


@app.get("/api/sessions/{session_id}", response_model=SessionModel)
def get_session(session_id: str, db_path: Optional[str] = None) -> SessionModel:
    """
    Retrieve a specific conversation session by ID.

    Args:
    ----
        session_id: The ID of the session to retrieve.
        db_path: Optional path to the sessions database.

    Returns:
    -------
        SessionModel: The found conversation session.

    Raises:
    ------
        HTTPException: 404 if session not found.

    """
    with _sessions_lock:
        with get_sessions_conn(db_path) as conn:
            row = conn.execute(
                "SELECT session_id, title, strftime(created_at, '%Y-%m-%d %H:%M:%S'), strftime(updated_at, '%Y-%m-%d %H:%M:%S'), messages_json FROM chat_sessions WHERE session_id = ?",
                [session_id],
            ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Session not found")

    try:
        msgs_raw = json.loads(row[4])
        msgs = [ChatMessageModel(**m) for m in msgs_raw]
    except Exception:
        msgs = []

    return SessionModel(
        session_id=row[0],
        title=row[1],
        created_at=str(row[2]),
        updated_at=str(row[3]),
        messages=msgs,
    )


@app.delete("/api/sessions/{session_id}")
def delete_session(session_id: str, db_path: Optional[str] = None) -> Dict[str, str]:
    """
    Delete a conversation session.

    Args:
    ----
        session_id: The ID of the session to delete.
        db_path: Optional path to the sessions database.

    Returns:
    -------
        Dict[str, str]: Deletion confirmation message.

    Raises:
    ------
        HTTPException: 404 if session not found.

    """
    with _sessions_lock:
        with get_sessions_conn(db_path) as conn:
            exists = conn.execute(
                "SELECT 1 FROM chat_sessions WHERE session_id = ?", [session_id]
            ).fetchone()
            if not exists:
                raise HTTPException(status_code=404, detail="Session not found")
            conn.execute("DELETE FROM chat_sessions WHERE session_id = ?", [session_id])
    return {"status": "deleted", "session_id": session_id}


class ExecuteSqlRequest(BaseModel):
    """Request model for the execute SQL endpoint."""

    query: str
    db_path: Optional[str] = None


class ExecuteSqlResponse(BaseModel):
    """Response model for the execute SQL endpoint."""

    sqlResult: Optional[List[Dict[str, SqlValue]]] = None
    error: Optional[ApiError] = None


def parse_error(e: Exception) -> ApiError:
    """
    Parse an exception into a structured ApiError.

    Args:
    ----
        e (Exception): The exception to parse.

    Returns:
    -------
        ApiError: The resulting ApiError object.

    """
    try:
        data = json.loads(str(e))
        return ApiError(**data)
    except Exception:
        return ApiError(error_code="backend.errorUnexpected", params={"error": str(e)})


@app.post("/api/execute-sql", response_model=ExecuteSqlResponse)
def execute_sql_endpoint(
    request: ExecuteSqlRequest, req: Request
) -> ExecuteSqlResponse:
    """
    Execute an arbitrary SQL query against the database.

    Args:
    ----
        request: The request containing the query and optional db_path.
        req: The incoming FastAPI HTTP request.

    Returns:
    -------
        The execution response containing results or error.

    """
    enforce_rate_limit(req)
    if not request.query.strip():
        return ExecuteSqlResponse(
            error=ApiError(error_code="backend.emptyMessage", params={})
        )

    try:
        db_path = validate_db_path(request.db_path)
        results = execute_sql(db_path, request.query)
        return ExecuteSqlResponse(sqlResult=results)
    except HTTPException:
        raise
    except ValueError as ve:
        logger.error(f"Database error during sql execution request: {ve}")
        return ExecuteSqlResponse(error=parse_error(ve))
    except Exception as e:
        logger.exception("Unexpected error during sql execution request.")
        return ExecuteSqlResponse(error=parse_error(e))


class TableDataResponse(BaseModel):
    """Response model for table data."""

    rows: List[Dict[str, SqlValue]]
    total_count: int = 0
    page: int = 1
    total_pages: int = 1


@app.get("/api/table/{table_name}", response_model=TableDataResponse)
def get_table_data(
    table_name: str,
    limit: int = 25,
    offset: int = 0,
    sort_by: Optional[str] = None,
    order: Optional[str] = "asc",
    search: Optional[str] = None,
    db_path: Optional[str] = None,
) -> TableDataResponse:
    """
    Return paginated rows from a specific table with optional sorting and searching.

    Args:
    ----
        table_name: The table name.
        limit: The limit.
        offset: The offset.
        sort_by: Optional column name to sort by.
        order: Sort order, either 'asc' or 'desc' (default: 'asc').
        search: Optional string to filter across columns.
        db_path: Optional path to the DuckDB database.

    Returns:
    -------
        The data with pagination metadata.

    Raises:
    ------
        HTTPException: on error.

    """
    target_db = validate_db_path(db_path)
    if not Path(target_db).exists():
        raise HTTPException(
            status_code=404,
            detail={"error_code": "backend.dbNotFound", "params": {"path": target_db}},
        )

    # Validate table name to prevent SQL injection
    if not table_name.isidentifier():
        raise HTTPException(
            status_code=400, detail={"error_code": "backend.invalidTable", "params": {}}
        )

    if limit <= 0 or offset < 0:
        raise HTTPException(
            status_code=400,
            detail={
                "error_code": "backend.invalidPagination",
                "params": {"limit": limit, "offset": offset},
            },
        )

    sort_order_clean = (order or "asc").strip().lower()
    if sort_order_clean not in ("asc", "desc"):
        raise HTTPException(
            status_code=400,
            detail={
                "error_code": "backend.invalidSortOrder",
                "params": {"order": order},
            },
        )

    conn = None
    try:
        conn = duckdb.connect(target_db, read_only=True)
        # Check if table exists
        tables = [row[0] for row in conn.execute("SHOW TABLES").fetchall()]
        if table_name not in tables:
            raise HTTPException(
                status_code=404,
                detail={"error_code": "backend.tableNotFound", "params": {}},
            )

        escaped_table = table_name.replace('"', '""')
        # Inspect columns for validation and search filtering
        cols_info = conn.execute(f'DESCRIBE "{escaped_table}"').fetchall()
        column_names = [c[0] for c in cols_info]

        if sort_by is not None and sort_by != "":
            if sort_by not in column_names:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error_code": "backend.invalidSortColumn",
                        "params": {"sort_by": sort_by},
                    },
                )

        # Build WHERE clause for search if provided
        where_clause = ""
        params: List[Any] = []
        if search is not None and search.strip():
            search_pattern = f"%{search.strip()}%"
            or_conditions = []
            for col in column_names:
                escaped_col = col.replace('"', '""')
                or_conditions.append(f'CAST("{escaped_col}" AS VARCHAR) ILIKE ?')
                params.append(search_pattern)
            where_clause = f"WHERE ({' OR '.join(or_conditions)})"

        # Limit to reasonable maximum
        limit = min(limit, 1000)

        count_query = f'SELECT COUNT(*) FROM "{escaped_table}" {where_clause}'.strip()
        count_result = conn.execute(count_query, params).fetchone()
        total_count = count_result[0] if count_result else 0
        total_pages = max(1, math.ceil(total_count / limit)) if limit > 0 else 1
        page = (offset // limit) + 1 if limit > 0 else 1

        # Enforce deterministic ordering
        if sort_by is not None and sort_by != "":
            escaped_sort = sort_by.replace('"', '""')
            order_by_clause = f'ORDER BY "{escaped_sort}" {sort_order_clean.upper()}'
        else:
            escaped_first = column_names[0].replace('"', '""')
            order_by_clause = f'ORDER BY "{escaped_first}" ASC'

        query = f'SELECT * FROM "{escaped_table}" {where_clause} {order_by_clause} LIMIT {limit} OFFSET {offset}'.strip()
        result = conn.execute(query, params)
        columns = [desc[0] for desc in result.description] if result.description else []
        rows = result.fetchall()
        output = [
            {col: sanitize_sql_value(val) for col, val in zip(columns, row)}
            for row in rows
        ]
        return TableDataResponse(
            rows=output, total_count=total_count, page=page, total_pages=total_pages
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching table data: {e}")
        err_msg = str(e).lower()
        if (
            "does not exist" in err_msg
            or "cannot open" in err_msg
            or "no such file" in err_msg
        ):
            raise HTTPException(
                status_code=404,
                detail={
                    "error_code": "backend.dbNotFound",
                    "params": {"error": str(e)},
                },
            )
        raise HTTPException(
            status_code=500,
            detail={"error_code": "backend.serverError", "params": {"error": str(e)}},
        )
    finally:
        if conn is not None:
            conn.close()


@app.get("/api/table/{table_name}/profile")
def get_table_profile(table_name: str, db_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Profile a table to return data quality, completeness, and column statistics.

    Args:
    ----
        table_name: The table name to profile.
        db_path: Optional path to the database file.

    Returns:
    -------
        Dict[str, Any]: Profiling summary dictionary.

    Raises:
    ------
        HTTPException: 400 for invalid table, 404 if db or table not found, 500 otherwise.

    """
    target_db = validate_db_path(db_path)
    if not Path(target_db).exists():
        raise HTTPException(
            status_code=404,
            detail={"error_code": "backend.dbNotFound", "params": {"path": target_db}},
        )

    if not table_name.isidentifier():
        raise HTTPException(
            status_code=400,
            detail={"error_code": "backend.invalidTable", "params": {}},
        )

    try:
        return profile_table(target_db, table_name)
    except ValueError as ve:
        raise HTTPException(
            status_code=404,
            detail={
                "error_code": "backend.tableNotFound",
                "params": {"error": str(ve)},
            },
        )
    except FileNotFoundError as fnf:
        raise HTTPException(
            status_code=404,
            detail={"error_code": "backend.dbNotFound", "params": {"error": str(fnf)}},
        )
    except Exception as e:
        logger.error(f"Error profiling table {table_name}: {e}")
        raise HTTPException(
            status_code=500,
            detail={"error_code": "backend.serverError", "params": {"error": str(e)}},
        )


@app.get("/api/export/excel")
def export_excel(table_name: str, db_path: Optional[str] = None) -> StreamingResponse:
    """
    Export table data formatted as a native OpenXML Excel (.xlsx) workbook.

    Args:
    ----
        table_name: Table to export.
        db_path: Optional path to database file.

    Returns:
    -------
        StreamingResponse: Streamed XLSX file with spreadsheetml MIME type.

    Raises:
    ------
        HTTPException: If table or DB is invalid.

    """
    target_db = validate_db_path(db_path)
    if not Path(target_db).exists():
        raise HTTPException(
            status_code=404,
            detail={"error_code": "backend.dbNotFound", "params": {"path": target_db}},
        )

    if not table_name.isidentifier():
        raise HTTPException(
            status_code=400,
            detail={"error_code": "backend.invalidTable", "params": {}},
        )

    conn = None
    try:
        conn = duckdb.connect(target_db, read_only=True)
        tables = [row[0] for row in conn.execute("SHOW TABLES").fetchall()]
        if table_name not in tables:
            conn.close()
            raise HTTPException(
                status_code=404,
                detail={"error_code": "backend.tableNotFound", "params": {}},
            )

        escaped_table = table_name.replace('"', '""')
        result = conn.execute(f'SELECT * FROM "{escaped_table}"')
        columns = [desc[0] for desc in result.description] if result.description else []

        def _generate_excel_chunks() -> Iterator[bytes]:
            """
            Stream table data formatted as native OpenXML workbook in chunks.

            Yields
            ------
                bytes: Binary XLSX chunk bytes.

            """
            try:
                from openpyxl import Workbook

                wb = Workbook(write_only=True)
                ws = wb.create_sheet(title=table_name[:31])
                ws.append(columns)

                while True:
                    batch = result.fetchmany(5000)
                    if not batch:
                        break
                    for r in batch:
                        formatted_row: List[Any] = []
                        for val in r:
                            if val is None:
                                formatted_row.append(None)
                            elif isinstance(val, (int, float, bool)):
                                formatted_row.append(val)
                            else:
                                s_val = str(val)
                                if s_val and s_val[0] in (
                                    "=",
                                    "+",
                                    "-",
                                    "@",
                                    "\t",
                                    "\r",
                                ):
                                    is_numeric = False
                                    if s_val[0] in ("+", "-"):
                                        try:
                                            float(s_val)
                                            is_numeric = True
                                        except ValueError:
                                            is_numeric = False
                                    if not is_numeric:
                                        s_val = f"'{s_val}"
                                formatted_row.append(s_val)
                        ws.append(formatted_row)

                output = io.BytesIO()
                wb.save(output)
                output.seek(0)

                chunk_size = 65536
                while True:
                    data = output.read(chunk_size)
                    if not data:
                        break
                    yield data
            finally:
                conn.close()

        return StreamingResponse(
            _generate_excel_chunks(),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={
                "Content-Disposition": f'attachment; filename="{table_name}_export.xlsx"'
            },
        )
    except HTTPException:
        raise
    except Exception as e:
        if conn is not None:
            conn.close()
        logger.error(f"Error exporting table to excel: {e}")
        raise HTTPException(
            status_code=500,
            detail={"error_code": "backend.serverError", "params": {"error": str(e)}},
        )


@app.get("/api/export/parquet")
def export_parquet(
    table_name: str,
    db_path: Optional[str] = None,
) -> StreamingResponse:
    """
    Export table data in columnar Apache Parquet format.

    Args:
    ----
        table_name: Table to export.
        db_path: Optional path to database file.

    Returns:
    -------
        StreamingResponse: Streamed Parquet file.

    Raises:
    ------
        HTTPException: If table or DB is invalid.

    """
    target_db = validate_db_path(db_path)
    if not Path(target_db).exists():
        raise HTTPException(
            status_code=404,
            detail={"error_code": "backend.dbNotFound", "params": {"path": target_db}},
        )

    if not table_name.isidentifier():
        raise HTTPException(
            status_code=400,
            detail={"error_code": "backend.invalidTable", "params": {}},
        )

    conn = None
    temp_parquet_file = None
    try:
        conn = duckdb.connect(target_db, read_only=True)
        tables = [row[0] for row in conn.execute("SHOW TABLES").fetchall()]
        if table_name not in tables:
            conn.close()
            raise HTTPException(
                status_code=404,
                detail={"error_code": "backend.tableNotFound", "params": {}},
            )

        escaped_table = table_name.replace('"', '""')
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
            temp_parquet_file = tmp.name

        escaped_out_path = temp_parquet_file.replace("'", "''")
        conn.execute(
            f"COPY \"{escaped_table}\" TO '{escaped_out_path}' (FORMAT PARQUET, COMPRESSION ZSTD)"
        )

        def _iter_parquet_chunks(
            file_path: str,
            db_conn: duckdb.DuckDBPyConnection,
            chunk_size: int = 65536,
        ) -> Iterator[bytes]:
            """
            Iterate over generated Parquet file in chunks and clean up database and filesystem upon exit.

            Args:
            ----
                file_path: Path to the temporary Parquet file.
                db_conn: Open DuckDB connection to close after streaming.
                chunk_size: Byte size of chunks to stream.

            Yields:
            ------
                bytes: Binary Parquet chunk.

            """
            try:
                with open(file_path, "rb") as f:
                    while True:
                        chunk = f.read(chunk_size)
                        if not chunk:
                            break
                        yield chunk
            finally:
                db_conn.close()
                Path(file_path).unlink(missing_ok=True)

        return StreamingResponse(
            _iter_parquet_chunks(temp_parquet_file, conn),
            media_type="application/vnd.apache.parquet",
            headers={
                "Content-Disposition": f'attachment; filename="{table_name}_export.parquet"'
            },
        )
    except HTTPException:
        raise
    except Exception as e:
        if conn is not None:
            conn.close()
        if temp_parquet_file:
            Path(temp_parquet_file).unlink(missing_ok=True)
        logger.error(f"Error exporting table to parquet: {e}")
        raise HTTPException(
            status_code=500,
            detail={"error_code": "backend.serverError", "params": {"error": str(e)}},
        )


class ColumnInfo(BaseModel):
    """Information about a column in a table."""

    name: str
    type: str


class TableInfo(BaseModel):
    """Information about a table and its columns."""

    name: str
    columns: List[ColumnInfo]


class SchemaResponse(BaseModel):
    """Response model for database schema."""

    tables: List[TableInfo]
    database: Optional[Dict[str, Any]] = None


@app.get("/api/status")
@app.get("/api/health")
def get_status(request: Request, db_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Return comprehensive system health, database status, and LLM availability.

    Args:
    ----
        request: FastAPI request object.
        db_path: Optional custom database path.

    Returns:
    -------
        Dictionary containing health status details.

    """
    target_db = validate_db_path(db_path) if db_path else None
    lang_header = request.headers.get("accept-language", "en")
    lang = lang_header.split(",")[0].split("-")[0].split(";")[0]
    return get_system_health(db_path=target_db, lang=lang).to_dict()


@app.get("/api/schema", response_model=SchemaResponse)
def get_schema(request: Request, db_path: Optional[str] = None) -> SchemaResponse:
    """
    Return structured schema for the frontend Schema Explorer.

    Args:
    ----
        request: FastAPI request.
        db_path: Optional target DuckDB database path.

    Returns:
    -------
        The structured schema response including database diagnostics.

    """
    target_db = validate_db_path(db_path)
    lang_header = request.headers.get("accept-language", "en")
    lang = lang_header.split(",")[0].split("-")[0].split(";")[0]
    db_health = check_database_health(target_db, lang=lang)

    conn = None
    try:
        conn = duckdb.connect(target_db, read_only=True)
        # Fetch tables
        tables = [row[0] for row in conn.execute("SHOW TABLES").fetchall()]

        schema_data = []
        for table in tables:
            # Fetch columns for each table safely quoting table name
            escaped_table = table.replace('"', '""')
            columns = conn.execute(f'DESCRIBE "{escaped_table}"').fetchall()
            col_info = [ColumnInfo(name=c[0], type=c[1]) for c in columns]
            schema_data.append(TableInfo(name=table, columns=col_info))

        return SchemaResponse(tables=schema_data, database=db_health.to_dict())
    except Exception as e:
        logger.error(f"Error fetching schema: {e}")
        return SchemaResponse(tables=[], database=db_health.to_dict())
    finally:
        if conn is not None:
            conn.close()


@app.get("/api/providers/status", response_model=ProvidersStatusResponse)
def get_providers_status() -> ProvidersStatusResponse:
    """
    Report connection health and credential status for all supported LLM providers.

    Returns
    -------
        ProvidersStatusResponse containing status for ollama, openai, anthropic, google.

    """
    statuses: List[ProviderStatus] = []

    # 1. Ollama
    ollama_healthy = False
    ollama_details = "Ollama service unavailable"
    ollama_base = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    if not ollama_base.startswith("http://") and not ollama_base.startswith("https://"):
        ollama_base = f"http://{ollama_base}"
    try:
        req = urllib.request.Request(f"{ollama_base}/api/tags")
        with urllib.request.urlopen(req, timeout=2) as resp:
            if resp.status == 200:
                ollama_healthy = True
                ollama_details = "Ollama service online"
    except Exception as e:
        ollama_details = f"Ollama unreachable: {e}"

    statuses.append(
        ProviderStatus(
            provider="ollama",
            configured=True,
            healthy=ollama_healthy,
            details=ollama_details,
            reachable=ollama_healthy,
            error_code="PROVIDER_ONLINE" if ollama_healthy else "PROVIDER_OFFLINE",
        )
    )

    # 2. OpenAI
    openai_conf = is_provider_configured("openai")
    statuses.append(
        ProviderStatus(
            provider="openai",
            configured=openai_conf,
            healthy=openai_conf,
            details="Configured via OPENAI_API_KEY"
            if openai_conf
            else "Missing OPENAI_API_KEY",
            reachable=openai_conf,
            error_code="PROVIDER_ONLINE" if openai_conf else "MISSING_API_KEY",
        )
    )

    # 3. Anthropic
    anthropic_conf = is_provider_configured("anthropic")
    statuses.append(
        ProviderStatus(
            provider="anthropic",
            configured=anthropic_conf,
            healthy=anthropic_conf,
            details="Configured via ANTHROPIC_API_KEY"
            if anthropic_conf
            else "Missing ANTHROPIC_API_KEY",
            reachable=anthropic_conf,
            error_code="PROVIDER_ONLINE" if anthropic_conf else "MISSING_API_KEY",
        )
    )

    # 4. Google
    google_conf = is_provider_configured("google")
    statuses.append(
        ProviderStatus(
            provider="google",
            configured=google_conf,
            healthy=google_conf,
            details="Configured via GEMINI_API_KEY/GOOGLE_API_KEY"
            if google_conf
            else "Missing GEMINI_API_KEY",
            reachable=google_conf,
            error_code="PROVIDER_ONLINE" if google_conf else "MISSING_API_KEY",
        )
    )

    return ProvidersStatusResponse(providers=statuses)


@app.get("/api/models", response_model=ModelsResponse)
def list_models(include_cloud: bool = True) -> ModelsResponse:
    """
    List available local Ollama models and configured cloud LLM providers.

    Args:
    ----
        include_cloud: Whether to include standard cloud provider models.

    Returns:
    -------
        ModelsResponse: List of available local and cloud models.

    """
    base_url = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
    if not base_url.startswith("http://") and not base_url.startswith("https://"):
        base_url = f"http://{base_url}"

    models: List[ModelInfo] = []
    try:
        req = urllib.request.Request(f"{base_url}/api/tags")
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode())

        for model in data.get("models", []):
            models.append(
                ModelInfo(
                    name=model.get("name"),
                    size=model.get("size"),
                    provider="ollama",
                    configured=True,
                    requires_key=False,
                    available=True,
                    reachable=True,
                    error_code=None,
                )
            )
    except Exception as e:
        logger.error(f"Error fetching models from local Ollama instance: {e}")

    if include_cloud:
        for prov, prov_models in CLOUD_PROVIDER_MODELS.items():
            conf = is_provider_configured(prov)
            for m in prov_models:
                models.append(
                    ModelInfo(
                        name=m,
                        provider=prov,
                        configured=conf,
                        requires_key=True,
                        available=conf,
                        reachable=conf,
                        error_code=None if conf else "MISSING_API_KEY",
                    )
                )

    return ModelsResponse(models=models)


class DatabaseItem(BaseModel):
    """Information about an available DuckDB database file."""

    name: str
    path: str
    size_bytes: int


class DatabaseListResponse(BaseModel):
    """Response model for the databases listing endpoint."""

    databases: List[DatabaseItem]
    current_db: str


@app.get("/api/databases", response_model=DatabaseListResponse)
def list_databases() -> DatabaseListResponse:
    """
    Discover and list available DuckDB database files in the workspace.

    Returns
    -------
        DatabaseListResponse containing list of found databases and current default.

    """
    default_db = os.environ.get("T1D_DB_PATH", "t1d.duckdb")
    all_files: List[Path] = []
    for s_dir in (Path("."), Path("./data")):
        if s_dir.exists():
            all_files.extend(s_dir.glob("*.duckdb"))

    found_dbs: List[DatabaseItem] = []
    seen_paths = set()
    for p in all_files:
        resolved = str(p.resolve())
        if resolved not in seen_paths:
            seen_paths.add(resolved)
            found_dbs.append(
                DatabaseItem(
                    name=p.name,
                    path=str(p),
                    size_bytes=p.stat().st_size if p.exists() else 0,
                )
            )

    return DatabaseListResponse(databases=found_dbs, current_db=default_db)


if os.environ.get("DEBUG"):  # pragma: no cover
    dist_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "web", "dist"
    )
    if os.path.exists(dist_path):
        logger.info(f"DEBUG mode enabled. Serving static frontend from {dist_path}")
        app.mount("/", StaticFiles(directory=dist_path, html=True), name="static")
    else:
        logger.warning(
            f"DEBUG mode enabled but static dist folder not found at {dist_path}"
        )
