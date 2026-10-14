"""Pydantic request and response schemas for the T1D Analytics REST API."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from t1d_analytics.sql_utils import SqlValue


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
    isError: Optional[bool] = None
    model: Optional[str] = None
    provider: Optional[str] = None


class ChatRequest(BaseModel):
    """
    Request model for the chat endpoint.

    Attributes
    ----------
        message: The natural language question or SQL query from the user.
        model: Model identifier or 'provider/model' string.
        provider: Optional explicit LLM provider ('ollama', 'openai', 'anthropic', 'google').
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


class ExecuteSqlRequest(BaseModel):
    """Request model for the execute SQL endpoint."""

    query: str
    db_path: Optional[str] = None


class ExecuteSqlResponse(BaseModel):
    """Response model for the execute SQL endpoint."""

    sqlResult: Optional[List[Dict[str, SqlValue]]] = None
    error: Optional[ApiError] = None


class TableDataResponse(BaseModel):
    """Response model for table data."""

    rows: List[Dict[str, SqlValue]]
    total_count: int = 0
    page: int = 1
    total_pages: int = 1


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


class DatabaseItem(BaseModel):
    """Information about an available DuckDB database file."""

    name: str
    path: str
    size_bytes: int


class DatabaseListResponse(BaseModel):
    """Response model for the databases listing endpoint."""

    databases: List[DatabaseItem]
    current_db: str


class TrainingJobRequest(BaseModel):
    """Request model for submitting an asynchronous background training job."""

    dataset_path: str
    output_dir: str
    backend: str = "cpu"
    model_name: str = "gemma4-sql"
    dry_run: bool = False
