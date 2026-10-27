"""Pydantic request and response schemas for the T1D Analytics REST API."""

from typing import Dict, List, Optional, Union

from pydantic import BaseModel

from t1d_analytics.sql_utils import SqlValue

JsonScalar = SqlValue
JsonLevel1 = Union[SqlValue, List[SqlValue]]
JsonLevel2 = Union[JsonLevel1, Dict[str, JsonLevel1]]
JsonNestedValue = Union[JsonLevel2, Dict[str, JsonLevel2]]


class ApiError(BaseModel):
    """Structured API error format."""

    error_code: str
    params: Dict[str, SqlValue] = {}


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
        auto_execute: Whether to automatically execute generated SQL for natural language queries.

    """

    message: str
    model: str = "gemma4"
    provider: Optional[str] = None
    api_key: Optional[str] = None
    db_path: Optional[str] = None
    session_id: Optional[str] = None
    history: Optional[List[ChatMessageModel]] = None
    auto_execute: bool = True


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
    rows: Optional[List[Dict[str, SqlValue]]] = None
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


class TableSchemaModel(BaseModel):
    """
    Structured representation of a DuckDB table schema for LLM prompt construction and RAG pruning.

    Attributes
    ----------
        table_name: Name of the database table.
        columns: Ordered list of column metadata (name and SQL data type).
        sample_row: Optional formatted sample row string from the table.

    """

    table_name: str
    columns: List[ColumnInfo] = []
    sample_row: Optional[str] = None

    def column_names(self) -> List[str]:
        """
        Return the list of column names for this table.

        Returns
        -------
            List[str]: Ordered column names.

        """
        return [col.name for col in self.columns]

    def to_prompt_block(self) -> str:
        """
        Format the table schema into the canonical multi-line prompt representation.

        Returns
        -------
            str: Formatted `Table: ... / Columns: ... / Sample row: ...` block.

        """
        col_desc = ", ".join([f"{c.name} ({c.type})" for c in self.columns])
        sample_desc = self.sample_row or "Sample row: (empty table)"
        return f"Table: {self.table_name}\nColumns: {col_desc}\n{sample_desc}"


class SchemaResponse(BaseModel):
    """Response model for database schema."""

    tables: List[TableInfo]
    database: Optional[Dict[str, JsonLevel2]] = None


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


class StatusResponse(BaseModel):
    """
    Canonical service status summary model for the `/api/status` endpoint and diagnostics.

    Attributes
    ----------
        status: High-level service readiness status ('ready', 'healthy', 'degraded', or 'error').
        tables_count: Total number of clinical tables in the database.
        total_rows: Total number of rows across all clinical tables.
        required_tables_present: Whether expected T1D clinical tables are present.
        backend_accessible: Whether the FastAPI backend is reachable.
        version: Backend application version string.
        database: Dictionary with detailed database diagnostics.
        ollama: Dictionary with Ollama LLM service diagnostics.
        providers: Dictionary with cloud LLM provider diagnostics.

    """

    status: str = "ready"
    tables_count: int = 0
    total_rows: int = 0
    required_tables_present: bool = False
    backend_accessible: bool = True
    version: str = "0.1.0"
    database: Dict[str, JsonLevel2] = {}
    ollama: Dict[str, JsonLevel2] = {}
    providers: Dict[str, Dict[str, JsonLevel2]] = {}

    def to_dict(
        self,
    ) -> Dict[
        str,
        Union[
            SqlValue,
            Dict[str, JsonLevel2],
            Dict[str, Dict[str, JsonLevel2]],
        ],
    ]:
        """
        Convert StatusResponse to a dictionary representation.

        Returns
        -------
            Dict[str, Union[SqlValue, Dict[str, JsonLevel2], Dict[str, Dict[str, JsonLevel2]]]]:
                Dictionary payload of status fields.

        """
        return {
            "status": self.status,
            "tables_count": self.tables_count,
            "total_rows": self.total_rows,
            "required_tables_present": self.required_tables_present,
            "backend_accessible": self.backend_accessible,
            "version": self.version,
            "database": self.database or {},
            "ollama": self.ollama or {},
            "providers": self.providers or {},
        }
