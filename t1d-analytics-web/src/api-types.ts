/**
 * @file api-types.ts
 * @module api-types
 * Canonical TypeScript wire contracts for FastAPI backend endpoints (`/api/chat`,
 * `/api/chat/stream`, `/api/chat/jobs`, `/api/execute-sql`, `/api/status`,
 * `/api/providers/status`, `/api/models`, `/api/databases`, `/api/schema`, `/api/table/*`).
 */

import { type SqlCellValue } from "./formatters";
import { type SqlGoldComparisonPayload } from "./state";

/**
 * Structured API error detail object returned by backend FastAPI/T1DError handlers.
 */
export interface ApiErrorPayload {
  /** Machine-readable i18n error key (e.g. `backend.invalidDatabasePath`). */
  error_code: string;
  /** Optional localized or fallback message string. */
  message?: string;
  /** Optional interpolation parameters for `i18next.t`. */
  params?: Record<string, SqlCellValue>;
}

/**
 * Input value accepted by `parseApiError`.
 */
export type ApiErrorInput =
  | string
  | number
  | boolean
  | null
  | undefined
  | ApiErrorPayload
  | Record<string, never>;

/**
 * Canonical chat turn response payload shared across `/api/chat`, `/api/chat/stream` (`done` event),
 * `/api/chat/jobs/{jobId}`, and `/api/execute-sql`.
 */
export interface ChatTurnResponse {
  /** Optional SSE event discriminator (`token`, `done`, `result`, `sql_generated`, `execution_progress`, `error`). */
  event?: string;
  /** Optional streamed token delta when `event === "token"`. */
  token?: string;
  /** Async job identifier (`chatjob-...`) when dispatched via `/api/chat/jobs` or `async_poll`. */
  job_id?: string | null;
  /** Lifecycle status (`"pending"`, `"complete"`, `"failed"`). */
  status?: string;
  /** Live accumulated partial text while `"pending"`. */
  partial_content?: string;
  /** Assistant response text or translation key. */
  content?: string;
  /** Generated or executed SQL query string. */
  sqlQuery?: string;
  /** Tabular SQL query result rows. */
  sqlResult?: Record<string, SqlCellValue>[];
  /** Structured or string API error payload. */
  error?: ApiErrorInput;
  /** Whether the queried DuckDB database uses synthetic seed data rather than authentic Jaeb trial data. */
  is_synthetic_data?: boolean;
  /** Whether deterministic rule-based SQL synthesis was used instead of an LLM. */
  is_rule_based_sql?: boolean;
  /** SQL generation mode (`"llm"`, `"rule_based_fallback"`, or `"literal_sql"`). */
  generation_mode?: string;
  /** Optional human-readable diagnostic reason why LLM inference failed prior to fallback synthesis. */
  llm_error?: string | null;
  /** Optional structured error code for the underlying LLM failure. */
  llm_error_code?: string | null;
  /** Database provenance classification (`"jaeb_authentic"`, `"synthetic_seed"`, `"custom"`). */
  data_provenance?: string;
  /** Optional gold-standard SQL evaluation metrics returned by the backend. */
  gold_comparison?: SqlGoldComparisonPayload | null;
  /** Whether fallback synthesis repaired or replaced a failed LLM attempt. */
  used_fallback?: boolean;
}

/**
 * Per-provider readiness metadata returned by `GET /api/providers/status` and `GET /api/status`.
 */
export interface ProviderStatusEntryPayload {
  /** Whether credentials or local daemon configuration are present. */
  configured: boolean;
  /** Credential source (`env`, `header`, `local`, `none`). */
  source?: string;
  /** Whether the provider endpoint responded to reachability checks. */
  reachable?: boolean;
  /** Discovered or configured model names for this provider. */
  models?: string[];
  /** Optional human-readable status message. */
  message?: string;
}

/**
 * Response contract for `GET /api/providers/status`.
 */
export interface ProvidersStatusResponsePayload {
  /** Dictionary of provider identifier (`ollama`, `openai`, `anthropic`, `google`) to status entry. */
  providers: Record<string, ProviderStatusEntryPayload>;
}

/**
 * Database health sub-payload inside `GET /api/status`.
 */
export interface DatabaseHealthPayload {
  /** Whether the DuckDB file exists on disk. */
  exists: boolean;
  /** Whether a read-only DuckDB connection succeeded. */
  accessible: boolean;
  /** Machine-readable database status code (`healthy`, `empty_db`, `missing_initial_data`, `missing_file`, `corrupted`). */
  status_code?: string;
  /** Number of tables discovered in the database. */
  table_count: number;
  /** List of table names present in the database. */
  tables?: string[];
  /** Whether expected initial clinical tables are present. */
  has_initial_data: boolean;
  /** Human-readable status summary. */
  message: string;
  /** Optional CLI remediation command or guidance. */
  remediation?: string | null;
  /** Size of the DuckDB file in bytes. */
  file_size_bytes?: number;
  /** Whether the active database contains synthetic seed data rather than authentic Jaeb clinical data. */
  is_synthetic_seed?: boolean;
  /** Optional recommended Jaeb DuckDB path configured on the server. */
  recommended_jaeb_db_path?: string | null;
}

/**
 * Local Ollama daemon health sub-payload inside `GET /api/status`.
 */
export interface OllamaStatusPayload {
  /** Whether the Ollama HTTP API responded. */
  accessible: boolean;
  /** Models currently pulled in local Ollama. */
  available_models: string[];
  /** Human-readable status message. */
  message: string;
  /** Optional remediation command when Ollama or default model is unavailable. */
  remediation?: string | null;
}

/**
 * Sessions DuckDB health sub-payload inside `GET /api/status`.
 */
export interface SessionsDatabaseHealthPayload {
  /** Whether `sessions.duckdb` exists on disk. */
  exists: boolean;
  /** Whether `sessions.duckdb` is readable. */
  accessible: boolean;
  /** Number of persisted chat sessions. */
  session_count?: number;
  /** Human-readable status summary. */
  message?: string;
}

/**
 * Response contract for `GET /api/status`, `GET /api/health`, and `GET /api/ready`.
 */
export interface StatusResponsePayload {
  /** Overall service health (`healthy`, `degraded`, `unhealthy`, `error`). */
  status: string;
  /** DuckDB health details. */
  database?: DatabaseHealthPayload;
  /** Sessions DuckDB health details. */
  sessions_database?: SessionsDatabaseHealthPayload;
  /** Local Ollama health details. */
  ollama?: OllamaStatusPayload;
  /** Optional dictionary of cloud/local provider statuses. */
  providers?: Record<string, ProviderStatusEntryPayload>;
}

/**
 * Single model entry returned by `GET /api/models`.
 */
export interface ModelCatalogEntryPayload {
  /** Model identifier (e.g. `gemma:2b`, `openai/gpt-4o-mini`, `sql`). */
  name: string;
  /** Human-readable display label. */
  label?: string;
  /** Owning provider (`ollama`, `openai`, `anthropic`, `google`, `duckdb`). */
  provider?: string;
  /** Whether required credentials are configured. */
  configured?: boolean;
  /** Whether the provider is currently reachable. */
  reachable?: boolean;
}

/**
 * Response contract for `GET /api/models`.
 */
export interface ModelsResponsePayload {
  /** Available SQL and LLM model entries. */
  models: ModelCatalogEntryPayload[];
}

/**
 * Schema column entry inside `GET /api/schema`.
 */
export interface SchemaColumnPayload {
  /** Column identifier. */
  name: string;
  /** DuckDB SQL data type. */
  type: string;
}

/**
 * Schema table entry inside `GET /api/schema`.
 */
export interface SchemaTablePayload {
  /** Table identifier. */
  name: string;
  /** Ordered list of table columns. */
  columns: SchemaColumnPayload[];
}

/**
 * Response contract for `GET /api/schema`.
 */
export interface SchemaResponsePayload {
  /** Tables and columns in the active DuckDB database. */
  tables?: SchemaTablePayload[];
  /** Legacy alias supported by older callers. */
  schema?: SchemaTablePayload[];
  /** Optional error payload if schema inspection failed. */
  error?: ApiErrorInput;
}

/**
 * Paginated table preview response contract for `GET /api/table/{table_name}`.
 */
export interface TableRowsResponsePayload {
  /** Ordered column names. */
  columns?: string[];
  /** Page row records. */
  rows: Record<string, SqlCellValue>[];
  /** Total matching row count across all pages. */
  total_rows?: number;
  /** Current 1-based page index. */
  page?: number;
  /** Total number of pages. */
  total_pages?: number;
}
