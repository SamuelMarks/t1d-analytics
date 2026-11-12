/**
 * @file state.ts
 * @module state
 * Manages the application state for the chat UI.
 */
import i18next from "./i18n";
import {
  deleteRemoteSession,
  fetchRemoteSessionById,
  fetchRemoteSessions,
  pushRemoteSession,
  type RemoteSessionPayload,
} from "./api-client";
import {
  type TableRelationship,
  CLINICAL_RELATIONSHIPS,
  resolveTableJoin,
} from "./cohort-sql";

export { type TableRelationship, CLINICAL_RELATIONSHIPS, resolveTableJoin };

/**
 * Safely decodes a URI component, returning the raw string if percent-encoding is malformed.
 * @param {string} raw - Raw URI component string.
 * @returns {string} Decoded URI component string.
 */
export function safeDecodeUriComponent(raw: string): string {
  try {
    return decodeURIComponent(raw);
  } catch {
    return raw;
  }
}

/**
 * Extracts a chat session identifier from a browser Location object.
 * Supports canonical `/chats/${ID}` paths as well as legacy `#${ID}` URL hashes.
 * @param {Pick<Location, "pathname" | "hash">} loc - Location object containing pathname and hash.
 * @returns {string | null} Extracted chat ID, or null when on the root route with no chat ID.
 */
export function extractChatIdFromLocation(
  loc: Pick<Location, "pathname" | "hash">,
): string | null {
  if (loc.hash && loc.hash.length > 1) {
    return safeDecodeUriComponent(loc.hash.slice(1));
  }
  const pathMatch = /^\/chats\/([^/?#]+)\/?$/.exec(loc.pathname);
  if (pathMatch) {
    return safeDecodeUriComponent(String(pathMatch[1]));
  }
  return null;
}

/**
 * Builds the canonical URL path for a given chat session ID.
 * @param {string | null} chatId - Active chat session identifier, or null.
 * @returns {string} Canonical `/chats/${ID}` path or `/` when no chat is active.
 */
export function buildChatPath(chatId: string | null): string {
  if (chatId) {
    return `/chats/${encodeURIComponent(chatId)}`;
  }
  return "/";
}

/** Default DuckDB database filename. */
export const DEFAULT_DB_NAME = "t1d.duckdb";

/** Default local LLM model identifier. */
export const DEFAULT_CHAT_MODEL = "gemma:2b";

/** Direct SQL execution mode identifier. */
export const RAW_SQL_MODEL = "sql";

/**
 * Shape of the serialized ChatState payload persisted in localStorage.
 */
export interface PersistedChatStatePayload {
  /** Persisted chat sessions array. */
  chats?: Chat[];
  /** Persisted active chat session identifier. */
  activeChatId?: string | null;
  /** Persisted monotonic chat counter. */
  chatCounter?: number;
  /** Persisted selected DuckDB database path. */
  currentDb?: string;
  /** Persisted discovered databases list. */
  availableDbs?: AvailableDatabaseItem[];
  /** Persisted server synchronization flag. */
  serverSyncEnabled?: boolean;
  /** Persisted SSE streaming mode flag. */
  streamingMode?: boolean;
  /** Persisted automatic SQL execution flag for NL-to-SQL turns. */
  autoExecuteSql?: boolean;
  /** Persisted tombstone list of deleted session IDs. */
  deletedChatIds?: string[];
}

/**
 * Mapping of legacy i18next HTML entities to their unescaped literal characters.
 */
const LEGACY_HTML_ENTITY_MAP: Readonly<Record<string, string>> = {
  "&#39;": "'",
  "&#x27;": "'",
  "&apos;": "'",
  "&quot;": '"',
  "&#34;": '"',
  "&#x2F;": "/",
  "&#x2f;": "/",
  "&#47;": "/",
  "&gt;": ">",
  "&lt;": "<",
  "&amp;": "&",
};

/**
 * Decodes legacy i18next HTML character entities in persisted message text.
 * @param {string} text - Raw persisted message content string.
 * @returns {string} Decoded string with HTML entities restored to literal characters.
 */
export function decodeLegacyHtmlEntities(text: string): string {
  return text.replace(
    /&(?:#39|#x27|apos|quot|#34|#x2[Ff]|#47|gt|lt|amp);/g,
    (entity) => String(LEGACY_HTML_ENTITY_MAP[entity]),
  );
}

/**
 * Sanitizes a single message record by returning a new object with decoded legacy HTML entities in its content.
 * @param {Message} msg - Raw message record.
 * @returns {Message} Newly constructed sanitized message record.
 */
export function sanitizeMessageRecord(msg: Message): Message {
  return {
    ...msg,
    content: decodeLegacyHtmlEntities(msg.content),
  };
}

/**
 * Sanitizes the messages of each chat in a stored chat list without mutating input chat objects in place.
 * @param {Chat[] | undefined} chats - Stored chats array.
 * @returns {Chat[]} Sanitized chats array.
 */
export function parseStoredChatList(chats?: Chat[]): Chat[] {
  if (!Array.isArray(chats)) {
    return [];
  }
  return chats.map((chat) =>
    Array.isArray(chat.messages)
      ? {
          ...chat,
          messages: chat.messages.map((msg) => sanitizeMessageRecord(msg)),
        }
      : { ...chat },
  );
}

/**
 * Merges a remote session payload into the local `chats` array and returns the merged `Chat`.
 * @param {Chat[]} chats - Mutable local chats list.
 * @param {RemoteSessionPayload<Message>} remoteSession - Remote session payload from `/api/sessions`.
 * @returns {Chat} Merged local chat record.
 */
export function mergeRemoteSessionIntoLocalChats(
  chats: Chat[],
  remoteSession: RemoteSessionPayload<Message>,
): Chat {
  const local = chats.find((c) => c.id === remoteSession.session_id);
  const remoteTime = remoteSession.updated_at
    ? new Date(remoteSession.updated_at).getTime()
    : 0;
  const remoteMessages = (
    Array.isArray(remoteSession.messages) ? remoteSession.messages : []
  ).map((msg) => sanitizeMessageRecord(msg));
  if (!local) {
    const created: Chat = {
      id: remoteSession.session_id,
      title: remoteSession.title,
      messages: remoteMessages,
      model: DEFAULT_CHAT_MODEL,
      isTemporary: false,
      updatedAt: remoteTime,
    };
    chats.push(created);
    return created;
  }
  if (remoteMessages.length > local.messages.length) {
    local.messages = remoteMessages;
    local.updatedAt = remoteTime;
  }
  if (
    remoteSession.title &&
    remoteSession.title !== local.title &&
    !local.titleModified
  ) {
    local.title = remoteSession.title;
    local.updatedAt = remoteTime;
  }
  return local;
}

/**
 * Parses a serialized ChatState JSON string into a typed PersistedChatStatePayload.
 * @param {string} rawJson - Serialized JSON string from localStorage.
 * @returns {PersistedChatStatePayload} Parsed state payload object.
 */
export function parsePersistedChatState(
  rawJson: string,
): PersistedChatStatePayload {
  const parsed: PersistedChatStatePayload = JSON.parse(rawJson);
  if (Array.isArray(parsed.chats)) {
    parsed.chats = parseStoredChatList(parsed.chats);
  }
  return parsed;
}

/**
 * Evaluation metrics comparing an LLM-generated SQL query against a Gold-Standard SQL query on the read-only DuckDB cursor.
 */
export interface SqlGoldComparisonPayload {
  /** Canonical gold-standard DuckDB SQL query. */
  gold_sql: string;
  /** SQL query generated by the target LLM. */
  pred_sql: string;
  /** Whether normalized `pred_sql` matches normalized `gold_sql`. */
  exact_match: boolean;
  /** Whether executing `pred_sql` and `gold_sql` on the read-only DuckDB cursor yields equivalent result sets. */
  execution_accuracy: boolean;
  /** Whether `pred_sql` raised a DuckDB syntax/binder/execution error. */
  syntax_error: boolean;
  /** Total rows returned by `gold_sql` on the read-only cursor. */
  gold_row_count: number;
  /** Total rows returned by `pred_sql` on the read-only cursor. */
  pred_row_count: number;
  /** Ordered column names returned by `gold_sql`. */
  gold_columns: string[];
  /** Ordered column names returned by `pred_sql`. */
  pred_columns: string[];
  /** Value/cell multiset approximation percentage (0.0 to 100.0) between predicted and gold results. */
  column_overlap_pct: number;
  /** Preview rows from executing `gold_sql` on the read-only DuckDB cursor. */
  gold_rows_preview: Record<string, string | number | boolean | null>[];
  /** Optional diagnostic error message if `pred_sql` failed execution. */
  error_message?: string | null;
}

/**
 * Represents a single chat message.
 */
export interface Message {
  /** The role of the message sender. */
  role: "user" | "assistant";
  /** The textual content of the message. */
  content: string;
  /** Optional tabular data returned from a SQL query execution. */
  sqlResult?: Record<string, string | number | boolean | null>[];
  /** Optional SQL query string generated or executed. */
  sqlQuery?: string;
  /** Optional flag indicating this message represents an error. */
  isError?: boolean;
  /** Optional model identifier used to process this message. */
  model?: string;
  /** Optional gold-standard SQL attached to a user prompt for NL-to-SQL benchmarking. */
  goldSql?: string;
  /** Optional gold-standard SQL comparison metrics returned by the backend. */
  goldComparison?: SqlGoldComparisonPayload;
  /** Optional flag indicating whether rule-based SQL fallback synthesis was used. */
  isRuleBasedSql?: boolean;
  /** Optional SQL generation mode ('llm', 'rule_based_fallback', or 'literal_sql'). */
  generationMode?: string;
  /** Optional human-readable diagnostic reason why LLM inference failed prior to fallback synthesis. */
  llmError?: string;
  /** Optional structured error code for the underlying LLM failure. */
  llmErrorCode?: string;
}

/**
 * Represents a chat session.
 */
export interface Chat {
  /** Unique identifier for the chat. */
  id: string;
  /** Display title for the chat. */
  title: string;
  /** The sequence of messages in the chat. */
  messages: Message[];
  /** The model identifier used in this chat. */
  model: string;
  /** Optional provider identifier (e.g. 'ollama', 'openai', 'anthropic', 'google'). */
  provider?: string;
  /** Whether the chat is a temporary placeholder. */
  isTemporary?: boolean;
  /** Optional timestamp (epoch ms) of last modification. */
  updatedAt?: number;
  /** Whether the chat title was locally modified and needs sync. */
  titleModified?: boolean;
}

/**
 * System health and diagnostic status tracking backend, database, and LLM.
 */
export interface SystemStatus {
  /** Whether backend API is reachable. */
  backendOnline: boolean;
  /** Overall system health level. */
  status: "healthy" | "degraded" | "error" | "offline";
  /** Whether database path is configured. */
  dbConfigured: boolean;
  /** Whether database file exists on disk. */
  dbExists: boolean;
  /** Whether database connection is active. */
  dbConnected: boolean;
  /** Database diagnostic status code. */
  dbStatusCode: string;
  /** Human-readable database status message. */
  dbMessage: string | null;
  /** Remediation advice for database. */
  dbRemediation: string | null;
  /** Count of tables present in database. */
  tableCount: number;
  /** Whether clinical trial initial data is present. */
  hasInitialData: boolean;
  /** Whether Ollama service is reachable. */
  ollamaOnline: boolean;
  /** Human-readable Ollama status message. */
  ollamaMessage: string | null;
  /** Remediation advice for Ollama service. */
  ollamaRemediation: string | null;
  /** Database file size in bytes. */
  dbFileSizeBytes?: number;
  /** Whether database file is writable. */
  dbWritable?: boolean;
  /** Whether database passed integrity check. */
  dbIntegrityOk?: boolean;
  /** Free disk space in bytes. */
  diskFreeBytes?: number;
  /** Ollama server version. */
  ollamaVersion?: string | null;
  /** Recommended authentic Jaeb DuckDB path for the current host. */
  recommendedJaebDbPath?: string;
}

/**
 * Safely reads a string item from browser localStorage.
 * @param {string} key - The storage key to read.
 * @returns {string | null} The stored string value, or null if unavailable or throwing.
 */
export function safeStorageGet(key: string): string | null {
  try {
    if (typeof localStorage === "undefined") return null;
    return localStorage.getItem(key);
  } catch (e) {
    console.warn(`Could not read '${key}' from localStorage`, e);
    return null;
  }
}

/**
 * Safely writes a string item to browser localStorage.
 * @param {string} key - The storage key to write.
 * @param {string} value - The string value to persist.
 * @returns {void}
 */
export function safeStorageSet(key: string, value: string): void {
  try {
    if (typeof localStorage === "undefined") return;
    localStorage.setItem(key, value);
  } catch (e) {
    console.warn(`Could not save '${key}' to localStorage`, e);
  }
}

/**
 * Safely removes an item from browser localStorage.
 * @param {string} key - The storage key to remove.
 * @returns {void}
 */
export function safeStorageRemove(key: string): void {
  try {
    if (typeof localStorage === "undefined") return;
    localStorage.removeItem(key);
  } catch (e) {
    console.warn(`Could not remove '${key}' from localStorage`, e);
  }
}

/**
 * Storage scope for sensitive API credentials (`"session"` for tab-scoped `sessionStorage`, `"local"` for `localStorage`).
 */
export type CredentialStorageScope = "session" | "local";

/**
 * Safely reads a string item from browser `sessionStorage`.
 * @param {string} key - The storage key to read.
 * @returns {string | null} Stored value or null if unavailable.
 */
export function safeSessionStorageGet(key: string): string | null {
  try {
    if (typeof sessionStorage === "undefined") return null;
    return sessionStorage.getItem(key);
  } catch (e) {
    console.warn(`Could not read '${key}' from sessionStorage`, e);
    return null;
  }
}

/**
 * Safely writes a string item to browser `sessionStorage`.
 * @param {string} key - The storage key to write.
 * @param {string} value - The string value to persist.
 * @returns {void}
 */
export function safeSessionStorageSet(key: string, value: string): void {
  try {
    if (typeof sessionStorage === "undefined") return;
    sessionStorage.setItem(key, value);
  } catch (e) {
    console.warn(`Could not save '${key}' to sessionStorage`, e);
  }
}

/**
 * Safely removes an item from browser `sessionStorage`.
 * @param {string} key - The storage key to remove.
 * @returns {void}
 */
export function safeSessionStorageRemove(key: string): void {
  try {
    if (typeof sessionStorage === "undefined") return;
    sessionStorage.removeItem(key);
  } catch (e) {
    console.warn(`Could not remove '${key}' from sessionStorage`, e);
  }
}

/**
 * Metadata for an available workspace DuckDB database file.
 */
export interface AvailableDatabaseItem {
  /** Display name of the database file. */
  name: string;
  /** Filesystem path to the database file. */
  path: string;
  /** File size in bytes. */
  size_bytes: number;
}

/**
 * Column metadata entry within a database table schema.
 */
export interface SchemaColumnMetadata {
  /** Column identifier name. */
  name: string;
  /** DuckDB SQL data type name. */
  type: string;
}

/**
 * Table metadata entry stored in `ChatState` from `/api/schema`.
 */
export interface SchemaTableMetadata {
  /** Table identifier name. */
  name: string;
  /** Ordered column definitions for the table. */
  columns: SchemaColumnMetadata[];
}

/**
 * Composable store managing cloud LLM provider API keys, backend `T1D_API_KEY` Bearer tokens,
 * and custom model configurations across `sessionStorage` or `localStorage`.
 */
export class ProviderCredentialsStore {
  /**
   * Returns the configured storage scope (`"session"` or `"local"`) for API credentials.
   * @returns {CredentialStorageScope} Active storage scope (defaults to `"local"` for backward compatibility unless set to `"session"`).
   */
  public getStorageScope(): CredentialStorageScope {
    const scope =
      safeSessionStorageGet("t1d_credential_storage_scope") ??
      safeStorageGet("t1d_credential_storage_scope");
    return scope === "session" ? "session" : "local";
  }

  /**
   * Updates the credential storage scope (`"session"` or `"local"`).
   * When switching to `"session"`, migrates existing keys out of `localStorage` into `sessionStorage`.
   * @param {CredentialStorageScope} scope - Target storage scope.
   * @returns {void}
   */
  public setStorageScope(scope: CredentialStorageScope): void {
    if (scope === "session") {
      safeSessionStorageSet("t1d_credential_storage_scope", "session");
      safeStorageSet("t1d_credential_storage_scope", "session");
      for (const p of ["openai", "anthropic", "google"]) {
        const storageKey = `t1d_api_key_${p}`;
        const existing = safeStorageGet(storageKey);
        if (existing) {
          safeSessionStorageSet(storageKey, existing);
          safeStorageRemove(storageKey);
        }
      }
      const backendKey = safeStorageGet("t1d_backend_api_key");
      if (backendKey) {
        safeSessionStorageSet("t1d_backend_api_key", backendKey);
        safeStorageRemove("t1d_backend_api_key");
      }
    } else {
      safeSessionStorageRemove("t1d_credential_storage_scope");
      safeStorageSet("t1d_credential_storage_scope", "local");
    }
  }

  /**
   * Retrieves a stored custom API key for a cloud LLM provider from `sessionStorage` or `localStorage`.
   * @param {string} provider - The provider identifier (e.g. 'openai', 'anthropic', 'google').
   * @returns {string | null} The API key if stored, otherwise null.
   */
  public getApiKey(provider: string): string | null {
    const keyName = `t1d_api_key_${provider.toLowerCase()}`;
    return safeSessionStorageGet(keyName) ?? safeStorageGet(keyName);
  }

  /**
   * Stores a custom API key for a cloud LLM provider in the active storage scope (`sessionStorage` or `localStorage`).
   * @param {string} provider - The provider identifier.
   * @param {string} key - The API key to store.
   * @returns {void}
   */
  public setApiKey(provider: string, key: string): void {
    const keyName = `t1d_api_key_${provider.toLowerCase()}`;
    const trimmed = key.trim();
    if (this.getStorageScope() === "session") {
      safeSessionStorageSet(keyName, trimmed);
      safeStorageRemove(keyName);
    } else {
      safeStorageSet(keyName, trimmed);
    }
  }

  /**
   * Removes a stored custom API key for a cloud LLM provider from both `sessionStorage` and `localStorage`.
   * @param {string} provider - The provider identifier.
   * @returns {void}
   */
  public removeApiKey(provider: string): void {
    const keyName = `t1d_api_key_${provider.toLowerCase()}`;
    safeSessionStorageRemove(keyName);
    safeStorageRemove(keyName);
  }

  /**
   * Retrieves the configured `T1D_API_KEY` backend Bearer authentication token, if present.
   * @returns {string | null} Backend Bearer token or null.
   */
  public getBackendApiKey(): string | null {
    return (
      safeSessionStorageGet("t1d_backend_api_key") ??
      safeStorageGet("t1d_backend_api_key")
    );
  }

  /**
   * Stores or clears the `T1D_API_KEY` backend Bearer authentication token in the active storage scope.
   * @param {string} token - Bearer token string (empty string removes the stored token).
   * @returns {void}
   */
  public setBackendApiKey(token: string): void {
    const trimmed = token.trim();
    if (!trimmed) {
      this.removeBackendApiKey();
      return;
    }
    if (this.getStorageScope() === "session") {
      safeSessionStorageSet("t1d_backend_api_key", trimmed);
      safeStorageRemove("t1d_backend_api_key");
    } else {
      safeStorageSet("t1d_backend_api_key", trimmed);
    }
  }

  /**
   * Removes the stored `T1D_API_KEY` backend Bearer authentication token from storage.
   * @returns {void}
   */
  public removeBackendApiKey(): void {
    safeSessionStorageRemove("t1d_backend_api_key");
    safeStorageRemove("t1d_backend_api_key");
  }

  /**
   * Retrieves all custom API keys stored for cloud LLM providers.
   * @returns {Record<string, string>} A mapping of provider names to configured API keys.
   */
  public getAllApiKeys(): Record<string, string> {
    const keys: Record<string, string> = {};
    const providers = ["openai", "anthropic", "google"];
    for (const p of providers) {
      const val = this.getApiKey(p);
      if (val) {
        keys[p] = val;
      }
    }
    return keys;
  }

  /**
   * Retrieves user-configured Mozilla any-llm model identifiers from localStorage.
   * @returns {string[]} Array of configured 'provider/model' or 'provider:model' identifiers.
   */
  public getConfiguredModels(): string[] {
    const raw = safeStorageGet("t1d_configured_any_llm_models");
    if (!raw) return [];
    return raw
      .split(",")
      .map((s) => s.trim())
      .filter(Boolean);
  }

  /**
   * Persists user-configured Mozilla any-llm model identifiers to localStorage.
   * @param {string} modelsCsv - Comma-separated 'provider/model' or 'provider:model' string.
   * @returns {void}
   */
  public setConfiguredModels(modelsCsv: string): void {
    const cleaned = modelsCsv
      .split(",")
      .map((s) => s.trim())
      .filter(Boolean)
      .join(", ");
    if (cleaned) {
      safeStorageSet("t1d_configured_any_llm_models", cleaned);
    } else {
      safeStorageRemove("t1d_configured_any_llm_models");
    }
  }

  /**
   * Removes user-configured Mozilla any-llm model identifiers from localStorage.
   * @returns {void}
   */
  public removeConfiguredModels(): void {
    safeStorageRemove("t1d_configured_any_llm_models");
  }
}

/**
 * Composable service handling bidirectional session synchronization with `/api/sessions`.
 */
export class SessionSyncService {
  /**
   * Pushes a single non-temporary local chat to `/api/sessions` if its messages or title changed.
   * @param {Chat} localChat - Local chat session.
   * @param {RemoteSessionPayload<Message> | undefined} remote - Matching remote session if present.
   * @returns {Promise<void>}
   */
  private async pushModifiedLocalChat(
    localChat: Chat,
    remote: RemoteSessionPayload<Message> | undefined,
  ): Promise<void> {
    if (localChat.isTemporary === true) return;
    const hasNewerMessages =
      !remote || localChat.messages.length > remote.messages.length;
    const hasModifiedTitle =
      localChat.titleModified === true && localChat.title !== remote?.title;

    if (hasNewerMessages || hasModifiedTitle) {
      await pushRemoteSession(
        localChat.id,
        localChat.title,
        localChat.messages,
      );
      if (remote) {
        remote.title = localChat.title;
      }
      localChat.titleModified = false;
    }
  }

  /**
   * Pushes modified local chats to `/api/sessions` and merges remote sessions into `chats`.
   * @param {Chat[]} chats - Mutable local chats list.
   * @param {Set<string>} deletedChatIds - Tombstone set of deleted chat IDs.
   * @returns {Promise<void>}
   */
  public async synchronizeChats(
    chats: Chat[],
    deletedChatIds: Set<string>,
  ): Promise<void> {
    const sessionsList = await fetchRemoteSessions<Message>();

    const serverMap = new Map<string, RemoteSessionPayload<Message>>();
    for (const s of sessionsList) {
      if (Boolean(s?.session_id)) {
        serverMap.set(s.session_id, s);
      }
    }

    for (const localChat of chats) {
      await this.pushModifiedLocalChat(localChat, serverMap.get(localChat.id));
    }

    for (const s of sessionsList) {
      if (!s?.session_id) continue;
      if (deletedChatIds.has(s.session_id)) {
        await deleteRemoteSession(s.session_id).catch(() => {});
        continue;
      }
      mergeRemoteSessionIntoLocalChats(chats, s);
    }
  }
}

/** Default recommended Jaeb T1D clinical trial database path. */
export const DEFAULT_RECOMMENDED_JAEB_DB_PATH = "./data/jaeb/t1d.duckdb";

/**
 * State manager for the Chat Application.
 */
export class ChatState {
  /** Collection of all chats. */
  public chats: Chat[] = [];
  /** Identifier of the currently active chat, or null if none. */
  public activeChatId: string | null = null;
  /** Counter used for auto-generating chat titles. */
  public chatCounter: number = 1;
  /** Monotonic sequence counter preventing same-millisecond chat ID collisions. */
  private idSequence: number = 0;
  /** Composable store for cloud LLM credentials and custom models. */
  public readonly credentialsStore: ProviderCredentialsStore =
    new ProviderCredentialsStore();
  /** Composable service for remote session synchronization. */
  public readonly sessionSyncService: SessionSyncService =
    new SessionSyncService();
  /** Diagnostic health status of backend and database. */
  public systemStatus: SystemStatus = {
    backendOnline: true,
    status: "healthy",
    dbConfigured: true,
    dbExists: true,
    dbConnected: true,
    dbStatusCode: "healthy",
    dbMessage: null,
    dbRemediation: null,
    tableCount: 0,
    hasInitialData: true,
    ollamaOnline: true,
    ollamaMessage: null,
    ollamaRemediation: null,
    recommendedJaebDbPath: DEFAULT_RECOMMENDED_JAEB_DB_PATH,
  };

  /** Currently selected DuckDB database path. */
  public currentDb: string = DEFAULT_DB_NAME;
  /** Default selected LLM model identifier. */
  public selectedModel: string = DEFAULT_CHAT_MODEL;
  /** List of discovered workspace DuckDB databases. */
  public availableDbs: AvailableDatabaseItem[] = [];
  /** Typed table and column schema metadata loaded from `/api/schema` for the active database. */
  public schemaTables: SchemaTableMetadata[] = [];
  /** Whether server-side chat session synchronization is enabled. */
  public serverSyncEnabled: boolean = false;
  /** Whether token-by-token SSE streaming mode is enabled. */
  public streamingMode: boolean = false;
  /** Whether LLM-generated SQL queries are automatically executed on the DuckDB read-only cursor. */
  public autoExecuteSql: boolean = true;
  /** Optional gold-standard SQL reference currently attached to the next prompt turn. */
  public activeGoldSql: string | null = null;
  /** Tombstone set of deleted chat session IDs to prevent server resurrection. */
  public deletedChatIds: Set<string> = new Set();
  /** Set of in-flight background remote session operations (delete/rename) awaited before full sync. */
  private readonly pendingSyncTasks: Set<Promise<void>> = new Set();

  /**
   * Initializes a new ChatState instance and hydrates from localStorage.
   */
  constructor() {
    this.loadFromLocalStorage();
  }

  /**
   * Tracks a background remote synchronization promise until it settles.
   * @param {Promise<unknown>} operation - Remote API promise to track.
   * @returns {void}
   */
  private trackBackgroundSync(operation: Promise<unknown>): void {
    const tracked: Promise<void> = operation
      .then(() => undefined)
      .catch(() => undefined)
      .finally(() => {
        this.pendingSyncTasks.delete(tracked);
      });
    this.pendingSyncTasks.add(tracked);
  }

  /**
   * Awaits all currently in-flight background remote sync operations.
   * @returns {Promise<void>}
   */
  public async flushPendingSyncTasks(): Promise<void> {
    if (this.pendingSyncTasks.size > 0) {
      await Promise.all(Array.from(this.pendingSyncTasks));
    }
  }

  /**
   * Saves the current state to localStorage.
   * @returns {void}
   */
  public saveToLocalStorage(): void {
    const data = {
      chats: this.chats,
      activeChatId: this.activeChatId,
      chatCounter: this.chatCounter,
      currentDb: this.currentDb,
      availableDbs: this.availableDbs,
      serverSyncEnabled: this.serverSyncEnabled,
      streamingMode: this.streamingMode,
      autoExecuteSql: this.autoExecuteSql,
      deletedChatIds: Array.from(this.deletedChatIds),
    };
    safeStorageSet("t1d_analytics_chats", JSON.stringify(data));
  }

  /**
   * Loads state from localStorage.
   * @returns {void}
   */
  private loadFromLocalStorage(): void {
    const data = safeStorageGet("t1d_analytics_chats");
    if (!data) return;
    try {
      const parsed = parsePersistedChatState(data);
      this.chats = parseStoredChatList(parsed.chats);
      this.activeChatId = parsed.activeChatId || null;
      this.chatCounter =
        parsed.chatCounter !== undefined && parsed.chatCounter > 0
          ? parsed.chatCounter
          : 1;
      if (parsed.currentDb) this.currentDb = parsed.currentDb;
      if (parsed.availableDbs) this.availableDbs = parsed.availableDbs;
      if (typeof parsed.serverSyncEnabled === "boolean") {
        this.serverSyncEnabled = parsed.serverSyncEnabled;
      }
      if (typeof parsed.streamingMode === "boolean") {
        this.streamingMode = parsed.streamingMode;
      }
      if (typeof parsed.autoExecuteSql === "boolean") {
        this.autoExecuteSql = parsed.autoExecuteSql;
      }
      if (Array.isArray(parsed.deletedChatIds)) {
        this.deletedChatIds = new Set(parsed.deletedChatIds);
      }
    } catch (e) {
      console.warn("Could not load from localStorage", e);
    }
  }

  /**
   * Updates the active database path.
   * @param {string} dbPath The selected database path.
   * @returns {void}
   */
  setCurrentDb(dbPath: string): void {
    this.currentDb = dbPath;
    this.saveToLocalStorage();
  }

  /**
   * Sets the available databases list.
   * @param {{ name: string; path: string; size_bytes: number }[]} dbs The available databases.
   * @param {string} [currentDb] Optional active database path.
   * @returns {void}
   */
  setAvailableDbs(
    dbs: { name: string; path: string; size_bytes: number }[],
    currentDb?: string,
  ): void {
    this.availableDbs = dbs;
    if (currentDb && !this.currentDb) {
      this.currentDb = currentDb;
    }
    this.saveToLocalStorage();
  }

  /**
   * Sets whether server sync is enabled.
   * @param {boolean} enabled Whether sync is active.
   * @returns {void}
   */
  setServerSyncEnabled(enabled: boolean): void {
    this.serverSyncEnabled = enabled;
    this.saveToLocalStorage();
  }

  /**
   * Sets whether LLM-generated SQL queries are automatically executed on the DuckDB read-only cursor.
   * @param {boolean} enabled Whether automatic SQL execution is active.
   * @returns {void}
   */
  setAutoExecuteSql(enabled: boolean): void {
    this.autoExecuteSql = enabled;
    this.saveToLocalStorage();
  }

  /**
   * Sets or clears the gold-standard SQL reference attached to the next prompt turn.
   * @param {string | null} goldSql Gold-standard SQL string or null to clear.
   * @returns {void}
   */
  setActiveGoldSql(goldSql: string | null): void {
    this.activeGoldSql = goldSql && goldSql.trim() ? goldSql.trim() : null;
  }

  /**
   * Synchronizes chat sessions bidirectionally with the server `/api/sessions`.
   * Pushes local chats to the server and merges server sessions into local state.
   * If the server is unreachable, gracefully falls back to local state and saves to localStorage.
   * @returns {Promise<boolean>} True if server sync succeeded, false if fell back to offline.
   */
  public async syncWithServer(): Promise<boolean> {
    try {
      await this.flushPendingSyncTasks();
      await this.sessionSyncService.synchronizeChats(
        this.chats,
        this.deletedChatIds,
      );

      const routeChatId =
        typeof window !== "undefined"
          ? extractChatIdFromLocation(window.location)
          : null;
      const firstChat = this.chats[0];
      if (routeChatId && this.chats.some((c) => c.id === routeChatId)) {
        this.activeChatId = routeChatId;
      } else if (!this.activeChatId && firstChat) {
        this.activeChatId = firstChat.id;
      }

      this.serverSyncEnabled = true;
      this.saveToLocalStorage();
      return true;
    } catch (e) {
      console.warn(
        "Server sync failed, falling back to offline localStorage:",
        e,
      );
      this.loadFromLocalStorage();
      this.saveToLocalStorage();
      return false;
    }
  }

  /**
   * Loads and activates a shared chat session by ID, fetching from `/api/sessions/{sessionId}`
   * when the session is not yet present in local state.
   * @param {string} sessionId - Unique chat session identifier from a `/chats/${ID}` URL.
   * @returns {Promise<Chat | null>} The activated Chat instance, or null if not found.
   */
  public async loadSharedSession(sessionId: string): Promise<Chat | null> {
    const existing = this.chats.find((c) => c.id === sessionId);
    if (existing) {
      this.activeChatId = existing.id;
      this.saveToLocalStorage();
      return existing;
    }
    try {
      const remote = await fetchRemoteSessionById<Message>(sessionId);
      if (!remote || !remote.session_id) {
        return null;
      }
      this.deletedChatIds.delete(remote.session_id);
      const merged = mergeRemoteSessionIntoLocalChats(this.chats, remote);
      this.activeChatId = remote.session_id;
      this.saveToLocalStorage();
      return merged;
    } catch {
      return null;
    }
  }

  /**
   * Creates a new chat and sets it as active.
   * @param {string} [title] Optional title for the chat.
   * @param {boolean} [isTemporary=false] Whether the chat should be treated as temporary.
   * @returns {Chat} The newly created chat.
   */
  createChat(title?: string, isTemporary: boolean = false): Chat {
    const chat: Chat = {
      id: `chat-${Date.now()}-${++this.idSequence}-${Math.floor(Math.random() * 1000)}`,
      title:
        title || i18next.t("app.chatNumber", { count: this.chatCounter++ }),
      messages: [],
      model: DEFAULT_CHAT_MODEL,
      isTemporary,
      updatedAt: Date.now(),
    };
    this.chats.push(chat);
    this.activeChatId = chat.id;
    this.saveToLocalStorage();
    return chat;
  }

  /**
   * Retrieves the currently active chat.
   * @returns {Chat | null} The active chat or null if none is active.
   */
  getActiveChat(): Chat | null {
    return this.chats.find((c) => c.id === this.activeChatId) || null;
  }

  /**
   * Sets the active chat by ID.
   * @param {string} id The ID of the chat to make active.
   * @returns {void}
   */
  setActiveChat(id: string): void {
    if (this.chats.some((c) => c.id === id)) {
      this.activeChatId = id;
      this.saveToLocalStorage();
    }
  }

  /**
   * Deletes a chat by ID. If the active chat is deleted, clears the active chat.
   * @param {string} id The ID of the chat to delete.
   * @returns {void}
   */
  deleteChat(id: string): void {
    const chat = this.chats.find((c) => c.id === id);
    if (!chat || chat.isTemporary) {
      return;
    }
    this.deletedChatIds.add(id);
    if (this.serverSyncEnabled) {
      this.trackBackgroundSync(deleteRemoteSession(id));
    }
    this.chats = this.chats.filter((c) => c.id !== id);
    if (this.activeChatId === id) {
      const lastChat = this.chats[this.chats.length - 1];
      this.activeChatId = lastChat ? lastChat.id : null;
    }

    if (this.chats.length === 0) {
      this.createChat(i18next.t("app.temporaryChat"), true);
    }
    this.saveToLocalStorage();
  }

  /**
   * Renames a chat.
   * @param {string} id The ID of the chat to rename.
   * @param {string} newTitle The new title.
   * @returns {void}
   */
  renameChat(id: string, newTitle: string): void {
    const chat = this.chats.find((c) => c.id === id);
    if (chat && newTitle.trim()) {
      chat.title = newTitle.trim();
      chat.updatedAt = Date.now();
      chat.titleModified = true;
      this.saveToLocalStorage();
      if (this.serverSyncEnabled && !chat.isTemporary) {
        this.trackBackgroundSync(
          pushRemoteSession(chat.id, chat.title, chat.messages),
        );
      }
    }
  }

  /**
   * Duplicates a chat, copying its messages and model, and appends " (Copy)" to the title.
   * @param {string} id The ID of the chat to duplicate.
   * @returns {Chat | null} The duplicated chat, or null if the original wasn't found.
   */
  duplicateChat(id: string): Chat | null {
    const original = this.chats.find((c) => c.id === id);
    if (!original) {
      return null;
    }

    const duplicate: Chat = {
      id: `chat-${Date.now()}-${++this.idSequence}-${Math.floor(Math.random() * 1000)}`,
      title: i18next.t("app.copyOf", { title: original.title }),
      messages: structuredClone(original.messages),
      model: original.model,
      isTemporary: false,
    };
    this.chats.push(duplicate);
    this.activeChatId = duplicate.id;
    this.saveToLocalStorage();
    return duplicate;
  }

  /**
   * Adds a message to the active chat.
   * @param {Message} message The message to add.
   * @returns {void}
   */
  addMessageToActiveChat(message: Message): void {
    const chat = this.getActiveChat();
    if (chat) {
      chat.messages.push(message);
      chat.updatedAt = Date.now();
      if (chat.isTemporary) {
        chat.isTemporary = false;
        chat.title = i18next.t("app.chatNumber", { count: this.chatCounter++ });
      }
      this.saveToLocalStorage();
    }
  }

  /**
   * Updates an existing message in place and persists state to localStorage.
   * @param {Message} targetMessage The target message instance to update.
   * @param {Partial<Message>} patch Partial message properties to apply.
   * @returns {void}
   */
  updateMessageInActiveChat(
    targetMessage: Message,
    patch: Partial<Message>,
  ): void {
    Object.assign(targetMessage, patch);
    this.saveToLocalStorage();
  }

  /**
   * Sets the model for the active chat.
   * @param {string} model The model identifier.
   * @returns {void}
   */
  setActiveChatModel(model: string): void {
    this.selectedModel = model;
    const chat = this.getActiveChat();
    if (chat) {
      chat.model = model;
      this.saveToLocalStorage();
    }
  }

  /**
   * Sets the global and active chat selected model.
   * @param {string} model The model identifier.
   * @returns {void}
   */
  setSelectedModel(model: string): void {
    this.setActiveChatModel(model);
  }

  /**
   * Sets the provider for the active chat.
   * @param {string | undefined} provider The provider identifier.
   * @returns {void}
   */
  setActiveChatProvider(provider?: string): void {
    const chat = this.getActiveChat();
    if (chat) {
      if (provider !== undefined) {
        chat.provider = provider;
      } else {
        delete chat.provider;
      }
      this.saveToLocalStorage();
    }
  }

  /**
   * Retrieves a stored custom API key for a cloud LLM provider from localStorage.
   * @param {string} provider The provider identifier (e.g. 'openai', 'anthropic', 'google').
   * @returns {string | null} The API key if stored, otherwise null.
   */
  getApiKey(provider: string): string | null {
    return this.credentialsStore.getApiKey(provider);
  }

  /**
   * Stores a custom API key for a cloud LLM provider in localStorage.
   * @param {string} provider The provider identifier.
   * @param {string} key The API key to store.
   * @returns {void}
   */
  setApiKey(provider: string, key: string): void {
    this.credentialsStore.setApiKey(provider, key);
  }

  /**
   * Removes a stored custom API key for a cloud LLM provider from localStorage.
   * @param {string} provider The provider identifier.
   * @returns {void}
   */
  removeApiKey(provider: string): void {
    this.credentialsStore.removeApiKey(provider);
  }

  /**
   * Retrieves all custom API keys stored for cloud LLM providers.
   * @returns {Record<string, string>} A mapping of provider names to configured API keys.
   */
  getAllApiKeys(): Record<string, string> {
    return this.credentialsStore.getAllApiKeys();
  }

  /**
   * Retrieves user-configured Mozilla any-llm model identifiers from localStorage.
   * @returns {string[]} Array of configured 'provider/model' or 'provider:model' identifiers.
   */
  getConfiguredModels(): string[] {
    return this.credentialsStore.getConfiguredModels();
  }

  /**
   * Persists user-configured Mozilla any-llm model identifiers to localStorage.
   * @param {string} modelsCsv Comma-separated 'provider/model' or 'provider:model' string.
   * @returns {void}
   */
  setConfiguredModels(modelsCsv: string): void {
    this.credentialsStore.setConfiguredModels(modelsCsv);
  }

  /**
   * Removes user-configured Mozilla any-llm model identifiers from localStorage.
   * @returns {void}
   */
  removeConfiguredModels(): void {
    this.credentialsStore.removeConfiguredModels();
  }

  /**
   * Stores the loaded database table and column schema metadata for the active database.
   * @param {SchemaTableMetadata[]} tables - Loaded schema table definitions.
   * @returns {void}
   */
  setSchemaTables(tables: SchemaTableMetadata[]): void {
    this.schemaTables = tables.map((t) => ({
      name: t.name,
      columns: [...t.columns],
    }));
  }

  /**
   * Returns column names for `tableName` from loaded `schemaTables`, or undefined when unknown.
   * @param {string} tableName - Target table identifier.
   * @returns {string[] | undefined} Ordered column names if the table is present in `schemaTables`.
   */
  getTableColumns(tableName: string): string[] | undefined {
    const target = tableName.trim().toLowerCase();
    if (!target) {
      return undefined;
    }
    const found = this.schemaTables.find(
      (t) => t.name.toLowerCase() === target,
    );
    return found ? found.columns.map((c) => c.name) : undefined;
  }

  /**
   * Updates the diagnostic status of the system.
   * @param {Partial<SystemStatus>} status The partial status updates.
   * @returns {void}
   */
  setSystemStatus(status: Partial<SystemStatus>): void {
    this.systemStatus = { ...this.systemStatus, ...status };
  }
}
