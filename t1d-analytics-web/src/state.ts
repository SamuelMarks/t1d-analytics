/**
 * @file state.ts
 * @module state
 * Manages the application state for the chat UI.
 */
import i18next from "./i18n";
import {
  type TableRelationship,
  CLINICAL_RELATIONSHIPS,
  resolveTableJoin,
} from "./cohort-sql";

export { type TableRelationship, CLINICAL_RELATIONSHIPS, resolveTableJoin };

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
  /** Persisted tombstone list of deleted session IDs. */
  deletedChatIds?: string[];
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
  return parsed;
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
 * State manager for the Chat Application.
 */
export class ChatState {
  /** Collection of all chats. */
  public chats: Chat[] = [];
  /** Identifier of the currently active chat, or null if none. */
  public activeChatId: string | null = null;
  /** Counter used for auto-generating chat titles. */
  public chatCounter: number = 1;
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
  };

  /** Currently selected DuckDB database path. */
  public currentDb: string = "t1d.duckdb";
  /** List of discovered workspace DuckDB databases. */
  public availableDbs: AvailableDatabaseItem[] = [];
  /** Whether server-side chat session synchronization is enabled. */
  public serverSyncEnabled: boolean = false;
  /** Whether token-by-token SSE streaming mode is enabled. */
  public streamingMode: boolean = false;
  /** Tombstone set of deleted chat session IDs to prevent server resurrection. */
  public deletedChatIds: Set<string> = new Set();

  /**
   * Initializes a new ChatState instance and hydrates from localStorage.
   */
  constructor() {
    this.loadFromLocalStorage();
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
      this.chats = parsed.chats || [];
      this.activeChatId = parsed.activeChatId || null;
      this.chatCounter = parsed.chatCounter || 1;
      if (parsed.currentDb) this.currentDb = parsed.currentDb;
      if (parsed.availableDbs) this.availableDbs = parsed.availableDbs;
      if (typeof parsed.serverSyncEnabled === "boolean") {
        this.serverSyncEnabled = parsed.serverSyncEnabled;
      }
      if (typeof parsed.streamingMode === "boolean") {
        this.streamingMode = parsed.streamingMode;
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
   * Synchronizes chat sessions bidirectionally with the server `/api/sessions`.
   * Pushes local chats to the server and merges server sessions into local state.
   * If the server is unreachable, gracefully falls back to local state and saves to localStorage.
   * @returns {Promise<boolean>} True if server sync succeeded, false if fell back to offline.
   */
  public async syncWithServer(): Promise<boolean> {
    try {
      // 1. Fetch remote sessions from server
      const resp = await fetch("/api/sessions");
      if (!resp.ok) {
        throw new Error(`Server returned status ${resp.status}`);
      }
      const rawData = await resp.json();
      const sessionsList: Array<{
        session_id: string;
        title: string;
        created_at: string;
        updated_at: string;
        messages: Message[];
      }> = Array.isArray(rawData)
        ? rawData
        : Array.isArray(rawData?.sessions)
          ? rawData.sessions
          : [];

      const serverMap = new Map<
        string,
        {
          session_id: string;
          title: string;
          created_at: string;
          updated_at: string;
          messages: Message[];
        }
      >();
      for (const s of sessionsList) {
        if (s && s.session_id) {
          serverMap.set(s.session_id, s);
        }
      }

      // 2. Upload any local non-temporary chats not on server or having newer local messages or modified titles
      for (const localChat of this.chats) {
        if (localChat.isTemporary) continue;
        const remote = serverMap.get(localChat.id);
        const hasNewerMessages =
          !remote || localChat.messages.length > remote.messages.length;
        const hasModifiedTitle =
          Boolean(localChat.titleModified) && localChat.title !== remote?.title;

        if (hasNewerMessages || hasModifiedTitle) {
          await fetch("/api/sessions", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              session_id: localChat.id,
              title: localChat.title,
              messages: localChat.messages,
            }),
          });
          localChat.titleModified = false;
        }
      }

      // 3. Merge server sessions into local chats
      for (const s of sessionsList) {
        if (!s || !s.session_id) continue;
        if (this.deletedChatIds.has(s.session_id)) {
          fetch(`/api/sessions/${encodeURIComponent(s.session_id)}`, {
            method: "DELETE",
          }).catch(() => {});
          continue;
        }
        const local = this.chats.find((c) => c.id === s.session_id);
        const remoteTime = s.updated_at ? new Date(s.updated_at).getTime() : 0;
        if (!local) {
          this.chats.push({
            id: s.session_id,
            title: s.title,
            messages: s.messages || [],
            model: "gemma4",
            isTemporary: false,
            updatedAt: remoteTime,
          });
        } else {
          if ((s.messages?.length || 0) > local.messages.length) {
            local.messages = s.messages;
            local.updatedAt = remoteTime;
          }
          if (s.title && s.title !== local.title && !local.titleModified) {
            local.title = s.title;
            local.updatedAt = remoteTime;
          }
        }
      }

      if (!this.activeChatId && this.chats.length > 0) {
        this.activeChatId = this.chats[0].id;
      }

      this.serverSyncEnabled = true;
      this.saveToLocalStorage();
      return true;
    } catch (e) {
      // Offline fallback: load from localStorage and persist locally
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
   * Creates a new chat and sets it as active.

   * @param {string} [title] Optional title for the chat.
   * @param {boolean} [isTemporary=false] Whether the chat should be treated as temporary.
   * @returns {Chat} The newly created chat.
   */
  createChat(title?: string, isTemporary: boolean = false): Chat {
    const chat: Chat = {
      id: `chat-${Date.now()}-${Math.floor(Math.random() * 1000)}`,
      title:
        title || i18next.t("app.chatNumber", { count: this.chatCounter++ }),
      messages: [],
      model: "gemma4",
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
      fetch(`/api/sessions/${encodeURIComponent(id)}`, {
        method: "DELETE",
      }).catch(() => {});
    }
    this.chats = this.chats.filter((c) => c.id !== id);
    if (this.activeChatId === id) {
      this.activeChatId =
        this.chats.length > 0 ? this.chats[this.chats.length - 1].id : null;
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
        fetch("/api/sessions", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            session_id: chat.id,
            title: chat.title,
            messages: chat.messages,
          }),
        }).catch(() => {});
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
      id: `chat-${Date.now()}-${Math.floor(Math.random() * 1000)}`,
      title: i18next.t("app.copyOf", { title: original.title }),
      messages: JSON.parse(JSON.stringify(original.messages)), // Deep copy
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
   * Sets the model for the active chat.
   * @param {string} model The model identifier.
   * @returns {void}
   */
  setActiveChatModel(model: string): void {
    const chat = this.getActiveChat();
    if (chat) {
      chat.model = model;
      this.saveToLocalStorage();
    }
  }

  /**
   * Sets the provider for the active chat.
   * @param {string | undefined} provider The provider identifier.
   * @returns {void}
   */
  setActiveChatProvider(provider?: string): void {
    const chat = this.getActiveChat();
    if (chat) {
      chat.provider = provider;
      this.saveToLocalStorage();
    }
  }

  /**
   * Retrieves a stored custom API key for a cloud LLM provider from localStorage.
   * @param {string} provider The provider identifier (e.g. 'openai', 'anthropic', 'google').
   * @returns {string | null} The API key if stored, otherwise null.
   */
  getApiKey(provider: string): string | null {
    return safeStorageGet(`t1d_api_key_${provider.toLowerCase()}`);
  }

  /**
   * Stores a custom API key for a cloud LLM provider in localStorage.
   * @param {string} provider The provider identifier.
   * @param {string} key The API key to store.
   * @returns {void}
   */
  setApiKey(provider: string, key: string): void {
    safeStorageSet(`t1d_api_key_${provider.toLowerCase()}`, key.trim());
  }

  /**
   * Removes a stored custom API key for a cloud LLM provider from localStorage.
   * @param {string} provider The provider identifier.
   * @returns {void}
   */
  removeApiKey(provider: string): void {
    safeStorageRemove(`t1d_api_key_${provider.toLowerCase()}`);
  }

  /**
   * Retrieves all custom API keys stored for cloud LLM providers.
   * @returns {Record<string, string>} A mapping of provider names to configured API keys.
   */
  getAllApiKeys(): Record<string, string> {
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
  getConfiguredModels(): string[] {
    const raw = safeStorageGet("t1d_configured_any_llm_models");
    if (!raw) return [];
    return raw
      .split(",")
      .map((s) => s.trim())
      .filter(Boolean);
  }

  /**
   * Persists user-configured Mozilla any-llm model identifiers to localStorage.
   * @param {string} modelsCsv Comma-separated 'provider/model' or 'provider:model' string.
   * @returns {void}
   */
  setConfiguredModels(modelsCsv: string): void {
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
  removeConfiguredModels(): void {
    safeStorageRemove("t1d_configured_any_llm_models");
  }

  /**
   * Updates the diagnostic health status of the system.
   * @param {Partial<SystemStatus>} status The partial status updates.
   * @returns {void}
   */
  setSystemStatus(status: Partial<SystemStatus>): void {
    this.systemStatus = { ...this.systemStatus, ...status };
  }
}
