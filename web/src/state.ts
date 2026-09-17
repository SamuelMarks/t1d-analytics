/**
 * @file state.ts
 * Manages the application state for the chat UI.
 */
import i18next from "./i18n";

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
}

/**
 * Relationship definition between two clinical tables.
 */
export interface TableRelationship {
  /** The source primary table name. */
  fromTable: string;
  /** The target secondary table name. */
  toTable: string;
  /** Primary key column in the source table. */
  fromColumn: string;
  /** Foreign key column in the target table. */
  toColumn: string;
  /** Suggested default join type. */
  defaultJoinType: "INNER JOIN" | "LEFT JOIN";
}

/**
 * Standard clinical relational ontology mapping primary and foreign keys.
 */
export const CLINICAL_RELATIONSHIPS: TableRelationship[] = [
  {
    fromTable: "patients",
    toTable: "cgms",
    fromColumn: "patient_id",
    toColumn: "patient_id",
    defaultJoinType: "INNER JOIN",
  },
  {
    fromTable: "patients",
    toTable: "dclp3",
    fromColumn: "patient_id",
    toColumn: "patient_id",
    defaultJoinType: "INNER JOIN",
  },
  {
    fromTable: "patients",
    toTable: "visits",
    fromColumn: "patient_id",
    toColumn: "patient_id",
    defaultJoinType: "LEFT JOIN",
  },
  {
    fromTable: "patients",
    toTable: "devices",
    fromColumn: "patient_id",
    toColumn: "patient_id",
    defaultJoinType: "LEFT JOIN",
  },
  {
    fromTable: "patients",
    toTable: "labs",
    fromColumn: "patient_id",
    toColumn: "patient_id",
    defaultJoinType: "LEFT JOIN",
  },
  {
    fromTable: "cgms",
    toTable: "dclp3",
    fromColumn: "patient_id",
    toColumn: "patient_id",
    defaultJoinType: "INNER JOIN",
  },
];

/**
 * Resolve foreign key join matching between two tables using known ontology or shared ID columns.
 * @param {string} table1 The primary table name.
 * @param {string} table2 The secondary table name.
 * @param {string[]} [table1Cols] Optional known columns of table 1.
 * @param {string[]} [table2Cols] Optional known columns of table 2.
 * @returns {TableRelationship | null} The resolved relationship or null if none discovered.
 */
export function resolveTableJoin(
  table1: string,
  table2: string,
  table1Cols?: string[],
  table2Cols?: string[],
): TableRelationship | null {
  const t1 = table1.toLowerCase();
  const t2 = table2.toLowerCase();

  // 1. Direct match in clinical ontology
  const found = CLINICAL_RELATIONSHIPS.find(
    (r) =>
      (r.fromTable.toLowerCase() === t1 && r.toTable.toLowerCase() === t2) ||
      (r.fromTable.toLowerCase() === t2 && r.toTable.toLowerCase() === t1),
  );
  if (found) {
    if (found.fromTable.toLowerCase() === t1) {
      return found;
    }
    return {
      fromTable: table1,
      toTable: table2,
      fromColumn: found.toColumn,
      toColumn: found.fromColumn,
      defaultJoinType: found.defaultJoinType,
    };
  }

  // 2. Ontology-aware common clinical keys if columns are available
  if (table1Cols && table2Cols) {
    const candidateKeys = [
      "patient_id",
      "subject_id",
      "pt_id",
      "id",
      "study_id",
      "device_id",
    ];
    for (const key of candidateKeys) {
      const match1 = table1Cols.find((c) => c.toLowerCase() === key);
      const match2 = table2Cols.find((c) => c.toLowerCase() === key);
      if (match1 && match2) {
        return {
          fromTable: table1,
          toTable: table2,
          fromColumn: match1,
          toColumn: match2,
          defaultJoinType: "INNER JOIN",
        };
      }
    }
  }

  // 3. Heuristic match if table name contains patient
  if (t1.includes("patient") || t2.includes("patient")) {
    return {
      fromTable: table1,
      toTable: table2,
      fromColumn: "patient_id",
      toColumn: "patient_id",
      defaultJoinType: "INNER JOIN",
    };
  }

  return null;
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

  public currentDb: string = "t1d.duckdb";
  public availableDbs: { name: string; path: string; size_bytes: number }[] =
    [];
  public serverSyncEnabled: boolean = false;
  /** Whether token-by-token SSE streaming mode is enabled. */
  public streamingMode: boolean = false;
  /** Tombstone set of deleted chat session IDs to prevent server resurrection. */
  public deletedChatIds: Set<string> = new Set();

  constructor() {
    this.loadFromLocalStorage();
  }

  /**
   * Saves the current state to localStorage.
   */
  public saveToLocalStorage(): void {
    try {
      if (typeof localStorage === "undefined") return;
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
      localStorage.setItem("t1d_analytics_chats", JSON.stringify(data));
    } catch (e) {
      console.warn("Could not save to localStorage", e);
    }
  }

  /**
   * Loads state from localStorage.
   */
  private loadFromLocalStorage(): void {
    try {
      if (typeof localStorage === "undefined") return;
      const data = localStorage.getItem("t1d_analytics_chats");
      if (data) {
        const parsed = JSON.parse(data);
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
      }
    } catch (e) {
      console.warn("Could not load from localStorage", e);
    }
  }

  /**
   * Updates the active database path.
   * @param {string} dbPath The selected database path.
   */
  setCurrentDb(dbPath: string): void {
    this.currentDb = dbPath;
    this.saveToLocalStorage();
  }

  /**
   * Sets the available databases list.
   * @param {{ name: string; path: string; size_bytes: number }[]} dbs The available databases.
   * @param {string} [currentDb] Optional active database path.
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

      // 2. Upload any local non-temporary chats not on server or having newer local messages
      for (const localChat of this.chats) {
        if (localChat.isTemporary) continue;
        const remote = serverMap.get(localChat.id);
        if (!remote || localChat.messages.length > remote.messages.length) {
          await fetch("/api/sessions", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              session_id: localChat.id,
              title: localChat.title,
              messages: localChat.messages,
            }),
          });
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
        if (!local) {
          this.chats.push({
            id: s.session_id,
            title: s.title,
            messages: s.messages || [],
            model: "gemma4",
            isTemporary: false,
          });
        } else {
          if ((s.messages?.length || 0) > local.messages.length) {
            local.messages = s.messages;
          }
          if (s.title && s.title !== local.title) {
            local.title = s.title;
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
   */
  renameChat(id: string, newTitle: string): void {
    const chat = this.chats.find((c) => c.id === id);
    if (chat && newTitle.trim()) {
      chat.title = newTitle.trim();
      this.saveToLocalStorage();
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
   */
  addMessageToActiveChat(message: Message): void {
    const chat = this.getActiveChat();
    if (chat) {
      chat.messages.push(message);
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
    try {
      return localStorage.getItem(`t1d_api_key_${provider.toLowerCase()}`);
    } catch {
      return null;
    }
  }

  /**
   * Stores a custom API key for a cloud LLM provider in localStorage.
   * @param {string} provider The provider identifier.
   * @param {string} key The API key to store.
   */
  setApiKey(provider: string, key: string): void {
    try {
      localStorage.setItem(`t1d_api_key_${provider.toLowerCase()}`, key.trim());
    } catch (e) {
      console.warn("Could not save API key to localStorage", e);
    }
  }

  /**
   * Removes a stored custom API key for a cloud LLM provider from localStorage.
   * @param {string} provider The provider identifier.
   */
  removeApiKey(provider: string): void {
    try {
      localStorage.removeItem(`t1d_api_key_${provider.toLowerCase()}`);
    } catch (e) {
      console.warn("Could not remove API key from localStorage", e);
    }
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
   * Updates the diagnostic health status of the system.
   * @param {Partial<SystemStatus>} status The partial status updates.
   */
  setSystemStatus(status: Partial<SystemStatus>): void {
    this.systemStatus = { ...this.systemStatus, ...status };
  }
}
