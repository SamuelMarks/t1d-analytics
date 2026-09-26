import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import { ChatState, resolveTableJoin } from "../src/state";

describe("ChatState", () => {
  let state: ChatState;

  beforeEach(() => {
    localStorage.clear();
    state = new ChatState();
  });

  afterEach(() => {
    localStorage.clear();
  });

  it("initializes empty state", () => {
    expect(state.chats).toEqual([]);
    expect(state.activeChatId).toBeNull();
    expect(state.getActiveChat()).toBeNull();
  });

  it("creates a new chat and sets it as active", () => {
    const chat = state.createChat();
    expect(state.chats.length).toBe(1);
    expect(state.activeChatId).toBe(chat.id);
    expect(chat.title).toBe("Chat #1");
    expect(chat.model).toBe("gemma4");
  });

  it("retrieves the active chat", () => {
    state.createChat();
    const chat2 = state.createChat();
    expect(state.getActiveChat()).toEqual(chat2);
  });

  it("sets the active chat by id", () => {
    const chat1 = state.createChat();
    state.createChat();
    state.setActiveChat(chat1.id);
    expect(state.activeChatId).toBe(chat1.id);
    expect(state.getActiveChat()).toEqual(chat1);
  });

  it("ignores setting active chat to invalid id", () => {
    const chat = state.createChat();
    state.setActiveChat("invalid-id");
    expect(state.activeChatId).toBe(chat.id);
  });

  it("deletes a chat and updates active chat if necessary", () => {
    const chat1 = state.createChat();
    const chat2 = state.createChat();

    // Delete non-active chat
    state.deleteChat(chat1.id);
    expect(state.chats.length).toBe(1);
    expect(state.activeChatId).toBe(chat2.id);

    // Delete active chat with no remaining chats
    state.deleteChat(chat2.id);
    expect(state.chats.length).toBe(1); // Temporary chat created!
    expect(state.chats[0].title).toBe("Temporary chat");
  });

  it("deletes active chat and sets next available chat as active", () => {
    state.createChat();
    const chat2 = state.createChat();
    const chat3 = state.createChat();
    // currently chat3 is active

    state.deleteChat(chat3.id);
    expect(state.chats.length).toBe(2);
    // Should fallback to chat2 (the last one remaining)
    expect(state.activeChatId).toBe(chat2.id);
  });

  it("renames a chat", () => {
    const chat = state.createChat();
    state.renameChat(chat.id, "My Custom Title");
    expect(state.chats[0].title).toBe("My Custom Title");

    // With server sync enabled, triggers POST fetch
    state.serverSyncEnabled = true;
    const fetchSpy = vi.spyOn(globalThis, "fetch").mockResolvedValueOnce({
      ok: true,
      json: async () => ({ status: "ok" }),
    } as any);
    state.renameChat(chat.id, "Synced Title");
    expect(fetchSpy).toHaveBeenCalled();

    // Catch branch when fetch rejects
    fetchSpy.mockRejectedValueOnce(new Error("Network failed"));
    state.renameChat(chat.id, "Catch Branch Title");
    expect(fetchSpy).toHaveBeenCalledTimes(2);
    fetchSpy.mockRestore();
    state.serverSyncEnabled = false;
  });

  it("does not rename chat if title is empty or chat not found", () => {
    const chat = state.createChat();
    state.renameChat(chat.id, "   ");
    expect(state.chats[0].title).toBe("Chat #1");

    state.renameChat("invalid", "New Title");
    expect(state.chats[0].title).toBe("Chat #1");
  });

  it("duplicates a chat", () => {
    const chat = state.createChat();
    state.addMessageToActiveChat({ role: "user", content: "Hello" });

    const duplicate = state.duplicateChat(chat.id);
    expect(duplicate).not.toBeNull();
    expect(duplicate?.title).toBe("Chat #1 (Copy)");
    expect(duplicate?.messages.length).toBe(1);
    expect(state.chats.length).toBe(2);
    expect(state.activeChatId).toBe(duplicate?.id);

    // Ensure deep copy
    state.addMessageToActiveChat({ role: "assistant", content: "Hi" });
    expect(duplicate?.messages.length).toBe(2);
    expect(chat.messages.length).toBe(1);
  });

  it("returns null when duplicating non-existent chat", () => {
    expect(state.duplicateChat("invalid")).toBeNull();
  });

  it("adds message to active chat", () => {
    state.createChat();
    state.addMessageToActiveChat({ role: "user", content: "Test message" });
    const chat = state.getActiveChat();
    expect(chat?.messages[0]).toEqual({
      role: "user",
      content: "Test message",
    });
  });

  it("ignores adding message if no active chat", () => {
    state.addMessageToActiveChat({ role: "user", content: "Test" });
    expect(state.chats.length).toBe(0);
  });

  it("sets model for active chat", () => {
    state.createChat();
    state.setActiveChatModel("sql");
    expect(state.getActiveChat()?.model).toBe("sql");
  });

  it("ignores setting model if no active chat", () => {
    state.setActiveChatModel("sql"); // should not throw
  });

  it("ignores deleting a temporary chat", () => {
    const chat = state.createChat("Temporary chat", true);
    state.deleteChat(chat.id);
    expect(state.chats.length).toBe(1);
  });

  it("adding message to temporary chat sets isTemporary to false and updates title", () => {
    const chat = state.createChat("Temporary chat", true);
    state.addMessageToActiveChat({ role: "user", content: "hello" });
    expect(chat.isTemporary).toBe(false);
    expect(chat.title).toBe("Chat #1");
  });

  it("handles localStorage setItem throwing an error", () => {
    const setItem = localStorage.setItem;
    localStorage.setItem = () => {
      throw new Error("Quota exceeded");
    };
    state.createChat();
    localStorage.setItem = setItem;
  });

  it("handles localStorage getItem throwing an error", () => {
    const getItem = localStorage.getItem;
    localStorage.getItem = () => {
      throw new Error("Access denied");
    };
    new ChatState();
    localStorage.getItem = getItem;
  });

  it("handles undefined localStorage gracefully", () => {
    const originalLocalStorage = global.localStorage;
    Object.defineProperty(global, "localStorage", {
      value: undefined,
      configurable: true,
    });

    const newState = new ChatState();
    newState.createChat();

    Object.defineProperty(global, "localStorage", {
      value: originalLocalStorage,
      configurable: true,
    });
  });
  it("handles loading partial state from localStorage", () => {
    localStorage.setItem("t1d_analytics_chats", JSON.stringify({}));
    const partialState = new ChatState();
    expect(partialState.chats).toEqual([]);
    expect(partialState.activeChatId).toBeNull();
    expect(partialState.chatCounter).toBe(1);
    expect(partialState.currentDb).toBe("t1d.duckdb");
    expect(partialState.availableDbs).toEqual([]);
    expect(partialState.serverSyncEnabled).toBe(false);
  });

  it("handles setting and loading database and server sync state", () => {
    state.currentDb = "";
    state.setAvailableDbs(
      [{ name: "study1", path: "/data/study1.duckdb", size_bytes: 1024 }],
      "study1",
    );
    expect(state.currentDb).toBe("study1");
    expect(state.availableDbs.length).toBe(1);

    state.setCurrentDb("custom_study.duckdb");
    expect(state.currentDb).toBe("custom_study.duckdb");

    state.setServerSyncEnabled(true);
    expect(state.serverSyncEnabled).toBe(true);

    const reloaded = new ChatState();
    expect(reloaded.currentDb).toBe("custom_study.duckdb");
    expect(reloaded.availableDbs.length).toBe(1);
    expect(reloaded.serverSyncEnabled).toBe(true);
  });

  it("handles syncWithServer success with remote merge and local upload", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch");
    // Local chat with 2 messages
    const chat1 = state.createChat("Local Chat");
    state.addMessageToActiveChat({ role: "user", content: "hello" });
    state.addMessageToActiveChat({ role: "assistant", content: "hi" });

    // Server returns one existing session with 1 message, and one new remote session
    fetchSpy.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        sessions: [
          {
            session_id: chat1.id,
            title: "Local Chat",
            created_at: "2026-01-01",
            updated_at: "2026-01-01",
            messages: [{ role: "user", content: "hello" }],
          },
          {
            session_id: "remote-session-2",
            title: "Remote Session",
            created_at: "2026-01-01",
            updated_at: "2026-01-01",
            messages: [{ role: "user", content: "remote query" }],
          },
          null as any,
          {} as any,
        ],
      }),
    } as any);

    // POST /api/sessions mock response for chat1 upload
    fetchSpy.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ session_id: chat1.id, status: "created" }),
    } as any);

    const ok = await state.syncWithServer();
    expect(ok).toBe(true);
    expect(state.serverSyncEnabled).toBe(true);
    expect(state.chats.some((c) => c.id === "remote-session-2")).toBe(true);

    // ActiveChatId set if empty
    state.activeChatId = null;
    fetchSpy.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        sessions: [
          {
            session_id: chat1.id,
            title: "Local Chat Updated",
            created_at: "2026-01-01",
            updated_at: "2026-01-01",
            messages: [
              { role: "user", content: "hello" },
              { role: "assistant", content: "hi" },
              { role: "user", content: "more" },
            ],
          },
        ],
      }),
    } as any);
    // Remote has 3 messages, remote-session-2 in local has 1 message so local remote-session-2 might trigger POST if not on server
    fetchSpy.mockResolvedValue({
      ok: true,
      json: async () => ({ status: "ok" }),
    } as any);
    await state.syncWithServer();
    expect(state.activeChatId).toBe(state.chats[0].id);
    expect(chat1.title).toBe("Local Chat Updated");
    expect(chat1.messages.length).toBe(3);
    fetchSpy.mockRestore();
  });

  it("handles syncWithServer network or server error with offline fallback", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch");
    fetchSpy.mockResolvedValueOnce({
      ok: false,
      status: 503,
    } as any);

    const ok = await state.syncWithServer();
    expect(ok).toBe(false);

    // Also test rejected promise
    fetchSpy.mockRejectedValueOnce(new Error("Network disconnect"));
    const ok2 = await state.syncWithServer();
    expect(ok2).toBe(false);
    fetchSpy.mockRestore();
  });

  it("tracks deleted chat tombstones and prevents server session resurrection on sync", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch");
    state.serverSyncEnabled = true;
    const chat = state.createChat("To Be Deleted");
    const chatId = chat.id;

    // Mock successful DELETE
    fetchSpy.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ status: "deleted" }),
    } as any);

    state.deleteChat(chatId);
    expect(state.deletedChatIds.has(chatId)).toBe(true);
    expect(fetchSpy).toHaveBeenCalledWith(
      `/api/sessions/${encodeURIComponent(chatId)}`,
      expect.objectContaining({ method: "DELETE" }),
    );

    // Now simulate syncWithServer where server returns the deleted session
    fetchSpy.mockResolvedValueOnce({
      ok: true,
      json: async () => [
        {
          session_id: chatId,
          title: "To Be Deleted",
          created_at: "2024-01-01",
          updated_at: "2024-01-01",
          messages: [],
        },
        {
          session_id: "other-session",
          title: "Other Session",
          created_at: "2024-01-01",
          updated_at: "2024-01-01",
          messages: [],
        },
      ],
    } as any);

    // Mock the propagation DELETE call for the tombstone
    fetchSpy.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ status: "deleted" }),
    } as any);

    await state.syncWithServer();

    // Verify deleted chat is NOT resurrected
    expect(state.chats.some((c) => c.id === chatId)).toBe(false);
    // Other session should be added
    expect(state.chats.some((c) => c.id === "other-session")).toBe(true);

    fetchSpy.mockRestore();
  });

  it("merges remote session updates when remote has more messages or title changed", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch");
    const local = state.createChat("Old Title");
    local.isTemporary = false;

    fetchSpy.mockResolvedValueOnce({
      ok: true,
      json: async () => [
        {
          session_id: local.id,
          title: "New Remote Title",
          created_at: "2024-01-01",
          updated_at: "2024-01-01",
          messages: [
            { role: "user", content: "Hello" },
            { role: "assistant", content: "Hi" },
          ],
        },
      ],
    } as any);

    const ok = await state.syncWithServer();
    expect(ok).toBe(true);
    expect(local.title).toBe("New Remote Title");
    expect(local.messages.length).toBe(2);

    // Test when titleModified is true but matches remote title exactly (falsy branch of title !== remote?.title)
    local.titleModified = true;
    local.title = "New Remote Title";
    fetchSpy.mockResolvedValueOnce({
      ok: true,
      json: async () => [
        {
          session_id: local.id,
          title: "New Remote Title",
          created_at: "2024-01-01",
          updated_at: "2024-01-01",
          messages: [
            { role: "user", content: "Hello" },
            { role: "assistant", content: "Hi" },
          ],
        },
      ],
    } as any);
    await state.syncWithServer();
    expect(local.title).toBe("New Remote Title");

    fetchSpy.mockRestore();
  });

  it("handles provider and custom API key management", () => {
    // 1. setActiveChatProvider
    const chat = state.createChat("Provider Chat");
    expect(chat.provider).toBeUndefined();
    state.setActiveChatProvider("openai");
    expect(chat.provider).toBe("openai");

    // When no active chat exists
    state.activeChatId = null;
    state.setActiveChatProvider("anthropic"); // should not throw

    // 2. get, set, remove API keys
    expect(state.getApiKey("openai")).toBeNull();
    state.setApiKey("openai", "sk-test-12345");
    expect(state.getApiKey("openai")).toBe("sk-test-12345");

    state.setApiKey("anthropic", "sk-ant-abc");
    expect(state.getApiKey("anthropic")).toBe("sk-ant-abc");

    const allKeys = state.getAllApiKeys();
    expect(allKeys.openai).toBe("sk-test-12345");
    expect(allKeys.anthropic).toBe("sk-ant-abc");
    expect(allKeys.google).toBeUndefined();

    state.removeApiKey("openai");
    expect(state.getApiKey("openai")).toBeNull();
    expect(state.getAllApiKeys().openai).toBeUndefined();

    // 3. Error branches when localStorage throws
    const origGet = localStorage.getItem;
    localStorage.getItem = () => {
      throw new Error("Quota exceeded");
    };
    expect(state.getApiKey("google")).toBeNull();
    localStorage.getItem = origGet;

    const origSet = localStorage.setItem;
    localStorage.setItem = () => {
      throw new Error("Storage disabled");
    };
    state.setApiKey("google", "test"); // should catch and not throw
    localStorage.setItem = origSet;

    const origRemove = localStorage.removeItem;
    localStorage.removeItem = () => {
      throw new Error("Cannot delete");
    };
    state.removeApiKey("google"); // should catch and not throw
    localStorage.removeItem = origRemove;
  });

  it("handles catch branches for remote session delete and null messages in syncWithServer", async () => {
    state.serverSyncEnabled = true;
    const chat = state.createChat("Delete fail chat");

    // Test deleteChat fetch catch when remote delete rejects
    const fetchRejectSpy = vi
      .spyOn(globalThis, "fetch")
      .mockRejectedValue(new Error("Network delete error"));
    state.deleteChat(chat.id);
    await new Promise((r) => setTimeout(r, 10));
    fetchRejectSpy.mockRestore();

    // Test syncWithServer handling sessions in deletedChatIds where DELETE fetch rejects
    const syncFetchSpy = vi
      .spyOn(globalThis, "fetch")
      .mockImplementation(async (_url, init) => {
        if (init?.method === "DELETE") {
          return Promise.reject(new Error("Remote delete failure"));
        }
        return {
          ok: true,
          json: async () => [
            { session_id: chat.id, title: "Old Deleted", messages: null },
            { session_id: "new-session", title: "New Session", messages: null },
          ],
        } as any;
      });

    // Also add a local chat that has messages to test local message count comparison
    const activeChat = state.createChat("Active Local");
    activeChat.id = "active-local";
    state.addMessageToActiveChat({ role: "user", content: "Hi" });

    await state.syncWithServer();
    await new Promise((r) => setTimeout(r, 10));
    syncFetchSpy.mockRestore();

    const newChat = state.chats.find((c) => c.id === "new-session");
    expect(newChat).toBeDefined();
    expect(newChat?.messages).toEqual([]);

    // Test rawData fallback when server returns an object without sessions array
    const syncEmptyFetch = vi.spyOn(globalThis, "fetch").mockResolvedValue({
      ok: true,
      json: async () => ({ unexpected: true }),
    } as any);
    await state.syncWithServer();
    syncEmptyFetch.mockRestore();

    // Test server session with fewer messages than local chat (line 427 false branch)
    // and server session with undefined/null messages for existing local chat (line 427 || 0 branch)
    const localWithMessages = state.createChat("Multi Message Local");
    state.addMessageToActiveChat({ role: "user", content: "M1" });
    state.addMessageToActiveChat({ role: "assistant", content: "M2" });
    const localWithNullServerMsgs = state.createChat(
      "Null Server Msgs Local",
      true,
    );
    const syncFewerFetch = vi.spyOn(globalThis, "fetch").mockResolvedValue({
      ok: true,
      json: async () => [
        {
          session_id: localWithMessages.id,
          title: "Multi Message Local",
          messages: [{ role: "user", content: "M1" }],
        },
        {
          session_id: localWithNullServerMsgs.id,
          title: "Null Server Msgs Local",
          messages: undefined,
        },
      ],
    } as any);
    await state.syncWithServer();
    syncFewerFetch.mockRestore();
  });

  it("updates systemStatus and handles currentDb initialization when already set", () => {
    state.setSystemStatus({
      status: "healthy",
      backendOnline: true,
    });
    expect(state.systemStatus.status).toBe("healthy");
    expect(state.systemStatus.backendOnline).toBe(true);

    // Test setAvailableDbs branch when this.currentDb is already set
    state.currentDb = "initial.duckdb";
    state.setAvailableDbs(
      [{ name: "test", path: "test.duckdb", size_bytes: 100 }],
      "ignored.duckdb",
    );
    expect(state.currentDb).toBe("initial.duckdb");
  });

  it("resolves clinical table relational joins correctly via ontology and heuristics", () => {
    // 1. Direct ontology matches (both directions)
    const rel1 = resolveTableJoin("patients", "cgms");
    expect(rel1).not.toBeNull();
    expect(rel1?.fromTable).toBe("patients");
    expect(rel1?.toTable).toBe("cgms");
    expect(rel1?.fromColumn).toBe("patient_id");
    expect(rel1?.toColumn).toBe("patient_id");
    expect(rel1?.defaultJoinType).toBe("INNER JOIN");

    const relReverse = resolveTableJoin("cgms", "patients");
    expect(relReverse).not.toBeNull();
    expect(relReverse?.fromTable).toBe("cgms");
    expect(relReverse?.toTable).toBe("patients");
    expect(relReverse?.fromColumn).toBe("patient_id");

    const relVisits = resolveTableJoin("patients", "visits");
    expect(relVisits?.defaultJoinType).toBe("LEFT JOIN");

    // 2. Common clinical keys via known column names
    const relCols = resolveTableJoin(
      "custom_demographics",
      "custom_sensors",
      ["subject_id", "name", "age"],
      ["subject_id", "glucose_val", "recorded_at"],
    );
    expect(relCols).not.toBeNull();
    expect(relCols?.fromColumn).toBe("subject_id");
    expect(relCols?.toColumn).toBe("subject_id");

    // 3. Fallback heuristic matching "patient"
    const relPatient = resolveTableJoin("cohort_patients", "unrelated_records");
    expect(relPatient).not.toBeNull();
    expect(relPatient?.fromColumn).toBe("patient_id");

    // 4. Completely unrelated tables without candidate keys
    const relNull = resolveTableJoin(
      "table_alpha",
      "table_beta",
      ["col_x", "col_y"],
      ["col_a", "col_b"],
    );
    expect(relNull).toBeNull();
  });
});
