/**
 * @file schema-explorer.test.ts
 * @module tests/schema-explorer.test
 * Unit tests for SchemaExplorerController sidebar rendering, model fallback resolution,
 * database selector switching, and session synchronization.
 */

import { describe, it, expect, beforeEach, vi, afterEach } from "vitest";
import i18next from "i18next";
import { ChatState } from "../src/state";
import { ChatUI } from "../src/ui";
import { flushPromises, getTestElement, mountChatUiTestDom } from "./setup";

describe("SchemaExplorerController & Database Switcher", () => {
  let state: ChatState;
  let ui: ChatUI;
  let mockFetch: ReturnType<typeof vi.fn>;

  beforeEach(async () => {
    mountChatUiTestDom();
    mockFetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ models: [], schema: [] }),
    });
    vi.stubGlobal("fetch", mockFetch);

    state = new ChatState();
    await i18next.changeLanguage("en");
    document.documentElement.lang = "en";
    ui = new ChatUI(state);
    await flushPromises();
  });

  afterEach(() => {
    ui.dispose();
    localStorage.clear();
    document.body.innerHTML = "";
    vi.restoreAllMocks();
  });

  it("handles schema explorer loading, toggling, and model fallbacks", async () => {
    mockFetch.mockImplementationOnce(async () => {
      throw new Error("Network Error");
    });
    await ui.schemaExplorerController.loadSchema();
    expect(document.getElementById("schema-content")?.innerHTML).toContain(
      "error-text",
    );

    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve({ tables: [] }),
    });
    await ui.schemaExplorerController.loadSchema();
    expect(document.getElementById("schema-content")?.innerHTML).toContain(
      "No tables found",
    );

    mockFetch.mockResolvedValueOnce({ ok: false, status: 500 });
    await ui.schemaExplorerController.loadSchema();
    expect(document.getElementById("schema-content")?.innerHTML).toContain(
      "error-text",
    );

    mockFetch.mockRejectedValueOnce(new Error("Network Error"));
    await ui.modelSelectorController.loadModels();

    const chat = state.createChat();
    chat.model = "missing_model";
    state.setActiveChat(chat.id);
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve({ models: [{ name: "gemma4" }] }),
    });
    await ui.modelSelectorController.loadModels();
    expect(ui.modelSelect.value).toBe("gemma4");

    // Unconfigured cloud model when chat is empty -> falls back to "sql"
    chat.model = "missing_model";
    chat.messages = [];
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () =>
        Promise.resolve({
          models: [
            {
              name: "openai/gpt-4o",
              configured: false,
              available: false,
              reachable: false,
            },
          ],
        }),
    });
    await ui.modelSelectorController.loadModels();
    expect(ui.modelSelect.value).toBe("sql");
    expect(chat.model).toBe("sql");

    // Unconfigured cloud model when chat has messages -> falls back to modelsList[0].name
    chat.model = "missing_model";
    chat.messages = [{ role: "user", content: "hi" }];
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () =>
        Promise.resolve({
          models: [
            {
              name: "openai/gpt-4o",
              configured: false,
              available: true,
              reachable: true,
            },
          ],
        }),
    });
    await ui.modelSelectorController.loadModels();
    expect(ui.modelSelect.value).toBe("openai/gpt-4o");

    // Unconfigured cloud model on server but localStorage key exists -> treated as ready
    state.setApiKey("openai", "sk-test");
    chat.model = "missing_model";
    chat.messages = [];
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () =>
        Promise.resolve({
          models: [
            {
              name: "openai/gpt-4o",
              configured: false,
              available: true,
              reachable: true,
            },
            {
              name: "local-unconfigured",
              configured: false,
              available: true,
              reachable: true,
            },
          ],
        }),
    });
    await ui.modelSelectorController.loadModels();
    expect(ui.modelSelect.value).toBe("openai/gpt-4o");
    state.removeApiKey("openai");

    // Unconfigured local model without slash when no API key is set
    chat.model = "missing_model";
    chat.messages = [];
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () =>
        Promise.resolve({
          models: [
            {
              name: "local-only",
              configured: false,
              available: true,
              reachable: true,
            },
          ],
        }),
    });
    await ui.modelSelectorController.loadModels();
    expect(ui.modelSelect.value).toBe("sql");

    // Empty modelsList when chat.model is not in options -> falls back to "sql"
    chat.model = "missing_model";
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve({ models: [] }),
    });
    await ui.modelSelectorController.loadModels();
    expect(ui.modelSelect.value).toBe("sql");
    expect(chat.model).toBe("sql");

    // renderActiveChat and NLP chip click when "gemma:2b" option is absent from #model-select
    chat.model = "gemma:2b";
    chat.messages = [];
    ui.render();
    expect(ui.modelSelect.value).toBe("sql");
    const nlpChipOffline =
      document.querySelectorAll<HTMLButtonElement>(".empty-state .chip")[4];
    nlpChipOffline?.click();
    expect(ui.modelSelect.value).toBe("sql");
    expect(chat.model).toBe("sql");

    chat.model = "gemma:2b";
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () =>
        Promise.resolve({ models: [{ name: "gemma:2b" }, { name: "other" }] }),
    });
    await ui.modelSelectorController.loadModels();
    expect(ui.modelSelect.value).toBe("gemma:2b");

    // Table header expand/collapse and table view button
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () =>
        Promise.resolve({
          tables: [
            { name: "users", columns: [{ name: "id", type: "INTEGER" }] },
          ],
          database: { status_code: "healthy" },
        }),
    });
    await ui.schemaExplorerController.loadSchema();

    const toggleHeaderBtn = document.querySelector(
      ".schema-table-header-toggle",
    );
    const tableDiv = document.querySelector(".schema-table");
    toggleHeaderBtn?.dispatchEvent(new MouseEvent("click"));
    expect(tableDiv?.classList.contains("expanded")).toBe(true);
    expect(toggleHeaderBtn?.getAttribute("aria-expanded")).toBe("true");

    // Click and keyboard-activate .schema-column to insert column identifier into #chat-input
    ui.chatInput.value = "SELECT";
    ui.chatInput.setSelectionRange(6, 6);
    const colLi = document.querySelector<HTMLElement>(".schema-column");
    colLi?.click();
    expect(ui.chatInput.value).toBe("SELECT id");

    ui.chatInput.value = "users.";
    ui.chatInput.setSelectionRange(6, 6);
    colLi?.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter" }));
    expect(ui.chatInput.value).toBe("users.id");

    colLi?.dispatchEvent(new KeyboardEvent("keydown", { key: "Tab" }));
    expect(ui.chatInput.value).toBe("users.id");

    toggleHeaderBtn?.dispatchEvent(new MouseEvent("click"));
    expect(tableDiv?.classList.contains("expanded")).toBe(false);
    expect(toggleHeaderBtn?.getAttribute("aria-expanded")).toBe("false");

    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve({ rows: [{ id: 1 }] }),
    });
    const viewBtn = document.querySelector(".table-view-btn");
    viewBtn?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    await flushPromises();
    expect(
      document.getElementById("table-modal")?.getAttribute("aria-hidden"),
    ).toBe("false");

    const toggleBtn = getTestElement("toggle-schema-btn");
    const header = document.querySelector(".schema-header");
    const explorer = document.getElementById("schema-explorer");
    header?.dispatchEvent(new MouseEvent("click"));
    expect(explorer?.classList.contains("collapsed")).toBe(true);
    header?.dispatchEvent(new MouseEvent("click"));
    expect(explorer?.classList.contains("collapsed")).toBe(false);

    ui.dispose();
    explorer?.classList.remove("collapsed");
    header?.remove();
    explorer?.prepend(toggleBtn);
    const ui2 = new ChatUI(state);
    document
      .getElementById("toggle-schema-btn")
      ?.dispatchEvent(new MouseEvent("click"));
    expect(explorer?.classList.contains("collapsed")).toBe(true);
    document
      .getElementById("toggle-schema-btn")
      ?.dispatchEvent(new MouseEvent("click"));
    expect(explorer?.classList.contains("collapsed")).toBe(false);
    ui2.dispose();
  });

  it("handles database switcher dropdown, session sync, and custom API key headers", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        databases: [
          { name: "t1d.duckdb", path: "t1d.duckdb", size_bytes: 10240 },
          {
            name: "cohort_2025.duckdb",
            path: "cohort_2025.duckdb",
            size_bytes: 20480,
          },
        ],
        current_db: "t1d.duckdb",
      }),
    });
    await ui.schemaExplorerController.fetchDatabases();

    const dbSelect = getTestElement<HTMLSelectElement>("db-select");
    expect(dbSelect.options.length).toBe(2);

    dbSelect.value = "cohort_2025.duckdb";
    dbSelect.dispatchEvent(new Event("change"));
    await flushPromises();
    expect(state.currentDb).toBe("cohort_2025.duckdb");

    dbSelect.value = "";
    dbSelect.dispatchEvent(new Event("change"));
    expect(state.currentDb).toBe("cohort_2025.duckdb");

    mockFetch.mockResolvedValueOnce({ ok: false, status: 500 });
    await ui.schemaExplorerController.fetchDatabases();
    mockFetch.mockRejectedValueOnce(new Error("DB list failed"));
    await ui.schemaExplorerController.fetchDatabases();

    state.setCurrentDb("");
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({}),
    });
    await ui.schemaExplorerController.fetchDatabases();
    expect(dbSelect.value).toBe("t1d.duckdb");

    state.setCurrentDb("custom_saved.duckdb");
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        databases: [],
      }),
    });
    await ui.schemaExplorerController.fetchDatabases();

    // Session sync
    state.chats = [];
    state.createChat();
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ status: "ok" }),
    });
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ sessions: [] }),
    });
    getTestElement<HTMLButtonElement>("sync-sessions-btn").click();
    await flushPromises();
    expect(state.chats.length).toBe(1);

    vi.spyOn(state, "syncWithServer").mockResolvedValueOnce(false);
    await ui.syncAllChatsWithServer();

    vi.spyOn(state, "syncWithServer").mockRejectedValueOnce(
      new Error("Sync err"),
    );
    await ui.syncAllChatsWithServer();

    // Send message when activeChatId is null
    state.chats = [];
    state.activeChatId = null;
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ content: "No active chat reply" }),
    });
    await ui.handleSendMessage("Hello without active chat");
    expect(state.chats.length).toBe(0);
    state.createChat();

    state.setServerSyncEnabled(true);
    mockFetch.mockResolvedValue({
      ok: true,
      json: async () => ({ content: "Synced reply", sessions: [] }),
    });
    await ui.handleSendMessage("Test with sync");
    state.setServerSyncEnabled(false);

    // Custom API key header injection for OpenAI, Anthropic, Google, and custom database payload
    state.setApiKey("openai", "sk-openai-custom");
    state.setApiKey("anthropic", "sk-ant-custom");
    state.setApiKey("google", "sk-goog-custom");

    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ content: "OpenAI reply" }),
    });
    await ui.handleSendMessage("Hello", "openai/gpt-4o");

    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ content: "Anthropic reply" }),
    });
    await ui.handleSendMessage("Hello", "anthropic/claude-3-5-sonnet");

    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ content: "Google reply" }),
    });
    await ui.handleSendMessage("Hello", "google/gemini-2.5-pro");

    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ content: null }),
    });
    await ui.handleSendMessage("Hello", "gemma4");

    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ content: "Hello" }),
    });
    await ui.handleSendMessage("Hi", "plainmodel");

    state.setCurrentDb("custom_clinical.duckdb");
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ content: "Hello custom db" }),
    });
    await ui.handleSendMessage("Hi", "sql");

    sessionStorage.setItem("t1d_backend_api_key", "secret-token");
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ databases: [], current_db: "t1d.duckdb" }),
    });
    await ui.schemaExplorerController.fetchDatabases();
    sessionStorage.removeItem("t1d_backend_api_key");

    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        tables: [],
        database: { status_code: "missing_file" },
      }),
    });
    await ui.schemaExplorerController.loadSchema();

    const triggerBtn = document.createElement("button");
    const msgCountBefore = state.getActiveChat()?.messages.length ?? 0;
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ sqlResult: [{ a: 1 }] }),
    });
    await ui.messageRenderer.executeDirectSqlAction("SELECT 1", triggerBtn);
    expect(triggerBtn.disabled).toBe(false);
    const activeMsgs = state.getActiveChat()?.messages ?? [];
    expect(activeMsgs.length).toBe(msgCountBefore + 1);
    expect(activeMsgs[activeMsgs.length - 1]?.sqlQuery).toBe("SELECT 1");
    expect(activeMsgs[activeMsgs.length - 1]?.content).toBe("1");
  });
});
