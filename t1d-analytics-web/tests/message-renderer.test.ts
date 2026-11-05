/**
 * @file message-renderer.test.ts
 * @module tests/message-renderer.test
 * Unit tests for MessageRenderer DOM node caching, Table 1 query chips, copy/play/refresh SQL buttons,
 * CSV/JSON export buttons, scalar vs table classification, and XSS sanitization.
 */

import { describe, it, expect, beforeEach, vi, afterEach } from "vitest";
import i18next from "i18next";
import { ChatState, type Message } from "../src/state";
import { ChatUI } from "../src/ui";
import {
  flushPromises,
  getTestElement,
  mountChatUiTestDom,
  queryAllTestElements,
  queryTestElement,
} from "./setup";

describe("MessageRenderer & SQL Action Controls", () => {
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
    i18next.changeLanguage("en");
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

  it("renders SQL results in a table, empty SQL results, and literal SQL errors", async () => {
    state.createChat();
    ui.render();

    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        content: "Success",
        sqlResult: [
          { patient: "Alice", bg_level: 110 },
          { patient: "Bob", bg_level: null },
        ],
        error: null,
      }),
    });

    const input = getTestElement<HTMLTextAreaElement>("chat-input");
    input.value = "Show patients";
    document
      .getElementById("chat-form")
      ?.dispatchEvent(new Event("submit", { cancelable: true }));
    await flushPromises();

    const tables = document.querySelectorAll(".sql-table");
    expect(tables.length).toBe(1);
    expect(tables[0]?.innerHTML).toContain("Alice");
    expect(tables[0]?.innerHTML).toContain("bg_level");
    expect(tables[0]?.innerHTML).toContain("NULL");
    expect(tables[0]?.querySelectorAll("th")[0]?.getAttribute("scope")).toBe(
      "col",
    );
    expect(
      tables[0]?.querySelector("caption")?.classList.contains("sr-only"),
    ).toBe(true);

    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        content: "Success empty",
        sqlResult: [],
        error: null,
      }),
    });
    input.value = "empty results";
    document
      .getElementById("chat-form")
      ?.dispatchEvent(new Event("submit", { cancelable: true }));
    await flushPromises();
    const emptyMsg = document.querySelectorAll(".sql-empty");
    expect(emptyMsg.length).toBe(1);
    expect(emptyMsg[0]?.textContent).toBe("No rows returned");

    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        content: "Executed literal SQL",
        sqlResult: null,
        error: "Syntax error",
      }),
    });
    const select = getTestElement<HTMLSelectElement>("model-select");
    select.value = "sql";
    select.dispatchEvent(new Event("change"));
    input.value = "SELECT * FROM patients";
    document
      .getElementById("chat-form")
      ?.dispatchEvent(new Event("submit", { cancelable: true }));
    await flushPromises();
    const msgs = state.getActiveChat()?.messages ?? [];
    const last = msgs[msgs.length - 1];
    expect(last?.content).toContain("Executed literal SQL");
    expect(last?.content).toContain("Error details: Syntax error");
  });

  it("handles copy-sql-btn, play-sql-btn, and refresh button interactions", async () => {
    const messages: Message[] = [
      {
        role: "assistant",
        content: "response",
        sqlQuery: "SELECT 1;",
      },
    ];
    state.chats = [
      {
        id: "chat-1",
        title: "Chat 1",
        model: "gemma4",
        messages,
      },
    ];
    state.activeChatId = "chat-1";
    ui.messageRenderer.renderMessages();

    vi.useFakeTimers();
    const writeTextMock = vi.fn().mockResolvedValue(undefined);
    Object.assign(navigator, {
      clipboard: { writeText: writeTextMock },
    });

    const copyBtn = queryTestElement<HTMLButtonElement>(".copy-sql-btn");
    copyBtn.click();
    await flushPromises();
    copyBtn.click();
    await flushPromises();
    expect(writeTextMock).toHaveBeenCalledWith("SELECT 1;");
    expect(copyBtn.innerHTML).toContain("polyline points=");
    vi.runAllTimers();
    expect(copyBtn.innerHTML).not.toContain("polyline points=");
    vi.useRealTimers();

    // Copy error branch
    const writeErrMock = vi.fn().mockRejectedValue(new Error("Copy failed"));
    Object.assign(navigator, {
      clipboard: { writeText: writeErrMock },
    });
    const consoleSpy = vi.spyOn(console, "error").mockImplementation(() => {});
    copyBtn.click();
    await flushPromises();
    expect(consoleSpy).toHaveBeenCalledWith(
      "Failed to copy text: ",
      expect.any(Error),
    );
    consoleSpy.mockRestore();

    // Play SQL button on assistant message and SQL user message
    const chat = state.createChat();
    chat.model = "sql";
    chat.messages.push({ role: "user", content: "SELECT 1" });
    state.setActiveChat(chat.id);
    ui.messageRenderer.renderMessages();

    const wrapper = document.querySelector(".sql-user-query-wrapper");
    expect(wrapper).not.toBeNull();
    expect(wrapper?.querySelector(".copy-sql-btn")).not.toBeNull();
    const playBtn = queryTestElement<HTMLButtonElement>(".play-sql-btn");

    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve({ sqlResult: [{ val: 1 }] }),
    });
    playBtn.click();
    await flushPromises();
    expect(state.getActiveChat()?.messages.length).toBe(2);

    // Play SQL error from backend
    chat.messages = [{ role: "user", content: "SELECT 1" }];
    ui.messageRenderer.renderMessages();
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve({ error: "backend.syntaxError" }),
    });
    queryTestElement<HTMLButtonElement>(".play-sql-btn").click();
    await flushPromises();
    expect(state.getActiveChat()?.messages[1]?.isError).toBe(true);

    // Play SQL no rows
    chat.messages = [{ role: "user", content: "SELECT 1" }];
    ui.messageRenderer.renderMessages();
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve({ sqlResult: [] }),
    });
    queryTestElement<HTMLButtonElement>(".play-sql-btn").click();
    await flushPromises();
    expect(state.getActiveChat()?.messages[1]?.content.length).toBeGreaterThan(
      0,
    );

    // Play SQL multiple columns
    chat.messages = [{ role: "user", content: "SELECT 1" }];
    ui.messageRenderer.renderMessages();
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve({ sqlResult: [{ a: 1, b: 2 }] }),
    });
    queryTestElement<HTMLButtonElement>(".play-sql-btn").click();
    await flushPromises();
    expect(state.getActiveChat()?.messages[1]?.content.length).toBeGreaterThan(
      0,
    );

    // Play SQL null scalar
    chat.messages = [{ role: "user", content: "SELECT NULL" }];
    ui.messageRenderer.renderMessages();
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve({ sqlResult: [{ val: null }] }),
    });
    queryTestElement<HTMLButtonElement>(".play-sql-btn").click();
    await flushPromises();
    expect(state.getActiveChat()?.messages[1]?.content).toBe("NULL");

    // Play SQL network Error & string rejection
    chat.messages = [{ role: "user", content: "SELECT 1" }];
    ui.messageRenderer.renderMessages();
    mockFetch.mockRejectedValueOnce(new Error("Network Error"));
    queryTestElement<HTMLButtonElement>(".play-sql-btn").click();
    await flushPromises();
    expect(state.getActiveChat()?.messages[1]?.isError).toBe(true);

    chat.messages = [{ role: "user", content: "SELECT 1" }];
    ui.messageRenderer.renderMessages();
    mockFetch.mockRejectedValueOnce("Network Error String");
    queryTestElement<HTMLButtonElement>(".play-sql-btn").click();
    await flushPromises();
    expect(state.getActiveChat()?.messages[1]?.content).toContain(
      "Network Error String",
    );

    // Refresh button interactions
    chat.messages = [
      {
        role: "assistant",
        content: "Table Data",
        sqlResult: [{ id: 1 }],
        sqlQuery: "SELECT 1",
        model: "sql",
      },
    ];
    ui.messageRenderer.renderMessages();
    const getRefreshBtn = () =>
      queryTestElement<HTMLButtonElement>("button[aria-label='Refresh Query']");

    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve({ sqlResult: [{ id: 1, val: 2 }] }),
    });
    getRefreshBtn().click();
    await flushPromises();
    expect(state.getActiveChat()?.messages[0]?.sqlResult?.[0]?.val).toBe(2);

    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve({ error: "backend.syntaxError" }),
    });
    getRefreshBtn().click();
    await flushPromises();
    expect(state.getActiveChat()?.messages[0]?.isError).toBe(true);

    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve({ sqlResult: [] }),
    });
    getRefreshBtn().click();
    await flushPromises();
    expect(state.getActiveChat()?.messages[0]?.content.length).toBeGreaterThan(
      0,
    );

    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve({ sqlResult: [{ a: 1, b: 2 }] }),
    });
    getRefreshBtn().click();
    await flushPromises();
    expect(state.getActiveChat()?.messages[0]?.content.length).toBeGreaterThan(
      0,
    );

    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve({ sqlResult: [{ val: 123 }] }),
    });
    getRefreshBtn().click();
    await flushPromises();
    expect(state.getActiveChat()?.messages[0]?.content).toBe("123");

    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve({ sqlResult: [{ val: null }] }),
    });
    getRefreshBtn().click();
    await flushPromises();
    expect(state.getActiveChat()?.messages[0]?.content).toBe("NULL");

    mockFetch.mockRejectedValueOnce(new Error("Network Error"));
    getRefreshBtn().click();
    await flushPromises();
    expect(state.getActiveChat()?.messages[0]?.isError).toBe(true);

    mockFetch.mockRejectedValueOnce("Network Error String");
    getRefreshBtn().click();
    await flushPromises();
    expect(state.getActiveChat()?.messages[0]?.content).toContain(
      "Network Error String",
    );
  });

  it("handles scalar value formatting, translated content, and sendMessage overrides", async () => {
    const chat = state.createChat();
    state.setActiveChat(chat.id);
    ui.render();

    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () =>
        Promise.resolve({
          content: "backend.literalSql",
          error: "",
        }),
    });
    const input = getTestElement<HTMLTextAreaElement>("chat-input");
    input.value = "test translated content";
    getTestElement<HTMLFormElement>("chat-form").dispatchEvent(
      new Event("submit"),
    );
    await flushPromises();
    expect(chat.messages[chat.messages.length - 1]?.content).toBe(
      "Executed literal SQL:",
    );

    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () =>
        Promise.resolve({
          content: "untranslated.content",
          error: "",
        }),
    });
    input.value = "test empty error";
    getTestElement<HTMLFormElement>("chat-form").dispatchEvent(
      new Event("submit"),
    );
    await flushPromises();
    expect(chat.messages[chat.messages.length - 1]?.content).toBe(
      "untranslated.content",
    );

    input.value = "some typed text";
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve({ response: "hello", sql_query: "" }),
    });
    await ui.handleSendMessage("override text");
    expect(input.value).toBe("some typed text");

    // Scalar result with non-empty content, empty content, and null value
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () =>
        Promise.resolve({
          content: "Generated SQL:",
          sqlQuery: "SELECT COUNT(*) AS count FROM patients",
          sqlResult: [{ count: 42 }],
        }),
    });
    await ui.handleSendMessage("how many patients?");
    expect(chat.messages[chat.messages.length - 1]?.content).toBe(
      "Generated SQL: **count:** 42",
    );

    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () =>
        Promise.resolve({
          content: "",
          sqlQuery: "SELECT 99 AS val",
          sqlResult: [{ val: 99 }],
        }),
    });
    await ui.handleSendMessage("SELECT 99");
    expect(chat.messages[chat.messages.length - 1]?.content).toBe("99");

    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () =>
        Promise.resolve({
          content: "",
          sqlQuery: "SELECT NULL AS val",
          sqlResult: [{ val: null }],
        }),
    });
    await ui.handleSendMessage("SELECT NULL");
    expect(chat.messages[chat.messages.length - 1]?.content).toBe("NULL");
  });

  it("announces messages, renders Table 1 chips, and sanitizes untrusted markdown HTML against XSS", async () => {
    vi.useFakeTimers();

    const announcer = document.getElementById("a11y-announcer");
    announcer?.remove();
    ui.announce("Should not throw when announcer is missing");
    if (announcer) document.body.appendChild(announcer);

    ui.announce("Critical error occurred", true);
    await vi.advanceTimersByTimeAsync(60);
    expect(
      document.getElementById("a11y-announcer")?.getAttribute("aria-live"),
    ).toBe("assertive");
    expect(document.getElementById("a11y-announcer")?.textContent).toBe(
      "Critical error occurred",
    );
    vi.useRealTimers();

    const chat = state.createChat();
    ui.render();
    const table1Chips = queryAllTestElements<HTMLButtonElement>(".table1-chip");
    expect(table1Chips.length).toBe(7);

    state.addMessageToActiveChat({
      role: "assistant",
      content:
        'Hello <script>window.__xssExecuted = true;</script><img src="x" onerror="alert(1)" /> **bold**',
    });
    ui.render();

    const messagesContainer = getTestElement("messages-container");
    expect(messagesContainer.querySelector("script")).toBeNull();
    const img = messagesContainer.querySelector("img");
    expect(img?.getAttribute("onerror")).toBeNull();
    expect(messagesContainer.innerHTML).toContain("<strong>bold</strong>");
    expect(chat.messages.length).toBe(1);
  });

  it("renders Gold-Standard SQL Evaluation Card (PASS, DIFF, syntax error, and fallback badges) and user message Edit/Retry button", () => {
    state.createChat();
    state.addMessageToActiveChat({
      role: "user",
      content: "Calculate mean age by treatment group",
      model: "gemma:2b",
      goldSql: "SELECT TxGroup, AVG(Age) FROM tblaptsummary GROUP BY TxGroup;",
    });
    state.addMessageToActiveChat({
      role: "assistant",
      content:
        "```sql\nSELECT TxGroup, AVG(Age) FROM tblaptsummary GROUP BY TxGroup;\n```",
      sqlQuery: "SELECT TxGroup, AVG(Age) FROM tblaptsummary GROUP BY TxGroup;",
      sqlResult: [{ TxGroup: "Experimental", avg_age: 14.2 }],
      model: "gemma:2b",
      isRuleBasedSql: false,
      generationMode: "llm",
      goldComparison: {
        gold_sql:
          "SELECT TxGroup, AVG(Age) FROM tblaptsummary GROUP BY TxGroup;",
        pred_sql:
          "SELECT TxGroup, AVG(Age) FROM tblaptsummary GROUP BY TxGroup;",
        exact_match: true,
        execution_accuracy: true,
        syntax_error: false,
        gold_row_count: 1,
        pred_row_count: 1,
        gold_columns: ["TxGroup", "avg_age"],
        pred_columns: ["TxGroup", "avg_age"],
        column_overlap_pct: 100.0,
        gold_rows_preview: [{ TxGroup: "Experimental", avg_age: 14.2 }],
      },
    });
    state.addMessageToActiveChat({
      role: "assistant",
      content: "Fallback attempt failed",
      model: "gemma:2b",
      isRuleBasedSql: true,
      generationMode: "rule_based_fallback",
      goldComparison: {
        gold_sql: "SELECT 1;",
        pred_sql: "",
        exact_match: false,
        execution_accuracy: false,
        syntax_error: true,
        gold_row_count: 0,
        pred_row_count: 0,
        gold_columns: [],
        pred_columns: [],
        column_overlap_pct: 0.0,
        gold_rows_preview: [],
        error_message: "Binder Error: column not found",
      },
    });

    ui.render();

    const cards = queryAllTestElements<HTMLElement>(
      ".gold-sql-comparison-card",
    );
    expect(cards.length).toBe(2);
    expect(cards[0]?.textContent).toContain("PASS");
    expect(cards[0]?.textContent).toContain("100.0%");
    expect(cards[1]?.textContent).toContain("DIFF");
    expect(cards[1]?.textContent).toContain("Binder Error: column not found");

    // Click the Edit/Retry button on the user message
    const retryBtn = queryTestElement<HTMLButtonElement>(".edit-retry-msg-btn");
    retryBtn.click();
    expect(ui.chatInput.value).toBe("Calculate mean age by treatment group");
    expect(state.activeGoldSql).toBe(
      "SELECT TxGroup, AVG(Age) FROM tblaptsummary GROUP BY TxGroup;",
    );

    // Also test user message without goldSql
    state.addMessageToActiveChat({
      role: "user",
      content: "Plain prompt without gold SQL",
    });
    ui.render();
    const retryBtns = queryAllTestElements<HTMLButtonElement>(
      ".edit-retry-msg-btn",
    );
    retryBtns[retryBtns.length - 1]?.click();
    expect(ui.chatInput.value).toBe("Plain prompt without gold SQL");
    expect(state.activeGoldSql).toBeNull();

    // Test fallback SQL synthesis diagnostic banner (both explicit llmError and parenthetical note) and >40 char cell tooltip
    const longText = "A".repeat(45);
    ui.messageRenderer.finalizeAssistantResponse(
      {
        content:
          "_Note: Local LLM inference was unavailable (Connection refused by local Ollama); synthesized schema-guided fallback SQL for `demographics`._\n\n```sql\nSELECT * FROM demographics LIMIT 10;\n```",
        sqlQuery: "SELECT * FROM demographics LIMIT 10;",
        sqlResult: [{ RecID: 1, ptid: 101, Notes: longText }],
        is_rule_based_sql: true,
        generation_mode: "rule_based_fallback",
        llm_error: "Connection refused by local Ollama",
        llm_error_code: "backend.llmConnectionError",
      },
      "gemma:2b",
    );
    state.addMessageToActiveChat({
      role: "assistant",
      content:
        "_Note: Local LLM inference was unavailable (Model not pulled); synthesized schema-guided fallback SQL for `patients`._",
    });
    state.addMessageToActiveChat({
      role: "assistant",
      content: "",
      isRuleBasedSql: true,
    });
    ui.render();

    const fallbackBanners = queryAllTestElements<HTMLElement>(
      ".fallback-synthesis-banner",
    );
    expect(fallbackBanners.length).toBe(2);
    expect(fallbackBanners[0]?.textContent).toContain(
      "Connection refused by local Ollama",
    );
    expect(fallbackBanners[0]?.textContent).toContain(
      "backend.llmConnectionError",
    );
    expect(fallbackBanners[1]?.textContent).toContain("Model not pulled");

    const configBtn = fallbackBanners[0]?.querySelector<HTMLButtonElement>(
      ".fallback-configure-btn",
    );
    expect(configBtn).not.toBeNull();
    configBtn?.click();

    const metaBadges = queryAllTestElements<HTMLElement>(
      ".sql-table-meta-badge",
    );
    expect(metaBadges[metaBadges.length - 1]?.textContent).toContain(
      "1 rows × 3 cols",
    );

    const cells = Array.from(
      queryAllTestElements<HTMLTableCellElement>(".sql-table td"),
    );
    const longCell = cells.find((c) => c.textContent === longText);
    expect(longCell?.title).toBe(longText);
  });
});
