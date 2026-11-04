/**
 * @file ui-chat.test.ts
 * @module tests/ui-chat.test
 * Unit tests for ChatUI core chat submission, SSE streaming, cancellation, and keyboard navigation.
 */

import { describe, it, expect, beforeEach, vi, afterEach } from "vitest";
import i18next from "i18next";
import { ChatState, type Message } from "../src/state";
import { consumeSseChatStream } from "../src/sse-stream";
import { ChatUI } from "../src/ui";
import {
  flushPromises,
  getTestElement,
  mountChatUiTestDom,
  queryAllTestElements,
  queryTestElement,
} from "./setup";

describe("ChatUI Core Chat & Streaming Interactions", () => {
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

  it("renders initial empty state and creates/switches chats", () => {
    const container = document.getElementById("messages-container");
    expect(container?.innerHTML).toContain("Select or create a chat to begin.");
    expect(getTestElement<HTMLTextAreaElement>("chat-input").disabled).toBe(
      true,
    );
    expect(
      document.getElementById("chat-input")?.getAttribute("aria-describedby"),
    ).toBe("chat-input-help");

    const input = getTestElement("chat-input");
    vi.spyOn(input, "focus");

    document.getElementById("new-chat-btn")?.click();
    expect(state.chats.length).toBe(1);

    // When Ollama is offline and no cloud keys exist, clicking newChatBtn sets model to 'sql'
    state.setSystemStatus({ ollamaOnline: false });
    document.getElementById("new-chat-btn")?.click();
    expect(state.getActiveChat()?.model).toBe("sql");

    // When Ollama is offline but a cloud key or configured model is present, newChatBtn keeps 'gemma:2b'
    state.setApiKey("openai", "sk-test");
    document.getElementById("new-chat-btn")?.click();
    expect(state.getActiveChat()?.model).toBe("gemma:2b");
    state.removeApiKey("openai");
    state.setApiKey("anthropic", "sk-ant");
    document.getElementById("new-chat-btn")?.click();
    expect(state.getActiveChat()?.model).toBe("gemma:2b");
    state.removeApiKey("anthropic");
    state.setApiKey("google", "sk-goog");
    document.getElementById("new-chat-btn")?.click();
    expect(state.getActiveChat()?.model).toBe("gemma:2b");
    state.removeApiKey("google");
    state.setConfiguredModels("groq:llama3");
    document.getElementById("new-chat-btn")?.click();
    expect(state.getActiveChat()?.model).toBe("gemma:2b");
    state.removeConfiguredModels();
    state.setSystemStatus({ ollamaOnline: true });
    state.chats = [state.chats[0]!];
    state.activeChatId = state.chats[0]!.id;
    ui.render();

    const list = document.getElementById("chat-list");
    expect(list?.children.length).toBe(1);
    expect(list?.innerHTML).toContain("Chat #1");
    expect(input.focus).toHaveBeenCalled();

    const chat2 = state.createChat();
    ui.render();

    const items = queryAllTestElements<HTMLElement>(".chat-item-title");
    items[0]?.click();
    expect(state.activeChatId).toBe(state.chats[0]?.id);
    expect(document.querySelector(".chat-item.active")?.textContent).toContain(
      "Chat #1",
    );

    items[1]?.dispatchEvent(
      new KeyboardEvent("keydown", {
        key: "Enter",
        bubbles: true,
        cancelable: true,
      }),
    );
    expect(state.activeChatId).toBe(chat2.id);

    state.setActiveChat(state.chats[0]?.id ?? "");
    ui.render();
    const newItems = queryAllTestElements<HTMLElement>(".chat-item-title");
    newItems[1]?.dispatchEvent(
      new KeyboardEvent("keydown", {
        key: " ",
        bubbles: true,
        cancelable: true,
      }),
    );
    expect(state.activeChatId).toBe(chat2.id);

    state.setActiveChat(state.chats[0]?.id ?? "");
    ui.render();
    const finalItems = queryAllTestElements<HTMLElement>(".chat-item-title");
    finalItems[1]?.dispatchEvent(
      new KeyboardEvent("keydown", {
        key: "Escape",
        bubbles: true,
        cancelable: true,
      }),
    );
    expect(state.activeChatId).toBe(state.chats[0]?.id);
  });

  it("handles sending a message successfully and announces response", async () => {
    vi.useFakeTimers();
    state.createChat();
    ui.render();

    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        content: "Generated SQL for: Hello world",
        sqlResult: null,
        error: null,
        is_synthetic_data: true,
        is_rule_based_sql: true,
      }),
    });

    const input = getTestElement<HTMLTextAreaElement>("chat-input");
    input.value = "Hello world";
    document
      .getElementById("chat-form")
      ?.dispatchEvent(new Event("submit", { cancelable: true }));

    expect(state.getActiveChat()?.messages.length).toBe(1);
    expect(input.value).toBe("");

    await flushPromises();
    await vi.advanceTimersByTimeAsync(60);

    expect(state.getActiveChat()?.messages.length).toBe(2);
    expect(state.getActiveChat()?.messages[1]?.content).toContain(
      "Generated SQL for: Hello world",
    );
    expect(
      ui.systemStatusController.jaebProvenanceAlert.classList.contains(
        "hidden",
      ),
    ).toBe(false);
    expect(
      ui.systemStatusController.ruleBasedSqlAlert.classList.contains("hidden"),
    ).toBe(false);

    // Second turn with authentic Jaeb data and active LLM hides both alert bars
    ui.modelSelect.value = "gemma4";
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        content: "Generated SQL from LLM on Jaeb DB",
        sqlResult: null,
        error: null,
        is_synthetic_data: false,
        is_rule_based_sql: false,
      }),
    });
    input.value = "Second question";
    document
      .getElementById("chat-form")
      ?.dispatchEvent(new Event("submit", { cancelable: true }));
    await flushPromises();
    await vi.advanceTimersByTimeAsync(60);
    expect(
      ui.systemStatusController.jaebProvenanceAlert.classList.contains(
        "hidden",
      ),
    ).toBe(true);
    expect(
      ui.systemStatusController.ruleBasedSqlAlert.classList.contains("hidden"),
    ).toBe(true);
    expect(document.getElementById("a11y-announcer")?.textContent).toContain(
      "received",
    );
    vi.useRealTimers();
  });

  it("handles fetch HTTP and network errors during chat submission", async () => {
    state.createChat();
    ui.render();

    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 500,
    });

    const input = getTestElement<HTMLTextAreaElement>("chat-input");
    input.value = "fail test";
    document
      .getElementById("chat-form")
      ?.dispatchEvent(new Event("submit", { cancelable: true }));
    await flushPromises();
    expect(state.getActiveChat()?.messages[1]?.content).toContain(
      "Error communicating with backend API: HTTP 500",
    );

    mockFetch.mockRejectedValueOnce(new Error("Network error"));
    input.value = "fail network";
    document
      .getElementById("chat-form")
      ?.dispatchEvent(new Event("submit", { cancelable: true }));
    await flushPromises();
    expect(state.getActiveChat()?.messages[3]?.content).toContain(
      "Error communicating with backend API: Network error",
    );

    mockFetch.mockRejectedValueOnce("String rejection");
    input.value = "fail throw str";
    document
      .getElementById("chat-form")
      ?.dispatchEvent(new Event("submit", { cancelable: true }));
    await flushPromises();
    expect(state.getActiveChat()?.messages[5]?.content).toContain(
      "Error communicating with backend API: String rejection",
    );
  });

  it("handles empty message guard, mobile sidebar, and dropdown menus/actions", () => {
    state.createChat();
    ui.render();
    document
      .getElementById("chat-form")
      ?.dispatchEvent(new Event("submit", { cancelable: true }));
    expect(state.getActiveChat()?.messages.length).toBe(0);

    document.getElementById("open-sidebar-btn")?.click();
    expect(document.body.classList.contains("sidebar-open")).toBe(true);
    document.getElementById("close-sidebar-btn")?.click();
    expect(document.body.classList.contains("sidebar-open")).toBe(false);
    document.getElementById("open-sidebar-btn")?.click();
    document.getElementById("overlay")?.click();
    expect(document.body.classList.contains("sidebar-open")).toBe(false);

    const dropdownBtn = queryTestElement(".dropdown-btn");
    dropdownBtn.click();
    const menu = queryTestElement(".dropdown-menu");
    expect(menu.classList.contains("show")).toBe(true);
    expect(dropdownBtn.getAttribute("aria-expanded")).toBe("true");

    dropdownBtn.click();
    expect(menu.classList.contains("show")).toBe(false);

    dropdownBtn.click();
    const container = queryTestElement(".dropdown-container");
    container.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    expect(menu.classList.contains("show")).toBe(true);

    document.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    expect(menu.classList.contains("show")).toBe(false);

    dropdownBtn.click();
    const textNode = document.createTextNode("test text");
    document.body.appendChild(textNode);
    document.dispatchEvent(new MouseEvent("click"));
    expect(menu.classList.contains("show")).toBe(false);
    textNode.remove();

    // Keyboard navigation on dropdown
    const items = queryAllTestElements<HTMLElement>(".dropdown-item");
    dropdownBtn.dispatchEvent(
      new KeyboardEvent("keydown", {
        key: "Enter",
        bubbles: true,
        cancelable: true,
      }),
    );
    expect(menu.classList.contains("show")).toBe(true);
    expect(document.activeElement).toBe(items[0]);

    menu.dispatchEvent(
      new KeyboardEvent("keydown", {
        key: "ArrowDown",
        bubbles: true,
        cancelable: true,
      }),
    );
    expect(document.activeElement).toBe(items[1]);

    menu.dispatchEvent(
      new KeyboardEvent("keydown", {
        key: "ArrowDown",
        bubbles: true,
        cancelable: true,
      }),
    );
    expect(document.activeElement).toBe(items[2]);

    menu.dispatchEvent(
      new KeyboardEvent("keydown", {
        key: "ArrowDown",
        bubbles: true,
        cancelable: true,
      }),
    );
    expect(document.activeElement).toBe(items[0]);

    menu.dispatchEvent(
      new KeyboardEvent("keydown", {
        key: "ArrowUp",
        bubbles: true,
        cancelable: true,
      }),
    );
    expect(document.activeElement).toBe(items[2]);

    menu.dispatchEvent(
      new KeyboardEvent("keydown", {
        key: "ArrowUp",
        bubbles: true,
        cancelable: true,
      }),
    );
    expect(document.activeElement).toBe(items[1]);

    menu.dispatchEvent(
      new KeyboardEvent("keydown", {
        key: "Escape",
        bubbles: true,
        cancelable: true,
      }),
    );
    expect(menu.classList.contains("show")).toBe(false);
    expect(document.activeElement).toBe(dropdownBtn);

    dropdownBtn.dispatchEvent(
      new KeyboardEvent("keydown", {
        key: "ArrowDown",
        bubbles: true,
        cancelable: true,
      }),
    );
    expect(menu.classList.contains("show")).toBe(true);

    menu.dispatchEvent(
      new KeyboardEvent("keydown", {
        key: "Home",
        bubbles: true,
        cancelable: true,
      }),
    );
    expect(document.activeElement).toBe(items[0]);

    menu.dispatchEvent(
      new KeyboardEvent("keydown", {
        key: "End",
        bubbles: true,
        cancelable: true,
      }),
    );
    expect(document.activeElement).toBe(items[2]);

    ui.sidebarController.closeAllDropdowns();
    dropdownBtn.dispatchEvent(
      new KeyboardEvent("keydown", {
        key: "ArrowUp",
        bubbles: true,
        cancelable: true,
      }),
    );
    expect(menu.classList.contains("show")).toBe(true);
    expect(document.activeElement).toBe(items[2]);

    menu.dispatchEvent(
      new KeyboardEvent("keydown", {
        key: "a",
        bubbles: true,
        cancelable: true,
      }),
    );
    dropdownBtn.dispatchEvent(
      new KeyboardEvent("keydown", {
        key: "a",
        bubbles: true,
        cancelable: true,
      }),
    );

    // Mobile sidebar keydown when open and closed
    const sidebarEl = getTestElement("sidebar");
    document.body.classList.remove("sidebar-open");
    sidebarEl.dispatchEvent(
      new KeyboardEvent("keydown", { key: "Escape", bubbles: true }),
    );
    document.body.classList.add("sidebar-open");
    sidebarEl.dispatchEvent(
      new KeyboardEvent("keydown", {
        key: "Escape",
        bubbles: true,
        cancelable: true,
      }),
    );
    expect(document.body.classList.contains("sidebar-open")).toBe(false);

    // Rename, duplicate, and delete
    vi.spyOn(window, "prompt").mockReturnValueOnce("Renamed Chat");
    queryTestElement(".dropdown-item").click();
    expect(state.chats[0]?.title).toBe("Renamed Chat");

    vi.spyOn(window, "prompt").mockReturnValueOnce(null);
    queryTestElement(".dropdown-item").click();
    expect(state.chats[0]?.title).toBe("Renamed Chat");

    queryAllTestElements<HTMLElement>(".dropdown-item")[1]?.click();
    expect(state.chats.length).toBe(2);
    expect(state.chats[1]?.title).toBe("Renamed Chat (Copy)");

    vi.spyOn(window, "confirm").mockReturnValueOnce(false);
    queryAllTestElements<HTMLElement>(".dropdown-item")[2]?.click();
    expect(state.chats.length).toBe(2);

    vi.spyOn(window, "confirm").mockReturnValue(true);
    queryAllTestElements<HTMLElement>(".dropdown-item")[2]?.click();
    queryAllTestElements<HTMLElement>(".dropdown-item")[2]?.click();
    expect(state.chats.length).toBe(1);
    expect(state.chats[0]?.title).toBe("Temporary chat");

    const orphanMenu = document.createElement("div");
    orphanMenu.className = "dropdown-menu show";
    document.body.appendChild(orphanMenu);
    ui.sidebarController.closeAllDropdowns();
    expect(orphanMenu.classList.contains("show")).toBe(false);
    orphanMenu.remove();
  });

  it("handles hashchange, document.title, chips, theme toggle, language switch, and Enter submission", async () => {
    ui.messageRenderer.renderActiveChat();
    expect(document.getElementById("messages-container")?.innerHTML).toContain(
      "Select or create a chat to begin.",
    );

    const chat1 = state.createChat();
    const chat2 = state.createChat();
    state.setActiveChat(chat1.id);
    ui.render();
    expect(document.title).toBe("Chat #1 - t1d-analytics");

    const input = getTestElement<HTMLTextAreaElement>("chat-input");
    const focusSpy = vi.spyOn(input, "focus");

    window.location.hash = `#${chat2.id}`;
    window.dispatchEvent(new Event("hashchange"));
    expect(state.activeChatId).toBe(chat2.id);
    expect(focusSpy).toHaveBeenCalled();

    window.location.hash = "";
    focusSpy.mockClear();
    window.dispatchEvent(new Event("hashchange"));
    expect(focusSpy).not.toHaveBeenCalled();

    window.location.hash = `#${chat2.id}`;
    focusSpy.mockClear();
    expect(focusSpy).not.toHaveBeenCalled();

    state.activeChatId = null;
    ui.render();
    expect(document.title).toBe("t1d-analytics");

    state.setActiveChat(chat1.id);
    ui.render();
    const chip = queryTestElement<HTMLButtonElement>(".chip");
    chip.click();
    await flushPromises();
    expect(state.getActiveChat()?.messages.length).toBe(0);
    expect(input.value).toBeTruthy();

    const table1Chips = queryAllTestElements<HTMLButtonElement>(".table1-chip");
    expect(table1Chips.length).toBe(7);
    for (const t1Chip of table1Chips) {
      t1Chip.click();
      expect(ui.modelSelect.value).toBe("sql");
      expect(state.getActiveChat()?.model).toBe("sql");
      expect(input.value).toContain("SELECT");
    }

    const select = getTestElement<HTMLSelectElement>("model-select");
    select.innerHTML = `<option value="gemma:2b">gemma:2b</option><option value="sql">SQL</option>`;
    const chips = queryAllTestElements<HTMLButtonElement>(".chip");
    const nlpChip = Array.from(chips).find((c) =>
      c.textContent?.includes("[NLP]"),
    );
    nlpChip?.click();
    expect(select.value).toBe("gemma:2b");
    expect(state.getActiveChat()?.model).toBe("gemma:2b");

    const themeBtn = getTestElement<HTMLButtonElement>("theme-toggle-btn");
    expect(document.body.classList.contains("light-mode")).toBe(false);
    themeBtn.click();
    expect(document.body.classList.contains("light-mode")).toBe(true);
    expect(localStorage.getItem("t1d_theme")).toBe("light");

    // Re-instantiating ChatUI hydrates light theme and initial window.location.hash
    window.location.hash = `#${chat2.id}`;
    const lightUi = new ChatUI(state);
    expect(document.body.classList.contains("light-mode")).toBe(true);
    expect(state.activeChatId).toBe(chat2.id);
    lightUi.dispose();

    const freshThemeBtn = getTestElement<HTMLButtonElement>("theme-toggle-btn");
    localStorage.setItem("t1d_theme", "dark");
    state.setActiveChat(chat1.id);
    window.location.hash = "";
    const darkUi = new ChatUI(state);
    expect(document.body.classList.contains("light-mode")).toBe(false);
    freshThemeBtn.click();
    expect(document.body.classList.contains("light-mode")).toBe(true);
    freshThemeBtn.click();
    expect(document.body.classList.contains("light-mode")).toBe(false);
    ui = darkUi;

    let submitted = false;
    const form = getTestElement<HTMLFormElement>("chat-form");
    form.addEventListener("submit", (e) => {
      e.preventDefault();
      submitted = true;
    });
    input.dispatchEvent(
      new KeyboardEvent("keydown", { key: "Enter", shiftKey: false }),
    );
    expect(submitted).toBe(true);

    const shiftEnter = new KeyboardEvent("keydown", {
      key: "Enter",
      shiftKey: true,
      cancelable: true,
    });
    input.dispatchEvent(shiftEnter);
    expect(shiftEnter.defaultPrevented).toBe(false);

    const keyA = new KeyboardEvent("keydown", { key: "A", cancelable: true });
    input.dispatchEvent(keyA);
    expect(keyA.defaultPrevented).toBe(false);

    const langSelect = getTestElement<HTMLSelectElement>("lang-select");
    const renderSpy = vi.spyOn(ui, "render");
    langSelect.value = "ja";
    langSelect.dispatchEvent(new Event("change"));
    await flushPromises();
    expect(renderSpy).toHaveBeenCalled();
    expect(document.documentElement.lang).toBe("ja");

    // Language switch while table modal is open
    const tableModal = getTestElement("table-modal");
    tableModal.setAttribute("aria-hidden", "false");
    ui.tableModalController.currentTable = "patients";
    langSelect.value = "en";
    langSelect.dispatchEvent(new Event("change"));
    await flushPromises();
    tableModal.setAttribute("aria-hidden", "true");

    await i18next.changeLanguage("en");
    document.documentElement.lang = "en";

    // Render assistant message with CGM rows (renderCgmCard truthy) and undefined content
    const cgmRows = Array.from({ length: 12 }, (_, idx) => ({
      timestamp: `2026-01-01T0${idx % 9}:00:00Z`,
      glucose: 90 + idx * 8,
    }));
    const cgmMsg: Message = {
      role: "assistant",
      content: "",
      sqlResult: cgmRows,
    };
    Reflect.deleteProperty(cgmMsg, "content");
    chat1.messages.push(cgmMsg);
    ui.messageRenderer.renderMessages();
    expect(document.querySelector(".cgm-analytics-card")).not.toBeNull();
  });

  it("handles SSE streaming toggle, stream consumption, and fallback to batch /api/chat", async () => {
    const streamBtn = getTestElement<HTMLButtonElement>("streaming-toggle-btn");
    expect(ui.streamingEnabled).toBe(false);

    streamBtn.click();
    expect(ui.streamingEnabled).toBe(true);
    expect(streamBtn.classList.contains("active")).toBe(true);

    const chat = state.createChat();
    ui.render();

    const sseHeaders = new Headers({ "content-type": "text/event-stream" });
    const streamTextChunks = [
      `data: {"event":"token","token":"SELECT * "}

`,
      `data: {"event":"token","token":"FROM patients;"}

`,
      `data: {"event":"done","content":"Generated SQL:","sqlQuery":"SELECT * FROM patients;","sqlResult":[{"id":1,"bg":120}],"error":null}

`,
    ];
    let chunkIdx = 0;

    const mockReader = {
      read: vi.fn().mockImplementation(async () => {
        if (chunkIdx < streamTextChunks.length) {
          const chunk = new TextEncoder().encode(streamTextChunks[chunkIdx++]);
          return { done: false, value: chunk };
        }
        return { done: true, value: undefined };
      }),
    };

    mockFetch.mockResolvedValueOnce({
      ok: true,
      headers: sseHeaders,
      body: { getReader: () => mockReader },
    });

    const input = getTestElement<HTMLTextAreaElement>("chat-input");
    input.value = "stream query";
    getTestElement<HTMLFormElement>("chat-form").dispatchEvent(
      new Event("submit", { cancelable: true }),
    );
    await flushPromises();

    expect(chat.messages.length).toBe(2);
    expect(chat.messages[1]?.sqlQuery).toBe("SELECT * FROM patients;");
    expect(chat.messages[1]?.sqlResult?.length).toBe(1);

    // Fallback when stream returns non-ok HTTP status
    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 500,
    });
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        content: "Fallback from 500 stream",
        sqlQuery: null,
        sqlResult: null,
        error: null,
      }),
    });
    await ui.handleSendMessage("stream 500 fallback");
    expect(chat.messages[chat.messages.length - 1]?.content).toContain(
      "Fallback from 500 stream",
    );

    // Fallback when stream fails
    mockFetch.mockRejectedValueOnce(new Error("Stream not supported"));
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        content: "Fallback response",
        sqlQuery: "SELECT 1;",
        sqlResult: [{ a: 1, b: 2 }],
        error: null,
      }),
    });
    input.value = "fallback query";
    getTestElement<HTMLFormElement>("chat-form").dispatchEvent(
      new Event("submit", { cancelable: true }),
    );
    await flushPromises();
    expect(chat.messages.length).toBe(6);
    expect(chat.messages[5]?.content).toContain("Fallback response");

    // Stream error event, scalar formatting, and malformed JSON chunks
    const streamErrorChunks = [
      `data: {bad json

`,
      `data: {"event":"done","content":"backend.literalSql","sqlQuery":"SELECT 1","sqlResult":null,"error":"backend.syntaxError"}

`,
    ];
    let idx1 = 0;
    mockFetch.mockResolvedValueOnce({
      ok: true,
      headers: sseHeaders,
      body: {
        getReader: () => ({
          read: vi.fn().mockImplementation(async () => {
            if (idx1 < streamErrorChunks.length) {
              return {
                done: false,
                value: new TextEncoder().encode(streamErrorChunks[idx1++]),
              };
            }
            return { done: true, value: undefined };
          }),
        }),
      },
    });
    await ui.handleSendMessage("stream err");
    expect(chat.messages[chat.messages.length - 1]?.isError).toBe(true);

    const streamScalarChunks = [
      `data: {"event":"done","content":"Count is","sqlQuery":"SELECT COUNT(*)","sqlResult":[{"count":42}],"error":null}

`,
    ];
    let idx2 = 0;
    mockFetch.mockResolvedValueOnce({
      ok: true,
      headers: sseHeaders,
      body: {
        getReader: () => ({
          read: vi.fn().mockImplementation(async () => {
            if (idx2 < streamScalarChunks.length) {
              return {
                done: false,
                value: new TextEncoder().encode(streamScalarChunks[idx2++]),
              };
            }
            return { done: true, value: undefined };
          }),
        }),
      },
    });
    await ui.handleSendMessage("stream scalar");
    expect(chat.messages[chat.messages.length - 1]?.content).toContain(
      "**count:** 42",
    );

    const streamNullScalarChunks = [
      `data: {"event":"done","content":"","sqlQuery":"SELECT NULL","sqlResult":[{"v":null}],"error":null}

`,
    ];
    let idx3 = 0;
    mockFetch.mockResolvedValueOnce({
      ok: true,
      headers: sseHeaders,
      body: {
        getReader: () => ({
          read: vi.fn().mockImplementation(async () => {
            if (idx3 < streamNullScalarChunks.length) {
              return {
                done: false,
                value: new TextEncoder().encode(streamNullScalarChunks[idx3++]),
              };
            }
            return { done: true, value: undefined };
          }),
        }),
      },
    });
    await ui.handleSendMessage("stream null scalar");
    expect(chat.messages[chat.messages.length - 1]?.content).toBe("NULL");

    // Multi-event single chunk with double newline
    const trailingChunks = [
      `data: {"event":"token","token":"SELECT 1"}

data: {"event":"done","content":"Trailing done","sqlQuery":"SELECT 1","sqlResult":[],"error":null}

`,
    ];
    let idxT = 0;
    mockFetch.mockResolvedValueOnce({
      ok: true,
      headers: sseHeaders,
      body: {
        getReader: () => ({
          read: vi.fn().mockImplementation(async () => {
            if (idxT < trailingChunks.length) {
              return {
                done: false,
                value: new TextEncoder().encode(trailingChunks[idxT++]),
              };
            }
            return { done: true, value: undefined };
          }),
        }),
      },
    });
    await ui.handleSendMessage("trailing buffer");
    expect(chat.messages[chat.messages.length - 1]?.content).toBe(
      "Trailing done",
    );

    // Stream without content in done event (falls back to streamText or "")
    const streamTokenOnlyDone = [
      `data: {"event":"token","token":"Streamed text"}

`,
      `data: {"event":"done","content":"","sqlQuery":null,"sqlResult":null,"error":null}

`,
    ];
    let idx4 = 0;
    mockFetch.mockResolvedValueOnce({
      ok: true,
      headers: sseHeaders,
      body: {
        getReader: () => ({
          read: vi.fn().mockImplementation(async () => {
            if (idx4 < streamTokenOnlyDone.length) {
              return {
                done: false,
                value: new TextEncoder().encode(streamTokenOnlyDone[idx4++]),
              };
            }
            return { done: true, value: undefined };
          }),
        }),
      },
    });
    await ui.handleSendMessage("stream token fallback");
    expect(chat.messages[chat.messages.length - 1]?.content).toBe(
      "Streamed text",
    );

    const streamEmptyDone = [
      `: comment line

`,
      `data: {"event":"done","content":"","sqlQuery":null,"sqlResult":null,"error":null}

`,
    ];
    let idx5 = 0;
    mockFetch.mockResolvedValueOnce({
      ok: true,
      headers: sseHeaders,
      body: {
        getReader: () => ({
          read: vi.fn().mockImplementation(async () => {
            if (idx5 < streamEmptyDone.length) {
              return {
                done: false,
                value: new TextEncoder().encode(streamEmptyDone[idx5++]),
              };
            }
            return { done: true, value: undefined };
          }),
        }),
      },
    });
    await ui.handleSendMessage("stream empty done");
    expect(chat.messages[chat.messages.length - 1]?.content).toBe("");

    // Stream without done event falls back to batch /api/chat
    const incompleteStreamChunks = [
      `data: {"event":"token","token":"partial"}

`,
    ];
    let incompleteIdx = 0;
    mockFetch.mockResolvedValueOnce({
      ok: true,
      headers: { get: () => "text/event-stream" },
      body: {
        getReader: () => ({
          read: vi.fn().mockImplementation(async () => {
            if (incompleteIdx < incompleteStreamChunks.length) {
              const val = new TextEncoder().encode(
                incompleteStreamChunks[incompleteIdx++],
              );
              return { done: false, value: val };
            }
            return { done: true, value: undefined };
          }),
        }),
      },
    });
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        content: "Batch fallback after incomplete stream",
        error: { error_code: "custom.untranslated_code", params: { k: "v" } },
      }),
    });
    await ui.handleSendMessage("incomplete stream fallback");
    expect(chat.messages[chat.messages.length - 1]?.content).toContain(
      "custom.untranslated_code",
    );

    mockFetch.mockResolvedValueOnce({
      ok: true,
      headers: { get: () => "application/json" },
    });
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        content: "Batch after non-sse header",
        error: 404,
      }),
    });
    await ui.handleSendMessage("non-sse header fallback");
    expect(chat.messages[chat.messages.length - 1]?.content).toContain("404");

    // Direct consumeSseChatStream unit coverage for non-object JSON, CRLF framing, and unterminated trailing frame
    const nonObjChunks = [
      `data: 42\r\n\r\n`,
      `data: {"event":"token","token":""}\r\n\r\n`,
    ];
    let idxNo = 0;
    const fakeStream = new ReadableStream<Uint8Array>({
      pull(controller) {
        if (idxNo < nonObjChunks.length) {
          controller.enqueue(new TextEncoder().encode(nonObjChunks[idxNo++]));
        } else {
          controller.close();
        }
      },
    });
    const resNo = await consumeSseChatStream(fakeStream.getReader(), () => {});
    expect(resNo.streamUsed).toBe(false);
    expect(resNo.streamText).toBe("");

    const crlfAndTrailingChunks = [
      `data: {"event":"token","token":"Hello "}\r\n\r\n`,
      `data: {"event":"done","content":"Final CRLF"}`,
    ];
    let idxCrlf = 0;
    const crlfStream = new ReadableStream<Uint8Array>({
      pull(controller) {
        if (idxCrlf < crlfAndTrailingChunks.length) {
          controller.enqueue(
            new TextEncoder().encode(crlfAndTrailingChunks[idxCrlf++]),
          );
        } else {
          controller.close();
        }
      },
    });
    const resCrlf = await consumeSseChatStream(
      crlfStream.getReader(),
      () => {},
    );
    expect(resCrlf.streamUsed).toBe(true);
    expect(resCrlf.streamText).toBe("Hello ");
    expect(resCrlf.finalData?.content).toBe("Final CRLF");

    streamBtn.click();
    expect(ui.streamingEnabled).toBe(false);
  });
});
