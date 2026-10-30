/**
 * @file ui-chat.test.ts
 * @module tests/ui-chat.test
 * Unit tests for ChatUI core chat interactions, SSE streaming, SQL query actions, schema explorer, and session sync.
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
    ui.destroy();
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

    // When Ollama is offline but a cloud key or configured model is present, newChatBtn keeps 'gemma4'
    state.setApiKey("openai", "sk-test");
    document.getElementById("new-chat-btn")?.click();
    expect(state.getActiveChat()?.model).toBe("gemma4");
    state.removeApiKey("openai");
    state.setApiKey("anthropic", "sk-ant");
    document.getElementById("new-chat-btn")?.click();
    expect(state.getActiveChat()?.model).toBe("gemma4");
    state.removeApiKey("anthropic");
    state.setApiKey("google", "sk-goog");
    document.getElementById("new-chat-btn")?.click();
    expect(state.getActiveChat()?.model).toBe("gemma4");
    state.removeApiKey("google");
    state.setConfiguredModels("groq:llama3");
    document.getElementById("new-chat-btn")?.click();
    expect(state.getActiveChat()?.model).toBe("gemma4");
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

    ui["closeAllDropdowns"]();
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
    ui["closeAllDropdowns"]();
    expect(orphanMenu.classList.contains("show")).toBe(false);
    orphanMenu.remove();
  });

  it("handles hashchange, document.title, chips, theme toggle, language switch, and Enter submission", async () => {
    ui["renderMessages"]();
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

    const select = getTestElement<HTMLSelectElement>("model-select");
    select.innerHTML = `<option value="gemma4">Gemma</option><option value="sql">SQL</option>`;
    const chips = queryAllTestElements<HTMLButtonElement>(".chip");
    const nlpChip = Array.from(chips).find((c) =>
      c.textContent?.includes("[NLP]"),
    );
    nlpChip?.click();
    expect(select.value).toBe("gemma4");
    expect(state.getActiveChat()?.model).toBe("gemma4");

    const themeBtn = getTestElement<HTMLButtonElement>("theme-toggle-btn");
    expect(document.body.classList.contains("light-mode")).toBe(false);
    themeBtn.click();
    expect(document.body.classList.contains("light-mode")).toBe(true);
    themeBtn.click();
    expect(document.body.classList.contains("light-mode")).toBe(false);

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

    // Language switch while table modal is open (with and without modal-title)
    const tableModal = getTestElement("table-modal");
    tableModal.removeAttribute("aria-hidden");
    ui["currentTable"] = "patients";
    langSelect.value = "en";
    langSelect.dispatchEvent(new Event("change"));
    await flushPromises();

    const modalTitle = document.getElementById("modal-title");
    modalTitle?.remove();
    langSelect.value = "ja";
    langSelect.dispatchEvent(new Event("change"));
    await flushPromises();
    if (modalTitle) tableModal.prepend(modalTitle);
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
    ui["renderMessages"]();
    expect(document.querySelector(".cgm-analytics-card")).not.toBeNull();
  });

  it("handles schema explorer loading, toggling, and model fallbacks", async () => {
    mockFetch.mockImplementationOnce(async () => {
      throw new Error("Network Error");
    });
    await ui["loadSchema"]();
    expect(document.getElementById("schema-content")?.innerHTML).toContain(
      "error-text",
    );

    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve({ tables: [] }),
    });
    await ui["loadSchema"]();
    expect(document.getElementById("schema-content")?.innerHTML).toContain(
      "No tables found",
    );

    mockFetch.mockResolvedValueOnce({ ok: false, status: 500 });
    await ui["loadSchema"]();
    expect(document.getElementById("schema-content")?.innerHTML).toContain(
      "error-text",
    );

    mockFetch.mockRejectedValueOnce(new Error("Network Error"));
    await ui["loadModels"]();

    const chat = state.createChat();
    chat.model = "missing_model";
    state.setActiveChat(chat.id);
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve({ models: [{ name: "gemma4" }] }),
    });
    await ui["loadModels"]();
    expect(ui["modelSelect"].value).toBe("gemma4");

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
    await ui["loadModels"]();
    expect(ui["modelSelect"].value).toBe("sql");
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
    await ui["loadModels"]();
    expect(ui["modelSelect"].value).toBe("openai/gpt-4o");

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
    await ui["loadModels"]();
    expect(ui["modelSelect"].value).toBe("openai/gpt-4o");
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
    await ui["loadModels"]();
    expect(ui["modelSelect"].value).toBe("sql");

    // Empty modelsList when chat.model is not in options -> falls back to "sql"
    chat.model = "missing_model";
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve({ models: [] }),
    });
    await ui["loadModels"]();
    expect(ui["modelSelect"].value).toBe("sql");
    expect(chat.model).toBe("sql");

    // renderActiveChat and NLP chip click when "gemma4" option is absent from #model-select
    chat.model = "gemma4";
    chat.messages = [];
    ui.render();
    expect(ui["modelSelect"].value).toBe("sql");
    const nlpChipOffline =
      document.querySelectorAll<HTMLButtonElement>(".empty-state .chip")[4];
    nlpChipOffline?.click();
    expect(ui["modelSelect"].value).toBe("sql");
    expect(chat.model).toBe("sql");

    chat.model = "gemma4";
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () =>
        Promise.resolve({ models: [{ name: "gemma4" }, { name: "other" }] }),
    });
    await ui["loadModels"]();
    expect(ui["modelSelect"].value).toBe("gemma4");

    // Table header expand/collapse and table view button
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () =>
        Promise.resolve({
          tables: [
            { name: "users", columns: [{ name: "id", type: "INTEGER" }] },
          ],
        }),
    });
    await ui["loadSchema"]();

    const toggleHeaderBtn = document.querySelector(
      ".schema-table-header-toggle",
    );
    const tableDiv = document.querySelector(".schema-table");
    toggleHeaderBtn?.dispatchEvent(new MouseEvent("click"));
    expect(tableDiv?.classList.contains("expanded")).toBe(true);
    expect(toggleHeaderBtn?.getAttribute("aria-expanded")).toBe("true");

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
    ).toBeNull();
    document.getElementById("close-modal-btn")?.click();

    // Schema explorer collapse/expand
    const toggleBtn = getTestElement("toggle-schema-btn");
    const header = document.querySelector(".schema-header");
    const explorer = document.getElementById("schema-explorer");
    header?.dispatchEvent(new MouseEvent("click"));
    expect(explorer?.classList.contains("collapsed")).toBe(true);
    header?.dispatchEvent(new MouseEvent("click"));
    expect(explorer?.classList.contains("collapsed")).toBe(false);

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
    ui2.destroy();

    document.getElementById("schema-explorer")?.remove();
    const sidebar = document.getElementById("sidebar");
    if (sidebar) sidebar.id = "sidebar-renamed";
    const ui3 = new ChatUI(state);
    await ui3["loadSchema"]();
    ui3.destroy();
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
    ui["renderMessages"]();

    vi.useFakeTimers();
    const writeTextMock = vi.fn().mockResolvedValue(undefined);
    Object.assign(navigator, {
      clipboard: { writeText: writeTextMock },
    });

    const copyBtn = queryTestElement<HTMLButtonElement>(".copy-sql-btn");
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
    ui["renderMessages"]();

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
    ui["renderMessages"]();
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve({ error: "backend.syntaxError" }),
    });
    queryTestElement<HTMLButtonElement>(".play-sql-btn").click();
    await flushPromises();
    expect(state.getActiveChat()?.messages[1]?.isError).toBe(true);

    // Play SQL no rows
    chat.messages = [{ role: "user", content: "SELECT 1" }];
    ui["renderMessages"]();
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
    ui["renderMessages"]();
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
    ui["renderMessages"]();
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve({ sqlResult: [{ val: null }] }),
    });
    queryTestElement<HTMLButtonElement>(".play-sql-btn").click();
    await flushPromises();
    expect(state.getActiveChat()?.messages[1]?.content).toBe("NULL");

    // Play SQL network Error & string rejection
    chat.messages = [{ role: "user", content: "SELECT 1" }];
    ui["renderMessages"]();
    mockFetch.mockRejectedValueOnce(new Error("Network Error"));
    queryTestElement<HTMLButtonElement>(".play-sql-btn").click();
    await flushPromises();
    expect(state.getActiveChat()?.messages[1]?.isError).toBe(true);

    chat.messages = [{ role: "user", content: "SELECT 1" }];
    ui["renderMessages"]();
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
    ui["renderMessages"]();
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
    await ui["handleSendMessage"]("override text");
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
    await ui["handleSendMessage"]("how many patients?");
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
    await ui["handleSendMessage"]("SELECT 99");
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
    await ui["handleSendMessage"]("SELECT NULL");
    expect(chat.messages[chat.messages.length - 1]?.content).toBe("NULL");
  });

  it("handles SSE streaming toggle, stream consumption, and fallback to batch /api/chat", async () => {
    const streamBtn = getTestElement<HTMLButtonElement>("streaming-toggle-btn");
    expect(ui["streamingEnabled"]).toBe(false);

    streamBtn.click();
    expect(ui["streamingEnabled"]).toBe(true);
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
    await ui["handleSendMessage"]("stream 500 fallback");
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
    await ui["handleSendMessage"]("stream err");
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
    await ui["handleSendMessage"]("stream scalar");
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
    await ui["handleSendMessage"]("stream null scalar");
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
    await ui["handleSendMessage"]("trailing buffer");
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
    await ui["handleSendMessage"]("stream token fallback");
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
    await ui["handleSendMessage"]("stream empty done");
    expect(chat.messages[chat.messages.length - 1]?.content).toBe("");

    // Direct consumeSseChatStream unit coverage for non-object JSON and empty token
    const nonObjChunks = [
      `data: 42

`,
      `data: {"event":"token","token":""}

`,
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

    streamBtn.click();
    expect(ui["streamingEnabled"]).toBe(false);
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
    await ui.fetchDatabases();

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
    await ui.fetchDatabases();
    mockFetch.mockRejectedValueOnce(new Error("DB list failed"));
    await ui.fetchDatabases();

    state.setCurrentDb("");
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({}),
    });
    await ui.fetchDatabases();
    expect(dbSelect.value).toBe("t1d.duckdb");

    state.setCurrentDb("custom_saved.duckdb");
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        databases: [],
      }),
    });
    await ui.fetchDatabases();
    expect(dbSelect.value).toBe("custom_saved.duckdb");

    const origDbSelect = ui["dbSelect"];
    ui["dbSelect"] = null;
    await ui.fetchDatabases();
    ui.renderDatabaseDropdown();
    ui["dbSelect"] = origDbSelect;

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
    await ui["handleSendMessage"]("Hello without active chat");
    expect(state.chats.length).toBe(0);
    state.createChat();

    state.setServerSyncEnabled(true);
    mockFetch.mockResolvedValue({
      ok: true,
      json: async () => ({ content: "Synced reply", sessions: [] }),
    });
    await ui["handleSendMessage"]("Test with sync");
    state.setServerSyncEnabled(false);

    // Custom API key header injection for OpenAI, Anthropic, Google, and custom database payload
    state.setApiKey("openai", "sk-openai-custom");
    state.setApiKey("anthropic", "sk-ant-custom");
    state.setApiKey("google", "sk-goog-custom");

    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ content: "OpenAI reply" }),
    });
    await ui["handleSendMessage"]("Hello", "openai/gpt-4o");

    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ content: "Anthropic reply" }),
    });
    await ui["handleSendMessage"]("Hello", "anthropic/claude-3-5-sonnet");

    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ content: "Google reply" }),
    });
    await ui["handleSendMessage"]("Hello", "google/gemini-2.5-pro");

    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ content: null }),
    });
    await ui["handleSendMessage"]("Hello", "gemma4");

    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ content: "Hello" }),
    });
    await ui["handleSendMessage"]("Hi", "plainmodel");

    state.setCurrentDb("custom_clinical.duckdb");
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ content: "Hello custom db" }),
    });
    await ui["handleSendMessage"]("Hi", "sql");

    const triggerBtn = document.createElement("button");
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ sqlResult: [{ a: 1 }] }),
    });
    await ui["executeDirectSqlAction"]("SELECT 1", triggerBtn);
  });

  it("announces messages and sanitizes untrusted markdown HTML against XSS", async () => {
    vi.useFakeTimers();

    // Null announcer guard
    const announcer = document.getElementById("a11y-announcer");
    announcer?.remove();
    ui["announce"]("Should not throw when announcer is missing");
    if (announcer) document.body.appendChild(announcer);

    // Assertive announcer branch
    ui["announce"]("Critical error occurred", true);
    await vi.advanceTimersByTimeAsync(60);
    expect(
      document.getElementById("a11y-announcer")?.getAttribute("aria-live"),
    ).toBe("assertive");
    expect(document.getElementById("a11y-announcer")?.textContent).toBe(
      "Critical error occurred",
    );
    vi.useRealTimers();

    // XSS sanitization in assistant messages
    state.createChat();
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
  });
});
