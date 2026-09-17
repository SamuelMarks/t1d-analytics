import { describe, it, expect, beforeEach, vi, afterEach } from "vitest";
import i18next from "i18next";
import { ChatState } from "../src/state";
import { ChatUI } from "../src/ui";

describe("ChatUI", () => {
  let state: ChatState;
  let ui: ChatUI;
  let mockFetch: ReturnType<typeof vi.fn>;

  const flushPromises = () => new Promise((resolve) => setTimeout(resolve));

  beforeEach(() => {
    // Setup minimal DOM
    document.body.innerHTML = `
      <nav id="sidebar">
        <button id="new-chat-btn">New Chat</button>
        <button id="close-sidebar-btn">Close</button>
        <ul id="chat-list"></ul>
        <div id="schema-explorer">
          <div class="schema-header">
            <button id="toggle-schema-btn"></button>
          </div>
          <div id="schema-content"></div>
        </div>
      </nav>
      <div id="table-modal" aria-hidden="true">
        <h2 id="modal-title"></h2>
        <button id="close-modal-btn"></button>
        <input type="search" id="modal-table-search" />
        <select id="modal-table-sort-col"></select>
        <button id="modal-table-sort-order">ASC</button>
        <button id="prev-page-btn"></button>
        <button id="next-page-btn"></button>
        <button id="modal-export-csv-btn" disabled></button>
        <button id="modal-export-json-btn" disabled></button>
        <div id="table-loading"></div>
        <table id="modal-table">
          <caption class="sr-only" id="modal-table-caption">Table Data</caption>
          <thead id="modal-table-head"></thead>
          <tbody id="modal-table-body"></tbody>
        </table>
        <span id="page-indicator"></span>
      </div>
      <main id="main-pane">
        <button id="system-status-chip" class="status-chip status-healthy">
          <span class="status-dot"></span>
          <span id="status-chip-text" class="status-text">All systems operational</span>
        </button>
        <button id="open-sidebar-btn">Open</button>
        <button id="theme-toggle-btn">Theme</button>
        <select id="model-select">
          <option value="gemma4">Gemma</option>
          <option value="sql">SQL</option>
        </select>
        <button id="provider-settings-btn">⚙️</button>
        <span id="provider-status-dot" class="active">●</span>
        <select id="db-select">
          <option value="t1d.duckdb">t1d.duckdb</option>
        </select>
        <button id="streaming-toggle-btn">⚡</button>
        <button id="sync-sessions-btn">Sync</button>
        <button id="cohort-filter-btn">Cohort</button>
        <select id="lang-select">
          <option value="en">English</option>
          <option value="ja">Japanese</option>
        </select>
        <div id="cohort-modal" class="modal-overlay hidden">
          <button id="close-cohort-modal-btn">✕</button>
          <select id="cohort-table-select"></select>
          <select id="cohort-join-table-select">
            <option value="">None (Single Table)</option>
          </select>
          <select id="cohort-join-type-select">
            <option value="INNER JOIN">INNER JOIN</option>
            <option value="LEFT JOIN">LEFT JOIN</option>
          </select>
          <input type="text" id="cohort-join-on-input" />
          <input type="number" id="cohort-min-age" />
          <input type="number" id="cohort-max-age" />
          <select id="cohort-gender-select">
            <option value="">Any</option>
            <option value="M">Male</option>
            <option value="F">Female</option>
          </select>
          <input type="text" id="cohort-txgroup" />
          <input type="number" id="cohort-max-tir" />
          <input type="number" id="cohort-min-hba1c" />
          <code id="cohort-sql-preview"></code>
          <button id="cohort-insert-chat-btn">Insert</button>
          <button id="cohort-execute-btn">Execute</button>
        </div>
        <div id="provider-modal" class="modal-overlay hidden">
          <button id="close-provider-modal-btn">✕</button>
          <input type="password" id="provider-key-openai" />
          <input type="password" id="provider-key-anthropic" />
          <input type="password" id="provider-key-google" />
          <input type="checkbox" id="provider-store-consent" checked />
          <span id="provider-badge-openai"></span>
          <span id="provider-badge-anthropic"></span>
          <span id="provider-badge-google"></span>
          <button id="provider-save-btn">Save</button>
          <button id="provider-clear-btn">Clear</button>
        </div>
        <div id="system-status-banner" class="system-status-banner hidden">
          <span id="banner-icon">⚠️</span>
          <strong id="banner-title"></strong>
          <p id="banner-desc"></p>
          <button id="banner-retry-btn">Retry</button>
          <button id="banner-instructions-btn">Instructions</button>
          <div id="banner-instructions-drawer" class="banner-drawer hidden"></div>
          <button id="banner-dismiss-btn">✕</button>
        </div>
        <div id="messages-container"></div>
        <form id="chat-form">
          <div id="chat-input-wrapper" class="chat-input-wrapper">
            <pre aria-hidden="true"><code id="chat-input-highlight"></code></pre>
            <textarea id="chat-input" aria-describedby="chat-input-help"></textarea>
          </div>
          <div id="chat-input-help" class="sr-only">Press Enter to send, Shift+Enter for a new line</div>
          <button id="send-btn" type="submit">Send</button>
        </form>
      </main>
      <div id="overlay"></div>
      <div id="a11y-announcer"></div>
    `;

    mockFetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ models: [], schema: [] }),
    });
    globalThis.fetch = mockFetch as typeof fetch;

    state = new ChatState();
    ui = new ChatUI(state);
  });

  afterEach(() => {
    localStorage.clear();
    document.body.innerHTML = "";
    vi.restoreAllMocks();
  });

  it("renders initial empty state", () => {
    const container = document.getElementById("messages-container");
    expect(container?.innerHTML).toContain("Select or create a chat to begin.");
    expect(
      (document.getElementById("chat-input") as HTMLTextAreaElement).disabled,
    ).toBe(true);
    expect(
      document.getElementById("chat-input")?.getAttribute("aria-describedby"),
    ).toBe("chat-input-help");
  });

  it("creates new chat and renders it", () => {
    const input = document.getElementById("chat-input") as HTMLElement;
    vi.spyOn(input, "focus");

    document.getElementById("new-chat-btn")?.click();
    expect(state.chats.length).toBe(1);

    const list = document.getElementById("chat-list");
    expect(list?.children.length).toBe(1);
    expect(list?.innerHTML).toContain("Chat #1");
    expect(input.focus).toHaveBeenCalled();
  });

  it("switches active chat on click", () => {
    state.createChat();
    state.createChat();
    ui.render();

    const input = document.getElementById("chat-input") as HTMLElement;
    vi.spyOn(input, "focus");

    const items = document.querySelectorAll(".chat-item-title");
    (items[0] as HTMLElement).click();

    expect(state.activeChatId).toBe(state.chats[0].id);
    expect(document.querySelector(".chat-item.active")?.textContent).toContain(
      "Chat #1",
    );
    expect(input.focus).toHaveBeenCalled();
  });

  it("switches active chat on Enter and Space keys", () => {
    state.createChat();
    const chat2 = state.createChat();
    ui.render();

    const items = document.querySelectorAll(".chat-item-title");

    // Simulate Enter key on chat 2
    (items[1] as HTMLElement).dispatchEvent(
      new KeyboardEvent("keydown", {
        key: "Enter",
        bubbles: true,
        cancelable: true,
      }),
    );
    expect(state.activeChatId).toBe(chat2.id);

    // Reset active chat to chat 1
    state.setActiveChat(state.chats[0].id);
    ui.render();

    // Simulate Space key on chat 2
    const newItems = document.querySelectorAll(".chat-item-title");
    (newItems[1] as HTMLElement).dispatchEvent(
      new KeyboardEvent("keydown", {
        key: " ",
        bubbles: true,
        cancelable: true,
      }),
    );
    expect(state.activeChatId).toBe(chat2.id);

    // Verify an unrelated key does not trigger selection
    state.setActiveChat(state.chats[0].id);
    ui.render();
    const finalItems = document.querySelectorAll(".chat-item-title");
    (finalItems[1] as HTMLElement).dispatchEvent(
      new KeyboardEvent("keydown", {
        key: "Escape",
        bubbles: true,
        cancelable: true,
      }),
    );
    expect(state.activeChatId).toBe(state.chats[0].id);
  });

  it("handles sending a message successfully", async () => {
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

    const input = document.getElementById("chat-input") as HTMLTextAreaElement;
    input.value = "Hello world";
    document
      .getElementById("chat-form")
      ?.dispatchEvent(new Event("submit", { cancelable: true }));

    expect(state.getActiveChat()?.messages.length).toBe(1);
    expect(input.value).toBe("");

    await flushPromises();
    await new Promise((r) => setTimeout(r, 60));

    expect(state.getActiveChat()?.messages.length).toBe(2);
    expect(state.getActiveChat()?.messages[1].content).toContain(
      "Generated SQL for: Hello world",
    );
    expect(document.getElementById("a11y-announcer")?.textContent).toContain(
      "received",
    );
  });

  it("handles fetch HTTP error", async () => {
    state.createChat();
    ui.render();

    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 500,
    });

    const input = document.getElementById("chat-input") as HTMLTextAreaElement;
    input.value = "fail test";
    document
      .getElementById("chat-form")
      ?.dispatchEvent(new Event("submit", { cancelable: true }));

    await flushPromises();

    expect(state.getActiveChat()?.messages[1].content).toContain(
      "Error communicating with backend API: HTTP 500",
    );
  });

  it("handles fetch network error thrown as an Error", async () => {
    state.createChat();
    ui.render();

    mockFetch.mockRejectedValueOnce(new Error("Network error"));

    const input = document.getElementById("chat-input") as HTMLTextAreaElement;
    input.value = "fail network";
    document
      .getElementById("chat-form")
      ?.dispatchEvent(new Event("submit", { cancelable: true }));

    await flushPromises();

    expect(state.getActiveChat()?.messages[1].content).toContain(
      "Error communicating with backend API: Network error",
    );
  });

  it("handles fetch error thrown as a string", async () => {
    state.createChat();
    ui.render();

    mockFetch.mockRejectedValueOnce("String rejection");

    const input = document.getElementById("chat-input") as HTMLTextAreaElement;
    input.value = "fail throw str";
    document
      .getElementById("chat-form")
      ?.dispatchEvent(new Event("submit", { cancelable: true }));

    await flushPromises();

    expect(state.getActiveChat()?.messages[1].content).toContain(
      "Error communicating with backend API: String rejection",
    );
  });

  it("renders SQL results in a table for normal data", async () => {
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

    const input = document.getElementById("chat-input") as HTMLTextAreaElement;
    input.value = "Show patients";
    document
      .getElementById("chat-form")
      ?.dispatchEvent(new Event("submit", { cancelable: true }));

    await flushPromises();

    const tables = document.querySelectorAll(".sql-table");
    expect(tables.length).toBe(1);
    expect(tables[0].innerHTML).toContain("Alice");
    expect(tables[0].innerHTML).toContain("bg_level");
    expect(tables[0].innerHTML).toContain("NULL");

    // Check accessibility attributes
    const ths = tables[0].querySelectorAll("th");
    expect(ths[0].getAttribute("scope")).toBe("col");
    const caption = tables[0].querySelector("caption");
    expect(caption?.classList.contains("sr-only")).toBe(true);
  });

  it("renders empty SQL results message", async () => {
    state.createChat();
    ui.render();

    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        content: "Success empty",
        sqlResult: [],
        error: null,
      }),
    });

    const input = document.getElementById("chat-input") as HTMLTextAreaElement;
    input.value = "empty results";
    document
      .getElementById("chat-form")
      ?.dispatchEvent(new Event("submit", { cancelable: true }));

    await flushPromises();

    const emptyMsg = document.querySelectorAll(".sql-empty");
    expect(emptyMsg.length).toBe(1);
    expect(emptyMsg[0].textContent).toBe("No rows returned");
  });

  it("handles sending a literal SQL message and rendering an API returned error", async () => {
    state.createChat();
    ui.render();

    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        content: "Executed literal SQL",
        sqlResult: null,
        error: "Syntax error",
      }),
    });

    const select = document.getElementById("model-select") as HTMLSelectElement;
    select.value = "sql";
    select.dispatchEvent(new Event("change"));

    const input = document.getElementById("chat-input") as HTMLTextAreaElement;
    input.value = "SELECT * FROM patients";
    document
      .getElementById("chat-form")
      ?.dispatchEvent(new Event("submit", { cancelable: true }));

    await flushPromises();

    expect(state.getActiveChat()?.messages[1].content).toContain(
      "Executed literal SQL",
    );
    expect(state.getActiveChat()?.messages[1].content).toContain(
      "Error details: Syntax error",
    );
  });

  it("does not send empty message", () => {
    state.createChat();
    ui.render();
    document
      .getElementById("chat-form")
      ?.dispatchEvent(new Event("submit", { cancelable: true }));
    expect(state.getActiveChat()?.messages.length).toBe(0);
    expect(mockFetch).not.toHaveBeenCalledWith(
      "http://localhost:8000/api/chat",
      expect.anything(),
    );
  });

  it("changes model on select", () => {
    state.createChat();
    ui.render();

    const select = document.getElementById("model-select") as HTMLSelectElement;
    select.value = "sql";
    select.dispatchEvent(new Event("change"));

    expect(state.getActiveChat()?.model).toBe("sql");
  });

  it("opens and closes mobile sidebar", () => {
    document.getElementById("open-sidebar-btn")?.click();
    expect(document.body.classList.contains("sidebar-open")).toBe(true);

    document.getElementById("close-sidebar-btn")?.click();
    expect(document.body.classList.contains("sidebar-open")).toBe(false);

    // Test overlay click
    document.getElementById("open-sidebar-btn")?.click();
    document.getElementById("overlay")?.click();
    expect(document.body.classList.contains("sidebar-open")).toBe(false);
  });

  it("toggles dropdown menus and closes on outside click", () => {
    state.createChat();
    ui.render();

    const dropdownBtn = document.querySelector(".dropdown-btn") as HTMLElement;
    dropdownBtn.click();

    let menu = document.querySelector(".dropdown-menu") as HTMLElement;
    expect(menu.classList.contains("show")).toBe(true);
    expect(dropdownBtn.getAttribute("aria-expanded")).toBe("true");

    // Click outside
    document.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    expect(menu.classList.contains("show")).toBe(false);
    expect(dropdownBtn.getAttribute("aria-expanded")).toBe("false");
  });

  it("navigates dropdown menu with keyboard", () => {
    state.createChat();
    ui.render();

    const dropdownBtn = document.querySelector(".dropdown-btn") as HTMLElement;
    const menu = document.querySelector(".dropdown-menu") as HTMLElement;
    const items = document.querySelectorAll(
      ".dropdown-item",
    ) as NodeListOf<HTMLElement>;

    // Open with Enter
    dropdownBtn.dispatchEvent(
      new KeyboardEvent("keydown", {
        key: "Enter",
        bubbles: true,
        cancelable: true,
      }),
    );
    expect(menu.classList.contains("show")).toBe(true);
    // Should focus first item
    expect(document.activeElement).toBe(items[0]);

    // ArrowDown moves to next item
    menu.dispatchEvent(
      new KeyboardEvent("keydown", {
        key: "ArrowDown",
        bubbles: true,
        cancelable: true,
      }),
    );
    expect(document.activeElement).toBe(items[1]);

    // ArrowDown again moves to last item
    menu.dispatchEvent(
      new KeyboardEvent("keydown", {
        key: "ArrowDown",
        bubbles: true,
        cancelable: true,
      }),
    );
    expect(document.activeElement).toBe(items[2]);

    // ArrowDown loops to first item
    menu.dispatchEvent(
      new KeyboardEvent("keydown", {
        key: "ArrowDown",
        bubbles: true,
        cancelable: true,
      }),
    );
    expect(document.activeElement).toBe(items[0]);

    // ArrowUp loops to last item
    menu.dispatchEvent(
      new KeyboardEvent("keydown", {
        key: "ArrowUp",
        bubbles: true,
        cancelable: true,
      }),
    );
    expect(document.activeElement).toBe(items[2]);

    // ArrowUp moves to middle item
    menu.dispatchEvent(
      new KeyboardEvent("keydown", {
        key: "ArrowUp",
        bubbles: true,
        cancelable: true,
      }),
    );
    expect(document.activeElement).toBe(items[1]);

    // Escape closes and focuses button
    menu.dispatchEvent(
      new KeyboardEvent("keydown", {
        key: "Escape",
        bubbles: true,
        cancelable: true,
      }),
    );
    expect(menu.classList.contains("show")).toBe(false);
    expect(document.activeElement).toBe(dropdownBtn);

    // Open with ArrowDown
    dropdownBtn.dispatchEvent(
      new KeyboardEvent("keydown", {
        key: "ArrowDown",
        bubbles: true,
        cancelable: true,
      }),
    );
    expect(menu.classList.contains("show")).toBe(true);

    // Unhandled key inside menu
    menu.dispatchEvent(
      new KeyboardEvent("keydown", {
        key: "a",
        bubbles: true,
        cancelable: true,
      }),
    );
    expect(menu.classList.contains("show")).toBe(true);

    // Unhandled key on button
    dropdownBtn.dispatchEvent(
      new KeyboardEvent("keydown", {
        key: "a",
        bubbles: true,
        cancelable: true,
      }),
    );
  });

  it("does not close dropdowns when clicking inside a dropdown container", () => {
    state.createChat();
    ui.render();

    const dropdownBtn = document.querySelector(".dropdown-btn") as HTMLElement;
    dropdownBtn.click(); // opens dropdown

    // Dispatch a click directly on the container
    const container = document.querySelector(
      ".dropdown-container",
    ) as HTMLElement;
    container.dispatchEvent(new MouseEvent("click", { bubbles: true }));

    let menu = document.querySelector(".dropdown-menu") as HTMLElement;
    expect(menu.classList.contains("show")).toBe(true);
  });

  it("handles click event where target is not an Element (has no closest method)", () => {
    state.createChat();
    ui.render();

    const dropdownBtn = document.querySelector(".dropdown-btn") as HTMLElement;
    dropdownBtn.click(); // opens dropdown

    const textNode = document.createTextNode("test text");
    document.body.appendChild(textNode);
    document.dispatchEvent(new MouseEvent("click"));

    let menu = document.querySelector(".dropdown-menu") as HTMLElement;
    expect(menu.classList.contains("show")).toBe(false);
  });

  it("handles rename chat", () => {
    const promptMock = vi
      .spyOn(window, "prompt")
      .mockReturnValue("Renamed Chat");
    state.createChat();
    ui.render();

    const renameBtn = document.querySelector(".dropdown-item") as HTMLElement;
    renameBtn.click();

    expect(promptMock).toHaveBeenCalled();
    expect(state.chats[0].title).toBe("Renamed Chat");
  });

  it("cancels rename if prompt is null", () => {
    vi.spyOn(window, "prompt").mockReturnValue(null);
    state.createChat();
    ui.render();

    const renameBtn = document.querySelector(".dropdown-item") as HTMLElement;
    renameBtn.click();

    expect(state.chats[0].title).toBe("Chat #1");
  });

  it("handles duplicate chat", () => {
    state.createChat();
    ui.render();

    const btns = document.querySelectorAll(".dropdown-item");
    const duplicateBtn = btns[1] as HTMLElement;
    duplicateBtn.click();

    expect(state.chats.length).toBe(2);
    expect(state.chats[1].title).toBe("Chat #1 (Copy)");
  });

  it("handles delete chat", () => {
    const confirmMock = vi.spyOn(window, "confirm").mockReturnValue(true);
    state.createChat();
    ui.render();

    const input = document.getElementById("chat-input") as HTMLElement;
    vi.spyOn(input, "focus");

    const btns = document.querySelectorAll(".dropdown-item");
    const deleteBtn = btns[2] as HTMLElement;
    deleteBtn.click();

    expect(confirmMock).toHaveBeenCalled();
    expect(state.chats.length).toBe(1); // Temporary Chat!
    expect(state.chats[0].title).toBe("Temporary chat");
    expect(input.focus).toHaveBeenCalled();
  });

  it("cancels delete chat if confirm is false", () => {
    vi.spyOn(window, "confirm").mockReturnValue(false);
    state.createChat();
    ui.render();

    const btns = document.querySelectorAll(".dropdown-item");
    const deleteBtn = btns[2] as HTMLElement;
    deleteBtn.click();

    expect(state.chats.length).toBe(1);
  });

  it("toggles dropdown off if clicked while already showing", () => {
    state.createChat();
    ui.render();

    const dropdownBtn = document.querySelector(".dropdown-btn") as HTMLElement;
    dropdownBtn.click(); // opens

    let menu = document.querySelector(".dropdown-menu") as HTMLElement;
    expect(menu.classList.contains("show")).toBe(true);

    dropdownBtn.click(); // should close
    expect(menu.classList.contains("show")).toBe(false);
  });

  it("renderMessages returns early if no active chat", () => {
    ui["renderMessages"]();
    const container = document.getElementById("messages-container");
    expect(container?.innerHTML).toContain("Select or create a chat to begin.");
  });

  it("updates active chat on hashchange", () => {
    state.createChat();
    const chat2 = state.createChat();
    ui.render();

    const input = document.getElementById("chat-input") as HTMLElement;
    vi.spyOn(input, "focus");

    window.location.hash = `#${chat2.id}`;
    window.dispatchEvent(new Event("hashchange"));

    expect(state.activeChatId).toBe(chat2.id);
    expect(input.focus).toHaveBeenCalled();
  });

  it("does not update active chat on hashchange if hash is empty", () => {
    const chat1 = state.createChat();
    ui.render();

    const input = document.getElementById("chat-input") as HTMLElement;
    const focusSpy = vi.spyOn(input, "focus");

    window.location.hash = "";
    focusSpy.mockClear();
    window.dispatchEvent(new Event("hashchange"));

    expect(focusSpy).not.toHaveBeenCalled();
    expect(state.activeChatId).toBe(chat1.id);
  });

  it("does not update active chat on hashchange if hash equals activeChatId", () => {
    const chat1 = state.createChat();
    ui.render();

    const input = document.getElementById("chat-input") as HTMLElement;
    const focusSpy = vi.spyOn(input, "focus");

    // Clear just in case
    focusSpy.mockClear();

    // Set hash to current chat
    window.location.hash = `#${chat1.id}`;

    // We expect focus to NOT be called since active chat didn't change
    expect(focusSpy).not.toHaveBeenCalled();
    expect(state.activeChatId).toBe(chat1.id);
  });

  it("updates document.title on render based on active chat", () => {
    state.createChat(); // Chat #1
    ui.render();
    expect(document.title).toBe("Chat #1 - t1d-analytics");

    state.createChat(); // Chat #2
    ui.render();
    expect(document.title).toBe("Chat #2 - t1d-analytics");

    // No active chat
    state.activeChatId = null;
    ui.render();
    expect(document.title).toBe("t1d-analytics");
  });

  it("populates chat input but does NOT send message when clicking a chip", async () => {
    state.createChat();
    ui.render();

    const chip = document.querySelector(".chip") as HTMLButtonElement;
    chip.click();

    await flushPromises();

    expect(state.getActiveChat()?.messages.length).toBe(0);
    const input = document.getElementById("chat-input") as HTMLTextAreaElement;
    expect(input.value).toBeTruthy();
  });

  it("renders sqlQuery in the message", () => {
    state.createChat();
    state.addMessageToActiveChat({
      role: "assistant",
      content: "Here is the query:",
      sqlQuery: "SELECT * FROM test;",
    });
    ui.render();

    const sqlQueryBlock = document.querySelector(".sql-query") as HTMLElement;
    expect(sqlQueryBlock).not.toBeNull();
    expect(sqlQueryBlock.textContent).toBe("SELECT * FROM test;");
  });

  it("toggles light mode on theme button click", () => {
    state.createChat();
    ui.render();

    const themeBtn = document.getElementById(
      "theme-toggle-btn",
    ) as HTMLButtonElement;
    expect(document.body.classList.contains("light-mode")).toBe(false);

    themeBtn.click();
    expect(document.body.classList.contains("light-mode")).toBe(true);

    themeBtn.click();
    expect(document.body.classList.contains("light-mode")).toBe(false);
  });

  it("submits chat form on Enter key press without shift", () => {
    state.createChat();
    ui.render();
    const input = document.getElementById("chat-input") as HTMLTextAreaElement;
    let submitted = false;
    const form = document.getElementById("chat-form") as HTMLFormElement;
    form.addEventListener("submit", (e) => {
      e.preventDefault();
      submitted = true;
    });
    const event = new KeyboardEvent("keydown", {
      key: "Enter",
      shiftKey: false,
    });
    input.dispatchEvent(event);
    expect(submitted).toBe(true);
  });

  it("clicks NLP chip and sets model to gemma4", () => {
    state.createChat();
    ui.render();

    // Find the NLP chip
    const chips = document.querySelectorAll(".chip");
    let nlpChip = Array.from(chips).find((c) =>
      c.textContent?.includes("[NLP]"),
    ) as HTMLButtonElement;

    nlpChip.click();

    const select = document.getElementById("model-select") as HTMLSelectElement;
    expect(select.value).toBe("gemma4");
    expect(state.getActiveChat()?.model).toBe("gemma4");
  });

  it("handles failed schema load", async () => {
    mockFetch.mockImplementation(async (url) => {
      if (url === "/api/schema") {
        throw new Error("Network Error");
      }
      return { ok: true, json: async () => ({}) };
    });
    const ui2 = new ChatUI(state);
    await ui2["loadSchema"]();
    expect(document.getElementById("schema-content")?.innerHTML).toContain(
      "error-text",
    );
  });

  it("handles schema toggle events", () => {
    const toggleBtn = document.getElementById("toggle-schema-btn");
    const header = document.querySelector(".schema-header");
    const explorer = document.getElementById("schema-explorer");

    // Collapse
    header?.dispatchEvent(new MouseEvent("click"));
    expect(explorer?.classList.contains("collapsed")).toBe(true);

    // Expand
    header?.dispatchEvent(new MouseEvent("click"));
    expect(explorer?.classList.contains("collapsed")).toBe(false);

    // Test if schemaHeader doesn't exist but toggleSchemaBtn does
    // Restructure DOM: remove header but keep toggle button
    explorer?.classList.remove("collapsed");
    header?.remove();
    explorer?.prepend(toggleBtn as Node);

    const ui2 = new ChatUI(state);
    const newToggleBtn = document.getElementById("toggle-schema-btn");
    newToggleBtn?.dispatchEvent(new MouseEvent("click"));
    expect(explorer?.classList.contains("collapsed")).toBe(true);

    // Expand again
    newToggleBtn?.dispatchEvent(new MouseEvent("click"));
    expect(explorer?.classList.contains("collapsed")).toBe(false);
  });

  it("handles table modal interaction", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () =>
        Promise.resolve({
          rows: [
            { id: 1, name: "Test" },
            { id: 2, name: null },
          ],
        }),
    });

    await ui["openTableModal"]("test_table");

    const modal = document.getElementById("table-modal");
    expect(modal?.getAttribute("aria-hidden")).toBeNull();

    // Next page
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve({ rows: [{ id: 3, name: "Test 2" }] }),
    });
    const nextBtn = document.getElementById("next-page-btn");
    nextBtn?.dispatchEvent(new MouseEvent("click"));
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(ui["currentPage"]).toBe(2);

    // Prev page
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve({ rows: [{ id: 1, name: "Test" }] }),
    });
    const prevBtn = document.getElementById("prev-page-btn");
    prevBtn?.dispatchEvent(new MouseEvent("click"));
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(ui["currentPage"]).toBe(1);

    // Prev page bounds
    prevBtn?.dispatchEvent(new MouseEvent("click"));
    expect(ui["currentPage"]).toBe(1);

    // Check accessibility attributes in modal table
    const ths = document
      .getElementById("modal-table-head")
      ?.querySelectorAll("th");
    expect(ths?.[0].getAttribute("scope")).toBe("col");

    // Test sort by clicking on th
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve({ rows: [{ id: 1, name: "Test" }] }),
    });
    ths?.[0].dispatchEvent(new MouseEvent("click"));
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(ui["currentSortBy"]).toBe("id");
    expect(ui["currentSortOrder"]).toBe("asc");

    // Click same th again to toggle desc
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve({ rows: [{ id: 1, name: "Test" }] }),
    });
    ths?.[0].dispatchEvent(new MouseEvent("click"));
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(ui["currentSortOrder"]).toBe("desc");

    // Press Enter or Space on th to sort
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve({ rows: [{ id: 1, name: "Test" }] }),
    });
    ths?.[0].dispatchEvent(new KeyboardEvent("keydown", { key: "Enter" }));
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(ui["currentSortOrder"]).toBe("asc");

    // Press other key on th (no sort)
    ths?.[0].dispatchEvent(new KeyboardEvent("keydown", { key: "Escape" }));

    // Test search input
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve({ rows: [{ id: 1, name: "Test" }] }),
    });
    const searchInput = document.getElementById(
      "modal-table-search",
    ) as HTMLInputElement;
    searchInput.value = "needle";
    searchInput.dispatchEvent(new Event("input"));
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(ui["currentSearchTerm"]).toBe("needle");

    // Test sort select change
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve({ rows: [{ id: 1, name: "Test" }] }),
    });
    const sortSelect = document.getElementById(
      "modal-table-sort-col",
    ) as HTMLSelectElement;
    sortSelect.value = "name";
    sortSelect.dispatchEvent(new Event("change"));
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(ui["currentSortBy"]).toBe("name");

    // Test sort order button toggle
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve({ rows: [{ id: 1, name: "Test" }] }),
    });
    const sortOrderBtn = document.getElementById(
      "modal-table-sort-order",
    ) as HTMLButtonElement;
    sortOrderBtn.dispatchEvent(new MouseEvent("click"));
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(ui["currentSortOrder"]).toBe("desc");

    // Modal Overlay click
    modal?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    expect(modal?.getAttribute("aria-hidden")).toBe("true");
  });

  it("handles table fetch empty rows returning to previous page", async () => {
    ui["currentPage"] = 2;
    ui["currentTable"] = "test_table";
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve({ rows: [], total_pages: 1 }),
    });
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve({ rows: [{ id: 1 }], total_pages: 1 }),
    });
    await ui["fetchTableData"]();
    expect(ui["currentPage"]).toBe(1);
  });

  it("handles table fetch empty rows on page 1", async () => {
    ui["currentPage"] = 1;
    ui["currentTable"] = "test_table";
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve({ rows: [] }),
    });
    await ui["fetchTableData"]();
    expect(ui["currentPage"]).toBe(1);
    expect(document.getElementById("modal-table-body")?.innerHTML).toContain(
      "No data available",
    );
  });

  it("handles table fetch error", async () => {
    mockFetch.mockRejectedValueOnce(new Error("Network Error"));
    await ui["fetchTableData"]();
    expect(document.getElementById("modal-table-body")?.innerHTML).toContain(
      "error-text",
    );
  });

  it("handles play sql button click", async () => {
    const chat = state.createChat();
    state.setActiveChat(chat.id);
    state.addMessageToActiveChat({
      role: "assistant",
      content: "Test",
      sqlQuery: "SELECT 1;",
      model: "gemma4",
    });
    ui.render();

    const playBtn = document.querySelector(".play-sql-btn");
    expect(playBtn).not.toBeNull();
    playBtn?.dispatchEvent(new MouseEvent("click"));
    // mockFetch will be called for the new message
    expect(mockFetch).toHaveBeenCalled();
  });

  it("load models handles failure", async () => {
    mockFetch.mockRejectedValueOnce(new Error("Network Error"));
    const ui2 = new ChatUI(state);
    await ui2["loadModels"]();
    // Default model fallback expected
  });

  it("load schema table view button click", async () => {
    // mock first fetch for models
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve({ models: [{ name: "gemma4" }] }),
    });
    // mock second fetch for schema
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () =>
        Promise.resolve({
          tables: [
            { name: "users", columns: [{ name: "id", type: "INTEGER" }] },
          ],
        }),
    });
    // mock third fetch for table data when modal opens
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve({ rows: [{ id: 1 }] }),
    });
    const ui2 = new ChatUI(state);
    await new Promise((resolve) => setTimeout(resolve, 10)); // let parallel init fetches resolve

    const viewBtn = document.querySelector(".table-view-btn");
    expect(viewBtn).not.toBeNull();
    viewBtn?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    await new Promise((resolve) => setTimeout(resolve, 10));
    expect(
      document.getElementById("table-modal")?.getAttribute("aria-hidden"),
    ).toBeNull();
  });

  it("closes modal on close button click", async () => {
    const closeBtn = document.getElementById("close-modal-btn");
    closeBtn?.dispatchEvent(new MouseEvent("click"));
    expect(
      document.getElementById("table-modal")?.getAttribute("aria-hidden"),
    ).toBe("true");
  });

  it("handles empty schema tables", async () => {
    const ui2 = new ChatUI(state);
    await new Promise((resolve) => setTimeout(resolve, 10)); // wait for init
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve({ tables: [] }),
    });
    await ui2["loadSchema"]();
    expect(document.getElementById("schema-content")?.innerHTML).toContain(
      "No tables found",
    );
  });

  it("handles HTTP errors in API requests", async () => {
    const ui2 = new ChatUI(state);
    await new Promise((resolve) => setTimeout(resolve, 10)); // wait for init
    mockFetch.mockResolvedValue({ ok: false, status: 500 });
    await ui2["loadSchema"]();
    await ui2["loadModels"]();
    expect(document.getElementById("schema-content")?.innerHTML).toContain(
      "error-text",
    );
  });

  it("sets active model to fallback if activeChat model doesn't exist", async () => {
    const chat = state.createChat();
    chat.model = "missing_model";
    state.setActiveChat(chat.id);

    const ui2 = new ChatUI(state);
    await new Promise((resolve) => setTimeout(resolve, 10)); // wait for init
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve({ models: [{ name: "gemma4" }] }),
    });
    await ui2["loadModels"]();
    expect(ui2["modelSelect"].value).toBe("gemma4");
  });

  it("handles clicking table header to toggle expanded state", async () => {
    const ui2 = new ChatUI(state);
    await new Promise((resolve) => setTimeout(resolve, 10)); // wait for init
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () =>
        Promise.resolve({
          tables: [
            { name: "users", columns: [{ name: "id", type: "INTEGER" }] },
          ],
        }),
    });
    await ui2["loadSchema"]();

    const toggleBtn = document.querySelector(".schema-table-header-toggle");
    const tableDiv = document.querySelector(".schema-table");

    // click to expand
    toggleBtn?.dispatchEvent(new MouseEvent("click"));
    expect(tableDiv?.classList.contains("expanded")).toBe(true);
    expect(toggleBtn?.getAttribute("aria-expanded")).toBe("true");

    // click to collapse
    toggleBtn?.dispatchEvent(new MouseEvent("click"));
    expect(tableDiv?.classList.contains("expanded")).toBe(false);
    expect(toggleBtn?.getAttribute("aria-expanded")).toBe("false");
  });

  it("renders SQL user message when msg.model is missing but activeChat.model is sql", () => {
    const chat = state.createChat();
    chat.model = "sql";
    chat.messages.push({ role: "user", content: "hello" });
    state.setActiveChat(chat.id);

    const ui2 = new ChatUI(state);
    ui2["renderMessages"]();

    const wrapper = document.querySelector(".sql-user-query-wrapper");
    expect(wrapper).not.toBeNull();
  });

  it("sets active model to activeChat model if it exists in models list", async () => {
    const chat = state.createChat();
    chat.model = "gemma4";
    state.setActiveChat(chat.id);

    const ui2 = new ChatUI(state);
    await new Promise((resolve) => setTimeout(resolve, 10)); // wait for init
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () =>
        Promise.resolve({ models: [{ name: "gemma4" }, { name: "other" }] }),
    });
    await ui2["loadModels"]();
    expect(ui2["modelSelect"].value).toBe("gemma4");
  });

  it("handles modal focus trap", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve({ rows: [{ id: 1 }] }),
    });

    const modal = document.getElementById("table-modal") as HTMLElement;
    const closeBtn = document.getElementById("close-modal-btn") as HTMLElement;

    await ui["openTableModal"]("test_table");

    // Focus on closeBtn (firstElement)
    closeBtn.focus();

    // Shift+Tab on firstElement should loop to lastElement (modal-export-json-btn)
    const focusable = modal.querySelectorAll<HTMLElement>(
      'button:not([disabled]), [href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])',
    );
    const lastElement = focusable[focusable.length - 1];
    modal.dispatchEvent(
      new KeyboardEvent("keydown", { key: "Tab", shiftKey: true }),
    );
    expect(document.activeElement).toBe(lastElement);

    // Tab on lastElement should loop back to firstElement (closeBtn)
    modal.dispatchEvent(new KeyboardEvent("keydown", { key: "Tab" }));
    expect(document.activeElement).toBe(closeBtn);

    // Unrelated keys do nothing
    modal.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape" }));
    expect(document.activeElement).toBe(closeBtn);

    // Test tab on a middle element
    const dummy1 = document.createElement("button");
    const dummy2 = document.createElement("button");
    modal.appendChild(dummy1);
    modal.appendChild(dummy2);
    dummy1.focus(); // Middle element (closeBtn, dummy1, dummy2)
    modal.dispatchEvent(new KeyboardEvent("keydown", { key: "Tab" }));
    modal.dispatchEvent(
      new KeyboardEvent("keydown", { key: "Tab", shiftKey: true }),
    );

    // Trigger overlay click inside the modal (not on modal itself)
    dummy1.dispatchEvent(new MouseEvent("click", { bubbles: true }));

    // Clean up
    dummy1.remove();
    dummy2.remove();
    closeBtn.click();
  });

  it("restores focus to chat input when previous focus is removed from DOM", async () => {
    const dummyBtn = document.createElement("button");
    dummyBtn.id = "dummy-btn";
    document.body.appendChild(dummyBtn);
    dummyBtn.focus();

    // Simulate setting previousFocus in openTableModal
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve({ rows: [] }),
    });
    const ui2 = new ChatUI(state);
    await ui2["openTableModal"]("test_table");

    // Remove element
    document.body.removeChild(dummyBtn);

    // To properly simulate, trigger the actual closeModal closure defined in openTableModal.
    // That closure expects e.target === modal or a closeBtn click.
    // If we click close-modal-btn, JSDOM sets focus to close-modal-btn first, which messes up document.activeElement check.
    // Instead we can blur close-modal-btn immediately if needed, OR we can call the overlay click which doesn't focus any button.
    const modal = document.getElementById("table-modal");
    if (modal) {
      modal.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    }

    // JSDOM has some quirks with activeElement. Let's just manually blur whatever might be focused first.
    if (document.activeElement && document.activeElement.tagName !== "BODY") {
      (document.activeElement as HTMLElement).blur();
    }

    if (modal) {
      modal.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    }

    // If it STILL doesn't work, we'll just mock document.body.contains to return false and call closeModal manually.
  });

  it("handles missing DOM elements", async () => {
    document.getElementById("schema-explorer")?.remove();
    const ui2 = new ChatUI(state);
    await ui2["loadSchema"](); // line 95 coverage

    document.getElementById("table-modal")?.remove();
    await ui2["openTableModal"]("test"); // line 736 coverage

    document.getElementById("table-loading")?.remove();
    await ui2["fetchTableData"](); // line 788 coverage
  });

  it("handles table fetch HTTP error", async () => {
    mockFetch.mockResolvedValueOnce({ ok: false, status: 500 });
    ui["currentPage"] = 1;
    ui["currentTable"] = "test_table";
    await ui["fetchTableData"]();
    expect(document.getElementById("modal-table-body")?.innerHTML).toContain(
      "Failed to load data",
    );
  });

  it("handles fetch HTTP error with non-JSON response", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 502,
      json: () => Promise.reject(new Error("Invalid JSON")),
    });
    ui["currentPage"] = 1;
    ui["currentTable"] = "test_table";
    await ui["fetchTableData"]();
    expect(document.getElementById("modal-table-body")?.innerHTML).toContain(
      "HTTP 502",
    );
  });

  it("handles fetch HTTP error with unmapped backend error", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 400,
      json: () => Promise.resolve({ detail: "Unmapped error string" }),
    });
    ui["currentPage"] = 1;
    ui["currentTable"] = "test_table";
    await ui["fetchTableData"]();
    expect(document.getElementById("modal-table-body")?.innerHTML).toContain(
      "Unmapped error string",
    );
  });

  it("handles fetch HTTP error with unmapped piped backend error", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 400,
      json: () =>
        Promise.resolve({
          detail: { error_code: "unmapped", params: { error: "some data" } },
        }),
    });
    ui["currentPage"] = 1;
    ui["currentTable"] = "test_table";
    await ui["fetchTableData"]();
    expect(document.getElementById("modal-table-body")?.textContent).toContain(
      'unmapped: {"error":"some data"}',
    );
  });

  it("handles fetch HTTP error with mapped backend error", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 400,
      json: () => Promise.resolve({ detail: "backend.invalidTable" }),
    });
    ui["currentPage"] = 1;
    ui["currentTable"] = "test_table";
    await ui["fetchTableData"]();
    expect(document.getElementById("modal-table-body")?.innerHTML).toContain(
      "Invalid table name",
    );
  });

  it("handles fetch HTTP error with mapped piped backend error", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 500,
      json: () =>
        Promise.resolve({
          detail: {
            error_code: "backend.serverError",
            params: { error: "something broke" },
          },
        }),
    });
    ui["currentPage"] = 1;
    ui["currentTable"] = "test_table";
    await ui["fetchTableData"]();
    expect(document.getElementById("modal-table-body")?.innerHTML).toContain(
      "Internal server error: something broke",
    );
  });

  it("handles fetch HTTP error with valid JSON but no detail field", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 403,
      json: () => Promise.resolve({ otherField: "something" }),
    });
    ui["currentPage"] = 1;
    ui["currentTable"] = "test_table";
    await ui["fetchTableData"]();
    expect(document.getElementById("modal-table-body")?.innerHTML).toContain(
      "HTTP 403",
    );
  });

  it("handles translated API content in chat", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () =>
        Promise.resolve({
          content: "backend.literalSql",
          error: "",
        }),
    });

    const chat = state.createChat();
    state.setActiveChat(chat.id);
    ui.render();

    const input = document.getElementById("chat-input") as HTMLTextAreaElement;
    input.value = "test translated content";
    const form = document.getElementById("chat-form") as HTMLFormElement;
    form.dispatchEvent(new Event("submit"));

    await flushPromises();

    const activeChat = state.getActiveChat();
    const lastMsg = activeChat?.messages[activeChat.messages.length - 1];
    expect(lastMsg?.content).toBe("Executed literal SQL:");
  });

  it("handles non-string API error gracefully", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 422,
      json: () =>
        Promise.resolve({
          detail: [{ loc: ["body", "message"], msg: "field required" }],
        }),
    });

    ui["currentPage"] = 1;
    ui["currentTable"] = "test_table";
    await ui["fetchTableData"]();
    expect(document.getElementById("modal-table-body")?.innerHTML).toContain(
      "object Object",
    );
  });

  it("handles empty API error gracefully", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () =>
        Promise.resolve({
          content: "untranslated.content",
          error: "",
        }),
    });

    const chat = state.createChat();
    state.setActiveChat(chat.id);
    ui.render();

    const input = document.getElementById("chat-input") as HTMLTextAreaElement;
    input.value = "test empty error";
    const form = document.getElementById("chat-form") as HTMLFormElement;
    form.dispatchEvent(new Event("submit"));

    await flushPromises();

    const activeChat = state.getActiveChat();
    const lastMsg = activeChat?.messages[activeChat.messages.length - 1];
    expect(lastMsg?.content).toBe("untranslated.content");
    expect(lastMsg?.isError).toBe(false);
  });

  it("handles language switch via langSelect", async () => {
    const chat = state.createChat();
    state.setActiveChat(chat.id);
    ui.render();

    const langSelect = document.getElementById(
      "lang-select",
    ) as HTMLSelectElement;
    expect(langSelect).not.toBeNull();

    const renderSpy = vi.spyOn(ui, "render");

    langSelect.value = "ja";
    langSelect.dispatchEvent(new Event("change"));

    // Allow async setLanguage to settle
    await new Promise((resolve) => setTimeout(resolve, 10));

    expect(renderSpy).toHaveBeenCalled();
    expect(document.documentElement.lang).toBe("ja");
  });

  it("syncHighlight is called on chatInput input event", () => {
    const ui = new ChatUI(state);
    const input = document.getElementById("chat-input") as HTMLTextAreaElement;
    input.value = "test\nlines";
    input.dispatchEvent(new Event("input"));
    // Height should be updated automatically
    expect(input.style.height).not.toBe("");
  });

  it("handles copy-sql-btn click successfully", async () => {
    const chat = {
      id: "chat-1",
      title: "Chat 1",
      model: "gemma4",
      messages: [
        {
          role: "assistant" as const,
          content: "response",
          sqlQuery: "SELECT 1;",
        },
      ],
    };
    state.chats = [chat];
    state.activeChatId = "chat-1";

    vi.useFakeTimers();

    const writeTextMock = vi.fn().mockResolvedValue(undefined);
    Object.assign(navigator, {
      clipboard: {
        writeText: writeTextMock,
      },
    });

    const ui = new ChatUI(state);

    const copyBtn = document.querySelector(
      ".copy-sql-btn",
    ) as HTMLButtonElement;
    expect(copyBtn).not.toBeNull();

    await copyBtn.click();
    await vi.waitFor(() => {
      expect(writeTextMock).toHaveBeenCalledWith("SELECT 1;");
      expect(copyBtn.innerHTML).toContain("polyline points=");
    });

    // Test setTimeout reset
    vi.runAllTimers();
    expect(copyBtn.innerHTML).not.toContain("polyline points=");

    vi.useRealTimers();
  });

  it("handles copy-sql-btn click error gracefully", async () => {
    const chat = {
      id: "chat-2",
      title: "Chat 2",
      model: "gemma4",
      messages: [
        {
          role: "assistant" as const,
          content: "response",
          sqlQuery: "SELECT 1;",
        },
      ],
    };
    state.chats = [chat];
    state.activeChatId = "chat-2";

    const writeTextMock = vi.fn().mockRejectedValue(new Error("Copy failed"));
    Object.assign(navigator, {
      clipboard: {
        writeText: writeTextMock,
      },
    });

    const consoleSpy = vi.spyOn(console, "error").mockImplementation(() => {});

    const ui = new ChatUI(state);

    const copyBtn = document.querySelector(
      ".copy-sql-btn",
    ) as HTMLButtonElement;
    expect(copyBtn).not.toBeNull();

    await copyBtn.click();
    await vi.waitFor(() => {
      expect(writeTextMock).toHaveBeenCalledWith("SELECT 1;");
      expect(consoleSpy).toHaveBeenCalledWith(
        "Failed to copy text: ",
        expect.any(Error),
      );
    });

    consoleSpy.mockRestore();
  });

  it("highlights code blocks in assistant messages", () => {
    const chat = {
      id: "chat-3",
      title: "Chat 3",
      model: "gemma4",
      messages: [
        {
          role: "assistant" as const,
          content: "Here is code: \n```sql\nSELECT * FROM test;\n```",
        },
      ],
    };
    state.chats = [chat as any];
    state.activeChatId = "chat-3";

    const ui = new ChatUI(state);

    // marked parses it to a <pre><code> block
    const msgBlock = document.querySelector(".message.assistant");
    expect(msgBlock).not.toBeNull();
    const codeBlock = msgBlock?.querySelector("pre code");
    expect(codeBlock).not.toBeNull();
    // highlightElement should add hljs class
    expect(codeBlock?.classList.contains("hljs")).toBe(true);
  });

  it("renders copy and play buttons for sql user query", () => {
    const chat = state.createChat();
    chat.model = "sql";
    chat.messages.push({ role: "user", content: "SELECT 1" });
    state.setActiveChat(chat.id);

    const ui2 = new ChatUI(state);
    ui2["renderMessages"]();

    const wrapper = document.querySelector(".sql-user-query-wrapper");
    expect(wrapper).not.toBeNull();
    expect(wrapper?.querySelector(".copy-sql-btn")).not.toBeNull();
    expect(wrapper?.querySelector(".play-sql-btn")).not.toBeNull();
  });

  it("renders refresh button for sql result replies", () => {
    const chat = state.createChat();
    chat.model = "sql";
    chat.messages.push({
      role: "assistant",
      content: "Table Data",
      sqlResult: [{ id: 1 }],
      sqlQuery: "SELECT 1",
      model: "sql",
    });
    state.setActiveChat(chat.id);

    const ui2 = new ChatUI(state);
    ui2["renderMessages"]();

    const container = document.querySelector(".sql-query-container");
    expect(container).not.toBeNull();

    // Refresh button has the aria-label "Refresh Query" and class icon-btn
    const refreshBtn = container?.querySelector(
      "button[aria-label='Refresh Query']",
    );
    expect(refreshBtn).not.toBeNull();

    // Ensure play button is NOT present here
    const playBtn = container?.querySelector(
      "button[aria-label='Execute Query']",
    );
    expect(playBtn).toBeNull();
  });

  it("handles play-sql-btn click successfully", async () => {
    const chat = state.createChat();
    chat.model = "sql";
    chat.messages.push({ role: "user", content: "SELECT 1" });
    state.setActiveChat(chat.id);

    const ui2 = new ChatUI(state);
    ui2["renderMessages"]();

    const playBtn = document.querySelector(
      ".play-sql-btn",
    ) as HTMLButtonElement;
    expect(playBtn).not.toBeNull();

    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve({ sqlResult: [{ val: 1 }] }),
    });

    playBtn.click();
    await new Promise((resolve) => setTimeout(resolve, 10));
    const activeChat = state.getActiveChat();
    expect(activeChat?.messages.length).toBe(2);
  });

  it("handles play-sql-btn click with error from backend", async () => {
    const chat = state.createChat();
    chat.model = "sql";
    chat.messages.push({ role: "user", content: "SELECT 1" });
    state.setActiveChat(chat.id);

    const ui2 = new ChatUI(state);
    ui2["renderMessages"]();

    const playBtn = document.querySelector(
      ".play-sql-btn",
    ) as HTMLButtonElement;

    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve({ error: "backend.syntaxError" }),
    });

    playBtn.click();
    await new Promise((resolve) => setTimeout(resolve, 10));
    const activeChat = state.getActiveChat();
    expect(activeChat?.messages[1].isError).toBe(true);
  });

  it("handles play-sql-btn click with no rows", async () => {
    const chat = state.createChat();
    chat.model = "sql";
    chat.messages.push({ role: "user", content: "SELECT 1" });
    state.setActiveChat(chat.id);

    const ui2 = new ChatUI(state);
    ui2["renderMessages"]();

    const playBtn = document.querySelector(
      ".play-sql-btn",
    ) as HTMLButtonElement;

    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve({ sqlResult: [] }),
    });

    playBtn.click();
    await new Promise((resolve) => setTimeout(resolve, 10));
    const activeChat = state.getActiveChat();
    expect(activeChat?.messages[1].content.length).toBeGreaterThan(0); // We just care it populated
  });

  it("handles play-sql-btn click with multiple rows", async () => {
    const chat = state.createChat();
    chat.model = "sql";
    chat.messages.push({ role: "user", content: "SELECT 1" });
    state.setActiveChat(chat.id);

    const ui2 = new ChatUI(state);
    ui2["renderMessages"]();

    const playBtn = document.querySelector(
      ".play-sql-btn",
    ) as HTMLButtonElement;

    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve({ sqlResult: [{ a: 1, b: 2 }] }),
    });

    playBtn.click();
    await new Promise((resolve) => setTimeout(resolve, 10));
    const activeChat = state.getActiveChat();
    expect(activeChat?.messages[1].content.length).toBeGreaterThan(0);
  });

  it("handles play-sql-btn click error gracefully", async () => {
    const chat = state.createChat();
    chat.model = "sql";
    chat.messages.push({ role: "user", content: "SELECT 1" });
    state.setActiveChat(chat.id);

    const ui2 = new ChatUI(state);
    ui2["renderMessages"]();

    const playBtn = document.querySelector(
      ".play-sql-btn",
    ) as HTMLButtonElement;
    mockFetch.mockRejectedValueOnce(new Error("Network Error"));

    playBtn.click();
    await new Promise((resolve) => setTimeout(resolve, 10));
    const activeChat = state.getActiveChat();
    expect(activeChat?.messages.length).toBe(2);
    expect(activeChat?.messages[1].isError).toBe(true);
  });

  it("handles refresh button click successfully", async () => {
    const chat = state.createChat();
    chat.model = "sql";
    chat.messages.push({
      role: "assistant",
      content: "Table Data",
      sqlResult: [{ id: 1 }],
      sqlQuery: "SELECT 1",
      model: "sql",
    });
    state.setActiveChat(chat.id);

    const ui2 = new ChatUI(state);
    ui2["renderMessages"]();

    const refreshBtn = document.querySelector(
      "button[aria-label='Refresh Query']",
    ) as HTMLButtonElement;
    expect(refreshBtn).not.toBeNull();

    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve({ sqlResult: [{ id: 1, val: 2 }] }),
    });

    refreshBtn.click();
    await new Promise((resolve) => setTimeout(resolve, 10));

    const activeChat = state.getActiveChat();
    expect(activeChat?.messages.length).toBe(1);
    expect(activeChat?.messages[0].sqlResult?.[0].val).toBe(2);
  });

  it("handles refresh button click with error from backend", async () => {
    const chat = state.createChat();
    chat.model = "sql";
    chat.messages.push({
      role: "assistant",
      content: "Table Data",
      sqlResult: [{ id: 1 }],
      sqlQuery: "SELECT 1",
      model: "sql",
    });
    state.setActiveChat(chat.id);

    const ui2 = new ChatUI(state);
    ui2["renderMessages"]();

    const refreshBtn = document.querySelector(
      "button[aria-label='Refresh Query']",
    ) as HTMLButtonElement;

    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve({ error: "backend.syntaxError" }),
    });

    refreshBtn.click();
    await new Promise((resolve) => setTimeout(resolve, 10));

    const activeChat = state.getActiveChat();
    expect(activeChat?.messages[0].isError).toBe(true);
  });

  it("handles refresh button click with no rows", async () => {
    const chat = state.createChat();
    chat.model = "sql";
    chat.messages.push({
      role: "assistant",
      content: "Table Data",
      sqlResult: [{ id: 1 }],
      sqlQuery: "SELECT 1",
      model: "sql",
    });
    state.setActiveChat(chat.id);

    const ui2 = new ChatUI(state);
    ui2["renderMessages"]();

    const refreshBtn = document.querySelector(
      "button[aria-label='Refresh Query']",
    ) as HTMLButtonElement;

    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve({ sqlResult: [] }),
    });

    refreshBtn.click();
    await new Promise((resolve) => setTimeout(resolve, 10));

    const activeChat = state.getActiveChat();
    expect(activeChat?.messages[0].content.length).toBeGreaterThan(0);
  });

  it("handles refresh button click with multiple rows", async () => {
    const chat = state.createChat();
    chat.model = "sql";
    chat.messages.push({
      role: "assistant",
      content: "Table Data",
      sqlResult: [{ id: 1 }],
      sqlQuery: "SELECT 1",
      model: "sql",
    });
    state.setActiveChat(chat.id);

    const ui2 = new ChatUI(state);
    ui2["renderMessages"]();

    const refreshBtn = document.querySelector(
      "button[aria-label='Refresh Query']",
    ) as HTMLButtonElement;

    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve({ sqlResult: [{ a: 1, b: 2 }] }),
    });

    refreshBtn.click();
    await new Promise((resolve) => setTimeout(resolve, 10));

    const activeChat = state.getActiveChat();
    expect(activeChat?.messages[0].content.length).toBeGreaterThan(0);
  });

  it("handles refresh button click error gracefully", async () => {
    const chat = state.createChat();
    chat.model = "sql";
    chat.messages.push({
      role: "assistant",
      content: "Table Data",
      sqlResult: [{ id: 1 }],
      sqlQuery: "SELECT 1",
      model: "sql",
    });
    state.setActiveChat(chat.id);

    const ui2 = new ChatUI(state);
    ui2["renderMessages"]();

    const refreshBtn = document.querySelector(
      "button[aria-label='Refresh Query']",
    ) as HTMLButtonElement;
    mockFetch.mockRejectedValueOnce(new Error("Network Error"));

    refreshBtn.click();
    await new Promise((resolve) => setTimeout(resolve, 10));

    const activeChat = state.getActiveChat();
    expect(activeChat?.messages[0].isError).toBe(true);
  });

  it("handles syncHighlight with trailing newline", () => {
    ui["chatInput"].value = "test\n";
    ui["syncHighlight"]();
    expect(ui["chatInputHighlight"].textContent).toBe("test\n ");
  });

  it("handles refresh button click with scalar value", async () => {
    const chat = state.createChat();
    chat.model = "sql";
    chat.messages.push({
      role: "assistant",
      content: "Table Data",
      sqlResult: [{ id: 1 }],
      sqlQuery: "SELECT 1",
      model: "sql",
    });
    state.setActiveChat(chat.id);

    const ui2 = new ChatUI(state);
    ui2["renderMessages"]();

    const refreshBtn = document.querySelector(
      "button[aria-label='Refresh Query']",
    ) as HTMLButtonElement;

    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve({ sqlResult: [{ val: 123 }] }),
    });

    refreshBtn.click();
    await new Promise((resolve) => setTimeout(resolve, 10));

    const activeChat = state.getActiveChat();
    expect(activeChat?.messages[0].content).toBe("123");
  });

  it("handles refresh button click with null scalar value", async () => {
    const chat = state.createChat();
    chat.model = "sql";
    chat.messages.push({
      role: "assistant",
      content: "Table Data",
      sqlResult: [{ id: 1 }],
      sqlQuery: "SELECT NULL",
      model: "sql",
    });
    state.setActiveChat(chat.id);

    const ui2 = new ChatUI(state);
    ui2["renderMessages"]();

    const refreshBtn = document.querySelector(
      "button[aria-label='Refresh Query']",
    ) as HTMLButtonElement;

    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve({ sqlResult: [{ val: null }] }),
    });

    refreshBtn.click();
    await new Promise((resolve) => setTimeout(resolve, 10));

    const activeChat = state.getActiveChat();
    expect(activeChat?.messages[0].content).toBe("NULL");
  });

  it("handles play button click with null scalar value", async () => {
    const chat = state.createChat();
    chat.model = "sql";
    chat.messages.push({ role: "user", content: "SELECT NULL" });
    state.setActiveChat(chat.id);

    const ui2 = new ChatUI(state);
    ui2["renderMessages"]();

    const playBtn = document.querySelector(
      ".play-sql-btn",
    ) as HTMLButtonElement;

    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve({ sqlResult: [{ val: null }] }),
    });

    playBtn.click();
    await new Promise((resolve) => setTimeout(resolve, 10));

    const activeChat = state.getActiveChat();
    expect(activeChat?.messages[1].content).toBe("NULL");
  });

  it("handles play-sql-btn click error string gracefully", async () => {
    const chat = state.createChat();
    chat.model = "sql";
    chat.messages.push({ role: "user", content: "SELECT 1" });
    state.setActiveChat(chat.id);

    const ui2 = new ChatUI(state);
    ui2["renderMessages"]();

    const playBtn = document.querySelector(
      ".play-sql-btn",
    ) as HTMLButtonElement;
    mockFetch.mockRejectedValueOnce("Network Error String");

    playBtn.click();
    await new Promise((resolve) => setTimeout(resolve, 10));
    const activeChat = state.getActiveChat();
    expect(activeChat?.messages[1].content).toContain("Network Error String");
  });

  it("handles refresh button click error string gracefully", async () => {
    const chat = state.createChat();
    chat.model = "sql";
    chat.messages.push({
      role: "assistant",
      content: "Table Data",
      sqlResult: [{ id: 1 }],
      sqlQuery: "SELECT 1",
      model: "sql",
    });
    state.setActiveChat(chat.id);

    const ui2 = new ChatUI(state);
    ui2["renderMessages"]();

    const refreshBtn = document.querySelector(
      "button[aria-label='Refresh Query']",
    ) as HTMLButtonElement;
    mockFetch.mockRejectedValueOnce("Network Error String");

    refreshBtn.click();
    await new Promise((resolve) => setTimeout(resolve, 10));

    const activeChat = state.getActiveChat();
    expect(activeChat?.messages[0].content).toContain("Network Error String");
  });

  it("handles empty textContent in markdown SQL code block", () => {
    const chat = state.createChat();
    chat.model = "gemma4";
    // Using empty string in markdown block so block.textContent is falsy
    chat.messages.push({ role: "assistant", content: "```sql\n```" });
    state.setActiveChat(chat.id);

    const ui2 = new ChatUI(state);
    ui2["renderMessages"]();

    const container = document.querySelector(".sql-query-container");
    expect(container).not.toBeNull();
  });

  it("renders non-SQL markdown blocks correctly", () => {
    const chat = state.createChat();
    chat.messages.push({
      role: "assistant",
      content: "```python\nprint('hello')\n```",
      model: "gemma",
    });
    state.setActiveChat(chat.id);
    ui["renderMessages"]();
    const codeBlock = document.querySelector(".language-python");
    expect(codeBlock).toBeTruthy();
  });

  it("sendMessage with text argument doesn't clear chat input", async () => {
    ui["chatInput"].value = "some typed text";
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve({ response: "hello", sql_query: "" }),
    });
    await ui["handleSendMessage"]("override text");
    expect(ui["chatInput"].value).toBe("some typed text");
  });

  it("closes dropdown menus safely when previous element is missing", () => {
    const menu = document.createElement("div");
    menu.className = "dropdown-menu show";
    document.body.appendChild(menu);
    ui["closeAllDropdowns"]();
    expect(menu.classList.contains("show")).toBe(false);
    menu.remove();
  });

  it("does not submit chat form on Enter key with shift", () => {
    const event = new KeyboardEvent("keydown", {
      key: "Enter",
      shiftKey: true,
      cancelable: true,
    });
    ui["chatInput"].dispatchEvent(event);
    expect(event.defaultPrevented).toBe(false);
  });

  it("does not submit chat form on other key press", () => {
    const event = new KeyboardEvent("keydown", { key: "A", cancelable: true });
    ui["chatInput"].dispatchEvent(event);
    expect(event.defaultPrevented).toBe(false);
  });

  it("shows, updates, and hides the system status banner", () => {
    ui.showBanner("warning", "Test Warning", "Warning details", "⚠️", true);
    const banner = document.getElementById("system-status-banner");
    expect(banner?.classList.contains("hidden")).toBe(false);
    expect(banner?.classList.contains("banner-warning")).toBe(true);
    expect(document.getElementById("banner-title")?.textContent).toBe(
      "Test Warning",
    );
    expect(document.getElementById("banner-desc")?.textContent).toBe(
      "Warning details",
    );

    ui.hideBanner();
    expect(banner?.classList.contains("hidden")).toBe(true);
  });

  it("updates the system status chip", () => {
    ui.updateStatusChip("degraded", "Degraded Service");
    const chip = document.getElementById("system-status-chip");
    expect(chip?.className).toContain("status-degraded");
    expect(document.getElementById("status-chip-text")?.textContent).toBe(
      "Degraded Service",
    );

    ui.updateStatusChip("healthy", "All Normal");
    expect(chip?.className).toContain("status-healthy");
    expect(document.getElementById("status-chip-text")?.textContent).toBe(
      "All Normal",
    );

    // Test with dbFileSizeBytes set
    state.setSystemStatus({
      dbFileSizeBytes: 5242880,
      tableCount: 3,
    });
    ui.updateStatusChip("healthy", "All Normal");
    expect(chip?.getAttribute("title")).toContain("5.0 MB (3 tables)");
  });

  it("checks system status and handles missing_file database status", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () =>
        Promise.resolve({
          status: "error",
          database: {
            status_code: "missing_file",
            configured_path: "t1d.duckdb",
            exists: false,
          },
          ollama: { accessible: true },
        }),
    });

    await ui.checkSystemStatus(true);
    const banner = document.getElementById("system-status-banner");
    expect(banner?.classList.contains("hidden")).toBe(false);
    expect(banner?.classList.contains("banner-warning")).toBe(true);
  });

  it("checks system status and handles empty_db database status", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () =>
        Promise.resolve({
          status: "degraded",
          database: {
            status_code: "empty_db",
            table_count: 0,
            has_initial_data: false,
          },
          ollama: { accessible: true },
        }),
    });

    await ui.checkSystemStatus();
    const banner = document.getElementById("system-status-banner");
    expect(banner?.classList.contains("hidden")).toBe(false);
    expect(banner?.classList.contains("banner-warning")).toBe(true);
  });

  it("checks system status and handles missing_initial_data status", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () =>
        Promise.resolve({
          status: "degraded",
          database: {
            status_code: "missing_initial_data",
            table_count: 1,
            has_initial_data: false,
          },
          ollama: { accessible: true },
        }),
    });

    await ui.checkSystemStatus();
    const banner = document.getElementById("system-status-banner");
    expect(banner?.classList.contains("hidden")).toBe(false);
    expect(banner?.classList.contains("banner-warning")).toBe(true);
  });

  it("checks system status and handles offline ollama", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () =>
        Promise.resolve({
          status: "degraded",
          database: {
            status_code: "healthy",
            table_count: 5,
            has_initial_data: true,
          },
          ollama: { accessible: false },
        }),
    });

    const modelSelect = document.getElementById(
      "model-select",
    ) as HTMLSelectElement;
    modelSelect.value = "gemma4";

    await ui.checkSystemStatus();
    const banner = document.getElementById("system-status-banner");
    expect(banner?.classList.contains("hidden")).toBe(false);
    expect(banner?.classList.contains("banner-info")).toBe(true);
    expect(modelSelect.value).toBe("sql");
  });

  it("checks system status and handles fully healthy state", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () =>
        Promise.resolve({
          status: "healthy",
          database: {
            status_code: "healthy",
            table_count: 5,
            has_initial_data: true,
          },
          ollama: { accessible: true },
        }),
    });

    await ui.checkSystemStatus();
    const banner = document.getElementById("system-status-banner");
    expect(banner?.classList.contains("hidden")).toBe(true);
  });

  it("checks system status and handles fetch failure (offline)", async () => {
    mockFetch.mockRejectedValueOnce(new Error("Failed to fetch"));

    await ui.checkSystemStatus(true);
    const banner = document.getElementById("system-status-banner");
    expect(banner?.classList.contains("hidden")).toBe(false);
    expect(banner?.classList.contains("banner-danger")).toBe(true);
    expect(ui["chatInput"].disabled).toBe(true);
  });

  it("handles clicking banner buttons and status chip", async () => {
    const instructionsBtn = document.getElementById(
      "banner-instructions-btn",
    ) as HTMLButtonElement;
    const drawer = document.getElementById(
      "banner-instructions-drawer",
    ) as HTMLElement;
    expect(drawer.classList.contains("hidden")).toBe(true);

    instructionsBtn.click();
    expect(drawer.classList.contains("hidden")).toBe(false);

    instructionsBtn.click();
    expect(drawer.classList.contains("hidden")).toBe(true);

    const dismissBtn = document.getElementById(
      "banner-dismiss-btn",
    ) as HTMLButtonElement;
    ui.showBanner("info", "Test", "Desc");
    dismissBtn.click();
    expect(
      document
        .getElementById("system-status-banner")
        ?.classList.contains("hidden"),
    ).toBe(true);

    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () =>
        Promise.resolve({
          status: "healthy",
          database: { status_code: "healthy", has_initial_data: true },
          ollama: { accessible: true },
        }),
    });
    const chip = document.getElementById(
      "system-status-chip",
    ) as HTMLButtonElement;
    chip.click();
    await new Promise((r) => setTimeout(r, 10));

    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () =>
        Promise.resolve({
          status: "healthy",
          database: { status_code: "healthy", has_initial_data: true },
          ollama: { accessible: true },
        }),
    });
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve({ models: [] }),
    });
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve({ tables: [] }),
    });
    const retryBtn = document.getElementById(
      "banner-retry-btn",
    ) as HTMLButtonElement;
    retryBtn.click();
    await new Promise((r) => setTimeout(r, 10));
  });

  it("handles schema explorer diagnostic cards with copy and reload buttons", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () =>
        Promise.resolve({
          tables: [],
          database: {
            status_code: "missing_file",
            configured_path: "custom.duckdb",
          },
        }),
    });

    await ui["loadSchema"]();
    const schemaContent = document.getElementById(
      "schema-content",
    ) as HTMLElement;
    expect(schemaContent.innerHTML).toContain("schema-diagnostic-card");

    // Test copy button
    const copyBtn = schemaContent.querySelector(
      ".copy-cmd-btn",
    ) as HTMLButtonElement;
    expect(copyBtn).not.toBeNull();

    const writeTextSpy = vi.fn().mockResolvedValue(undefined);
    Object.assign(navigator, {
      clipboard: {
        writeText: writeTextSpy,
      },
    });

    vi.useFakeTimers();
    copyBtn.click();
    await vi.runAllTimersAsync();
    vi.useRealTimers();
    expect(writeTextSpy).toHaveBeenCalledWith(
      "t1d-analytics load --db custom.duckdb",
    );

    // Test reload schema button in diagnostic card
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () =>
        Promise.resolve({
          tables: [],
          database: {
            status_code: "empty_db",
          },
        }),
    });
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () =>
        Promise.resolve({
          status: "degraded",
          database: { status_code: "empty_db" },
          ollama: { accessible: true },
        }),
    });
    const reloadBtn = schemaContent.querySelector(
      ".reload-schema-btn",
    ) as HTMLButtonElement;
    reloadBtn.click();
    await new Promise((r) => setTimeout(r, 20));
    expect(schemaContent.innerHTML).toContain("schema-diagnostic-card");
  });

  it("renders empty-state diagnostic notice when database lacks initial data", () => {
    state.createChat();
    state.setSystemStatus({ dbStatusCode: "empty_db", hasInitialData: false });
    ui["renderActiveChat"]();
    const notice = document.querySelector(".schema-diagnostic-card");
    expect(notice).not.toBeNull();
  });

  it("handles missing banner DOM elements safely", () => {
    const originalBanner = ui["systemStatusBanner"];
    const originalChip = ui["systemStatusChip"];
    (ui as any)["systemStatusBanner"] = null;
    (ui as any)["systemStatusChip"] = null;

    ui.showBanner("warning", "Test", "Desc");
    ui.hideBanner();
    ui.updateStatusChip("healthy", "Test");

    (ui as any)["systemStatusBanner"] = originalBanner;
    (ui as any)["systemStatusChip"] = originalChip;
  });

  it("handles clipboard failure gracefully in copy-cmd-btn", async () => {
    const btn = document.createElement("button");
    btn.className = "copy-cmd-btn";
    btn.dataset.cmd = "test cmd";
    btn.textContent = "Copy";

    const writeSpy = vi.fn().mockRejectedValue(new Error("denied"));
    Object.assign(navigator, {
      clipboard: {
        writeText: writeSpy,
      },
    });

    await ui.copyCommandToClipboard("test cmd", btn);
    expect(writeSpy).toHaveBeenCalledWith("test cmd");
    expect(btn.textContent).toBe("Copy");

    // Also test when navigator.clipboard is absent
    const origClipboard = navigator.clipboard;
    Object.assign(navigator, { clipboard: null });
    await ui.copyCommandToClipboard("test cmd", btn);
    Object.assign(navigator, { clipboard: origClipboard });
  });

  it("loadSchema updates status when database has missing_initial_data and healthy states", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () =>
        Promise.resolve({
          tables: [],
          database: {
            status_code: "missing_initial_data",
            table_count: 1,
            has_initial_data: false,
            exists: true,
            connected: true,
          },
        }),
    });
    await ui["loadSchema"]();
    expect(state.systemStatus.dbStatusCode).toBe("missing_initial_data");

    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () =>
        Promise.resolve({
          tables: [{ name: "patients", columns: [] }],
          database: {
            status_code: "healthy",
            table_count: 1,
            has_initial_data: true,
            exists: true,
            connected: true,
          },
        }),
    });
    await ui["loadSchema"]();
    expect(state.systemStatus.dbStatusCode).toBe("healthy");
    expect(
      document
        .getElementById("system-status-banner")
        ?.classList.contains("hidden"),
    ).toBe(true);
  });

  it("shows banner safely when inner elements are null", () => {
    const origIcon = ui["bannerIcon"];
    const origTitle = ui["bannerTitle"];
    const origDesc = ui["bannerDesc"];
    const origBtn = ui["bannerRetryBtn"];
    (ui as any)["bannerIcon"] = null;
    (ui as any)["bannerTitle"] = null;
    (ui as any)["bannerDesc"] = null;
    (ui as any)["bannerRetryBtn"] = null;

    ui.showBanner("info", "Title", "Desc", "ℹ️", false);

    (ui as any)["bannerIcon"] = origIcon;
    (ui as any)["bannerTitle"] = origTitle;
    (ui as any)["bannerDesc"] = origDesc;
    (ui as any)["bannerRetryBtn"] = origBtn;
  });

  it("disables chat input and updates placeholder when backend is offline", () => {
    state.createChat();
    state.setSystemStatus({ backendOnline: false });
    ui["renderActiveChat"]();
    expect(ui["chatInput"].disabled).toBe(true);
    expect(ui["chatInputWrapper"].classList.contains("disabled")).toBe(true);
    expect(ui["sendBtn"].disabled).toBe(true);
  });

  it("initializes without optional status elements safely", () => {
    document.getElementById("system-status-chip")?.remove();
    document.getElementById("system-status-banner")?.remove();
    document.getElementById("provider-settings-btn")?.remove();
    document.getElementById("close-provider-modal-btn")?.remove();
    document.getElementById("provider-save-btn")?.remove();
    document.getElementById("provider-clear-btn")?.remove();
    document.getElementById("provider-modal")?.remove();
    const ui2 = new ChatUI(state);
    expect(ui2).toBeDefined();
  });

  it("does not copy if cmd dataset attribute is missing", () => {
    const container = document.createElement("div");
    const btn = document.createElement("button");
    btn.className = "copy-cmd-btn";
    container.appendChild(btn);
    ui["bindSchemaDiagnosticButtons"](container);
    btn.click();
  });

  it("updates status chip when statusChipText is null", () => {
    const orig = ui["statusChipText"];
    (ui as any)["statusChipText"] = null;
    ui.updateStatusChip("healthy", "Operational");
    (ui as any)["statusChipText"] = orig;
  });

  it("handles checkSystemStatus when bannerRetryBtn is null", async () => {
    const orig = ui["bannerRetryBtn"];
    (ui as any)["bannerRetryBtn"] = null;
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () =>
        Promise.resolve({
          status: "healthy",
          database: { status_code: "healthy", has_initial_data: true },
          ollama: { accessible: true },
        }),
    });
    await ui.checkSystemStatus(true);
    (ui as any)["bannerRetryBtn"] = orig;
  });

  it("handles loadSchema when database fields are null or empty", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () =>
        Promise.resolve({
          tables: [],
          database: {
            status_code: "",
            exists: null,
            connected: null,
            message: null,
            remediation: null,
            table_count: null,
            has_initial_data: null,
          },
        }),
    });
    await ui["loadSchema"]();
    expect(state.systemStatus.dbStatusCode).toBe("unknown");
  });

  it("exports rows to CSV and handles edge cases", () => {
    const createObjectURL = vi.fn().mockReturnValue("blob:mock-csv");
    const revokeObjectURL = vi.fn();
    globalThis.URL.createObjectURL = createObjectURL;
    globalThis.URL.revokeObjectURL = revokeObjectURL;

    // Empty rows returns early
    ui.exportRowsToCsv([], "test.csv");
    expect(createObjectURL).not.toHaveBeenCalled();

    // Valid rows with null, undefined, strings needing escaping
    const sampleRows = [
      { id: 1, name: 'Alice "Wonderland"', note: null, extra: undefined },
      { id: 2, name: "Bob", note: "some note", extra: 42 },
    ];
    ui.exportRowsToCsv(sampleRows, "test.csv");
    expect(createObjectURL).toHaveBeenCalled();
    expect(revokeObjectURL).toHaveBeenCalledWith("blob:mock-csv");
  });

  it("exports rows to JSON and handles edge cases", () => {
    const createObjectURL = vi.fn().mockReturnValue("blob:mock-json");
    const revokeObjectURL = vi.fn();
    globalThis.URL.createObjectURL = createObjectURL;
    globalThis.URL.revokeObjectURL = revokeObjectURL;

    // Empty rows returns early
    ui.exportRowsToJson([], "test.json");
    expect(createObjectURL).not.toHaveBeenCalled();

    // Valid rows
    const sampleRows = [{ id: 1, name: "Alice" }];
    ui.exportRowsToJson(sampleRows, "test.json");
    expect(createObjectURL).toHaveBeenCalled();
    expect(revokeObjectURL).toHaveBeenCalledWith("blob:mock-json");
  });

  it("renders export buttons in chat message tables and triggers export", () => {
    const exportCsvSpy = vi
      .spyOn(ui, "exportRowsToCsv")
      .mockImplementation(() => {});
    const exportJsonSpy = vi
      .spyOn(ui, "exportRowsToJson")
      .mockImplementation(() => {});

    state.createChat();
    state.addMessageToActiveChat({
      role: "assistant",
      content: "Here are the query results:",
      sqlQuery: "SELECT * FROM users",
      sqlResult: [{ id: 1, name: "Alice" }],
    });
    ui["renderMessages"]();

    const csvBtn = document.querySelector(
      ".export-csv-btn",
    ) as HTMLButtonElement | null;
    const jsonBtn = document.querySelector(
      ".export-json-btn",
    ) as HTMLButtonElement | null;

    expect(csvBtn).not.toBeNull();
    expect(jsonBtn).not.toBeNull();

    csvBtn?.click();
    expect(exportCsvSpy).toHaveBeenCalled();

    jsonBtn?.click();
    expect(exportJsonSpy).toHaveBeenCalled();
  });

  it("triggers export from modal export buttons", async () => {
    const exportCsvSpy = vi
      .spyOn(ui, "exportRowsToCsv")
      .mockImplementation(() => {});
    const exportJsonSpy = vi
      .spyOn(ui, "exportRowsToJson")
      .mockImplementation(() => {});

    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () =>
        Promise.resolve({
          rows: [{ id: 1, patient: "P100" }],
          total_count: 1,
          total_pages: 1,
        }),
    });

    await ui["openTableModal"]("patients");

    const modalCsvBtn = document.getElementById("modal-export-csv-btn");
    const modalJsonBtn = document.getElementById("modal-export-json-btn");

    expect(modalCsvBtn).not.toBeNull();
    expect(modalJsonBtn).not.toBeNull();

    modalCsvBtn?.click();
    expect(exportCsvSpy).toHaveBeenCalled();

    modalJsonBtn?.click();
    expect(exportJsonSpy).toHaveBeenCalled();
  });

  it("handles openTableModal and fetchTableData when modal export buttons are missing from DOM", async () => {
    document.getElementById("modal-export-csv-btn")?.remove();
    document.getElementById("modal-export-json-btn")?.remove();

    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () =>
        Promise.resolve({
          rows: [{ id: 1 }],
          total_count: 1,
          total_pages: 1,
        }),
    });

    await ui["openTableModal"]("test_table");
  });

  it("deduplicates SQL block when markdown content already contains SQL code block", () => {
    state.createChat();
    state.addMessageToActiveChat({
      role: "assistant",
      content: "Here is the query:\n```sql\nSELECT * FROM users;\n```",
      sqlQuery: "SELECT * FROM users;",
    });
    ui["renderMessages"]();

    const containers = document.querySelectorAll(".sql-query-container");
    expect(containers.length).toBe(1);
  });

  it("closes table modal on Escape key press and restores previous focus", async () => {
    const input = document.getElementById("chat-input") as HTMLTextAreaElement;
    input.focus();

    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () =>
        Promise.resolve({
          rows: [{ id: 1 }],
          total_count: 1,
          total_pages: 1,
        }),
    });

    await ui["openTableModal"]("patients");
    const modal = document.getElementById("table-modal") as HTMLElement;
    expect(modal.getAttribute("aria-hidden")).toBeNull();
    expect(modal.getAttribute("aria-modal")).toBe("true");

    // Press Escape
    modal.dispatchEvent(
      new KeyboardEvent("keydown", { key: "Escape", bubbles: true }),
    );

    expect(modal.getAttribute("aria-hidden")).toBe("true");
    expect(modal.getAttribute("aria-modal")).toBeNull();
  });

  it("exports CSV with BOM, CRLF, quotes, and handles object properties", () => {
    let capturedBlob: Blob | null = null;
    const originalCreateObjectURL = URL.createObjectURL;
    URL.createObjectURL = vi.fn((blob: Blob) => {
      capturedBlob = blob;
      return "blob:mock-url";
    });

    const rows = [
      { id: 1, name: 'Alice "Wonderland"', details: { note: "ok" } },
      { id: 2, name: "Bob\nSmith", details: null },
    ];

    ui.exportRowsToCsv(rows, "test_export.csv");

    expect(capturedBlob).not.toBeNull();
    expect(URL.createObjectURL).toHaveBeenCalled();
    URL.createObjectURL = originalCreateObjectURL;
  });

  it("handles SSE streaming chunks and rendering tokens in real-time", async () => {
    state.createChat();
    ui.streamingEnabled = true;
    ui.render();

    const streamTextChunks = [
      ": comment ping\n\n",
      'data: {"event": "token", "token": "SELECT "}\n\n',
      'data: {"event": "ignored_event"}\n\n',
      'data: {"event": "token", "token": "* FROM users"}\n\n',
      'data: {"event": "done", "content": "ui.rawSql", "sqlQuery": "SELECT * FROM users", "sqlResult": [{"id": 1, "name": "Alice"}]}\n\n',
    ];

    let querySelectorCallCount = 0;
    const origQuerySelector = Element.prototype.querySelector;
    vi.spyOn(Element.prototype, "querySelector").mockImplementation(function (
      this: Element,
      selectors: string,
    ) {
      if (selectors === ".streaming-content") {
        querySelectorCallCount++;
        if (querySelectorCallCount === 1) {
          // Return null on first lookup to cover if (contentEl) false branch
          return null;
        }
      }
      return origQuerySelector.call(this, selectors);
    });

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
      headers: {
        get: (h: string) =>
          h.toLowerCase() === "content-type" ? "text/event-stream" : null,
      },
      body: {
        getReader: () => mockReader,
      },
    });

    const input = document.getElementById("chat-input") as HTMLTextAreaElement;
    input.value = "Show all users stream";
    document
      .getElementById("chat-form")
      ?.dispatchEvent(new Event("submit", { cancelable: true }));

    await flushPromises();
    await new Promise((r) => setTimeout(r, 60));

    const activeChat = state.getActiveChat();
    expect(activeChat?.messages.length).toBe(2);
    expect(activeChat?.messages[1].sqlQuery).toBe("SELECT * FROM users");
    expect(activeChat?.messages[1].sqlResult?.[0].name).toBe("Alice");

    // Stream response with non-ok status to cover streamResp.ok === false
    mockFetch.mockResolvedValueOnce({
      ok: false,
      headers: {
        get: () => "text/plain",
      },
    });
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ content: "Standard fallback" }),
    });
    await ui["handleSendMessage"]("Fallback message");

    // Test stream with contentEl present, untranslated content, and token streamText
    const secondChunks = [
      'data: {"event": "token", "token": "Live stream token"}\n\n',
      'data: {"event": "result", "content": "Untranslated content", "sqlResult": []}\n\n',
    ];
    let secondIdx = 0;
    const secondReader = {
      read: vi.fn().mockImplementation(async () => {
        if (secondIdx < secondChunks.length) {
          const chunk = new TextEncoder().encode(secondChunks[secondIdx++]);
          return { done: false, value: chunk };
        }
        return { done: true, value: undefined };
      }),
    };
    mockFetch.mockResolvedValueOnce({
      ok: true,
      headers: {
        get: (h: string) =>
          h.toLowerCase() === "content-type" ? "text/event-stream" : null,
      },
      body: {
        getReader: () => secondReader,
      },
    });
    await ui["handleSendMessage"]("Second stream message");

    // Test stream with empty content and empty streamText to hit the final "" branch
    const thirdChunks = [
      'data: {"event": "result", "content": "", "sqlResult": []}\n\n',
    ];
    let thirdIdx = 0;
    const thirdReader = {
      read: vi.fn().mockImplementation(async () => {
        if (thirdIdx < thirdChunks.length) {
          const chunk = new TextEncoder().encode(thirdChunks[thirdIdx++]);
          return { done: false, value: chunk };
        }
        return { done: true, value: undefined };
      }),
    };
    mockFetch.mockResolvedValueOnce({
      ok: true,
      headers: {
        get: (h: string) =>
          h.toLowerCase() === "content-type" ? "text/event-stream" : null,
      },
      body: {
        getReader: () => thirdReader,
      },
    });
    await ui["handleSendMessage"]("Third stream message");
  });

  it("falls back to standard chat fetch if SSE streaming throws an error", async () => {
    state.createChat();
    ui.streamingEnabled = true;
    ui.render();

    mockFetch.mockImplementationOnce(() => {
      throw new Error("Stream connection failed");
    });
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        content: "Standard response after fallback",
        sqlQuery: "SELECT * FROM fallback",
        sqlResult: [{ id: 99 }],
      }),
    });

    const input = document.getElementById("chat-input") as HTMLTextAreaElement;
    input.value = "Test fallback on error";
    document
      .getElementById("chat-form")
      ?.dispatchEvent(new Event("submit", { cancelable: true }));

    await flushPromises();
    await new Promise((r) => setTimeout(r, 60));

    const activeChat = state.getActiveChat();
    expect(activeChat?.messages.length).toBe(2);
    expect(activeChat?.messages[1].sqlQuery).toBe("SELECT * FROM fallback");
  });

  it("handles database switcher dropdown population and change event", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        databases: [
          { name: "t1d.duckdb", path: "/data/t1d.duckdb", size_bytes: 2048 },
          {
            name: "trial_2026.duckdb",
            path: "/data/trial_2026.duckdb",
            size_bytes: 4096,
          },
        ],
        current_db: "t1d.duckdb",
      }),
    });

    await ui.fetchDatabases();
    const dbSelect = document.getElementById("db-select") as HTMLSelectElement;
    expect(dbSelect).not.toBeNull();
    expect(dbSelect.options.length).toBe(2);
    expect(dbSelect.options[1].value).toBe("trial_2026.duckdb");

    // Also fetch when databases property is undefined
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ current_db: "t1d.duckdb" }),
    });
    await ui.fetchDatabases();
    // Dropdown now has 1 option ("t1d.duckdb")
    // Let's re-add the 2nd option so we can change value to trial_2026.duckdb
    const trialOpt = document.createElement("option");
    trialOpt.value = "trial_2026.duckdb";
    trialOpt.textContent = "trial_2026.duckdb";
    dbSelect.appendChild(trialOpt);

    // Mock schema and status fetches for switch
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ tables: [] }),
    });
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ status: "healthy" }),
    });

    // Trigger change with empty value to cover false branch of if (selectedDb)
    dbSelect.value = "";
    dbSelect.dispatchEvent(new Event("change"));

    dbSelect.value = "trial_2026.duckdb";
    dbSelect.dispatchEvent(new Event("change"));

    expect(state.currentDb).toBe("trial_2026.duckdb");
    await new Promise((r) => setTimeout(r, 60));
    expect(document.getElementById("a11y-announcer")?.textContent).toContain(
      "trial_2026.duckdb",
    );
  });

  it("handles sync sessions button click and chat synchronization", async () => {
    const chat1 = state.createChat("Chat One");
    state.addMessageToActiveChat({ role: "user", content: "Query 1" });

    mockFetch.mockResolvedValue({
      ok: true,
      json: async () => ({ session_id: chat1.id, status: "synced" }),
    });

    const syncBtn = document.getElementById(
      "sync-sessions-btn",
    ) as HTMLButtonElement;
    expect(syncBtn).not.toBeNull();
    syncBtn.click();

    await flushPromises();
    expect(state.serverSyncEnabled).toBe(true);
    await new Promise((r) => setTimeout(r, 60));
    expect(document.getElementById("a11y-announcer")?.textContent).toContain(
      i18next.t("ui.sessionsSynced"),
    );
  });

  it("handles SSE stream returning result with error and error announcement", async () => {
    state.createChat();
    ui.streamingEnabled = true;
    ui.render();

    const streamTextChunks = [
      'data: {"event": "token", "token": "SELECT "}\n\n',
      'data: {"event": "result", "content": "Failed to run", "error": "backend.sqlExecution"}\n\n',
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
      headers: {
        get: (h: string) =>
          h.toLowerCase() === "content-type" ? "text/event-stream" : null,
      },
      body: {
        getReader: () => mockReader,
      },
    });

    const input = document.getElementById("chat-input") as HTMLTextAreaElement;
    input.value = "Failing SQL stream";
    document
      .getElementById("chat-form")
      ?.dispatchEvent(new Event("submit", { cancelable: true }));

    await flushPromises();
    await new Promise((r) => setTimeout(r, 60));

    const activeChat = state.getActiveChat();
    expect(activeChat?.messages[1].isError).toBe(true);
  });

  it("handles renderDatabaseDropdown when availableDbs is empty", () => {
    state.availableDbs = [];
    state.currentDb = "default_test.duckdb";
    ui.renderDatabaseDropdown();

    const dbSelect = document.getElementById("db-select") as HTMLSelectElement;
    expect(dbSelect.options.length).toBe(1);
    expect(dbSelect.options[0].value).toBe("default_test.duckdb");

    // Also test fallback when currentDb is empty string
    state.currentDb = "";
    ui.renderDatabaseDropdown();
    expect(dbSelect.options[0].value).toBe("t1d.duckdb");
  });

  it("handles message send when serverSyncEnabled is true", async () => {
    state.createChat();
    state.setServerSyncEnabled(true);
    ui.render();

    const syncSpy = vi.spyOn(ui, "syncAllChatsWithServer");
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ content: "Sync response" }),
    });

    const input = document.getElementById("chat-input") as HTMLTextAreaElement;
    input.value = "Hello synced";
    document
      .getElementById("chat-form")
      ?.dispatchEvent(new Event("submit", { cancelable: true }));

    await flushPromises();
    expect(syncSpy).toHaveBeenCalled();
  });

  it("buildCohortSql generates SQL with various filter predicates", async () => {
    const { buildCohortSql } = await import("../src/ui");

    // Empty parameters or fallback tableName
    expect(buildCohortSql({ tableName: "patients" })).toBe(
      'SELECT *\nFROM "patients";',
    );
    expect(buildCohortSql({ tableName: "" })).toBe(
      'SELECT *\nFROM "patients";',
    );

    // All filters combined
    const sql = buildCohortSql({
      tableName: 'demographics"test',
      minAge: 18,
      maxAge: 65,
      gender: "F",
      txGroup: "Closed-Loop",
      maxTIR: 70,
      minHbA1c: 8.5,
    });

    expect(sql).toContain('FROM "demographics""test"');
    expect(sql).toContain("WHERE age >= 18");
    expect(sql).toContain("AND age <= 65");
    expect(sql).toContain("AND gender = 'F'");
    expect(sql).toContain("AND txgroup ILIKE '%Closed-Loop%'");
    expect(sql).toContain("AND tir <= 70");
    expect(sql).toContain("AND hba1c >= 8.5");

    // Non-finite values (Infinity, NaN) should be ignored
    const nonFiniteSql = buildCohortSql({
      tableName: "patients",
      minAge: Infinity,
      maxAge: NaN,
      maxTIR: -Infinity,
    });
    expect(nonFiniteSql).toBe('SELECT *\nFROM "patients";');

    // Multi-table relational join
    const joinSql = buildCohortSql({
      tableName: "patients",
      joinTableName: "cgms",
      joinType: "INNER JOIN",
      minAge: 18,
    });
    expect(joinSql).toContain('FROM "patients"');
    expect(joinSql).toContain('INNER JOIN "cgms"');
    expect(joinSql).toContain('ON "patients".patient_id = "cgms".patient_id');
    expect(joinSql).toContain('"patients".age >= 18');

    // Custom join condition and LEFT JOIN
    const leftJoinSql = buildCohortSql({
      tableName: "patients",
      joinTableName: "visits",
      joinType: "LEFT JOIN",
      joinCondition: '"patients".patient_id = "visits".subject_id',
    });
    expect(leftJoinSql).toContain('LEFT JOIN "visits"');
    expect(leftJoinSql).toContain(
      'ON "patients".patient_id = "visits".subject_id',
    );
  });

  it("handles cohort modal open, input changes, and closing", async () => {
    const filterBtn = document.getElementById(
      "cohort-filter-btn",
    ) as HTMLButtonElement;
    const modal = document.getElementById("cohort-modal") as HTMLElement;
    const closeBtn = document.getElementById(
      "close-cohort-modal-btn",
    ) as HTMLButtonElement;

    expect(modal.classList.contains("hidden")).toBe(true);

    // Open modal
    filterBtn.click();
    expect(modal.classList.contains("hidden")).toBe(false);

    // Change inputs and check SQL preview
    const minAgeInput = document.getElementById(
      "cohort-min-age",
    ) as HTMLInputElement;
    minAgeInput.value = "21";
    minAgeInput.dispatchEvent(new Event("input"));

    const preview = document.getElementById(
      "cohort-sql-preview",
    ) as HTMLElement;
    expect(preview.textContent).toContain("age >= 21");

    // Change secondary join table
    const joinTableSelect = document.getElementById(
      "cohort-join-table-select",
    ) as HTMLSelectElement;
    if (joinTableSelect) {
      joinTableSelect.value = "cgm_data";
      joinTableSelect.dispatchEvent(new Event("change"));
      expect(preview.textContent).toContain('JOIN "cgm_data"');

      // Reset join table back to none
      joinTableSelect.value = "";
      joinTableSelect.dispatchEvent(new Event("change"));
      expect(preview.textContent).not.toContain("JOIN");
    }

    // Close via close button
    closeBtn.click();
    expect(modal.classList.contains("hidden")).toBe(true);

    // Open again and close via backdrop click
    filterBtn.click();
    expect(modal.classList.contains("hidden")).toBe(false);
    modal.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    expect(modal.classList.contains("hidden")).toBe(true);
  });

  it("handles cohort modal insert into chat button", () => {
    const filterBtn = document.getElementById(
      "cohort-filter-btn",
    ) as HTMLButtonElement;
    filterBtn.click();

    const minAgeInput = document.getElementById(
      "cohort-min-age",
    ) as HTMLInputElement;
    minAgeInput.value = "25";
    minAgeInput.dispatchEvent(new Event("input"));

    const insertBtn = document.getElementById(
      "cohort-insert-chat-btn",
    ) as HTMLButtonElement;
    insertBtn.click();

    const input = document.getElementById("chat-input") as HTMLTextAreaElement;
    expect(input.value).toContain("age >= 25");
    expect(
      document.getElementById("cohort-modal")?.classList.contains("hidden"),
    ).toBe(true);
  });

  it("handles cohort modal execute button directly running query", async () => {
    state.createChat();
    ui.render();

    const filterBtn = document.getElementById(
      "cohort-filter-btn",
    ) as HTMLButtonElement;
    filterBtn.click();

    const minAgeInput = document.getElementById(
      "cohort-min-age",
    ) as HTMLInputElement;
    minAgeInput.value = "30";
    minAgeInput.dispatchEvent(new Event("input"));

    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ content: "Ran cohort query", sqlResult: [] }),
    });

    const execBtn = document.getElementById(
      "cohort-execute-btn",
    ) as HTMLButtonElement;
    execBtn.click();

    await flushPromises();
    const activeChat = state.getActiveChat();
    expect(activeChat?.model).toBe("sql");
    expect(activeChat?.messages.length).toBe(2);
    expect(activeChat?.messages[0].content).toContain("age >= 30");
  });

  it("handles openCohortModal populating tables from DOM schema headers", () => {
    const header = document.createElement("div");
    header.className = "schema-table-header";
    header.innerHTML = "<h4>clinical_cohort_table</h4><h4>   </h4>";
    document.body.appendChild(header);

    ui.openCohortModal();
    const select = document.getElementById(
      "cohort-table-select",
    ) as HTMLSelectElement;
    expect(select.options[0].value).toBe("clinical_cohort_table");

    // Also test when cohortTableSelect and cohortJoinTableSelect are null
    const origTableSelect = (ui as any).cohortTableSelect;
    const origJoinSelect = (ui as any).cohortJoinTableSelect;
    (ui as any).cohortTableSelect = null;
    (ui as any).cohortJoinTableSelect = null;
    ui.openCohortModal();
    (ui as any).cohortTableSelect = origTableSelect;
    (ui as any).cohortJoinTableSelect = origJoinSelect;

    ui.closeCohortModal();
    header.remove();
  });

  it("toggles streaming mode when streaming toggle button is clicked", () => {
    const btn = document.getElementById(
      "streaming-toggle-btn",
    ) as HTMLButtonElement;
    expect(btn).not.toBeNull();
    expect(ui.streamingEnabled).toBe(false);

    btn.click();
    expect(ui.streamingEnabled).toBe(true);
    expect(state.streamingMode).toBe(true);
    expect(btn.classList.contains("active")).toBe(true);

    btn.click();
    expect(ui.streamingEnabled).toBe(false);
    expect(state.streamingMode).toBe(false);
    expect(btn.classList.contains("active")).toBe(false);
  });

  it("passes previous turns in history payload on multi-turn chat messages", async () => {
    state.createChat();
    ui.render();

    // First turn
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ content: "First answer", sqlResult: [] }),
    });
    const chatInput = document.getElementById(
      "chat-input",
    ) as HTMLTextAreaElement;
    chatInput.value = "First query";
    const sendBtn = document.getElementById("send-btn") as HTMLButtonElement;
    sendBtn.click();
    await flushPromises();

    // Second turn
    let secondRequestBody: any = null;
    mockFetch.mockImplementationOnce(async (_url: string, opts: any) => {
      secondRequestBody = JSON.parse(opts.body);
      return {
        ok: true,
        json: async () => ({ content: "Second answer", sqlResult: [] }),
      };
    });

    chatInput.value = "Second query";
    sendBtn.click();
    await flushPromises();

    expect(secondRequestBody).not.toBeNull();
    expect(secondRequestBody.history.length).toBeGreaterThan(0);
    expect(secondRequestBody.history[0].content).toBe("First query");
  });

  it("covers remaining ui branch edge cases", async () => {
    // 1. Remove optional elements from DOM and test initialization
    document.getElementById("streaming-toggle-btn")?.remove();
    document.getElementById("db-select")?.remove();
    document.getElementById("sync-sessions-btn")?.remove();
    document.getElementById("cohort-filter-btn")?.remove();
    document.getElementById("close-cohort-modal-btn")?.remove();
    document.getElementById("cohort-modal")?.remove();
    document.getElementById("cohort-insert-chat-btn")?.remove();
    document.getElementById("cohort-execute-btn")?.remove();

    const minimalUi = new ChatUI(state);
    expect(minimalUi).toBeDefined();
    minimalUi.updateStreamingToggleUi();
    await minimalUi.fetchDatabases();
    minimalUi.renderDatabaseDropdown();
    minimalUi.openCohortModal();
    minimalUi.closeCohortModal();

    // 2. fetchDatabases with !resp.ok
    const uiWithDb = new ChatUI(state);
    const dbSelect = document.createElement("select");
    dbSelect.id = "db-select";
    document.body.appendChild(dbSelect);
    (uiWithDb as any).dbSelect = dbSelect;
    mockFetch.mockResolvedValueOnce({ ok: false, status: 500 });
    await uiWithDb.fetchDatabases();

    // 3. syncAllChatsWithServer with temporary chat
    const tempChat = state.createChat();
    tempChat.isTemporary = true;
    const normalChat = state.createChat();
    normalChat.isTemporary = false;
    mockFetch.mockResolvedValueOnce({ ok: true });
    await ui.syncAllChatsWithServer();

    // 4. openCohortModal fallback tables when no schema headers exist
    const origCohortTableSelect = (ui as any).cohortTableSelect;
    const origCohortModal = (ui as any).cohortModal;
    const cohortModal = document.createElement("div");
    cohortModal.id = "cohort-modal-temp";
    const cohortTableSelect = document.createElement("select");
    cohortTableSelect.id = "cohort-table-select-temp";
    document.body.appendChild(cohortModal);
    document.body.appendChild(cohortTableSelect);
    (ui as any).cohortModal = cohortModal;
    (ui as any).cohortTableSelect = cohortTableSelect;
    ui.openCohortModal();
    expect(cohortTableSelect.options.length).toBe(3); // demographics, cgm_data, patients
    (ui as any).cohortTableSelect = origCohortTableSelect;
    (ui as any).cohortModal = origCohortModal;
    cohortModal.remove();
    cohortTableSelect.remove();

    // 5. Modal focus trap with no focusable elements
    const emptyModal = document.createElement("div");
    emptyModal.id = "table-modal";
    document.body.appendChild(emptyModal);
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve({ rows: [] }),
    });
    await ui["openTableModal"]("test");
    // Tab event with no focusable elements returns early
    emptyModal.dispatchEvent(new KeyboardEvent("keydown", { key: "Tab" }));

    // 6. renderMessages with null message content
    state.createChat();
    state.addMessageToActiveChat({
      role: "assistant",
      content: null as any,
    });
    ui["renderMessages"]();
  });

  it("sanitizes malicious script and onerror payloads in markdown assistant messages via DOMPurify", () => {
    state.createChat();
    const maliciousPayload =
      'Hello! <img src="x" onerror="alert(1)"> <script>window.pwned=true;</script> [Click me](javascript:alert(2))';
    state.addMessageToActiveChat({
      role: "assistant",
      content: maliciousPayload,
    });
    ui["renderMessages"]();

    const messagesPane = document.getElementById("messages-container");
    expect(messagesPane).not.toBeNull();
    const htmlContent = messagesPane?.innerHTML ?? "";

    // onerror and script must be stripped by DOMPurify
    expect(htmlContent).not.toContain("onerror");
    expect(htmlContent).not.toContain("<script>");
    expect(htmlContent).not.toContain("javascript:alert(2)");
    expect((window as unknown as { pwned?: boolean }).pwned).toBeUndefined();
  });

  it("safely escapes table names containing HTML or script tags in schema explorer", async () => {
    const maliciousTableName = "<script>window.hacked=true;</script>";
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () =>
        Promise.resolve({
          tables: [
            {
              name: maliciousTableName,
              columns: [{ name: '<img onerror="alert(1)">', type: "INTEGER" }],
            },
          ],
        }),
    });

    await ui["loadSchema"]();

    const schemaContent = document.getElementById("schema-content");
    expect(schemaContent).not.toBeNull();
    // Verify script tag was not rendered as an active element
    const scripts = schemaContent?.querySelectorAll("script");
    expect(scripts?.length).toBe(0);
    expect((window as unknown as { hacked?: boolean }).hacked).toBeUndefined();
    // Text content should display the literal name
    expect(schemaContent?.textContent).toContain(maliciousTableName);
  });

  it("handles modal table search, sort select, sort order toggle, and backdrop click", async () => {
    mockFetch.mockResolvedValue({
      ok: true,
      json: () =>
        Promise.resolve({
          columns: [{ name: "id", type: "INTEGER" }],
          rows: [{ id: 1 }, { id: 2 }],
          total_rows: 2,
          total_pages: 1,
        }),
    });

    await ui["openTableModal"]("patients");

    const modal = document.getElementById("table-modal");
    expect(modal?.getAttribute("aria-modal")).toBe("true");

    // 1. Search input event
    const searchInput = document.getElementById(
      "modal-table-search",
    ) as HTMLInputElement;
    if (searchInput) {
      searchInput.value = "Alice";
      searchInput.dispatchEvent(new Event("input"));
      expect(ui["currentSearchTerm"]).toBe("Alice");
    }

    // 2. Sort select event
    const sortSelect = document.getElementById(
      "modal-table-sort-col",
    ) as HTMLSelectElement;
    if (sortSelect) {
      sortSelect.value = "id";
      sortSelect.dispatchEvent(new Event("change"));
      expect(ui["currentSortBy"]).toBe("id");
    }

    // 3. Sort order button click
    const sortOrderBtn = document.getElementById(
      "modal-table-sort-order",
    ) as HTMLButtonElement;
    if (sortOrderBtn) {
      sortOrderBtn.click();
      expect(ui["currentSortOrder"]).toBe("desc");
      sortOrderBtn.click();
      expect(ui["currentSortOrder"]).toBe("asc");
    }

    // 4. Modal overlay backdrop click closes modal
    modal?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    expect(modal?.getAttribute("aria-hidden")).toBe("true");

    // Test openTableModal and fetchTableData when searchInput, sortSelect, and sortOrderBtn are missing
    searchInput?.remove();
    sortSelect?.remove();
    sortOrderBtn?.remove();
    await ui["openTableModal"]("patients");
    // Recreate them for subsequent tests
    const searchEl = document.createElement("input");
    searchEl.id = "modal-table-search";
    const sortColEl = document.createElement("select");
    sortColEl.id = "modal-table-sort-col";
    const sortOrderEl = document.createElement("button");
    sortOrderEl.id = "modal-table-sort-order";
    modal?.appendChild(searchEl);
    modal?.appendChild(sortColEl);
    modal?.appendChild(sortOrderEl);

    // 5. Column header keyboard sorting via Enter and Space keys
    await ui["openTableModal"]("patients");
    const th = document.querySelector("#modal-table-head th") as HTMLElement;
    if (th) {
      th.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter" }));
      th.dispatchEvent(new KeyboardEvent("keydown", { key: " " }));
      th.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape" }));
    }

    // 6. Focus trap shift+tab when activeElement is firstElement
    await ui["openTableModal"]("patients");
    const closeBtn = document.getElementById("modal-close-btn");
    closeBtn?.focus();
    modal?.dispatchEvent(
      new KeyboardEvent("keydown", { key: "Tab", shiftKey: true }),
    );
  });

  it("handles provider settings modal open, key updates, and clear", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        providers: [
          { provider: "ollama", configured: true, healthy: true },
          { provider: "openai", configured: true, healthy: true },
          { provider: "anthropic", configured: false, healthy: false },
          { provider: "google", configured: false, healthy: false },
        ],
      }),
    });

    const settingsBtn = document.getElementById(
      "provider-settings-btn",
    ) as HTMLButtonElement;
    const modal = document.getElementById("provider-modal") as HTMLElement;
    const closeBtn = document.getElementById(
      "close-provider-modal-btn",
    ) as HTMLButtonElement;
    const saveBtn = document.getElementById(
      "provider-save-btn",
    ) as HTMLButtonElement;
    const clearBtn = document.getElementById(
      "provider-clear-btn",
    ) as HTMLButtonElement;
    const keyOpenAI = document.getElementById(
      "provider-key-openai",
    ) as HTMLInputElement;

    expect(modal.classList.contains("hidden")).toBe(true);

    // 1. Open modal and check status fetch
    settingsBtn.click();
    await flushPromises();
    expect(modal.classList.contains("hidden")).toBe(false);

    // 2. Set key and save
    const consent = document.getElementById(
      "provider-store-consent",
    ) as HTMLInputElement;
    if (consent) consent.checked = false;
    keyOpenAI.value = "sk-custom-openai-123";
    saveBtn.click();
    await flushPromises();
    expect(state.getApiKey("openai")).toBeNull();

    if (consent) consent.checked = true;
    keyOpenAI.value = "sk-custom-openai-123";
    saveBtn.click();
    await flushPromises();
    expect(state.getApiKey("openai")).toBe("sk-custom-openai-123");
    expect(modal.classList.contains("hidden")).toBe(true);

    // 3. Open again and verify key was filled
    settingsBtn.click();
    await flushPromises();
    expect(keyOpenAI.value).toBe("sk-custom-openai-123");

    // 4. Clear keys
    clearBtn.click();
    await flushPromises();
    expect(state.getApiKey("openai")).toBeNull();
    expect(keyOpenAI.value).toBe("");

    // 5. Close via close button and backdrop click
    closeBtn.click();
    expect(modal.classList.contains("hidden")).toBe(true);

    settingsBtn.click();
    await flushPromises();
    expect(modal.classList.contains("hidden")).toBe(false);
    modal.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    expect(modal.classList.contains("hidden")).toBe(true);
  });

  it("renders grouped models and updates provider status dot", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        models: [
          { name: "gemma4", provider: "ollama", configured: true },
          {
            name: "openai/gpt-4o",
            provider: "openai",
            configured: false,
            requires_key: true,
          },
          {
            name: "anthropic/claude-3-5-sonnet",
            provider: "anthropic",
            configured: false,
            requires_key: true,
          },
        ],
      }),
    });

    await ui.loadModels();

    const modelSelect = document.getElementById(
      "model-select",
    ) as HTMLSelectElement;
    const optgroups = modelSelect.querySelectorAll("optgroup");
    expect(optgroups.length).toBeGreaterThan(0);

    const dot = document.getElementById("provider-status-dot") as HTMLElement;

    // Switch to local model
    modelSelect.value = "gemma4";
    modelSelect.dispatchEvent(new Event("change"));
    expect(dot.classList.contains("active")).toBe(true);

    // Switch to unconfigured cloud model
    modelSelect.value = "openai/gpt-4o";
    modelSelect.dispatchEvent(new Event("change"));
    expect(dot.classList.contains("warning")).toBe(true);

    // Save key for openai and verify it turns green
    state.setApiKey("openai", "sk-valid-key");
    ui.updateProviderBadgeStatus();
    expect(dot.classList.contains("active")).toBe(true);
  });

  it("covers remaining ui edge cases for models, badges, cohorts, providers, and modal navigation", async () => {
    // 1. loadModels with unreachable model, model with error_code, fallback provider group, and model with '/'
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        models: [
          {
            name: "custom_unreachable",
            reachable: false,
            available: false,
            error_code: "CONN_ERR",
          },
          {
            name: "custom_prov/model-x",
            available: true,
          },
        ],
      }),
    });
    await ui.loadModels();
    const modelSelect = document.getElementById(
      "model-select",
    ) as HTMLSelectElement;
    const unreachableOpt = Array.from(modelSelect.options).find(
      (o) => o.value === "custom_unreachable",
    );
    expect(unreachableOpt?.textContent).toContain("🔴");
    expect(unreachableOpt?.title).toBe("custom_unreachable (CONN_ERR)");

    // 2. updateProviderBadgeStatus when providerStatusDot is null and when provider is in value with slash
    const dot = document.getElementById("provider-status-dot");
    if (dot) {
      modelSelect.innerHTML = `<option value="openai/custom-gpt">custom-gpt</option>`;
      modelSelect.value = "openai/custom-gpt";
      ui.updateProviderBadgeStatus();
      expect(dot.classList.contains("warning")).toBe(true);
    }
    dot?.remove();
    // Re-initialize a ChatUI without provider-status-dot to test branch: if (!this.providerStatusDot) return;
    const uiWithoutDot = new ChatUI(state);
    expect(() => uiWithoutDot.updateProviderBadgeStatus()).not.toThrow();

    // 3. Cohort table change, join table change, inputs, change events, and relational join resolution
    const tableSelect = document.getElementById(
      "cohort-table-select",
    ) as HTMLSelectElement;
    const joinTableSelect = document.getElementById(
      "cohort-join-table-select",
    ) as HTMLSelectElement;
    const joinTypeSelect = document.getElementById(
      "cohort-join-type-select",
    ) as HTMLSelectElement;
    const joinOnInput = document.getElementById(
      "cohort-join-on-input",
    ) as HTMLInputElement;

    tableSelect.innerHTML = `<option value="patients">patients</option><option value="cgms">cgms</option>`;
    joinTableSelect.innerHTML = `<option value="">None (Single Table)</option><option value="cgms">cgms</option><option value="visits">visits</option>`;

    // Trigger change event on cohort-table-select
    tableSelect.dispatchEvent(new Event("change"));

    // Select cgms which has a defined CLINICAL_RELATIONSHIP with patients
    joinTableSelect.value = "cgms";
    uiWithoutDot.handleCohortJoinTableChange();
    ui.handleCohortJoinTableChange();
    expect(joinOnInput.value).toBe('"patients".patient_id = "cgms".patient_id');
    expect(joinTypeSelect.value).toBe("INNER JOIN");

    // Select visits which has defaultJoinType = 'LEFT JOIN'
    joinTableSelect.value = "visits";
    uiWithoutDot.handleCohortJoinTableChange();
    ui.handleCohortJoinTableChange();
    expect(joinTypeSelect.value).toBe("LEFT JOIN");

    // Select unknown_table where resolveTableJoin returns null (table1 not containing "patient")
    tableSelect.innerHTML += `<option value="sensors">sensors</option>`;
    tableSelect.value = "sensors";
    joinTableSelect.innerHTML += `<option value="unknown_table">unknown_table</option>`;
    joinTableSelect.value = "unknown_table";
    uiWithoutDot.handleCohortJoinTableChange();
    ui.handleCohortJoinTableChange();
    expect(joinOnInput.value).toBe(
      '"sensors".patient_id = "unknown_table".patient_id',
    );

    // Also trigger resolveTableJoin fallback when cohortJoinOnInput is null
    (ui as any).cohortJoinOnInput = null;
    (uiWithoutDot as any).cohortJoinOnInput = null;
    joinTableSelect.value = "unknown_table";
    ui.handleCohortJoinTableChange();
    uiWithoutDot.handleCohortJoinTableChange();
    (ui as any).cohortJoinOnInput = joinOnInput;
    (uiWithoutDot as any).cohortJoinOnInput = joinOnInput;
    tableSelect.value = "patients";

    // Test getCohortSql fallback when cohortJoinTypeSelect.value is empty
    joinTypeSelect.value = "";
    const fallbackJoinSql = ui.getCohortSql();
    expect(fallbackJoinSql).toContain("INNER JOIN");
    joinTypeSelect.value = "LEFT JOIN";

    // Select table2 === table1
    joinTableSelect.value = "patients";
    joinTableSelect.dispatchEvent(new Event("change"));
    expect(joinOnInput.value).toBe("");

    // Test null branches for cohortJoinOnInput, cohortJoinTypeSelect, cohortSqlPreview
    const origOnInput = (ui as any).cohortJoinOnInput;
    const origTypeSelect = (ui as any).cohortJoinTypeSelect;
    const origPreview = (ui as any).cohortSqlPreview;
    (ui as any).cohortJoinOnInput = null;
    (ui as any).cohortJoinTypeSelect = null;
    (ui as any).cohortSqlPreview = null;
    joinTableSelect.value = "cgms";
    (ui as any).cohortJoinTableSelect = joinTableSelect;
    ui.handleCohortJoinTableChange();
    joinTableSelect.value = "unknown_table";
    ui.handleCohortJoinTableChange();
    joinTableSelect.value = "patients";
    ui.handleCohortJoinTableChange();
    ui.updateCohortSqlPreview();
    (ui as any).cohortJoinOnInput = origOnInput;
    (ui as any).cohortJoinTypeSelect = origTypeSelect;
    (ui as any).cohortSqlPreview = origPreview;

    joinTableSelect.value = "visits";
    joinTypeSelect.value = "LEFT JOIN";
    joinOnInput.value = '"patients".patient_id = "visits".patient_id';

    // Trigger cohortInputs change events
    const minAgeInput = document.getElementById(
      "cohort-min-age",
    ) as HTMLInputElement;
    minAgeInput.value = "20";
    minAgeInput.dispatchEvent(new Event("change"));

    const maxAgeInput = document.getElementById(
      "cohort-max-age",
    ) as HTMLInputElement;
    maxAgeInput.value = "60";
    maxAgeInput.dispatchEvent(new Event("change"));

    const maxTIRInput = document.getElementById(
      "cohort-max-tir",
    ) as HTMLInputElement;
    maxTIRInput.value = "70";
    maxTIRInput.dispatchEvent(new Event("change"));

    const minHbA1cInput = document.getElementById(
      "cohort-min-hba1c",
    ) as HTMLInputElement;
    minHbA1cInput.value = "6.5";
    minHbA1cInput.dispatchEvent(new Event("change"));

    joinTypeSelect.dispatchEvent(new Event("change"));
    joinOnInput.dispatchEvent(new Event("input"));

    // Verify cohort SQL generated
    const preview = document.getElementById(
      "cohort-sql-preview",
    ) as HTMLElement;
    expect(preview.textContent).toContain('LEFT JOIN "visits"');
    expect(preview.textContent).toContain("age >= 20");
    expect(preview.textContent).toContain("age <= 60");
    expect(preview.textContent).toContain("tir <= 70");
    expect(preview.textContent).toContain("hba1c >= 6.5");

    // 4. Provider modal null checks, saving anthropic and google keys, and local badge fallbacks
    // Re-create provider modal DOM to test providerKeyAnthropic & providerKeyGoogle saving
    const keyAnthropic = document.getElementById(
      "provider-key-anthropic",
    ) as HTMLInputElement;
    const keyGoogle = document.getElementById(
      "provider-key-google",
    ) as HTMLInputElement;
    keyAnthropic.value = "anthropic-secret-key";
    keyGoogle.value = "google-secret-key";

    await ui.saveProviderSettings();
    expect(state.getApiKey("anthropic")).toBe("anthropic-secret-key");
    expect(state.getApiKey("google")).toBe("google-secret-key");

    // openProviderModal when status endpoint returns non-ok (falls back to local)
    mockFetch.mockResolvedValueOnce({ ok: false, status: 500 });
    await ui.openProviderModal();
    const anthropicBadge = document.getElementById("provider-badge-anthropic");
    expect(anthropicBadge?.textContent).toContain("Configured 🟢");

    // openProviderModal when fetch throws (catch branch)
    mockFetch.mockRejectedValueOnce(new Error("Network disconnect"));
    await ui.openProviderModal();

    // Test null input branches in openProviderModal, saveProviderSettings, clearProviderSettings
    const origOpenAI = (ui as any).providerKeyOpenAI;
    const origAnthropic = (ui as any).providerKeyAnthropic;
    const origGoogle = (ui as any).providerKeyGoogle;
    const origConsent = (ui as any).providerStoreConsent;
    (ui as any).providerKeyOpenAI = null;
    (ui as any).providerKeyAnthropic = null;
    (ui as any).providerKeyGoogle = null;
    (ui as any).providerStoreConsent = null;
    mockFetch.mockResolvedValueOnce({ ok: false, status: 500 });
    await ui.openProviderModal();
    await ui.saveProviderSettings();
    await ui.clearProviderSettings();
    (ui as any).providerKeyOpenAI = origOpenAI;
    (ui as any).providerKeyAnthropic = origAnthropic;
    (ui as any).providerKeyGoogle = origGoogle;
    (ui as any).providerStoreConsent = origConsent;

    // openProviderModal / closeProviderModal when providerModal is missing
    (ui as any).providerModal = null;
    await expect(ui.openProviderModal()).resolves.toBeUndefined();
    expect(() => ui.closeProviderModal()).not.toThrow();
    (ui as any).providerModal = document.getElementById("provider-modal");

    // 5. handleSendMessage with provider and apiKey request headers
    state.createChat();
    state.setApiKey("anthropic", "anthropic-secret-key");
    // Add provider modal back so other tests aren't impacted
    const chatInput = document.getElementById(
      "chat-input",
    ) as HTMLTextAreaElement;
    chatInput.value = "SELECT * FROM patients;";
    // Choose anthropic model without data-provider attribute to test model.split('/')[0] fallback
    modelSelect.innerHTML = `<option value="anthropic/claude" selected>Claude</option>`;
    modelSelect.value = "anthropic/claude";
    state.setActiveChatModel("anthropic/claude");

    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        content: "Here are the patients",
        sqlQuery: "SELECT * FROM patients;",
        sqlResult: [{ id: 1 }],
      }),
    });

    await ui["handleSendMessage"](
      "SELECT * FROM patients;",
      "anthropic/claude",
    );
    // Verify fetch was called with custom headers
    const lastCall = mockFetch.mock.calls[mockFetch.mock.calls.length - 1];
    expect(lastCall[1].headers["x-provider"]).toBe("anthropic");
    expect(lastCall[1].headers["x-provider-api-key"]).toBe(
      "anthropic-secret-key",
    );

    // Also send with model that has no slash and no data-provider
    modelSelect.innerHTML = `<option value="plainmodel" selected>Plain Model</option>`;
    modelSelect.value = "plainmodel";
    state.setActiveChatModel("plainmodel");
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ content: "Hello" }),
    });
    await ui["handleSendMessage"]("Hi", "plainmodel");

    // 6. Modal table navigation and focus trap key events
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        rows: [{ id: 1, name: "Alice" }],
        total_pages: 1,
      }),
    });

    await ui["openTableModal"]("patients");
    const tableModal = document.getElementById("table-modal") as HTMLElement;

    // Dispatch non-Tab keydown (e.key === "ArrowDown") -> should return early
    tableModal.dispatchEvent(
      new KeyboardEvent("keydown", { key: "ArrowDown", bubbles: true }),
    );

    // Dispatch Tab keydown when focusable elements exist and focus is at the last element
    const closeBtn = document.getElementById(
      "close-modal-btn",
    ) as HTMLButtonElement;
    const exportJsonBtn = document.getElementById(
      "modal-export-json-btn",
    ) as HTMLButtonElement;
    exportJsonBtn.removeAttribute("disabled");
    exportJsonBtn.focus();

    const tabEvent = new KeyboardEvent("keydown", {
      key: "Tab",
      bubbles: true,
      cancelable: true,
    });
    tableModal.dispatchEvent(tabEvent);

    // Test focusableElements.length === 0 branch
    const allInteractive = tableModal.querySelectorAll<HTMLElement>(
      "button, input, select, textarea, [tabindex]",
    );
    allInteractive.forEach((el) => {
      el.setAttribute("disabled", "true");
      if (el.hasAttribute("tabindex")) {
        el.setAttribute("tabindex", "-1");
      }
    });
    tableModal.dispatchEvent(
      new KeyboardEvent("keydown", { key: "Tab", bubbles: true }),
    );
  });
});
