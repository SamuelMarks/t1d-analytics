import { describe, it, expect, beforeEach, vi, afterEach } from "vitest";
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
        <button id="prev-page-btn"></button>
        <button id="next-page-btn"></button>
        <div id="table-loading"></div>
        <table id="modal-table">
          <caption class="sr-only" id="modal-table-caption">Table Data</caption>
          <thead id="modal-table-head"></thead>
          <tbody id="modal-table-body"></tbody>
        </table>
        <span id="page-indicator"></span>
      </div>
      <main id="main-pane">
        <button id="open-sidebar-btn">Open</button>
        <button id="theme-toggle-btn">Theme</button>
        <select id="model-select">
          <option value="gemma4">Gemma</option>
          <option value="sql">SQL</option>
        </select>
        <select id="lang-select">
          <option value="en">English</option>
          <option value="ja">Japanese</option>
        </select>
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

    mockFetch = vi.fn();
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
    mockFetch.mockRejectedValueOnce(new Error("Network Error"));
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

    // Modal Overlay click
    modal?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    expect(modal?.getAttribute("aria-hidden")).toBe("true");
  });

  it("handles table fetch empty rows returning to previous page", async () => {
    ui["currentPage"] = 2;
    ui["currentTable"] = "test_table";
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve({ rows: [] }),
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

    // We fetch 1 row, so next and prev buttons are disabled.
    // The only focusable element is closeBtn.
    // Tab on closeBtn should loop back to closeBtn.
    closeBtn.focus();
    modal.dispatchEvent(new KeyboardEvent("keydown", { key: "Tab" }));
    expect(document.activeElement).toBe(closeBtn);

    modal.dispatchEvent(
      new KeyboardEvent("keydown", { key: "Tab", shiftKey: true }),
    );
    expect(document.activeElement).toBe(closeBtn);

    // Unrelated keys do nothing
    modal.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape" }));
    expect(document.activeElement).toBe(closeBtn);

    // Clean up
    closeBtn.click();
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
      json: () => Promise.resolve({ detail: "unmapped|some data" }),
    });
    ui["currentPage"] = 1;
    ui["currentTable"] = "test_table";
    await ui["fetchTableData"]();
    expect(document.getElementById("modal-table-body")?.innerHTML).toContain(
      "unmapped|some data",
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
        Promise.resolve({ detail: "backend.serverError|something broke" }),
    });
    ui["currentPage"] = 1;
    ui["currentTable"] = "test_table";
    await ui["fetchTableData"]();
    expect(document.getElementById("modal-table-body")?.innerHTML).toContain(
      "Internal server error: something broke",
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
});
