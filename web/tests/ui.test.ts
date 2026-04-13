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
      </nav>
      <main id="main-pane">
        <button id="open-sidebar-btn">Open</button>
        <button id="theme-toggle-btn">Theme</button>
        <select id="model-select">
          <option value="gemma4">Gemma</option>
          <option value="sql">SQL</option>
        </select>
        <div id="messages-container"></div>
        <form id="chat-form">
          <input id="chat-input" type="text" />
          <button id="send-btn" type="submit">Send</button>
        </form>
      </main>
      <div id="overlay"></div>
    `;

    mockFetch = vi.fn();
    globalThis.fetch = mockFetch as any;

    state = new ChatState();
    ui = new ChatUI(state);
  });

  afterEach(() => {
    document.body.innerHTML = "";
    vi.restoreAllMocks();
  });

  it("renders initial empty state", () => {
    const container = document.getElementById("messages-container");
    expect(container?.innerHTML).toContain("Select or create a chat to begin.");
    expect(
      (document.getElementById("chat-input") as HTMLInputElement).disabled,
    ).toBe(true);
  });

  it("creates new chat and renders it", () => {
    document.getElementById("new-chat-btn")?.click();
    expect(state.chats.length).toBe(1);

    const list = document.getElementById("chat-list");
    expect(list?.children.length).toBe(1);
    expect(list?.innerHTML).toContain("Chat #1");
  });

  it("switches active chat on click", () => {
    state.createChat();
    state.createChat();
    ui.render();

    const items = document.querySelectorAll(".chat-item-title");
    (items[0] as HTMLElement).click();

    expect(state.activeChatId).toBe(state.chats[0].id);
    expect(document.querySelector(".chat-item.active")?.textContent).toContain(
      "Chat #1",
    );
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

    const input = document.getElementById("chat-input") as HTMLInputElement;
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
  });

  it("handles fetch HTTP error", async () => {
    state.createChat();
    ui.render();

    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 500,
    });

    const input = document.getElementById("chat-input") as HTMLInputElement;
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

    const input = document.getElementById("chat-input") as HTMLInputElement;
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

    const input = document.getElementById("chat-input") as HTMLInputElement;
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

    const input = document.getElementById("chat-input") as HTMLInputElement;
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

    const input = document.getElementById("chat-input") as HTMLInputElement;
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

    const input = document.getElementById("chat-input") as HTMLInputElement;
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
    expect(mockFetch).not.toHaveBeenCalled();
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

    // Click outside
    document.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    expect(menu.classList.contains("show")).toBe(false);
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

    const btns = document.querySelectorAll(".dropdown-item");
    const deleteBtn = btns[2] as HTMLElement;
    deleteBtn.click();

    expect(confirmMock).toHaveBeenCalled();
    expect(state.chats.length).toBe(1); // Temporary Chat!
    expect(state.chats[0].title).toBe("Temporary chat");
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
    (ui as any).renderMessages();
    const container = document.getElementById("messages-container");
    expect(container?.innerHTML).toContain("Select or create a chat to begin.");
  });

  it("updates active chat on hashchange", () => {
    state.createChat();
    const chat2 = state.createChat();
    ui.render();
    window.location.hash = `#${chat2.id}`;
    window.dispatchEvent(new Event("hashchange"));
    expect(state.activeChatId).toBe(chat2.id);
  });

  it("populates chat input and sends message when clicking a chip", async () => {
    state.createChat();
    ui.render();

    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        content: "Chip response",
        sqlResult: null,
        error: null,
      }),
    });

    const chip = document.querySelector(".chip") as HTMLButtonElement;
    chip.click();

    await flushPromises();

    expect(state.getActiveChat()?.messages.length).toBe(2);
    expect(state.getActiveChat()?.messages[0].content).toBe(chip.textContent);
    expect(state.getActiveChat()?.messages[1].content).toBe("Chip response");
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

  it("toggles dark mode on theme button click", () => {
    state.createChat();
    ui.render();

    const themeBtn = document.getElementById(
      "theme-toggle-btn",
    ) as HTMLButtonElement;
    expect(document.body.classList.contains("dark-mode")).toBe(false);

    themeBtn.click();
    expect(document.body.classList.contains("dark-mode")).toBe(true);

    themeBtn.click();
    expect(document.body.classList.contains("dark-mode")).toBe(false);
  });
});
