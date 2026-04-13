/**
 * @file ui.ts
 * DOM manipulation and event binding for the Chat UI.
 */

import { ChatState } from "./state";

/**
 * UI controller class that binds the state to the DOM elements.
 */
export class ChatUI {
  private state: ChatState;

  // DOM Elements
  private chatListEl: HTMLElement;
  private newChatBtn: HTMLButtonElement;
  private messagesContainer: HTMLElement;
  private chatForm: HTMLFormElement;
  private chatInput: HTMLInputElement;
  private sendBtn: HTMLButtonElement;
  private modelSelect: HTMLSelectElement;
  private openSidebarBtn: HTMLButtonElement;
  private closeSidebarBtn: HTMLButtonElement;
  private themeToggleBtn: HTMLButtonElement;
  private overlay: HTMLElement;

  /**
   * Initializes the ChatUI.
   * @param {ChatState} state The global state object.
   */
  constructor(state: ChatState) {
    this.state = state;

    // Bind DOM elements
    this.chatListEl = document.getElementById("chat-list") as HTMLElement;
    this.newChatBtn = document.getElementById(
      "new-chat-btn",
    ) as HTMLButtonElement;
    this.messagesContainer = document.getElementById(
      "messages-container",
    ) as HTMLElement;
    this.chatForm = document.getElementById("chat-form") as HTMLFormElement;
    this.chatInput = document.getElementById("chat-input") as HTMLInputElement;
    this.sendBtn = document.getElementById("send-btn") as HTMLButtonElement;
    this.modelSelect = document.getElementById(
      "model-select",
    ) as HTMLSelectElement;

    this.openSidebarBtn = document.getElementById(
      "open-sidebar-btn",
    ) as HTMLButtonElement;
    this.closeSidebarBtn = document.getElementById(
      "close-sidebar-btn",
    ) as HTMLButtonElement;
    this.themeToggleBtn = document.getElementById(
      "theme-toggle-btn",
    ) as HTMLButtonElement;
    this.overlay = document.getElementById("overlay") as HTMLElement;

    this.bindEvents();
    this.render();

    // Hash routing
    window.addEventListener("hashchange", () => {
      const hashId = window.location.hash.slice(1);
      if (hashId && hashId !== this.state.activeChatId) {
        this.state.setActiveChat(hashId);
        this.render();
      }
    });

    if (window.location.hash) {
      const hashId = window.location.hash.slice(1);
      this.state.setActiveChat(hashId);
      this.render();
    }
  }

  /**
   * Binds global and static DOM events.
   */
  private bindEvents(): void {
    this.newChatBtn.addEventListener("click", () => {
      this.state.createChat();
      this.render();
      this.closeMobileSidebar();
    });

    this.chatForm.addEventListener("submit", (e) => {
      e.preventDefault();
      this.handleSendMessage();
    });

    this.modelSelect.addEventListener("change", () => {
      this.state.setActiveChatModel(this.modelSelect.value);
    });

    this.themeToggleBtn.addEventListener("click", () => {
      document.body.classList.toggle("dark-mode");
    });

    // Mobile sidebar toggles
    this.openSidebarBtn.addEventListener("click", () => {
      document.body.classList.add("sidebar-open");
    });

    const closeSidebar = () => {
      this.closeMobileSidebar();
    };

    this.closeSidebarBtn.addEventListener("click", closeSidebar);
    this.overlay.addEventListener("click", closeSidebar);

    // Global click listener to close dropdowns
    document.addEventListener("click", (e) => {
      const target = e.target as HTMLElement;
      if (target && typeof target.closest === "function") {
        if (!target.closest(".dropdown-container")) {
          this.closeAllDropdowns();
        }
      } else {
        // e.g. target is Document
        this.closeAllDropdowns();
      }
    });
  }

  /**
   * Closes the mobile sidebar if it's open.
   */
  private closeMobileSidebar(): void {
    document.body.classList.remove("sidebar-open");
  }

  /**
   * Handles sending a user message.
   */
  private async handleSendMessage(): Promise<void> {
    const content = this.chatInput.value.trim();
    if (!content) return;

    this.state.addMessageToActiveChat({ role: "user", content });
    this.chatInput.value = "";

    this.renderSidebar();
    this.renderMessages();
    this.scrollToBottom();

    const model = this.modelSelect.value;
    const dbPath = "t1d.duckdb"; // Backend database path

    try {
      const response = await fetch("http://localhost:8000/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: content, model, db_path: dbPath }),
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      const data = await response.json();

      let assistantMsg = data.content;
      if (data.error) {
        assistantMsg += `\nError details: ${data.error}`;
      }

      this.state.addMessageToActiveChat({
        role: "assistant",
        content: assistantMsg,
        sqlResult: data.sqlResult,
        sqlQuery: data.sqlQuery,
      });
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : String(err);
      this.state.addMessageToActiveChat({
        role: "assistant",
        content: `Error communicating with backend API: ${errorMsg}`,
      });
    }

    this.renderSidebar();
    this.renderMessages();
    this.scrollToBottom();
  }

  /**
   * Scrolls the message container to the bottom.
   */
  private scrollToBottom(): void {
    this.messagesContainer.scrollTop = this.messagesContainer.scrollHeight;
  }

  /**
   * Closes all open dropdown menus.
   */
  private closeAllDropdowns(): void {
    document.querySelectorAll(".dropdown-menu.show").forEach((menu) => {
      menu.classList.remove("show");
    });
  }

  /**
   * Main render function that updates the entire UI.
   */
  public render(): void {
    if (
      this.state.activeChatId &&
      window.location.hash !== `#${this.state.activeChatId}`
    ) {
      window.history.replaceState(null, "", `#${this.state.activeChatId}`);
    }
    this.renderSidebar();
    this.renderActiveChat();
  }

  /**
   * Renders the sidebar chat list.
   */
  private renderSidebar(): void {
    this.chatListEl.innerHTML = "";

    this.state.chats.forEach((chat) => {
      const li = document.createElement("li");
      li.className = `chat-item ${chat.id === this.state.activeChatId ? "active" : ""}`;

      const titleSpan = document.createElement("span");
      titleSpan.className = "chat-item-title";
      titleSpan.textContent = chat.title;
      titleSpan.addEventListener("click", () => {
        this.state.setActiveChat(chat.id);
        this.render();
        this.closeMobileSidebar();
      });

      const dropdownContainer = document.createElement("div");
      dropdownContainer.className = "dropdown-container";

      const dropdownBtn = document.createElement("button");
      dropdownBtn.className = "dropdown-btn";
      dropdownBtn.innerHTML = "&#8942;"; // 3 vertical dots

      const dropdownMenu = document.createElement("div");
      dropdownMenu.className = "dropdown-menu";

      const renameBtn = document.createElement("button");
      renameBtn.className = "dropdown-item";
      renameBtn.textContent = "Rename";
      renameBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        this.closeAllDropdowns();
        const newTitle = prompt("Enter new chat title:", chat.title);
        if (newTitle !== null) {
          this.state.renameChat(chat.id, newTitle);
          this.renderSidebar();
        }
      });

      const duplicateBtn = document.createElement("button");
      duplicateBtn.className = "dropdown-item";
      duplicateBtn.textContent = "Duplicate";
      duplicateBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        this.closeAllDropdowns();
        this.state.duplicateChat(chat.id);
        this.render();
      });

      const deleteBtn = document.createElement("button");
      deleteBtn.className = "dropdown-item danger";
      deleteBtn.textContent = "Delete";
      deleteBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        this.closeAllDropdowns();
        if (confirm(`Are you sure you want to delete "${chat.title}"?`)) {
          this.state.deleteChat(chat.id);
          this.render();
        }
      });

      dropdownBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        const isShowing = dropdownMenu.classList.contains("show");
        this.closeAllDropdowns();
        if (!isShowing) {
          dropdownMenu.classList.add("show");
        }
      });

      dropdownMenu.appendChild(renameBtn);
      dropdownMenu.appendChild(duplicateBtn);
      dropdownMenu.appendChild(deleteBtn);

      dropdownContainer.appendChild(dropdownBtn);
      dropdownContainer.appendChild(dropdownMenu);

      li.appendChild(titleSpan);
      li.appendChild(dropdownContainer);

      this.chatListEl.appendChild(li);
    });
  }

  /**
   * Renders the active chat in the main pane.
   */
  private renderActiveChat(): void {
    const activeChat = this.state.getActiveChat();

    if (!activeChat) {
      this.messagesContainer.innerHTML =
        '<div class="empty-state">Select or create a chat to begin.</div>';
      this.chatInput.disabled = true;
      this.sendBtn.disabled = true;
      this.modelSelect.disabled = true;
      return;
    }

    this.chatInput.disabled = false;
    this.sendBtn.disabled = false;
    this.modelSelect.disabled = false;
    this.modelSelect.value = activeChat.model;

    this.renderMessages();
  }

  /**
   * Renders only the messages of the active chat.
   */
  private renderMessages(): void {
    const activeChat = this.state.getActiveChat();
    if (!activeChat) return;

    this.messagesContainer.innerHTML = "";

    if (activeChat.messages.length === 0) {
      const emptyStateDiv = document.createElement("div");
      emptyStateDiv.className = "empty-state";
      emptyStateDiv.innerHTML =
        '<p style="margin-bottom: 1rem;">No messages yet. Send a message to start, or try an example query:</p>';

      const chipsContainer = document.createElement("div");
      chipsContainer.className = "chips-container";

      const exampleQueries = [
        "What is the average HbA1c across all participants in the PEDAP dataset?",
        "Show me the average Time in Range (TIR) for the AIDE T1D study participants.",
        "SELECT study, AVG(time_in_range) as avg_tir FROM cgm_metrics GROUP BY study;",
        "SELECT age_group, AVG(hba1c) FROM demographics JOIN clinical_results USING(pt_id) WHERE study = 'T1DEXIP' GROUP BY age_group;",
        "How did exercise impact average glucose levels in the T1DexiP study?",
      ];

      exampleQueries.forEach((query) => {
        const chip = document.createElement("button");
        chip.className = "chip";
        chip.textContent = query;
        chip.addEventListener("click", () => {
          this.chatInput.value = query;
          this.handleSendMessage();
        });
        chipsContainer.appendChild(chip);
      });

      emptyStateDiv.appendChild(chipsContainer);
      this.messagesContainer.appendChild(emptyStateDiv);
      return;
    }

    activeChat.messages.forEach((msg) => {
      const msgDiv = document.createElement("div");
      msgDiv.className = `message ${msg.role}`;

      const contentDiv = document.createElement("div");
      contentDiv.textContent = msg.content;
      msgDiv.appendChild(contentDiv);

      if (msg.sqlQuery) {
        const queryContainer = document.createElement("div");
        queryContainer.className = "sql-query-container";
        const codeBlock = document.createElement("pre");
        codeBlock.className = "sql-query";
        codeBlock.textContent = msg.sqlQuery;
        queryContainer.appendChild(codeBlock);
        msgDiv.appendChild(queryContainer);
      }

      if (msg.sqlResult) {
        if (msg.sqlResult.length > 0) {
          const tableContainer = document.createElement("div");
          tableContainer.className = "sql-table-container";

          const table = document.createElement("table");
          table.className = "sql-table";

          const thead = document.createElement("thead");
          const headerRow = document.createElement("tr");
          const columns = Object.keys(msg.sqlResult[0]);

          columns.forEach((col) => {
            const th = document.createElement("th");
            th.textContent = col;
            headerRow.appendChild(th);
          });
          thead.appendChild(headerRow);
          table.appendChild(thead);

          const tbody = document.createElement("tbody");
          msg.sqlResult.forEach((row) => {
            const tr = document.createElement("tr");
            columns.forEach((col) => {
              const td = document.createElement("td");
              const cellValue = row[col];
              td.textContent =
                cellValue === null || cellValue === undefined
                  ? "NULL"
                  : String(cellValue);
              tr.appendChild(td);
            });
            tbody.appendChild(tr);
          });
          table.appendChild(tbody);

          tableContainer.appendChild(table);
          msgDiv.appendChild(tableContainer);
        } else {
          const emptyDiv = document.createElement("div");
          emptyDiv.className = "sql-empty";
          emptyDiv.textContent = "No rows returned";
          msgDiv.appendChild(emptyDiv);
        }
      }

      this.messagesContainer.appendChild(msgDiv);
    });
  }
}
