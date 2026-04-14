/**
 * @file ui.ts
 * DOM manipulation and event binding for the Chat UI.
 */

import hljs from "highlight.js/lib/core";
import sql from "highlight.js/lib/languages/sql";
import "highlight.js/styles/github-dark.css";
import { ChatState } from "./state";

hljs.registerLanguage("sql", sql);

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
  private chatInputWrapper: HTMLElement;
  private chatInputHighlight: HTMLElement;
  private chatInput: HTMLTextAreaElement;
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
    this.chatInputWrapper = document.getElementById(
      "chat-input-wrapper",
    ) as HTMLElement;
    this.chatInputHighlight = document.getElementById(
      "chat-input-highlight",
    ) as HTMLElement;
    this.chatInput = document.getElementById(
      "chat-input",
    ) as HTMLTextAreaElement;
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
    this.loadModels();
    this.loadSchema();
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
   * Fetches the database schema and renders it in the sidebar.
   */
  private async loadSchema(): Promise<void> {
    const schemaContent = document.getElementById("schema-content");
    if (!schemaContent) return;

    try {
      const response = await fetch("http://localhost:8000/api/schema");
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      const data = await response.json();

      schemaContent.innerHTML = ""; // Clear loading message

      if (!data.tables || data.tables.length === 0) {
        schemaContent.innerHTML =
          '<div class="schema-loading">No tables found. Load data first.</div>';
        return;
      }

      data.tables.forEach(
        (table: {
          name: string;
          columns: { name: string; type: string }[];
        }) => {
          const tableDiv = document.createElement("div");
          tableDiv.className = "schema-table";

          const headerDiv = document.createElement("div");
          headerDiv.className = "schema-table-header";
          headerDiv.innerHTML = `
          <div class="schema-table-header-left">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <polyline points="9 18 15 12 9 6"></polyline>
            </svg>
            <span>${table.name}</span>
          </div>
          <button class="icon-btn table-view-btn" title="View Table Data" data-table="${table.name}">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
              <line x1="3" y1="9" x2="21" y2="9"></line>
              <line x1="9" y1="21" x2="9" y2="9"></line>
            </svg>
          </button>
        `;

          headerDiv.addEventListener("click", () => {
            tableDiv.classList.toggle("expanded");
          });

          const viewBtn = headerDiv.querySelector(".table-view-btn");
          viewBtn?.addEventListener("click", (e) => {
            e.stopPropagation();
            this.openTableModal(table.name);
          });
          const columnsList = document.createElement("ul");
          columnsList.className = "schema-columns";

          table.columns.forEach((col) => {
            const colLi = document.createElement("li");
            colLi.className = "schema-column";
            colLi.innerHTML = `
            <span>${col.name}</span>
            <span class="schema-column-type">${col.type}</span>
          `;
            columnsList.appendChild(colLi);
          });

          tableDiv.appendChild(headerDiv);
          tableDiv.appendChild(columnsList);
          schemaContent.appendChild(tableDiv);
        },
      );
    } catch (error) {
      console.error("Failed to load schema:", error);
      schemaContent.innerHTML =
        '<div class="schema-loading error-text">Failed to load schema</div>';
    }
  }

  /**
   * Fetches available models from the backend and populates the dropdown.
   */
  private async loadModels(): Promise<void> {
    try {
      const response = await fetch("http://localhost:8000/api/models");
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      const data = await response.json();

      // Preserve the "Literal SQL" option
      this.modelSelect.innerHTML = '<option value="sql">Literal SQL</option>';

      data.models.forEach((model: { name: string; size?: number }) => {
        const option = document.createElement("option");
        option.value = model.name;
        option.textContent = model.name;
        this.modelSelect.appendChild(option);
      });

      // Reset the current model to match the active chat
      const activeChat = this.state.getActiveChat();
      if (activeChat) {
        // If the model exists, set it, else fall back to the first available model that's not 'sql', or 'gemma4'
        const exists = Array.from(this.modelSelect.options).some(
          (opt) => opt.value === activeChat.model,
        );
        if (exists) {
          this.modelSelect.value = activeChat.model;
        } else if (data.models.length > 0) {
          this.modelSelect.value = data.models[0].name;
          this.state.setActiveChatModel(data.models[0].name);
        }
      }
    } catch (error) {
      console.warn("Failed to fetch models from API:", error);
      // Fallback is handled by the default HTML structure if we didn't wipe it,
      // but if we got here before wiping, it's fine.
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

    this.chatInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        this.chatForm.dispatchEvent(
          new Event("submit", { cancelable: true, bubbles: true }),
        );
      }
    });

    this.chatInput.addEventListener("input", () => {
      this.syncHighlight();
    });

    this.modelSelect.addEventListener("change", () => {
      this.state.setActiveChatModel(this.modelSelect.value);
      this.syncHighlight();
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

    // Schema Explorer Toggle
    const toggleSchemaBtn = document.getElementById("toggle-schema-btn");
    const schemaExplorer = document.getElementById("schema-explorer");
    if (toggleSchemaBtn && schemaExplorer) {
      // Allow clicking the entire header to toggle
      const schemaHeader = document.querySelector(".schema-header");
      if (schemaHeader) {
        schemaHeader.addEventListener("click", () => {
          schemaExplorer.classList.toggle("collapsed");
        });
      } else {
        toggleSchemaBtn.addEventListener("click", () => {
          schemaExplorer.classList.toggle("collapsed");
        });
      }
    }

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
   * Syncs the textarea value to the highlight block and resizes it.
   */
  private syncHighlight(): void {
    let value = this.chatInput.value;

    // Ensure trailing newlines don't collapse
    if (value.endsWith("\n")) {
      value += " ";
    }

    this.chatInputHighlight.textContent = value;

    if (
      this.modelSelect.value === "sql" ||
      value.trim().toUpperCase().startsWith("SELECT")
    ) {
      this.chatInputHighlight.className = "language-sql";
      delete this.chatInputHighlight.dataset.highlighted;
      hljs.highlightElement(this.chatInputHighlight);
    } else {
      this.chatInputHighlight.className = "";
      this.chatInputHighlight.textContent = value;
    }

    // Auto-resize both textarea and wrapper
    this.chatInput.style.height = "auto";
    const scrollHeight = this.chatInput.scrollHeight;
    this.chatInput.style.height = scrollHeight + "px";
  }

  /**
   * Handles sending a user message.
   */
  private async handleSendMessage(
    text?: string,
    modelOverride?: string,
  ): Promise<void> {
    const content = text ?? this.chatInput.value.trim();
    if (!content) return;

    this.state.addMessageToActiveChat({ role: "user", content });

    if (text === undefined) {
      this.chatInput.value = "";
      this.syncHighlight(); // Reset height and highlight block
    }

    this.renderSidebar();
    this.renderMessages();
    this.scrollToBottom();

    // UI state while loading
    this.chatInput.disabled = true;
    this.chatInputWrapper.classList.add("disabled");
    this.sendBtn.disabled = true;
    this.modelSelect.disabled = true;

    // Show a loading indicator
    const loadingDiv = document.createElement("div");
    loadingDiv.className = "message assistant loading";
    loadingDiv.innerHTML =
      '<div class="typing-indicator"><span></span><span></span><span></span></div><div style="font-size: 0.85em; margin-top: 0.5rem; opacity: 0.7;">Querying...</div>';
    this.messagesContainer.appendChild(loadingDiv);
    this.scrollToBottom();

    const model = modelOverride ?? this.modelSelect.value;

    try {
      const response = await fetch("http://localhost:8000/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: content, model }),
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
        isError: !!data.error,
        model: model,
      });
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : String(err);
      this.state.addMessageToActiveChat({
        role: "assistant",
        content: `Error communicating with backend API: ${errorMsg}`,
        isError: true,
        model: model,
      });
    } finally {
      this.chatInput.disabled = false;
      this.chatInputWrapper.classList.remove("disabled");
      this.sendBtn.disabled = false;
      this.modelSelect.disabled = false;
      this.chatInput.focus();
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
      this.chatInputWrapper.classList.add("disabled");
      this.sendBtn.disabled = true;
      this.modelSelect.disabled = true;
      return;
    }

    this.chatInput.disabled = false;
    this.chatInputWrapper.classList.remove("disabled");
    this.sendBtn.disabled = false;
    this.modelSelect.disabled = false;
    this.modelSelect.value = activeChat.model;

    // Ensure input field and highlight blocks accurately reflect all saved state or empty initial load
    this.syncHighlight();

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
        {
          label: "[SQL] Patient Demographics",
          query:
            "-- Patient Demographics & Treatment Breakdown\n\nSELECT \n    TxGroup, \n    COUNT(*) as total_patients, \n    ROUND(AVG(AgeAsOfRandDt), 1) as avg_age,\n    SUM(NumSevHypo) as total_severe_hypo_events\nFROM tblaptsummary \nGROUP BY TxGroup;",
        },
        {
          label: "[SQL] Adverse Events Frequency",
          query:
            "-- Frequency of Adverse Events\n\nSELECT \n    event as adverse_event_type, \n    COUNT(*) as occurrence_count \nFROM adverseevents \nGROUP BY event \nORDER BY occurrence_count DESC;",
        },
        {
          label: "[SQL] Pump Manufacturers",
          query:
            "-- Patient Pump Manufacturers\n\nSELECT \n    Pt_PumpManuf as pump_manufacturer, \n    COUNT(*) as user_count \nFROM subjects \nWHERE Pt_PumpManuf IS NOT NULL \nGROUP BY Pt_PumpManuf \nORDER BY user_count DESC;",
        },
        {
          label: "[SQL] HbA1c by Demographics",
          query:
            "-- Joining Subject Summaries with HbA1c Lab Results\n\nSELECT \n    t.TxGroup, \n    t.Gender,\n    ROUND(AVG(h.HbA1c), 2) as average_hba1c,\n    COUNT(h.HbA1c) as total_tests_run\nFROM tblaptsummary t\nJOIN hba1c h ON t.PtID = h.PtID\nWHERE h.HbA1c IS NOT NULL\nGROUP BY t.TxGroup, t.Gender\nORDER BY t.TxGroup, average_hba1c DESC;",
        },
        {
          label: "[NLP] Show First 5 Patients",
          query: "Show me the first 5 patients in the demographics table.",
        },
      ];

      exampleQueries.forEach((item) => {
        const chip = document.createElement("button");
        chip.className = "chip";
        chip.textContent = item.label;
        chip.addEventListener("click", () => {
          this.chatInput.value = item.query;
          if (item.label.startsWith("[SQL]")) {
            this.modelSelect.value = "sql";
            this.state.setActiveChatModel("sql");
          } else {
            this.modelSelect.value = "gemma4";
            this.state.setActiveChatModel("gemma4");
          }
          this.chatInput.focus();

          this.syncHighlight();

          // Optional: handleSendMessage() could be called here if auto-send is desired,
          // but the user wants to see the query properly below it first.
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

      if (msg.role === "assistant" && msg.model) {
        const header = document.createElement("div");
        header.className = "message-header";
        const badge = document.createElement("span");
        badge.className = "model-badge";
        badge.textContent =
          msg.model === "sql" ? "Raw SQL" : `Model: ${msg.model}`;
        header.appendChild(badge);
        msgDiv.appendChild(header);
      }

      const contentDiv = document.createElement("div");
      contentDiv.textContent = msg.content;

      if (msg.isError) {
        contentDiv.className = "error-text";
      }

      msgDiv.appendChild(contentDiv);

      if (msg.sqlQuery) {
        const queryContainer = document.createElement("div");
        queryContainer.className = "sql-query-container";
        const preBlock = document.createElement("pre");
        const codeBlock = document.createElement("code");
        codeBlock.className = "language-sql sql-query";
        codeBlock.textContent = msg.sqlQuery;

        hljs.highlightElement(codeBlock);
        preBlock.appendChild(codeBlock);

        const headerDiv = document.createElement("div");
        headerDiv.className = "sql-query-header";

        // Only show Play button if this is an NLP generated query (not literal SQL)
        if (msg.model !== "sql" && msg.role === "assistant" && !msg.sqlResult) {
          const playBtn = document.createElement("button");
          playBtn.className = "play-sql-btn";
          playBtn.innerHTML = `
            <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <polygon points="5 3 19 12 5 21 5 3"></polygon>
            </svg>
            Run SQL
          `;
          playBtn.addEventListener("click", () => {
            this.handleSendMessage(msg.sqlQuery, "sql");
          });
          headerDiv.appendChild(playBtn);
        }

        queryContainer.appendChild(headerDiv);
        queryContainer.appendChild(preBlock);
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

  // Modal State
  private currentTable: string = "";
  private currentPage: number = 1;
  private readonly ROWS_PER_PAGE = 25;

  private async openTableModal(tableName: string) {
    this.currentTable = tableName;
    this.currentPage = 1;

    const modal = document.getElementById("table-modal");
    const modalTitle = document.getElementById("modal-title");
    const closeBtn = document.getElementById("close-modal-btn");
    const prevBtn = document.getElementById(
      "prev-page-btn",
    ) as HTMLButtonElement;
    const nextBtn = document.getElementById(
      "next-page-btn",
    ) as HTMLButtonElement;

    if (!modal || !modalTitle || !closeBtn || !prevBtn || !nextBtn) return;

    modalTitle.textContent = `Table: ${tableName}`;
    modal.removeAttribute("aria-hidden");

    // Close listeners
    const closeModal = () => {
      modal.setAttribute("aria-hidden", "true");
      // Clean up event listeners if needed
      closeBtn.removeEventListener("click", closeModal);
      modal.removeEventListener("click", overlayClick);
    };

    const overlayClick = (e: MouseEvent) => {
      if (e.target === modal) closeModal();
    };

    closeBtn.addEventListener("click", closeModal);
    modal.addEventListener("click", overlayClick);

    // Pagination listeners (use clean ones)
    const prevFn = () => {
      if (this.currentPage > 1) {
        this.currentPage--;
        this.fetchTableData();
      }
    };
    const nextFn = () => {
      this.currentPage++;
      this.fetchTableData();
    };

    // Replace elements to clear old listeners
    const newPrevBtn = prevBtn.cloneNode(true) as HTMLButtonElement;
    const newNextBtn = nextBtn.cloneNode(true) as HTMLButtonElement;
    prevBtn.parentNode?.replaceChild(newPrevBtn, prevBtn);
    nextBtn.parentNode?.replaceChild(newNextBtn, nextBtn);

    newPrevBtn.addEventListener("click", prevFn);
    newNextBtn.addEventListener("click", nextFn);

    await this.fetchTableData();
  }

  private async fetchTableData() {
    const loading = document.getElementById("table-loading");
    const thead = document.getElementById("modal-table-head");
    const tbody = document.getElementById("modal-table-body");
    const prevBtn = document.getElementById(
      "prev-page-btn",
    ) as HTMLButtonElement;
    const nextBtn = document.getElementById(
      "next-page-btn",
    ) as HTMLButtonElement;
    const indicator = document.getElementById("page-indicator");

    if (!loading || !thead || !tbody || !prevBtn || !nextBtn || !indicator)
      return;

    loading.style.display = "block";
    thead.innerHTML = "";
    tbody.innerHTML = "";
    prevBtn.disabled = true;
    nextBtn.disabled = true;

    try {
      const limit = this.ROWS_PER_PAGE;
      const offset = (this.currentPage - 1) * limit;
      const response = await fetch(
        `http://localhost:8000/api/table/${this.currentTable}?limit=${limit}&offset=${offset}`,
      );
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = await response.json();

      loading.style.display = "none";
      indicator.textContent = `Page ${this.currentPage}`;

      if (data.rows.length === 0 && this.currentPage > 1) {
        // No more rows
        this.currentPage--;
        indicator.textContent = `Page ${this.currentPage}`;
        return;
      }

      if (data.rows.length > 0) {
        // Render header
        const columns = Object.keys(data.rows[0]);
        const trHead = document.createElement("tr");
        columns.forEach((col) => {
          const th = document.createElement("th");
          th.textContent = col;
          trHead.appendChild(th);
        });
        thead.appendChild(trHead);

        // Render rows
        data.rows.forEach(
          (row: Record<string, string | number | boolean | null>) => {
            const tr = document.createElement("tr");
            columns.forEach((col) => {
              const td = document.createElement("td");
              td.textContent = row[col] === null ? "NULL" : String(row[col]);
              tr.appendChild(td);
            });
            tbody.appendChild(tr);
          },
        );

        prevBtn.disabled = this.currentPage === 1;
        nextBtn.disabled = data.rows.length < limit;
      } else {
        tbody.innerHTML = "<tr><td>No data available.</td></tr>";
        prevBtn.disabled = true;
        nextBtn.disabled = true;
      }
    } catch (error) {
      console.error("Failed to fetch table data:", error);
      loading.style.display = "none";
      tbody.innerHTML = `<tr><td class="error-text">Failed to load data: ${error}</td></tr>`;
    }
  }
}
