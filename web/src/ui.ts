/**
 * @file ui.ts
 * DOM manipulation and event binding for the Chat UI.
 */

import { marked } from "marked";
import hljs from "highlight.js/lib/core";
import sql from "highlight.js/lib/languages/sql";
import markdown from "highlight.js/lib/languages/markdown";
import "highlight.js/styles/github-dark.css";
import { ChatState } from "./state";
import i18next, { setLanguage } from "./i18n";

hljs.registerLanguage("sql", sql);
hljs.registerLanguage("markdown", markdown);

/**
 * Parses a backend error string or detail object into a translated string.
 */
function parseApiError(errorStr: any): string {
  if (typeof errorStr !== "string") return String(errorStr);
  if (errorStr.includes("|")) {
    const [key, ...rest] = errorStr.split("|");
    return i18next.exists(key)
      ? i18next.t(key, { error: rest.join("|") })
      : errorStr;
  }
  return i18next.exists(errorStr) ? i18next.t(errorStr) : errorStr;
}

/**
 * Helper to fetch and automatically throw parsed backend errors.
 */
async function fetchWithBackendError(
  url: string,
  options?: RequestInit,
): Promise<Response> {
  const response = await fetch(url, options);
  if (!response.ok) {
    let errorMsg = `HTTP ${response.status}`;
    try {
      const errData = await response.json();
      if (errData.detail) {
        errorMsg = parseApiError(errData.detail);
      }
    } catch (e) {
      // Ignore JSON parse errors
    }
    throw new Error(errorMsg);
  }
  return response;
}

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
  private langSelect: HTMLSelectElement;
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
    this.langSelect = document.getElementById(
      "lang-select",
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
        this.chatInput.focus();
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
      const response = await fetchWithBackendError(
        "http://localhost:8000/api/schema",
      );
      const data = await response.json();

      schemaContent.innerHTML = ""; // Clear loading message

      if (!data.tables || data.tables.length === 0) {
        schemaContent.innerHTML = `<div class="schema-loading" role="status" aria-live="polite" data-i18n="ui.noTables">${i18next.t("ui.noTables")}</div>`;
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
          <button class="schema-table-header-toggle" aria-expanded="false" aria-controls="schema-table-${table.name}">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true">
              <polyline points="9 18 15 12 9 6"></polyline>
            </svg>
            <span>${table.name}</span>
          </button>
          <button class="icon-btn table-view-btn" aria-label="${i18next.t("aria.viewTableData")}" title="${i18next.t("ui.viewTableData")}" data-table="${table.name}" data-i18n-aria-label="aria.viewTableData" data-i18n-title="ui.viewTableData">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
              <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
              <line x1="3" y1="9" x2="21" y2="9"></line>
              <line x1="9" y1="21" x2="9" y2="9"></line>
            </svg>
          </button>
        `;

          const toggleBtn = headerDiv.querySelector(
            ".schema-table-header-toggle",
          ) as HTMLButtonElement;

          /**
           * Toggles the expansion state of the table schema view.
           */
          const toggleExpand = () => {
            const isExpanded = tableDiv.classList.toggle("expanded");
            toggleBtn.setAttribute(
              "aria-expanded",
              isExpanded ? "true" : "false",
            );
          };
          toggleBtn.addEventListener("click", toggleExpand);

          const viewBtn = headerDiv.querySelector(".table-view-btn");
          viewBtn?.addEventListener("click", (e) => {
            e.stopPropagation();
            this.openTableModal(table.name);
          });
          const columnsList = document.createElement("ul");
          columnsList.className = "schema-columns";
          columnsList.id = `schema-table-${table.name}`;

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
      schemaContent.innerHTML = `<div class="schema-loading error-text" role="alert" aria-live="assertive" data-i18n="ui.failedSchema">${i18next.t("ui.failedSchema")}</div>`;
    }
  }

  /**
   * Fetches available models from the backend and populates the dropdown.
   */
  private async loadModels(): Promise<void> {
    try {
      const response = await fetchWithBackendError(
        "http://localhost:8000/api/models",
      );
      const data = await response.json();

      // Preserve the "Literal SQL" option
      this.modelSelect.innerHTML = `<option value="sql" data-i18n="app.literalSql">${i18next.t("app.literalSql")}</option>`;

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
      this.chatInput.focus();
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

    if (this.langSelect) {
      this.langSelect.value = i18next.language;
      this.langSelect.addEventListener("change", async () => {
        await setLanguage(this.langSelect.value);
        // Re-render UI components that generate dynamic text
        this.render();
      });
    }

    this.themeToggleBtn.addEventListener("click", () => {
      const isLight = document.body.classList.toggle("light-mode");
      this.themeToggleBtn.setAttribute(
        "aria-pressed",
        isLight ? "false" : "true",
      );
      this.themeToggleBtn.setAttribute(
        "aria-label",
        isLight
          ? i18next.t("aria.switchToDarkMode")
          : i18next.t("aria.switchToLightMode"),
      );
      this.themeToggleBtn.innerHTML = isLight ? "☾" : "☀";
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
      const toggleFn = () => {
        const isCollapsed = schemaExplorer.classList.toggle("collapsed");
        toggleSchemaBtn.setAttribute(
          "aria-expanded",
          isCollapsed ? "false" : "true",
        );
      };

      toggleSchemaBtn.setAttribute("aria-expanded", "true"); // Initially expanded

      const schemaHeader = document.querySelector(".schema-header");
      if (schemaHeader) {
        schemaHeader.addEventListener("click", toggleFn);
      } else {
        toggleSchemaBtn.addEventListener("click", toggleFn);
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
   * Resizes the textarea height automatically.
   */
  private syncHighlight(): void {
    let value = this.chatInput.value;

    // Ensure trailing newlines don't collapse
    if (value.endsWith("\n")) {
      value += " ";
    }

    if (this.chatInputHighlight) {
      this.chatInputHighlight.textContent = value;

      if (this.modelSelect.value === "sql") {
        this.chatInputHighlight.className = "language-sql";
      } else {
        this.chatInputHighlight.className = "language-markdown";
      }
      delete this.chatInputHighlight.dataset.highlighted;
      hljs.highlightElement(this.chatInputHighlight);
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

    this.state.addMessageToActiveChat({
      role: "user",
      content,
      model: modelOverride || this.modelSelect.value,
    });

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
    loadingDiv.setAttribute("role", "status");
    loadingDiv.setAttribute("aria-live", "polite");
    loadingDiv.innerHTML = `<div class="typing-indicator"><span></span><span></span><span></span></div><div style="font-size: 0.85em; margin-top: 0.5rem; opacity: 0.7;">${i18next.t("ui.querying")}</div>`;
    this.messagesContainer.appendChild(loadingDiv);
    this.scrollToBottom();

    const model = modelOverride ?? this.modelSelect.value;

    try {
      const response = await fetchWithBackendError(
        "http://localhost:8000/api/chat",
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ message: content, model }),
        },
      );

      const data = await response.json();

      let assistantMsg = i18next.exists(data.content)
        ? i18next.t(data.content)
        : data.content;
      if (data.error) {
        const parsedError = parseApiError(data.error);
        assistantMsg += `\n${i18next.t("ui.errorDetails", { error: parsedError })}`;
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
        content: i18next.t("ui.errorComm", { error: errorMsg }),
        isError: true,
        model: model,
      });
    } finally {
      this.chatInput.disabled = false;
      this.chatInputWrapper.classList.remove("disabled");
      this.sendBtn.disabled = false;
      this.modelSelect.disabled = false;
      this.chatInput.focus();

      const announcer = document.getElementById("a11y-announcer");
      if (announcer) {
        announcer.textContent = i18next.t(
          "aria.messageReceived",
          "Message received",
        );
      }
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
      const btn = menu.previousElementSibling;
      if (btn && btn.classList.contains("dropdown-btn")) {
        btn.setAttribute("aria-expanded", "false");
      }
    });
  }

  /**
   * Main render function that updates the entire UI.
   * Also updates the document title for accessibility.
   */
  public render(): void {
    if (
      this.state.activeChatId &&
      window.location.hash !== `#${this.state.activeChatId}`
    ) {
      window.history.replaceState(null, "", `#${this.state.activeChatId}`);
    }

    const activeChat = this.state.getActiveChat();
    const appTitle = i18next.t("app.title");
    document.title = activeChat
      ? `${activeChat.title} - ${appTitle}`
      : appTitle;

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
      titleSpan.setAttribute("role", "button");
      titleSpan.setAttribute("tabindex", "0");

      const activateChat = () => {
        this.state.setActiveChat(chat.id);
        this.render();
        this.closeMobileSidebar();
        this.chatInput.focus();
      };

      titleSpan.addEventListener("click", activateChat);
      titleSpan.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          activateChat();
        }
      });

      const dropdownContainer = document.createElement("div");
      dropdownContainer.className = "dropdown-container";

      const dropdownBtn = document.createElement("button");
      dropdownBtn.className = "dropdown-btn";
      dropdownBtn.innerHTML = "&#8942;"; // 3 vertical dots
      dropdownBtn.setAttribute("aria-label", i18next.t("aria.chatOptions"));
      dropdownBtn.setAttribute("aria-haspopup", "menu");
      dropdownBtn.setAttribute("aria-expanded", "false");
      dropdownBtn.setAttribute("aria-controls", `dropdown-menu-${chat.id}`);

      const dropdownMenu = document.createElement("div");
      dropdownMenu.id = `dropdown-menu-${chat.id}`;
      dropdownMenu.className = "dropdown-menu";
      dropdownMenu.setAttribute("role", "menu");

      const renameBtn = document.createElement("button");
      renameBtn.className = "dropdown-item";
      renameBtn.textContent = i18next.t("ui.rename");
      renameBtn.setAttribute("role", "menuitem");
      renameBtn.setAttribute("tabindex", "-1");
      renameBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        this.closeAllDropdowns();
        dropdownBtn.focus();
        const newTitle = prompt(i18next.t("ui.newChatTitle"), chat.title);
        if (newTitle !== null) {
          this.state.renameChat(chat.id, newTitle);
          this.renderSidebar();
        }
      });

      const duplicateBtn = document.createElement("button");
      duplicateBtn.className = "dropdown-item";
      duplicateBtn.textContent = i18next.t("ui.duplicate");
      duplicateBtn.setAttribute("role", "menuitem");
      duplicateBtn.setAttribute("tabindex", "-1");
      duplicateBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        this.closeAllDropdowns();
        dropdownBtn.focus();
        this.state.duplicateChat(chat.id);
        this.render();
      });

      const deleteBtn = document.createElement("button");
      deleteBtn.className = "dropdown-item danger";
      deleteBtn.textContent = i18next.t("ui.delete");
      deleteBtn.setAttribute("role", "menuitem");
      deleteBtn.setAttribute("tabindex", "-1");
      deleteBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        this.closeAllDropdowns();
        dropdownBtn.focus();
        if (confirm(i18next.t("ui.deleteConfirm", { title: chat.title }))) {
          this.state.deleteChat(chat.id);
          this.render();
          this.chatInput.focus();
        }
      });

      const menuItems = [renameBtn, duplicateBtn, deleteBtn];

      dropdownBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        const isShowing = dropdownMenu.classList.contains("show");
        this.closeAllDropdowns();
        if (!isShowing) {
          dropdownMenu.classList.add("show");
          dropdownBtn.setAttribute("aria-expanded", "true");
          renameBtn.focus();
        }
      });

      dropdownBtn.addEventListener("keydown", (e) => {
        if (e.key === "ArrowDown" || e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          dropdownBtn.click();
        }
      });

      dropdownMenu.addEventListener("keydown", (e) => {
        const index = menuItems.indexOf(
          document.activeElement as HTMLButtonElement,
        );
        if (e.key === "ArrowDown") {
          e.preventDefault();
          const nextIndex = (index + 1) % menuItems.length;
          menuItems[nextIndex].focus();
        } else if (e.key === "ArrowUp") {
          e.preventDefault();
          const prevIndex = (index - 1 + menuItems.length) % menuItems.length;
          menuItems[prevIndex].focus();
        } else if (e.key === "Escape") {
          e.preventDefault();
          this.closeAllDropdowns();
          dropdownBtn.focus();
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
      this.messagesContainer.innerHTML = `<div class="empty-state" role="status" aria-live="polite" data-i18n="app.emptyState">${i18next.t("app.emptyState")}</div>`;
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
      emptyStateDiv.setAttribute("role", "status");
      emptyStateDiv.setAttribute("aria-live", "polite");
      emptyStateDiv.innerHTML = `<p style="margin-bottom: 1rem;">${i18next.t("ui.noMessagesYet")}</p>`;

      const chipsContainer = document.createElement("div");
      chipsContainer.className = "chips-container";

      const exampleQueries = [
        {
          label: i18next.t("ui.exampleQueries.patientDemographics"),
          query:
            "-- Patient Demographics & Treatment Breakdown\n\nSELECT \n    TxGroup, \n    COUNT(*) as total_patients, \n    ROUND(AVG(AgeAsOfRandDt), 1) as avg_age,\n    SUM(NumSevHypo) as total_severe_hypo_events\nFROM tblaptsummary \nGROUP BY TxGroup;",
        },
        {
          label: i18next.t("ui.exampleQueries.adverseEvents"),
          query:
            "-- Frequency of Adverse Events\n\nSELECT \n    event as adverse_event_type, \n    COUNT(*) as occurrence_count \nFROM adverseevents \nGROUP BY event \nORDER BY occurrence_count DESC;",
        },
        {
          label: i18next.t("ui.exampleQueries.pumpManufacturers"),
          query:
            "-- Patient Pump Manufacturers\n\nSELECT \n    Pt_PumpManuf as pump_manufacturer, \n    COUNT(*) as user_count \nFROM subjects \nWHERE Pt_PumpManuf IS NOT NULL \nGROUP BY Pt_PumpManuf \nORDER BY user_count DESC;",
        },
        {
          label: i18next.t("ui.exampleQueries.hba1cDemographics"),
          query:
            "-- Joining Subject Summaries with HbA1c Lab Results\n\nSELECT \n    t.TxGroup, \n    t.Gender,\n    ROUND(AVG(h.HbA1c), 2) as average_hba1c,\n    COUNT(h.HbA1c) as total_tests_run\nFROM tblaptsummary t\nJOIN hba1c h ON t.PtID = h.PtID\nWHERE h.HbA1c IS NOT NULL\nGROUP BY t.TxGroup, t.Gender\nORDER BY t.TxGroup, average_hba1c DESC;",
        },
        {
          label: i18next.t("ui.exampleQueries.nlpFirst5"),
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

      if (msg.model) {
        const header = document.createElement("div");
        header.className = "message-header";
        const badge = document.createElement("span");
        badge.className = "model-badge";
        badge.textContent =
          msg.model === "sql"
            ? i18next.t("ui.rawSql")
            : i18next.t("ui.modelName", { name: msg.model });

        // Assistant messages align header left by default in flex (or start).
        // User messages align header right. We handle this via CSS usually,
        // but we can add a specific class if needed.
        if (msg.role === "user") {
          header.style.justifyContent = "flex-end";
          badge.style.backgroundColor = "rgba(255, 255, 255, 0.2)";
          badge.style.color = "var(--user-msg-text)";
        }

        header.appendChild(badge);
        msgDiv.appendChild(header);
      }

      const contentDiv = document.createElement("div");

      const isSqlMessage =
        msg.role === "user" &&
        (msg.model === "sql" || (!msg.model && activeChat.model === "sql"));

      if (isSqlMessage) {
        contentDiv.className = "sql-user-query-wrapper";
        const preBlock = document.createElement("pre");
        preBlock.style.margin = "0";
        preBlock.style.background = "transparent";
        preBlock.style.padding = "0";
        preBlock.style.whiteSpace = "pre-wrap";
        preBlock.tabIndex = 0;
        preBlock.setAttribute("aria-label", i18next.t("aria.codeBlock"));

        const codeBlock = document.createElement("code");
        codeBlock.className = "language-sql";
        codeBlock.style.background = "transparent";
        codeBlock.style.padding = "0";
        codeBlock.textContent = msg.content;

        hljs.highlightElement(codeBlock);
        preBlock.appendChild(codeBlock);
        contentDiv.appendChild(preBlock);
      } else {
        contentDiv.innerHTML = marked.parse(msg.content) as string;
        const codeBlocks = contentDiv.querySelectorAll("pre code");
        codeBlocks.forEach((block) => {
          hljs.highlightElement(block as HTMLElement);
        });
      }

      if (msg.isError) {
        contentDiv.className = "error-text";
        contentDiv.setAttribute("role", "alert");
        contentDiv.setAttribute("aria-live", "assertive");
      }

      msgDiv.appendChild(contentDiv);

      if (msg.sqlQuery) {
        const queryContainer = document.createElement("div");
        queryContainer.className = "sql-query-container";
        const preBlock = document.createElement("pre");
        preBlock.tabIndex = 0;
        preBlock.setAttribute("aria-label", i18next.t("aria.codeBlock"));
        const codeBlock = document.createElement("code");
        codeBlock.className = "language-sql sql-query";
        codeBlock.textContent = msg.sqlQuery;

        hljs.highlightElement(codeBlock);
        preBlock.appendChild(codeBlock);

        const headerDiv = document.createElement("div");
        headerDiv.className = "sql-query-header";

        // Add copy button for all SQL queries
        const copyBtn = document.createElement("button");
        copyBtn.className = "copy-sql-btn icon-btn";
        copyBtn.setAttribute("aria-label", "Copy SQL");
        copyBtn.title = "Copy SQL";
        copyBtn.innerHTML = `
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
            <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
          </svg>
        `;
        copyBtn.addEventListener("click", async () => {
          try {
            await navigator.clipboard.writeText(msg.sqlQuery!);
            const originalSvg = copyBtn.innerHTML;
            copyBtn.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"></polyline></svg>`;
            setTimeout(() => {
              copyBtn.innerHTML = originalSvg;
            }, 2000);
          } catch (err) {
            console.error("Failed to copy text: ", err);
          }
        });
        headerDiv.appendChild(copyBtn);

        // Only show Play button if this is an NLP generated query (not literal SQL)
        if (msg.model !== "sql" && msg.role === "assistant" && !msg.sqlResult) {
          const playBtn = document.createElement("button");
          playBtn.className = "play-sql-btn";
          playBtn.setAttribute("aria-label", i18next.t("aria.playQuery"));
          playBtn.innerHTML = `
            <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
              <polygon points="5 3 19 12 5 21 5 3"></polygon>
            </svg>
            ${i18next.t("aria.playQuery")}
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

          const caption = document.createElement("caption");
          caption.className = "sr-only";
          caption.textContent = i18next.t("app.tableData");
          table.appendChild(caption);

          const thead = document.createElement("thead");
          const headerRow = document.createElement("tr");
          const columns = Object.keys(msg.sqlResult[0]);

          columns.forEach((col) => {
            const th = document.createElement("th");
            th.textContent = col;
            th.setAttribute("scope", "col");
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
          emptyDiv.textContent = i18next.t("ui.noRows");
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

  /**
   * Opens the table modal and loads the first page of data.
   * @param {string} tableName - The name of the table to display.
   */
  private async openTableModal(tableName: string) {
    const previousFocus = document.activeElement as HTMLElement | null;
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

    modalTitle.textContent = i18next.t("ui.tableName", { name: tableName });
    modal.removeAttribute("aria-hidden");

    // Focus the first interactive element
    closeBtn.focus();

    // Trap focus inside modal
    const trapFocus = (e: KeyboardEvent) => {
      if (e.key !== "Tab") return;
      const focusableElements = modal.querySelectorAll<HTMLElement>(
        'button:not([disabled]), [href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])',
      );
      const firstElement = focusableElements[0];
      const lastElement = focusableElements[focusableElements.length - 1];

      if (e.shiftKey) {
        if (document.activeElement === firstElement) {
          e.preventDefault();
          lastElement?.focus();
        }
      } else {
        if (document.activeElement === lastElement) {
          e.preventDefault();
          firstElement?.focus();
        }
      }
    };

    modal.addEventListener("keydown", trapFocus);

    // Close listeners
    const closeModal = () => {
      modal.setAttribute("aria-hidden", "true");
      // Clean up event listeners if needed
      closeBtn.removeEventListener("click", closeModal);
      modal.removeEventListener("click", overlayClick);
      modal.removeEventListener("keydown", trapFocus);

      // Restore focus to where it was before opening the modal
      if (previousFocus) {
        previousFocus.focus();
      }
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

  /**
   * Fetches the current page of data for the opened table and updates the modal UI.
   */
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
      const response = await fetchWithBackendError(
        `http://localhost:8000/api/table/${this.currentTable}?limit=${limit}&offset=${offset}`,
      );
      const data = await response.json();

      loading.style.display = "none";
      indicator.textContent = i18next.t("app.page", { page: this.currentPage });

      if (data.rows.length === 0 && this.currentPage > 1) {
        // No more rows
        this.currentPage--;
        indicator.textContent = i18next.t("app.page", {
          page: this.currentPage,
        });
        return;
      }

      if (data.rows.length > 0) {
        // Render header
        const columns = Object.keys(data.rows[0]);
        const trHead = document.createElement("tr");
        columns.forEach((col) => {
          const th = document.createElement("th");
          th.textContent = col;
          th.setAttribute("scope", "col");
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
        tbody.innerHTML = `<tr><td>${i18next.t("ui.noDataAvailable")}</td></tr>`;
        prevBtn.disabled = true;
        nextBtn.disabled = true;
      }
    } catch (error) {
      console.error("Failed to fetch table data:", error);
      loading.style.display = "none";
      tbody.innerHTML = `<tr><td class="error-text" role="alert" aria-live="assertive">${i18next.t("ui.failedData", { error })}</td></tr>`;
    }
  }
}
