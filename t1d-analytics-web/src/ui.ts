/**
 * @file ui.ts
 * @module ui
 * Main Chat UI coordinator delegating modal, schema, status, sidebar, model selection,
 * and message rendering operations to composable sub-controllers.
 */

import "highlight.js/styles/github-dark.css";
import { parseResponseJson } from "./api-client";
import { buildCohortSql, normalizeJoinType } from "./cohort-sql";
import {
  type CohortBuilderHost,
  CohortBuilderModalController,
} from "./controllers/cohort-modal-controller";
import {
  type ModelSelectorHost,
  ModelSelectorController,
} from "./controllers/model-selector-controller";
import {
  type ProviderSettingsHost,
  ProviderSettingsModalController,
} from "./controllers/provider-modal-controller";
import {
  type SchemaExplorerHost,
  SchemaExplorerController,
} from "./controllers/schema-explorer-controller";
import {
  type SidebarHost,
  SidebarController,
} from "./controllers/sidebar-controller";
import {
  type SystemStatusHost,
  SystemStatusController,
  isDbStatusCode,
} from "./controllers/system-status-controller";
import {
  type TablePreviewHost,
  TablePreviewModalController,
} from "./controllers/table-modal-controller";
import { getBuildTimestampUtc } from "./formatters";
import i18next, { setLanguage } from "./i18n";
import { type MessageRendererHost, MessageRenderer } from "./message-renderer";
import { requireElementById, trapModalFocus } from "./modal-utils";
import { isReadOnlySqlStatement, updateInputHighlight } from "./sql-highlight";
import { dispatchChatRequest } from "./sse-stream";
import { ChatState, safeStorageGet, safeStorageSet } from "./state";

export {
  buildCohortSql,
  CohortBuilderModalController,
  isDbStatusCode,
  ModelSelectorController,
  normalizeJoinType,
  parseResponseJson,
  ProviderSettingsModalController,
  requireElementById,
  SchemaExplorerController,
  SidebarController,
  SystemStatusController,
  TablePreviewModalController,
  trapModalFocus,
};

/** Active ChatUI instance so re-instantiation in tests automatically disposes previous listeners/timers. */
let activeChatUIInstance: ChatUI | null = null;

/**
 * Lean application coordinator wiring `ChatState` and composable UI sub-controllers.
 */
export class ChatUI
  implements
    CohortBuilderHost,
    ModelSelectorHost,
    ProviderSettingsHost,
    SystemStatusHost,
    TablePreviewHost,
    SchemaExplorerHost,
    SidebarHost,
    MessageRendererHost
{
  /** Global application state store. */
  public state: ChatState;
  /** AbortController for global document/window listeners bound by this ChatUI instance. */
  private readonly globalListenerController: AbortController;
  /** Sub-controller for the Table Preview Modal. */
  public readonly tableModalController: TablePreviewModalController;
  /** Sub-controller for the Visual Cohort Builder Modal. */
  public readonly cohortModalController: CohortBuilderModalController;
  /** Sub-controller for the Provider Settings Modal. */
  public readonly providerModalController: ProviderSettingsModalController;
  /** Sub-controller for System Status & Diagnostics. */
  public readonly systemStatusController: SystemStatusController;
  /** Sub-controller for Schema Explorer & Database Selector. */
  public readonly schemaExplorerController: SchemaExplorerController;
  /** Sub-controller for Sidebar chat list and mobile navigation drawer. */
  public readonly sidebarController: SidebarController;
  /** Sub-controller for the AI/SQL Model Selector dropdown. */
  public readonly modelSelectorController: ModelSelectorController;
  /** Renderer for chat messages, SQL blocks, and CGM analytics cards. */
  public readonly messageRenderer: MessageRenderer;
  /** Scrollable container element holding chat messages. */
  public readonly messagesContainer: HTMLElement;
  /** Form element wrapping the chat input and send button. */
  public readonly chatForm: HTMLFormElement;
  /** Wrapper element for the syntax-highlighted textarea input. */
  public readonly chatInputWrapper: HTMLElement;
  /** Code element displaying live syntax highlighting behind the textarea. */
  public readonly chatInputHighlight: HTMLElement;
  /** Textarea element where the user types queries or SQL. */
  public readonly chatInput: HTMLTextAreaElement;
  /** Submit button element for sending messages. */
  public readonly sendBtn: HTMLButtonElement;
  /** Dropdown select element for choosing the active LLM or SQL engine. */
  public readonly modelSelect: HTMLSelectElement;
  /** Button element to manually trigger server session synchronization. */
  public readonly syncSessionsBtn: HTMLButtonElement;
  /** Button element to toggle real-time SSE token streaming. */
  public readonly streamingToggleBtn: HTMLButtonElement;
  /** Whether real-time SSE token streaming is enabled in the UI. */
  public streamingEnabled: boolean = false;
  /** Dropdown select element for switching the UI language/locale. */
  public readonly langSelect: HTMLSelectElement;
  /** Button element to toggle between dark mode and light mode. */
  public readonly themeToggleBtn: HTMLButtonElement;

  /**
   * Initializes the ChatUI coordinator and its composable sub-controllers.
   * @param {ChatState} state - The global state store.
   */
  constructor(state: ChatState) {
    if (activeChatUIInstance) {
      activeChatUIInstance.dispose();
    }
    this.globalListenerController = new AbortController();
    activeChatUIInstance = this;
    this.state = state;

    this.messagesContainer =
      requireElementById<HTMLElement>("messages-container");
    this.chatForm = requireElementById<HTMLFormElement>("chat-form");
    this.chatInputWrapper =
      requireElementById<HTMLElement>("chat-input-wrapper");
    this.chatInputHighlight = requireElementById<HTMLElement>(
      "chat-input-highlight",
    );
    this.chatInput = requireElementById<HTMLTextAreaElement>("chat-input");
    this.sendBtn = requireElementById<HTMLButtonElement>("send-btn");
    this.syncSessionsBtn =
      requireElementById<HTMLButtonElement>("sync-sessions-btn");
    this.streamingToggleBtn = requireElementById<HTMLButtonElement>(
      "streaming-toggle-btn",
    );
    this.langSelect = requireElementById<HTMLSelectElement>("lang-select");
    this.themeToggleBtn =
      requireElementById<HTMLButtonElement>("theme-toggle-btn");

    this.modelSelectorController = new ModelSelectorController(this);
    this.modelSelect = this.modelSelectorController.modelSelect;
    this.tableModalController = new TablePreviewModalController(this);
    this.cohortModalController = new CohortBuilderModalController(this);
    this.providerModalController = new ProviderSettingsModalController(this);
    this.systemStatusController = new SystemStatusController(this);
    this.schemaExplorerController = new SchemaExplorerController(this);
    this.sidebarController = new SidebarController(
      this,
      this.globalListenerController.signal,
    );
    this.messageRenderer = new MessageRenderer(this);

    const savedTheme = safeStorageGet("t1d_theme");
    if (savedTheme === "light") {
      document.body.classList.add("light-mode");
    } else if (savedTheme === "dark") {
      document.body.classList.remove("light-mode");
    }

    this.streamingEnabled = this.state.streamingMode;
    this.updateStreamingToggleUi();
    requireElementById<HTMLElement>("build-timestamp").textContent =
      `built on ${getBuildTimestampUtc()}`;

    if (window.location.hash) {
      this.state.setActiveChat(window.location.hash.slice(1));
    }

    this.bindEvents();
    void this.modelSelectorController.loadModels();
    void this.schemaExplorerController.loadSchema();
    this.render();

    window.addEventListener(
      "hashchange",
      () => {
        const hashId = window.location.hash.slice(1);
        if (hashId && hashId !== this.state.activeChatId) {
          this.state.setActiveChat(hashId);
          this.render();
          this.chatInput.focus();
        }
      },
      { signal: this.globalListenerController.signal },
    );
  }

  /**
   * Disposes global event listeners, modal listeners, and active status polling timers.
   * @returns {void}
   */
  public dispose(): void {
    this.systemStatusController.stopPolling();
    this.globalListenerController.abort();
    this.tableModalController.abortListeners();
    this.cohortModalController.dispose();
    this.providerModalController.dispose();
    if (activeChatUIInstance === this) {
      activeChatUIInstance = null;
    }
  }

  /**
   * Updates the aria-label, aria-pressed, and icon of the theme toggle button.
   * @returns {void}
   */
  private updateThemeButtonLabel(): void {
    const isLight = document.body.classList.contains("light-mode");
    const ariaKey = isLight
      ? "aria.switchToDarkMode"
      : "aria.switchToLightMode";
    this.themeToggleBtn.setAttribute("data-i18n-aria-label", ariaKey);
    this.themeToggleBtn.setAttribute("aria-label", i18next.t(ariaKey));
    this.themeToggleBtn.setAttribute(
      "aria-pressed",
      isLight ? "false" : "true",
    );
    this.themeToggleBtn.textContent = isLight ? "☾" : "☀";
  }

  /**
   * Updates the visual indicator and accessibility attributes on the streaming toggle button.
   * @returns {void}
   */
  public updateStreamingToggleUi(): void {
    const label = this.streamingEnabled
      ? i18next.t("ui.streamingOn", "Streaming Mode: ON (⚡ SSE)")
      : i18next.t("ui.streamingOff", "Streaming Mode: OFF (Standard Batch)");
    this.streamingToggleBtn.classList.toggle("active", this.streamingEnabled);
    this.streamingToggleBtn.style.color = this.streamingEnabled
      ? "var(--warning-accent, #f1c40f)"
      : "";
    this.streamingToggleBtn.title = label;
    this.streamingToggleBtn.setAttribute("aria-label", label);
    this.streamingToggleBtn.setAttribute(
      "aria-pressed",
      this.streamingEnabled ? "true" : "false",
    );
  }

  /**
   * Binds form submission, streaming toggle, sync button, locale selector, and theme toggle events.
   * @returns {void}
   */
  private bindEvents(): void {
    this.streamingToggleBtn.addEventListener("click", () => {
      this.streamingEnabled = !this.streamingEnabled;
      this.state.streamingMode = this.streamingEnabled;
      this.state.saveToLocalStorage();
      this.updateStreamingToggleUi();
    });

    this.chatForm.addEventListener("submit", (e) => {
      e.preventDefault();
      void this.handleSendMessage();
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

    this.syncSessionsBtn.addEventListener("click", () => {
      void this.syncAllChatsWithServer();
    });

    this.langSelect.value = i18next.language;
    this.langSelect.addEventListener("change", async () => {
      await setLanguage(this.langSelect.value);
      this.updateThemeButtonLabel();
      this.updateStreamingToggleUi();
      this.providerModalController.updateProviderBadgeStatus();
      this.schemaExplorerController.renderDatabaseDropdown();
      this.providerModalController.updateModalBadgesFromLocal();
      await this.tableModalController.refreshIfOpen();
      void this.systemStatusController.check();
      this.render();
    });

    this.updateThemeButtonLabel();
    this.themeToggleBtn.addEventListener("click", () => {
      const isLight = document.body.classList.toggle("light-mode");
      safeStorageSet("t1d_theme", isLight ? "light" : "dark");
      this.updateThemeButtonLabel();
    });
  }

  /**
   * Resizes the textarea height automatically and synchronizes syntax highlighting.
   * @returns {void}
   */
  public syncHighlight(): void {
    updateInputHighlight(
      this.chatInput,
      this.chatInputHighlight,
      this.modelSelect.value,
    );
  }

  /**
   * Announces a message to screen readers using the aria-live region.
   * @param {string} message - The text message to announce to assistive technology.
   * @param {boolean} [isAssertive=false] - Whether to use an assertive live region instead of polite.
   * @returns {void}
   */
  public announce(message: string, isAssertive: boolean = false): void {
    const announcer = document.getElementById("a11y-announcer");
    if (!announcer) return;
    announcer.textContent = "";
    announcer.setAttribute("aria-live", isAssertive ? "assertive" : "polite");
    setTimeout(() => {
      announcer.textContent = message;
    }, 50);
  }

  /**
   * Handles sending a user message and processing the streaming or batch response.
   * @param {string} [text] - Optional message text override (used when executing SQL directly).
   * @param {string} [modelOverride] - Optional model identifier override (e.g. 'sql').
   * @returns {Promise<void>}
   */
  public async handleSendMessage(
    text?: string,
    modelOverride?: string,
  ): Promise<void> {
    const content = text ?? this.chatInput.value.trim();
    if (!content) return;

    const model =
      modelOverride ??
      (this.modelSelect.value === "sql" || isReadOnlySqlStatement(content)
        ? "sql"
        : this.modelSelect.value);

    this.state.addMessageToActiveChat({ role: "user", content, model });

    if (text === undefined) {
      this.state.setActiveChatModel(this.modelSelect.value);
      this.chatInput.value = "";
      this.syncHighlight();
    }

    this.sidebarController.renderSidebar();
    this.messageRenderer.renderMessages();
    this.scrollToBottom();

    this.chatInput.disabled = true;
    this.chatInputWrapper.classList.add("disabled");
    this.sendBtn.disabled = true;
    this.modelSelect.disabled = true;

    const loadingDiv = this.messageRenderer.createLoadingIndicator();
    this.messagesContainer.appendChild(loadingDiv);
    this.scrollToBottom();

    const selectedOption = this.modelSelect.selectedOptions?.[0];
    const provider =
      selectedOption?.getAttribute("data-provider") ||
      (model.includes("/") ? model.split("/")[0] : undefined);
    const apiKey = provider ? this.state.getApiKey(provider) : undefined;

    const requestHeaders: Record<string, string> = {
      "Content-Type": "application/json",
      ...(provider ? { "x-provider": provider } : {}),
      ...(apiKey ? { "x-provider-api-key": apiKey } : {}),
    };

    const activeChat = this.state.getActiveChat();
    const history = (activeChat?.messages ?? []).slice(0, -1).map((m) => ({
      role: m.role,
      content: m.content,
      sqlQuery: m.sqlQuery,
      sqlResult: m.sqlResult,
      model: m.model,
    }));

    const dbPath =
      this.state.currentDb && this.state.currentDb !== "t1d.duckdb"
        ? this.state.currentDb
        : undefined;

    const bodyJson = JSON.stringify({
      message: content,
      model,
      provider,
      ...(dbPath ? { db_path: dbPath } : {}),
      history,
      auto_execute: model === "sql",
    });

    try {
      const outcome = await dispatchChatRequest({
        bodyJson,
        headers: requestHeaders,
        streamingEnabled: this.streamingEnabled,
        loadingDiv,
        onTokenScroll: () => this.scrollToBottom(),
      });
      loadingDiv.remove();
      this.messageRenderer.finalizeAssistantResponse(
        outcome.data,
        model,
        outcome.fallbackText,
      );
    } catch (err) {
      loadingDiv.remove();
      const errorMsg = err instanceof Error ? err.message : String(err);
      const announcedErr = i18next.t("ui.errorComm", { error: errorMsg });
      this.announce(announcedErr, true);
      this.state.addMessageToActiveChat({
        role: "assistant",
        content: announcedErr,
        isError: true,
        model,
      });
    }

    if (this.state.serverSyncEnabled) {
      void this.syncAllChatsWithServer();
    }

    this.chatInput.disabled = false;
    this.chatInputWrapper.classList.remove("disabled");
    this.sendBtn.disabled = false;
    this.modelSelect.disabled = false;
    this.chatInput.focus();

    this.announce(i18next.t("aria.messageReceived", "Message received"));
    this.sidebarController.renderSidebar();
    this.messageRenderer.renderMessages();
    this.scrollToBottom();
  }

  /**
   * Synchronizes all active non-temporary chats with the backend persistent store.
   * @returns {Promise<void>}
   */
  public async syncAllChatsWithServer(): Promise<void> {
    try {
      const ok = await this.state.syncWithServer();
      if (ok) {
        this.sidebarController.renderSidebar();
        this.messageRenderer.renderMessages();
        this.announce(i18next.t("ui.sessionsSynced"));
      }
    } catch {
      // Ignore sync network errors
    }
  }

  /**
   * Scrolls the message container to the bottom.
   * @returns {void}
   */
  private scrollToBottom(): void {
    this.messagesContainer.scrollTop = this.messagesContainer.scrollHeight;
  }

  /**
   * Main render function that updates the sidebar, active chat pane, and document title.
   * @returns {void}
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

    this.sidebarController.renderSidebar();
    this.messageRenderer.renderActiveChat();
  }
}
