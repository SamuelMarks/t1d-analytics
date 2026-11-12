/**
 * @file ui.ts
 * @module ui
 * Main Chat UI coordinator wiring unidirectional domain events (`TypedEventTarget<AppEvents>`)
 * and narrow injected dependencies across composable sub-controllers.
 */

import "highlight.js/styles/github-dark.css";
import { CohortBuilderModalController } from "./controllers/cohort-modal-controller";
import { ModelSelectorController } from "./controllers/model-selector-controller";
import { ProviderSettingsModalController } from "./controllers/provider-modal-controller";
import { SchemaExplorerController } from "./controllers/schema-explorer-controller";
import { SidebarController } from "./controllers/sidebar-controller";
import { SystemStatusController } from "./controllers/system-status-controller";
import { TablePreviewModalController } from "./controllers/table-modal-controller";
import { TemplatePickerController } from "./controllers/template-picker-controller";
import {
  TypedEventTarget,
  type AppEventBus,
  type AppEvents,
} from "./event-bus";
import { getBuildTimestampUtc } from "./formatters";
import i18next, { setLanguage } from "./i18n";
import { MessageRenderer } from "./message-renderer";
import { optionalElementById, requireElementById } from "./modal-utils";
import { isReadOnlySqlStatement, updateInputHighlight } from "./sql-highlight";
import { dispatchChatRequest } from "./sse-stream";
import {
  buildChatPath,
  ChatState,
  extractChatIdFromLocation,
  safeStorageGet,
  safeStorageSet,
} from "./state";

/**
 * Optional startup configuration for `ChatUI`.
 */
export interface ChatUIOptions {
  /** Whether to trigger an initial `/api/status` health check on construction. */
  checkStatusOnInit?: boolean;
  /** Optional background `/api/status` polling interval in milliseconds. */
  statusPollIntervalMs?: number;
}

/**
 * Lean application coordinator wiring `ChatState`, `TypedEventTarget<AppEvents>`,
 * and composable UI sub-controllers without circular host interfaces or module singletons.
 */
export class ChatUI {
  /** Global application state store. */
  public state: ChatState;
  /** Unidirectional domain event bus decoupling sub-controllers from `ChatUI`. */
  public readonly events: AppEventBus;
  /** AbortController for global document/window/bus listeners bound by this `ChatUI` instance. */
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
  /** Sub-controller for the Persistent Prompt & SQL Template Library modal. */
  public readonly templatePickerController: TemplatePickerController;
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
  /** Optional button element to cancel an in-flight chat generation request. */
  public readonly stopBtn: HTMLButtonElement | null;
  /** Optional button element to toggle automatic DuckDB execution of LLM-generated SQL. */
  public readonly autoExecuteToggleBtn: HTMLButtonElement | null;
  /** Active AbortController for an in-flight chat request, or null when idle. */
  private activeRequestController: AbortController | null = null;
  /** Active history navigation index in `getTypedMessageHistory()`, or null when not browsing history. */
  public inputHistoryIndex: number | null = null;
  /** Saved draft text that was in `chatInput` prior to starting `ArrowUp` history navigation. */
  public savedInputDraft: string | null = null;
  /** Ordered list of user-submitted messages recorded during the current UI session. */
  public sessionTypedHistory: string[] = [];
  /** Dropdown select element for choosing the active LLM or SQL engine. */
  public readonly modelSelect: HTMLSelectElement;
  /** Button element to manually trigger server session synchronization. */
  public readonly syncSessionsBtn: HTMLButtonElement;
  /** Button element to toggle real-time SSE token streaming. */
  public readonly streamingToggleBtn: HTMLButtonElement;
  /** Whether real-time SSE token streaming is enabled in the UI. */
  public streamingEnabled: boolean = false;
  /** Whether to dispatch via `POST /api/chat/jobs` before `/api/chat`. */
  public asyncJobPollingEnabled: boolean = false;
  /** Polling interval in milliseconds when waiting on a pending chat job. */
  public pollIntervalMs: number = 800;
  /** Dropdown select element for switching the UI language/locale. */
  public readonly langSelect: HTMLSelectElement;
  /** Button element to toggle between dark mode and light mode. */
  public readonly themeToggleBtn: HTMLButtonElement;

  /**
   * Initializes the `ChatUI` coordinator and its composable sub-controllers.
   * @param {ChatState} state - The global state store.
   * @param {ChatUIOptions} [options] - Optional startup health-check and polling settings.
   */
  constructor(state: ChatState, options?: ChatUIOptions) {
    this.globalListenerController = new AbortController();
    this.events = new TypedEventTarget<AppEvents>();
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
    this.stopBtn = optionalElementById("stop-btn", HTMLButtonElement);
    this.autoExecuteToggleBtn = optionalElementById(
      "auto-execute-toggle-btn",
      HTMLButtonElement,
    );
    this.syncSessionsBtn =
      requireElementById<HTMLButtonElement>("sync-sessions-btn");
    this.streamingToggleBtn = requireElementById<HTMLButtonElement>(
      "streaming-toggle-btn",
    );
    this.langSelect = requireElementById<HTMLSelectElement>("lang-select");
    this.themeToggleBtn =
      requireElementById<HTMLButtonElement>("theme-toggle-btn");

    this.wireEventBusSubscriptions();

    this.modelSelectorController = new ModelSelectorController({
      state: this.state,
      events: this.events,
    });
    this.modelSelect = this.modelSelectorController.modelSelect;
    this.tableModalController = new TablePreviewModalController({
      state: this.state,
      events: this.events,
    });
    this.cohortModalController = new CohortBuilderModalController({
      state: this.state,
      events: this.events,
    });
    this.providerModalController = new ProviderSettingsModalController({
      state: this.state,
      events: this.events,
      modelSelect: this.modelSelect,
      reloadModels: () => this.modelSelectorController.loadModels(),
    });
    this.systemStatusController = new SystemStatusController({
      state: this.state,
      events: this.events,
      modelSelect: this.modelSelect,
    });
    this.schemaExplorerController = new SchemaExplorerController({
      state: this.state,
      events: this.events,
      chatInput: this.chatInput,
    });
    this.sidebarController = new SidebarController({
      state: this.state,
      events: this.events,
      chatInput: this.chatInput,
      signal: this.globalListenerController.signal,
    });
    this.templatePickerController = new TemplatePickerController({
      state: this.state,
      events: this.events,
      modelSelect: this.modelSelect,
      onSendBenchmark: (question, model, goldSql) =>
        this.handleSendMessage(question, model, goldSql),
    });
    this.messageRenderer = new MessageRenderer({
      state: this.state,
      events: this.events,
      messagesContainer: this.messagesContainer,
      chatInputWrapper: this.chatInputWrapper,
      chatInput: this.chatInput,
      sendBtn: this.sendBtn,
      modelSelect: this.modelSelect,
      onOpenProviderSettings: () => {
        this.events.emit("openProviderSettings", undefined);
      },
      onOpenTableExplorer: (tableName) => {
        void this.tableModalController.open(tableName);
      },
      onBindDiagnosticButtons: (container) => {
        this.schemaExplorerController.bindSchemaDiagnosticButtons(container);
      },
    });

    const savedTheme = safeStorageGet("t1d_theme");
    if (savedTheme === "light") {
      document.body.classList.add("light-mode");
    } else if (savedTheme === "dark") {
      document.body.classList.remove("light-mode");
    }

    this.streamingEnabled = this.state.streamingMode;
    this.updateStreamingToggleUi();
    this.updateAutoExecuteToggleUi();
    requireElementById<HTMLElement>("build-timestamp").textContent =
      `built on ${getBuildTimestampUtc()}`;

    const initialChatId = extractChatIdFromLocation(window.location);
    if (initialChatId) {
      if (this.state.chats.some((c) => c.id === initialChatId)) {
        this.state.setActiveChat(initialChatId);
      } else {
        void this.hydrateSharedChat(initialChatId);
      }
    }

    this.bindEvents();
    if (options?.checkStatusOnInit === true) {
      void this.systemStatusController.check();
    }
    if (
      options?.statusPollIntervalMs !== undefined &&
      options.statusPollIntervalMs > 0
    ) {
      this.systemStatusController.startPolling(options.statusPollIntervalMs);
    }
    void this.modelSelectorController.loadModels();
    void this.schemaExplorerController.loadSchema();
    this.render();

    /**
     * Synchronizes active chat state with browser history `popstate` / `hashchange` events.
     * @returns {void}
     */
    const onRouteChange = (): void => {
      this.handleLocationChange();
    };
    window.addEventListener("popstate", onRouteChange, {
      signal: this.globalListenerController.signal,
    });
    window.addEventListener("hashchange", onRouteChange, {
      signal: this.globalListenerController.signal,
    });
  }

  /**
   * Wires unidirectional domain event bus subscriptions owned by the coordinator.
   * @returns {void}
   */
  private wireEventBusSubscriptions(): void {
    const signal = this.globalListenerController.signal;

    this.events.on(
      "announce",
      ({ message, isAssertive }) => {
        this.announce(message, Boolean(isAssertive));
      },
      signal,
    );

    this.events.on(
      "renderRequested",
      () => {
        this.render();
      },
      signal,
    );

    this.events.on(
      "composerSyncRequested",
      ({ focus }) => {
        if (focus === true) {
          this.chatInput.focus();
        }
        this.syncHighlight();
      },
      signal,
    );

    this.events.on(
      "cancelActiveRequest",
      () => {
        this.cancelActiveRequest();
      },
      signal,
    );

    this.events.on(
      "templateSelected",
      (selection) => {
        if (selection.targetModel !== "keep") {
          this.modelSelect.value = selection.targetModel;
          this.state.setActiveChatModel(selection.targetModel);
        }
        this.state.setActiveGoldSql(selection.goldSql);
        if (selection.autoSend === true) {
          void this.handleSendMessage(
            selection.promptText,
            this.modelSelect.value,
            selection.goldSql ?? undefined,
          );
          return;
        }
        this.chatInput.value = selection.promptText;
        this.chatInput.focus();
        this.syncHighlight();
      },
      signal,
    );
  }

  /**
   * Handles browser URL changes (`popstate` or legacy `hashchange`), activating the target chat
   * locally or hydrating it from `/api/sessions/{id}`.
   * @returns {void}
   */
  public handleLocationChange(): void {
    const routeChatId = extractChatIdFromLocation(window.location);
    if (routeChatId && routeChatId !== this.state.activeChatId) {
      this.cancelActiveRequest();
      if (this.state.chats.some((c) => c.id === routeChatId)) {
        this.state.setActiveChat(routeChatId);
        this.render();
        this.chatInput.focus();
      } else {
        void this.hydrateSharedChat(routeChatId);
      }
    }
  }

  /**
   * Fetches and activates a shared chat session by ID from the backend if not yet in local state.
   * @param {string} chatId - Unique chat session identifier from a `/chats/${ID}` URL.
   * @returns {Promise<void>}
   */
  public async hydrateSharedChat(chatId: string): Promise<void> {
    const loaded = await this.state.loadSharedSession(chatId);
    if (loaded) {
      this.render();
      this.chatInput.focus();
    }
  }

  /**
   * Disposes global event listeners, modal listeners, sub-controllers, and active status polling timers.
   * @returns {void}
   */
  public dispose(): void {
    this.cancelActiveRequest();
    this.globalListenerController.abort();
    this.systemStatusController.dispose();
    this.modelSelectorController.dispose();
    this.schemaExplorerController.dispose();
    this.tableModalController.dispose();
    this.cohortModalController.dispose();
    this.providerModalController.dispose();
    this.templatePickerController.dispose();
  }

  /**
   * Cancels any currently in-flight chat generation request.
   * @returns {void}
   */
  public cancelActiveRequest(): void {
    if (this.activeRequestController) {
      this.activeRequestController.abort();
      this.activeRequestController = null;
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
   * Updates the visual indicator and accessibility attributes on the Auto-Run SQL toggle button.
   * @returns {void}
   */
  public updateAutoExecuteToggleUi(): void {
    if (!this.autoExecuteToggleBtn) return;
    const isAuto = this.state.autoExecuteSql;
    const label = isAuto
      ? i18next.t("ui.autoExecuteOn", "Auto-Run SQL: ON (Read-Only DuckDB)")
      : i18next.t("ui.autoExecuteOff", "Auto-Run SQL: OFF (Manual Play)");
    this.autoExecuteToggleBtn.classList.toggle("active", isAuto);
    this.autoExecuteToggleBtn.title = label;
    this.autoExecuteToggleBtn.setAttribute("aria-label", label);
    this.autoExecuteToggleBtn.setAttribute(
      "aria-pressed",
      isAuto ? "true" : "false",
    );
  }

  /**
   * Binds form submission, streaming toggle, sync button, locale selector, and theme toggle events.
   * @returns {void}
   */
  private bindEvents(): void {
    const signal = this.globalListenerController.signal;

    this.streamingToggleBtn.addEventListener(
      "click",
      () => {
        this.streamingEnabled = !this.streamingEnabled;
        this.state.streamingMode = this.streamingEnabled;
        this.state.saveToLocalStorage();
        this.updateStreamingToggleUi();
      },
      { signal },
    );

    this.autoExecuteToggleBtn?.addEventListener(
      "click",
      () => {
        this.state.setAutoExecuteSql(!this.state.autoExecuteSql);
        this.updateAutoExecuteToggleUi();
      },
      { signal },
    );

    this.stopBtn?.addEventListener(
      "click",
      () => {
        this.cancelActiveRequest();
      },
      { signal },
    );

    this.chatForm.addEventListener(
      "submit",
      (e) => {
        e.preventDefault();
        void this.handleSendMessage();
      },
      { signal },
    );

    this.chatInput.addEventListener(
      "keydown",
      (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
          e.preventDefault();
          this.chatForm.dispatchEvent(
            new Event("submit", { cancelable: true, bubbles: true }),
          );
          return;
        }
        if (
          e.key === "ArrowUp" &&
          !e.shiftKey &&
          !e.altKey &&
          !e.ctrlKey &&
          !e.metaKey
        ) {
          this.handleInputArrowUp(e);
          return;
        }
        if (
          e.key === "ArrowDown" &&
          !e.shiftKey &&
          !e.altKey &&
          !e.ctrlKey &&
          !e.metaKey
        ) {
          this.handleInputArrowDown(e);
        }
      },
      { signal },
    );

    this.chatInput.addEventListener(
      "input",
      () => {
        this.inputHistoryIndex = null;
        this.savedInputDraft = null;
        if (this.state.activeGoldSql !== null) {
          this.state.setActiveGoldSql(null);
        }
        this.syncHighlight();
      },
      { signal },
    );

    this.chatInput.addEventListener(
      "scroll",
      () => {
        const highlightPre = this.chatInputHighlight.parentElement;
        if (highlightPre) {
          highlightPre.scrollTop = this.chatInput.scrollTop;
        }
      },
      { signal },
    );

    this.syncSessionsBtn.addEventListener(
      "click",
      () => {
        void this.syncAllChatsWithServer();
      },
      { signal },
    );

    this.langSelect.value = i18next.language;
    this.langSelect.addEventListener(
      "change",
      () => {
        void (async (): Promise<void> => {
          await setLanguage(this.langSelect.value);
          this.updateThemeButtonLabel();
          this.updateStreamingToggleUi();
          this.updateAutoExecuteToggleUi();
          this.providerModalController.updateProviderBadgeStatus();
          this.schemaExplorerController.renderDatabaseDropdown();
          this.providerModalController.updateModalBadgesFromLocal();
          await this.tableModalController.refreshIfOpen();
          void this.systemStatusController.check();
          this.render();
        })();
      },
      { signal },
    );

    this.updateThemeButtonLabel();
    this.themeToggleBtn.addEventListener(
      "click",
      () => {
        const isLight = document.body.classList.toggle("light-mode");
        safeStorageSet("t1d_theme", isLight ? "light" : "dark");
        this.updateThemeButtonLabel();
      },
      { signal },
    );
  }

  /**
   * Collects all user messages from the active chat and session history, deduplicating entries.
   * @returns {string[]} Ordered list of user-typed messages.
   */
  public getTypedMessageHistory(): string[] {
    const combined: string[] = [];
    const activeMessages = this.state.getActiveChat()?.messages ?? [];
    for (const msg of activeMessages) {
      if (msg.role === "user" && msg.content.trim().length > 0) {
        if (
          combined.length === 0 ||
          combined[combined.length - 1] !== msg.content
        ) {
          combined.push(msg.content);
        }
      }
    }
    for (const item of this.sessionTypedHistory) {
      if (!combined.includes(item)) {
        combined.push(item);
      }
    }
    return combined;
  }

  /**
   * Checks whether the textarea caret is collapsed at the very top line of the input box.
   * @param {HTMLTextAreaElement} textarea - Target textarea element.
   * @returns {boolean} True when the caret is at the very top line.
   */
  public isCaretAtVeryTop(textarea: HTMLTextAreaElement): boolean {
    return (
      textarea.selectionStart === textarea.selectionEnd &&
      !textarea.value.slice(0, textarea.selectionStart).includes(`\n`)
    );
  }

  /**
   * Checks whether the textarea caret is collapsed at the very bottom line of the input box.
   * @param {HTMLTextAreaElement} textarea - Target textarea element.
   * @returns {boolean} True when the caret is at the very bottom line.
   */
  public isCaretAtVeryBottom(textarea: HTMLTextAreaElement): boolean {
    return (
      textarea.selectionStart === textarea.selectionEnd &&
      !textarea.value.slice(textarea.selectionEnd).includes(`\n`)
    );
  }

  /**
   * Handles `ArrowUp` key navigation at the very top of `#chat-input` to recall previous messages.
   * @param {KeyboardEvent} e - Keyboard event triggered on `chatInput`.
   * @returns {void}
   */
  private handleInputArrowUp(e: KeyboardEvent): void {
    if (!this.isCaretAtVeryTop(this.chatInput)) return;
    const history = this.getTypedMessageHistory();
    if (history.length === 0) return;

    if (this.inputHistoryIndex === null) {
      this.savedInputDraft = this.chatInput.value;
      this.inputHistoryIndex = history.length - 1;
    } else if (this.inputHistoryIndex > 0) {
      this.inputHistoryIndex--;
    }
    e.preventDefault();
    this.chatInput.value = String(history[this.inputHistoryIndex]);
    this.chatInput.setSelectionRange(0, 0);
    this.syncHighlight();
  }

  /**
   * Handles `ArrowDown` key navigation at the very bottom of `#chat-input` to step forward or restore a prior draft.
   * @param {KeyboardEvent} e - Keyboard event triggered on `chatInput`.
   * @returns {void}
   */
  private handleInputArrowDown(e: KeyboardEvent): void {
    if (
      this.inputHistoryIndex === null ||
      !this.isCaretAtVeryBottom(this.chatInput)
    ) {
      return;
    }
    const history = this.getTypedMessageHistory();
    if (this.inputHistoryIndex < history.length - 1) {
      this.inputHistoryIndex++;
      e.preventDefault();
      this.chatInput.value = String(history[this.inputHistoryIndex]);
      const len = this.chatInput.value.length;
      this.chatInput.setSelectionRange(len, len);
      this.syncHighlight();
      return;
    }
    if (this.savedInputDraft !== null && this.savedInputDraft.length > 0) {
      e.preventDefault();
      this.chatInput.value = this.savedInputDraft;
      const len = this.chatInput.value.length;
      this.chatInput.setSelectionRange(len, len);
      this.inputHistoryIndex = null;
      this.savedInputDraft = null;
      this.syncHighlight();
    } else {
      this.inputHistoryIndex = null;
      this.savedInputDraft = null;
    }
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
   * Toggles composer input/button disabled states while a chat request is in flight.
   * @param {boolean} busy - True to disable inputs and show `#stop-btn`, false to re-enable.
   * @returns {void}
   */
  private setComposerBusyState(busy: boolean): void {
    this.chatInput.disabled = busy;
    this.chatInputWrapper.classList.toggle("disabled", busy);
    this.sendBtn.disabled = busy;
    this.modelSelect.disabled = busy;
    if (busy) {
      this.stopBtn?.classList.remove("hidden");
    } else {
      this.stopBtn?.classList.add("hidden");
      this.chatInput.focus();
    }
  }

  /**
   * Resolves provider and Bearer authentication headers for a chat turn.
   * @param {string | undefined} provider - Resolved provider identifier.
   * @returns {Record<string, string>} HTTP request header map.
   */
  private buildChatAuthHeaders(
    provider: string | undefined,
  ): Record<string, string> {
    const apiKey = provider ? this.state.getApiKey(provider) : undefined;
    const backendToken = this.state.credentialsStore.getBackendApiKey();
    return {
      "Content-Type": "application/json",
      ...(backendToken ? { Authorization: `Bearer ${backendToken}` } : {}),
      ...(provider ? { "x-provider": provider } : {}),
      ...(apiKey ? { "x-provider-api-key": apiKey } : {}),
    };
  }

  /**
   * Builds HTTP request headers and JSON body for a `/api/chat` or `/api/chat/stream` turn.
   * @param {string} content - User prompt or SQL statement.
   * @param {string} model - Target model identifier.
   * @param {string | undefined} effectiveGoldSql - Optional gold-standard SQL reference.
   * @returns {{ requestHeaders: Record<string, string>; bodyJson: string }} Serialized headers and payload.
   */
  private buildChatRequestEnvelope(
    content: string,
    model: string,
    effectiveGoldSql: string | undefined,
  ): { requestHeaders: Record<string, string>; bodyJson: string } {
    const selectedOption = this.modelSelect.selectedOptions?.[0];
    const provider =
      selectedOption?.getAttribute("data-provider") ||
      (model.includes("/") ? model.split("/")[0] : undefined);
    const requestHeaders = this.buildChatAuthHeaders(provider);

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
      auto_execute: model === "sql" || this.state.autoExecuteSql,
      ...(effectiveGoldSql ? { gold_sql: effectiveGoldSql } : {}),
    });

    return { requestHeaders, bodyJson };
  }

  /**
   * Updates synthetic-data and rule-based SQL alert banners after a completed chat turn.
   * @param {boolean | undefined} isSyntheticData - Synthetic seed flag from backend.
   * @param {boolean | undefined} isRuleBasedSql - Rule-based SQL synthesis flag from backend.
   * @param {string} model - Executed model identifier.
   * @returns {void}
   */
  private applyTurnProvenanceAlerts(
    isSyntheticData: boolean | undefined,
    isRuleBasedSql: boolean | undefined,
    model: string,
  ): void {
    if (isSyntheticData === true) {
      this.systemStatusController.showSyntheticDataAlert();
    } else if (isSyntheticData === false) {
      this.systemStatusController.hideSyntheticDataAlert();
    }
    if (isRuleBasedSql === true) {
      this.systemStatusController.showRuleBasedSqlAlert();
    } else if (isRuleBasedSql === false && model !== "sql") {
      this.systemStatusController.hideRuleBasedSqlAlert();
    }
  }

  /**
   * Resolves the effective model and gold-SQL reference for an outgoing turn and appends the user message.
   * @param {string} content - Non-empty prompt or SQL statement.
   * @param {boolean} clearComposer - Whether to clear `chatInput` after recording the user turn.
   * @param {string} [modelOverride] - Optional explicit model override.
   * @param {string} [goldSqlOverride] - Optional explicit gold-SQL override.
   * @returns {{ model: string; effectiveGoldSql: string | undefined }} Resolved model and gold SQL.
   */
  private recordOutgoingUserTurn(
    content: string,
    clearComposer: boolean,
    modelOverride?: string,
    goldSqlOverride?: string,
  ): { model: string; effectiveGoldSql: string | undefined } {
    const model =
      modelOverride ??
      (this.modelSelect.value === "sql" || isReadOnlySqlStatement(content)
        ? "sql"
        : this.modelSelect.value);
    const effectiveGoldSql =
      goldSqlOverride ??
      (model !== "sql" ? (this.state.activeGoldSql ?? undefined) : undefined);
    this.state.setActiveGoldSql(null);

    this.state.addMessageToActiveChat({
      role: "user",
      content,
      model,
      ...(effectiveGoldSql ? { goldSql: effectiveGoldSql } : {}),
    });
    if (
      this.sessionTypedHistory[this.sessionTypedHistory.length - 1] !== content
    ) {
      this.sessionTypedHistory.push(content);
    }
    this.inputHistoryIndex = null;
    this.savedInputDraft = null;

    if (clearComposer) {
      this.state.setActiveChatModel(this.modelSelect.value);
      this.chatInput.value = "";
      this.syncHighlight();
    }
    return { model, effectiveGoldSql };
  }

  /**
   * Handles sending a user message and processing the streaming or batch response.
   * @param {string} [text] - Optional message text override (used when executing SQL directly).
   * @param {string} [modelOverride] - Optional model identifier override (e.g. 'sql').
   * @param {string} [goldSqlOverride] - Optional gold-standard SQL reference override for NL-to-SQL benchmarking.
   * @returns {Promise<void>}
   */
  public async handleSendMessage(
    text?: string,
    modelOverride?: string,
    goldSqlOverride?: string,
  ): Promise<void> {
    const content = text ?? this.chatInput.value.trim();
    if (!content) return;

    const { model, effectiveGoldSql } = this.recordOutgoingUserTurn(
      content,
      text === undefined,
      modelOverride,
      goldSqlOverride,
    );

    this.sidebarController.renderSidebar();
    this.messageRenderer.renderMessages();
    this.scrollToBottom();
    this.setComposerBusyState(true);

    const loadingDiv = this.messageRenderer.createLoadingIndicator();
    this.messagesContainer.appendChild(loadingDiv);
    this.scrollToBottom();

    const { requestHeaders, bodyJson } = this.buildChatRequestEnvelope(
      content,
      model,
      effectiveGoldSql,
    );

    this.activeRequestController = new AbortController();
    try {
      const outcome = await dispatchChatRequest({
        bodyJson,
        headers: requestHeaders,
        streamingEnabled: this.streamingEnabled,
        loadingDiv,
        onTokenScroll: () => this.scrollToBottom(),
        signal: this.activeRequestController.signal,
        useAsyncJobEndpoint: this.asyncJobPollingEnabled && model !== "sql",
        pollIntervalMs: this.pollIntervalMs,
      });
      loadingDiv.remove();
      this.applyTurnProvenanceAlerts(
        outcome.data.is_synthetic_data,
        outcome.data.is_rule_based_sql,
        model,
      );
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
    } finally {
      this.activeRequestController = null;
      this.setComposerBusyState(false);
    }

    if (this.state.serverSyncEnabled) {
      void this.syncAllChatsWithServer();
    }

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
        this.render();
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
    const activeChat = this.state.getActiveChat();
    const targetPath = buildChatPath(activeChat ? activeChat.id : null);
    if (
      window.location.pathname !== targetPath ||
      window.location.hash !== ""
    ) {
      window.history.replaceState(null, "", targetPath);
    }

    const appTitle = i18next.t("app.title");
    document.title = activeChat
      ? `${activeChat.title} - ${appTitle}`
      : appTitle;

    this.sidebarController.renderSidebar();
    this.messageRenderer.renderActiveChat();
  }
}
