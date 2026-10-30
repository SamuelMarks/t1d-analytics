/**
 * @file ui.ts
 * @module ui
 * Main Chat UI coordinator delegating modal, schema, status, and message rendering
 * operations to composable sub-controllers.
 */

import "highlight.js/styles/github-dark.css";
import { ChatState, type Message } from "./state";
import i18next, { setLanguage } from "./i18n";
import {
  type SqlCellValue,
  classifySqlResult,
  getBuildTimestampUtc,
} from "./formatters";
import {
  type ExportableCellValue,
  exportRowsToCsv as exportRowsToCsvHelper,
  exportRowsToJson as exportRowsToJsonHelper,
} from "./exporters";
import { consumeSseChatStream, type SseFinalChatData } from "./sse-stream";
import {
  type ApiErrorInput,
  fetchWithBackendError,
  parseApiError,
  parseResponseJson,
} from "./api-client";
import {
  type CohortFilterParams,
  buildCohortSql,
  normalizeJoinType,
} from "./cohort-sql";
import { trapModalFocus } from "./modal-utils";
import { isReadOnlySqlStatement, updateInputHighlight } from "./sql-highlight";
import {
  type DbStatusCode,
  type SystemStatusApiResponse,
  type SystemStatusDbPayload,
  type SystemStatusHost,
  SystemStatusController,
  isDbStatusCode,
} from "./controllers/system-status-controller";
import {
  type CohortBuilderHost,
  CohortBuilderModalController,
} from "./controllers/cohort-modal-controller";
import {
  type ProviderSettingsHost,
  type ProvidersStatusPayload,
  ProviderSettingsModalController,
} from "./controllers/provider-modal-controller";
import {
  type TableDataPayload,
  type TablePreviewHost,
  TablePreviewModalController,
} from "./controllers/table-modal-controller";
import {
  type DatabasesListApiResponse,
  type SchemaExplorerApiResponse,
  type SchemaExplorerHost,
  SchemaExplorerController,
} from "./controllers/schema-explorer-controller";
import {
  type MessageRendererHost,
  type SchemaDiagnosticCardOptions,
  type SvgChildSpec,
  MessageRenderer,
  createSchemaDiagnosticCard,
  createSvgIcon,
} from "./message-renderer";

export {
  type CohortFilterParams,
  buildCohortSql,
  isReadOnlySqlStatement,
  normalizeJoinType,
  parseResponseJson,
  trapModalFocus,
  updateInputHighlight,
  type DbStatusCode,
  type SystemStatusHost,
  SystemStatusController,
  isDbStatusCode,
  type CohortBuilderHost,
  CohortBuilderModalController,
  type ProviderSettingsHost,
  ProviderSettingsModalController,
  type TablePreviewHost,
  TablePreviewModalController,
  type SchemaExplorerHost,
  SchemaExplorerController,
  type MessageRendererHost,
  type SchemaDiagnosticCardOptions,
  type SvgChildSpec,
  MessageRenderer,
  createSchemaDiagnosticCard,
  createSvgIcon,
};

/** Alias for database health diagnostics payload. */
export type DatabaseHealthApiPayload = SystemStatusDbPayload;

/** Alias for `/api/status` response payload. */
export type HealthApiResponse = SystemStatusApiResponse;

/** Alias for `/api/schema` response payload. */
export type SchemaApiResponse = SchemaExplorerApiResponse;

/** Alias for `/api/databases` response payload. */
export type DatabasesApiResponse = DatabasesListApiResponse;

/** Alias for `/api/providers/status` response payload. */
export type ProvidersStatusApiResponse = ProvidersStatusPayload;

/** Alias for `/api/table/{table_name}` response payload. */
export type TableDataApiResponse = TableDataPayload;

/**
 * Model metadata entry returned by `/api/models`.
 */
export interface ModelApiItem {
  /** Model identifier string. */
  name: string;
  /** Model size in bytes (for local Ollama models). */
  size?: number;
  /** Provider identifier ('ollama', 'openai', 'anthropic', 'google', etc.). */
  provider?: string;
  /** Whether credentials are configured on the server. */
  configured?: boolean;
  /** Whether an API key is required. */
  requires_key?: boolean;
  /** Whether the model is available for selection. */
  available?: boolean;
  /** Whether the provider is reachable. */
  reachable?: boolean;
  /** Optional error code when unconfigured or offline. */
  error_code?: string;
}

/**
 * Response payload returned by `/api/models`.
 */
export interface ModelsApiResponse {
  /** Available models across local and cloud providers. */
  models?: ModelApiItem[];
}

/**
 * Response payload returned by `/api/chat` and `/api/execute-sql`.
 */
export interface ChatApiResponse {
  /** Assistant response text or translation key. */
  content?: string;
  /** Generated or executed SQL query string. */
  sqlQuery?: string;
  /** Tabular SQL query result rows. */
  sqlResult?: Record<string, SqlCellValue>[];
  /** Structured or string API error payload. */
  error?: ApiErrorInput;
}

/**
 * Looks up a typed element by ID using `document.querySelector<T>("#" + id)`.
 * @template T - Target HTML element subtype.
 * @param {string} id - The DOM element ID.
 * @returns {T | null} The matched element or null.
 */
function queryElementById<T extends HTMLElement>(id: string): T | null {
  return document.querySelector<T>(`#${id}`);
}

/**
 * Looks up a required typed element by ID and throws a descriptive Error if missing.
 * @template T - Target HTML element subtype.
 * @param {string} id - The DOM element ID.
 * @returns {T} The matched element.
 * @throws {Error} When the required mount element is not found in the document.
 */
export function requireElementById<T extends HTMLElement>(id: string): T {
  const el = document.querySelector<T>(`#${id}`);
  if (!el) {
    throw new Error(`Required DOM element #${id} not found`);
  }
  return el;
}

/** Active ChatUI instance so re-instantiation in tests automatically disposes previous listeners/timers. */
let activeChatUIInstance: ChatUI | null = null;

/**
 * UI controller class that binds the state to the DOM elements and coordinates sub-controllers.
 */
export class ChatUI
  implements
    CohortBuilderHost,
    ProviderSettingsHost,
    SystemStatusHost,
    TablePreviewHost,
    SchemaExplorerHost,
    MessageRendererHost
{
  /** Global application state store. */
  public state: ChatState;
  /** AbortController for global document/window listeners bound by this ChatUI instance. */
  private globalListenerController: AbortController;
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
  /** Renderer for chat messages, SQL blocks, and CGM analytics cards. */
  public readonly messageRenderer: MessageRenderer;

  // DOM Elements
  /** Sidebar list element containing chat sessions. */
  public chatListEl: HTMLElement;
  /** Button element to create a new chat session. */
  public newChatBtn: HTMLButtonElement;
  /** Scrollable container element holding chat messages. */
  public messagesContainer: HTMLElement;
  /** Form element wrapping the chat input and send button. */
  public chatForm: HTMLFormElement;
  /** Wrapper element for the syntax-highlighted textarea input. */
  public chatInputWrapper: HTMLElement;
  /** Code element displaying live syntax highlighting behind the textarea. */
  public chatInputHighlight: HTMLElement;
  /** Textarea element where the user types queries or SQL. */
  public chatInput: HTMLTextAreaElement;
  /** Submit button element for sending messages. */
  public sendBtn: HTMLButtonElement;
  /** Dropdown select element for choosing the active LLM or SQL engine. */
  public modelSelect: HTMLSelectElement;
  /** Dropdown select element for choosing the active DuckDB database file. */
  public dbSelect: HTMLSelectElement | null;
  /** Button element to manually trigger server session synchronization. */
  public syncSessionsBtn: HTMLButtonElement | null;
  /** Button element to toggle real-time SSE token streaming. */
  public streamingToggleBtn: HTMLButtonElement | null;
  /** Whether real-time SSE token streaming is enabled in the UI. */
  public streamingEnabled: boolean = false;
  /** Dropdown select element for switching the UI language/locale. */
  public langSelect: HTMLSelectElement;
  /** Mobile header button element to open the navigation sidebar. */
  public openSidebarBtn: HTMLButtonElement;
  /** Mobile sidebar button element to close the navigation sidebar. */
  public closeSidebarBtn: HTMLButtonElement;
  /** Button element to toggle between dark mode and light mode. */
  public themeToggleBtn: HTMLButtonElement;
  /** Backdrop overlay element for the mobile navigation sidebar. */
  public overlay: HTMLElement;

  // Diagnostic and status elements
  /** Header status chip button displaying system health. */
  public systemStatusChip: HTMLButtonElement | null;
  /** Text span inside the system status chip. */
  public statusChipText: HTMLElement | null;
  /** Alert banner element for displaying database or backend warnings. */
  public systemStatusBanner: HTMLElement | null;
  /** Heading element inside the system status banner. */
  public bannerTitle: HTMLElement | null;
  /** Description paragraph inside the system status banner. */
  public bannerDesc: HTMLElement | null;
  /** Icon span inside the system status banner. */
  public bannerIcon: HTMLElement | null;
  /** Retry connection button inside the system status banner. */
  public bannerRetryBtn: HTMLButtonElement | null;
  /** Button to toggle setup instructions inside the system status banner. */
  public bannerInstructionsBtn: HTMLButtonElement | null;
  /** Collapsible drawer containing CLI setup instructions. */
  public bannerInstructionsDrawer: HTMLElement | null;
  /** Button to dismiss the system status alert banner. */
  public bannerDismissBtn: HTMLButtonElement | null;

  // Cohort filter elements
  /** Header button to open the Visual Cohort Query Builder modal. */
  public cohortFilterBtn: HTMLButtonElement | null;
  /** Modal dialog container for the Visual Cohort Query Builder. */
  public cohortModal: HTMLElement | null;
  /** Button to close the Visual Cohort Query Builder modal. */
  public closeCohortModalBtn: HTMLButtonElement | null;
  /** Select dropdown for the primary cohort table. */
  public cohortTableSelect: HTMLSelectElement | null;
  /** Select dropdown for the optional joined secondary table. */
  public cohortJoinTableSelect: HTMLSelectElement | null;
  /** Select dropdown for the SQL join type (INNER JOIN or LEFT JOIN). */
  public cohortJoinTypeSelect: HTMLSelectElement | null;
  /** Input field for the SQL join ON condition. */
  public cohortJoinOnInput: HTMLInputElement | null;
  /** Number input for minimum patient age filter. */
  public cohortMinAge: HTMLInputElement | null;
  /** Number input for maximum patient age filter. */
  public cohortMaxAge: HTMLInputElement | null;
  /** Select dropdown for biological sex/gender filter. */
  public cohortGenderSelect: HTMLSelectElement | null;
  /** Text input for clinical treatment group filter. */
  public cohortTxGroup: HTMLInputElement | null;
  /** Number input for maximum Time-in-Range (TIR) threshold. */
  public cohortMaxTIR: HTMLInputElement | null;
  /** Number input for minimum HbA1c threshold. */
  public cohortMinHbA1c: HTMLInputElement | null;
  /** Preformatted text element showing the live generated DuckDB SQL. */
  public cohortSqlPreview: HTMLElement | null;
  /** Button to insert the generated cohort SQL into the chat input. */
  public cohortInsertChatBtn: HTMLButtonElement | null;
  /** Button to immediately execute the generated cohort SQL. */
  public cohortExecuteBtn: HTMLButtonElement | null;

  // Provider settings elements
  /** Header button to open the Cloud LLM Providers & API Keys modal. */
  public providerSettingsBtn: HTMLButtonElement | null;
  /** Indicator dot displaying the configuration status of the active provider. */
  public providerStatusDot: HTMLElement | null;
  /** Modal dialog container for Cloud LLM Providers & API Keys settings. */
  public providerModal: HTMLElement | null;
  /** Button to close the Cloud LLM Providers modal. */
  public closeProviderModalBtn: HTMLButtonElement | null;
  /** Password input for the OpenAI API key. */
  public providerKeyOpenAI: HTMLInputElement | null;
  /** Password input for the Anthropic API key. */
  public providerKeyAnthropic: HTMLInputElement | null;
  /** Password input for the Google Gemini API key. */
  public providerKeyGoogle: HTMLInputElement | null;
  /** Checkbox input for user consent to store API keys in localStorage. */
  public providerStoreConsent: HTMLInputElement | null;
  /** Button to save configured provider API keys. */
  public providerSaveBtn: HTMLButtonElement | null;
  /** Button to clear all stored provider API keys. */
  public providerClearBtn: HTMLButtonElement | null;

  // Modal accessibility focus tracking and listeners
  /** Element that held focus prior to opening the cohort modal. */
  public previousFocusCohort: Element | null = null;
  /** Element that held focus prior to opening the provider settings modal. */
  public previousFocusProvider: Element | null = null;
  /** Keydown listener for focus trapping inside the cohort modal. */
  public onCohortModalKeyDown: ((e: KeyboardEvent) => void) | null = null;
  /** Keydown listener for focus trapping inside the provider modal. */
  public onProviderModalKeyDown: ((e: KeyboardEvent) => void) | null = null;

  // Table Preview Modal State
  /** Name of the database table currently displayed in the table preview modal. */
  public currentTable: string = "";
  /** Current 1-based page number in the table preview modal. */
  public currentPage: number = 1;
  /** Number of rows fetched per page in the table preview modal. */
  public readonly ROWS_PER_PAGE = 25;
  /** Array of row objects currently rendered in the table preview modal. */
  public currentModalRows: Array<Record<string, SqlCellValue>> = [];
  /** Column name currently used to sort the table preview modal, or empty string for default. */
  public currentSortBy: string = "";
  /** Active sort direction ('asc' or 'desc') in the table preview modal. */
  public currentSortOrder: "asc" | "desc" = "asc";
  /** Active search filter term in the table preview modal. */
  public currentSearchTerm: string = "";

  /**
   * Initializes the ChatUI.
   * @param {ChatState} state The global state object.
   */
  constructor(state: ChatState) {
    if (activeChatUIInstance) {
      activeChatUIInstance.dispose();
    }
    this.globalListenerController = new AbortController();
    activeChatUIInstance = this;
    this.state = state;
    this.tableModalController = new TablePreviewModalController(this);
    this.cohortModalController = new CohortBuilderModalController(this);
    this.providerModalController = new ProviderSettingsModalController(this);
    this.systemStatusController = new SystemStatusController(this);
    this.schemaExplorerController = new SchemaExplorerController(this);
    this.messageRenderer = new MessageRenderer(this);

    // Bind required DOM elements
    this.chatListEl = requireElementById<HTMLElement>("chat-list");
    this.newChatBtn = requireElementById<HTMLButtonElement>("new-chat-btn");
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
    this.modelSelect = requireElementById<HTMLSelectElement>("model-select");
    this.dbSelect = queryElementById<HTMLSelectElement>("db-select");
    this.syncSessionsBtn =
      queryElementById<HTMLButtonElement>("sync-sessions-btn");
    this.langSelect = requireElementById<HTMLSelectElement>("lang-select");

    this.openSidebarBtn =
      requireElementById<HTMLButtonElement>("open-sidebar-btn");
    this.closeSidebarBtn =
      requireElementById<HTMLButtonElement>("close-sidebar-btn");
    this.themeToggleBtn =
      requireElementById<HTMLButtonElement>("theme-toggle-btn");
    this.overlay = requireElementById<HTMLElement>("overlay");

    // Diagnostic and status elements
    this.systemStatusChip =
      queryElementById<HTMLButtonElement>("system-status-chip");
    this.statusChipText = document.getElementById("status-chip-text");
    this.systemStatusBanner = document.getElementById("system-status-banner");
    this.bannerTitle = document.getElementById("banner-title");
    this.bannerDesc = document.getElementById("banner-desc");
    this.bannerIcon = document.getElementById("banner-icon");
    this.bannerRetryBtn =
      queryElementById<HTMLButtonElement>("banner-retry-btn");
    this.bannerInstructionsBtn = queryElementById<HTMLButtonElement>(
      "banner-instructions-btn",
    );
    this.bannerInstructionsDrawer = document.getElementById(
      "banner-instructions-drawer",
    );
    this.bannerDismissBtn =
      queryElementById<HTMLButtonElement>("banner-dismiss-btn");

    // Cohort filter elements
    this.cohortFilterBtn =
      queryElementById<HTMLButtonElement>("cohort-filter-btn");
    this.cohortModal = document.getElementById("cohort-modal");
    this.closeCohortModalBtn = queryElementById<HTMLButtonElement>(
      "close-cohort-modal-btn",
    );
    this.cohortTableSelect = queryElementById<HTMLSelectElement>(
      "cohort-table-select",
    );
    this.cohortJoinTableSelect = queryElementById<HTMLSelectElement>(
      "cohort-join-table-select",
    );
    this.cohortJoinTypeSelect = queryElementById<HTMLSelectElement>(
      "cohort-join-type-select",
    );
    this.cohortJoinOnInput = queryElementById<HTMLInputElement>(
      "cohort-join-on-input",
    );
    this.cohortMinAge = queryElementById<HTMLInputElement>("cohort-min-age");
    this.cohortMaxAge = queryElementById<HTMLInputElement>("cohort-max-age");
    this.cohortGenderSelect = queryElementById<HTMLSelectElement>(
      "cohort-gender-select",
    );
    this.cohortTxGroup = queryElementById<HTMLInputElement>("cohort-txgroup");
    this.cohortMaxTIR = queryElementById<HTMLInputElement>("cohort-max-tir");
    this.cohortMinHbA1c =
      queryElementById<HTMLInputElement>("cohort-min-hba1c");
    this.cohortSqlPreview = document.getElementById("cohort-sql-preview");
    this.cohortInsertChatBtn = queryElementById<HTMLButtonElement>(
      "cohort-insert-chat-btn",
    );
    this.cohortExecuteBtn =
      queryElementById<HTMLButtonElement>("cohort-execute-btn");

    // Provider settings elements
    this.providerSettingsBtn = queryElementById<HTMLButtonElement>(
      "provider-settings-btn",
    );
    this.providerStatusDot = document.getElementById("provider-status-dot");
    this.providerModal = document.getElementById("provider-modal");
    this.closeProviderModalBtn = queryElementById<HTMLButtonElement>(
      "close-provider-modal-btn",
    );
    this.providerKeyOpenAI = queryElementById<HTMLInputElement>(
      "provider-key-openai",
    );
    this.providerKeyAnthropic = queryElementById<HTMLInputElement>(
      "provider-key-anthropic",
    );
    this.providerKeyGoogle = queryElementById<HTMLInputElement>(
      "provider-key-google",
    );
    this.providerStoreConsent = queryElementById<HTMLInputElement>(
      "provider-store-consent",
    );
    this.providerSaveBtn =
      queryElementById<HTMLButtonElement>("provider-save-btn");
    this.providerClearBtn =
      queryElementById<HTMLButtonElement>("provider-clear-btn");

    this.streamingToggleBtn = queryElementById<HTMLButtonElement>(
      "streaming-toggle-btn",
    );
    this.streamingEnabled = this.state.streamingMode;
    this.updateStreamingToggleUi();

    const buildTimestampEl = requireElementById<HTMLElement>("build-timestamp");
    buildTimestampEl.textContent = `built on ${getBuildTimestampUtc()}`;

    this.bindEvents();
    void this.loadModels();
    void this.loadSchema();
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

    if (window.location.hash) {
      const hashId = window.location.hash.slice(1);
      this.state.setActiveChat(hashId);
      this.render();
    }
  }

  /**
   * Starts periodic background polling of `/api/status`.
   * @param {number} [intervalMs=30000] - Polling interval in milliseconds.
   * @returns {void}
   */
  public startStatusPolling(intervalMs: number = 30000): void {
    this.systemStatusController.startPolling(intervalMs);
  }

  /**
   * Stops any active background system status polling interval.
   * @returns {void}
   */
  public stopStatusPolling(): void {
    this.systemStatusController.stopPolling();
  }

  /**
   * Disposes global event listeners, modal listeners, and active status polling timers.
   * @returns {void}
   */
  public dispose(): void {
    this.stopStatusPolling();
    this.globalListenerController.abort();
    this.tableModalController.abortListeners();
    if (this.cohortModal && this.onCohortModalKeyDown) {
      this.cohortModal.removeEventListener(
        "keydown",
        this.onCohortModalKeyDown,
      );
      this.onCohortModalKeyDown = null;
    }
    if (this.providerModal && this.onProviderModalKeyDown) {
      this.providerModal.removeEventListener(
        "keydown",
        this.onProviderModalKeyDown,
      );
      this.onProviderModalKeyDown = null;
    }
    if (activeChatUIInstance === this) {
      activeChatUIInstance = null;
    }
  }

  /**
   * Alias for `dispose()` to clean up intervals and global listeners.
   * @returns {void}
   */
  public destroy(): void {
    this.dispose();
  }

  /**
   * Shows the system status alert banner with the specified severity and messaging.
   * @param {"danger" | "warning" | "info"} severity - The alert level.
   * @param {string} title - The title to display.
   * @param {string} desc - The description message.
   * @param {string} [icon="⚠️"] - The icon to render.
   * @param {boolean} [showRetry=true] - Whether to show the retry button.
   * @returns {void}
   */
  public showBanner(
    severity: "danger" | "warning" | "info",
    title: string,
    desc: string,
    icon: string = "⚠️",
    showRetry: boolean = true,
  ): void {
    this.systemStatusController.showBanner(
      severity,
      title,
      desc,
      icon,
      showRetry,
    );
  }

  /**
   * Hides the system status alert banner.
   * @returns {void}
   */
  public hideBanner(): void {
    this.systemStatusController.hideBanner();
  }

  /**
   * Updates the status chip in the top header.
   * @param {"healthy" | "degraded" | "error" | "offline"} level - The status severity.
   * @param {string} text - The status text.
   * @returns {void}
   */
  public updateStatusChip(
    level: "healthy" | "degraded" | "error" | "offline",
    text: string,
  ): void {
    this.systemStatusController.updateStatusChip(level, text);
  }

  /**
   * Checks system health and updates UI indicators accordingly.
   * @param {boolean} [isManualRetry=false] - Whether this is a manual user retry.
   * @param {string} [dbPath] - Optional custom database path.
   * @returns {Promise<void>}
   */
  public async checkSystemStatus(
    isManualRetry: boolean = false,
    dbPath?: string,
  ): Promise<void> {
    await this.systemStatusController.check(isManualRetry, dbPath);
  }

  /**
   * Copies text to the user's clipboard and provides visual and screen reader feedback.
   * @param {string} cmd - The command string to copy.
   * @param {HTMLButtonElement} btn - The button providing visual feedback.
   * @returns {Promise<void>}
   */
  public async copyCommandToClipboard(
    cmd: string,
    btn: HTMLButtonElement,
  ): Promise<void> {
    if (!navigator?.clipboard) return;
    try {
      await navigator.clipboard.writeText(cmd);
      const copiedMsg = i18next.t("status.commandCopied");
      btn.textContent = copiedMsg;
      this.announce(copiedMsg);
      setTimeout(() => {
        btn.textContent = i18next.t("status.copyCommand");
      }, 2000);
    } catch {
      // Ignore clipboard write errors
    }
  }

  /**
   * Binds interaction events to action buttons inside diagnostic cards.
   * @param {HTMLElement} container - The container holding diagnostic cards.
   * @returns {void}
   */
  public bindSchemaDiagnosticButtons(container: HTMLElement): void {
    this.schemaExplorerController.bindSchemaDiagnosticButtons(container);
  }

  /**
   * Opens the table preview modal for a table.
   * @param {string} tableName - Table identifier to preview.
   * @returns {Promise<void>}
   */
  public async openTablePreview(tableName: string): Promise<void> {
    await this.tableModalController.open(tableName);
  }

  /**
   * Fetches the database schema and renders it in the sidebar.
   * @param {string} [dbPath] - Optional custom database path.
   * @returns {Promise<void>}
   */
  public async loadSchema(dbPath?: string): Promise<void> {
    await this.schemaExplorerController.loadSchema(dbPath);
  }

  /**
   * Fetches available models from the backend and populates the dropdown grouped by provider.
   * @returns {Promise<void>}
   */
  public async loadModels(): Promise<void> {
    try {
      const customModels = this.state.getConfiguredModels();
      const modelsUrl =
        customModels.length > 0
          ? `/api/models?configured_models=${encodeURIComponent(customModels.join(","))}`
          : "/api/models";
      const response = await fetchWithBackendError(modelsUrl);
      const data = await parseResponseJson<ModelsApiResponse>(response);

      const sqlOption = document.createElement("option");
      sqlOption.value = "sql";
      sqlOption.setAttribute("data-i18n", "app.literalSql");
      sqlOption.textContent = i18next.t("app.literalSql");
      this.modelSelect.replaceChildren(sqlOption);

      const groups: Record<string, HTMLOptGroupElement> = {
        ollama: document.createElement("optgroup"),
        openai: document.createElement("optgroup"),
        anthropic: document.createElement("optgroup"),
        google: document.createElement("optgroup"),
      };
      groups.ollama.label = i18next.t("provider.groupOllama", "Ollama (Local)");
      groups.openai.label = i18next.t("provider.groupOpenAI", "OpenAI");
      groups.anthropic.label = i18next.t(
        "provider.groupAnthropic",
        "Anthropic",
      );
      groups.google.label = i18next.t("provider.groupGoogle", "Google");

      const modelsList: ModelApiItem[] = Array.isArray(data.models)
        ? [...data.models]
        : [];

      for (const customModel of customModels) {
        const normalized = customModel.includes(":")
          ? customModel.replace(":", "/")
          : customModel;
        if (!modelsList.some((m) => m.name === normalized)) {
          const prov = normalized.includes("/")
            ? normalized.split("/")[0]
            : "ollama";
          modelsList.push({
            name: normalized,
            provider: prov,
            configured: true,
            requires_key: prov !== "ollama",
            available: true,
            reachable: true,
          });
        }
      }

      modelsList.forEach((model) => {
        const prov =
          model.provider ||
          (model.name.includes("/") ? model.name.split("/")[0] : "ollama");
        const hasKey = Boolean(this.state.getApiKey(prov));
        const isConfigured = model.configured || hasKey;
        const isReachable = model.reachable !== false;
        const isAvailable = model.available !== false && isConfigured;

        let badge = "🟢";
        if (!isReachable) {
          badge = "🔴";
        } else if (!isConfigured) {
          badge = "🟡";
        }

        const option = document.createElement("option");
        option.value = model.name;
        option.setAttribute("data-provider", prov);
        option.setAttribute("data-configured", isConfigured ? "true" : "false");
        option.textContent = `${model.name} ${badge}`;
        if (!isAvailable && model.error_code) {
          option.title = `${model.name} (${model.error_code})`;
        }

        if (!groups[prov]) {
          const dynamicGroup = document.createElement("optgroup");
          dynamicGroup.label = `${prov.charAt(0).toUpperCase() + prov.slice(1)} (any-llm)`;
          groups[prov] = dynamicGroup;
        }
        groups[prov].appendChild(option);
      });

      for (const grp of Object.values(groups)) {
        if (grp.children.length > 0) {
          this.modelSelect.appendChild(grp);
        }
      }

      const hasReachableOllama = modelsList.some((m) => {
        const prov =
          m.provider ||
          (m.name.includes("/") ? m.name.split("/")[0] : "ollama");
        return (
          prov === "ollama" &&
          m.available !== false &&
          m.reachable !== false &&
          m.configured !== false
        );
      });
      this.state.setSystemStatus({ ollamaOnline: hasReachableOllama });

      const activeChat = this.state.getActiveChat();
      if (activeChat) {
        const exists = Array.from(this.modelSelect.options).some(
          (opt) => opt.value === activeChat.model,
        );
        if (exists) {
          this.modelSelect.value = activeChat.model;
        } else if (modelsList.length > 0) {
          const firstReady = modelsList.find((m) => {
            const prov =
              m.provider ||
              (m.name.includes("/") ? m.name.split("/")[0] : "ollama");
            return (
              m.available !== false &&
              m.reachable !== false &&
              ((prov === "ollama" && m.configured !== false) ||
                Boolean(this.state.getApiKey(prov)))
            );
          });
          const fallbackModel = firstReady
            ? firstReady.name
            : activeChat.messages.length === 0
              ? "sql"
              : modelsList[0].name;
          this.modelSelect.value = fallbackModel;
          this.state.setActiveChatModel(fallbackModel);
        } else {
          this.modelSelect.value = "sql";
          this.state.setActiveChatModel("sql");
        }
      }
      this.updateProviderBadgeStatus();
    } catch (error) {
      console.warn("Failed to fetch models from API:", error);
    }
  }

  /**
   * Updates the provider status dot badge based on the currently selected model.
   * @returns {void}
   */
  public updateProviderBadgeStatus(): void {
    this.providerModalController.updateProviderBadgeStatus();
  }

  /**
   * Updates the aria-label and icon of the theme toggle button based on current mode and language.
   * @returns {void}
   */
  private updateThemeButtonLabel(): void {
    const isLight = document.body.classList.contains("light-mode");
    const ariaKey = isLight
      ? "aria.switchToDarkMode"
      : "aria.switchToLightMode";
    this.themeToggleBtn.setAttribute("data-i18n-aria-label", ariaKey);
    this.themeToggleBtn.setAttribute("aria-label", i18next.t(ariaKey));
    this.themeToggleBtn.textContent = isLight ? "☾" : "☀";
  }

  /**
   * Updates the visual indicator and accessibility attributes on the streaming toggle button.
   * @returns {void}
   */
  public updateStreamingToggleUi(): void {
    if (!this.streamingToggleBtn) return;
    if (this.streamingEnabled) {
      const label = i18next.t("ui.streamingOn", "Streaming Mode: ON (⚡ SSE)");
      this.streamingToggleBtn.classList.add("active");
      this.streamingToggleBtn.style.color = "var(--warning-accent, #f1c40f)";
      this.streamingToggleBtn.title = label;
      this.streamingToggleBtn.setAttribute("aria-label", label);
      this.streamingToggleBtn.setAttribute("aria-pressed", "true");
    } else {
      const label = i18next.t(
        "ui.streamingOff",
        "Streaming Mode: OFF (Standard Batch)",
      );
      this.streamingToggleBtn.classList.remove("active");
      this.streamingToggleBtn.style.color = "";
      this.streamingToggleBtn.title = label;
      this.streamingToggleBtn.setAttribute("aria-label", label);
      this.streamingToggleBtn.setAttribute("aria-pressed", "false");
    }
  }

  /**
   * Binds global and static DOM events.
   * @returns {void}
   */
  private bindEvents(): void {
    if (this.streamingToggleBtn) {
      this.streamingToggleBtn.addEventListener("click", () => {
        this.streamingEnabled = !this.streamingEnabled;
        this.state.streamingMode = this.streamingEnabled;
        this.state.saveToLocalStorage();
        this.updateStreamingToggleUi();
      });
    }

    this.newChatBtn.addEventListener("click", () => {
      this.state.createChat();
      const hasCloudKeys = Boolean(
        this.state.getApiKey("openai") ||
        this.state.getApiKey("anthropic") ||
        this.state.getApiKey("google") ||
        this.state.getConfiguredModels().length > 0,
      );
      if (!this.state.systemStatus.ollamaOnline && !hasCloudKeys) {
        this.state.setActiveChatModel("sql");
      }
      this.render();
      this.closeMobileSidebar();
      this.chatInput.focus();
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

    this.modelSelect.addEventListener("change", () => {
      this.state.setActiveChatModel(this.modelSelect.value);
      this.updateProviderBadgeStatus();
      this.syncHighlight();
    });

    if (this.dbSelect) {
      this.dbSelect.addEventListener("change", () => {
        const selectedDb = this.dbSelect?.value;
        if (selectedDb) {
          this.state.setCurrentDb(selectedDb);
          void this.loadSchema(selectedDb);
          void this.systemStatusController.check(true, selectedDb);
          this.announce(i18next.t("ui.databaseSwitched", { db: selectedDb }));
        }
      });
    }

    if (this.syncSessionsBtn) {
      this.syncSessionsBtn.addEventListener("click", () => {
        void this.syncAllChatsWithServer();
      });
    }

    if (this.providerSettingsBtn) {
      this.providerSettingsBtn.addEventListener("click", () => {
        void this.providerModalController.open();
      });
    }

    if (this.closeProviderModalBtn) {
      this.closeProviderModalBtn.addEventListener("click", () => {
        this.providerModalController.close();
      });
    }

    if (this.providerSaveBtn) {
      this.providerSaveBtn.addEventListener("click", () => {
        void this.saveProviderSettings();
      });
    }

    if (this.providerClearBtn) {
      this.providerClearBtn.addEventListener("click", () => {
        void this.clearProviderSettings();
      });
    }

    if (this.providerModal) {
      this.providerModal.addEventListener("click", (e) => {
        if (e.target === this.providerModal) {
          this.providerModalController.close();
        }
      });
    }

    if (this.cohortFilterBtn) {
      this.cohortFilterBtn.addEventListener("click", () => {
        this.cohortModalController.open();
      });
    }

    if (this.closeCohortModalBtn) {
      this.closeCohortModalBtn.addEventListener("click", () => {
        this.cohortModalController.close();
      });
    }

    if (this.cohortModal) {
      this.cohortModal.addEventListener("click", (e) => {
        if (e.target === this.cohortModal) {
          this.cohortModalController.close();
        }
      });
    }

    if (this.cohortTableSelect) {
      this.cohortTableSelect.addEventListener("change", () => {
        this.handleCohortJoinTableChange();
        this.updateCohortSqlPreview();
      });
    }

    if (this.cohortJoinTableSelect) {
      this.cohortJoinTableSelect.addEventListener("change", () => {
        this.handleCohortJoinTableChange();
        this.updateCohortSqlPreview();
      });
    }

    if (this.cohortJoinTypeSelect) {
      this.cohortJoinTypeSelect.addEventListener("change", () => {
        this.updateCohortSqlPreview();
      });
    }

    if (this.cohortJoinOnInput) {
      this.cohortJoinOnInput.addEventListener("input", () => {
        this.updateCohortSqlPreview();
      });
    }

    const cohortInputs = [
      this.cohortMinAge,
      this.cohortMaxAge,
      this.cohortGenderSelect,
      this.cohortTxGroup,
      this.cohortMaxTIR,
      this.cohortMinHbA1c,
    ];
    for (const inp of cohortInputs) {
      if (inp) {
        inp.addEventListener("input", () => this.updateCohortSqlPreview());
        inp.addEventListener("change", () => this.updateCohortSqlPreview());
      }
    }

    if (this.cohortInsertChatBtn) {
      this.cohortInsertChatBtn.addEventListener("click", () => {
        const sql = this.getCohortSql();
        this.chatInput.value = sql;
        this.syncHighlight();
        this.closeCohortModal();
        this.chatInput.focus();
      });
    }

    if (this.cohortExecuteBtn) {
      this.cohortExecuteBtn.addEventListener("click", () => {
        const sql = this.getCohortSql();
        this.closeCohortModal();
        this.modelSelect.value = "sql";
        this.state.setActiveChatModel("sql");
        void this.handleSendMessage(sql, "sql");
      });
    }

    this.langSelect.value = i18next.language;
    this.langSelect.addEventListener("change", async () => {
      await setLanguage(this.langSelect.value);
      this.updateThemeButtonLabel();
      this.updateStreamingToggleUi();
      this.updateProviderBadgeStatus();
      this.renderDatabaseDropdown();
      this.updateModalBadgesFromLocal();
      const tableModal = document.getElementById("table-modal");
      if (
        tableModal &&
        tableModal.getAttribute("aria-hidden") !== "true" &&
        this.currentTable
      ) {
        const modalTitle = document.getElementById("modal-title");
        if (modalTitle) {
          modalTitle.textContent = i18next.t("ui.tableName", {
            name: this.currentTable,
          });
        }
        void this.fetchTableData();
      }
      void this.systemStatusController.check();
      this.render();
    });

    this.updateThemeButtonLabel();

    this.themeToggleBtn.addEventListener("click", () => {
      const isLight = document.body.classList.toggle("light-mode");
      this.themeToggleBtn.setAttribute(
        "aria-pressed",
        isLight ? "false" : "true",
      );
      this.updateThemeButtonLabel();
    });

    if (this.bannerRetryBtn) {
      this.bannerRetryBtn.addEventListener("click", () => {
        void this.systemStatusController.check(true);
        void this.loadModels();
        void this.loadSchema();
      });
    }

    if (this.bannerInstructionsBtn && this.bannerInstructionsDrawer) {
      this.bannerInstructionsBtn.addEventListener("click", () => {
        const isHidden =
          this.bannerInstructionsDrawer?.classList.toggle("hidden");
        this.bannerInstructionsBtn?.setAttribute(
          "aria-expanded",
          isHidden ? "false" : "true",
        );
      });
    }

    if (this.bannerDismissBtn) {
      this.bannerDismissBtn.addEventListener("click", () => {
        this.hideBanner();
      });
    }

    if (this.systemStatusChip) {
      this.systemStatusChip.addEventListener("click", () => {
        void this.systemStatusController.check(true);
      });
    }

    this.openSidebarBtn.addEventListener("click", () => {
      document.body.classList.add("sidebar-open");
      this.openSidebarBtn.setAttribute("aria-expanded", "true");
      this.closeSidebarBtn?.focus();
    });

    const closeSidebar = (): void => {
      this.closeMobileSidebar();
      this.openSidebarBtn.focus();
    };

    this.closeSidebarBtn.addEventListener("click", closeSidebar);
    this.overlay.addEventListener("click", closeSidebar);
    const sidebarEl = document.getElementById("sidebar");
    if (sidebarEl) {
      sidebarEl.addEventListener("keydown", (e: KeyboardEvent) => {
        if (document.body.classList.contains("sidebar-open")) {
          trapModalFocus(e, sidebarEl, closeSidebar);
        }
      });
    }

    const toggleSchemaBtn = document.getElementById("toggle-schema-btn");
    const schemaExplorer = document.getElementById("schema-explorer");
    if (toggleSchemaBtn && schemaExplorer) {
      const toggleFn = (): void => {
        const isCollapsed = schemaExplorer.classList.toggle("collapsed");
        toggleSchemaBtn.setAttribute(
          "aria-expanded",
          isCollapsed ? "false" : "true",
        );
      };

      toggleSchemaBtn.setAttribute("aria-expanded", "true");

      const schemaHeader = document.querySelector(".schema-header");
      if (schemaHeader) {
        schemaHeader.addEventListener("click", toggleFn);
      } else {
        toggleSchemaBtn.addEventListener("click", toggleFn);
      }
    }

    document.addEventListener(
      "click",
      (e) => {
        const target = e.target;
        if (target instanceof Element) {
          if (!target.closest(".dropdown-container")) {
            this.closeAllDropdowns();
          }
        } else {
          this.closeAllDropdowns();
        }
      },
      { signal: this.globalListenerController.signal },
    );
  }

  /**
   * Closes the mobile sidebar if it's open.
   * @returns {void}
   */
  private closeMobileSidebar(): void {
    document.body.classList.remove("sidebar-open");
    this.openSidebarBtn?.setAttribute("aria-expanded", "false");
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
   * Executes a SQL query against `/api/execute-sql` and either appends a new assistant message or updates an existing message in place.
   * @param {string} queryToPlay - The SQL query string to execute.
   * @param {HTMLButtonElement} triggerBtn - The button element that triggered execution.
   * @param {Message} [messageToUpdate] - Optional existing assistant message to update in place.
   * @returns {Promise<void>}
   */
  public async executeDirectSqlAction(
    queryToPlay: string,
    triggerBtn: HTMLButtonElement,
    messageToUpdate?: Message,
  ): Promise<void> {
    triggerBtn.disabled = true;
    try {
      const dbPathOpt =
        this.state.currentDb && this.state.currentDb !== "t1d.duckdb"
          ? { db_path: this.state.currentDb }
          : {};
      const response = await fetchWithBackendError("/api/execute-sql", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: queryToPlay, ...dbPathOpt }),
      });
      const data = await parseResponseJson<ChatApiResponse>(response);

      let contentStr = "";
      let sqlRes: Record<string, SqlCellValue>[] | undefined = data.sqlResult;

      if (data.error) {
        const parsedError = parseApiError(data.error);
        contentStr = i18next.t("ui.errorDetails", { error: parsedError });
        this.announce(contentStr, true);
      } else {
        const classified = classifySqlResult(sqlRes);
        if (classified.kind === "empty") {
          contentStr = i18next.t("ui.noRows", "No rows returned");
          this.announce(contentStr, false);
          sqlRes = [];
        } else if (classified.kind === "scalar") {
          contentStr = classified.formattedValue;
          sqlRes = undefined;
        } else {
          contentStr = i18next.t("app.tableData", "Table Data");
        }
      }

      if (messageToUpdate) {
        messageToUpdate.content = contentStr;
        messageToUpdate.sqlResult = sqlRes;
        messageToUpdate.isError = !!data.error;
        this.state.saveToLocalStorage();
      } else {
        this.state.addMessageToActiveChat({
          role: "assistant",
          content: contentStr,
          sqlResult: sqlRes,
          sqlQuery: queryToPlay,
          isError: !!data.error,
          model: "sql",
        });
      }
      this.renderMessages();
      this.scrollToBottom();
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : String(err);
      const announcedErr = i18next.t("ui.errorComm", { error: errorMsg });
      if (messageToUpdate) {
        messageToUpdate.content = announcedErr;
        messageToUpdate.sqlResult = undefined;
        messageToUpdate.isError = true;
        this.state.saveToLocalStorage();
      } else {
        this.announce(announcedErr, true);
        this.state.addMessageToActiveChat({
          role: "assistant",
          content: announcedErr,
          isError: true,
          sqlQuery: queryToPlay,
          model: "sql",
        });
      }
      this.renderMessages();
      this.scrollToBottom();
    } finally {
      triggerBtn.disabled = false;
    }
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

    const effectiveModel =
      modelOverride ??
      (this.modelSelect.value === "sql" || isReadOnlySqlStatement(content)
        ? "sql"
        : this.modelSelect.value);

    this.state.addMessageToActiveChat({
      role: "user",
      content,
      model: effectiveModel,
    });

    if (text === undefined) {
      this.state.setActiveChatModel(this.modelSelect.value);
      this.chatInput.value = "";
      this.syncHighlight();
    }

    this.renderSidebar();
    this.renderMessages();
    this.scrollToBottom();

    this.chatInput.disabled = true;
    this.chatInputWrapper.classList.add("disabled");
    this.sendBtn.disabled = true;
    this.modelSelect.disabled = true;

    const loadingDiv = document.createElement("div");
    loadingDiv.className = "message assistant loading";
    loadingDiv.setAttribute("role", "status");
    loadingDiv.setAttribute("aria-live", "polite");
    const queryingLabel = i18next.t("ui.querying");
    const typingIndicator = document.createElement("div");
    typingIndicator.className = "typing-indicator";
    typingIndicator.setAttribute("aria-label", queryingLabel);
    typingIndicator.append(
      document.createElement("span"),
      document.createElement("span"),
      document.createElement("span"),
    );
    const srSpan = document.createElement("span");
    srSpan.className = "sr-only";
    srSpan.textContent = queryingLabel;
    typingIndicator.appendChild(srSpan);
    const loadingTextDiv = document.createElement("div");
    loadingTextDiv.style.fontSize = "0.85em";
    loadingTextDiv.style.marginTop = "0.5rem";
    loadingTextDiv.style.opacity = "0.7";
    loadingTextDiv.textContent = queryingLabel;
    loadingDiv.append(typingIndicator, loadingTextDiv);
    this.messagesContainer.appendChild(loadingDiv);
    this.scrollToBottom();

    const model = effectiveModel;
    const selectedOption = this.modelSelect.selectedOptions?.[0];
    const provider =
      selectedOption?.getAttribute("data-provider") ||
      (model.includes("/") ? model.split("/")[0] : undefined);
    const apiKey = provider ? this.state.getApiKey(provider) : undefined;

    const requestHeaders: Record<string, string> = {
      "Content-Type": "application/json",
    };
    if (provider) {
      requestHeaders["x-provider"] = provider;
    }
    if (apiKey) {
      requestHeaders["x-provider-api-key"] = apiKey;
    }

    const activeChat = this.state.getActiveChat();
    const historyPayload = (activeChat?.messages ?? [])
      .slice(0, -1)
      .map((m) => ({
        role: m.role,
        content: m.content,
        sqlQuery: m.sqlQuery,
        sqlResult: m.sqlResult,
        model: m.model,
      }));

    const dbPathPayload =
      this.state.currentDb && this.state.currentDb !== "t1d.duckdb"
        ? this.state.currentDb
        : undefined;

    const payload = {
      message: content,
      model,
      provider,
      api_key: apiKey,
      ...(dbPathPayload ? { db_path: dbPathPayload } : {}),
      history: historyPayload,
      auto_execute: model === "sql",
    };

    let streamUsed = false;
    let streamText = "";
    let finalData: SseFinalChatData | null = null;

    if (this.streamingEnabled) {
      try {
        const streamResp = await fetch("/api/chat/stream", {
          method: "POST",
          headers: requestHeaders,
          body: JSON.stringify(payload),
        });

        if (
          streamResp.ok &&
          streamResp.headers
            ?.get("content-type")
            ?.includes("text/event-stream") &&
          streamResp.body?.getReader
        ) {
          const reader = streamResp.body.getReader();
          const contentEl = document.createElement("div");
          contentEl.className = "message-content streaming-content";
          loadingDiv.replaceChildren(contentEl);

          const sseOutcome = await consumeSseChatStream(reader, (accText) => {
            contentEl.textContent = accText;
            this.scrollToBottom();
          });
          streamUsed = sseOutcome.streamUsed;
          streamText = sseOutcome.streamText;
          finalData = sseOutcome.finalData;
        }
      } catch {
        streamUsed = false;
      }
    }

    if (streamUsed && finalData) {
      loadingDiv.remove();
      let assistantMsg =
        finalData.content && i18next.exists(finalData.content)
          ? i18next.t(finalData.content)
          : finalData.content || streamText || "";
      if (finalData.error) {
        const parsedError = parseApiError(finalData.error);
        assistantMsg += `
${i18next.t("ui.errorDetails", { error: parsedError })}`;
        this.announce(assistantMsg, true);
      } else {
        const classifiedStream = classifySqlResult(finalData.sqlResult);
        if (classifiedStream.kind === "scalar") {
          assistantMsg = assistantMsg
            ? `${assistantMsg} **${classifiedStream.column}:** ${classifiedStream.formattedValue}`
            : classifiedStream.formattedValue;
        }
      }

      this.state.addMessageToActiveChat({
        role: "assistant",
        content: assistantMsg,
        sqlResult: finalData.sqlResult,
        sqlQuery: finalData.sqlQuery,
        isError: !!finalData.error,
        model,
      });
    } else {
      try {
        const response = await fetchWithBackendError("/api/chat", {
          method: "POST",
          headers: requestHeaders,
          body: JSON.stringify(payload),
        });

        const data = await parseResponseJson<ChatApiResponse>(response);
        loadingDiv.remove();

        let assistantMsg =
          data.content && i18next.exists(data.content)
            ? i18next.t(data.content)
            : (data.content ?? "");
        if (data.error) {
          const parsedError = parseApiError(data.error);
          assistantMsg += `
${i18next.t("ui.errorDetails", { error: parsedError })}`;
          this.announce(assistantMsg, true);
        } else {
          const classifiedBatch = classifySqlResult(data.sqlResult);
          if (classifiedBatch.kind === "scalar") {
            assistantMsg = assistantMsg
              ? `${assistantMsg} **${classifiedBatch.column}:** ${classifiedBatch.formattedValue}`
              : classifiedBatch.formattedValue;
          }
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
        loadingDiv.remove();
        const errorMsg = err instanceof Error ? err.message : String(err);
        const announcedErr = i18next.t("ui.errorComm", { error: errorMsg });
        this.announce(announcedErr, true);
        this.state.addMessageToActiveChat({
          role: "assistant",
          content: announcedErr,
          isError: true,
          model: model,
        });
      }
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

    this.renderSidebar();
    this.renderMessages();
    this.scrollToBottom();
  }

  /**
   * Fetches available DuckDB databases from the backend and updates the dropdown.
   * @returns {Promise<void>}
   */
  public async fetchDatabases(): Promise<void> {
    await this.schemaExplorerController.fetchDatabases();
  }

  /**
   * Renders options in the database selector dropdown.
   * @returns {void}
   */
  public renderDatabaseDropdown(): void {
    this.schemaExplorerController.renderDatabaseDropdown();
  }

  /**
   * Synchronizes all active non-temporary chats with the backend persistent store.
   * @returns {Promise<void>}
   */
  public async syncAllChatsWithServer(): Promise<void> {
    try {
      const ok = await this.state.syncWithServer();
      if (ok) {
        this.renderSidebar();
        this.renderMessages();
        this.announce(i18next.t("ui.sessionsSynced"));
      }
    } catch {
      // Ignore sync network errors
    }
  }

  /**
   * Generates the cohort filter DuckDB SQL query from active modal inputs.
   * @returns {string} The generated DuckDB SQL statement.
   */
  public getCohortSql(): string {
    return this.cohortModalController.getCohortSql();
  }

  /**
   * Updates join condition and types when target or secondary tables change.
   * @returns {void}
   */
  public handleCohortJoinTableChange(): void {
    this.cohortModalController.handleJoinTableChange();
  }

  /**
   * Updates the live DuckDB SQL preview in the cohort modal.
   * @returns {void}
   */
  public updateCohortSqlPreview(): void {
    this.cohortModalController.updateSqlPreview();
  }

  /**
   * Opens the Visual Cohort Query Builder modal and populates available clinical tables.
   * @returns {void}
   */
  public openCohortModal(): void {
    this.cohortModalController.open();
  }

  /**
   * Closes the Visual Cohort Query Builder modal.
   * @returns {void}
   */
  public closeCohortModal(): void {
    this.cohortModalController.close();
  }

  /**
   * Opens the Cloud LLM Providers & API Keys configuration modal.
   * @returns {Promise<void>}
   */
  public async openProviderModal(): Promise<void> {
    await this.providerModalController.open();
  }

  /**
   * Updates badges in the provider modal using locally stored keys.
   * @returns {void}
   */
  public updateModalBadgesFromLocal(): void {
    this.providerModalController.updateModalBadgesFromLocal();
  }

  /**
   * Updates an individual provider status badge element.
   * @param {string} provider - Provider identifier.
   * @param {boolean} isConfigured - Whether provider has active credentials.
   * @returns {void}
   */
  public updateModalBadge(provider: string, isConfigured: boolean): void {
    this.providerModalController.updateModalBadge(provider, isConfigured);
  }

  /**
   * Closes the Cloud LLM Providers & API Keys configuration modal.
   * @returns {void}
   */
  public closeProviderModal(): void {
    this.providerModalController.close();
  }

  /**
   * Saves provider API keys from the modal into localStorage.
   * @returns {Promise<void>}
   */
  public async saveProviderSettings(): Promise<void> {
    await this.providerModalController.saveSettings();
  }

  /**
   * Clears stored custom provider API keys from localStorage.
   * @returns {Promise<void>}
   */
  public async clearProviderSettings(): Promise<void> {
    await this.providerModalController.clearSettings();
  }

  /**
   * Scrolls the message container to the bottom.
   * @returns {void}
   */
  private scrollToBottom(): void {
    this.messagesContainer.scrollTop = this.messagesContainer.scrollHeight;
  }

  /**
   * Closes all open dropdown menus.
   * @returns {void}
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

    this.renderSidebar();
    this.renderActiveChat();
  }

  /**
   * Renders the sidebar chat list.
   * @returns {void}
   */
  private renderSidebar(): void {
    this.chatListEl.replaceChildren();

    this.state.chats.forEach((chat) => {
      const isActive = chat.id === this.state.activeChatId;
      const li = document.createElement("li");
      li.className = `chat-item ${isActive ? "active" : ""}`;

      const titleBtn = document.createElement("button");
      titleBtn.type = "button";
      titleBtn.className = "chat-item-title";
      titleBtn.textContent = chat.title;
      if (isActive) {
        titleBtn.setAttribute("aria-current", "page");
      }

      const activateChat = (): void => {
        this.state.setActiveChat(chat.id);
        this.render();
        this.closeMobileSidebar();
        this.chatInput.focus();
      };

      titleBtn.addEventListener("click", activateChat);
      titleBtn.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          activateChat();
        }
      });

      const dropdownContainer = document.createElement("div");
      dropdownContainer.className = "dropdown-container";

      const dropdownBtn = document.createElement("button");
      dropdownBtn.className = "dropdown-btn";
      dropdownBtn.textContent = "\u22EE";
      dropdownBtn.setAttribute("aria-label", i18next.t("aria.chatOptions"));
      dropdownBtn.setAttribute("data-i18n-aria-label", "aria.chatOptions");
      dropdownBtn.setAttribute("aria-haspopup", "menu");
      dropdownBtn.setAttribute("aria-expanded", "false");
      dropdownBtn.setAttribute("aria-controls", `dropdown-menu-${chat.id}`);

      const dropdownMenu = document.createElement("div");
      dropdownMenu.id = `dropdown-menu-${chat.id}`;
      dropdownMenu.className = "dropdown-menu";
      dropdownMenu.setAttribute("role", "menu");
      dropdownMenu.setAttribute(
        "aria-label",
        i18next.t("aria.chatActions", "Chat actions"),
      );
      dropdownMenu.setAttribute("data-i18n-aria-label", "aria.chatActions");

      const renameBtn = document.createElement("button");
      renameBtn.className = "dropdown-item";
      renameBtn.textContent = i18next.t("ui.rename");
      renameBtn.setAttribute(
        "aria-label",
        i18next.t("aria.editChatTitle", "Edit chat title"),
      );
      renameBtn.setAttribute("data-i18n-aria-label", "aria.editChatTitle");
      renameBtn.setAttribute("role", "menuitem");
      renameBtn.setAttribute("tabindex", "-1");
      renameBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        this.closeAllDropdowns();
        dropdownBtn.focus();
        const newTitle = prompt(i18next.t("ui.newChatTitle"), chat.title);
        if (newTitle !== null) {
          this.state.renameChat(chat.id, newTitle);
          this.render();
        }
      });

      const duplicateBtn = document.createElement("button");
      duplicateBtn.className = "dropdown-item";
      duplicateBtn.textContent = i18next.t("ui.duplicate");
      duplicateBtn.setAttribute(
        "aria-label",
        i18next.t("aria.duplicateChat", "Duplicate chat"),
      );
      duplicateBtn.setAttribute("data-i18n-aria-label", "aria.duplicateChat");
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
      deleteBtn.setAttribute(
        "aria-label",
        i18next.t("aria.deleteChat", "Delete chat"),
      );
      deleteBtn.setAttribute("data-i18n-aria-label", "aria.deleteChat");
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
        } else if (e.key === "ArrowUp") {
          e.preventDefault();
          this.closeAllDropdowns();
          dropdownMenu.classList.add("show");
          dropdownBtn.setAttribute("aria-expanded", "true");
          deleteBtn.focus();
        }
      });

      dropdownMenu.addEventListener("keydown", (e) => {
        const index = menuItems.findIndex(
          (item) => item === document.activeElement,
        );
        if (e.key === "ArrowDown") {
          e.preventDefault();
          const nextIndex = (index + 1) % menuItems.length;
          menuItems[nextIndex].focus();
        } else if (e.key === "ArrowUp") {
          e.preventDefault();
          const prevIndex = (index - 1 + menuItems.length) % menuItems.length;
          menuItems[prevIndex].focus();
        } else if (e.key === "Home") {
          e.preventDefault();
          menuItems[0].focus();
        } else if (e.key === "End") {
          e.preventDefault();
          menuItems[menuItems.length - 1].focus();
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

      li.appendChild(titleBtn);
      li.appendChild(dropdownContainer);

      this.chatListEl.appendChild(li);
    });
  }

  /**
   * Renders the active chat in the main pane.
   * @returns {void}
   */
  public renderActiveChat(): void {
    const activeChat = this.state.getActiveChat();

    if (!activeChat) {
      const emptyDiv = document.createElement("div");
      emptyDiv.className = "empty-state";
      emptyDiv.setAttribute("role", "status");
      emptyDiv.setAttribute("aria-live", "polite");
      emptyDiv.setAttribute("data-i18n", "app.emptyState");
      emptyDiv.textContent = i18next.t("app.emptyState");
      this.messagesContainer.replaceChildren(emptyDiv);
      this.chatInput.disabled = true;
      this.chatInputWrapper.classList.add("disabled");
      this.sendBtn.disabled = true;
      this.modelSelect.disabled = true;
      return;
    }

    if (!this.state.systemStatus.backendOnline) {
      this.chatInput.disabled = true;
      this.chatInputWrapper.classList.add("disabled");
      this.chatInput.placeholder = i18next.t("status.inputOffline");
      this.sendBtn.disabled = true;
    } else {
      this.chatInput.disabled = false;
      this.chatInputWrapper.classList.remove("disabled");
      this.chatInput.placeholder = i18next.t("app.typeMessage");
      this.sendBtn.disabled = false;
    }

    this.modelSelect.disabled = false;
    const hasActiveModelOption = Array.from(this.modelSelect.options).some(
      (opt) => opt.value === activeChat.model,
    );
    this.modelSelect.value = hasActiveModelOption ? activeChat.model : "sql";
    this.updateProviderBadgeStatus();
    this.syncHighlight();
    this.renderMessages();
  }

  /**
   * Renders only the messages of the active chat.
   * @returns {void}
   */
  public renderMessages(): void {
    this.messageRenderer.renderMessages();
  }

  /**
   * Opens the table modal and loads the first page of data.
   * @param {string} tableName - The name of the table to display.
   * @returns {Promise<void>}
   */
  public async openTableModal(tableName: string): Promise<void> {
    await this.tableModalController.open(tableName);
  }

  /**
   * Fetches the current page of data for the opened table and updates the modal UI.
   * @returns {Promise<void>}
   */
  public async fetchTableData(): Promise<void> {
    await this.tableModalController.fetchTableData();
  }

  /**
   * Export row objects as a downloadable CSV file.
   * @param {Array<Record<string, ExportableCellValue>>} rows - The data rows to export.
   * @param {string} filename - The filename for the downloaded file.
   * @returns {void}
   */
  public exportRowsToCsv(
    rows: Array<Record<string, ExportableCellValue>>,
    filename: string,
  ): void {
    exportRowsToCsvHelper(rows, filename);
  }

  /**
   * Export row objects as a downloadable JSON file.
   * @param {Array<Record<string, ExportableCellValue>>} rows - The data rows to export.
   * @param {string} filename - The filename for the downloaded file.
   * @returns {void}
   */
  public exportRowsToJson(
    rows: Array<Record<string, ExportableCellValue>>,
    filename: string,
  ): void {
    exportRowsToJsonHelper(rows, filename);
  }
}
