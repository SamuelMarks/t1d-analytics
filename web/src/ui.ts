/**
 * @file ui.ts
 * DOM manipulation and event binding for the Chat UI.
 */

import { marked } from "marked";
import DOMPurify from "dompurify";
import hljs from "highlight.js/lib/core";
import sql from "highlight.js/lib/languages/sql";
import markdown from "highlight.js/lib/languages/markdown";
import python from "highlight.js/lib/languages/python";
import "highlight.js/styles/github-dark.css";
import { ChatState, resolveTableJoin } from "./state";
import i18next, { setLanguage } from "./i18n";
import { renderCgmCard } from "./chart";

hljs.registerLanguage("sql", sql);
hljs.registerLanguage("markdown", markdown);
hljs.registerLanguage("python", python);

/**
 * Parses a backend error string or detail object into a translated string.
 */
function parseApiError(errorObj: unknown): string {
  if (typeof errorObj === "string") {
    return i18next.exists(errorObj)
      ? (i18next.t(errorObj) as string)
      : errorObj;
  }
  if (errorObj && typeof errorObj === "object" && "error_code" in errorObj) {
    const key = (errorObj as { error_code: string }).error_code;
    const params = (errorObj as { params?: Record<string, unknown> }).params;
    return i18next.exists(key)
      ? (i18next.t(key, params as any) as string)
      : `${key}: ${JSON.stringify(params)}`;
  }
  return String(errorObj);
}

/**
 * Helper to fetch and automatically throw parsed backend errors.
 * @param {string} url The URL to fetch.
 * @param {RequestInit} [options] Fetch options.
 * @returns {Promise<Response>} The fetch response.
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
 * Clinical cohort filter parameters for query generation.
 */
export interface CohortFilterParams {
  /** Target primary table name. */
  tableName: string;
  /** Optional secondary table name for relational join. */
  joinTableName?: string;
  /** Relational join condition type ('INNER JOIN' or 'LEFT JOIN'). */
  joinType?: "INNER JOIN" | "LEFT JOIN";
  /** Explicit ON condition for table joining. */
  joinCondition?: string;
  /** Minimum patient age filter. */
  minAge?: number;
  /** Maximum patient age filter. */
  maxAge?: number;
  /** Biological sex or gender. */
  gender?: string;
  /** Clinical treatment group (e.g. Closed-Loop, MDI). */
  txGroup?: string;
  /** Time in Range maximum threshold percentage. */
  maxTIR?: number;
  /** Minimum glycated hemoglobin (HbA1c) threshold. */
  minHbA1c?: number;
}

/**
 * Construct an optimized DuckDB SQL query from clinician cohort filter parameters, supporting relational joins.
 * @param {CohortFilterParams} params - The selected filter criteria.
 * @returns {string} The generated DuckDB SQL query.
 */
export function buildCohortSql(params: CohortFilterParams): string {
  const table1 = params.tableName || "patients";
  const table2 = params.joinTableName?.trim();
  const joinType = params.joinType || "INNER JOIN";
  const joinOn = params.joinCondition?.trim();

  const isJoin = Boolean(table2 && table2 !== table1);
  const prefix = isJoin ? `"${table1.replace(/"/g, '""')}".` : "";
  const predicates: string[] = [];

  if (params.minAge !== undefined && Number.isFinite(params.minAge)) {
    predicates.push(`${prefix}age >= ${params.minAge}`);
  }
  if (params.maxAge !== undefined && Number.isFinite(params.maxAge)) {
    predicates.push(`${prefix}age <= ${params.maxAge}`);
  }
  if (params.gender) {
    const safeGender = params.gender.replace(/'/g, "''");
    predicates.push(`${prefix}gender = '${safeGender}'`);
  }
  if (params.txGroup) {
    const safeTx = params.txGroup.replace(/'/g, "''");
    predicates.push(`${prefix}txgroup ILIKE '%${safeTx}%'`);
  }
  if (params.maxTIR !== undefined && Number.isFinite(params.maxTIR)) {
    predicates.push(`${prefix}tir <= ${params.maxTIR}`);
  }
  if (params.minHbA1c !== undefined && Number.isFinite(params.minHbA1c)) {
    predicates.push(`${prefix}hba1c >= ${params.minHbA1c}`);
  }

  let fromClause = `FROM "${table1.replace(/"/g, '""')}"`;
  if (isJoin && table2) {
    const defaultOn = `"${table1.replace(/"/g, '""')}".patient_id = "${table2.replace(/"/g, '""')}".patient_id`;
    fromClause += `\n${joinType} "${table2.replace(/"/g, '""')}"\n  ON ${joinOn || defaultOn}`;
  }

  const whereClause =
    predicates.length > 0 ? `\nWHERE ${predicates.join("\n  AND ")}` : "";
  return `SELECT *\n${fromClause}${whereClause};`;
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
  private dbSelect: HTMLSelectElement | null;
  private syncSessionsBtn: HTMLButtonElement | null;
  private streamingToggleBtn: HTMLButtonElement | null;
  public streamingEnabled: boolean = false;
  private langSelect: HTMLSelectElement;
  private openSidebarBtn: HTMLButtonElement;
  private closeSidebarBtn: HTMLButtonElement;
  private themeToggleBtn: HTMLButtonElement;
  private overlay: HTMLElement;

  // Diagnostic and status elements
  private systemStatusChip: HTMLButtonElement | null;
  private statusChipText: HTMLElement | null;
  private systemStatusBanner: HTMLElement | null;
  private bannerTitle: HTMLElement | null;
  private bannerDesc: HTMLElement | null;
  private bannerIcon: HTMLElement | null;
  private bannerRetryBtn: HTMLButtonElement | null;
  private bannerInstructionsBtn: HTMLButtonElement | null;
  private bannerInstructionsDrawer: HTMLElement | null;
  private bannerDismissBtn: HTMLButtonElement | null;

  // Cohort filter elements
  private cohortFilterBtn: HTMLButtonElement | null;
  private cohortModal: HTMLElement | null;
  private closeCohortModalBtn: HTMLButtonElement | null;
  private cohortTableSelect: HTMLSelectElement | null;
  private cohortJoinTableSelect: HTMLSelectElement | null;
  private cohortJoinTypeSelect: HTMLSelectElement | null;
  private cohortJoinOnInput: HTMLInputElement | null;
  private cohortMinAge: HTMLInputElement | null;
  private cohortMaxAge: HTMLInputElement | null;
  private cohortGenderSelect: HTMLSelectElement | null;
  private cohortTxGroup: HTMLInputElement | null;
  private cohortMaxTIR: HTMLInputElement | null;
  private cohortMinHbA1c: HTMLInputElement | null;
  private cohortSqlPreview: HTMLElement | null;
  private cohortInsertChatBtn: HTMLButtonElement | null;
  private cohortExecuteBtn: HTMLButtonElement | null;

  // Provider settings elements
  private providerSettingsBtn: HTMLButtonElement | null;
  private providerStatusDot: HTMLElement | null;
  private providerModal: HTMLElement | null;
  private closeProviderModalBtn: HTMLButtonElement | null;
  private providerKeyOpenAI: HTMLInputElement | null;
  private providerKeyAnthropic: HTMLInputElement | null;
  private providerKeyGoogle: HTMLInputElement | null;
  private providerStoreConsent: HTMLInputElement | null;
  private providerSaveBtn: HTMLButtonElement | null;
  private providerClearBtn: HTMLButtonElement | null;
  private providerBadgeOpenAI: HTMLElement | null;
  private providerBadgeAnthropic: HTMLElement | null;
  private providerBadgeGoogle: HTMLElement | null;

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
    this.dbSelect = document.getElementById(
      "db-select",
    ) as HTMLSelectElement | null;
    this.syncSessionsBtn = document.getElementById(
      "sync-sessions-btn",
    ) as HTMLButtonElement | null;
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

    // Diagnostic and status elements
    this.systemStatusChip = document.getElementById(
      "system-status-chip",
    ) as HTMLButtonElement | null;
    this.statusChipText = document.getElementById("status-chip-text");
    this.systemStatusBanner = document.getElementById("system-status-banner");
    this.bannerTitle = document.getElementById("banner-title");
    this.bannerDesc = document.getElementById("banner-desc");
    this.bannerIcon = document.getElementById("banner-icon");
    this.bannerRetryBtn = document.getElementById(
      "banner-retry-btn",
    ) as HTMLButtonElement | null;
    this.bannerInstructionsBtn = document.getElementById(
      "banner-instructions-btn",
    ) as HTMLButtonElement | null;
    this.bannerInstructionsDrawer = document.getElementById(
      "banner-instructions-drawer",
    );
    this.bannerDismissBtn = document.getElementById(
      "banner-dismiss-btn",
    ) as HTMLButtonElement | null;

    // Cohort filter elements
    this.cohortFilterBtn = document.getElementById(
      "cohort-filter-btn",
    ) as HTMLButtonElement | null;
    this.cohortModal = document.getElementById("cohort-modal");
    this.closeCohortModalBtn = document.getElementById(
      "close-cohort-modal-btn",
    ) as HTMLButtonElement | null;
    this.cohortTableSelect = document.getElementById(
      "cohort-table-select",
    ) as HTMLSelectElement | null;
    this.cohortJoinTableSelect = document.getElementById(
      "cohort-join-table-select",
    ) as HTMLSelectElement | null;
    this.cohortJoinTypeSelect = document.getElementById(
      "cohort-join-type-select",
    ) as HTMLSelectElement | null;
    this.cohortJoinOnInput = document.getElementById(
      "cohort-join-on-input",
    ) as HTMLInputElement | null;
    this.cohortMinAge = document.getElementById(
      "cohort-min-age",
    ) as HTMLInputElement | null;
    this.cohortMaxAge = document.getElementById(
      "cohort-max-age",
    ) as HTMLInputElement | null;
    this.cohortGenderSelect = document.getElementById(
      "cohort-gender-select",
    ) as HTMLSelectElement | null;
    this.cohortTxGroup = document.getElementById(
      "cohort-txgroup",
    ) as HTMLInputElement | null;
    this.cohortMaxTIR = document.getElementById(
      "cohort-max-tir",
    ) as HTMLInputElement | null;
    this.cohortMinHbA1c = document.getElementById(
      "cohort-min-hba1c",
    ) as HTMLInputElement | null;
    this.cohortSqlPreview = document.getElementById("cohort-sql-preview");
    this.cohortInsertChatBtn = document.getElementById(
      "cohort-insert-chat-btn",
    ) as HTMLButtonElement | null;
    this.cohortExecuteBtn = document.getElementById(
      "cohort-execute-btn",
    ) as HTMLButtonElement | null;

    // Provider settings elements
    this.providerSettingsBtn = document.getElementById(
      "provider-settings-btn",
    ) as HTMLButtonElement | null;
    this.providerStatusDot = document.getElementById("provider-status-dot");
    this.providerModal = document.getElementById("provider-modal");
    this.closeProviderModalBtn = document.getElementById(
      "close-provider-modal-btn",
    ) as HTMLButtonElement | null;
    this.providerKeyOpenAI = document.getElementById(
      "provider-key-openai",
    ) as HTMLInputElement | null;
    this.providerKeyAnthropic = document.getElementById(
      "provider-key-anthropic",
    ) as HTMLInputElement | null;
    this.providerKeyGoogle = document.getElementById(
      "provider-key-google",
    ) as HTMLInputElement | null;
    this.providerStoreConsent = document.getElementById(
      "provider-store-consent",
    ) as HTMLInputElement | null;
    this.providerSaveBtn = document.getElementById(
      "provider-save-btn",
    ) as HTMLButtonElement | null;
    this.providerClearBtn = document.getElementById(
      "provider-clear-btn",
    ) as HTMLButtonElement | null;
    this.providerBadgeOpenAI = document.getElementById("provider-badge-openai");
    this.providerBadgeAnthropic = document.getElementById(
      "provider-badge-anthropic",
    );
    this.providerBadgeGoogle = document.getElementById("provider-badge-google");

    this.streamingToggleBtn = document.getElementById(
      "streaming-toggle-btn",
    ) as HTMLButtonElement | null;
    this.streamingEnabled = this.state.streamingMode;
    this.updateStreamingToggleUi();

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
   * Shows the system status alert banner with the specified severity and messaging.
   * @param {"danger" | "warning" | "info"} severity The alert level.
   * @param {string} title The title to display.
   * @param {string} desc The description message.
   * @param {string} icon The icon to render.
   * @param {boolean} showRetry Whether to show the retry button.
   */
  public showBanner(
    severity: "danger" | "warning" | "info",
    title: string,
    desc: string,
    icon: string = "⚠️",
    showRetry: boolean = true,
  ): void {
    if (!this.systemStatusBanner) return;
    this.systemStatusBanner.className = `system-status-banner banner-${severity}`;
    if (this.bannerIcon) this.bannerIcon.textContent = icon;
    if (this.bannerTitle) this.bannerTitle.textContent = title;
    if (this.bannerDesc) this.bannerDesc.textContent = desc;
    if (this.bannerRetryBtn) {
      this.bannerRetryBtn.style.display = showRetry ? "inline-block" : "none";
    }
    this.systemStatusBanner.classList.remove("hidden");
  }

  /**
   * Hides the system status alert banner.
   */
  public hideBanner(): void {
    if (this.systemStatusBanner) {
      this.systemStatusBanner.classList.add("hidden");
    }
  }

  /**
   * Updates the status chip in the top header.
   * @param {"healthy" | "degraded" | "error" | "offline"} level The status severity.
   * @param {string} text The status text.
   */
  public updateStatusChip(
    level: "healthy" | "degraded" | "error" | "offline",
    text: string,
  ): void {
    if (!this.systemStatusChip) return;
    this.systemStatusChip.className = `status-chip status-${level}`;
    let tooltip = `System status: ${text}`;
    if (this.state.systemStatus.dbFileSizeBytes) {
      const mb = (
        this.state.systemStatus.dbFileSizeBytes /
        (1024 * 1024)
      ).toFixed(1);
      tooltip += ` | DB: ${mb} MB (${this.state.systemStatus.tableCount} tables)`;
    }
    this.systemStatusChip.setAttribute("aria-label", tooltip);
    this.systemStatusChip.setAttribute("title", tooltip);
    if (this.statusChipText) {
      this.statusChipText.textContent = text;
    }
  }

  /**
   * Checks system health and updates UI indicators accordingly.
   * @param {boolean} [isManualRetry=false] Whether this is a manual user retry.
   * @param {string} [dbPath] Optional custom database path.
   */
  public async checkSystemStatus(
    isManualRetry: boolean = false,
    dbPath?: string,
  ): Promise<void> {
    if (isManualRetry && this.bannerRetryBtn) {
      this.bannerRetryBtn.textContent = i18next.t("status.reconnecting");
    }

    try {
      const targetDb = dbPath || this.state.currentDb;
      const url =
        targetDb && targetDb !== "t1d.duckdb"
          ? `/api/status?db_path=${encodeURIComponent(targetDb)}`
          : "/api/status";
      const response = await fetchWithBackendError(url);
      const health = await response.json();

      this.state.setSystemStatus({
        backendOnline: true,
        status: health.status,
        dbConfigured: Boolean(health.database?.exists),
        dbExists: Boolean(health.database?.exists),
        dbConnected: Boolean(health.database?.connected),
        dbStatusCode: health.database?.status_code || "unknown",
        dbMessage: health.database?.message || null,
        dbRemediation: health.database?.remediation || null,
        tableCount: Number(health.database?.table_count) || 0,
        hasInitialData: Boolean(health.database?.has_initial_data),
        ollamaOnline: Boolean(health.ollama?.accessible),
        ollamaMessage: health.ollama?.message || null,
        ollamaRemediation: health.ollama?.remediation || null,
        dbFileSizeBytes: Number(health.database?.file_size_bytes) || 0,
        dbWritable: Boolean(health.database?.writable),
        dbIntegrityOk: health.database?.integrity_ok !== false,
        diskFreeBytes: Number(health.database?.disk_free_bytes) || 0,
        ollamaVersion: health.ollama?.version || null,
      });

      // Handle database statuses
      if (health.database?.status_code === "missing_file") {
        this.showBanner(
          "warning",
          i18next.t("status.dbMissingTitle"),
          i18next.t("status.dbMissingDesc", {
            path: health.database.configured_path,
          }),
          "📁",
          true,
        );
        this.updateStatusChip("error", i18next.t("status.dbMissingTitle"));
      } else if (health.database?.status_code === "empty_db") {
        this.showBanner(
          "warning",
          i18next.t("status.dbEmptyTitle"),
          i18next.t("status.dbEmptyDesc"),
          "🗄️",
          true,
        );
        this.updateStatusChip("degraded", i18next.t("status.dbEmptyTitle"));
      } else if (health.database?.status_code === "missing_initial_data") {
        this.showBanner(
          "warning",
          i18next.t("status.dbMissingDataTitle"),
          i18next.t("status.dbMissingDataDesc"),
          "📊",
          true,
        );
        this.updateStatusChip(
          "degraded",
          i18next.t("status.dbMissingDataTitle"),
        );
      } else if (!health.ollama?.accessible) {
        this.showBanner(
          "info",
          i18next.t("status.ollamaOfflineTitle"),
          i18next.t("status.ollamaOfflineDesc"),
          "ℹ️",
          false,
        );
        this.updateStatusChip(
          "degraded",
          i18next.t("status.ollamaOfflineTitle"),
        );
        if (this.modelSelect && this.modelSelect.value !== "sql") {
          this.modelSelect.value = "sql";
        }
      } else {
        this.hideBanner();
        this.updateStatusChip("healthy", i18next.t("status.systemHealthy"));
      }

      this.render();
    } catch {
      this.state.setSystemStatus({
        backendOnline: false,
        status: "offline",
      });
      const backendUrl = window.location.origin;
      this.showBanner(
        "danger",
        i18next.t("status.offlineTitle"),
        i18next.t("status.offlineDesc", { url: backendUrl }),
        "⚠️",
        true,
      );
      this.updateStatusChip("offline", i18next.t("status.systemOffline"));
      this.render();
    } finally {
      if (this.bannerRetryBtn) {
        this.bannerRetryBtn.textContent = i18next.t("status.retry");
      }
    }
  }

  /**
   * Copies text to the user's clipboard and provides visual button feedback.
   * @param {string} cmd The command string to copy.
   * @param {HTMLButtonElement} btn The button providing visual feedback.
   */
  public async copyCommandToClipboard(
    cmd: string,
    btn: HTMLButtonElement,
  ): Promise<void> {
    if (!navigator?.clipboard) return;
    try {
      await navigator.clipboard.writeText(cmd);
      btn.textContent = i18next.t("status.commandCopied");
      setTimeout(() => {
        btn.textContent = i18next.t("status.copyCommand");
      }, 2000);
    } catch {
      // Ignore clipboard write errors
    }
  }

  /**
   * Binds interaction events to action buttons inside diagnostic cards.
   * @param {HTMLElement} container The container holding diagnostic cards.
   */
  private bindSchemaDiagnosticButtons(container: HTMLElement): void {
    container.querySelectorAll(".copy-cmd-btn").forEach((btn) => {
      btn.addEventListener("click", (e) => {
        const target = e.currentTarget as HTMLButtonElement;
        const cmd = target.dataset.cmd;
        if (cmd) {
          void this.copyCommandToClipboard(cmd, target);
        }
      });
    });

    container.querySelectorAll(".reload-schema-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        this.loadSchema();
        this.checkSystemStatus(true);
      });
    });
  }

  /**
   * Fetches the database schema and renders it in the sidebar.
   * @param {string} [dbPath] Optional custom database path.
   */
  public async loadSchema(dbPath?: string): Promise<void> {
    const schemaContent = document.getElementById("schema-content");
    if (!schemaContent) return;

    try {
      const targetDb = dbPath || this.state.currentDb;
      const url =
        targetDb && targetDb !== "t1d.duckdb"
          ? `/api/schema?db_path=${encodeURIComponent(targetDb)}`
          : "/api/schema";
      const response = await fetchWithBackendError(url);
      const data = await response.json();

      schemaContent.innerHTML = ""; // Clear loading message

      if (data.database) {
        const dbStatus = data.database;
        const isHealthy = dbStatus.status_code === "healthy";
        this.state.setSystemStatus({
          backendOnline: true,
          status: isHealthy ? "healthy" : "degraded",
          dbConfigured: Boolean(dbStatus.exists),
          dbExists: Boolean(dbStatus.exists),
          dbConnected: Boolean(dbStatus.connected),
          dbStatusCode: dbStatus.status_code || "unknown",
          dbMessage: dbStatus.message || null,
          dbRemediation: dbStatus.remediation || null,
          tableCount: Number(dbStatus.table_count) || 0,
          hasInitialData: Boolean(dbStatus.has_initial_data),
          dbFileSizeBytes: Number(dbStatus.file_size_bytes) || 0,
          dbWritable: Boolean(dbStatus.writable),
          dbIntegrityOk: dbStatus.integrity_ok !== false,
          diskFreeBytes: Number(dbStatus.disk_free_bytes) || 0,
        });

        if (dbStatus.status_code === "missing_file") {
          this.showBanner(
            "warning",
            i18next.t("status.dbMissingTitle"),
            i18next.t("status.dbMissingDesc", {
              path: dbStatus.configured_path,
            }),
            "📁",
            true,
          );
          this.updateStatusChip("error", i18next.t("status.dbMissingTitle"));
        } else if (dbStatus.status_code === "empty_db") {
          this.showBanner(
            "warning",
            i18next.t("status.dbEmptyTitle"),
            i18next.t("status.dbEmptyDesc"),
            "🗄️",
            true,
          );
          this.updateStatusChip("degraded", i18next.t("status.dbEmptyTitle"));
        } else if (dbStatus.status_code === "missing_initial_data") {
          this.showBanner(
            "warning",
            i18next.t("status.dbMissingDataTitle"),
            i18next.t("status.dbMissingDataDesc"),
            "📊",
            true,
          );
          this.updateStatusChip(
            "degraded",
            i18next.t("status.dbMissingDataTitle"),
          );
        } else {
          this.hideBanner();
          this.updateStatusChip("healthy", i18next.t("status.systemHealthy"));
        }
      }

      if (!data.tables || data.tables.length === 0) {
        const dbStatus = data.database;
        if (dbStatus && dbStatus.status_code === "missing_file") {
          schemaContent.innerHTML = `
            <div class="schema-diagnostic-card" role="alert">
              <div class="schema-diagnostic-header">
                <span>📁</span>
                <span>${i18next.t("status.dbMissingTitle")}</span>
              </div>
              <p class="schema-diagnostic-desc">${i18next.t(
                "status.dbMissingDesc",
                { path: dbStatus.configured_path },
              )}</p>
              <div class="schema-command-box">
                <code>t1d-analytics load --db ${dbStatus.configured_path}</code>
                <button class="btn btn-xs btn-secondary copy-cmd-btn" data-cmd="t1d-analytics load --db ${dbStatus.configured_path}">${i18next.t("status.copyCommand")}</button>
              </div>
              <button class="btn btn-xs btn-primary reload-schema-btn">${i18next.t("status.reloadSchema")}</button>
            </div>
          `;
        } else if (dbStatus && dbStatus.status_code === "empty_db") {
          schemaContent.innerHTML = `
            <div class="schema-diagnostic-card" role="alert">
              <div class="schema-diagnostic-header">
                <span>🗄️</span>
                <span>${i18next.t("status.dbEmptyTitle")}</span>
              </div>
              <p class="schema-diagnostic-desc">${i18next.t("status.dbEmptyDesc")}</p>
              <div class="schema-command-box">
                <code>t1d-analytics load</code>
                <button class="btn btn-xs btn-secondary copy-cmd-btn" data-cmd="t1d-analytics load">${i18next.t("status.copyCommand")}</button>
              </div>
              <button class="btn btn-xs btn-primary reload-schema-btn">${i18next.t("status.reloadSchema")}</button>
            </div>
          `;
        } else {
          schemaContent.innerHTML = `<div class="schema-loading" role="status" aria-live="polite" data-i18n="ui.noTables">${i18next.t("ui.noTables")}</div>`;
        }
        this.bindSchemaDiagnosticButtons(schemaContent);
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

          const toggleBtn = document.createElement("button");
          toggleBtn.className = "schema-table-header-toggle";
          toggleBtn.setAttribute("aria-expanded", "false");
          toggleBtn.setAttribute("aria-controls", `schema-table-${table.name}`);
          toggleBtn.innerHTML = `
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true">
              <polyline points="9 18 15 12 9 6"></polyline>
            </svg>
          `;
          const titleSpan = document.createElement("span");
          titleSpan.textContent = table.name;
          toggleBtn.appendChild(titleSpan);

          const viewBtn = document.createElement("button");
          viewBtn.className = "icon-btn table-view-btn";
          viewBtn.setAttribute("aria-label", i18next.t("aria.viewTableData"));
          viewBtn.setAttribute("title", i18next.t("ui.viewTableData"));
          viewBtn.setAttribute("data-table", table.name);
          viewBtn.setAttribute("data-i18n-aria-label", "aria.viewTableData");
          viewBtn.setAttribute("data-i18n-title", "ui.viewTableData");
          viewBtn.innerHTML = `
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
              <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
              <line x1="3" y1="9" x2="21" y2="9"></line>
              <line x1="9" y1="21" x2="9" y2="9"></line>
            </svg>
          `;

          headerDiv.appendChild(toggleBtn);
          headerDiv.appendChild(viewBtn);

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

          viewBtn.addEventListener("click", (e) => {
            e.stopPropagation();
            this.openTableModal(table.name);
          });
          const columnsList = document.createElement("ul");
          columnsList.className = "schema-columns";
          columnsList.id = `schema-table-${table.name}`;

          table.columns.forEach((col) => {
            const colLi = document.createElement("li");
            colLi.className = "schema-column";
            const colNameSpan = document.createElement("span");
            colNameSpan.textContent = col.name;
            const colTypeSpan = document.createElement("span");
            colTypeSpan.className = "schema-column-type";
            colTypeSpan.textContent = col.type;
            colLi.appendChild(colNameSpan);
            colLi.appendChild(colTypeSpan);
            columnsList.appendChild(colLi);
          });

          tableDiv.appendChild(headerDiv);
          tableDiv.appendChild(columnsList);
          schemaContent.appendChild(tableDiv);
        },
      );
    } catch (error) {
      console.error("Failed to load schema:", error);
      const errMsg = i18next.t("ui.failedSchema");
      this.announce(errMsg, true);
      const backendUrl = window.location.origin;
      schemaContent.innerHTML = `
        <div class="schema-diagnostic-card error-text" role="alert">
          <div class="schema-diagnostic-header">
            <span>⚠️</span>
            <span>${i18next.t("status.offlineTitle")}</span>
          </div>
          <p class="schema-diagnostic-desc">${i18next.t("status.offlineDesc", { url: backendUrl })}</p>
          <button class="btn btn-xs btn-primary reload-schema-btn">${i18next.t("status.reloadSchema")}</button>
        </div>
      `;
      this.bindSchemaDiagnosticButtons(schemaContent);
      this.state.setSystemStatus({
        backendOnline: false,
        status: "offline",
      });
      this.showBanner(
        "danger",
        i18next.t("status.offlineTitle"),
        i18next.t("status.offlineDesc", { url: backendUrl }),
        "⚠️",
        true,
      );
      this.updateStatusChip("offline", i18next.t("status.systemOffline"));
    }
  }

  /**
   * Fetches available models from the backend and populates the dropdown grouped by provider.
   */
  public async loadModels(): Promise<void> {
    try {
      const response = await fetchWithBackendError("/api/models");
      const data = await response.json();

      // Preserve the "Literal SQL" option
      this.modelSelect.innerHTML = `<option value="sql" data-i18n="app.literalSql">${i18next.t("app.literalSql")}</option>`;

      const groups: Record<string, HTMLOptGroupElement> = {
        ollama: document.createElement("optgroup"),
        openai: document.createElement("optgroup"),
        anthropic: document.createElement("optgroup"),
        google: document.createElement("optgroup"),
      };
      groups.ollama.label = "Ollama (Local)";
      groups.openai.label = "OpenAI";
      groups.anthropic.label = "Anthropic";
      groups.google.label = "Google";

      const modelsList = Array.isArray(data.models) ? data.models : [];
      modelsList.forEach(
        (model: {
          name: string;
          size?: number;
          provider?: string;
          configured?: boolean;
          requires_key?: boolean;
          available?: boolean;
          reachable?: boolean;
          error_code?: string;
        }) => {
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
          option.textContent = `${model.name} ${badge}`;
          if (!isAvailable && model.error_code) {
            option.title = `${model.name} (${model.error_code})`;
          }

          const targetGroup = groups[prov] || groups.ollama;
          targetGroup.appendChild(option);
        },
      );

      for (const grp of Object.values(groups)) {
        if (grp.children.length > 0) {
          this.modelSelect.appendChild(grp);
        }
      }

      // Reset the current model to match the active chat
      const activeChat = this.state.getActiveChat();
      if (activeChat) {
        const exists = Array.from(this.modelSelect.options).some(
          (opt) => opt.value === activeChat.model,
        );
        if (exists) {
          this.modelSelect.value = activeChat.model;
        } else if (modelsList.length > 0) {
          this.modelSelect.value = modelsList[0].name;
          this.state.setActiveChatModel(modelsList[0].name);
        }
      }
      this.updateProviderBadgeStatus();
    } catch (error) {
      console.warn("Failed to fetch models from API:", error);
    }
  }

  /**
   * Updates the provider status dot badge based on the currently selected model.
   */
  public updateProviderBadgeStatus(): void {
    if (!this.providerStatusDot) return;
    const selectedOption = this.modelSelect.selectedOptions?.[0];
    const prov =
      selectedOption?.getAttribute("data-provider") ||
      (this.modelSelect.value.includes("/")
        ? this.modelSelect.value.split("/")[0]
        : "ollama");

    this.state.setActiveChatProvider(prov);
    if (prov === "ollama" || this.modelSelect.value === "sql") {
      this.providerStatusDot.className = "provider-status-dot active";
      this.providerStatusDot.title = "Local Engine Active";
    } else {
      const hasKey = Boolean(this.state.getApiKey(prov));
      if (hasKey) {
        this.providerStatusDot.className = "provider-status-dot active";
        this.providerStatusDot.title = `${prov} Active (Configured)`;
      } else {
        this.providerStatusDot.className = "provider-status-dot warning";
        this.providerStatusDot.title = `${prov} Missing API Key (Click ⚙️ to configure)`;
      }
    }
  }

  /**
   * Updates the aria-label and icon of the theme toggle button based on current mode and language.
   */
  private updateThemeButtonLabel(): void {
    const isLight = document.body.classList.contains("light-mode");
    const ariaKey = isLight
      ? "aria.switchToDarkMode"
      : "aria.switchToLightMode";
    this.themeToggleBtn.setAttribute("data-i18n-aria-label", ariaKey);
    this.themeToggleBtn.setAttribute("aria-label", i18next.t(ariaKey));
    this.themeToggleBtn.innerHTML = isLight ? "☾" : "☀";
  }

  /**
   * Updates the visual indicator and accessibility attributes on the streaming toggle button.
   */
  public updateStreamingToggleUi(): void {
    if (!this.streamingToggleBtn) return;
    if (this.streamingEnabled) {
      this.streamingToggleBtn.classList.add("active");
      this.streamingToggleBtn.style.color = "#f1c40f";
      this.streamingToggleBtn.title = "Streaming Mode: ON (⚡ SSE)";
      this.streamingToggleBtn.setAttribute(
        "aria-label",
        "Streaming Mode: ON (⚡ SSE)",
      );
    } else {
      this.streamingToggleBtn.classList.remove("active");
      this.streamingToggleBtn.style.color = "";
      this.streamingToggleBtn.title = "Streaming Mode: OFF (Standard Batch)";
      this.streamingToggleBtn.setAttribute(
        "aria-label",
        "Streaming Mode: OFF (Standard Batch)",
      );
    }
  }

  /**
   * Binds global and static DOM events.
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
      this.updateProviderBadgeStatus();
      this.syncHighlight();
    });

    if (this.dbSelect) {
      this.dbSelect.addEventListener("change", () => {
        const selectedDb = this.dbSelect?.value;
        if (selectedDb) {
          this.state.setCurrentDb(selectedDb);
          void this.loadSchema(selectedDb);
          void this.checkSystemStatus(true, selectedDb);
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
        void this.openProviderModal();
      });
    }

    if (this.closeProviderModalBtn) {
      this.closeProviderModalBtn.addEventListener("click", () => {
        this.closeProviderModal();
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
          this.closeProviderModal();
        }
      });
    }

    if (this.cohortFilterBtn) {
      this.cohortFilterBtn.addEventListener("click", () => {
        this.openCohortModal();
      });
    }

    if (this.closeCohortModalBtn) {
      this.closeCohortModalBtn.addEventListener("click", () => {
        this.closeCohortModal();
      });
    }

    if (this.cohortModal) {
      this.cohortModal.addEventListener("click", (e) => {
        if (e.target === this.cohortModal) {
          this.closeCohortModal();
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
      // Re-render UI components that generate dynamic text
      this.checkSystemStatus();
      this.render();
    });

    // Set initial label
    this.updateThemeButtonLabel();

    this.themeToggleBtn.addEventListener("click", () => {
      const isLight = document.body.classList.toggle("light-mode");
      this.themeToggleBtn.setAttribute(
        "aria-pressed",
        isLight ? "false" : "true",
      );
      this.updateThemeButtonLabel();
    });

    // Status banner & chip interactions
    if (this.bannerRetryBtn) {
      this.bannerRetryBtn.addEventListener("click", () => {
        this.checkSystemStatus(true);
        this.loadModels();
        this.loadSchema();
      });
    }

    if (this.bannerInstructionsBtn && this.bannerInstructionsDrawer) {
      this.bannerInstructionsBtn.addEventListener("click", () => {
        this.bannerInstructionsDrawer?.classList.toggle("hidden");
      });
    }

    if (this.bannerDismissBtn) {
      this.bannerDismissBtn.addEventListener("click", () => {
        this.hideBanner();
      });
    }

    if (this.systemStatusChip) {
      this.systemStatusChip.addEventListener("click", () => {
        this.checkSystemStatus(true);
      });
    }

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

    this.chatInputHighlight!.textContent = value;

    if (this.modelSelect.value === "sql") {
      this.chatInputHighlight!.className = "language-sql";
    } else {
      this.chatInputHighlight!.className = "language-markdown";
    }
    delete this.chatInputHighlight!.dataset.highlighted;
    hljs.highlightElement(this.chatInputHighlight!);

    // Auto-resize both textarea and wrapper
    this.chatInput.style.height = "auto";
    const scrollHeight = this.chatInput.scrollHeight;
    this.chatInput.style.height = scrollHeight + "px";
  }

  /**
   * Announces a message to screen readers using the aria-live region.
   */
  private announce(message: string, isAssertive: boolean = false): void {
    const announcer = document.getElementById("a11y-announcer");
    if (!announcer) return;
    announcer.textContent = "";
    announcer.setAttribute("aria-live", isAssertive ? "assertive" : "polite");
    setTimeout(() => {
      announcer.textContent = message;
    }, 50);
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

    const payload = {
      message: content,
      model,
      provider,
      api_key: apiKey,
      db_path: this.state.currentDb,
      history: historyPayload,
    };

    let streamUsed = false;
    let streamText = "";
    let finalData: {
      content?: string;
      sqlResult?: Record<string, string | number | boolean | null>[];
      sqlQuery?: string;
      error?: unknown;
    } | null = null;

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
          const decoder = new TextDecoder();
          let buffer = "";

          loadingDiv.innerHTML = `<div class="message-content streaming-content"></div>`;
          const contentEl = loadingDiv.querySelector(
            ".streaming-content",
          ) as HTMLElement;

          while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });
            const parts = buffer.split("\n\n");
            buffer = parts.pop() || "";
            for (const part of parts) {
              const line = part.trim();
              if (line.startsWith("data: ")) {
                try {
                  const eventData = JSON.parse(line.slice(6));
                  if (eventData.event === "token" && eventData.token) {
                    streamText += eventData.token;
                    if (contentEl) {
                      contentEl.textContent = streamText;
                      this.scrollToBottom();
                    }
                  } else if (
                    eventData.event === "result" ||
                    eventData.event === "done"
                  ) {
                    finalData = eventData;
                    streamUsed = true;
                  }
                } catch {
                  // ignore JSON parse errors on partial chunks
                }
              }
            }
          }
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
        assistantMsg += `\n${i18next.t("ui.errorDetails", { error: parsedError })}`;
        this.announce(assistantMsg, true);
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

        const data = await response.json();
        loadingDiv.remove();

        let assistantMsg =
          data.content && i18next.exists(data.content)
            ? i18next.t(data.content)
            : (data.content ?? "");
        if (data.error) {
          const parsedError = parseApiError(data.error);
          assistantMsg += `\n${i18next.t("ui.errorDetails", { error: parsedError })}`;
          this.announce(assistantMsg, true);
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
   */
  public async fetchDatabases(): Promise<void> {
    if (!this.dbSelect) return;
    try {
      const resp = await fetch("/api/databases");
      if (!resp.ok) return;
      const data = await resp.json();
      const dbs: { name: string; path: string; size_bytes: number }[] =
        data.databases || [];
      this.state.setAvailableDbs(dbs, data.current_db);
      this.renderDatabaseDropdown();
    } catch {
      // Ignore network failures fetching database list
    }
  }

  /**
   * Renders options in the database selector dropdown.
   */
  public renderDatabaseDropdown(): void {
    if (!this.dbSelect) return;
    this.dbSelect.innerHTML = "";
    if (this.state.availableDbs.length === 0) {
      const opt = document.createElement("option");
      opt.value = this.state.currentDb || "t1d.duckdb";
      opt.textContent = opt.value;
      this.dbSelect.appendChild(opt);
      return;
    }
    for (const db of this.state.availableDbs) {
      const opt = document.createElement("option");
      opt.value = db.name;
      opt.textContent = `${db.name} (${Math.round(db.size_bytes / 1024)} KB)`;
      if (
        db.name === this.state.currentDb ||
        db.path === this.state.currentDb
      ) {
        opt.selected = true;
      }
      this.dbSelect.appendChild(opt);
    }
  }

  /**
   * Synchronizes all active non-temporary chats with the backend persistent store.
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
    const tableName = this.cohortTableSelect?.value || "patients";
    const joinTableName = this.cohortJoinTableSelect?.value || undefined;
    const joinType =
      (this.cohortJoinTypeSelect?.value as "INNER JOIN" | "LEFT JOIN") ||
      "INNER JOIN";
    const joinCondition = this.cohortJoinOnInput?.value?.trim() || undefined;
    const minAge = this.cohortMinAge?.value
      ? parseFloat(this.cohortMinAge.value)
      : undefined;
    const maxAge = this.cohortMaxAge?.value
      ? parseFloat(this.cohortMaxAge.value)
      : undefined;
    const gender = this.cohortGenderSelect?.value || undefined;
    const txGroup = this.cohortTxGroup?.value?.trim() || undefined;
    const maxTIR = this.cohortMaxTIR?.value
      ? parseFloat(this.cohortMaxTIR.value)
      : undefined;
    const minHbA1c = this.cohortMinHbA1c?.value
      ? parseFloat(this.cohortMinHbA1c.value)
      : undefined;

    return buildCohortSql({
      tableName,
      joinTableName,
      joinType,
      joinCondition,
      minAge,
      maxAge,
      gender,
      txGroup,
      maxTIR,
      minHbA1c,
    });
  }

  /**
   * Updates join condition and types when target or secondary tables change.
   */
  public handleCohortJoinTableChange(): void {
    const table1 = this.cohortTableSelect?.value || "patients";
    const table2 = this.cohortJoinTableSelect?.value;
    if (!table2 || table2 === table1) {
      if (this.cohortJoinOnInput) this.cohortJoinOnInput.value = "";
      return;
    }

    const rel = resolveTableJoin(table1, table2);
    if (rel) {
      if (this.cohortJoinTypeSelect) {
        this.cohortJoinTypeSelect.value = rel.defaultJoinType;
      }
      if (this.cohortJoinOnInput) {
        this.cohortJoinOnInput.value = `"${table1}".${rel.fromColumn} = "${table2}".${rel.toColumn}`;
      }
    } else {
      if (this.cohortJoinOnInput) {
        this.cohortJoinOnInput.value = `"${table1}".patient_id = "${table2}".patient_id`;
      }
    }
  }

  /**
   * Updates the live DuckDB SQL preview in the cohort modal.
   */
  public updateCohortSqlPreview(): void {
    if (this.cohortSqlPreview) {
      this.cohortSqlPreview.textContent = this.getCohortSql();
    }
  }

  /**
   * Opens the Visual Cohort Query Builder modal and populates available clinical tables.
   */
  public openCohortModal(): void {
    if (!this.cohortModal) return;
    const tableHeaders = document.querySelectorAll(".schema-table-header h4");
    const foundTables: string[] = [];
    tableHeaders.forEach((h) => {
      const tName = h.textContent?.trim();
      if (tName) foundTables.push(tName);
    });
    const finalTables =
      foundTables.length > 0
        ? foundTables
        : ["demographics", "cgm_data", "patients"];

    if (this.cohortTableSelect) {
      this.cohortTableSelect.innerHTML = "";
      for (const t of finalTables) {
        const opt = document.createElement("option");
        opt.value = t;
        opt.textContent = t;
        this.cohortTableSelect.appendChild(opt);
      }
    }

    if (this.cohortJoinTableSelect) {
      this.cohortJoinTableSelect.innerHTML =
        '<option value="">None (Single Table)</option>';
      for (const t of finalTables) {
        const opt = document.createElement("option");
        opt.value = t;
        opt.textContent = t;
        this.cohortJoinTableSelect.appendChild(opt);
      }
    }

    this.handleCohortJoinTableChange();
    this.updateCohortSqlPreview();
    this.cohortModal.classList.remove("hidden");
    this.cohortMinAge?.focus();
  }

  /**
   * Closes the Visual Cohort Query Builder modal.
   */
  public closeCohortModal(): void {
    if (!this.cohortModal) return;
    this.cohortModal.classList.add("hidden");
    this.cohortFilterBtn?.focus();
  }

  /**
   * Opens the Cloud LLM Providers & API Keys configuration modal.
   */
  public async openProviderModal(): Promise<void> {
    if (!this.providerModal) return;

    if (this.providerKeyOpenAI) {
      this.providerKeyOpenAI.value = this.state.getApiKey("openai") || "";
    }
    if (this.providerKeyAnthropic) {
      this.providerKeyAnthropic.value = this.state.getApiKey("anthropic") || "";
    }
    if (this.providerKeyGoogle) {
      this.providerKeyGoogle.value = this.state.getApiKey("google") || "";
    }

    try {
      const resp = await fetch("/api/providers/status");
      if (resp.ok) {
        const data = await resp.json();
        for (const p of data.providers || []) {
          this.updateModalBadge(
            p.provider,
            p.configured || Boolean(this.state.getApiKey(p.provider)),
          );
        }
      } else {
        this.updateModalBadgesFromLocal();
      }
    } catch {
      this.updateModalBadgesFromLocal();
    }

    this.providerModal.classList.remove("hidden");
    this.providerKeyOpenAI?.focus();
  }

  /**
   * Updates badges in the provider modal using locally stored keys.
   */
  private updateModalBadgesFromLocal(): void {
    for (const p of ["openai", "anthropic", "google"]) {
      this.updateModalBadge(p, Boolean(this.state.getApiKey(p)));
    }
  }

  /**
   * Updates an individual provider status badge element.
   * @param {string} provider Provider identifier.
   * @param {boolean} isConfigured Whether provider has active credentials.
   */
  public updateModalBadge(provider: string, isConfigured: boolean): void {
    const el = document.getElementById(`provider-badge-${provider}`);
    if (!el) return;
    if (isConfigured) {
      el.className = "provider-status-badge badge-active";
      el.textContent = "Configured 🟢";
    } else {
      el.className = "provider-status-badge badge-missing";
      el.textContent = "Missing Key 🟡";
    }
  }

  /**
   * Closes the Cloud LLM Providers & API Keys configuration modal.
   */
  public closeProviderModal(): void {
    if (!this.providerModal) return;
    this.providerModal.classList.add("hidden");
    this.providerSettingsBtn?.focus();
  }

  /**
   * Saves provider API keys from the modal into localStorage.
   */
  public async saveProviderSettings(): Promise<void> {
    const consent = this.providerStoreConsent
      ? this.providerStoreConsent.checked
      : true;
    if (consent) {
      if (this.providerKeyOpenAI) {
        const val = this.providerKeyOpenAI.value.trim();
        if (val) this.state.setApiKey("openai", val);
      }
      if (this.providerKeyAnthropic) {
        const val = this.providerKeyAnthropic.value.trim();
        if (val) this.state.setApiKey("anthropic", val);
      }
      if (this.providerKeyGoogle) {
        const val = this.providerKeyGoogle.value.trim();
        if (val) this.state.setApiKey("google", val);
      }
    }
    await this.loadModels();
    this.closeProviderModal();
    this.announce(i18next.t("ui.settingsSaved", "Provider settings saved"));
  }

  /**
   * Clears stored custom provider API keys from localStorage.
   */
  public async clearProviderSettings(): Promise<void> {
    this.state.removeApiKey("openai");
    this.state.removeApiKey("anthropic");
    this.state.removeApiKey("google");
    if (this.providerKeyOpenAI) this.providerKeyOpenAI.value = "";
    if (this.providerKeyAnthropic) this.providerKeyAnthropic.value = "";
    if (this.providerKeyGoogle) this.providerKeyGoogle.value = "";
    this.updateModalBadgesFromLocal();
    await this.loadModels();
    this.announce(i18next.t("ui.settingsCleared", "Provider keys cleared"));
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
      dropdownBtn.setAttribute("data-i18n-aria-label", "aria.chatOptions");
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

      if (
        this.state.systemStatus.dbStatusCode === "empty_db" ||
        this.state.systemStatus.dbStatusCode === "missing_initial_data" ||
        this.state.systemStatus.dbStatusCode === "missing_file"
      ) {
        const dbNotice = document.createElement("div");
        dbNotice.className = "schema-diagnostic-card";
        dbNotice.style.marginBottom = "1rem";
        dbNotice.innerHTML = `
          <div class="schema-diagnostic-header">
            <span>ℹ️</span>
            <span>${i18next.t("status.emptyDbPrompt")}</span>
          </div>
          <div class="schema-command-box">
            <code>t1d-analytics load</code>
            <button class="btn btn-xs btn-secondary copy-cmd-btn" data-cmd="t1d-analytics load">${i18next.t("status.copyCommand")}</button>
          </div>
        `;
        this.bindSchemaDiagnosticButtons(dbNotice);
        emptyStateDiv.appendChild(dbNotice);
      }

      emptyStateDiv.insertAdjacentHTML(
        "beforeend",
        `<p style="margin-bottom: 1rem;">${i18next.t("ui.noMessagesYet")}</p>`,
      );

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

      const createCopyBtn = (textToCopy: string) => {
        const btn = document.createElement("button");
        btn.className = "copy-sql-btn icon-btn";
        btn.setAttribute(
          "aria-label",
          i18next.t("aria.copyQuery", "Copy Query"),
        );
        btn.setAttribute("data-i18n-aria-label", "aria.copyQuery");
        btn.title = i18next.t("ui.copyQuery", "Copy Query");
        btn.setAttribute("data-i18n-title", "ui.copyQuery");
        btn.innerHTML = `
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
            <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
          </svg>
        `;
        btn.addEventListener("click", async () => {
          try {
            await navigator.clipboard.writeText(textToCopy);
            const originalSvg = btn.innerHTML;
            btn.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"></polyline></svg>`;
            setTimeout(() => {
              btn.innerHTML = originalSvg;
            }, 2000);
          } catch (err) {
            console.error("Failed to copy text: ", err);
          }
        });
        return btn;
      };

      const createPlayBtn = (queryToPlay: string) => {
        const btn = document.createElement("button");
        btn.className = "play-sql-btn";
        btn.setAttribute(
          "aria-label",
          i18next.t("aria.playQuery", "Execute Query"),
        );
        btn.setAttribute("data-i18n-aria-label", "aria.playQuery");
        btn.innerHTML = `
          <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
            <polygon points="5 3 19 12 5 21 5 3"></polygon>
          </svg>
          <span data-i18n="app.play">${i18next.t("app.play", "Play")}</span>
        `;
        btn.addEventListener("click", async () => {
          btn.disabled = true;
          try {
            const response = await fetchWithBackendError("/api/execute-sql", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ query: queryToPlay }),
            });
            const data = await response.json();

            let contentStr = "";
            let sqlRes = data.sqlResult;

            if (data.error) {
              const parsedError = parseApiError(data.error);
              contentStr = i18next.t("ui.errorDetails", { error: parsedError });
              this.announce(contentStr, true);
            } else if (!sqlRes || sqlRes.length === 0) {
              contentStr = i18next.t("ui.noRows", "No rows returned");
              this.announce(contentStr, false);
              sqlRes = []; // ensure it triggers table/empty rendering logic
            } else if (
              sqlRes.length === 1 &&
              Object.keys(sqlRes[0]).length === 1
            ) {
              // Scalar result
              const key = Object.keys(sqlRes[0])[0];
              const val = sqlRes[0][key];
              contentStr = val === null ? "NULL" : String(val);
              sqlRes = undefined; // Do not render as table
            } else {
              // Full table
              contentStr = i18next.t("app.tableData", "Table Data");
            }

            this.state.addMessageToActiveChat({
              role: "assistant",
              content: contentStr,
              sqlResult: sqlRes,
              sqlQuery: queryToPlay,
              isError: !!data.error,
              model: "sql",
            });
            this.renderMessages();
            this.scrollToBottom();
          } catch (err) {
            const errorMsg = err instanceof Error ? err.message : String(err);
            const announcedErr = i18next.t("ui.errorComm", { error: errorMsg });
            this.announce(announcedErr, true);
            this.state.addMessageToActiveChat({
              role: "assistant",
              content: announcedErr,
              isError: true,
              sqlQuery: queryToPlay,
              model: "sql",
            });
            this.renderMessages();
            this.scrollToBottom();
          } finally {
            btn.disabled = false;
          }
        });
        return btn;
      };

      const createRefreshBtn = (queryToPlay: string, messageToUpdate: any) => {
        const btn = document.createElement("button");
        btn.className = "copy-sql-btn icon-btn";
        btn.setAttribute(
          "aria-label",
          i18next.t("aria.refreshQuery", "Refresh Query"),
        );
        btn.setAttribute("data-i18n-aria-label", "aria.refreshQuery");
        btn.title = i18next.t("ui.refreshQuery", "Refresh Query");
        btn.setAttribute("data-i18n-title", "ui.refreshQuery");
        btn.innerHTML = `
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <polyline points="23 4 23 10 17 10"></polyline>
            <polyline points="1 20 1 14 7 14"></polyline>
            <path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"></path>
          </svg>
        `;
        btn.addEventListener("click", async () => {
          btn.disabled = true;
          try {
            const response = await fetchWithBackendError("/api/execute-sql", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ query: queryToPlay }),
            });
            const data = await response.json();

            let contentStr = "";
            let sqlRes = data.sqlResult;

            if (data.error) {
              const parsedError = parseApiError(data.error);
              contentStr = i18next.t("ui.errorDetails", { error: parsedError });
              this.announce(contentStr, true);
            } else if (!sqlRes || sqlRes.length === 0) {
              contentStr = i18next.t("ui.noRows", "No rows returned");
              this.announce(contentStr, false);
              sqlRes = []; // ensure it triggers table/empty rendering logic
            } else if (
              sqlRes.length === 1 &&
              Object.keys(sqlRes[0]).length === 1
            ) {
              // Scalar result
              const key = Object.keys(sqlRes[0])[0];
              const val = sqlRes[0][key];
              contentStr = val === null ? "NULL" : String(val);
              sqlRes = undefined; // Do not render as table
            } else {
              // Full table
              contentStr = i18next.t("app.tableData", "Table Data");
            }

            messageToUpdate.content = contentStr;
            messageToUpdate.sqlResult = sqlRes;
            messageToUpdate.isError = !!data.error;

            this.state.saveToLocalStorage();
            this.renderMessages();
            this.scrollToBottom();
          } catch (err) {
            const errorMsg = err instanceof Error ? err.message : String(err);
            messageToUpdate.content = i18next.t("ui.errorComm", {
              error: errorMsg,
            });
            messageToUpdate.sqlResult = undefined;
            messageToUpdate.isError = true;
            this.state.saveToLocalStorage();
            this.renderMessages();
            this.scrollToBottom();
          } finally {
            btn.disabled = false;
          }
        });
        return btn;
      };

      if (isSqlMessage) {
        contentDiv.className = "sql-user-query-wrapper";

        const headerDiv = document.createElement("div");
        headerDiv.className = "sql-query-header";
        headerDiv.appendChild(createCopyBtn(msg.content));
        headerDiv.appendChild(createPlayBtn(msg.content));

        const preBlock = document.createElement("pre");
        preBlock.style.margin = "0";
        preBlock.style.background = "transparent";
        preBlock.style.padding = "0";
        preBlock.style.whiteSpace = "pre-wrap";
        preBlock.tabIndex = 0;
        preBlock.setAttribute("aria-label", i18next.t("aria.codeBlock"));
        preBlock.setAttribute("data-i18n-aria-label", "aria.codeBlock");

        const codeBlock = document.createElement("code");
        codeBlock.className = "language-sql";
        codeBlock.style.background = "transparent";
        codeBlock.style.padding = "0";
        codeBlock.textContent = msg.content;

        hljs.highlightElement(codeBlock);
        preBlock.appendChild(codeBlock);

        contentDiv.appendChild(headerDiv);
        contentDiv.appendChild(preBlock);
      } else {
        const rawHtml = marked.parse(msg.content ?? "") as string;
        contentDiv.innerHTML = DOMPurify.sanitize(rawHtml, {
          ADD_ATTR: ["target"],
        });
        const codeBlocks = contentDiv.querySelectorAll("pre code");
        codeBlocks.forEach((block) => {
          hljs.highlightElement(block as HTMLElement);

          // Add copy and play button for SQL markdown blocks
          if (block.className.includes("language-sql")) {
            const preElement = block.parentElement!;
            const wrapper = document.createElement("div");
            wrapper.className = "sql-query-container";

            const headerDiv = document.createElement("div");
            headerDiv.className = "sql-query-header";
            const queryStr = block.textContent!;
            headerDiv.appendChild(createCopyBtn(queryStr));
            headerDiv.appendChild(createPlayBtn(queryStr));

            preElement.parentNode!.insertBefore(wrapper, preElement);
            wrapper.appendChild(headerDiv);
            wrapper.appendChild(preElement);
          }
        });
      }

      if (msg.isError) {
        contentDiv.className = "error-text";
        contentDiv.setAttribute("role", "alert");
        contentDiv.setAttribute("aria-live", "assertive");
      }

      msgDiv.appendChild(contentDiv);

      const hasMarkdownSqlBlock =
        contentDiv.querySelector(".sql-query-container") !== null;
      if (msg.sqlQuery && !hasMarkdownSqlBlock) {
        const queryContainer = document.createElement("div");
        queryContainer.className = "sql-query-container";
        const preBlock = document.createElement("pre");
        preBlock.tabIndex = 0;
        preBlock.setAttribute("aria-label", i18next.t("aria.codeBlock"));
        preBlock.setAttribute("data-i18n-aria-label", "aria.codeBlock");
        const codeBlock = document.createElement("code");
        codeBlock.className = "language-sql sql-query";
        codeBlock.textContent = msg.sqlQuery;

        hljs.highlightElement(codeBlock);
        preBlock.appendChild(codeBlock);

        const headerDiv = document.createElement("div");
        headerDiv.className = "sql-query-header";

        headerDiv.appendChild(createCopyBtn(msg.sqlQuery));

        if (msg.role === "assistant" && msg.model === "sql") {
          headerDiv.appendChild(createRefreshBtn(msg.sqlQuery, msg));
        } else {
          headerDiv.appendChild(createPlayBtn(msg.sqlQuery));
        }

        queryContainer.appendChild(headerDiv);
        queryContainer.appendChild(preBlock);
        msgDiv.appendChild(queryContainer);
      }
      if (msg.sqlResult) {
        if (msg.sqlResult.length > 0) {
          const tableContainer = document.createElement("div");
          tableContainer.className = "sql-table-container";

          const toolbar = document.createElement("div");
          toolbar.className = "sql-table-toolbar";
          const csvBtn = document.createElement("button");
          csvBtn.className = "btn-secondary export-csv-btn";
          csvBtn.textContent = "CSV";
          csvBtn.setAttribute("aria-label", "Export CSV");
          csvBtn.addEventListener("click", () =>
            this.exportRowsToCsv(
              msg.sqlResult as Array<Record<string, unknown>>,
              "query_results.csv",
            ),
          );

          const jsonBtn = document.createElement("button");
          jsonBtn.className = "btn-secondary export-json-btn";
          jsonBtn.textContent = "JSON";
          jsonBtn.setAttribute("aria-label", "Export JSON");
          jsonBtn.addEventListener("click", () =>
            this.exportRowsToJson(
              msg.sqlResult as Array<Record<string, unknown>>,
              "query_results.json",
            ),
          );

          toolbar.appendChild(csvBtn);
          toolbar.appendChild(jsonBtn);
          tableContainer.appendChild(toolbar);

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

          const cgmCard = renderCgmCard(
            msg.sqlResult as Array<Record<string, unknown>>,
          );
          if (cgmCard) {
            tableContainer.appendChild(cgmCard);
          }

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
  private currentModalRows: Array<Record<string, unknown>> = [];
  private currentSortBy: string = "";
  private currentSortOrder: "asc" | "desc" = "asc";
  private currentSearchTerm: string = "";

  /**
   * Opens the table modal and loads the first page of data.
   * @param {string} tableName - The name of the table to display.
   */
  private async openTableModal(tableName: string) {
    const previousFocus = document.activeElement as HTMLElement | null;
    this.currentTable = tableName;
    this.currentPage = 1;
    this.currentSortBy = "";
    this.currentSortOrder = "asc";
    this.currentSearchTerm = "";

    const modal = document.getElementById("table-modal");
    const modalTitle = document.getElementById("modal-title");
    const closeBtn = document.getElementById("close-modal-btn");
    const prevBtn = document.getElementById(
      "prev-page-btn",
    ) as HTMLButtonElement;
    const nextBtn = document.getElementById(
      "next-page-btn",
    ) as HTMLButtonElement;

    const searchInput = document.getElementById(
      "modal-table-search",
    ) as HTMLInputElement | null;
    const sortSelect = document.getElementById(
      "modal-table-sort-col",
    ) as HTMLSelectElement | null;
    const sortOrderBtn = document.getElementById(
      "modal-table-sort-order",
    ) as HTMLButtonElement | null;

    if (!modal || !modalTitle || !closeBtn || !prevBtn || !nextBtn) return;

    if (searchInput) {
      searchInput.value = "";
    }
    if (sortSelect) {
      sortSelect.innerHTML = '<option value="">(Default)</option>';
    }
    if (sortOrderBtn) {
      sortOrderBtn.textContent = "ASC";
    }

    modalTitle.textContent = i18next.t("ui.tableName", { name: tableName });
    modal.removeAttribute("aria-hidden");
    modal.setAttribute("aria-modal", "true");

    // Wire up modal export buttons if present
    const modalExportCsvBtn = document.getElementById(
      "modal-export-csv-btn",
    ) as HTMLButtonElement | null;
    const modalExportJsonBtn = document.getElementById(
      "modal-export-json-btn",
    ) as HTMLButtonElement | null;

    if (modalExportCsvBtn) {
      modalExportCsvBtn.onclick = () =>
        this.exportRowsToCsv(
          this.currentModalRows,
          `${this.currentTable}_page${this.currentPage}.csv`,
        );
    }
    if (modalExportJsonBtn) {
      modalExportJsonBtn.onclick = () =>
        this.exportRowsToJson(
          this.currentModalRows,
          `${this.currentTable}_page${this.currentPage}.json`,
        );
    }

    // Focus the first interactive element
    closeBtn.focus();

    // Close listeners
    const closeModal = () => {
      modal.setAttribute("aria-hidden", "true");
      modal.removeAttribute("aria-modal");
      // Clean up event listeners
      closeBtn.removeEventListener("click", closeModal);
      modal.removeEventListener("click", overlayClick);
      modal.removeEventListener("keydown", onModalKeyDown);

      // Restore focus to where it was before opening the modal
      if (previousFocus && document.body.contains(previousFocus)) {
        previousFocus.focus();
      } else {
        document.getElementById("chat-input")?.focus();
      }
    };

    // Trap focus and handle Escape inside modal
    const onModalKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.preventDefault();
        closeModal();
        return;
      }
      if (e.key !== "Tab") return;
      const focusableElements = modal.querySelectorAll<HTMLElement>(
        'button:not([disabled]), [href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])',
      );
      if (focusableElements.length === 0) return;
      const firstElement = focusableElements[0];
      const lastElement = focusableElements[focusableElements.length - 1];

      if (e.shiftKey) {
        if (document.activeElement === firstElement) {
          e.preventDefault();
          lastElement.focus();
        }
      } else {
        if (document.activeElement === lastElement) {
          e.preventDefault();
          firstElement.focus();
        }
      }
    };

    modal.addEventListener("keydown", onModalKeyDown);

    const overlayClick = (e: MouseEvent) => {
      if (e.target === modal) closeModal();
    };

    closeBtn.addEventListener("click", closeModal);
    modal.addEventListener("click", overlayClick);

    // Wire search input with debouncing or immediate input
    if (searchInput) {
      const newSearchInput = searchInput.cloneNode(true) as HTMLInputElement;
      searchInput.parentNode?.replaceChild(newSearchInput, searchInput);
      newSearchInput.addEventListener("input", () => {
        this.currentSearchTerm = newSearchInput.value.trim();
        this.currentPage = 1;
        this.fetchTableData();
      });
    }

    // Wire sort controls
    if (sortSelect) {
      const newSortSelect = sortSelect.cloneNode(true) as HTMLSelectElement;
      sortSelect.parentNode?.replaceChild(newSortSelect, sortSelect);
      newSortSelect.addEventListener("change", () => {
        this.currentSortBy = newSortSelect.value;
        this.currentPage = 1;
        this.fetchTableData();
      });
    }

    if (sortOrderBtn) {
      const newSortOrderBtn = sortOrderBtn.cloneNode(true) as HTMLButtonElement;
      sortOrderBtn.parentNode?.replaceChild(newSortOrderBtn, sortOrderBtn);
      newSortOrderBtn.addEventListener("click", () => {
        this.currentSortOrder =
          this.currentSortOrder === "asc" ? "desc" : "asc";
        newSortOrderBtn.textContent = this.currentSortOrder.toUpperCase();
        this.currentPage = 1;
        this.fetchTableData();
      });
    }

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
    const sortSelect = document.getElementById(
      "modal-table-sort-col",
    ) as HTMLSelectElement | null;
    const sortOrderBtn = document.getElementById(
      "modal-table-sort-order",
    ) as HTMLButtonElement | null;

    if (!loading || !thead || !tbody || !prevBtn || !nextBtn || !indicator)
      return;

    loading.style.display = "block";
    prevBtn.disabled = true;
    nextBtn.disabled = true;
    const modalExportCsvBtn = document.getElementById(
      "modal-export-csv-btn",
    ) as HTMLButtonElement | null;
    const modalExportJsonBtn = document.getElementById(
      "modal-export-json-btn",
    ) as HTMLButtonElement | null;
    if (modalExportCsvBtn) modalExportCsvBtn.disabled = true;
    if (modalExportJsonBtn) modalExportJsonBtn.disabled = true;

    try {
      const limit = this.ROWS_PER_PAGE;
      const offset = (this.currentPage - 1) * limit;
      let url = `/api/table/${encodeURIComponent(this.currentTable)}?limit=${limit}&offset=${offset}`;
      if (this.currentSortBy) {
        url += `&sort_by=${encodeURIComponent(this.currentSortBy)}&order=${this.currentSortOrder}`;
      }
      if (this.currentSearchTerm) {
        url += `&search=${encodeURIComponent(this.currentSearchTerm)}`;
      }

      const response = await fetchWithBackendError(url);
      const data = await response.json();

      loading.style.display = "none";
      const totalPagesStr = data.total_pages ? ` / ${data.total_pages}` : "";
      indicator.textContent = `${i18next.t("app.page", { page: this.currentPage })}${totalPagesStr}`;

      if (data.rows.length === 0 && this.currentPage > 1) {
        // No more rows: navigate back to previous page
        this.currentPage--;
        await this.fetchTableData();
        return;
      }

      this.currentModalRows = data.rows;
      thead.innerHTML = "";
      tbody.innerHTML = "";

      if (data.rows.length > 0) {
        // Render header
        const columns = Object.keys(data.rows[0]);

        // Populate sortSelect options if not already populated with these columns
        if (sortSelect && sortSelect.options.length <= 1) {
          columns.forEach((col) => {
            const opt = document.createElement("option");
            opt.value = col;
            opt.textContent = col;
            sortSelect.appendChild(opt);
          });
        }
        if (sortSelect) {
          sortSelect.value = this.currentSortBy;
        }
        if (sortOrderBtn) {
          sortOrderBtn.textContent = this.currentSortOrder.toUpperCase();
        }

        const trHead = document.createElement("tr");
        columns.forEach((col) => {
          const th = document.createElement("th");
          th.textContent = col;
          th.setAttribute("scope", "col");
          th.setAttribute("tabindex", "0");
          th.setAttribute("role", "columnheader");
          th.setAttribute(
            "aria-sort",
            this.currentSortBy === col
              ? this.currentSortOrder === "asc"
                ? "ascending"
                : "descending"
              : "none",
          );

          if (this.currentSortBy === col) {
            th.classList.add(
              this.currentSortOrder === "asc" ? "sorted-asc" : "sorted-desc",
            );
          }

          const onThSort = () => {
            if (this.currentSortBy === col) {
              this.currentSortOrder =
                this.currentSortOrder === "asc" ? "desc" : "asc";
            } else {
              this.currentSortBy = col;
              this.currentSortOrder = "asc";
            }
            this.currentPage = 1;
            this.fetchTableData();
          };

          th.addEventListener("click", onThSort);
          th.addEventListener("keydown", (e: KeyboardEvent) => {
            if (e.key === "Enter" || e.key === " ") {
              e.preventDefault();
              onThSort();
            }
          });

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
        const hasMore = data.total_pages
          ? this.currentPage < data.total_pages
          : data.rows.length >= limit;
        nextBtn.disabled = !hasMore;
        if (modalExportCsvBtn) modalExportCsvBtn.disabled = false;
        if (modalExportJsonBtn) modalExportJsonBtn.disabled = false;
        this.announce(i18next.t("app.tableData") + " loaded", false);
      } else {
        const msg = i18next.t("ui.noDataAvailable");
        this.announce(msg, false);
        tbody.innerHTML = "";
        const tr = document.createElement("tr");
        const td = document.createElement("td");
        td.textContent = msg;
        tr.appendChild(td);
        tbody.appendChild(tr);
        prevBtn.disabled = true;
        nextBtn.disabled = true;
      }
    } catch (error) {
      console.error("Failed to fetch table data:", error);
      loading.style.display = "none";
      const errMsg = i18next.t("ui.failedData", {
        error,
        interpolation: { escapeValue: false },
      });
      this.announce(errMsg, true);
      tbody.innerHTML = "";
      const tr = document.createElement("tr");
      const td = document.createElement("td");
      td.className = "error-text";
      td.setAttribute("role", "alert");
      td.setAttribute("aria-live", "assertive");
      td.textContent = errMsg;
      tr.appendChild(td);
      tbody.appendChild(tr);
    }
  }

  /**
   * Export row objects as a downloadable CSV file.
   * @param {Array<Record<string, unknown>>} rows - The data rows to export.
   * @param {string} filename - The filename for the downloaded file.
   */
  public exportRowsToCsv(
    rows: Array<Record<string, unknown>>,
    filename: string,
  ): void {
    if (!rows || rows.length === 0) return;
    const columns = Object.keys(rows[0]);
    const headerLine = columns
      .map((col) => `"${col.replace(/"/g, '""')}"`)
      .join(",");
    const dataLines = rows.map((row) =>
      columns
        .map((col) => {
          const val = row[col];
          if (val === null || val === undefined) return '""';
          const strVal =
            typeof val === "object" ? JSON.stringify(val) : String(val);
          return `"${strVal.replace(/"/g, '""')}"`;
        })
        .join(","),
    );
    const csvContent = "\uFEFF" + [headerLine, ...dataLines].join("\r\n");
    const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.setAttribute("href", url);
    link.setAttribute("download", filename);
    link.style.display = "none";
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  }

  /**
   * Export row objects as a downloadable JSON file.
   * @param {Array<Record<string, unknown>>} rows - The data rows to export.
   * @param {string} filename - The filename for the downloaded file.
   */
  public exportRowsToJson(
    rows: Array<Record<string, unknown>>,
    filename: string,
  ): void {
    if (!rows || rows.length === 0) return;
    const jsonContent = JSON.stringify(rows, null, 2);
    const blob = new Blob([jsonContent], {
      type: "application/json;charset=utf-8;",
    });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.setAttribute("href", url);
    link.setAttribute("download", filename);
    link.style.display = "none";
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  }
}
