/**
 * @file system-status-controller.ts
 * @module controllers/system-status-controller
 * Composable sub-controller and diagnostic status types for System Health polling,
 * status chip updates, and diagnostic alert banners.
 */

import i18next from "i18next";
import { fetchWithBackendError, parseResponseJson } from "../api-client";
import { formatNumber } from "../formatters";
import { requireElementById } from "../modal-utils";
import { DEFAULT_RECOMMENDED_JAEB_DB_PATH, type ChatState } from "../state";

/**
 * Known database diagnostic status codes returned by the backend.
 */
export type DbStatusCode =
  | "healthy"
  | "missing_file"
  | "empty_db"
  | "missing_initial_data"
  | "corrupted_or_unreadable"
  | "not_configured"
  | "unknown";

/**
 * Type guard checking whether a string is a valid `DbStatusCode`.
 * @param {string | undefined} code - Raw status code from API response.
 * @returns {code is DbStatusCode} True if code is a known `DbStatusCode`.
 */
export function isDbStatusCode(code: string | undefined): code is DbStatusCode {
  return (
    code === "healthy" ||
    code === "missing_file" ||
    code === "empty_db" ||
    code === "missing_initial_data" ||
    code === "corrupted_or_unreadable" ||
    code === "not_configured" ||
    code === "unknown"
  );
}

/**
 * Database health diagnostics block returned by `/api/status` and `/api/schema`.
 */
export interface SystemStatusDbPayload {
  /** Whether the DuckDB file exists on disk. */
  exists?: boolean;
  /** Whether a read-only connection succeeded. */
  connected?: boolean;
  /** Diagnostic status code string. */
  status_code?: string;
  /** Configured filesystem path for the database. */
  configured_path?: string;
  /** Human-readable diagnostic message. */
  message?: string | null;
  /** Suggested remediation command or action. */
  remediation?: string | null;
  /** Number of tables discovered in the database. */
  table_count?: number;
  /** Whether initial clinical tables are present. */
  has_initial_data?: boolean;
  /** Database file size in bytes. */
  file_size_bytes?: number;
  /** Whether the database directory/file is writable. */
  writable?: boolean;
  /** Whether PRAGMA integrity checks passed. */
  integrity_ok?: boolean;
  /** Free disk space in bytes. */
  disk_free_bytes?: number;
  /** Database provenance ('jaeb_authentic', 'synthetic_seed', 'custom'). */
  data_provenance?: string;
  /** Whether the database is using synthetic/faked seed data instead of authentic Jaeb datasets. */
  is_synthetic_faked_data?: boolean;
  /** Recommended authentic Jaeb DuckDB path for the current host. */
  recommended_jaeb_db_path?: string;
}

/**
 * Alias for `SystemStatusApiResponse` representing the `/api/status` health payload.
 */
export type SystemStatusResponsePayload = SystemStatusApiResponse;

/**
 * Response payload returned by `/api/status`.
 */
export interface SystemStatusApiResponse {
  /** Overall system health status ('healthy', 'degraded', 'error', 'offline'). */
  status: "healthy" | "degraded" | "error" | "offline";
  /** Database health diagnostics block. */
  database?: SystemStatusDbPayload;
  /** Local Ollama service health diagnostics block. */
  ollama?: {
    /** Whether Ollama is reachable over HTTP. */
    accessible?: boolean;
    /** Diagnostic message for Ollama. */
    message?: string | null;
    /** Remediation guidance for Ollama. */
    remediation?: string | null;
    /** Detected Ollama version string. */
    version?: string | null;
  };
}

/**
 * Host contract required by `SystemStatusController`.
 */
export interface SystemStatusHost {
  /** Global application state store. */
  state: ChatState;
  /** Dropdown select element for choosing the active LLM or SQL engine. */
  modelSelect: HTMLSelectElement;
  /** Reloads available models in the model selector. */
  reloadModels(): Promise<void>;
  /** Reloads the database schema in the schema explorer. */
  reloadSchema(dbPath?: string): Promise<void>;
  /** Re-renders the host UI. */
  render(): void;
}

/**
 * Composable sub-controller for System Status & Diagnostics.
 */
export class SystemStatusController {
  /** Host UI reference. */
  private readonly ui: SystemStatusHost;
  /** Header status chip button displaying system health. */
  public readonly systemStatusChip: HTMLButtonElement;
  /** Text span inside the system status chip. */
  public readonly statusChipText: HTMLElement;
  /** Alert banner element for displaying database or backend warnings. */
  public readonly systemStatusBanner: HTMLElement;
  /** Heading element inside the system status banner. */
  public readonly bannerTitle: HTMLElement;
  /** Description paragraph inside the system status banner. */
  public readonly bannerDesc: HTMLElement;
  /** Icon span inside the system status banner. */
  public readonly bannerIcon: HTMLElement;
  /** Retry connection button inside the system status banner. */
  public readonly bannerRetryBtn: HTMLButtonElement;
  /** Button to toggle setup instructions inside the system status banner. */
  public readonly bannerInstructionsBtn: HTMLButtonElement;
  /** Collapsible drawer containing CLI setup instructions. */
  public readonly bannerInstructionsDrawer: HTMLElement;
  /** Button to dismiss the system status alert banner. */
  public readonly bannerDismissBtn: HTMLButtonElement;
  /** Alert row below navbar warning when synthetic/faked non-Jaeb data is active. */
  public readonly jaebProvenanceAlert: HTMLElement;
  /** Description span inside the synthetic data alert row. */
  public readonly jaebProvenanceAlertDesc: HTMLElement;
  /** Dismiss ✖ button for the synthetic data alert row. */
  public readonly jaebProvenanceAlertDismissBtn: HTMLButtonElement;
  /** Alert row below navbar warning when deterministic rule-based SQL generation is active instead of an LLM. */
  public readonly ruleBasedSqlAlert: HTMLElement;
  /** Dismiss ✖ button for the rule-based SQL alert row. */
  public readonly ruleBasedSqlAlertDismissBtn: HTMLButtonElement;
  /** Interval timer ID for periodic system status polling, or null when inactive. */
  private statusPollTimer: ReturnType<typeof setInterval> | null = null;

  /**
   * Creates a new `SystemStatusController`.
   * @param {SystemStatusHost} ui - Host UI instance.
   */
  constructor(ui: SystemStatusHost) {
    this.ui = ui;
    this.systemStatusChip =
      requireElementById<HTMLButtonElement>("system-status-chip");
    this.statusChipText = requireElementById<HTMLElement>("status-chip-text");
    this.systemStatusBanner = requireElementById<HTMLElement>(
      "system-status-banner",
    );
    this.bannerTitle = requireElementById<HTMLElement>("banner-title");
    this.bannerDesc = requireElementById<HTMLElement>("banner-desc");
    this.bannerIcon = requireElementById<HTMLElement>("banner-icon");
    this.bannerRetryBtn =
      requireElementById<HTMLButtonElement>("banner-retry-btn");
    this.bannerInstructionsBtn = requireElementById<HTMLButtonElement>(
      "banner-instructions-btn",
    );
    this.bannerInstructionsDrawer = requireElementById<HTMLElement>(
      "banner-instructions-drawer",
    );
    this.bannerDismissBtn =
      requireElementById<HTMLButtonElement>("banner-dismiss-btn");
    this.jaebProvenanceAlert = requireElementById<HTMLElement>(
      "jaeb-provenance-alert",
    );
    this.jaebProvenanceAlertDesc = requireElementById<HTMLElement>(
      "jaeb-provenance-alert-desc",
    );
    this.jaebProvenanceAlertDismissBtn = requireElementById<HTMLButtonElement>(
      "jaeb-provenance-alert-dismiss",
    );
    this.ruleBasedSqlAlert = requireElementById<HTMLElement>(
      "rule-based-sql-alert",
    );
    this.ruleBasedSqlAlertDismissBtn = requireElementById<HTMLButtonElement>(
      "rule-based-sql-alert-dismiss",
    );

    this.bindEvents();
  }

  /**
   * Binds DOM event listeners for the status chip and diagnostic alert banner.
   * @returns {void}
   */
  private bindEvents(): void {
    this.bannerRetryBtn.addEventListener("click", () => {
      void this.check(true);
      void this.ui.reloadModels();
      void this.ui.reloadSchema();
    });

    this.bannerInstructionsBtn.addEventListener("click", () => {
      const isHidden = this.bannerInstructionsDrawer.classList.toggle("hidden");
      this.bannerInstructionsBtn.setAttribute(
        "aria-expanded",
        isHidden ? "false" : "true",
      );
    });

    this.bannerDismissBtn.addEventListener("click", () => {
      this.hideBanner();
    });

    this.jaebProvenanceAlertDismissBtn.addEventListener("click", () => {
      this.hideSyntheticDataAlert();
    });

    this.jaebProvenanceAlert.addEventListener("keydown", (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        this.hideSyntheticDataAlert();
      }
    });

    this.ruleBasedSqlAlertDismissBtn.addEventListener("click", () => {
      this.hideRuleBasedSqlAlert();
    });

    this.ruleBasedSqlAlert.addEventListener("keydown", (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        this.hideRuleBasedSqlAlert();
      }
    });

    this.systemStatusChip.addEventListener("click", () => {
      void this.check(true);
    });

    i18next.on("languageChanged", () => {
      const jaebPath =
        this.ui.state.systemStatus.recommendedJaebDbPath ||
        DEFAULT_RECOMMENDED_JAEB_DB_PATH;
      this.jaebProvenanceAlertDesc.textContent = i18next.t(
        "status.syntheticDataAlertDesc",
        { jaebPath },
      );
    });
  }

  /**
   * Shows the visual alert row below the navbar indicating synthetic/faked non-Jaeb data is active.
   * @param {string} [recommendedPath] - Optional recommended Jaeb database path to display.
   * @returns {void}
   */
  public showSyntheticDataAlert(recommendedPath?: string): void {
    const jaebPath =
      recommendedPath ||
      this.ui.state.systemStatus.recommendedJaebDbPath ||
      DEFAULT_RECOMMENDED_JAEB_DB_PATH;
    this.jaebProvenanceAlertDesc.textContent = i18next.t(
      "status.syntheticDataAlertDesc",
      { jaebPath },
    );
    this.jaebProvenanceAlert.classList.remove("hidden");
    this.jaebProvenanceAlert.removeAttribute("aria-hidden");
  }

  /**
   * Hides the visual alert row below the navbar for synthetic/faked non-Jaeb data.
   * @returns {void}
   */
  public hideSyntheticDataAlert(): void {
    this.jaebProvenanceAlert.classList.add("hidden");
    this.jaebProvenanceAlert.setAttribute("aria-hidden", "true");
  }

  /**
   * Shows the visual alert row below the navbar indicating deterministic rule-based SQL generation is active instead of an LLM.
   * @returns {void}
   */
  public showRuleBasedSqlAlert(): void {
    this.ruleBasedSqlAlert.classList.remove("hidden");
    this.ruleBasedSqlAlert.removeAttribute("aria-hidden");
  }

  /**
   * Hides the visual alert row below the navbar for deterministic rule-based SQL generation.
   * @returns {void}
   */
  public hideRuleBasedSqlAlert(): void {
    this.ruleBasedSqlAlert.classList.add("hidden");
    this.ruleBasedSqlAlert.setAttribute("aria-hidden", "true");
  }

  /**
   * Starts periodic background polling of `/api/status`.
   * @param {number} [intervalMs=30000] - Polling interval in milliseconds.
   * @returns {void}
   */
  public startPolling(intervalMs: number = 30000): void {
    this.stopPolling();
    this.statusPollTimer = setInterval(() => {
      void this.check();
    }, intervalMs);
  }

  /**
   * Stops any active background system status polling interval.
   * @returns {void}
   */
  public stopPolling(): void {
    if (this.statusPollTimer !== null) {
      clearInterval(this.statusPollTimer);
      this.statusPollTimer = null;
    }
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
    this.systemStatusBanner.className = `system-status-banner banner-${severity}`;
    this.bannerIcon.textContent = icon;
    this.bannerTitle.textContent = title;
    this.bannerDesc.textContent = desc;
    this.bannerRetryBtn.style.display = showRetry ? "inline-block" : "none";
    if (!showRetry) {
      this.bannerInstructionsDrawer.classList.add("hidden");
      this.bannerInstructionsBtn.setAttribute("aria-expanded", "false");
    }
    this.systemStatusBanner.classList.remove("hidden");
    this.systemStatusBanner.removeAttribute("aria-hidden");
  }

  /**
   * Hides the system status alert banner.
   * @returns {void}
   */
  public hideBanner(): void {
    this.systemStatusBanner.classList.add("hidden");
    this.systemStatusBanner.setAttribute("aria-hidden", "true");
    this.bannerInstructionsDrawer.classList.add("hidden");
    this.bannerInstructionsBtn.setAttribute("aria-expanded", "false");
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
    this.systemStatusChip.className = `status-chip status-${level}`;
    let tooltip: string;
    if (this.ui.state.systemStatus.dbFileSizeBytes) {
      const mbNum = this.ui.state.systemStatus.dbFileSizeBytes / (1024 * 1024);
      const mb = formatNumber(mbNum, {
        minimumFractionDigits: 1,
        maximumFractionDigits: 1,
      });
      const tables = formatNumber(this.ui.state.systemStatus.tableCount);
      tooltip = i18next.t("status.chipTooltipDb", {
        status: text,
        mb,
        tables,
        defaultValue: `System status: ${text} | DB: ${mb} MB (${tables} tables)`,
      });
    } else {
      tooltip = i18next.t("status.chipTooltip", {
        status: text,
        defaultValue: `System status: ${text}`,
      });
    }
    this.systemStatusChip.setAttribute("aria-label", tooltip);
    this.systemStatusChip.setAttribute("title", tooltip);
    this.statusChipText.textContent = text;
  }

  /**
   * Applies banner and status chip updates for a parsed `SystemStatusApiResponse`.
   * @param {SystemStatusApiResponse} health - Parsed `/api/status` response.
   * @returns {void}
   */
  public applyHealthPayload(health: SystemStatusApiResponse): void {
    if (
      health.database?.is_synthetic_faked_data === false ||
      health.database?.data_provenance === "jaeb_authentic"
    ) {
      this.hideSyntheticDataAlert();
    } else if (health.database?.is_synthetic_faked_data === true) {
      const jaebPath =
        health.database?.recommended_jaeb_db_path ||
        DEFAULT_RECOMMENDED_JAEB_DB_PATH;
      this.showSyntheticDataAlert(jaebPath);
    }

    if (!health.ollama?.accessible || this.ui.modelSelect.value === "sql") {
      this.showRuleBasedSqlAlert();
    } else {
      this.hideRuleBasedSqlAlert();
    }

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
      this.updateStatusChip("degraded", i18next.t("status.dbMissingDataTitle"));
    } else if (!health.ollama?.accessible) {
      this.showBanner(
        "info",
        i18next.t("status.ollamaOfflineTitle"),
        i18next.t("status.ollamaOfflineDesc"),
        "ℹ️",
        false,
      );
      this.updateStatusChip("degraded", i18next.t("status.ollamaOfflineTitle"));
      if (this.ui.modelSelect.value !== "sql") {
        this.ui.modelSelect.value = "sql";
        const currentChat = this.ui.state.getActiveChat();
        if (currentChat && currentChat.messages.length === 0) {
          this.ui.state.setActiveChatModel("sql");
        }
      }
    } else {
      this.hideBanner();
      this.updateStatusChip("healthy", i18next.t("status.systemHealthy"));
    }
  }

  /**
   * Checks system health and updates UI indicators accordingly.
   * @param {boolean} [isManualRetry=false] - Whether this is a manual user retry.
   * @param {string} [dbPath] - Optional custom database path.
   * @returns {Promise<void>}
   */
  public async check(
    isManualRetry: boolean = false,
    dbPath?: string,
  ): Promise<void> {
    if (isManualRetry) {
      this.bannerRetryBtn.textContent = i18next.t("status.reconnecting");
    }

    try {
      const targetDb = dbPath || this.ui.state.currentDb;
      const url =
        targetDb !== "t1d.duckdb"
          ? `/api/status?db_path=${encodeURIComponent(targetDb)}`
          : "/api/status";
      const response = await fetchWithBackendError(url);
      const health = await parseResponseJson<SystemStatusApiResponse>(response);

      this.ui.state.setSystemStatus({
        backendOnline: true,
        status: health.status,
        dbConfigured: Boolean(health.database?.exists),
        dbExists: Boolean(health.database?.exists),
        dbConnected: Boolean(health.database?.connected),
        dbStatusCode: isDbStatusCode(health.database?.status_code)
          ? health.database.status_code
          : "unknown",
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
        recommendedJaebDbPath:
          health.database?.recommended_jaeb_db_path ||
          DEFAULT_RECOMMENDED_JAEB_DB_PATH,
      });

      this.applyHealthPayload(health);
      this.ui.render();
    } catch {
      this.ui.state.setSystemStatus({
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
      this.ui.render();
    } finally {
      this.bannerRetryBtn.textContent = i18next.t("status.retry");
    }
  }
}
