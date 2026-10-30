/**
 * @file system-status-controller.ts
 * @module controllers/system-status-controller
 * Composable sub-controller and diagnostic status types for System Health polling,
 * status chip updates, and diagnostic alert banners.
 */

import i18next from "i18next";
import { fetchWithBackendError, parseResponseJson } from "../api-client";
import { formatNumber } from "../formatters";
import { type ChatState } from "../state";

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

const VALID_DB_STATUS_CODES: ReadonlySet<DbStatusCode> = new Set<DbStatusCode>([
  "healthy",
  "missing_file",
  "empty_db",
  "missing_initial_data",
  "corrupted_or_unreadable",
  "not_configured",
  "unknown",
]);

/**
 * Type guard checking whether a string is a valid `DbStatusCode`.
 * @param {string | undefined} code - Raw status code from API response.
 * @returns {code is DbStatusCode} True if code is a known `DbStatusCode`.
 */
export function isDbStatusCode(code: string | undefined): code is DbStatusCode {
  return (
    typeof code === "string" && VALID_DB_STATUS_CODES.has(code as DbStatusCode)
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
}

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
  /** Header status chip button displaying system health. */
  systemStatusChip: HTMLButtonElement | null;
  /** Text span inside the system status chip. */
  statusChipText: HTMLElement | null;
  /** Alert banner element for displaying database or backend warnings. */
  systemStatusBanner: HTMLElement | null;
  /** Heading element inside the system status banner. */
  bannerTitle: HTMLElement | null;
  /** Description paragraph inside the system status banner. */
  bannerDesc: HTMLElement | null;
  /** Icon span inside the system status banner. */
  bannerIcon: HTMLElement | null;
  /** Retry connection button inside the system status banner. */
  bannerRetryBtn: HTMLButtonElement | null;
  /** Button to toggle setup instructions inside the system status banner. */
  bannerInstructionsBtn: HTMLButtonElement | null;
  /** Collapsible drawer containing CLI setup instructions. */
  bannerInstructionsDrawer: HTMLElement | null;
  /** Dropdown select element for choosing the active LLM or SQL engine. */
  modelSelect: HTMLSelectElement;
  /** Re-renders the host UI. */
  render(): void;
  /** Checks system status (allows host spy delegation). */
  checkSystemStatus(isManualRetry?: boolean, dbPath?: string): Promise<void>;
}

/**
 * Composable sub-controller for System Status & Diagnostics.
 */
export class SystemStatusController {
  /** Host UI reference. */
  private readonly ui: SystemStatusHost;
  /** Interval timer ID for periodic system status polling, or null when inactive. */
  private statusPollTimer: number | null = null;

  /**
   * Creates a new `SystemStatusController`.
   * @param {SystemStatusHost} ui - Host UI instance.
   */
  constructor(ui: SystemStatusHost) {
    this.ui = ui;
  }

  /**
   * Starts periodic background polling of `/api/status`.
   * @param {number} intervalMs - Polling interval in milliseconds.
   * @returns {void}
   */
  public startPolling(intervalMs: number): void {
    this.stopPolling();
    this.statusPollTimer = window.setInterval(() => {
      void this.ui.checkSystemStatus();
    }, intervalMs);
  }

  /**
   * Stops any active background system status polling interval.
   * @returns {void}
   */
  public stopPolling(): void {
    if (this.statusPollTimer !== null) {
      window.clearInterval(this.statusPollTimer);
      this.statusPollTimer = null;
    }
  }

  /**
   * Shows the system status alert banner with the specified severity and messaging.
   * @param {"danger" | "warning" | "info"} severity - The alert level.
   * @param {string} title - The title to display.
   * @param {string} desc - The description message.
   * @param {string} icon - The icon to render.
   * @param {boolean} showRetry - Whether to show the retry button.
   * @returns {void}
   */
  public showBanner(
    severity: "danger" | "warning" | "info",
    title: string,
    desc: string,
    icon: string,
    showRetry: boolean,
  ): void {
    if (!this.ui.systemStatusBanner) return;
    this.ui.systemStatusBanner.className = `system-status-banner banner-${severity}`;
    if (this.ui.bannerIcon) this.ui.bannerIcon.textContent = icon;
    if (this.ui.bannerTitle) this.ui.bannerTitle.textContent = title;
    if (this.ui.bannerDesc) this.ui.bannerDesc.textContent = desc;
    if (this.ui.bannerRetryBtn) {
      this.ui.bannerRetryBtn.style.display = showRetry
        ? "inline-block"
        : "none";
    }
    if (!showRetry) {
      this.ui.bannerInstructionsDrawer?.classList.add("hidden");
      this.ui.bannerInstructionsBtn?.setAttribute("aria-expanded", "false");
    }
    this.ui.systemStatusBanner.classList.remove("hidden");
    this.ui.systemStatusBanner.removeAttribute("aria-hidden");
  }

  /**
   * Hides the system status alert banner.
   * @returns {void}
   */
  public hideBanner(): void {
    if (this.ui.systemStatusBanner) {
      this.ui.systemStatusBanner.classList.add("hidden");
      this.ui.systemStatusBanner.setAttribute("aria-hidden", "true");
    }
    this.ui.bannerInstructionsDrawer?.classList.add("hidden");
    this.ui.bannerInstructionsBtn?.setAttribute("aria-expanded", "false");
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
    if (!this.ui.systemStatusChip) return;
    this.ui.systemStatusChip.className = `status-chip status-${level}`;
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
    this.ui.systemStatusChip.setAttribute("aria-label", tooltip);
    this.ui.systemStatusChip.setAttribute("title", tooltip);
    if (this.ui.statusChipText) {
      this.ui.statusChipText.textContent = text;
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
    if (isManualRetry && this.ui.bannerRetryBtn) {
      this.ui.bannerRetryBtn.textContent = i18next.t("status.reconnecting");
    }

    try {
      const targetDb = dbPath || this.ui.state.currentDb;
      const url =
        targetDb && targetDb !== "t1d.duckdb"
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
      });

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
        if (this.ui.modelSelect && this.ui.modelSelect.value !== "sql") {
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
      if (this.ui.bannerRetryBtn) {
        this.ui.bannerRetryBtn.textContent = i18next.t("status.retry");
      }
    }
  }
}
