/**
 * @file schema-explorer-controller.ts
 * @module controllers/schema-explorer-controller
 * Composable sub-controller managing the Database Selector dropdown,
 * Schema Explorer sidebar tree, and diagnostic empty-state cards.
 */

import i18next from "i18next";
import { fetchWithBackendError, parseResponseJson } from "../api-client";
import { formatNumber } from "../formatters";
import { createSchemaDiagnosticCard, createSvgIcon } from "../message-renderer";
import { type ChatState } from "../state";
import {
  isDbStatusCode,
  type SystemStatusDbPayload,
} from "./system-status-controller";

/**
 * Response payload returned by `/api/schema`.
 */
export interface SchemaExplorerApiResponse {
  /** Database health diagnostics block. */
  database?: SystemStatusDbPayload;
  /** Array of discovered database tables and their columns. */
  tables?: Array<{
    /** Table identifier. */
    name: string;
    /** Columns belonging to the table. */
    columns: Array<{
      /** Column identifier. */
      name: string;
      /** SQL data type string. */
      type: string;
    }>;
  }>;
}

/**
 * Response payload returned by `/api/databases`.
 */
export interface DatabasesListApiResponse {
  /** Discovered workspace DuckDB databases. */
  databases?: Array<{
    /** Database filename. */
    name: string;
    /** Database path. */
    path: string;
    /** File size in bytes. */
    size_bytes: number;
  }>;
  /** Currently active default database path. */
  current_db?: string;
}

/**
 * Host contract required by `SchemaExplorerController`.
 */
export interface SchemaExplorerHost {
  /** Global application state store. */
  state: ChatState;
  /** Dropdown select element for choosing the active DuckDB database file. */
  dbSelect: HTMLSelectElement | null;
  /** Shows the system status alert banner. */
  showBanner(
    severity: "danger" | "warning" | "info",
    title: string,
    desc: string,
    icon?: string,
    showRetry?: boolean,
  ): void;
  /** Hides the system status alert banner. */
  hideBanner(): void;
  /** Updates the status chip in the top header. */
  updateStatusChip(
    level: "healthy" | "degraded" | "error" | "offline",
    text: string,
  ): void;
  /** Checks system health. */
  checkSystemStatus(isManualRetry?: boolean, dbPath?: string): Promise<void>;
  /** Opens the table preview modal for a table. */
  openTablePreview(tableName: string): Promise<void>;
  /** Copies a CLI command to the clipboard. */
  copyCommandToClipboard(cmd: string, btn: HTMLButtonElement): Promise<void>;
  /** Announces a message to screen readers. */
  announce(message: string, isAssertive?: boolean): void;
}

/**
 * Composable sub-controller for the Schema Explorer sidebar and database selector.
 */
export class SchemaExplorerController {
  /** Host UI reference. */
  private readonly ui: SchemaExplorerHost;

  /**
   * Creates a new `SchemaExplorerController`.
   * @param {SchemaExplorerHost} ui - Host UI instance.
   */
  constructor(ui: SchemaExplorerHost) {
    this.ui = ui;
  }

  /**
   * Binds interaction events to action buttons inside diagnostic cards.
   * @param {HTMLElement} container - The container holding diagnostic cards.
   * @returns {void}
   */
  public bindSchemaDiagnosticButtons(container: HTMLElement): void {
    container
      .querySelectorAll<HTMLButtonElement>(".copy-cmd-btn")
      .forEach((btn) => {
        btn.addEventListener("click", () => {
          const cmd = btn.dataset.cmd;
          if (cmd) {
            void this.ui.copyCommandToClipboard(cmd, btn);
          }
        });
      });

    container.querySelectorAll(".reload-schema-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        void this.loadSchema();
        void this.ui.checkSystemStatus(true);
      });
    });
  }

  /**
   * Fetches available DuckDB databases from the backend and updates the dropdown.
   * @returns {Promise<void>}
   */
  public async fetchDatabases(): Promise<void> {
    if (!this.ui.dbSelect) return;
    try {
      const resp = await fetch("/api/databases");
      if (!resp.ok) return;
      const data = await parseResponseJson<DatabasesListApiResponse>(resp);
      const dbs: { name: string; path: string; size_bytes: number }[] =
        data.databases || [];
      this.ui.state.setAvailableDbs(dbs, data.current_db);
      this.renderDatabaseDropdown();
    } catch {
      // Ignore network failures fetching database list
    }
  }

  /**
   * Renders options in the database selector dropdown.
   * @returns {void}
   */
  public renderDatabaseDropdown(): void {
    if (!this.ui.dbSelect) return;
    this.ui.dbSelect.replaceChildren();
    if (this.ui.state.availableDbs.length === 0) {
      const opt = document.createElement("option");
      opt.value = this.ui.state.currentDb || "t1d.duckdb";
      opt.textContent = opt.value;
      this.ui.dbSelect.appendChild(opt);
      return;
    }
    for (const db of this.ui.state.availableDbs) {
      const opt = document.createElement("option");
      opt.value = db.name;
      const kbSize = formatNumber(Math.round(db.size_bytes / 1024));
      opt.textContent = i18next.t("ui.dbSizeKb", {
        name: db.name,
        size: kbSize,
        defaultValue: `${db.name} (${kbSize} KB)`,
      });
      if (
        db.name === this.ui.state.currentDb ||
        db.path === this.ui.state.currentDb
      ) {
        opt.selected = true;
      }
      this.ui.dbSelect.appendChild(opt);
    }
  }

  /**
   * Fetches the database schema and renders it in the sidebar.
   * @param {string} [dbPath] - Optional custom database path.
   * @returns {Promise<void>}
   */
  public async loadSchema(dbPath?: string): Promise<void> {
    const schemaContent = document.getElementById("schema-content");
    if (!schemaContent) return;

    try {
      const targetDb = dbPath || this.ui.state.currentDb;
      const url =
        targetDb && targetDb !== "t1d.duckdb"
          ? `/api/schema?db_path=${encodeURIComponent(targetDb)}`
          : "/api/schema";
      const response = await fetchWithBackendError(url);
      const data = await parseResponseJson<SchemaExplorerApiResponse>(response);

      schemaContent.replaceChildren();

      if (data.database) {
        const dbStatus = data.database;
        const isHealthy = dbStatus.status_code === "healthy";
        this.ui.state.setSystemStatus({
          backendOnline: true,
          status: isHealthy ? "healthy" : "degraded",
          dbConfigured: Boolean(dbStatus.exists),
          dbExists: Boolean(dbStatus.exists),
          dbConnected: Boolean(dbStatus.connected),
          dbStatusCode: isDbStatusCode(dbStatus.status_code)
            ? dbStatus.status_code
            : "unknown",
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
          this.ui.showBanner(
            "warning",
            i18next.t("status.dbMissingTitle"),
            i18next.t("status.dbMissingDesc", {
              path: dbStatus.configured_path,
            }),
            "📁",
            true,
          );
          this.ui.updateStatusChip("error", i18next.t("status.dbMissingTitle"));
        } else if (dbStatus.status_code === "empty_db") {
          this.ui.showBanner(
            "warning",
            i18next.t("status.dbEmptyTitle"),
            i18next.t("status.dbEmptyDesc"),
            "🗄️",
            true,
          );
          this.ui.updateStatusChip(
            "degraded",
            i18next.t("status.dbEmptyTitle"),
          );
        } else if (dbStatus.status_code === "missing_initial_data") {
          this.ui.showBanner(
            "warning",
            i18next.t("status.dbMissingDataTitle"),
            i18next.t("status.dbMissingDataDesc"),
            "📊",
            true,
          );
          this.ui.updateStatusChip(
            "degraded",
            i18next.t("status.dbMissingDataTitle"),
          );
        } else {
          this.ui.hideBanner();
          this.ui.updateStatusChip(
            "healthy",
            i18next.t("status.systemHealthy"),
          );
        }
      }

      if (!data.tables || data.tables.length === 0) {
        const dbStatus = data.database;
        if (dbStatus && dbStatus.status_code === "missing_file") {
          schemaContent.replaceChildren(
            createSchemaDiagnosticCard({
              icon: "📁",
              title: i18next.t("status.dbMissingTitle"),
              description: i18next.t("status.dbMissingDesc", {
                path: dbStatus.configured_path,
              }),
              command: `t1d-analytics load --db ${dbStatus.configured_path}`,
              showReload: true,
              isAlert: true,
            }),
          );
        } else if (dbStatus && dbStatus.status_code === "empty_db") {
          schemaContent.replaceChildren(
            createSchemaDiagnosticCard({
              icon: "🗄️",
              title: i18next.t("status.dbEmptyTitle"),
              description: i18next.t("status.dbEmptyDesc"),
              command: "t1d-analytics load",
              showReload: true,
              isAlert: true,
            }),
          );
        } else {
          const noTablesDiv = document.createElement("div");
          noTablesDiv.className = "schema-loading";
          noTablesDiv.setAttribute("role", "status");
          noTablesDiv.setAttribute("aria-live", "polite");
          noTablesDiv.setAttribute("data-i18n", "ui.noTables");
          noTablesDiv.textContent = i18next.t("ui.noTables");
          schemaContent.replaceChildren(noTablesDiv);
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
          toggleBtn.appendChild(
            createSvgIcon(
              12,
              [{ tag: "polyline", attrs: { points: "9 18 15 12 9 6" } }],
              "none",
              true,
            ),
          );
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
          viewBtn.appendChild(
            createSvgIcon(
              14,
              [
                {
                  tag: "rect",
                  attrs: {
                    x: "3",
                    y: "3",
                    width: "18",
                    height: "18",
                    rx: "2",
                    ry: "2",
                  },
                },
                {
                  tag: "line",
                  attrs: { x1: "3", y1: "9", x2: "21", y2: "9" },
                },
                {
                  tag: "line",
                  attrs: { x1: "9", y1: "21", x2: "9", y2: "9" },
                },
              ],
              "none",
              true,
            ),
          );

          headerDiv.appendChild(toggleBtn);
          headerDiv.appendChild(viewBtn);

          const toggleExpand = (): void => {
            const isExpanded = tableDiv.classList.toggle("expanded");
            toggleBtn.setAttribute(
              "aria-expanded",
              isExpanded ? "true" : "false",
            );
          };
          toggleBtn.addEventListener("click", toggleExpand);

          viewBtn.addEventListener("click", (e) => {
            e.stopPropagation();
            void this.ui.openTablePreview(table.name);
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
      this.ui.announce(errMsg, true);
      const backendUrl = window.location.origin;
      schemaContent.replaceChildren(
        createSchemaDiagnosticCard({
          icon: "⚠️",
          title: i18next.t("status.offlineTitle"),
          description: i18next.t("status.offlineDesc", { url: backendUrl }),
          showReload: true,
          isError: true,
        }),
      );
      this.bindSchemaDiagnosticButtons(schemaContent);
      this.ui.state.setSystemStatus({
        backendOnline: false,
        status: "offline",
      });
      this.ui.showBanner(
        "danger",
        i18next.t("status.offlineTitle"),
        i18next.t("status.offlineDesc", { url: backendUrl }),
        "⚠️",
        true,
      );
      this.ui.updateStatusChip("offline", i18next.t("status.systemOffline"));
    }
  }
}
