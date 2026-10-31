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
import { requireElementById } from "../modal-utils";
import { type ChatState } from "../state";
import {
  isDbStatusCode,
  type SystemStatusController,
  type SystemStatusDbPayload,
} from "./system-status-controller";
import { type TablePreviewModalController } from "./table-modal-controller";

/**
 * Table definition entry in `SchemaExplorerApiResponse`.
 */
export interface SchemaTableItem {
  /** Table identifier. */
  name: string;
  /** Columns belonging to the table. */
  columns: Array<{
    /** Column identifier. */
    name: string;
    /** SQL data type string. */
    type: string;
  }>;
}

/**
 * Response payload returned by `/api/schema`.
 */
export interface SchemaExplorerApiResponse {
  /** Database health diagnostics block. */
  database?: SystemStatusDbPayload;
  /** Array of discovered database tables and their columns. */
  tables?: SchemaTableItem[];
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
  /** System status sub-controller. */
  systemStatusController: SystemStatusController;
  /** Table preview modal sub-controller. */
  tableModalController: TablePreviewModalController;
  /** Announces a message to screen readers. */
  announce(message: string, isAssertive?: boolean): void;
}

/**
 * Composable sub-controller for the Schema Explorer sidebar and database selector.
 */
export class SchemaExplorerController {
  /** Host UI reference. */
  private readonly ui: SchemaExplorerHost;
  /** Dropdown select element for choosing the active DuckDB database file. */
  public readonly dbSelect: HTMLSelectElement;
  /** Schema explorer sidebar container element. */
  public readonly schemaExplorer: HTMLElement;
  /** Toggle schema explorer button element. */
  public readonly toggleSchemaBtn: HTMLButtonElement;
  /** Schema content tree container element. */
  public readonly schemaContent: HTMLElement;

  /**
   * Creates a new `SchemaExplorerController`.
   * @param {SchemaExplorerHost} ui - Host UI instance.
   */
  constructor(ui: SchemaExplorerHost) {
    this.ui = ui;
    this.dbSelect = requireElementById<HTMLSelectElement>("db-select");
    this.schemaExplorer = requireElementById<HTMLElement>("schema-explorer");
    this.toggleSchemaBtn =
      requireElementById<HTMLButtonElement>("toggle-schema-btn");
    this.schemaContent = requireElementById<HTMLElement>("schema-content");

    this.bindEvents();
  }

  /**
   * Binds DOM event listeners for database selection and schema explorer toggling.
   * @returns {void}
   */
  private bindEvents(): void {
    this.dbSelect.addEventListener("change", () => {
      const selectedDb = this.dbSelect.value;
      if (selectedDb) {
        this.ui.state.setCurrentDb(selectedDb);
        void this.loadSchema(selectedDb);
        void this.ui.systemStatusController.check(true, selectedDb);
        this.ui.announce(i18next.t("ui.databaseSwitched", { db: selectedDb }));
      }
    });

    const toggleFn = (): void => {
      const isCollapsed = this.schemaExplorer.classList.toggle("collapsed");
      this.toggleSchemaBtn.setAttribute(
        "aria-expanded",
        isCollapsed ? "false" : "true",
      );
    };

    this.toggleSchemaBtn.setAttribute("aria-expanded", "true");
    const schemaHeader = this.schemaExplorer.querySelector(".schema-header");
    if (schemaHeader) {
      schemaHeader.addEventListener("click", toggleFn);
    } else {
      this.toggleSchemaBtn.addEventListener("click", toggleFn);
    }
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
      this.ui.announce(copiedMsg);
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
    container
      .querySelectorAll<HTMLButtonElement>(".copy-cmd-btn[data-cmd]")
      .forEach((btn) => {
        btn.addEventListener("click", () => {
          const cmd = btn.getAttribute("data-cmd")!;
          void this.copyCommandToClipboard(cmd, btn);
        });
      });

    container.querySelectorAll(".reload-schema-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        void this.loadSchema();
        void this.ui.systemStatusController.check(true);
      });
    });
  }

  /**
   * Fetches available DuckDB databases from the backend and updates the dropdown.
   * @returns {Promise<void>}
   */
  public async fetchDatabases(): Promise<void> {
    try {
      const resp = await fetch("/api/databases");
      if (!resp.ok) return;
      const data = await parseResponseJson<DatabasesListApiResponse>(resp);
      const dbs = data.databases || [];
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
    this.dbSelect.replaceChildren();
    if (this.ui.state.availableDbs.length === 0) {
      const opt = document.createElement("option");
      opt.value = this.ui.state.currentDb || "t1d.duckdb";
      opt.textContent = opt.value;
      this.dbSelect.appendChild(opt);
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
      this.dbSelect.appendChild(opt);
    }
  }

  /**
   * Updates `ChatState.systemStatus`, the status banner, and the status chip from `data.database`.
   * @param {SystemStatusDbPayload} dbStatus - Database health payload.
   * @returns {void}
   */
  private applySchemaDbStatus(dbStatus: SystemStatusDbPayload): void {
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
      tableCount: Number(dbStatus.table_count) || 0,
      hasInitialData: Boolean(dbStatus.has_initial_data),
    });

    if (dbStatus.status_code === "missing_file") {
      this.ui.systemStatusController.showBanner(
        "warning",
        i18next.t("status.dbMissingTitle"),
        i18next.t("status.dbMissingDesc", { path: dbStatus.configured_path }),
        "📁",
        true,
      );
      this.ui.systemStatusController.updateStatusChip(
        "error",
        i18next.t("status.dbMissingTitle"),
      );
    } else if (dbStatus.status_code === "empty_db") {
      this.ui.systemStatusController.showBanner(
        "warning",
        i18next.t("status.dbEmptyTitle"),
        i18next.t("status.dbEmptyDesc"),
        "🗄️",
        true,
      );
      this.ui.systemStatusController.updateStatusChip(
        "degraded",
        i18next.t("status.dbEmptyTitle"),
      );
    } else if (dbStatus.status_code === "missing_initial_data") {
      this.ui.systemStatusController.showBanner(
        "warning",
        i18next.t("status.dbMissingDataTitle"),
        i18next.t("status.dbMissingDataDesc"),
        "📊",
        true,
      );
      this.ui.systemStatusController.updateStatusChip(
        "degraded",
        i18next.t("status.dbMissingDataTitle"),
      );
    } else {
      this.ui.systemStatusController.hideBanner();
      this.ui.systemStatusController.updateStatusChip(
        "healthy",
        i18next.t("status.systemHealthy"),
      );
    }
  }

  /**
   * Renders the empty-schema state into `schemaContent` when no tables are returned.
   * @param {SystemStatusDbPayload | undefined} dbStatus - Optional database health metadata.
   * @returns {void}
   */
  private renderEmptyTablesState(
    dbStatus: SystemStatusDbPayload | undefined,
  ): void {
    if (dbStatus && dbStatus.status_code === "missing_file") {
      this.schemaContent.replaceChildren(
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
      this.schemaContent.replaceChildren(
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
      this.schemaContent.replaceChildren(noTablesDiv);
    }
    this.bindSchemaDiagnosticButtons(this.schemaContent);
  }

  /**
   * Builds a single `.schema-table` DOM node for a table and its columns.
   * @param {SchemaTableItem} table - Table metadata entry.
   * @returns {HTMLDivElement} Constructed table element.
   */
  private buildTableElement(table: SchemaTableItem): HTMLDivElement {
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
    titleSpan.className = "schema-table-name";
    titleSpan.setAttribute("data-table", table.name);
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
          { tag: "line", attrs: { x1: "3", y1: "9", x2: "21", y2: "9" } },
          { tag: "line", attrs: { x1: "9", y1: "21", x2: "9", y2: "9" } },
        ],
        "none",
        true,
      ),
    );

    headerDiv.append(toggleBtn, viewBtn);

    toggleBtn.addEventListener("click", () => {
      const isExpanded = tableDiv.classList.toggle("expanded");
      toggleBtn.setAttribute("aria-expanded", isExpanded ? "true" : "false");
    });

    viewBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      void this.ui.tableModalController.open(table.name);
    });

    const columnsList = document.createElement("ul");
    columnsList.className = "schema-columns";
    columnsList.id = `schema-table-${table.name}`;

    table.columns.forEach((col) => {
      const colLi = document.createElement("li");
      colLi.className = "schema-column";
      const colNameSpan = document.createElement("span");
      colNameSpan.className = "col-name";
      colNameSpan.textContent = col.name;
      const colTypeSpan = document.createElement("span");
      colTypeSpan.className = "schema-column-type";
      colTypeSpan.textContent = col.type;
      colLi.append(colNameSpan, colTypeSpan);
      columnsList.appendChild(colLi);
    });

    tableDiv.append(headerDiv, columnsList);
    return tableDiv;
  }

  /**
   * Fetches the database schema and renders it in the sidebar.
   * @param {string} [dbPath] - Optional custom database path.
   * @returns {Promise<void>}
   */
  public async loadSchema(dbPath?: string): Promise<void> {
    try {
      const targetDb = dbPath || this.ui.state.currentDb;
      const url =
        targetDb !== "t1d.duckdb"
          ? `/api/schema?db_path=${encodeURIComponent(targetDb)}`
          : "/api/schema";
      const response = await fetchWithBackendError(url);
      const data = await parseResponseJson<SchemaExplorerApiResponse>(response);

      this.schemaContent.replaceChildren();

      if (data.database) {
        this.applySchemaDbStatus(data.database);
      }

      if (!data.tables || data.tables.length === 0) {
        this.renderEmptyTablesState(data.database);
        return;
      }

      data.tables.forEach((table) => {
        this.schemaContent.appendChild(this.buildTableElement(table));
      });
    } catch (error) {
      console.error("Failed to load schema:", error);
      const errMsg = i18next.t("ui.failedSchema");
      this.ui.announce(errMsg, true);
      const backendUrl = window.location.origin;
      this.schemaContent.replaceChildren(
        createSchemaDiagnosticCard({
          icon: "⚠️",
          title: i18next.t("status.offlineTitle"),
          description: i18next.t("status.offlineDesc", { url: backendUrl }),
          showReload: true,
          isError: true,
        }),
      );
      this.bindSchemaDiagnosticButtons(this.schemaContent);
      this.ui.state.setSystemStatus({
        backendOnline: false,
        status: "offline",
      });
      this.ui.systemStatusController.showBanner(
        "danger",
        i18next.t("status.offlineTitle"),
        i18next.t("status.offlineDesc", { url: backendUrl }),
        "⚠️",
        true,
      );
      this.ui.systemStatusController.updateStatusChip(
        "offline",
        i18next.t("status.systemOffline"),
      );
    }
  }
}
