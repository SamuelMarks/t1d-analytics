/**
 * @file table-modal-controller.ts
 * @module controllers/table-modal-controller
 * Composable sub-controller managing the Table Preview Modal state, pagination,
 * column sorting, search filtering, and CSV/JSON/Excel/Parquet export triggers.
 */

import i18next from "i18next";
import { fetchWithBackendError, parseResponseJson } from "../api-client";
import { exportRowsToCsv, exportRowsToJson } from "../exporters";
import { formatNumber, type SqlCellValue } from "../formatters";
import { requireElementById, trapModalFocus } from "../modal-utils";
import { type ChatState } from "../state";

/**
 * Response payload returned by `/api/table/{table_name}`.
 */
export interface TableDataPayload {
  /** Total number of pages for the current pagination limit. */
  total_pages?: number;
  /** Current page of row records. */
  rows: Array<Record<string, SqlCellValue>>;
}

/**
 * Host contract required by `TablePreviewModalController`.
 */
export interface TablePreviewHost {
  /** Global application state store. */
  state: ChatState;
  /** Announces a message to screen readers. */
  announce(message: string, isAssertive?: boolean): void;
}

/**
 * Composable sub-controller for the Table Preview Modal.
 */
export class TablePreviewModalController {
  /** Host UI reference. */
  private readonly ui: TablePreviewHost;
  /** Modal dialog container. */
  public readonly modal: HTMLElement;
  /** Modal title heading element. */
  public readonly modalTitle: HTMLElement;
  /** Close modal button element. */
  public readonly closeBtn: HTMLButtonElement;
  /** Table search input element. */
  public readonly searchInput: HTMLInputElement;
  /** Sort column select element. */
  public readonly sortColSelect: HTMLSelectElement;
  /** Sort order toggle button element. */
  public readonly sortOrderBtn: HTMLButtonElement;
  /** Previous page button element. */
  public readonly prevBtn: HTMLButtonElement;
  /** Next page button element. */
  public readonly nextBtn: HTMLButtonElement;
  /** Page indicator span element. */
  public readonly pageIndicator: HTMLElement;
  /** Export page CSV button element. */
  public readonly csvBtn: HTMLButtonElement;
  /** Export page JSON button element. */
  public readonly jsonBtn: HTMLButtonElement;
  /** Export full table CSV button element. */
  public readonly fullCsvBtn: HTMLButtonElement;
  /** Export full table Excel button element. */
  public readonly excelBtn: HTMLButtonElement;
  /** Export full table Parquet button element. */
  public readonly parquetBtn: HTMLButtonElement;
  /** Loading spinner element inside the table modal. */
  public readonly loadingEl: HTMLElement;
  /** Table head element inside the table modal. */
  public readonly theadEl: HTMLElement;
  /** Table body element inside the table modal. */
  public readonly tbodyEl: HTMLElement;
  /** Active AbortController for listeners bound during the currently open table preview modal. */
  private listenerController: AbortController = new AbortController();
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
  /** Element that held focus before the modal opened. */
  private previousFocus: Element | null = null;

  /**
   * Creates a new `TablePreviewModalController`.
   * @param {TablePreviewHost} ui - Host UI instance.
   */
  constructor(ui: TablePreviewHost) {
    this.ui = ui;
    this.modal = requireElementById<HTMLElement>("table-modal");
    this.modalTitle = requireElementById<HTMLElement>("modal-title");
    this.closeBtn = requireElementById<HTMLButtonElement>("close-modal-btn");
    this.searchInput =
      requireElementById<HTMLInputElement>("modal-table-search");
    this.sortColSelect = requireElementById<HTMLSelectElement>(
      "modal-table-sort-col",
    );
    this.sortOrderBtn = requireElementById<HTMLButtonElement>(
      "modal-table-sort-order",
    );
    this.prevBtn = requireElementById<HTMLButtonElement>("prev-page-btn");
    this.nextBtn = requireElementById<HTMLButtonElement>("next-page-btn");
    this.pageIndicator = requireElementById<HTMLElement>("page-indicator");
    this.csvBtn = requireElementById<HTMLButtonElement>("modal-export-csv-btn");
    this.jsonBtn = requireElementById<HTMLButtonElement>(
      "modal-export-json-btn",
    );
    this.fullCsvBtn = requireElementById<HTMLButtonElement>(
      "modal-export-full-csv-btn",
    );
    this.excelBtn = requireElementById<HTMLButtonElement>(
      "modal-export-excel-btn",
    );
    this.parquetBtn = requireElementById<HTMLButtonElement>(
      "modal-export-parquet-btn",
    );
    this.loadingEl = requireElementById<HTMLElement>("table-loading");
    this.theadEl = requireElementById<HTMLElement>("modal-table-head");
    this.tbodyEl = requireElementById<HTMLElement>("modal-table-body");
  }

  /**
   * Aborts any active modal event listeners.
   * @returns {void}
   */
  public abortListeners(): void {
    this.listenerController.abort();
  }

  /**
   * Refreshes the localized title and data when the modal is currently open.
   * @returns {Promise<void>}
   */
  public async refreshIfOpen(): Promise<void> {
    if (
      this.modal.getAttribute("aria-hidden") !== "true" &&
      this.currentTable
    ) {
      this.modalTitle.textContent = i18next.t("ui.tableName", {
        name: this.currentTable,
      });
      await this.fetchTableData();
    }
  }

  /**
   * Closes the table preview modal and restores focus.
   * @returns {void}
   */
  public close(): void {
    this.modal.setAttribute("aria-hidden", "true");
    this.modal.removeAttribute("aria-modal");
    this.listenerController.abort();
    if (
      this.previousFocus instanceof HTMLElement &&
      document.body.contains(this.previousFocus)
    ) {
      this.previousFocus.focus();
    } else {
      document.getElementById("chat-input")?.focus();
    }
  }

  /**
   * Opens a backend full-table export URL.
   * @param {"csv" | "excel" | "parquet"} format - Export format endpoint.
   * @returns {void}
   */
  private triggerBackendExport(format: "csv" | "excel" | "parquet"): void {
    const dbParam =
      this.ui.state.currentDb !== "t1d.duckdb"
        ? `&db_path=${encodeURIComponent(this.ui.state.currentDb)}`
        : "";
    window.open(
      `/api/export/${format}?table_name=${encodeURIComponent(this.currentTable)}${dbParam}`,
    );
  }

  /**
   * Opens the Table Preview modal for the specified table name.
   * @param {string} tableName - Table identifier to preview.
   * @returns {Promise<void>}
   */
  public async open(tableName: string): Promise<void> {
    this.previousFocus = document.activeElement;
    this.currentTable = tableName;
    this.currentPage = 1;
    this.currentSortBy = "";
    this.currentSortOrder = "asc";
    this.currentSearchTerm = "";

    this.searchInput.value = "";
    const defaultOpt = document.createElement("option");
    defaultOpt.value = "";
    defaultOpt.setAttribute("data-i18n", "table.sortDefault");
    defaultOpt.textContent = i18next.t("table.sortDefault", "(Default)");
    this.sortColSelect.replaceChildren(defaultOpt);
    this.sortOrderBtn.setAttribute("data-i18n", "table.sortAsc");
    this.sortOrderBtn.textContent = i18next.t("table.sortAsc", "ASC");

    this.modalTitle.textContent = i18next.t("ui.tableName", {
      name: tableName,
    });
    this.modal.removeAttribute("aria-hidden");
    this.modal.setAttribute("aria-modal", "true");

    this.csvBtn.onclick = () =>
      exportRowsToCsv(
        this.currentModalRows,
        `${this.currentTable}_page${this.currentPage}.csv`,
      );
    this.jsonBtn.onclick = () =>
      exportRowsToJson(
        this.currentModalRows,
        `${this.currentTable}_page${this.currentPage}.json`,
      );
    this.fullCsvBtn.onclick = () => this.triggerBackendExport("csv");
    this.excelBtn.onclick = () => this.triggerBackendExport("excel");
    this.parquetBtn.onclick = () => this.triggerBackendExport("parquet");

    this.closeBtn.focus();

    this.listenerController.abort();
    this.listenerController = new AbortController();
    const { signal } = this.listenerController;

    this.modal.addEventListener(
      "keydown",
      (e: KeyboardEvent) => trapModalFocus(e, this.modal, () => this.close()),
      { signal },
    );
    this.closeBtn.addEventListener("click", () => this.close(), { signal });
    this.modal.addEventListener(
      "click",
      (e: MouseEvent) => {
        if (e.target === this.modal) this.close();
      },
      { signal },
    );

    this.searchInput.addEventListener(
      "input",
      () => {
        this.currentSearchTerm = this.searchInput.value.trim();
        this.currentPage = 1;
        void this.fetchTableData();
      },
      { signal },
    );

    this.sortColSelect.addEventListener(
      "change",
      () => {
        this.currentSortBy = this.sortColSelect.value;
        this.currentPage = 1;
        void this.fetchTableData();
      },
      { signal },
    );

    this.sortOrderBtn.addEventListener(
      "click",
      () => {
        this.currentSortOrder =
          this.currentSortOrder === "asc" ? "desc" : "asc";
        const sortKey =
          this.currentSortOrder === "asc" ? "table.sortAsc" : "table.sortDesc";
        const sortLabel = i18next.t(
          sortKey,
          this.currentSortOrder.toUpperCase(),
        );
        this.sortOrderBtn.setAttribute("data-i18n", sortKey);
        this.sortOrderBtn.textContent = sortLabel;
        this.sortOrderBtn.setAttribute(
          "aria-label",
          `${i18next.t("table.toggleSortOrder", "Toggle sort order")} (${sortLabel})`,
        );
        this.ui.announce(sortLabel);
        this.currentPage = 1;
        void this.fetchTableData();
      },
      { signal },
    );

    this.prevBtn.addEventListener(
      "click",
      () => {
        if (this.currentPage > 1) {
          this.currentPage--;
          void this.fetchTableData();
        }
      },
      { signal },
    );
    this.nextBtn.addEventListener(
      "click",
      () => {
        this.currentPage++;
        void this.fetchTableData();
      },
      { signal },
    );

    await this.fetchTableData();
  }

  /**
   * Updates the disabled state of all 5 export buttons in the modal footer.
   * @param {boolean} disabled - Whether the buttons should be disabled.
   * @returns {void}
   */
  private setExportButtonsDisabled(disabled: boolean): void {
    this.csvBtn.disabled = disabled;
    this.jsonBtn.disabled = disabled;
    this.fullCsvBtn.disabled = disabled;
    this.excelBtn.disabled = disabled;
    this.parquetBtn.disabled = disabled;
  }

  /**
   * Renders non-empty rows and sortable column headers into the table modal.
   * @param {TableDataPayload} data - Loaded table data payload.
   * @returns {void}
   */
  private renderPopulatedRows(data: TableDataPayload): void {
    const columns = Object.keys(data.rows[0]);

    if (this.sortColSelect.options.length <= 1) {
      columns.forEach((col) => {
        const opt = document.createElement("option");
        opt.value = col;
        opt.textContent = col;
        this.sortColSelect.appendChild(opt);
      });
    }
    this.sortColSelect.value = this.currentSortBy;

    const sortKey =
      this.currentSortOrder === "asc" ? "table.sortAsc" : "table.sortDesc";
    this.sortOrderBtn.setAttribute("data-i18n", sortKey);
    this.sortOrderBtn.textContent = i18next.t(
      sortKey,
      this.currentSortOrder.toUpperCase(),
    );

    const trHead = document.createElement("tr");
    columns.forEach((col) => {
      const th = document.createElement("th");
      th.setAttribute("scope", "col");
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

      const sortBtn = document.createElement("button");
      sortBtn.type = "button";
      sortBtn.className = "table-sort-header-btn";
      sortBtn.textContent = col;

      const onThSort = (): void => {
        if (this.currentSortBy === col) {
          this.currentSortOrder =
            this.currentSortOrder === "asc" ? "desc" : "asc";
        } else {
          this.currentSortBy = col;
          this.currentSortOrder = "asc";
        }
        this.currentPage = 1;
        void this.fetchTableData();
      };

      th.addEventListener("click", onThSort);
      th.addEventListener("keydown", (e: KeyboardEvent) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          onThSort();
        }
      });

      th.appendChild(sortBtn);
      trHead.appendChild(th);
    });
    this.theadEl.appendChild(trHead);

    data.rows.forEach((row) => {
      const tr = document.createElement("tr");
      columns.forEach((col) => {
        const td = document.createElement("td");
        td.textContent =
          row[col] === null || row[col] === undefined
            ? "NULL"
            : String(row[col]);
        tr.appendChild(td);
      });
      this.tbodyEl.appendChild(tr);
    });

    this.prevBtn.disabled = this.currentPage === 1;
    const hasMore = data.total_pages
      ? this.currentPage < data.total_pages
      : data.rows.length >= this.ROWS_PER_PAGE;
    this.nextBtn.disabled = !hasMore;
    this.setExportButtonsDisabled(false);
    this.ui.announce(
      i18next.t("aria.tableDataLoaded", "Table Data loaded"),
      false,
    );
  }

  /**
   * Fetches the current page of data for the opened table and updates the modal UI.
   * @returns {Promise<void>}
   */
  public async fetchTableData(): Promise<void> {
    this.loadingEl.style.display = "block";
    this.prevBtn.disabled = true;
    this.nextBtn.disabled = true;
    this.setExportButtonsDisabled(true);

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
      const data = await parseResponseJson<TableDataPayload>(response);

      this.loadingEl.style.display = "none";
      const formattedPage = formatNumber(this.currentPage);
      if (data.total_pages) {
        const formattedTotal = formatNumber(data.total_pages);
        this.pageIndicator.textContent = i18next.t("app.pageOfTotal", {
          page: formattedPage,
          total: formattedTotal,
          defaultValue: `Page ${formattedPage} / ${formattedTotal}`,
        });
      } else {
        this.pageIndicator.textContent = i18next.t("app.page", {
          page: formattedPage,
        });
      }

      if (data.rows.length === 0 && this.currentPage > 1) {
        this.currentPage--;
        await this.fetchTableData();
        return;
      }

      this.currentModalRows = data.rows;
      this.theadEl.replaceChildren();
      this.tbodyEl.replaceChildren();

      if (data.rows.length > 0) {
        this.renderPopulatedRows(data);
      } else {
        const msg = i18next.t("ui.noDataAvailable");
        this.ui.announce(msg, false);
        const tr = document.createElement("tr");
        const td = document.createElement("td");
        td.textContent = msg;
        tr.appendChild(td);
        this.tbodyEl.replaceChildren(tr);
        this.prevBtn.disabled = true;
        this.nextBtn.disabled = true;
      }
    } catch (error) {
      console.error("Failed to fetch table data:", error);
      this.loadingEl.style.display = "none";
      const errMsg = i18next.t("ui.failedData", {
        error,
        interpolation: { escapeValue: false },
      });
      this.ui.announce(errMsg, true);
      const tr = document.createElement("tr");
      const td = document.createElement("td");
      td.className = "error-text";
      td.setAttribute("role", "alert");
      td.setAttribute("aria-live", "assertive");
      td.textContent = errMsg;
      tr.appendChild(td);
      this.tbodyEl.replaceChildren(tr);
    }
  }
}
