/**
 * @file table-modal-controller.ts
 * @module controllers/table-modal-controller
 * Composable sub-controller managing the Table Preview Modal state, pagination,
 * column sorting, debounced search filtering, and CSV/JSON/Excel/Parquet exports via `ModalDialogController`.
 */

import i18next from "i18next";
import { fetchWithBackendError, parseResponseJson } from "../api-client";
import { type TableRowsResponsePayload } from "../api-types";
import { type AppEventBus } from "../event-bus";
import {
  exportRowsToCsv,
  exportRowsToJson,
  triggerBlobDownload,
} from "../exporters";
import { formatNumber, type SqlCellValue } from "../formatters";
import { ModalDialogController, requireElementById } from "../modal-utils";
import { DEFAULT_DB_NAME, type ChatState } from "../state";

export { type TableRowsResponsePayload as TableDataPayload };

/**
 * Narrow injected dependencies for `TablePreviewModalController`.
 */
export interface TablePreviewModalDeps {
  /** Shared application state store. */
  state: ChatState;
  /** Unidirectional domain event bus. */
  events: AppEventBus;
}

/**
 * Composable sub-controller for the Table Preview Modal (`#table-modal`).
 */
export class TablePreviewModalController {
  /** Shared application state store. */
  private readonly state: ChatState;
  /** Domain event bus. */
  private readonly events: AppEventBus;
  /** Composable modal dialog lifecycle controller. */
  private readonly dialog: ModalDialogController;
  /** Persistent AbortController for domain event bus subscriptions. */
  private readonly busAbortController = new AbortController();
  /** Modal overlay container for previewing database table rows. */
  public readonly modal: HTMLElement;
  /** Heading element inside the table preview modal. */
  public readonly modalTitle: HTMLElement;
  /** Button to close the table preview modal. */
  public readonly closeBtn: HTMLButtonElement;
  /** Search input element for filtering rows in the table preview modal. */
  public readonly searchInput: HTMLInputElement;
  /** Column select dropdown for sorting rows in the table preview modal. */
  public readonly sortColSelect: HTMLSelectElement;
  /** Button to toggle ascending/descending sort order in the table preview modal. */
  public readonly sortOrderBtn: HTMLButtonElement;
  /** Previous page button in the table preview modal pagination controls. */
  public readonly prevBtn: HTMLButtonElement;
  /** Next page button in the table preview modal pagination controls. */
  public readonly nextBtn: HTMLButtonElement;
  /** Span displaying the current page number in the table preview modal. */
  public readonly pageIndicator: HTMLElement;
  /** Button to export the current table modal page as CSV. */
  public readonly csvBtn: HTMLButtonElement;
  /** Button to export the current table modal page as JSON. */
  public readonly jsonBtn: HTMLButtonElement;
  /** Button to export the entire table as CSV from the backend. */
  public readonly fullCsvBtn: HTMLButtonElement;
  /** Button to export the entire table as Excel (.xlsx) from the backend. */
  public readonly excelBtn: HTMLButtonElement;
  /** Button to export the entire table as Parquet from the backend. */
  public readonly parquetBtn: HTMLButtonElement;
  /** Loading spinner element inside the table preview modal. */
  public readonly loadingEl: HTMLElement;
  /** `<thead>` element inside the table preview modal. */
  public readonly theadEl: HTMLElement;
  /** `<tbody>` element inside the table preview modal. */
  public readonly tbodyEl: HTMLElement;

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
  /** Monotonic sequence counter used to discard stale out-of-order fetchTableData responses. */
  private requestSequence: number = 0;
  /** Debounce timer handle for search input keystrokes. */
  private searchDebounceTimer: ReturnType<typeof setTimeout> | null = null;
  /** Active AbortController for cancelling superseded in-flight `/api/table/*` requests. */
  private fetchAbortController: AbortController | null = null;
  /** Search input debounce interval in milliseconds. */
  public static readonly SEARCH_DEBOUNCE_MS = 250;

  /**
   * Creates a new `TablePreviewModalController`.
   * @param {TablePreviewModalDeps} deps - Narrow state and event bus dependencies.
   */
  constructor(deps: TablePreviewModalDeps) {
    this.state = deps.state;
    this.events = deps.events;
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

    this.dialog = new ModalDialogController({
      modalEl: this.modal,
      closeBtn: this.closeBtn,
      initialFocusEl: this.closeBtn,
      onClose: () => {
        this.cancelPendingOperations();
      },
    });

    this.events.on(
      "openTablePreview",
      ({ tableName }) => {
        void this.open(tableName);
      },
      this.busAbortController.signal,
    );
  }

  /**
   * Emits a screen-reader announcement via the domain event bus.
   * @param {string} message - Message string to announce.
   * @param {boolean} [isAssertive=false] - Whether to use assertive politeness.
   * @returns {void}
   */
  private announce(message: string, isAssertive: boolean = false): void {
    this.events.emit("announce", { message, isAssertive });
  }

  /**
   * Cancels any pending search debounce timer and in-flight fetch request.
   * @returns {void}
   */
  private cancelPendingOperations(): void {
    if (this.searchDebounceTimer !== null) {
      clearTimeout(this.searchDebounceTimer);
      this.searchDebounceTimer = null;
    }
    this.fetchAbortController?.abort();
    this.fetchAbortController = null;
  }

  /**
   * Disposes all modal and bus listeners when tearing down the UI.
   * @returns {void}
   */
  public dispose(): void {
    this.cancelPendingOperations();
    this.dialog.dispose();
    this.busAbortController.abort();
  }

  /**
   * Refreshes the localized title and data when the modal is currently open.
   * @returns {Promise<void>}
   */
  public async refreshIfOpen(): Promise<void> {
    if (this.dialog.isOpen() && this.currentTable) {
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
    this.dialog.close();
  }

  /**
   * Downloads a full-table server export via authenticated `fetchWithBackendError` + `Blob` download.
   * @param {"csv" | "excel" | "parquet"} format - Export format endpoint.
   * @returns {Promise<void>}
   */
  public async triggerServerExport(
    format: "csv" | "excel" | "parquet",
  ): Promise<void> {
    const dbParam =
      this.state.currentDb !== DEFAULT_DB_NAME
        ? `&db_path=${encodeURIComponent(this.state.currentDb)}`
        : "";
    const url = `/api/export/${format}?table_name=${encodeURIComponent(this.currentTable)}${dbParam}`;
    const ext = format === "excel" ? "xlsx" : format;
    try {
      const response = await fetchWithBackendError(url);
      const blob = await response.blob();
      triggerBlobDownload(blob, `${this.currentTable}.${ext}`);
    } catch (error) {
      const errMsg = i18next.t("ui.failedData", {
        error,
        interpolation: { escapeValue: false },
      });
      this.announce(errMsg, true);
    }
  }

  /**
   * Opens the Table Preview modal for the specified table name.
   * @param {string} tableName - Table identifier to preview.
   * @returns {Promise<void>}
   */
  public async open(tableName: string): Promise<void> {
    this.dialog.captureOpenerFocus();
    this.cancelPendingOperations();
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
    this.fullCsvBtn.onclick = () => {
      void this.triggerServerExport("csv");
    };
    this.excelBtn.onclick = () => {
      void this.triggerServerExport("excel");
    };
    this.parquetBtn.onclick = () => {
      void this.triggerServerExport("parquet");
    };

    const signal = this.dialog.open(this.closeBtn);

    this.searchInput.addEventListener(
      "input",
      () => {
        if (this.searchDebounceTimer !== null) {
          clearTimeout(this.searchDebounceTimer);
        }
        this.fetchAbortController?.abort();
        this.searchDebounceTimer = setTimeout(() => {
          this.searchDebounceTimer = null;
          this.currentSearchTerm = this.searchInput.value.trim();
          this.currentPage = 1;
          void this.fetchTableData();
        }, TablePreviewModalController.SEARCH_DEBOUNCE_MS);
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
        this.announce(sortLabel);
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
   * @param {TableRowsResponsePayload} data - Loaded table data payload.
   * @returns {void}
   */
  private renderPopulatedRows(data: TableRowsResponsePayload): void {
    const columns = Object.keys(Object(data.rows[0]));

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

      /**
       * Toggles or selects column sort direction and reloads page 1.
       * @returns {void}
       */
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
    const hasMore =
      data.total_pages !== undefined && data.total_pages > 0
        ? this.currentPage < data.total_pages
        : data.rows.length >= this.ROWS_PER_PAGE;
    this.nextBtn.disabled = !hasMore;
    this.setExportButtonsDisabled(false);
    this.announce(
      i18next.t("aria.tableDataLoaded", "Table Data loaded"),
      false,
    );
  }

  /**
   * Fetches the current page of data for the opened table and updates the modal UI.
   * @returns {Promise<void>}
   */
  public async fetchTableData(): Promise<void> {
    const requestId = ++this.requestSequence;
    this.fetchAbortController?.abort();
    const reqAbort = new AbortController();
    this.fetchAbortController = reqAbort;
    this.loadingEl.style.display = "block";
    this.prevBtn.disabled = true;
    this.nextBtn.disabled = true;
    this.setExportButtonsDisabled(true);

    try {
      const limit = this.ROWS_PER_PAGE;
      const offset = (this.currentPage - 1) * limit;
      let url = `/api/table/${encodeURIComponent(this.currentTable)}?limit=${limit}&offset=${offset}`;
      if (this.state.currentDb !== DEFAULT_DB_NAME) {
        url += `&db_path=${encodeURIComponent(this.state.currentDb)}`;
      }
      if (this.currentSortBy) {
        url += `&sort_by=${encodeURIComponent(this.currentSortBy)}&order=${this.currentSortOrder}`;
      }
      if (this.currentSearchTerm) {
        url += `&search=${encodeURIComponent(this.currentSearchTerm)}`;
      }

      const response = await fetchWithBackendError(url, {
        signal: reqAbort.signal,
      });
      const data = await parseResponseJson<TableRowsResponsePayload>(response);

      if (requestId !== this.requestSequence) {
        return;
      }

      this.loadingEl.style.display = "none";
      const formattedPage = formatNumber(this.currentPage);
      if (data.total_pages !== undefined && data.total_pages > 0) {
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
        this.announce(msg, false);
        const tr = document.createElement("tr");
        const td = document.createElement("td");
        td.textContent = msg;
        tr.appendChild(td);
        this.tbodyEl.replaceChildren(tr);
        this.prevBtn.disabled = true;
        this.nextBtn.disabled = true;
      }
    } catch (error) {
      if (requestId !== this.requestSequence) {
        return;
      }
      console.error("Failed to fetch table data:", error);
      this.loadingEl.style.display = "none";
      const errMsg = i18next.t("ui.failedData", {
        error,
        interpolation: { escapeValue: false },
      });
      this.announce(errMsg, true);
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
