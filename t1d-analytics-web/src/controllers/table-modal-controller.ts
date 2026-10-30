/**
 * @file table-modal-controller.ts
 * @module controllers/table-modal-controller
 * Composable sub-controller managing the Table Preview Modal state, pagination,
 * column sorting, search filtering, and CSV/JSON/Excel/Parquet exports.
 */

import i18next from "i18next";
import { fetchWithBackendError, parseResponseJson } from "../api-client";
import { type ExportableCellValue } from "../exporters";
import { formatNumber, type SqlCellValue } from "../formatters";
import { trapModalFocus } from "../modal-utils";
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
  /** Name of the database table currently displayed in the table preview modal. */
  currentTable: string;
  /** Current 1-based page number in the table preview modal. */
  currentPage: number;
  /** Number of rows fetched per page in the table preview modal. */
  readonly ROWS_PER_PAGE: number;
  /** Array of row objects currently rendered in the table preview modal. */
  currentModalRows: Array<Record<string, SqlCellValue>>;
  /** Column name currently used to sort the table preview modal, or empty string for default. */
  currentSortBy: string;
  /** Active sort direction ('asc' or 'desc') in the table preview modal. */
  currentSortOrder: "asc" | "desc";
  /** Active search filter term in the table preview modal. */
  currentSearchTerm: string;
  /** Exports rows to CSV. */
  exportRowsToCsv(
    rows: Array<Record<string, ExportableCellValue>>,
    filename: string,
  ): void;
  /** Exports rows to JSON. */
  exportRowsToJson(
    rows: Array<Record<string, ExportableCellValue>>,
    filename: string,
  ): void;
  /** Announces a message to screen readers. */
  announce(message: string, isAssertive?: boolean): void;
}

/**
 * Looks up a typed element by ID using `document.querySelector<T>("#" + id)`.
 * @template T - Target HTML element subtype.
 * @param {string} id - The DOM element ID.
 * @returns {T | null} The matched element or null.
 */
function queryModalEl<T extends HTMLElement>(id: string): T | null {
  return document.querySelector<T>(`#${id}`);
}

/**
 * Composable sub-controller for the Table Preview Modal.
 */
export class TablePreviewModalController {
  /** Host UI reference. */
  private readonly ui: TablePreviewHost;
  /** Active AbortController for listeners bound during the currently open table preview modal. */
  private listenerController: AbortController = new AbortController();

  /**
   * Creates a new `TablePreviewModalController`.
   * @param {TablePreviewHost} ui - Host UI instance.
   */
  constructor(ui: TablePreviewHost) {
    this.ui = ui;
  }

  /**
   * Aborts any active modal event listeners.
   * @returns {void}
   */
  public abortListeners(): void {
    this.listenerController.abort();
  }

  /**
   * Opens the Table Preview modal for the specified table name.
   * @param {string} tableName - Table identifier to preview.
   * @returns {Promise<void>}
   */
  public async open(tableName: string): Promise<void> {
    const previousFocus = document.activeElement;
    this.ui.currentTable = tableName;
    this.ui.currentPage = 1;
    this.ui.currentSortBy = "";
    this.ui.currentSortOrder = "asc";
    this.ui.currentSearchTerm = "";

    const modal = document.getElementById("table-modal");
    const modalTitle = document.getElementById("modal-title");
    const closeBtn = document.getElementById("close-modal-btn");
    const prevBtn = queryModalEl<HTMLButtonElement>("prev-page-btn");
    const nextBtn = queryModalEl<HTMLButtonElement>("next-page-btn");

    const searchInput = queryModalEl<HTMLInputElement>("modal-table-search");
    const sortSelect = queryModalEl<HTMLSelectElement>("modal-table-sort-col");
    const sortOrderBtn = queryModalEl<HTMLButtonElement>(
      "modal-table-sort-order",
    );

    if (!modal || !modalTitle || !closeBtn || !prevBtn || !nextBtn) return;

    if (searchInput) {
      searchInput.value = "";
    }
    if (sortSelect) {
      const defaultOpt = document.createElement("option");
      defaultOpt.value = "";
      defaultOpt.setAttribute("data-i18n", "table.sortDefault");
      defaultOpt.textContent = i18next.t("table.sortDefault", "(Default)");
      sortSelect.replaceChildren(defaultOpt);
    }
    if (sortOrderBtn) {
      sortOrderBtn.setAttribute("data-i18n", "table.sortAsc");
      sortOrderBtn.textContent = i18next.t("table.sortAsc", "ASC");
    }

    modalTitle.textContent = i18next.t("ui.tableName", { name: tableName });
    modal.removeAttribute("aria-hidden");
    modal.setAttribute("aria-modal", "true");

    const modalExportCsvBtn = queryModalEl<HTMLButtonElement>(
      "modal-export-csv-btn",
    );
    const modalExportJsonBtn = queryModalEl<HTMLButtonElement>(
      "modal-export-json-btn",
    );
    const modalExportFullCsvBtn = queryModalEl<HTMLButtonElement>(
      "modal-export-full-csv-btn",
    );
    const modalExportExcelBtn = queryModalEl<HTMLButtonElement>(
      "modal-export-excel-btn",
    );
    const modalExportParquetBtn = queryModalEl<HTMLButtonElement>(
      "modal-export-parquet-btn",
    );

    if (modalExportCsvBtn) {
      modalExportCsvBtn.onclick = () =>
        this.ui.exportRowsToCsv(
          this.ui.currentModalRows,
          `${this.ui.currentTable}_page${this.ui.currentPage}.csv`,
        );
    }
    if (modalExportJsonBtn) {
      modalExportJsonBtn.onclick = () =>
        this.ui.exportRowsToJson(
          this.ui.currentModalRows,
          `${this.ui.currentTable}_page${this.ui.currentPage}.json`,
        );
    }
    if (modalExportFullCsvBtn) {
      modalExportFullCsvBtn.onclick = () => {
        const dbParam =
          this.ui.state.currentDb && this.ui.state.currentDb !== "t1d.duckdb"
            ? `&db_path=${encodeURIComponent(this.ui.state.currentDb)}`
            : "";
        window.open(
          `/api/export/csv?table_name=${encodeURIComponent(this.ui.currentTable)}${dbParam}`,
        );
      };
    }
    if (modalExportExcelBtn) {
      modalExportExcelBtn.onclick = () => {
        const dbParam =
          this.ui.state.currentDb && this.ui.state.currentDb !== "t1d.duckdb"
            ? `&db_path=${encodeURIComponent(this.ui.state.currentDb)}`
            : "";
        window.open(
          `/api/export/excel?table_name=${encodeURIComponent(this.ui.currentTable)}${dbParam}`,
        );
      };
    }
    if (modalExportParquetBtn) {
      modalExportParquetBtn.onclick = () => {
        const dbParam =
          this.ui.state.currentDb && this.ui.state.currentDb !== "t1d.duckdb"
            ? `&db_path=${encodeURIComponent(this.ui.state.currentDb)}`
            : "";
        window.open(
          `/api/export/parquet?table_name=${encodeURIComponent(this.ui.currentTable)}${dbParam}`,
        );
      };
    }

    closeBtn.focus();

    this.listenerController.abort();
    this.listenerController = new AbortController();
    const modalSignal = this.listenerController.signal;

    const closeModal = (): void => {
      modal.setAttribute("aria-hidden", "true");
      modal.removeAttribute("aria-modal");
      this.listenerController.abort();

      if (
        previousFocus instanceof HTMLElement &&
        document.body.contains(previousFocus)
      ) {
        previousFocus.focus();
      } else {
        document.getElementById("chat-input")?.focus();
      }
    };

    const onModalKeyDown = (e: KeyboardEvent): void => {
      trapModalFocus(e, modal, closeModal);
    };

    modal.addEventListener("keydown", onModalKeyDown, { signal: modalSignal });

    const overlayClick = (e: MouseEvent): void => {
      if (e.target === modal) closeModal();
    };

    closeBtn.addEventListener("click", closeModal, { signal: modalSignal });
    modal.addEventListener("click", overlayClick, { signal: modalSignal });

    if (searchInput) {
      searchInput.addEventListener(
        "input",
        () => {
          this.ui.currentSearchTerm = searchInput.value.trim();
          this.ui.currentPage = 1;
          void this.fetchTableData();
        },
        { signal: modalSignal },
      );
    }

    if (sortSelect) {
      sortSelect.addEventListener(
        "change",
        () => {
          this.ui.currentSortBy = sortSelect.value;
          this.ui.currentPage = 1;
          void this.fetchTableData();
        },
        { signal: modalSignal },
      );
    }

    if (sortOrderBtn) {
      sortOrderBtn.addEventListener(
        "click",
        () => {
          this.ui.currentSortOrder =
            this.ui.currentSortOrder === "asc" ? "desc" : "asc";
          const sortKey =
            this.ui.currentSortOrder === "asc"
              ? "table.sortAsc"
              : "table.sortDesc";
          const sortLabel = i18next.t(
            sortKey,
            this.ui.currentSortOrder.toUpperCase(),
          );
          sortOrderBtn.setAttribute("data-i18n", sortKey);
          sortOrderBtn.textContent = sortLabel;
          sortOrderBtn.setAttribute(
            "aria-label",
            `${i18next.t("table.toggleSortOrder", "Toggle sort order")} (${sortLabel})`,
          );
          this.ui.announce(sortLabel);
          this.ui.currentPage = 1;
          void this.fetchTableData();
        },
        { signal: modalSignal },
      );
    }

    const prevFn = (): void => {
      if (this.ui.currentPage > 1) {
        this.ui.currentPage--;
        void this.fetchTableData();
      }
    };
    const nextFn = (): void => {
      this.ui.currentPage++;
      void this.fetchTableData();
    };

    prevBtn.addEventListener("click", prevFn, { signal: modalSignal });
    nextBtn.addEventListener("click", nextFn, { signal: modalSignal });

    await this.fetchTableData();
  }

  /**
   * Fetches the current page of data for the opened table and updates the modal UI.
   * @returns {Promise<void>}
   */
  public async fetchTableData(): Promise<void> {
    const loading = document.getElementById("table-loading");
    const thead = document.getElementById("modal-table-head");
    const tbody = document.getElementById("modal-table-body");
    const prevBtn = queryModalEl<HTMLButtonElement>("prev-page-btn");
    const nextBtn = queryModalEl<HTMLButtonElement>("next-page-btn");
    const indicator = document.getElementById("page-indicator");
    const sortSelect = queryModalEl<HTMLSelectElement>("modal-table-sort-col");
    const sortOrderBtn = queryModalEl<HTMLButtonElement>(
      "modal-table-sort-order",
    );

    if (!loading || !thead || !tbody || !prevBtn || !nextBtn || !indicator)
      return;

    loading.style.display = "block";
    prevBtn.disabled = true;
    nextBtn.disabled = true;
    const modalExportCsvBtn = queryModalEl<HTMLButtonElement>(
      "modal-export-csv-btn",
    );
    const modalExportJsonBtn = queryModalEl<HTMLButtonElement>(
      "modal-export-json-btn",
    );
    const modalExportFullCsvBtn = queryModalEl<HTMLButtonElement>(
      "modal-export-full-csv-btn",
    );
    const modalExportExcelBtn = queryModalEl<HTMLButtonElement>(
      "modal-export-excel-btn",
    );
    const modalExportParquetBtn = queryModalEl<HTMLButtonElement>(
      "modal-export-parquet-btn",
    );

    if (modalExportCsvBtn) modalExportCsvBtn.disabled = true;
    if (modalExportJsonBtn) modalExportJsonBtn.disabled = true;
    if (modalExportFullCsvBtn) modalExportFullCsvBtn.disabled = true;
    if (modalExportExcelBtn) modalExportExcelBtn.disabled = true;
    if (modalExportParquetBtn) modalExportParquetBtn.disabled = true;

    try {
      const limit = this.ui.ROWS_PER_PAGE;
      const offset = (this.ui.currentPage - 1) * limit;
      let url = `/api/table/${encodeURIComponent(this.ui.currentTable)}?limit=${limit}&offset=${offset}`;
      if (this.ui.currentSortBy) {
        url += `&sort_by=${encodeURIComponent(this.ui.currentSortBy)}&order=${this.ui.currentSortOrder}`;
      }
      if (this.ui.currentSearchTerm) {
        url += `&search=${encodeURIComponent(this.ui.currentSearchTerm)}`;
      }

      const response = await fetchWithBackendError(url);
      const data = await parseResponseJson<TableDataPayload>(response);

      loading.style.display = "none";
      const formattedPage = formatNumber(this.ui.currentPage);
      if (data.total_pages) {
        const formattedTotal = formatNumber(data.total_pages);
        indicator.textContent = i18next.t("app.pageOfTotal", {
          page: formattedPage,
          total: formattedTotal,
          defaultValue: `Page ${formattedPage} / ${formattedTotal}`,
        });
      } else {
        indicator.textContent = i18next.t("app.page", {
          page: formattedPage,
        });
      }

      if (data.rows.length === 0 && this.ui.currentPage > 1) {
        this.ui.currentPage--;
        await this.fetchTableData();
        return;
      }

      this.ui.currentModalRows = data.rows;
      thead.replaceChildren();
      tbody.replaceChildren();

      if (data.rows.length > 0) {
        const columns = Object.keys(data.rows[0]);

        if (sortSelect && sortSelect.options.length <= 1) {
          columns.forEach((col) => {
            const opt = document.createElement("option");
            opt.value = col;
            opt.textContent = col;
            sortSelect.appendChild(opt);
          });
        }
        if (sortSelect) {
          sortSelect.value = this.ui.currentSortBy;
        }
        if (sortOrderBtn) {
          const sortKey =
            this.ui.currentSortOrder === "asc"
              ? "table.sortAsc"
              : "table.sortDesc";
          sortOrderBtn.setAttribute("data-i18n", sortKey);
          sortOrderBtn.textContent = i18next.t(
            sortKey,
            this.ui.currentSortOrder.toUpperCase(),
          );
        }

        const trHead = document.createElement("tr");
        columns.forEach((col) => {
          const th = document.createElement("th");
          th.setAttribute("scope", "col");
          th.setAttribute(
            "aria-sort",
            this.ui.currentSortBy === col
              ? this.ui.currentSortOrder === "asc"
                ? "ascending"
                : "descending"
              : "none",
          );

          if (this.ui.currentSortBy === col) {
            th.classList.add(
              this.ui.currentSortOrder === "asc" ? "sorted-asc" : "sorted-desc",
            );
          }

          const sortBtn = document.createElement("button");
          sortBtn.type = "button";
          sortBtn.className = "table-sort-header-btn";
          sortBtn.textContent = col;

          const onThSort = (): void => {
            if (this.ui.currentSortBy === col) {
              this.ui.currentSortOrder =
                this.ui.currentSortOrder === "asc" ? "desc" : "asc";
            } else {
              this.ui.currentSortBy = col;
              this.ui.currentSortOrder = "asc";
            }
            this.ui.currentPage = 1;
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
        thead.appendChild(trHead);

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

        prevBtn.disabled = this.ui.currentPage === 1;
        const hasMore = data.total_pages
          ? this.ui.currentPage < data.total_pages
          : data.rows.length >= limit;
        nextBtn.disabled = !hasMore;
        if (modalExportCsvBtn) modalExportCsvBtn.disabled = false;
        if (modalExportJsonBtn) modalExportJsonBtn.disabled = false;
        if (modalExportFullCsvBtn) modalExportFullCsvBtn.disabled = false;
        if (modalExportExcelBtn) modalExportExcelBtn.disabled = false;
        if (modalExportParquetBtn) modalExportParquetBtn.disabled = false;
        this.ui.announce(
          i18next.t("aria.tableDataLoaded", "Table Data loaded"),
          false,
        );
      } else {
        const msg = i18next.t("ui.noDataAvailable");
        this.ui.announce(msg, false);
        const tr = document.createElement("tr");
        const td = document.createElement("td");
        td.textContent = msg;
        tr.appendChild(td);
        tbody.replaceChildren(tr);
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
      this.ui.announce(errMsg, true);
      const tr = document.createElement("tr");
      const td = document.createElement("td");
      td.className = "error-text";
      td.setAttribute("role", "alert");
      td.setAttribute("aria-live", "assertive");
      td.textContent = errMsg;
      tr.appendChild(td);
      tbody.replaceChildren(tr);
    }
  }
}
