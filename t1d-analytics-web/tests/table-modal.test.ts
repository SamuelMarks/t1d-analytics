/**
 * @file table-modal.test.ts
 * @module tests/table-modal.test
 * Unit tests for TablePreviewModalController, pagination, sorting, search filtering, and data exports.
 */

import { describe, it, expect, beforeEach, vi, afterEach } from "vitest";
import i18next from "i18next";
import { exportRowsToCsv, exportRowsToJson } from "../src/exporters";
import { ChatState } from "../src/state";
import { TablePreviewModalController } from "../src/controllers/table-modal-controller";
import { ChatUI } from "../src/ui";
import { flushPromises, getTestElement, mountChatUiTestDom } from "./setup";

describe("TablePreviewModalController & Data Exports", () => {
  let state: ChatState;
  let ui: ChatUI;
  let mockFetch: ReturnType<typeof vi.fn>;

  beforeEach(async () => {
    mountChatUiTestDom();
    mockFetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ models: [], tables: [] }),
    });
    vi.stubGlobal("fetch", mockFetch);

    state = new ChatState();
    await i18next.changeLanguage("en");
    document.documentElement.lang = "en";
    ui = new ChatUI(state);
    await flushPromises();
  });

  afterEach(() => {
    ui.dispose();
    localStorage.clear();
    document.body.innerHTML = "";
    vi.restoreAllMocks();
  });

  it("handles table modal interaction, pagination, sorting, and search", async () => {
    expect(ui.tableModalController).toBeInstanceOf(TablePreviewModalController);
    ui.tableModalController.close();

    state.setCurrentDb("custom_table.duckdb");
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        columns: ["id", "name"],
        rows: [
          { id: 1, name: "Test" },
          { id: 2, name: null },
        ],
        total_rows: 50,
        page: 1,
        total_pages: 2,
      }),
    });

    await ui.tableModalController.open("test_table");
    expect(mockFetch).toHaveBeenCalledWith(
      "/api/table/test_table?limit=25&offset=0&db_path=custom_table.duckdb",
      expect.objectContaining({ signal: expect.any(AbortSignal) }),
    );
    state.setCurrentDb("t1d.duckdb");

    // Clicking prevBtn on page 1 does nothing
    ui.tableModalController.prevBtn.disabled = false;
    ui.tableModalController.prevBtn.click();
    expect(ui.tableModalController.currentPage).toBe(1);

    const modal = ui.tableModalController.modal;
    expect(modal.getAttribute("aria-modal")).toBe("true");

    // Next page
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        columns: ["id", "name"],
        rows: [{ id: 3, name: "Test 2" }],
        total_rows: 50,
        page: 2,
        total_pages: 2,
      }),
    });
    ui.tableModalController.nextBtn.click();
    await flushPromises();
    expect(ui.tableModalController.currentPage).toBe(2);

    // Prev page
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        columns: ["id", "name"],
        rows: [{ id: 1, name: "Test" }],
        total_rows: 50,
        page: 1,
        total_pages: 2,
      }),
    });
    ui.tableModalController.prevBtn.click();
    await flushPromises();
    expect(ui.tableModalController.currentPage).toBe(1);

    // Sort by clicking on th
    const ths = ui.tableModalController.theadEl.querySelectorAll("th");
    expect(ths[0]?.getAttribute("scope")).toBe("col");

    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        columns: ["id", "name"],
        rows: [{ id: 1, name: "Test" }],
        total_rows: 1,
        page: 1,
        total_pages: 1,
      }),
    });
    ths[0]?.click();
    await flushPromises();
    expect(ui.tableModalController.currentSortBy).toBe("id");
    expect(ui.tableModalController.currentSortOrder).toBe("asc");

    // Click same th again to toggle desc
    const thsAfterAsc = ui.tableModalController.theadEl.querySelectorAll("th");
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        columns: ["id", "name"],
        rows: [{ id: 1, name: "Test" }],
        total_rows: 1,
        page: 1,
        total_pages: 1,
      }),
    });
    thsAfterAsc[0]?.click();
    await flushPromises();
    expect(ui.tableModalController.currentSortOrder).toBe("desc");

    // Click same th a 3rd time to toggle back to asc, and test keyboard Enter/Space sorting
    const thsAfterDesc = ui.tableModalController.theadEl.querySelectorAll("th");
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        rows: [{ id: 1, name: "Test" }],
      }),
    });
    thsAfterDesc[0]?.dispatchEvent(
      new KeyboardEvent("keydown", { key: "Enter", bubbles: true }),
    );
    await flushPromises();
    expect(ui.tableModalController.currentSortOrder).toBe("asc");

    // Non-Enter/Space key does nothing
    ui.tableModalController.theadEl
      .querySelectorAll("th")[0]
      ?.dispatchEvent(
        new KeyboardEvent("keydown", { key: "Tab", bubbles: true }),
      );

    // Next page returning 0 rows on page > 1 auto-decrements back to page 1
    ui.tableModalController.currentPage = 2;
    mockFetch
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({ rows: [], total_pages: 1 }),
      })
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({ rows: [{ id: 1, name: "Test" }], total_pages: 1 }),
      });
    await ui.tableModalController.fetchTableData();
    expect(ui.tableModalController.currentPage).toBe(1);

    // Search input with debounce
    vi.useFakeTimers();
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        columns: ["id", "name"],
        rows: [{ id: 1, name: "Test" }],
        total_rows: 1,
        page: 1,
        total_pages: 1,
      }),
    });
    const searchInput = ui.tableModalController.searchInput;
    searchInput.value = "first";
    searchInput.dispatchEvent(new Event("input"));
    searchInput.value = "needle";
    searchInput.dispatchEvent(new Event("input"));
    await vi.advanceTimersByTimeAsync(300);
    vi.useRealTimers();
    await flushPromises();
    expect(ui.tableModalController.currentSearchTerm).toBe("needle");

    // Sort select change
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        columns: ["id", "name"],
        rows: [{ id: 1, name: "Test" }],
        total_rows: 1,
        page: 1,
        total_pages: 1,
      }),
    });
    const sortSelect = ui.tableModalController.sortColSelect;
    sortSelect.value = "name";
    sortSelect.dispatchEvent(new Event("change"));
    await flushPromises();
    expect(ui.tableModalController.currentSortBy).toBe("name");

    // Sort order button toggle (desc -> asc and asc -> desc)
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        columns: ["id", "name"],
        rows: [{ id: 1, name: "Test" }],
        total_rows: 1,
        page: 1,
        total_pages: 1,
      }),
    });
    ui.tableModalController.sortOrderBtn.click();
    await flushPromises();
    expect(ui.tableModalController.currentSortOrder).toBe("desc");

    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        columns: ["id", "name"],
        rows: [{ id: 1, name: "Test" }],
        total_rows: 1,
        page: 1,
        total_pages: 1,
      }),
    });
    ui.tableModalController.sortOrderBtn.click();
    await flushPromises();
    expect(ui.tableModalController.currentSortOrder).toBe("asc");

    // Refresh if open on locale change
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        columns: ["id"],
        rows: [{ id: 1 }],
        total_rows: 1,
        page: 1,
        total_pages: 1,
      }),
    });
    await ui.tableModalController.refreshIfOpen();

    // Backdrop click closes modal
    modal.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    expect(modal.getAttribute("aria-hidden")).toBe("true");
  });

  it("handles table fetch empty rows, errors, HTTP error mapping, and Escape key", async () => {
    const input = getTestElement<HTMLTextAreaElement>("chat-input");
    input.disabled = false;
    input.focus();

    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        columns: ["id"],
        rows: [],
        total_rows: 0,
        page: 1,
        total_pages: 0,
      }),
    });
    await ui.tableModalController.open("empty_table");
    expect(ui.tableModalController.tbodyEl.textContent).toContain(
      "No data available",
    );

    mockFetch.mockRejectedValueOnce("Non-Error failure");
    await ui.tableModalController.fetchTableData();
    expect(ui.tableModalController.tbodyEl.textContent).toContain(
      "Non-Error failure",
    );

    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 500,
      json: async () => ({
        detail: {
          error_code: "backend.serverError",
          params: { error: "something broke" },
        },
      }),
    });
    await ui.tableModalController.fetchTableData();
    expect(ui.tableModalController.tbodyEl.textContent).toContain(
      "Internal server error: something broke",
    );

    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 502,
      json: async () => ({ detail: {} }),
    });
    await ui.tableModalController.fetchTableData();
    expect(ui.tableModalController.tbodyEl.textContent).toContain(
      "[object Object]",
    );

    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 502,
      json: async () => ({ detail: "Plain gateway error" }),
    });
    await ui.tableModalController.fetchTableData();
    expect(ui.tableModalController.tbodyEl.textContent).toContain(
      "Plain gateway error",
    );

    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 503,
      json: async () => ({}),
    });
    await ui.tableModalController.fetchTableData();
    expect(ui.tableModalController.tbodyEl.textContent).toContain("HTTP 503");

    // Escape key closes modal and restores focus
    ui.tableModalController.modal.dispatchEvent(
      new KeyboardEvent("keydown", { key: "Escape", bubbles: true }),
    );
    expect(ui.tableModalController.modal.getAttribute("aria-hidden")).toBe(
      "true",
    );

    // Out-of-order stale response and error in fetchTableData are discarded when a newer request is in flight
    let resolveStale!: (value: unknown) => void;
    let rejectStaleErr!: (reason?: unknown) => void;
    mockFetch
      .mockImplementationOnce(
        () =>
          new Promise((resolve) => {
            resolveStale = resolve;
          }),
      )
      .mockImplementationOnce(
        () =>
          new Promise((_, reject) => {
            rejectStaleErr = reject;
          }),
      )
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          columns: ["id"],
          rows: [{ id: 999 }],
          total_rows: 1,
          page: 1,
          total_pages: 1,
        }),
      });

    const staleSuccessPromise = ui.tableModalController.fetchTableData();
    const staleErrorPromise = ui.tableModalController.fetchTableData();
    const freshPromise = ui.tableModalController.fetchTableData();

    await freshPromise;
    expect(ui.tableModalController.currentModalRows).toEqual([{ id: 999 }]);

    resolveStale({
      ok: true,
      json: async () => ({
        columns: ["id"],
        rows: [{ id: 111 }],
        total_rows: 1,
        page: 1,
        total_pages: 1,
      }),
    });
    rejectStaleErr(new Error("Stale network error"));
    await staleSuccessPromise;
    await staleErrorPromise;
    expect(ui.tableModalController.currentModalRows).toEqual([{ id: 999 }]);
  });

  it("exports rows to CSV and JSON and triggers export from modal and chat buttons", async () => {
    vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => {});
    let capturedBlob: Blob | null = null;
    const createObjectURL = vi.fn().mockImplementation((blob: Blob) => {
      capturedBlob = blob;
      return "blob:mock-csv";
    });
    const revokeObjectURL = vi.fn();
    globalThis.URL.createObjectURL = createObjectURL;
    globalThis.URL.revokeObjectURL = revokeObjectURL;

    const cr = String.fromCharCode(13);
    const lf = String.fromCharCode(10);
    const tab = String.fromCharCode(9);
    const sampleRows = [
      {
        id: 1,
        name: 'Alice "Wonderland"',
        note: null,
        extra: undefined,
        formula1: "=1+1",
        formula2: "  @SUM",
        formula3: `${tab}Tab`,
        numPos: "+12.5",
        numNeg: "-42",
        spacesOnly: "   ",
        details: { note: "ok" },
      },
      {
        id: 2,
        name: `Bob${lf}Smith`,
        note: "some note",
        extra: 42,
        formula1: "+CMD",
        formula2: "-CALC",
        formula3: `${cr}Return`,
        spacesOnly: "",
        otherKey: "disparate",
        details: null,
      },
    ];
    exportRowsToCsv([], "empty.csv");
    exportRowsToCsv(sampleRows, "test.csv");
    expect(createObjectURL).toHaveBeenCalled();
    expect(revokeObjectURL).toHaveBeenCalledWith("blob:mock-csv");
    expect(capturedBlob).not.toBeNull();
    const csvText = await new Promise<string>((resolve) => {
      const reader = new FileReader();
      reader.onload = () => resolve(String(reader.result));
      if (capturedBlob) {
        reader.readAsText(capturedBlob);
      }
    });
    expect(csvText).toContain(`${cr}${lf}`);
    expect(csvText).toContain('Alice ""Wonderland""');
    expect(csvText).toContain(`"Bob${lf}Smith"`);
    expect(csvText).toContain('"{""note"":""ok""}"');
    expect(csvText).toContain("'=1+1");
    expect(csvText).toContain("'  @SUM");
    expect(csvText).toContain("'+CMD");
    expect(csvText).toContain("'-CALC");
    expect(csvText).toContain('"+12.5","-42"');

    createObjectURL.mockClear();
    exportRowsToJson([], "empty.json");
    exportRowsToJson([{ id: 1, name: "Alice" }], "test.json");
    expect(createObjectURL).toHaveBeenCalled();

    state.createChat();
    state.addMessageToActiveChat({
      role: "assistant",
      content: "Here are the query results:",
      sqlQuery: "SELECT * FROM users",
      sqlResult: [{ id: 1, name: "Alice" }],
    });
    ui.messageRenderer.renderMessages();

    createObjectURL.mockClear();
    document.querySelector<HTMLButtonElement>(".export-csv-btn")?.click();
    expect(createObjectURL).toHaveBeenCalledTimes(1);
    document.querySelector<HTMLButtonElement>(".export-json-btn")?.click();
    expect(createObjectURL).toHaveBeenCalledTimes(2);

    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        columns: ["id", "patient"],
        rows: [{ id: 1, patient: "P100" }],
        total_rows: 1,
        page: 1,
        total_pages: 1,
      }),
    });
    await ui.tableModalController.open("patients");

    ui.tableModalController.csvBtn.click();
    expect(createObjectURL).toHaveBeenCalledTimes(3);
    ui.tableModalController.jsonBtn.click();
    expect(createObjectURL).toHaveBeenCalledTimes(4);

    createObjectURL.mockClear();
    state.credentialsStore.setBackendApiKey("secret-bearer-123");
    state.setCurrentDb("my_custom.duckdb");

    for (const expectedUrl of [
      "/api/export/csv?table_name=patients&db_path=my_custom.duckdb",
      "/api/export/excel?table_name=patients&db_path=my_custom.duckdb",
      "/api/export/parquet?table_name=patients&db_path=my_custom.duckdb",
    ]) {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        blob: async () => new Blob(["export-bytes"]),
      });
      if (expectedUrl.includes("/csv")) {
        ui.tableModalController.fullCsvBtn.click();
      } else if (expectedUrl.includes("/excel")) {
        ui.tableModalController.excelBtn.click();
      } else {
        ui.tableModalController.parquetBtn.click();
      }
      await flushPromises();
      expect(mockFetch).toHaveBeenLastCalledWith(
        expectedUrl,
        expect.objectContaining({
          headers: expect.objectContaining({
            Authorization: "Bearer secret-bearer-123",
          }),
        }),
      );
    }
    expect(createObjectURL).toHaveBeenCalledTimes(3);

    state.credentialsStore.removeBackendApiKey();
    state.setCurrentDb("t1d.duckdb");
    for (const btn of [
      ui.tableModalController.fullCsvBtn,
      ui.tableModalController.excelBtn,
      ui.tableModalController.parquetBtn,
    ]) {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        blob: async () => new Blob(["export-bytes"]),
      });
      btn.click();
      await flushPromises();
    }
    expect(createObjectURL).toHaveBeenCalledTimes(6);

    // Server export HTTP failure announces error cleanly without throwing
    const announceSpy = vi.spyOn(ui, "announce");
    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 401,
      json: async () => ({ detail: "Unauthorized export" }),
    });
    await ui.tableModalController.triggerServerExport("csv");
    expect(announceSpy).toHaveBeenCalledWith(
      expect.stringContaining("Unauthorized export"),
      true,
    );
    announceSpy.mockRestore();

    // Closing modal while a search debounce timer is pending clears the timer
    vi.useFakeTimers();
    ui.tableModalController.searchInput.value = "cancelled_query";
    ui.tableModalController.searchInput.dispatchEvent(new Event("input"));
    ui.tableModalController.closeBtn.click();
    await vi.advanceTimersByTimeAsync(300);
    vi.useRealTimers();
    expect(ui.tableModalController.modal.getAttribute("aria-hidden")).toBe(
      "true",
    );
  });
});
