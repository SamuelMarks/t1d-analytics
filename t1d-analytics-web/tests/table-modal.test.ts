/**
 * @file table-modal.test.ts
 * @module tests/table-modal.test
 * Unit tests for TablePreviewModalController, pagination, sorting, search filtering, and data exports.
 */

import { describe, it, expect, beforeEach, vi, afterEach } from "vitest";
import i18next from "i18next";
import { ChatState } from "../src/state";
import { ChatUI, TablePreviewModalController } from "../src/ui";
import { flushPromises, getTestElement, mountChatUiTestDom } from "./setup";

describe("TablePreviewModalController & Data Exports", () => {
  let state: ChatState;
  let ui: ChatUI;
  let mockFetch: ReturnType<typeof vi.fn>;

  beforeEach(async () => {
    mountChatUiTestDom();
    mockFetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ models: [], schema: [] }),
    });
    vi.stubGlobal("fetch", mockFetch);

    state = new ChatState();
    i18next.changeLanguage("en");
    document.documentElement.lang = "en";
    ui = new ChatUI(state);
    await flushPromises();
  });

  afterEach(() => {
    ui.destroy();
    localStorage.clear();
    document.body.innerHTML = "";
    vi.restoreAllMocks();
  });

  it("handles table modal interaction, pagination, sorting, and search", async () => {
    expect(ui.tableModalController).toBeInstanceOf(TablePreviewModalController);

    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () =>
        Promise.resolve({
          rows: [
            { id: 1, name: "Test" },
            { id: 2, name: null },
          ],
        }),
    });

    await ui.tableModalController.open("test_table");

    const modal = document.getElementById("table-modal");
    expect(modal?.getAttribute("aria-hidden")).toBeNull();

    // Next page
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve({ rows: [{ id: 3, name: "Test 2" }] }),
    });
    const nextBtn = document.getElementById("next-page-btn");
    nextBtn?.dispatchEvent(new MouseEvent("click"));
    await flushPromises();
    expect(ui["currentPage"]).toBe(2);

    // Prev page
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve({ rows: [{ id: 1, name: "Test" }] }),
    });
    const prevBtn = document.getElementById("prev-page-btn");
    prevBtn?.dispatchEvent(new MouseEvent("click"));
    await flushPromises();
    expect(ui["currentPage"]).toBe(1);

    // Prev page bounds
    prevBtn?.dispatchEvent(new MouseEvent("click"));
    expect(ui["currentPage"]).toBe(1);

    // Check accessibility attributes in modal table
    const ths = document
      .getElementById("modal-table-head")
      ?.querySelectorAll("th");
    expect(ths?.[0]?.getAttribute("scope")).toBe("col");

    // Test sort by clicking on th
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve({ rows: [{ id: 1, name: "Test" }] }),
    });
    ths?.[0]?.dispatchEvent(new MouseEvent("click"));
    await flushPromises();
    expect(ui["currentSortBy"]).toBe("id");
    expect(ui["currentSortOrder"]).toBe("asc");

    // Click same th again to toggle desc
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve({ rows: [{ id: 1, name: "Test" }] }),
    });
    ths?.[0]?.dispatchEvent(new MouseEvent("click"));
    await flushPromises();
    expect(ui["currentSortOrder"]).toBe("desc");

    // Press Enter or Space on th to sort
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve({ rows: [{ id: 1, name: "Test" }] }),
    });
    ths?.[0]?.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter" }));
    await flushPromises();
    expect(ui["currentSortOrder"]).toBe("asc");

    // Press other key on th (no sort)
    ths?.[0]?.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape" }));

    // Test search input
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve({ rows: [{ id: 1, name: "Test" }] }),
    });
    const searchInput = getTestElement<HTMLInputElement>("modal-table-search");
    searchInput.value = "needle";
    searchInput.dispatchEvent(new Event("input"));
    await flushPromises();
    expect(ui["currentSearchTerm"]).toBe("needle");

    // Test sort select change
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve({ rows: [{ id: 1, name: "Test" }] }),
    });
    const sortSelect = getTestElement<HTMLSelectElement>(
      "modal-table-sort-col",
    );
    sortSelect.value = "name";
    sortSelect.dispatchEvent(new Event("change"));
    await flushPromises();
    expect(ui["currentSortBy"]).toBe("name");

    // Test sort order button toggle (asc -> desc and desc -> asc)
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve({ rows: [{ id: 1, name: "Test" }] }),
    });
    const sortOrderBtn = getTestElement<HTMLButtonElement>(
      "modal-table-sort-order",
    );
    sortOrderBtn.dispatchEvent(new MouseEvent("click"));
    await flushPromises();
    expect(ui["currentSortOrder"]).toBe("desc");

    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve({ rows: [{ id: 1, name: "Test" }] }),
    });
    sortOrderBtn.dispatchEvent(new MouseEvent("click"));
    await flushPromises();
    expect(ui["currentSortOrder"]).toBe("asc");

    // Modal Overlay click
    modal?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    expect(modal?.getAttribute("aria-hidden")).toBe("true");
  });

  it("handles table fetch empty rows, errors, and HTTP error mapping", async () => {
    ui["currentPage"] = 2;
    ui["currentTable"] = "test_table";
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve({ rows: [], total_pages: 1 }),
    });
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve({ rows: [{ id: 1 }], total_pages: 1 }),
    });
    await ui["fetchTableData"]();
    expect(ui["currentPage"]).toBe(1);

    ui["currentPage"] = 1;
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve({ rows: [] }),
    });
    await ui["fetchTableData"]();
    expect(ui["currentPage"]).toBe(1);
    expect(document.getElementById("modal-table-body")?.innerHTML).toContain(
      "No data available",
    );

    mockFetch.mockRejectedValueOnce(new Error("Network Error"));
    await ui["fetchTableData"]();
    expect(document.getElementById("modal-table-body")?.innerHTML).toContain(
      "error-text",
    );

    mockFetch.mockResolvedValueOnce({ ok: false, status: 500 });
    await ui["fetchTableData"]();
    expect(document.getElementById("modal-table-body")?.innerHTML).toContain(
      "Failed to load data",
    );

    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 502,
      json: () => Promise.reject(new Error("Invalid JSON")),
    });
    await ui["fetchTableData"]();
    expect(document.getElementById("modal-table-body")?.innerHTML).toContain(
      "HTTP 502",
    );

    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 400,
      json: () => Promise.resolve({ detail: "Unmapped error string" }),
    });
    await ui["fetchTableData"]();
    expect(document.getElementById("modal-table-body")?.innerHTML).toContain(
      "Unmapped error string",
    );

    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 400,
      json: () =>
        Promise.resolve({
          detail: { error_code: "unmapped", params: { error: "some data" } },
        }),
    });
    await ui["fetchTableData"]();
    expect(document.getElementById("modal-table-body")?.textContent).toContain(
      'unmapped: {"error":"some data"}',
    );

    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 400,
      json: () => Promise.resolve({ detail: "backend.invalidTable" }),
    });
    await ui["fetchTableData"]();
    expect(document.getElementById("modal-table-body")?.innerHTML).toContain(
      "Invalid table name",
    );

    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 500,
      json: () =>
        Promise.resolve({
          detail: {
            error_code: "backend.serverError",
            params: { error: "something broke" },
          },
        }),
    });
    await ui["fetchTableData"]();
    expect(document.getElementById("modal-table-body")?.innerHTML).toContain(
      "Internal server error: something broke",
    );

    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 403,
      json: () => Promise.resolve({ otherField: "something" }),
    });
    await ui["fetchTableData"]();
    expect(document.getElementById("modal-table-body")?.innerHTML).toContain(
      "HTTP 403",
    );

    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 422,
      json: () =>
        Promise.resolve({
          detail: [{ loc: ["body", "message"], msg: "field required" }],
        }),
    });
    await ui["fetchTableData"]();
    expect(document.getElementById("modal-table-body")?.innerHTML).toContain(
      "object Object",
    );
  });

  it("handles modal focus trap, Escape key dismissal, and focus restoration", async () => {
    const input = getTestElement<HTMLTextAreaElement>("chat-input");
    input.disabled = false;
    input.focus();

    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve({ rows: [{ id: 1 }] }),
    });

    const modal = getTestElement("table-modal");
    const closeBtn = getTestElement("close-modal-btn");

    await ui["openTableModal"]("test_table");
    expect(modal.getAttribute("aria-hidden")).toBeNull();
    expect(modal.getAttribute("aria-modal")).toBe("true");

    closeBtn.focus();
    const focusable = modal.querySelectorAll<HTMLElement>(
      'button:not([disabled]), [href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])',
    );
    const lastElement = focusable[focusable.length - 1];
    modal.dispatchEvent(
      new KeyboardEvent("keydown", { key: "Tab", shiftKey: true }),
    );
    expect(document.activeElement).toBe(lastElement);

    modal.dispatchEvent(new KeyboardEvent("keydown", { key: "Tab" }));
    expect(document.activeElement).toBe(closeBtn);

    modal.dispatchEvent(
      new KeyboardEvent("keydown", { key: "ArrowDown", bubbles: true }),
    );
    expect(document.activeElement).toBe(closeBtn);

    const dummy1 = document.createElement("button");
    const dummy2 = document.createElement("button");
    modal.appendChild(dummy1);
    modal.appendChild(dummy2);
    dummy1.focus();
    modal.dispatchEvent(new KeyboardEvent("keydown", { key: "Tab" }));
    modal.dispatchEvent(
      new KeyboardEvent("keydown", { key: "Tab", shiftKey: true }),
    );
    dummy1.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    dummy1.remove();
    dummy2.remove();

    // Press Escape to close
    modal.dispatchEvent(
      new KeyboardEvent("keydown", { key: "Escape", bubbles: true }),
    );
    expect(modal.getAttribute("aria-hidden")).toBe("true");
    expect(modal.getAttribute("aria-modal")).toBeNull();

    // Restore focus to chat input when previous focus is removed from DOM
    const dummyBtn = document.createElement("button");
    document.body.appendChild(dummyBtn);
    dummyBtn.focus();

    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve({ rows: [] }),
    });
    await ui["openTableModal"]("test_table");
    document.body.removeChild(dummyBtn);
    modal.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    expect(modal.getAttribute("aria-hidden")).toBe("true");
    expect(document.activeElement).toBe(input);

    // Close button click
    closeBtn.dispatchEvent(new MouseEvent("click"));
    expect(modal.getAttribute("aria-hidden")).toBe("true");
  });

  it("handles missing modal DOM elements and focusableElements.length === 0 safely", async () => {
    const searchInput = document.getElementById("modal-table-search");
    const sortSelect = document.getElementById("modal-table-sort-col");
    const sortOrderBtn = document.getElementById("modal-table-sort-order");
    searchInput?.remove();
    sortSelect?.remove();
    sortOrderBtn?.remove();

    document.getElementById("modal-export-csv-btn")?.remove();
    document.getElementById("modal-export-json-btn")?.remove();
    document.getElementById("modal-export-full-csv-btn")?.remove();
    document.getElementById("modal-export-excel-btn")?.remove();
    document.getElementById("modal-export-parquet-btn")?.remove();

    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () =>
        Promise.resolve({
          rows: [{ id: 1 }],
          total_count: 1,
          total_pages: 1,
        }),
    });
    await ui["openTableModal"]("patients");

    const th = document.querySelector<HTMLElement>("#modal-table-head th");
    if (th) {
      mockFetch.mockResolvedValue({
        ok: true,
        json: () => Promise.resolve({ rows: [{ id: 1 }], total_pages: 1 }),
      });
      th.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter" }));
      th.dispatchEvent(new KeyboardEvent("keydown", { key: " " }));
      th.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape" }));
    }

    const tableModal = getTestElement("table-modal");
    const allInteractive = tableModal.querySelectorAll<HTMLElement>(
      "button, input, select, textarea, [tabindex]",
    );
    allInteractive.forEach((el) => {
      el.setAttribute("disabled", "true");
      if (el.hasAttribute("tabindex")) {
        el.setAttribute("tabindex", "-1");
      }
    });
    tableModal.dispatchEvent(
      new KeyboardEvent("keydown", { key: "Tab", bubbles: true }),
    );

    document.getElementById("table-modal")?.remove();
    await ui["openTableModal"]("test");

    document.getElementById("table-loading")?.remove();
    await ui["fetchTableData"]();
  });

  it("exports rows to CSV and JSON and triggers export from modal and chat buttons", async () => {
    let capturedBlob: Blob | null = null;
    const createObjectURL = vi.fn().mockImplementation((blob: Blob) => {
      capturedBlob = blob;
      return "blob:mock-csv";
    });
    const revokeObjectURL = vi.fn();
    globalThis.URL.createObjectURL = createObjectURL;
    globalThis.URL.revokeObjectURL = revokeObjectURL;

    ui.exportRowsToCsv([], "test.csv");
    expect(createObjectURL).not.toHaveBeenCalled();

    ui.exportRowsToCsv([{}, {}], "test.csv");
    expect(createObjectURL).not.toHaveBeenCalled();

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
        formula2: "@SUM",
        formula3: `${tab}Tab`,
        numPos: "+12.5",
        numNeg: "-42",
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
        otherKey: "disparate",
        details: null,
      },
    ];
    ui.exportRowsToCsv(sampleRows, "test.csv");
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
    expect(csvText).toContain("'@SUM");
    expect(csvText).toContain("'+CMD");
    expect(csvText).toContain("'-CALC");
    expect(csvText).toContain('"+12.5","-42"');

    createObjectURL.mockClear();
    ui.exportRowsToJson([], "test.json");
    expect(createObjectURL).not.toHaveBeenCalled();

    ui.exportRowsToJson([{ id: 1, name: "Alice" }], "test.json");
    expect(createObjectURL).toHaveBeenCalled();

    const exportCsvSpy = vi
      .spyOn(ui, "exportRowsToCsv")
      .mockImplementation(() => {});
    const exportJsonSpy = vi
      .spyOn(ui, "exportRowsToJson")
      .mockImplementation(() => {});

    state.createChat();
    state.addMessageToActiveChat({
      role: "assistant",
      content: "Here are the query results:",
      sqlQuery: "SELECT * FROM users",
      sqlResult: [{ id: 1, name: "Alice" }],
    });
    ui["renderMessages"]();

    document.querySelector<HTMLButtonElement>(".export-csv-btn")?.click();
    expect(exportCsvSpy).toHaveBeenCalled();
    document.querySelector<HTMLButtonElement>(".export-json-btn")?.click();
    expect(exportJsonSpy).toHaveBeenCalled();

    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () =>
        Promise.resolve({
          rows: [{ id: 1, patient: "P100" }],
          total_count: 1,
          total_pages: 1,
        }),
    });
    await ui["openTableModal"]("patients");

    document.getElementById("modal-export-csv-btn")?.click();
    expect(exportCsvSpy).toHaveBeenCalledTimes(2);
    document.getElementById("modal-export-json-btn")?.click();
    expect(exportJsonSpy).toHaveBeenCalledTimes(2);

    const windowOpenSpy = vi
      .spyOn(window, "open")
      .mockImplementation(() => null);
    const modalFullCsvBtn = document.getElementById(
      "modal-export-full-csv-btn",
    );
    const modalExcelBtn = document.getElementById("modal-export-excel-btn");
    const modalParquetBtn = document.getElementById("modal-export-parquet-btn");

    state.setCurrentDb("my_custom.duckdb");
    modalFullCsvBtn?.click();
    expect(windowOpenSpy).toHaveBeenCalledWith(
      expect.stringContaining(
        "/api/export/csv?table_name=patients&db_path=my_custom.duckdb",
      ),
    );
    modalExcelBtn?.click();
    expect(windowOpenSpy).toHaveBeenCalledWith(
      expect.stringContaining(
        "/api/export/excel?table_name=patients&db_path=my_custom.duckdb",
      ),
    );
    modalParquetBtn?.click();
    expect(windowOpenSpy).toHaveBeenCalledWith(
      expect.stringContaining(
        "/api/export/parquet?table_name=patients&db_path=my_custom.duckdb",
      ),
    );

    state.setCurrentDb("");
    modalFullCsvBtn?.click();
    expect(windowOpenSpy).toHaveBeenCalledWith(
      "/api/export/csv?table_name=patients",
    );
    modalExcelBtn?.click();
    expect(windowOpenSpy).toHaveBeenCalledWith(
      "/api/export/excel?table_name=patients",
    );
    modalParquetBtn?.click();
    expect(windowOpenSpy).toHaveBeenCalledWith(
      "/api/export/parquet?table_name=patients",
    );

    state.setCurrentDb("t1d.duckdb");
    modalFullCsvBtn?.click();
    expect(windowOpenSpy).toHaveBeenCalledWith(
      "/api/export/csv?table_name=patients",
    );
    windowOpenSpy.mockRestore();
  });
});
