/**
 * @file system-status.test.ts
 * @module tests/system-status.test
 * Unit tests for SystemStatusController, SchemaExplorerController, system health polling, diagnostic banners, and schema diagnostic cards.
 */

import { describe, it, expect, beforeEach, vi, afterEach } from "vitest";
import i18next from "i18next";
import { ChatState } from "../src/state";
import { ChatUI, SystemStatusController, isDbStatusCode } from "../src/ui";
import {
  flushPromises,
  getTestElement,
  mountChatUiTestDom,
  queryTestElement,
} from "./setup";

describe("SystemStatusController & SchemaExplorerController Diagnostics", () => {
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

  it("shows, updates, and hides the system status banner and status chip", () => {
    expect(ui.systemStatusController).toBeInstanceOf(SystemStatusController);
    expect(isDbStatusCode("healthy")).toBe(true);
    expect(isDbStatusCode("empty_db")).toBe(true);
    expect(isDbStatusCode("missing_initial_data")).toBe(true);
    expect(isDbStatusCode("missing_file")).toBe(true);
    expect(isDbStatusCode("corrupted_or_unreadable")).toBe(true);
    expect(isDbStatusCode("not_configured")).toBe(true);
    expect(isDbStatusCode("unknown")).toBe(true);
    expect(isDbStatusCode("")).toBe(false);
    expect(isDbStatusCode(undefined)).toBe(false);

    ui.systemStatusController.showBanner(
      "warning",
      "Test Warning",
      "Warning details",
      "⚠️",
      true,
    );
    const banner = ui.systemStatusController.systemStatusBanner;
    expect(banner.classList.contains("hidden")).toBe(false);
    expect(banner.classList.contains("banner-warning")).toBe(true);
    expect(ui.systemStatusController.bannerTitle.textContent).toBe(
      "Test Warning",
    );
    expect(ui.systemStatusController.bannerDesc.textContent).toBe(
      "Warning details",
    );

    ui.systemStatusController.showBanner(
      "info",
      "No Retry",
      "Details",
      "ℹ️",
      false,
    );
    expect(ui.systemStatusController.bannerRetryBtn.style.display).toBe("none");

    ui.systemStatusController.hideBanner();
    expect(banner.classList.contains("hidden")).toBe(true);

    ui.systemStatusController.updateStatusChip("degraded", "Degraded Service");
    expect(ui.systemStatusController.systemStatusChip.className).toContain(
      "status-degraded",
    );
    expect(ui.systemStatusController.statusChipText.textContent).toBe(
      "Degraded Service",
    );

    // Test synthetic data alert and rule-based SQL alert show/hide, ✖ click, and Escape key
    ui.systemStatusController.showSyntheticDataAlert();
    expect(
      ui.systemStatusController.jaebProvenanceAlert.classList.contains(
        "hidden",
      ),
    ).toBe(false);
    ui.systemStatusController.jaebProvenanceAlert.dispatchEvent(
      new KeyboardEvent("keydown", { key: "Enter", bubbles: true }),
    );
    expect(
      ui.systemStatusController.jaebProvenanceAlert.classList.contains(
        "hidden",
      ),
    ).toBe(false);
    ui.systemStatusController.jaebProvenanceAlert.dispatchEvent(
      new KeyboardEvent("keydown", { key: "Escape", bubbles: true }),
    );
    expect(
      ui.systemStatusController.jaebProvenanceAlert.classList.contains(
        "hidden",
      ),
    ).toBe(true);
    ui.systemStatusController.showSyntheticDataAlert();
    ui.systemStatusController.jaebProvenanceAlertDismissBtn.click();
    expect(
      ui.systemStatusController.jaebProvenanceAlert.classList.contains(
        "hidden",
      ),
    ).toBe(true);

    ui.systemStatusController.showRuleBasedSqlAlert();
    expect(
      ui.systemStatusController.ruleBasedSqlAlert.classList.contains("hidden"),
    ).toBe(false);
    ui.systemStatusController.ruleBasedSqlAlert.dispatchEvent(
      new KeyboardEvent("keydown", { key: "Enter", bubbles: true }),
    );
    expect(
      ui.systemStatusController.ruleBasedSqlAlert.classList.contains("hidden"),
    ).toBe(false);
    ui.systemStatusController.ruleBasedSqlAlert.dispatchEvent(
      new KeyboardEvent("keydown", { key: "Escape", bubbles: true }),
    );
    expect(
      ui.systemStatusController.ruleBasedSqlAlert.classList.contains("hidden"),
    ).toBe(true);
    ui.systemStatusController.showRuleBasedSqlAlert();
    ui.systemStatusController.ruleBasedSqlAlertDismissBtn.click();
    expect(
      ui.systemStatusController.ruleBasedSqlAlert.classList.contains("hidden"),
    ).toBe(true);
  });

  it("checks system status across missing_file, empty_db, missing_initial_data, offline ollama, healthy, and HTTP/network errors", async () => {
    state.createChat();
    // 1. missing_file
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        status: "error",
        database: {
          status_code: "missing_file",
          configured_path: "t1d.duckdb",
        },
        ollama: { accessible: true },
      }),
    });
    await ui.systemStatusController.check(true);
    expect(ui.systemStatusController.systemStatusBanner.className).toContain(
      "banner-warning",
    );

    // 2. empty_db
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        status: "degraded",
        database: { status_code: "empty_db", configured_path: "t1d.duckdb" },
        ollama: { accessible: true },
      }),
    });
    await ui.systemStatusController.check();
    expect(ui.systemStatusController.systemStatusBanner.className).toContain(
      "banner-warning",
    );

    // 3. missing_initial_data
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        status: "degraded",
        database: { status_code: "missing_initial_data", message: "No data" },
        ollama: { accessible: true },
      }),
    });
    await ui.systemStatusController.check();
    expect(ui.systemStatusController.systemStatusBanner.className).toContain(
      "banner-warning",
    );

    // 4. offline ollama switches empty chat model to sql
    state.createChat();
    ui.modelSelect.innerHTML = `<option value="gemma4">Gemma</option><option value="sql">SQL</option>`;
    ui.modelSelect.value = "gemma4";
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        status: "degraded",
        database: {
          status_code: "healthy",
          table_count: 5,
          file_size_bytes: 2097152,
        },
        ollama: { accessible: false },
      }),
    });
    await ui.systemStatusController.check();
    expect(ui.systemStatusController.systemStatusBanner.className).toContain(
      "banner-info",
    );
    expect(ui.modelSelect.value).toBe("sql");

    // 4b. offline ollama when activeChat has messages
    state.addMessageToActiveChat({ role: "user", content: "hi" });
    ui.modelSelect.value = "gemma4";
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        status: "degraded",
        database: { status_code: "healthy", table_count: 5 },
        ollama: { accessible: false },
      }),
    });
    await ui.systemStatusController.check();
    expect(ui.modelSelect.value).toBe("sql");

    // 5. healthy with custom dbPath and unknown status_code
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        status: "healthy",
        database: {
          status_code: "custom_code",
          table_count: 3,
          is_synthetic_faked_data: true,
          data_provenance: "synthetic_seed",
          recommended_jaeb_db_path: "/mnt/data/t1d/jaeb/t1d.duckdb",
        },
        ollama: { accessible: true },
      }),
    });
    await ui.systemStatusController.check(false, "custom.duckdb");
    expect(
      ui.systemStatusController.systemStatusBanner.classList.contains("hidden"),
    ).toBe(true);
    expect(
      ui.systemStatusController.jaebProvenanceAlert.classList.contains(
        "hidden",
      ),
    ).toBe(false);
    expect(
      ui.systemStatusController.jaebProvenanceAlertDesc.textContent,
    ).toContain("/mnt/data/t1d/jaeb/t1d.duckdb");
    expect(state.systemStatus.recommendedJaebDbPath).toBe(
      "/mnt/data/t1d/jaeb/t1d.duckdb",
    );

    await i18next.changeLanguage("ja");
    expect(
      ui.systemStatusController.jaebProvenanceAlertDesc.textContent,
    ).toContain("/mnt/data/t1d/jaeb/t1d.duckdb");
    await i18next.changeLanguage("en");

    ui.systemStatusController.applyHealthPayload({
      status: "healthy",
      database: {
        exists: true,
        connected: true,
        status_code: "healthy",
        is_synthetic_faked_data: true,
      },
      ollama: { accessible: true },
    });
    expect(
      ui.systemStatusController.jaebProvenanceAlertDesc.textContent,
    ).toContain("/Volumes/TOSHIBA_EXT/stanford/t1d/jaeb/t1d.duckdb");

    state.systemStatus.recommendedJaebDbPath = "";
    await i18next.changeLanguage("en");
    ui.systemStatusController.showSyntheticDataAlert();
    expect(
      ui.systemStatusController.jaebProvenanceAlertDesc.textContent,
    ).toContain("/Volumes/TOSHIBA_EXT/stanford/t1d/jaeb/t1d.duckdb");

    // 5b. authentic Jaeb database with LLM selected hides both alerts
    ui.modelSelect.value = "gemma4";
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        status: "healthy",
        database: {
          status_code: "healthy",
          table_count: 544,
          is_synthetic_faked_data: false,
          data_provenance: "jaeb_authentic",
        },
        ollama: { accessible: true },
      }),
    });
    await ui.systemStatusController.check(false, "jaeb.duckdb");
    expect(
      ui.systemStatusController.jaebProvenanceAlert.classList.contains(
        "hidden",
      ),
    ).toBe(true);
    expect(
      ui.systemStatusController.ruleBasedSqlAlert.classList.contains("hidden"),
    ).toBe(true);

    // 6. HTTP 500 error -> offline banner
    mockFetch.mockResolvedValueOnce({ ok: false, status: 500 });
    await ui.systemStatusController.check(true);
    expect(state.systemStatus.backendOnline).toBe(false);
    expect(ui.systemStatusController.systemStatusBanner.className).toContain(
      "banner-danger",
    );
  });

  it("handles clicking banner buttons, status chip, and polling lifecycle", async () => {
    const instructionsBtn = ui.systemStatusController.bannerInstructionsBtn;
    const drawer = ui.systemStatusController.bannerInstructionsDrawer;
    expect(drawer.classList.contains("hidden")).toBe(true);

    instructionsBtn.click();
    expect(drawer.classList.contains("hidden")).toBe(false);
    expect(instructionsBtn.getAttribute("aria-expanded")).toBe("true");

    instructionsBtn.click();
    expect(drawer.classList.contains("hidden")).toBe(true);
    expect(instructionsBtn.getAttribute("aria-expanded")).toBe("false");

    ui.systemStatusController.showBanner("info", "Test", "Desc");
    ui.systemStatusController.bannerDismissBtn.click();
    expect(
      ui.systemStatusController.systemStatusBanner.classList.contains("hidden"),
    ).toBe(true);

    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        status: "healthy",
        database: { status_code: "healthy", table_count: 2 },
      }),
    });
    ui.systemStatusController.systemStatusChip.click();
    await flushPromises();

    mockFetch.mockResolvedValue({
      ok: true,
      json: async () => ({
        status: "healthy",
        database: { status_code: "healthy", table_count: 2 },
        models: [],
        tables: [],
      }),
    });
    ui.systemStatusController.bannerRetryBtn.click();
    await flushPromises();

    vi.useFakeTimers();
    const checkSpy = vi
      .spyOn(ui.systemStatusController, "check")
      .mockResolvedValue(undefined);
    ui.systemStatusController.startPolling();
    ui.systemStatusController.startPolling(1000);
    vi.advanceTimersByTime(1000);
    expect(checkSpy).toHaveBeenCalledTimes(1);
    ui.systemStatusController.stopPolling();
    ui.systemStatusController.stopPolling();
    vi.advanceTimersByTime(1000);
    expect(checkSpy).toHaveBeenCalledTimes(1);
    checkSpy.mockRestore();
    vi.useRealTimers();
  });

  it("handles schema explorer diagnostic cards, database selector, and clipboard copy", async () => {
    // 1. missing_file diagnostic card
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        tables: [],
        database: {
          status_code: "missing_file",
          configured_path: "t1d.duckdb",
        },
      }),
    });
    await ui.schemaExplorerController.loadSchema("custom.duckdb");
    const schemaContent = getTestElement("schema-content");
    expect(
      schemaContent.querySelector(".schema-diagnostic-card"),
    ).not.toBeNull();

    const copyBtn = queryTestElement<HTMLButtonElement>(
      ".copy-cmd-btn",
      schemaContent,
    );
    expect(copyBtn).not.toBeNull();

    const writeTextSpy = vi.fn().mockResolvedValue(undefined);
    Object.assign(navigator, { clipboard: { writeText: writeTextSpy } });
    vi.useFakeTimers();
    copyBtn?.click();
    await vi.runAllTimersAsync();
    vi.useRealTimers();
    expect(writeTextSpy).toHaveBeenCalled();

    // Clipboard error and missing clipboard handling
    writeTextSpy.mockRejectedValueOnce(new Error("denied"));
    await ui.schemaExplorerController.copyCommandToClipboard("test", copyBtn!);
    const origClipboard = navigator.clipboard;
    Object.assign(navigator, { clipboard: null });
    await ui.schemaExplorerController.copyCommandToClipboard("test", copyBtn!);
    Object.assign(navigator, { clipboard: origClipboard });

    // 2. empty_db diagnostic card + reload button
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        tables: [],
        database: { status_code: "empty_db", configured_path: "t1d.duckdb" },
      }),
    });
    await ui.schemaExplorerController.loadSchema();
    const reloadBtn = queryTestElement<HTMLButtonElement>(
      ".reload-schema-btn",
      schemaContent,
    );
    mockFetch.mockResolvedValue({
      ok: true,
      json: async () => ({
        status: "healthy",
        tables: [
          {
            name: "patients",
            columns: [{ name: "PtID", type: "INTEGER" }],
          },
        ],
        database: { status_code: "healthy" },
      }),
    });
    reloadBtn?.click();
    await flushPromises();
    expect(schemaContent.querySelector(".schema-table")).not.toBeNull();

    // 3. missing_initial_data and unknown status_code with empty tables
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        tables: [],
        database: { status_code: "missing_initial_data" },
      }),
    });
    await ui.schemaExplorerController.loadSchema();
    expect(schemaContent.querySelector(".schema-loading")).not.toBeNull();

    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        tables: [
          {
            name: "patients",
            columns: [{ name: "PtID", type: "INTEGER" }],
          },
        ],
        database: {
          status_code: "custom_unknown",
          recommended_jaeb_db_path: "/mnt/data/t1d/jaeb/t1d.duckdb",
        },
      }),
    });
    await ui.schemaExplorerController.loadSchema();

    // 4. fetchDatabases and database dropdown switching
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        databases: [
          {
            name: "t1d.duckdb",
            path: "t1d.duckdb",
            size_bytes: 2 * 1024 * 1024,
          },
          {
            name: "secondary.duckdb",
            path: "secondary.duckdb",
            size_bytes: 0,
          },
        ],
        current_db: "secondary.duckdb",
      }),
    });
    await ui.schemaExplorerController.fetchDatabases();
    const dbSelect = ui.schemaExplorerController.dbSelect;
    expect(dbSelect.options.length).toBe(2);
    dbSelect.value = "secondary.duckdb";
    dbSelect.dispatchEvent(new Event("change"));
    await flushPromises();
    expect(state.currentDb).toBe("secondary.duckdb");

    mockFetch.mockResolvedValueOnce({ ok: false, status: 500 });
    await ui.schemaExplorerController.fetchDatabases();
    expect(state.availableDbs.length).toBe(2);

    mockFetch.mockRejectedValueOnce(new Error("net err"));
    await ui.schemaExplorerController.fetchDatabases();
    expect(state.availableDbs.length).toBe(2);
    expect(dbSelect.options.length).toBe(2);
  });
});
