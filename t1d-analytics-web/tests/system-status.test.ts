/**
 * @file system-status.test.ts
 * @module tests/system-status.test
 * Unit tests for SystemStatusController, system health polling, diagnostic banners, and schema explorer diagnostic cards.
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

describe("SystemStatusController & Health Diagnostics", () => {
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

  it("shows, updates, and hides the system status banner", () => {
    ui.showBanner("warning", "Test Warning", "Warning details", "⚠️", true);
    const banner = document.getElementById("system-status-banner");
    expect(banner?.classList.contains("hidden")).toBe(false);
    expect(banner?.classList.contains("banner-warning")).toBe(true);
    expect(document.getElementById("banner-title")?.textContent).toBe(
      "Test Warning",
    );
    expect(document.getElementById("banner-desc")?.textContent).toBe(
      "Warning details",
    );

    ui.hideBanner();
    expect(banner?.classList.contains("hidden")).toBe(true);
  });

  it("updates the system status chip", () => {
    ui.updateStatusChip("degraded", "Degraded Service");
    const chip = document.getElementById("system-status-chip");
    expect(chip?.className).toContain("status-degraded");
    expect(document.getElementById("status-chip-text")?.textContent).toBe(
      "Degraded Service",
    );

    ui.updateStatusChip("healthy", "All Normal");
    expect(chip?.className).toContain("status-healthy");
    expect(document.getElementById("status-chip-text")?.textContent).toBe(
      "All Normal",
    );

    state.setSystemStatus({
      dbFileSizeBytes: 5242880,
      tableCount: 3,
    });
    ui.updateStatusChip("healthy", "All Normal");
    expect(chip?.getAttribute("title")).toContain("5.0 MB (3 tables)");
  });

  it("checks system status and handles missing_file database status", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () =>
        Promise.resolve({
          status: "error",
          database: {
            status_code: "missing_file",
            configured_path: "t1d.duckdb",
            exists: false,
          },
          ollama: { accessible: true },
        }),
    });

    await ui.checkSystemStatus(true);
    const banner = document.getElementById("system-status-banner");
    expect(banner?.classList.contains("hidden")).toBe(false);
    expect(banner?.classList.contains("banner-warning")).toBe(true);
  });

  it("checks system status and handles empty_db database status", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () =>
        Promise.resolve({
          status: "degraded",
          database: {
            status_code: "empty_db",
            table_count: 0,
            has_initial_data: false,
          },
          ollama: { accessible: true },
        }),
    });

    await ui.checkSystemStatus();
    const banner = document.getElementById("system-status-banner");
    expect(banner?.classList.contains("hidden")).toBe(false);
    expect(banner?.classList.contains("banner-warning")).toBe(true);
  });

  it("checks system status and handles missing_initial_data status", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () =>
        Promise.resolve({
          status: "degraded",
          database: {
            status_code: "missing_initial_data",
            table_count: 1,
            has_initial_data: false,
          },
          ollama: { accessible: true },
        }),
    });

    await ui.checkSystemStatus();
    const banner = document.getElementById("system-status-banner");
    expect(banner?.classList.contains("hidden")).toBe(false);
    expect(banner?.classList.contains("banner-warning")).toBe(true);
  });

  it("checks system status and handles offline ollama", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () =>
        Promise.resolve({
          status: "degraded",
          database: {
            status_code: "healthy",
            table_count: 5,
            has_initial_data: true,
          },
          ollama: { accessible: false },
        }),
    });

    const modelSelect = getTestElement<HTMLSelectElement>("model-select");
    modelSelect.value = "gemma4";

    await ui.checkSystemStatus();
    const banner = document.getElementById("system-status-banner");
    expect(banner?.classList.contains("hidden")).toBe(false);
    expect(banner?.classList.contains("banner-info")).toBe(true);
    expect(modelSelect.value).toBe("sql");

    // Also test when an untouched empty active chat exists so its model is updated to 'sql'
    const chat = state.createChat();
    expect(chat.model).toBe("gemma4");
    modelSelect.value = "gemma4";
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () =>
        Promise.resolve({
          status: "degraded",
          database: {
            status_code: "healthy",
            table_count: 5,
            has_initial_data: true,
          },
          ollama: { accessible: false },
        }),
    });
    await ui.checkSystemStatus();
    expect(chat.model).toBe("sql");
  });

  it("checks system status and handles fully healthy state", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () =>
        Promise.resolve({
          status: "healthy",
          database: {
            status_code: "healthy",
            table_count: 5,
            has_initial_data: true,
          },
          ollama: { accessible: true },
        }),
    });

    await ui.checkSystemStatus();
    const banner = document.getElementById("system-status-banner");
    expect(banner?.classList.contains("hidden")).toBe(true);
  });

  it("checks system status and handles fetch failure (offline)", async () => {
    mockFetch.mockRejectedValueOnce(new Error("Failed to fetch"));

    await ui.checkSystemStatus(true);
    const banner = document.getElementById("system-status-banner");
    expect(banner?.classList.contains("hidden")).toBe(false);
    expect(banner?.classList.contains("banner-danger")).toBe(true);
    expect(ui["chatInput"].disabled).toBe(true);
  });

  it("handles clicking banner buttons and status chip", async () => {
    const instructionsBtn = getTestElement<HTMLButtonElement>(
      "banner-instructions-btn",
    );
    const drawer = getTestElement("banner-instructions-drawer");
    expect(drawer.classList.contains("hidden")).toBe(true);

    instructionsBtn.click();
    expect(drawer.classList.contains("hidden")).toBe(false);
    expect(instructionsBtn.getAttribute("aria-expanded")).toBe("true");

    instructionsBtn.click();
    expect(drawer.classList.contains("hidden")).toBe(true);
    expect(instructionsBtn.getAttribute("aria-expanded")).toBe("false");

    const dismissBtn = getTestElement<HTMLButtonElement>("banner-dismiss-btn");
    ui.showBanner("info", "Test", "Desc");
    dismissBtn.click();
    expect(
      document
        .getElementById("system-status-banner")
        ?.classList.contains("hidden"),
    ).toBe(true);

    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () =>
        Promise.resolve({
          status: "healthy",
          database: { status_code: "healthy", has_initial_data: true },
          ollama: { accessible: true },
        }),
    });
    const chip = getTestElement<HTMLButtonElement>("system-status-chip");
    chip.click();
    await flushPromises();

    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () =>
        Promise.resolve({
          status: "healthy",
          database: { status_code: "healthy", has_initial_data: true },
          ollama: { accessible: true },
        }),
    });
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve({ models: [] }),
    });
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve({ tables: [] }),
    });
    const retryBtn = getTestElement<HTMLButtonElement>("banner-retry-btn");
    retryBtn.click();
    await flushPromises();
  });

  it("handles schema explorer diagnostic cards with copy and reload buttons", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () =>
        Promise.resolve({
          tables: [],
          database: {
            status_code: "missing_file",
            configured_path: "custom.duckdb",
          },
        }),
    });

    await ui["loadSchema"]();
    const schemaContent = getTestElement("schema-content");
    expect(schemaContent.innerHTML).toContain("schema-diagnostic-card");

    const copyBtn = queryTestElement<HTMLButtonElement>(
      ".copy-cmd-btn",
      schemaContent,
    );
    expect(copyBtn).not.toBeNull();

    const writeTextSpy = vi.fn().mockResolvedValue(undefined);
    Object.assign(navigator, {
      clipboard: {
        writeText: writeTextSpy,
      },
    });

    vi.useFakeTimers();
    copyBtn.click();
    await vi.runAllTimersAsync();
    vi.useRealTimers();
    expect(writeTextSpy).toHaveBeenCalledWith(
      "t1d-analytics load --db custom.duckdb",
    );

    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () =>
        Promise.resolve({
          tables: [],
          database: {
            status_code: "empty_db",
          },
        }),
    });
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () =>
        Promise.resolve({
          status: "degraded",
          database: { status_code: "empty_db" },
          ollama: { accessible: true },
        }),
    });
    const reloadBtn = queryTestElement<HTMLButtonElement>(
      ".reload-schema-btn",
      schemaContent,
    );
    reloadBtn.click();
    await flushPromises();
    expect(schemaContent.innerHTML).toContain("schema-diagnostic-card");
  });

  it("renders empty-state diagnostic notice when database lacks initial data", () => {
    state.createChat();
    state.setSystemStatus({ dbStatusCode: "empty_db", hasInitialData: false });
    ui["renderActiveChat"]();
    const notice = document.querySelector(".schema-diagnostic-card");
    expect(notice).not.toBeNull();
  });

  it("handles missing banner DOM elements safely", () => {
    const originalBanner = ui["systemStatusBanner"];
    const originalChip = ui["systemStatusChip"];
    ui["systemStatusBanner"] = null;
    ui["systemStatusChip"] = null;

    ui.showBanner("warning", "Test", "Desc");
    ui.hideBanner();
    ui.updateStatusChip("healthy", "Test");

    ui["systemStatusBanner"] = originalBanner;
    ui["systemStatusChip"] = originalChip;
  });

  it("handles clipboard failure gracefully in copy-cmd-btn", async () => {
    const btn = document.createElement("button");
    btn.className = "copy-cmd-btn";
    btn.dataset.cmd = "test cmd";
    btn.textContent = "Copy";

    const writeSpy = vi.fn().mockRejectedValue(new Error("denied"));
    Object.assign(navigator, {
      clipboard: {
        writeText: writeSpy,
      },
    });

    await ui.copyCommandToClipboard("test cmd", btn);
    expect(writeSpy).toHaveBeenCalledWith("test cmd");
    expect(btn.textContent).toBe("Copy");

    const origClipboard = navigator.clipboard;
    Object.assign(navigator, { clipboard: null });
    await ui.copyCommandToClipboard("test cmd", btn);
    Object.assign(navigator, { clipboard: origClipboard });
  });

  it("loadSchema updates status when database has missing_initial_data and healthy states", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () =>
        Promise.resolve({
          tables: [],
          database: {
            status_code: "missing_initial_data",
            table_count: 1,
            has_initial_data: false,
            exists: true,
            connected: true,
          },
        }),
    });
    await ui["loadSchema"]();
    expect(state.systemStatus.dbStatusCode).toBe("missing_initial_data");

    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () =>
        Promise.resolve({
          tables: [{ name: "patients", columns: [] }],
          database: {
            status_code: "healthy",
            table_count: 1,
            has_initial_data: true,
            exists: true,
            connected: true,
          },
        }),
    });
    await ui["loadSchema"]();
    expect(state.systemStatus.dbStatusCode).toBe("healthy");
    expect(
      document
        .getElementById("system-status-banner")
        ?.classList.contains("hidden"),
    ).toBe(true);
  });

  it("shows banner safely when inner elements are null and handles null statusChipText/bannerRetryBtn", async () => {
    const origIcon = ui["bannerIcon"];
    const origTitle = ui["bannerTitle"];
    const origDesc = ui["bannerDesc"];
    const origBtn = ui["bannerRetryBtn"];
    ui["bannerIcon"] = null;
    ui["bannerTitle"] = null;
    ui["bannerDesc"] = null;
    ui["bannerRetryBtn"] = null;

    ui.showBanner("info", "Title", "Desc", "ℹ️", false);

    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () =>
        Promise.resolve({
          status: "healthy",
          database: { status_code: "healthy", has_initial_data: true },
          ollama: { accessible: true },
        }),
    });
    await ui.checkSystemStatus(true);

    ui["bannerIcon"] = origIcon;
    ui["bannerTitle"] = origTitle;
    ui["bannerDesc"] = origDesc;
    ui["bannerRetryBtn"] = origBtn;

    const origChipText = ui["statusChipText"];
    ui["statusChipText"] = null;
    ui.updateStatusChip("healthy", "Operational");
    ui["statusChipText"] = origChipText;
  });

  it("disables chat input and updates placeholder when backend is offline", () => {
    state.createChat();
    state.setSystemStatus({ backendOnline: false });
    ui["renderActiveChat"]();
    expect(ui["chatInput"].disabled).toBe(true);
    expect(ui["chatInputWrapper"].classList.contains("disabled")).toBe(true);
    expect(ui["sendBtn"].disabled).toBe(true);
  });

  it("initializes without optional status elements safely and ignores missing cmd attribute", () => {
    document.getElementById("system-status-chip")?.remove();
    document.getElementById("system-status-banner")?.remove();
    document.getElementById("provider-settings-btn")?.remove();
    document.getElementById("close-provider-modal-btn")?.remove();
    document.getElementById("provider-save-btn")?.remove();
    document.getElementById("provider-clear-btn")?.remove();
    document.getElementById("provider-modal")?.remove();
    const ui2 = new ChatUI(state);
    expect(ui2).toBeDefined();

    const container = document.createElement("div");
    const btn = document.createElement("button");
    btn.className = "copy-cmd-btn";
    container.appendChild(btn);
    ui2["bindSchemaDiagnosticButtons"](container);
    btn.click();
    ui2.destroy();
  });

  it("handles loadSchema when database fields are null or empty and validates SystemStatusController", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () =>
        Promise.resolve({
          tables: [],
          database: {
            status_code: "",
            exists: null,
            connected: null,
            message: null,
            remediation: null,
            table_count: null,
            has_initial_data: null,
          },
        }),
    });
    await ui["loadSchema"]();
    expect(state.systemStatus.dbStatusCode).toBe("unknown");

    expect(isDbStatusCode("healthy")).toBe(true);
    expect(isDbStatusCode("")).toBe(false);
    expect(isDbStatusCode(undefined)).toBe(false);
    expect(ui.systemStatusController).toBeInstanceOf(SystemStatusController);

    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        status: "healthy",
        database: { status_code: "", exists: true, connected: true },
        ollama: { accessible: true },
      }),
    });
    await ui.systemStatusController.check();
    expect(state.systemStatus.dbStatusCode).toBe("unknown");
  });

  it("manages status polling lifecycle and dispose/destroy cleanup", async () => {
    vi.useFakeTimers();
    const statusSpy = vi
      .spyOn(ui, "checkSystemStatus")
      .mockResolvedValue(undefined);
    ui.startStatusPolling();
    vi.advanceTimersByTime(30000);
    expect(statusSpy).toHaveBeenCalledTimes(1);
    ui.stopStatusPolling();
    vi.advanceTimersByTime(30000);
    expect(statusSpy).toHaveBeenCalledTimes(1);
    statusSpy.mockRestore();
    vi.useRealTimers();

    ui.openCohortModal();
    await ui.openProviderModal();
    ui.destroy();
    ui.dispose();
  });
});
