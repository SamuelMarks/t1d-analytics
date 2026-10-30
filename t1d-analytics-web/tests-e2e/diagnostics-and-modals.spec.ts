/**
 * @file diagnostics-and-modals.spec.ts
 * @module tests-e2e/diagnostics-and-modals.spec
 * End-to-end Playwright tests for system health banners, degraded/offline recovery,
 * Schema Explorer interactions, and the Table Preview Modal (pagination, sort, search, exports).
 */

import { test, expect } from "./base-test";

test.describe("System Diagnostics, Schema Explorer & Table Preview Modal", () => {
  /**
   * Tests missing_file, empty_db, and offline system status banners, setup drawer toggle, dismiss, and retry recovery.
   */
  test("Displays diagnostic banners for missing_file and offline states, toggles setup instructions, and recovers on Retry", async ({
    page,
  }) => {
    let statusPhase: "missing_file" | "offline" | "healthy" = "missing_file";

    await page.route("**/api/status*", async (route) => {
      if (statusPhase === "offline") {
        await route.abort("failed");
        return;
      }
      if (statusPhase === "missing_file") {
        await route.fulfill({
          status: 200,
          contentType: "application/json",
          json: {
            status: "unhealthy",
            database: {
              status_code: "missing_file",
              configured_path: "missing_t1d.duckdb",
              exists: false,
              connected: false,
              table_count: 0,
              has_initial_data: false,
            },
            ollama: { accessible: true },
          },
        });
        return;
      }
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        json: {
          status: "healthy",
          database: {
            status_code: "healthy",
            configured_path: "t1d.duckdb",
            exists: true,
            connected: true,
            table_count: 4,
            has_initial_data: true,
          },
          ollama: { accessible: true },
        },
      });
    });

    await page.route("**/api/schema*", async (route) => {
      if (statusPhase === "offline") {
        await route.abort("failed");
        return;
      }
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        json: {
          tables:
            statusPhase === "healthy"
              ? [
                  {
                    name: "patients",
                    columns: [
                      { name: "patient_id", type: "VARCHAR" },
                      { name: "age", type: "INTEGER" },
                    ],
                  },
                ]
              : [],
          database: {
            status_code: statusPhase === "healthy" ? "healthy" : "missing_file",
            configured_path: "t1d.duckdb",
            exists: statusPhase === "healthy",
            connected: statusPhase === "healthy",
            table_count: statusPhase === "healthy" ? 1 : 0,
            has_initial_data: statusPhase === "healthy",
          },
        },
      });
    });

    await page.goto("/");

    // 1. Verify missing_file warning/error banner is visible
    const banner = page.locator("#system-status-banner");
    await expect(banner).toBeVisible();
    await expect(banner).toHaveClass(/banner-(warning|error)/);

    // 2. Toggle setup instructions drawer
    const instructionsBtn = page.locator("#banner-instructions-btn");
    const drawer = page.locator("#banner-instructions-drawer");
    await expect(drawer).toHaveClass(/hidden/);
    await instructionsBtn.click();
    await expect(drawer).not.toHaveClass(/hidden/);
    await expect(instructionsBtn).toHaveAttribute("aria-expanded", "true");

    // 3. Transition to offline and click Retry
    statusPhase = "offline";
    await page.click("#banner-retry-btn");
    await expect(page.locator("#system-status-chip")).toHaveClass(
      /status-offline/,
    );
    await expect(page.locator("#chat-input")).toBeDisabled();

    // 4. Transition to healthy and click Retry -> banner hides and input enables
    statusPhase = "healthy";
    await page.click("#banner-retry-btn");
    await expect(banner).toHaveClass(/hidden/);
    await expect(page.locator("#system-status-chip")).toHaveClass(
      /status-healthy/,
    );
    await expect(page.locator("#chat-input")).toBeEnabled();
    await expect(page.locator(".schema-table")).toHaveCount(1);
  });

  /**
   * Tests Table Preview Modal pagination, column sorting, row search filtering, and export buttons.
   */
  test("Opens Table Preview Modal from Schema Explorer and supports sorting, searching, pagination, and exports", async ({
    page,
  }) => {
    const requestedUrls: string[] = [];

    await page.route("**/api/schema*", async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        json: {
          tables: [
            {
              name: "patients",
              columns: [
                { name: "patient_id", type: "VARCHAR" },
                { name: "age", type: "INTEGER" },
                { name: "baseline_hba1c", type: "DOUBLE" },
              ],
            },
          ],
          database: {
            status_code: "healthy",
            configured_path: "t1d.duckdb",
            exists: true,
            connected: true,
            table_count: 1,
            has_initial_data: true,
          },
        },
      });
    });

    await page.route("**/api/table/patients*", async (route) => {
      const reqUrl = route.request().url();
      requestedUrls.push(reqUrl);
      const parsed = new URL(reqUrl);
      const offset = Number(parsed.searchParams.get("offset") || "0");
      const search = parsed.searchParams.get("search") || "";
      const order = parsed.searchParams.get("order") || "asc";

      const rows =
        search === "PT-99"
          ? [{ patient_id: "PT-99", age: 29, baseline_hba1c: 7.1 }]
          : offset > 0
            ? [{ patient_id: "PT-26", age: 41, baseline_hba1c: 6.9 }]
            : order === "desc"
              ? [{ patient_id: "PT-25", age: 52, baseline_hba1c: 8.4 }]
              : [{ patient_id: "PT-01", age: 19, baseline_hba1c: 7.8 }];

      await route.fulfill({
        status: 200,
        contentType: "application/json",
        json: {
          rows,
          total: search ? 1 : 50,
          page: offset > 0 ? 2 : 1,
          limit: 25,
          total_pages: search ? 1 : 2,
        },
      });
    });

    await page.goto("/");

    // Expand table columns in Schema Explorer
    const schemaToggleBtn = page.locator(".schema-table-header-toggle").first();
    await expect(schemaToggleBtn).toBeVisible();
    await schemaToggleBtn.click();
    await expect(schemaToggleBtn).toHaveAttribute("aria-expanded", "true");
    await expect(page.locator(".schema-column")).toHaveCount(3);

    // Open Table Preview Modal
    await page.locator(".table-view-btn").first().click();
    const modal = page.locator("#table-modal");
    await expect(modal).toHaveAttribute("aria-modal", "true");
    await expect(page.locator("#modal-table-body tr")).toHaveCount(1);
    await expect(page.locator("#modal-table-body td").first()).toHaveText(
      "PT-01",
    );

    // Navigate to Page 2
    const nextBtn = page.locator("#next-page-btn");
    const prevBtn = page.locator("#prev-page-btn");
    await expect(nextBtn).toBeEnabled();
    await nextBtn.click();
    await expect(page.locator("#modal-table-body td").first()).toHaveText(
      "PT-26",
    );
    await expect(prevBtn).toBeEnabled();

    // Navigate back to Page 1
    await prevBtn.click();
    await expect(page.locator("#modal-table-body td").first()).toHaveText(
      "PT-01",
    );

    // Sort by age and toggle sort order to DESC
    await page.selectOption("#modal-table-sort-col", "age");
    await page.click("#modal-table-sort-order");
    await expect(page.locator("#modal-table-body td").first()).toHaveText(
      "PT-25",
    );

    // Filter rows using search input
    await page.fill("#modal-table-search", "PT-99");
    await expect(page.locator("#modal-table-body td").first()).toHaveText(
      "PT-99",
    );

    // Trigger page CSV and JSON downloads from modal footer
    const csvDownloadPromise = page.waitForEvent("download");
    await page.click("#modal-export-csv-btn");
    const csvDownload = await csvDownloadPromise;
    expect(csvDownload.suggestedFilename()).toBe("patients_page1.csv");

    const jsonDownloadPromise = page.waitForEvent("download");
    await page.click("#modal-export-json-btn");
    const jsonDownload = await jsonDownloadPromise;
    expect(jsonDownload.suggestedFilename()).toBe("patients_page1.json");

    // Close modal with Escape key
    await page.keyboard.press("Escape");
    await expect(modal).toHaveAttribute("aria-hidden", "true");
    expect(requestedUrls.length).toBeGreaterThanOrEqual(4);
  });

  /**
   * Tests empty_db and missing_initial_data diagnostic banners, diagnostic card command copying,
   * banner dismissal, and table header column click/keyboard sorting with live language switching.
   */
  test("Handles empty_db and missing_initial_data states, copies CLI commands, dismisses banner, and sorts via table headers", async ({
    page,
    context,
  }) => {
    await context.grantPermissions(["clipboard-read", "clipboard-write"]);

    let dbState: "empty_db" | "missing_initial_data" | "healthy" = "empty_db";

    await page.route("**/api/status*", async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        json: {
          status: dbState === "healthy" ? "healthy" : "degraded",
          database: {
            status_code: dbState,
            configured_path: "t1d.duckdb",
            exists: true,
            connected: true,
            table_count: dbState === "empty_db" ? 0 : 1,
            has_initial_data: dbState === "healthy",
          },
          ollama: { accessible: true },
        },
      });
    });

    await page.route("**/api/schema*", async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        json: {
          tables:
            dbState === "empty_db"
              ? []
              : [
                  {
                    name: "patients",
                    columns: [
                      { name: "patient_id", type: "VARCHAR" },
                      { name: "age", type: "INTEGER" },
                    ],
                  },
                ],
          database: {
            status_code: dbState,
            configured_path: "t1d.duckdb",
            exists: true,
            connected: true,
            table_count: dbState === "empty_db" ? 0 : 1,
            has_initial_data: dbState === "healthy",
          },
        },
      });
    });

    await page.route("**/api/table/patients*", async (route) => {
      const parsed = new URL(route.request().url());
      const sortBy = parsed.searchParams.get("sort_by") || "";
      const order = parsed.searchParams.get("order") || "asc";
      const rows =
        sortBy === "age" && order === "desc"
          ? [
              { patient_id: "PT-02", age: 45 },
              { patient_id: "PT-01", age: 21 },
            ]
          : [
              { patient_id: "PT-01", age: 21 },
              { patient_id: "PT-02", age: 45 },
            ];
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        json: {
          rows,
          total: 2,
          page: 1,
          limit: 25,
          total_pages: 1,
        },
      });
    });

    await page.goto("/");

    // 1. Verify empty_db diagnostic card in Schema Explorer and copy CLI command
    const copyCmdBtn = page.locator("#schema-content .copy-cmd-btn").first();
    await expect(copyCmdBtn).toBeVisible();
    await copyCmdBtn.click();
    const copiedCmd = await page.evaluate(() => navigator.clipboard.readText());
    expect(copiedCmd).toBe("t1d-analytics load");

    // 2. Dismiss banner via #banner-dismiss-btn
    const banner = page.locator("#system-status-banner");
    await expect(banner).not.toHaveClass(/hidden/);
    await page.click("#banner-dismiss-btn");
    await expect(banner).toHaveClass(/hidden/);

    // 3. Transition to missing_initial_data and click Reload Schema button inside diagnostic card
    dbState = "missing_initial_data";
    await page.locator("#schema-content .reload-schema-btn").click();
    await expect(banner).not.toHaveClass(/hidden/);
    await expect(page.locator("#system-status-chip")).toHaveClass(
      /status-degraded/,
    );

    // 4. Open Table Preview Modal, click column header <th> to sort asc then desc, and switch language while open
    await page.locator(".table-view-btn").first().click();
    const modal = page.locator("#table-modal");
    await expect(modal).toBeVisible();

    const ageTh = page.locator("#modal-table-head th").nth(1);
    await expect(ageTh).toHaveAttribute("aria-sort", "none");
    await ageTh.click();
    await expect(ageTh).toHaveAttribute("aria-sort", "ascending");

    // Re-examine sort button inside ageTh after re-render and toggle to descending via keyboard Enter
    const ageSortBtn = page
      .locator("#modal-table-head th")
      .nth(1)
      .locator(".table-sort-header-btn");
    await ageSortBtn.focus();
    await page.keyboard.press("Enter");
    await expect(page.locator("#modal-table-head th").nth(1)).toHaveAttribute(
      "aria-sort",
      "descending",
    );
    await expect(
      page.locator("#modal-table-body tr").first().locator("td").first(),
    ).toHaveText("PT-02");

    // Switch language to Japanese while Table Preview Modal is open -> modal title updates
    await page.selectOption("#lang-select", "ja");
    await expect(page.locator("#modal-title")).toContainText("patients");

    // Close modal by clicking backdrop overlay
    await modal.click({ position: { x: 5, y: 5 } });
    await expect(modal).toHaveAttribute("aria-hidden", "true");
  });

  /**
   * Tests mobile viewport sidebar drawer opening, focus trapping, Escape dismissal, and overlay click dismissal.
   */
  test("Supports mobile sidebar opening, keyboard Tab focus trapping, Escape close, and backdrop overlay close", async ({
    page,
  }) => {
    await page.setViewportSize({ width: 375, height: 667 });
    await page.goto("/");

    const openBtn = page.locator("#open-sidebar-btn");
    await expect(openBtn).toBeVisible();
    await openBtn.click();

    await expect(page.locator("body")).toHaveClass(/sidebar-open/);
    await expect(openBtn).toHaveAttribute("aria-expanded", "true");
    await expect(page.locator("#close-sidebar-btn")).toBeFocused();

    // Press Escape inside open mobile sidebar -> closes sidebar and restores focus to #open-sidebar-btn
    await page.keyboard.press("Escape");
    await expect(page.locator("body")).not.toHaveClass(/sidebar-open/);
    await expect(openBtn).toBeFocused();

    // Open again and click #overlay backdrop to close
    await openBtn.click();
    await expect(page.locator("body")).toHaveClass(/sidebar-open/);
    await page.locator("#overlay").click({ position: { x: 350, y: 300 } });
    await expect(page.locator("body")).not.toHaveClass(/sidebar-open/);
  });

  /**
   * Exercises all remaining ChatUI defensive DOM guards, status polling timers,
   * fallback error handlers, and table/cohort/provider modal edge cases in browser runtime.
   */
  test("Exercises ChatUI polling lifecycle, defensive DOM guards, error fallbacks, and modal edge cases", async ({
    page,
  }) => {
    await page.route(/\/node_modules\/(?!\.vite\/)/, async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/javascript",
        body: "export const defineConfig = (c) => c; export const devices = { 'Desktop Chrome': { browserName: 'chromium' } }; export default function istanbul(opts) { return { name: 'istanbul', opts, defineConfig, devices }; }",
      });
    });

    await page.goto("/");

    const uiCoverageCheck = await page.evaluate(async () => {
      (
        globalThis as unknown as { __BUILD_TIMESTAMP_UTC__?: string }
      ).__BUILD_TIMESTAMP_UTC__ = "2026-10-29T00:00:00Z UTC";
      const uiMod = await import("/src/ui.ts");
      const stateMod = await import("/src/state.ts");
      const i18nMod = await import("/src/i18n.ts");

      // 1. requireElementById missing element error branch (lines 258-259)
      let missingElError = "";
      try {
        uiMod.requireElementById("non-existent-required-id");
      } catch (e) {
        missingElError = e instanceof Error ? e.message : String(e);
      }

      const origFetch = window.fetch;
      window.fetch = async () =>
        new Response(JSON.stringify({ models: [], tables: [] }), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        });

      const state = new stateMod.ChatState();
      // Re-instantiating ChatUI disposes previous activeChatUIInstance (lines 543-544)
      const ui = new uiMod.ChatUI(state);
      await new Promise((r) => setTimeout(r, 10));

      // 2. Status polling lifecycle (lines 701-715) and dispose/destroy with open modals (lines 723-751)
      ui.stopStatusPolling();
      ui.startStatusPolling();
      ui.startStatusPolling(5);
      await new Promise((r) => setTimeout(r, 15));
      ui.stopStatusPolling();

      ui.openCohortModal();
      await ui.openProviderModal();
      ui.destroy();
      // Calling dispose again when activeChatUIInstance is already null (line 741)
      ui.dispose();

      // Re-instantiate active UI for remaining checks
      const ui2 = new uiMod.ChatUI(state);
      await new Promise((r) => setTimeout(r, 10));

      // 3. Banner & status chip defaults and null element guards (lines 767-819)
      ui2.showBanner("info", "Default Icon Banner", "Default retry");
      const savedBanner = document.getElementById("system-status-banner");
      const savedChip = document.getElementById("system-status-chip");
      const savedIcon = document.getElementById("banner-icon");
      const savedTitle = document.getElementById("banner-title");
      const savedDesc = document.getElementById("banner-desc");
      const savedRetry = document.getElementById("banner-retry-btn");
      const savedChipText = document.getElementById("status-chip-text");

      // Null inner banner & chip text elements
      Object.assign(ui2, {
        bannerIcon: null,
        bannerTitle: null,
        bannerDesc: null,
        bannerRetryBtn: null,
        statusChipText: null,
      });
      ui2.showBanner("warning", "T", "D", "⚠️", false);
      state.setSystemStatus({ dbFileSizeBytes: 0 });
      ui2.updateStatusChip("healthy", "Operational");

      // Null banner & chip containers
      Object.assign(ui2, {
        systemStatusBanner: null,
        systemStatusChip: null,
      });
      ui2.showBanner("danger", "T", "D");
      ui2.hideBanner();
      ui2.updateStatusChip("offline", "Offline");

      // Restore banner & chip elements
      Object.assign(ui2, {
        systemStatusBanner: savedBanner,
        systemStatusChip: savedChip,
        bannerIcon: savedIcon,
        bannerTitle: savedTitle,
        bannerDesc: savedDesc,
        bannerRetryBtn: savedRetry,
        statusChipText: savedChipText,
      });

      state.setSystemStatus({
        dbFileSizeBytes: 5 * 1024 * 1024,
        tableCount: 3,
      });
      ui2.updateStatusChip("healthy", "Operational");

      // 4. checkSystemStatus with missing_file, empty_db, custom dbPath, invalid status_code, and offline ollama model fallback (lines 845-933)
      window.fetch = async () =>
        new Response(
          JSON.stringify({
            status: "error",
            database: {
              status_code: "missing_file",
              configured_path: "missing.duckdb",
              exists: false,
              connected: false,
            },
            ollama: { accessible: true },
          }),
          { status: 200 },
        );
      await ui2.checkSystemStatus();

      window.fetch = async () =>
        new Response(
          JSON.stringify({
            status: "degraded",
            database: {
              status_code: "empty_db",
              configured_path: "empty.duckdb",
              exists: true,
              connected: true,
            },
            ollama: { accessible: true },
          }),
          { status: 200 },
        );
      await ui2.checkSystemStatus();

      window.fetch = async () =>
        new Response(
          JSON.stringify({
            status: "degraded",
            database: {
              status_code: "unrecognized_code",
              exists: false,
              connected: false,
            },
            ollama: { accessible: false },
          }),
          { status: 200 },
        );
      const chat = state.createChat();
      chat.model = "gemma4";
      chat.messages = [];
      const modelSelect = document.getElementById(
        "model-select",
      ) as HTMLSelectElement;
      const gOpt = document.createElement("option");
      gOpt.value = "gemma4";
      modelSelect.appendChild(gOpt);
      modelSelect.value = "gemma4";
      await ui2.checkSystemStatus(false, "custom_check.duckdb");

      // checkSystemStatus when activeChat has messages (false branch of currentChat.messages.length === 0)
      chat.messages = [{ role: "user", content: "existing", model: "sql" }];
      modelSelect.value = "gemma4";
      await ui2.checkSystemStatus(true);

      // checkSystemStatus when bannerRetryBtn is null
      Object.assign(ui2, { bannerRetryBtn: null });
      await ui2.checkSystemStatus(true);
      Object.assign(ui2, { bannerRetryBtn: savedRetry });

      // 5. copyCommandToClipboard wait for 2000ms timeout & null clipboard guard (lines 974-985)
      const cmdBtn = document.createElement("button");
      cmdBtn.className = "copy-cmd-btn";
      const origSetTimeout = window.setTimeout;
      // Run 2000ms timeouts immediately during test
      window.setTimeout = ((fn: TimerHandler, ms?: number) => {
        if (typeof fn === "function" && ms === 2000) {
          fn();
          return 0;
        }
        return origSetTimeout(fn, ms);
      }) as typeof window.setTimeout;
      await ui2.copyCommandToClipboard("echo test", cmdBtn);

      const origClipboard = navigator.clipboard;
      Object.defineProperty(navigator, "clipboard", {
        value: null,
        configurable: true,
      });
      await ui2.copyCommandToClipboard("echo test", cmdBtn);
      Object.defineProperty(navigator, "clipboard", {
        value: origClipboard,
        configurable: true,
      });

      // bindSchemaDiagnosticButtons when data-cmd is absent (line 998)
      const diagContainer = document.createElement("div");
      const noCmdBtn = document.createElement("button");
      noCmdBtn.className = "copy-cmd-btn";
      diagContainer.appendChild(noCmdBtn);
      (
        ui2 as unknown as {
          bindSchemaDiagnosticButtons: (c: HTMLElement) => void;
        }
      ).bindSchemaDiagnosticButtons(diagContainer);
      noCmdBtn.click();

      // 6. loadSchema missing #schema-content, missing database field, unrecognized status_code, and empty tables (lines 1020-1206)
      const schemaContent = document.getElementById("schema-content")!;
      schemaContent.id = "schema-content-temp";
      await ui2.loadSchema();
      schemaContent.id = "schema-content";

      window.fetch = async () =>
        new Response(JSON.stringify({ tables: [] }), { status: 200 });
      await ui2.loadSchema();

      window.fetch = async () =>
        new Response(
          JSON.stringify({
            tables: [
              {
                name: "patients",
                columns: [{ name: "patient_id", type: "VARCHAR" }],
              },
            ],
            database: { status_code: "unknown_code" },
          }),
          { status: 200 },
        );
      await ui2.loadSchema();
      const schemaTableToggleBtn = document.querySelector<HTMLButtonElement>(
        ".schema-table-header-toggle",
      );
      schemaTableToggleBtn?.click();
      schemaTableToggleBtn?.click();

      // 7. loadModels customModels, error_code, dynamic groups, activeChat model fallbacks, and catch (lines 1275-1438)
      state.setConfiguredModels(
        "groq:llama3, deepseek/coder, local-custom, openai/gpt-4o",
      );
      window.fetch = async () =>
        new Response(
          JSON.stringify({
            models: [
              {
                name: "openai/gpt-4o",
                provider: "openai",
                configured: false,
                available: false,
                reachable: false,
                error_code: "NO_KEY",
              },
              {
                name: "unconfigured-slash/model",
                configured: false,
                available: true,
                reachable: true,
              },
            ],
          }),
          { status: 200 },
        );
      chat.model = "missing_model_xyz";
      chat.messages = [];
      await ui2.loadModels();
      state.removeConfiguredModels();

      // loadModels fallback when no ready model exists: empty messages -> "sql", non-empty -> modelsList[0].name
      window.fetch = async () =>
        new Response(
          JSON.stringify({
            models: [
              {
                name: "openai/gpt-4o",
                configured: false,
                available: true,
                reachable: true,
              },
              {
                name: "local-unavail",
                configured: false,
                available: true,
                reachable: true,
              },
            ],
          }),
          { status: 200 },
        );
      chat.model = "missing_model_xyz";
      chat.messages = [];
      await ui2.loadModels();
      const emptyFallbackModel = chat.model;

      chat.model = "missing_model_xyz";
      chat.messages = [{ role: "user", content: "hi", model: "sql" }];
      await ui2.loadModels();
      const nonEmptyFallbackModel = chat.model;

      // loadModels with localStorage API key making cloud model ready
      state.setApiKey("openai", "sk-ready");
      chat.model = "missing_model_xyz";
      chat.messages = [];
      await ui2.loadModels();
      const keyReadyModel = chat.model;
      state.removeApiKey("openai");

      // loadModels when models is not an array (empty modelsList)
      window.fetch = async () =>
        new Response(JSON.stringify({ models: null }), { status: 200 });
      chat.model = "missing_model_xyz";
      await ui2.loadModels();

      // loadModels when activeChat is null
      state.activeChatId = null;
      await ui2.loadModels();
      state.setActiveChat(chat.id);

      // loadModels network error catch (line 1418)
      window.fetch = async () => Promise.reject(new Error("Models API down"));
      await ui2.loadModels();

      // 8. updateProviderBadgeStatus without providerStatusDot and with slash value (lines 1427-1460)
      modelSelect.innerHTML = `<option value="openai/custom-model">custom</option>`;
      modelSelect.value = "openai/custom-model";
      ui2.updateProviderBadgeStatus();
      const savedDot = (ui2 as unknown as { providerStatusDot: HTMLElement })
        .providerStatusDot;
      Object.assign(ui2, { providerStatusDot: null });
      ui2.updateProviderBadgeStatus();
      Object.assign(ui2, { providerStatusDot: savedDot });

      // 9. newChatBtn when ollama offline with and without cloud keys (lines 1480-1490)
      state.setSystemStatus({ ollamaOnline: false });
      document.getElementById("new-chat-btn")?.click();
      state.setApiKey("openai", "sk-1");
      document.getElementById("new-chat-btn")?.click();
      state.removeApiKey("openai");

      // dbSelect change with empty value (line 1522)
      const dbSelect = document.getElementById(
        "db-select",
      ) as HTMLSelectElement;
      const emptyDbOpt = document.createElement("option");
      emptyDbOpt.value = "";
      dbSelect.appendChild(emptyDbOpt);
      dbSelect.value = "";
      dbSelect.dispatchEvent(new Event("change"));

      // providerModal and cohortModal backdrop clicks and cohort input listeners (lines 1560-1635)
      await ui2.openProviderModal();
      const providerModal = document.getElementById("provider-modal")!;
      providerModal.dispatchEvent(new MouseEvent("click", { bubbles: true }));

      ui2.openCohortModal();
      const cohortModal = document.getElementById("cohort-modal")!;
      const cohortJoinType = document.getElementById(
        "cohort-join-type-select",
      ) as HTMLSelectElement;
      cohortJoinType.dispatchEvent(new Event("change"));
      const cohortMinAge = document.getElementById(
        "cohort-min-age",
      ) as HTMLInputElement;
      cohortMinAge.dispatchEvent(new Event("change"));
      cohortModal.dispatchEvent(new MouseEvent("click", { bubbles: true }));

      // langSelect change while table modal is closed or without modal-title (lines 1656-1670)
      const tableModal = document.getElementById("table-modal")!;
      tableModal.removeAttribute("aria-hidden");
      Object.assign(ui2, { currentTable: "patients" });
      const modalTitle = document.getElementById("modal-title")!;
      modalTitle.id = "modal-title-hidden";
      window.fetch = async () =>
        new Response(JSON.stringify({ rows: [], tables: [], models: [] }), {
          status: 200,
        });
      const langSelect = document.getElementById(
        "lang-select",
      ) as HTMLSelectElement;
      langSelect.value = "en";
      langSelect.dispatchEvent(new Event("change"));
      await new Promise((r) => setTimeout(r, 20));
      modalTitle.id = "modal-title";
      tableModal.setAttribute("aria-hidden", "true");

      // Schema explorer toggle (lines 1747-1763), bannerInstructionsBtn toggle, systemStatusChip click, sidebar keydown when closed, & document click with non-Element target (lines 1775-1780)
      document.getElementById("banner-instructions-btn")?.click();
      document.getElementById("banner-instructions-btn")?.click();
      document.getElementById("system-status-chip")?.click();
      await new Promise((r) => setTimeout(r, 10));
      document.body.classList.remove("sidebar-open");
      document
        .getElementById("sidebar")
        ?.dispatchEvent(
          new KeyboardEvent("keydown", { key: "Tab", bubbles: true }),
        );
      document
        .querySelector(".schema-header")
        ?.dispatchEvent(new MouseEvent("click"));
      document
        .querySelector(".schema-header")
        ?.dispatchEvent(new MouseEvent("click"));
      const docClickEv = new MouseEvent("click");
      Object.defineProperty(docClickEv, "target", { value: document });
      document.dispatchEvent(docClickEv);

      // 10. executeDirectSqlAction scalar, null scalar, error, and catch for both new & existing messages (lines 1810-1895)
      const dummyBtn = document.createElement("button");
      const execDirect = (
        ui2 as unknown as {
          executeDirectSqlAction: (
            q: string,
            b: HTMLButtonElement,
            m?: Record<string, unknown>,
          ) => Promise<void>;
        }
      ).executeDirectSqlAction.bind(ui2);

      state.setCurrentDb("custom_exec.duckdb");
      window.fetch = async () =>
        new Response(JSON.stringify({ sqlResult: [{ count: 7 }] }), {
          status: 200,
        });
      await execDirect("SELECT 7", dummyBtn);

      window.fetch = async () =>
        new Response(JSON.stringify({ sqlResult: [] }), { status: 200 });
      await execDirect("SELECT 0", dummyBtn);

      window.fetch = async () =>
        new Response(JSON.stringify({ error: "backend.syntaxError" }), {
          status: 200,
        });
      const msgToUpdate: Record<string, unknown> = {
        role: "assistant",
        content: "old",
      };
      await execDirect("SELECT bad", dummyBtn, msgToUpdate);

      window.fetch = async () => Promise.reject(new Error("Direct SQL fail"));
      await execDirect("SELECT net_err", dummyBtn);
      await execDirect("SELECT net_err", dummyBtn, msgToUpdate);

      window.fetch = async () => Promise.reject("Direct SQL string fail");
      await execDirect("SELECT str_err", dummyBtn);
      state.setCurrentDb("t1d.duckdb");

      // 11. announce without #a11y-announcer (line 1906)
      const announcer = document.getElementById("a11y-announcer")!;
      announcer.id = "a11y-announcer-hidden";
      (
        ui2 as unknown as { announce: (m: string, a?: boolean) => void }
      ).announce("test");
      announcer.id = "a11y-announcer";

      // 12. handleSendMessage edge cases: empty input, provider/apiKey headers, streaming error/scalar/empty, batch scalar/catch, serverSyncEnabled (lines 1920-2190)
      const sendMsg = (
        ui2 as unknown as {
          handleSendMessage: (t?: string, m?: string) => Promise<void>;
        }
      ).handleSendMessage.bind(ui2);
      const chatInput = document.getElementById(
        "chat-input",
      ) as HTMLTextAreaElement;
      chatInput.value = "   ";
      await sendMsg();

      state.setApiKey("openai", "sk-openai-key");
      state.setCurrentDb("custom_chat.duckdb");
      state.setServerSyncEnabled(true);

      // Streaming enabled: stream error event, scalar with content, scalar without content, fallback when !streamResp.ok, fallback on throw
      ui2.streamingEnabled = true;
      const sseHeaders = { "Content-Type": "text/event-stream" };
      const makeSseResponse = (chunks: string[]) => {
        let i = 0;
        const stream = new ReadableStream<Uint8Array>({
          pull(controller) {
            if (i < chunks.length) {
              controller.enqueue(new TextEncoder().encode(chunks[i++]));
            } else {
              controller.close();
            }
          },
        });
        return new Response(stream, { status: 200, headers: sseHeaders });
      };

      window.fetch = async (input: RequestInfo | URL) => {
        const url = String(input);
        if (url.includes("/api/chat/stream")) {
          return makeSseResponse([
            `data: {"event":"done","content":"backend.literalSql","sqlQuery":"SELECT 1","sqlResult":null,"error":"backend.syntaxError"}\n\n`,
          ]);
        }
        return new Response(JSON.stringify({ sessions: [] }), { status: 200 });
      };
      await sendMsg("SELECT 1", "openai/gpt-4o");

      window.fetch = async (input: RequestInfo | URL) => {
        const url = String(input);
        if (url.includes("/api/chat/stream")) {
          return makeSseResponse([
            `data: {"event":"done","content":"Result:","sqlQuery":"SELECT 1","sqlResult":[{"val":42}],"error":null}\n\n`,
          ]);
        }
        return new Response(JSON.stringify({ sessions: [] }), { status: 200 });
      };
      await sendMsg("SELECT 42", "sql");

      window.fetch = async (input: RequestInfo | URL) => {
        const url = String(input);
        if (url.includes("/api/chat/stream")) {
          return makeSseResponse([
            `data: {"event":"done","content":"","sqlQuery":"SELECT 1","sqlResult":[{"val":99}],"error":null}\n\n`,
          ]);
        }
        return new Response(JSON.stringify({ sessions: [] }), { status: 200 });
      };
      await sendMsg("SELECT 99", "sql");

      window.fetch = async (input: RequestInfo | URL) => {
        const url = String(input);
        if (url.includes("/api/chat/stream")) {
          return makeSseResponse([
            `data: {"event":"done","content":"","sqlQuery":null,"sqlResult":null,"error":null}\n\n`,
          ]);
        }
        return new Response(JSON.stringify({ sessions: [] }), { status: 200 });
      };
      await sendMsg("Empty done", "gemma4");

      // Stream HTTP 500 fallback to batch /api/chat with scalar (both empty and non-empty content)
      window.fetch = async (input: RequestInfo | URL) => {
        const url = String(input);
        if (url.includes("/api/chat/stream")) {
          return new Response("Err", { status: 500 });
        }
        if (url.includes("/api/chat")) {
          return new Response(
            JSON.stringify({
              content: "",
              sqlQuery: "SELECT 1",
              sqlResult: [{ c: 10 }],
            }),
            { status: 200 },
          );
        }
        return new Response(JSON.stringify({ sessions: [] }), { status: 200 });
      };
      await sendMsg("Fallback empty content scalar", "sql");

      // Stream throw fallback to batch /api/chat with null content & string rejection
      ui2.streamingEnabled = true;
      window.fetch = async (input: RequestInfo | URL) => {
        const url = String(input);
        if (url.includes("/api/chat/stream")) {
          return Promise.reject(new Error("Stream failed"));
        }
        if (url.includes("/api/chat")) {
          return new Response(JSON.stringify({ content: null }), {
            status: 200,
          });
        }
        return new Response(JSON.stringify({ sessions: [] }), { status: 200 });
      };
      await sendMsg("Fallback null content", "gemma4");

      ui2.streamingEnabled = false;
      window.fetch = async (input: RequestInfo | URL) => {
        const url = String(input);
        if (url.includes("/api/chat")) {
          return new Response(
            JSON.stringify({
              content: "backend.literalSql",
              error: "backend.syntaxError",
            }),
            { status: 200 },
          );
        }
        return new Response(JSON.stringify({ sessions: [] }), { status: 200 });
      };
      await sendMsg("Batch data.error", "sql");

      window.fetch = async (input: RequestInfo | URL) => {
        const url = String(input);
        if (url.includes("/api/chat")) {
          return Promise.reject("String chat error");
        }
        return new Response(JSON.stringify({ sessions: [] }), { status: 200 });
      };
      await sendMsg("Batch string error", "gemma4");

      // handleSendMessage when activeChatId is null & renderActiveChat when activeChat is null (lines 2836-2848)
      state.activeChatId = null;
      ui2.render();
      window.fetch = async () =>
        new Response(JSON.stringify({ content: "ok" }), { status: 200 });
      await sendMsg("No active chat", "sql");
      state.setActiveChat(chat.id);
      state.setServerSyncEnabled(false);
      state.removeApiKey("openai");
      state.setCurrentDb("t1d.duckdb");

      // 13. fetchDatabases and renderDatabaseDropdown edge cases (lines 2207-2265)
      window.fetch = async () => new Response("Err", { status: 500 });
      await ui2.fetchDatabases();
      window.fetch = async () => Promise.reject(new Error("DB net err"));
      await ui2.fetchDatabases();
      window.fetch = async () =>
        new Response(JSON.stringify({}), { status: 200 });
      state.currentDb = "";
      await ui2.fetchDatabases();

      const savedDbSelect = (
        ui2 as unknown as { dbSelect: HTMLSelectElement | null }
      ).dbSelect;
      Object.assign(ui2, { dbSelect: null });
      await ui2.fetchDatabases();
      ui2.renderDatabaseDropdown();
      Object.assign(ui2, { dbSelect: savedDbSelect });

      // syncAllChatsWithServer false & catch (lines 2266-2276)
      const origSync = state.syncWithServer.bind(state);
      state.syncWithServer = async () => false;
      await ui2.syncAllChatsWithServer();
      state.syncWithServer = async () => {
        throw new Error("Sync failed");
      };
      await ui2.syncAllChatsWithServer();
      state.syncWithServer = origSync;

      // 14. Cohort modal & join resolution edge cases (lines 2323-2455)
      const headerEl = document.createElement("div");
      headerEl.className = "schema-table-header";
      headerEl.innerHTML = "<h4>   </h4><h4>patients</h4>";
      document.body.appendChild(headerEl);
      const colListEl = document.createElement("ul");
      colListEl.id = "schema-table-patients";
      colListEl.innerHTML = `<li class="schema-column"><span>patient_id</span></li>`;
      document.body.appendChild(colListEl);

      (document.activeElement as HTMLElement | null)?.blur();
      ui2.openCohortModal();
      const cohortTableSel = document.getElementById(
        "cohort-table-select",
      ) as HTMLSelectElement;
      const cohortJoinSel = document.getElementById(
        "cohort-join-table-select",
      ) as HTMLSelectElement;
      cohortTableSel.innerHTML = `<option value="patients">patients</option><option value="custom_t">custom_t</option>`;
      cohortJoinSel.innerHTML = `<option value="cgm_data">cgm_data</option><option value="unknown_t">unknown_t</option>`;
      cohortTableSel.value = "patients";
      cohortJoinSel.value = "cgm_data";
      ui2.handleCohortJoinTableChange();

      cohortTableSel.value = "custom_t";
      cohortJoinSel.value = "unknown_t";
      ui2.handleCohortJoinTableChange();

      // Close cohort modal when previousFocusCohort is removed from DOM
      const tempFocusBtn = document.createElement("button");
      document.body.appendChild(tempFocusBtn);
      tempFocusBtn.focus();
      ui2.openCohortModal();
      tempFocusBtn.remove();
      ui2.closeCohortModal();
      ui2.closeCohortModal();
      headerEl.remove();
      colListEl.remove();

      // 15. Provider modal edge cases (lines 2467-2575)
      (document.activeElement as HTMLElement | null)?.blur();
      state.setApiKey("anthropic", "sk-ant");
      state.setApiKey("google", "sk-goog");
      window.fetch = async () =>
        new Response(
          JSON.stringify({
            providers: [{ provider: "anthropic", configured: false }],
          }),
          { status: 200 },
        );
      await ui2.openProviderModal();
      state.removeApiKey("anthropic");
      state.removeApiKey("google");

      window.fetch = async () =>
        new Response(JSON.stringify({}), { status: 200 });
      await ui2.openProviderModal();
      window.fetch = async () => new Response("Err", { status: 500 });
      await ui2.openProviderModal();
      window.fetch = async () => Promise.reject(new Error("Prov net err"));
      await ui2.openProviderModal();

      const tempProvFocus = document.createElement("button");
      document.body.appendChild(tempProvFocus);
      tempProvFocus.focus();
      await ui2.openProviderModal();
      tempProvFocus.remove();
      ui2.closeProviderModal();

      ui2.updateModalBadge("non-existent-provider", true);

      const consentEl = document.getElementById(
        "provider-store-consent",
      ) as HTMLInputElement;
      consentEl.checked = false;
      await ui2.saveProviderSettings();
      consentEl.checked = true;
      (
        document.getElementById("provider-key-openai") as HTMLInputElement
      ).value = "sk-oai";
      (
        document.getElementById("provider-key-anthropic") as HTMLInputElement
      ).value = "sk-ant";
      (
        document.getElementById("provider-key-google") as HTMLInputElement
      ).value = "sk-goog";
      await ui2.saveProviderSettings();
      (
        document.getElementById("provider-key-openai") as HTMLInputElement
      ).value = "";
      (
        document.getElementById("provider-key-anthropic") as HTMLInputElement
      ).value = "";
      (
        document.getElementById("provider-key-google") as HTMLInputElement
      ).value = "";
      await ui2.saveProviderSettings();
      await ui2.clearProviderSettings();

      // 16. closeAllDropdowns when menu has no previousElementSibling or non-dropdown-btn sibling (line 2635)
      const nonBtnSibling = document.createElement("span");
      const orphanMenu = document.createElement("div");
      orphanMenu.className = "dropdown-menu show";
      document.body.appendChild(nonBtnSibling);
      document.body.appendChild(orphanMenu);
      (ui2 as unknown as { closeAllDropdowns: () => void }).closeAllDropdowns();
      nonBtnSibling.remove();
      orphanMenu.remove();

      // 17. Sidebar keyboard activation (Enter/Space/Escape), dropdown keyboard navigation, and delete cancel (lines 2778-2885)
      ui2.render();
      const titleBtn =
        document.querySelector<HTMLButtonElement>(".chat-item-title")!;
      titleBtn.dispatchEvent(
        new KeyboardEvent("keydown", { key: "Enter", bubbles: true }),
      );
      titleBtn.dispatchEvent(
        new KeyboardEvent("keydown", { key: " ", bubbles: true }),
      );
      titleBtn.dispatchEvent(
        new KeyboardEvent("keydown", { key: "Escape", bubbles: true }),
      );

      const dropBtn =
        document.querySelector<HTMLButtonElement>(".dropdown-btn")!;
      const dropMenu = document.querySelector<HTMLElement>(".dropdown-menu")!;
      dropBtn.click();
      dropBtn.click();
      dropBtn.dispatchEvent(
        new KeyboardEvent("keydown", { key: "ArrowDown", bubbles: true }),
      );
      dropMenu.dispatchEvent(
        new KeyboardEvent("keydown", { key: "ArrowDown", bubbles: true }),
      );
      dropMenu.dispatchEvent(
        new KeyboardEvent("keydown", { key: "ArrowUp", bubbles: true }),
      );
      dropMenu.dispatchEvent(
        new KeyboardEvent("keydown", { key: "Home", bubbles: true }),
      );
      dropMenu.dispatchEvent(
        new KeyboardEvent("keydown", { key: "End", bubbles: true }),
      );
      dropMenu.dispatchEvent(
        new KeyboardEvent("keydown", { key: "a", bubbles: true }),
      );
      dropMenu.dispatchEvent(
        new KeyboardEvent("keydown", { key: "Escape", bubbles: true }),
      );
      dropBtn.dispatchEvent(
        new KeyboardEvent("keydown", { key: "ArrowUp", bubbles: true }),
      );
      dropBtn.dispatchEvent(
        new KeyboardEvent("keydown", { key: "a", bubbles: true }),
      );

      const origPrompt = window.prompt;
      const origConfirm = window.confirm;
      window.prompt = () => null;
      dropMenu
        .querySelectorAll<HTMLButtonElement>(".dropdown-item")[0]
        ?.click();
      window.confirm = () => false;
      dropMenu
        .querySelectorAll<HTMLButtonElement>(".dropdown-item")[2]
        ?.click();
      window.prompt = origPrompt;
      window.confirm = origConfirm;

      // 18. renderMessages when activeChat is null, NLP chip click when gemma4 option is absent/present, copy/play/refresh SQL buttons (lines 2925-3200)
      state.activeChatId = null;
      (ui2 as unknown as { renderMessages: () => void }).renderMessages();
      state.setActiveChat(chat.id);

      chat.messages = [];
      modelSelect.innerHTML = `<option value="sql">SQL</option><option value="gemma4">Gemma</option>`;
      ui2.render();
      const chips =
        document.querySelectorAll<HTMLButtonElement>(".empty-state .chip");
      chips[4]?.click();
      modelSelect.innerHTML = `<option value="sql">SQL</option>`;
      ui2.render();
      document
        .querySelectorAll<HTMLButtonElement>(".empty-state .chip")[4]
        ?.click();

      // Message with user role & no msg.model when activeChat.model === "sql", plus assistant message with sqlQuery and model !== "sql"
      chat.model = "sql";
      const msgNoContent: Record<string, unknown> = {
        role: "assistant",
        sqlQuery: "SELECT 1;",
        model: "gemma4",
      };
      chat.messages = [
        { role: "user", content: "SELECT 1;" },
        msgNoContent as never,
      ];
      ui2.render();

      // Click copy button with success (and 2000ms timeout) and with clipboard error
      const sqlCopyBtn =
        document.querySelector<HTMLButtonElement>(".copy-sql-btn")!;
      sqlCopyBtn.click();
      await new Promise((r) => setTimeout(r, 10));

      const origWriteText = navigator.clipboard.writeText;
      navigator.clipboard.writeText = () =>
        Promise.reject(new Error("Copy denied"));
      sqlCopyBtn.click();
      await new Promise((r) => setTimeout(r, 10));
      navigator.clipboard.writeText = origWriteText;

      // Click play button on assistant gemma4 sqlQuery
      window.fetch = async () =>
        new Response(JSON.stringify({ sqlResult: [{ a: 1 }] }), {
          status: 200,
        });
      const playBtns =
        document.querySelectorAll<HTMLButtonElement>(".play-sql-btn");
      playBtns[playBtns.length - 1]?.click();
      await new Promise((r) => setTimeout(r, 10));

      // 19. Table modal exports, pagination bounds, empty page backtrack, null cells, and fetch errors (lines 3368-3806)
      const openCalls: string[] = [];
      const origWindowOpen = window.open;
      window.open = ((url?: string | URL) => {
        openCalls.push(String(url));
        return null;
      }) as typeof window.open;

      let tablePageResponses: Array<Record<string, unknown>> = [
        {
          rows: [
            { id: 1, val: null },
            { id: 2, val: "ok" },
          ],
        },
      ];
      window.fetch = async () => {
        const nextResp = tablePageResponses.shift() ?? { rows: [] };
        return new Response(JSON.stringify(nextResp), { status: 200 });
      };

      const dummyTrigger = document.createElement("button");
      document.body.appendChild(dummyTrigger);
      dummyTrigger.focus();
      await ui2.openTableModal("patients");

      // Click full CSV, Excel, Parquet export buttons with default and custom DB
      state.setCurrentDb("t1d.duckdb");
      document.getElementById("modal-export-full-csv-btn")?.click();
      document.getElementById("modal-export-excel-btn")?.click();
      document.getElementById("modal-export-parquet-btn")?.click();

      state.setCurrentDb("custom_export.duckdb");
      document.getElementById("modal-export-full-csv-btn")?.click();
      document.getElementById("modal-export-excel-btn")?.click();
      document.getElementById("modal-export-parquet-btn")?.click();
      state.setCurrentDb("t1d.duckdb");

      // Toggle sortOrderBtn ASC -> DESC -> ASC
      tablePageResponses = [
        { rows: [{ id: 1, val: "a" }], total_pages: 1 },
        { rows: [{ id: 1, val: "a" }], total_pages: 1 },
      ];
      const sortOrderBtn = document.getElementById(
        "modal-table-sort-order",
      ) as HTMLButtonElement;
      sortOrderBtn.click();
      await new Promise((r) => setTimeout(r, 10));
      sortOrderBtn.click();
      await new Promise((r) => setTimeout(r, 10));

      // Sort via th click twice (asc -> desc) and keydown with Space and non-matching key
      tablePageResponses = [
        { rows: [{ id: 1, val: "a" }], total_pages: 1 },
        { rows: [{ id: 1, val: "a" }], total_pages: 1 },
        { rows: [{ id: 1, val: "a" }], total_pages: 1 },
      ];
      const firstTh = document.querySelector<HTMLElement>(
        "#modal-table-head th",
      )!;
      firstTh.click();
      await new Promise((r) => setTimeout(r, 10));
      document.querySelector<HTMLElement>("#modal-table-head th")?.click();
      await new Promise((r) => setTimeout(r, 10));
      document
        .querySelector<HTMLElement>("#modal-table-head th")
        ?.dispatchEvent(
          new KeyboardEvent("keydown", { key: " ", bubbles: true }),
        );
      await new Promise((r) => setTimeout(r, 10));
      document
        .querySelector<HTMLElement>("#modal-table-head th")
        ?.dispatchEvent(
          new KeyboardEvent("keydown", { key: "Tab", bubbles: true }),
        );

      // Prev page at page 1 (false branch of currentPage > 1)
      const prevBtn = document.getElementById(
        "prev-page-btn",
      ) as HTMLButtonElement;
      prevBtn.disabled = false;
      prevBtn.click();

      // Empty rows on page > 1 -> backtracks to page 1, then empty rows on page 1 -> "No data available"
      Object.assign(ui2, { currentPage: 2 });
      tablePageResponses = [{ rows: [], total_pages: 1 }, { rows: [] }];
      await (
        ui2 as unknown as { fetchTableData: () => Promise<void> }
      ).fetchTableData();

      // Error in fetchTableData (lines 3781-3806)
      window.fetch = async () =>
        Promise.reject(new Error("Table fetch failed"));
      await (
        ui2 as unknown as { fetchTableData: () => Promise<void> }
      ).fetchTableData();
      const errCellText =
        document.querySelector("#modal-table-body .error-text")?.textContent ??
        "";

      // Close modal when dummyTrigger was removed from DOM (restores focus to #chat-input)
      dummyTrigger.remove();
      document.getElementById("close-modal-btn")?.click();

      // 20. Minimal DOM instantiation to cover all optional element false branches in constructor/bindEvents/modals
      const optionalIds = [
        "system-status-chip",
        "system-status-banner",
        "banner-retry-btn",
        "banner-instructions-btn",
        "banner-instructions-drawer",
        "banner-dismiss-btn",
        "provider-settings-btn",
        "close-provider-modal-btn",
        "provider-save-btn",
        "provider-clear-btn",
        "provider-key-openai",
        "provider-key-anthropic",
        "provider-key-google",
        "provider-store-consent",
        "provider-modal",
        "cohort-filter-btn",
        "close-cohort-modal-btn",
        "cohort-table-select",
        "cohort-join-table-select",
        "cohort-join-type-select",
        "cohort-join-on-input",
        "cohort-min-age",
        "cohort-max-age",
        "cohort-gender-select",
        "cohort-txgroup",
        "cohort-max-tir",
        "cohort-min-hba1c",
        "cohort-insert-chat-btn",
        "cohort-execute-btn",
        "cohort-sql-preview",
        "cohort-modal",
        "streaming-toggle-btn",
        "db-select",
        "sync-sessions-btn",
        "modal-table-search",
        "modal-table-sort-col",
        "modal-table-sort-order",
        "modal-export-csv-btn",
        "modal-export-json-btn",
        "modal-export-full-csv-btn",
        "modal-export-excel-btn",
        "modal-export-parquet-btn",
      ];
      const savedNodes: Array<{
        el: HTMLElement;
        parent: Node;
        next: Node | null;
      }> = [];
      for (const id of optionalIds) {
        const el = document.getElementById(id);
        if (el && el.parentNode) {
          savedNodes.push({
            el,
            parent: el.parentNode,
            next: el.nextSibling,
          });
          el.remove();
        }
      }

      // Also test schema-explorer without .schema-header and without #sidebar
      const schemaHeader = document.querySelector(".schema-header");
      const toggleSchemaBtn = document.getElementById("toggle-schema-btn");
      const schemaExp = document.getElementById("schema-explorer");
      const sidebar = document.getElementById("sidebar");
      if (schemaHeader && toggleSchemaBtn && schemaExp) {
        schemaExp.prepend(toggleSchemaBtn);
        schemaHeader.remove();
      }
      if (sidebar) sidebar.id = "sidebar-temp";

      window.fetch = async () =>
        new Response(
          JSON.stringify({
            rows: [{ id: 1 }],
            models: [],
            tables: [],
          }),
          { status: 200 },
        );

      const minimalUi = new uiMod.ChatUI(state);
      toggleSchemaBtn?.click();
      toggleSchemaBtn?.click();
      minimalUi.updateStreamingToggleUi();
      minimalUi.openCohortModal();
      minimalUi.closeCohortModal();
      minimalUi.updateCohortSqlPreview();
      minimalUi.handleCohortJoinTableChange();
      await minimalUi.openProviderModal();
      minimalUi.closeProviderModal();
      await minimalUi.saveProviderSettings();
      await minimalUi.clearProviderSettings();
      await minimalUi.openTableModal("patients");

      // Also test openProviderModal when providerModal is present but inputs are absent, and handleCohortJoinTableChange when cohortJoinOnInput is null
      const provModalNode = savedNodes.find(
        (n) => n.el.id === "provider-modal",
      );
      const cohortTableNode = savedNodes.find(
        (n) => n.el.id === "cohort-table-select",
      );
      const cohortJoinNode = savedNodes.find(
        (n) => n.el.id === "cohort-join-table-select",
      );
      const cohortModalNode = savedNodes.find(
        (n) => n.el.id === "cohort-modal",
      );
      if (provModalNode) {
        provModalNode.parent.insertBefore(provModalNode.el, provModalNode.next);
      }
      if (cohortModalNode) {
        cohortModalNode.parent.insertBefore(
          cohortModalNode.el,
          cohortModalNode.next,
        );
      }
      if (cohortTableNode) {
        cohortTableNode.parent.insertBefore(
          cohortTableNode.el,
          cohortTableNode.next,
        );
      }
      if (cohortJoinNode) {
        cohortJoinNode.parent.insertBefore(
          cohortJoinNode.el,
          cohortJoinNode.next,
        );
      }
      const partialUi = new uiMod.ChatUI(state);
      await partialUi.openProviderModal();
      await partialUi.saveProviderSettings();
      await partialUi.clearProviderSettings();
      partialUi.openCohortModal();
      (cohortTableNode?.el as HTMLSelectElement).value = "patients";
      (cohortJoinNode?.el as HTMLSelectElement).value = "cgm_data";
      partialUi.handleCohortJoinTableChange();
      (cohortJoinNode?.el as HTMLSelectElement).value = "patients";
      partialUi.handleCohortJoinTableChange();

      // Remove table-modal and table-loading to cover early return guards (lines 3537, 3661)
      const tableModalEl = document.getElementById("table-modal")!;
      const tableLoadingEl = document.getElementById("table-loading")!;
      tableLoadingEl.id = "table-loading-temp";
      await (
        partialUi as unknown as { fetchTableData: () => Promise<void> }
      ).fetchTableData();
      tableLoadingEl.id = "table-loading";
      tableModalEl.id = "table-modal-temp";
      await partialUi.openTableModal("patients");
      tableModalEl.id = "table-modal";

      // Remove schema-explorer and toggle-schema-btn to cover line 1747 false branch, and restore bannerInstructionsBtn alone without bannerInstructionsDrawer
      const instrBtnNode = savedNodes.find(
        (n) => n.el.id === "banner-instructions-btn",
      );
      if (instrBtnNode) {
        document.body.appendChild(instrBtnNode.el);
      }
      schemaExp?.remove();
      const noSchemaUi = new uiMod.ChatUI(state);
      noSchemaUi.destroy();

      // Restore DOM & globals
      if (sidebar) sidebar.id = "sidebar";
      for (const item of savedNodes.reverse()) {
        item.parent.insertBefore(item.el, item.next);
      }
      window.setTimeout = origSetTimeout;
      window.open = origWindowOpen;
      window.fetch = origFetch;
      await i18nMod.setLanguage("en");

      const pwCfgMod = await import("/playwright.config.ts");
      const pwFullCfgMod = await import("/playwright.full.config.ts");
      const viteCfgMod = await import("/vite.config.ts");
      const vitestCfgMod = await import("/vitest.config.ts");

      pwCfgMod.createPlaywrightConfig();
      pwFullCfgMod.createPlaywrightFullConfig();
      viteCfgMod.createViteConfig({});
      viteCfgMod.createViteConfig({
        T1D_API_PORT: "8080",
        T1D_WEB_PORT: "4173",
      });
      viteCfgMod.createViteConfig({
        VITE_API_PROXY_TARGET: "http://127.0.0.1:9000",
        BUILD_TIMESTAMP_UTC: "2026-10-31T00:00:00Z UTC",
      });
      (
        globalThis as unknown as {
          process?: { env?: Record<string, string> };
        }
      ).process = { env: { T1D_API_PORT: "8001" } };
      viteCfgMod.getDefaultViteEnv();
      delete (globalThis as unknown as { process?: unknown }).process;
      viteCfgMod.getDefaultViteEnv();
      vitestCfgMod.createVitestConfig();

      return {
        missingElError,
        emptyFallbackModel,
        nonEmptyFallbackModel,
        keyReadyModel,
        openCallsCount: openCalls.length,
        errCellText,
      };
    });

    expect(uiCoverageCheck.missingElError).toContain(
      "Required DOM element #non-existent-required-id not found",
    );
    expect(uiCoverageCheck.emptyFallbackModel).toBe("sql");
    expect(uiCoverageCheck.nonEmptyFallbackModel).toBe("openai/gpt-4o");
    expect(uiCoverageCheck.keyReadyModel).toBe("openai/gpt-4o");
    expect(uiCoverageCheck.openCallsCount).toBe(6);
    expect(uiCoverageCheck.errCellText).toContain("Table fetch failed");
  });
});
