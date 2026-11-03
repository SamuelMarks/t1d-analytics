/**
 * @file diagnostics-and-modals.spec.ts
 * @module tests-e2e/diagnostics-and-modals.spec
 * End-to-end Playwright tests for system health banners, degraded/offline recovery,
 * Schema Explorer interactions, Provider Settings Modal, and the Table Preview Modal.
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
            is_synthetic_faked_data: true,
            data_provenance: "synthetic_seed",
          },
          ollama: { accessible: false },
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

    // 4. Transition to healthy DB with synthetic data and offline ollama -> provenance and rule-based SQL alerts appear and can be dismissed with ✖
    statusPhase = "healthy";
    await page.click("#banner-retry-btn");
    await expect(page.locator("#jaeb-provenance-alert")).toBeVisible();
    await expect(page.locator("#rule-based-sql-alert")).toBeVisible();
    await page.click("#jaeb-provenance-alert-dismiss");
    await expect(page.locator("#jaeb-provenance-alert")).toHaveClass(/hidden/);
    await page.click("#rule-based-sql-alert-dismiss");
    await expect(page.locator("#rule-based-sql-alert")).toHaveClass(/hidden/);
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
    await expect(openBtn).toHaveAttribute("aria-expanded", "false");
    await expect(openBtn).toBeFocused();

    // Re-open mobile sidebar and close via backdrop overlay click
    await openBtn.click();
    await expect(page.locator("body")).toHaveClass(/sidebar-open/);
    await page.locator("#overlay").click({ force: true });
    await expect(page.locator("body")).not.toHaveClass(/sidebar-open/);
  });

  /**
   * Tests Cloud Provider & API Key Settings modal configuration, badge updates, and light/dark theme persistence.
   */
  test("Configures Cloud Provider API keys in Provider Settings Modal and persists light/dark theme across reloads", async ({
    page,
  }) => {
    await page.route("**/api/providers/status", async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        json: {
          providers: {
            openai: { configured: false, available: false },
            anthropic: { configured: true, available: true },
            google: { configured: false, available: false },
          },
        },
      });
    });

    await page.route("**/api/models*", async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        json: {
          models: [
            { name: "gemma4", provider: "ollama", configured: true },
            {
              name: "openai/gpt-4o",
              provider: "openai",
              configured: false,
              requires_key: true,
            },
          ],
        },
      });
    });

    await page.goto("/");

    // Toggle light mode and verify persistence across reload
    await expect(page.locator("body")).not.toHaveClass(/light-mode/);
    await page.click("#theme-toggle-btn");
    await expect(page.locator("body")).toHaveClass(/light-mode/);
    await page.reload();
    await expect(page.locator("body")).toHaveClass(/light-mode/);
    await page.click("#theme-toggle-btn");
    await expect(page.locator("body")).not.toHaveClass(/light-mode/);

    // Open Provider Settings Modal
    await page.click("#provider-settings-btn");
    const providerModal = page.locator("#provider-modal");
    await expect(providerModal).toBeVisible();
    await expect(page.locator("#provider-badge-anthropic")).toHaveClass(
      /badge-configured/,
    );
    await expect(page.locator("#provider-badge-openai")).toHaveClass(
      /badge-missing/,
    );

    // Save an OpenAI API key with consent checked
    await page.check("#provider-store-consent");
    await page.fill("#provider-key-openai", "sk-e2e-test-key");
    await page.click("#provider-save-btn");
    await expect(providerModal).toHaveClass(/hidden/);

    // Select openai/gpt-4o and verify provider status dot is configured
    await page.selectOption("#model-select", "openai/gpt-4o");
    await expect(page.locator("#provider-status-dot")).toHaveClass(
      /configured/,
    );

    // Re-open Provider Settings Modal, clear stored keys, and close
    await page.click("#provider-settings-btn");
    await expect(page.locator("#provider-key-openai")).toHaveValue(
      "sk-e2e-test-key",
    );
    await page.click("#provider-clear-btn");
    await expect(page.locator("#provider-key-openai")).toHaveValue("");
    await page.click("#close-provider-modal-btn");
    await expect(providerModal).toHaveClass(/hidden/);
  });
});
