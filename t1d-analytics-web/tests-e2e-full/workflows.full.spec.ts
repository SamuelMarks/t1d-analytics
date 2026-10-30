/**
 * @file workflows.full.spec.ts
 * @module tests-e2e-full/workflows.full.spec
 * Full-stack end-to-end Playwright tests for clinical workflows against a live backend.
 */

import { test, expect } from "./base-test";
import AxeBuilder from "@axe-core/playwright";

test.describe("App Workflows Full E2E", () => {
  test.beforeEach(async ({ page }) => {
    await page.goto("/");
  });

  /**
   * Tests the Schema Explorer UI behavior including toggling and viewing table data.
   */
  test("Schema Explorer: view schema and open table data modal", async ({
    page,
  }) => {
    // Verify the schema table is rendered (from real duckdb)
    // There may be multiple tables (e.g. patients, tiny), so we find the one containing "tiny"
    const schemaTable = page
      .locator(".schema-table")
      .filter({ hasText: "tiny" })
      .first();
    await expect(schemaTable).toBeVisible({ timeout: 10000 });
    await expect(schemaTable.locator(".schema-table-header span")).toHaveText(
      "tiny",
    );

    // Expand the table columns view
    await schemaTable.locator(".schema-table-header-toggle").click();
    await expect(schemaTable.locator(".schema-columns")).toBeVisible();
    await expect(schemaTable.locator(".schema-column").first()).toContainText(
      "id",
    );

    // Open table data modal
    await schemaTable.locator(".table-view-btn").click();
    const modal = page.locator("#table-modal");
    await expect(modal).toBeVisible();

    // Verify modal content
    const modalTitle = page.locator("#modal-title");
    await expect(modalTitle).toHaveText("Table: tiny");

    // Close modal
    await page.locator("#close-modal-btn").click();
    await expect(modal).not.toBeVisible();
  });

  /**
   * Tests chat management features such as creating, renaming, and duplicating chats.
   */
  test("Chat Management: rename and duplicate chat", async ({ page }) => {
    // Send a message so we have a non-temporary chat
    await page.selectOption("#model-select", "sql");
    await page.fill("#chat-input", "SELECT * FROM tiny LIMIT 1");
    await page.click("#send-btn");

    // Wait for chat item to appear in the list
    const chatItem = page.locator(".chat-item").first();
    await expect(chatItem).toBeVisible();

    // The chat list gets populated asynchronously
    await page.waitForFunction(() => {
      const items = document.querySelectorAll(".chat-item");
      return (
        items.length > 0 && items[0].querySelector(".chat-item-title") !== null
      );
    });

    // Open dropdown menu
    await chatItem.locator(".dropdown-btn").click();

    // Prepare for prompt dialog
    page.once("dialog", (dialog) => {
      dialog.accept("My Custom Chat");
    });

    // Rename
    await chatItem
      .locator(".dropdown-item")
      .filter({ hasText: "Rename" })
      .click();
    await expect(chatItem.locator(".chat-item-title")).toHaveText(
      "My Custom Chat",
    );

    // Duplicate
    await chatItem.locator(".dropdown-btn").click();
    await chatItem
      .locator(".dropdown-item")
      .filter({ hasText: "Duplicate" })
      .click();

    // Now there should be two chats
    const chats = page.locator(".chat-item");
    await expect(chats).toHaveCount(2);
    await expect(chats.nth(1).locator(".chat-item-title")).toHaveText(
      "My Custom Chat (Copy)",
    );
  });

  /**
   * Tests executing SQL manually via the Play button on an assistant message.
   */
  test("SQL Execution: execute query manually from assistant message", async ({
    page,
  }) => {
    await page.selectOption("#model-select", "sql");
    await page.fill("#chat-input", "SELECT * FROM tiny;");
    await page.click("#send-btn");

    // Wait for response containing SQL
    const assistantMessage = page.locator(".message.assistant").first();
    await expect(assistantMessage).toBeVisible();
    await expect(assistantMessage.locator(".sql-query")).toHaveText(
      "SELECT * FROM tiny;",
    );

    // Get table value
    const sqlTable = assistantMessage.locator(".sql-table");
    await expect(sqlTable).toBeVisible();
    await expect(sqlTable.locator("td").first()).toHaveText("1");

    // The refresh button modifies the message in-place
    const refreshBtn = assistantMessage.locator(
      'button[data-i18n-title="ui.refreshQuery"]',
    );
    await expect(refreshBtn).toBeVisible();
    await refreshBtn.click();

    // Verify it refreshed by waiting for the table to still be there
    await expect(sqlTable).toBeVisible();
    await expect(sqlTable.locator("td").first()).toHaveText("1");
    await expect(sqlTable.locator("td").nth(1)).toHaveText("Test A");
  });

  /**
   * Tests the theme toggling functionality (Dark/Light mode).
   */
  test("Theme Toggle: switches between light and dark mode", async ({
    page,
  }) => {
    const themeBtn = page.locator("#theme-toggle-btn");
    const body = page.locator("body");

    await expect(body).not.toHaveClass(/light-mode/);
    await themeBtn.click();
    await expect(body).toHaveClass(/light-mode/);
    await themeBtn.click();
    await expect(body).not.toHaveClass(/light-mode/);
  });

  /**
   * Tests language selection updates the UI text asynchronously.
   */
  test("Language Selection: changing language updates i18n texts", async ({
    page,
  }) => {
    const langSelect = page.locator("#lang-select");

    // Switch to Japanese
    await langSelect.selectOption("ja");

    // The new chat button text should change away from "+ New Chat"
    const newChatBtn = page.locator("#new-chat-btn");
    await expect(newChatBtn).not.toHaveText("+ New Chat", { timeout: 5000 });
  });

  /**
   * Tests keyboard shortcut Shift+Enter for newline in chat input.
   */
  test("Chat Input: Shift+Enter adds newline, Enter sends", async ({
    page,
  }) => {
    const chatInput = page.locator("#chat-input");

    // Wait for models to load and enable input
    await expect(chatInput).toBeEnabled();

    // Use page.keyboard to simulate Shift+Enter properly
    await chatInput.click();
    await page.keyboard.type("Line 1");
    await page.keyboard.down("Shift");
    await page.keyboard.press("Enter");
    await page.keyboard.up("Shift");
    await page.keyboard.type("Line 2");

    // The value should contain a newline
    const value = await chatInput.inputValue();
    expect(value).toBe("Line 1\nLine 2");

    // Now press Enter without shift to send
    await page.keyboard.press("Enter");

    // Should have sent the message
    await expect(page.locator(".message.user").first()).toContainText(
      "Line 1\nLine 2",
    );
  });

  test("Modal Close: restores focus correctly or falls back to chat input", async ({
    page,
  }) => {
    // Open the schema explorer and click the table view button
    const schemaTable = page.locator(".schema-table").first();
    await expect(schemaTable).toBeVisible({ timeout: 10000 });

    const tableViewBtn = schemaTable.locator(".table-view-btn");
    await tableViewBtn.focus();
    await tableViewBtn.click();

    const modal = page.locator("#table-modal");
    await expect(modal).toBeVisible();

    // Now let's dynamically remove the button that opened the modal to simulate it disappearing from DOM
    await page.evaluate(() => {
      document.querySelector(".table-view-btn")?.remove();
    });

    // Close the modal
    await page.locator("#close-modal-btn").click();
    await expect(modal).not.toBeVisible();

    // Since previousFocus is gone, it should fallback to chat input
    const chatInput = page.locator("#chat-input");
    await expect(chatInput).toBeFocused();
  });

  test("Accessibility: complete accessibility audit", async ({ page }) => {
    // Wait for it to settle
    await expect(page.locator(".schema-table").first()).toBeVisible({
      timeout: 10000,
    });
    const accessibilityScanResults = await new AxeBuilder({ page }).analyze();
    expect(accessibilityScanResults.violations).toEqual([]);
  });

  test("Keyboard Navigation: toggles schema and opens modal", async ({
    page,
  }) => {
    // Navigate via tab to schema toggle and use enter
    await expect(page.locator(".schema-table").first()).toBeVisible({
      timeout: 10000,
    });

    await page.locator("#toggle-schema-btn").focus();
    await page.keyboard.press("Enter");

    // Verify it opened or toggled. It's normally open initially, so wait for it to collapse.
    const schemaContent = page.locator("#schema-content");
    await expect(schemaContent).toHaveCount(1);
    await page.keyboard.press("Enter");

    // Let's tab to the schema table toggle
    const tableToggle = page.locator(".schema-table-header-toggle").first();
    await tableToggle.focus();
    await page.keyboard.press("Enter");
    const expanded = await tableToggle.getAttribute("aria-expanded");
    expect(expanded).toMatch(/true|false/);

    // Tab to the view button
    const viewBtn = page.locator(".table-view-btn").first();
    await viewBtn.focus();
    await page.keyboard.press("Enter");

    // Modal opens, check focus is trapped
    const modal = page.locator("#table-modal");
    await expect(modal).toBeVisible();
    await expect(page.locator("#close-modal-btn")).toBeFocused();

    // Tab cycles focus within modal
    await page.keyboard.press("Tab"); // Next focusable (maybe prev btn or something else)
    await page.keyboard.down("Shift");
    await page.keyboard.press("Tab");
    await page.keyboard.up("Shift");
    await expect(page.locator("#close-modal-btn")).toBeFocused();

    // Press enter to close
    await page.keyboard.press("Enter");
    await expect(modal).not.toBeVisible();
    await expect(viewBtn).toBeFocused(); // focus returned to view button
  });

  /**
   * Tests exporting data directly from the table modal.
   */
  test("Table Modal: exports table data", async ({ page }) => {
    const schemaTable = page
      .locator(".schema-table")
      .filter({ hasText: "tiny" })
      .first();
    await expect(schemaTable).toBeVisible({ timeout: 10000 });

    await schemaTable.locator(".table-view-btn").click();
    const modal = page.locator("#table-modal");
    await expect(modal).toBeVisible();

    // Verify export buttons exist and are enabled
    const exportCsvBtn = page.locator("#modal-export-csv-btn");
    await expect(exportCsvBtn).toBeVisible();
    await expect(exportCsvBtn).toBeEnabled();

    // Set up download promise
    const downloadPromise = page.waitForEvent("download");
    await exportCsvBtn.click();
    const download = await downloadPromise;
    expect(download.suggestedFilename()).toContain(".csv");

    await page.locator("#close-modal-btn").click();
    await expect(modal).not.toBeVisible();
  });

  /**
   * Tests multi-turn chat sessions where multiple queries execute consecutively.
   */
  test("Multi-turn Chat: executes consecutive queries in the same session", async ({
    page,
  }) => {
    await page.selectOption("#model-select", "sql");

    // Turn 1
    await page.fill("#chat-input", "SELECT id, name FROM tiny WHERE id = 1;");
    await page.click("#send-btn");

    const firstMsg = page.locator(".message.assistant").first();
    await expect(firstMsg).toBeVisible({ timeout: 10000 });
    await expect(firstMsg.locator(".sql-table td").first()).toHaveText("1");

    // Turn 2
    await page.fill("#chat-input", "SELECT id, name FROM tiny WHERE id = 2;");
    await page.click("#send-btn");

    // Should now have 2 assistant messages
    const assistantMsgs = page.locator(".message.assistant");
    await expect(assistantMsgs).toHaveCount(2, { timeout: 10000 });
    await expect(
      assistantMsgs.nth(1).locator(".sql-table td").first(),
    ).toHaveText("2");
  });

  /**
   * Tests live Clinical CGM Analytics card rendering when querying cgm_readings from DuckDB.
   */
  test("Clinical CGM Analytics: renders live TIR, AGP, and Histogram SVGs from cgm_readings", async ({
    page,
  }) => {
    await page.selectOption("#model-select", "sql");
    await page.fill(
      "#chat-input",
      "SELECT subject_id, timestamp, glucose_value FROM cgm_readings LIMIT 50;",
    );
    await page.click("#send-btn");

    const cgmCard = page.locator(".cgm-analytics-card").first();
    await expect(cgmCard).toBeVisible({ timeout: 10000 });
    const svgCount = await cgmCard.locator("svg").count();
    expect(svgCount).toBeGreaterThanOrEqual(2);
  });

  /**
   * Tests Visual Cohort Query Builder executing a live DuckDB query against seeded tables.
   */
  test("Visual Cohort Builder: builds and executes live cohort query against DuckDB", async ({
    page,
  }) => {
    await expect(page.locator(".schema-table").first()).toBeVisible({
      timeout: 10000,
    });

    await page.click("#cohort-filter-btn");
    const cohortModal = page.locator("#cohort-modal");
    await expect(cohortModal).toBeVisible();

    await page.selectOption("#cohort-table-select", "patients");
    await page.fill("#cohort-min-age", "20");

    const previewSql = await page.locator("#cohort-sql-preview").textContent();
    expect(previewSql).toContain('FROM "patients"');
    expect(previewSql).toContain("age >= 20");

    await page.click("#cohort-execute-btn");
    await expect(cohortModal).toBeHidden();

    const assistantMsg = page.locator(".message.assistant").first();
    await expect(assistantMsg.locator(".sql-table")).toBeVisible({
      timeout: 10000,
    });
  });

  /**
   * Tests live sorting, searching, and JSON export in the Table Preview Modal against /api/table/tiny.
   */
  test("Table Modal Live Controls: sorts, searches, and exports JSON from live backend", async ({
    page,
  }) => {
    const schemaTable = page
      .locator(".schema-table")
      .filter({ hasText: "tiny" })
      .first();
    await expect(schemaTable).toBeVisible({ timeout: 10000 });

    await schemaTable.locator(".table-view-btn").click();
    const modal = page.locator("#table-modal");
    await expect(modal).toBeVisible();
    await expect(page.locator("#modal-table-body tr")).toHaveCount(2);

    // Sort by id DESC
    await page.selectOption("#modal-table-sort-col", "id");
    await page.click("#modal-table-sort-order");
    await expect(
      page.locator("#modal-table-body tr").first().locator("td").first(),
    ).toHaveText("2");

    // Search for "Test A"
    await page.fill("#modal-table-search", "Test A");
    await expect(page.locator("#modal-table-body tr")).toHaveCount(1);
    await expect(
      page.locator("#modal-table-body tr").first().locator("td").first(),
    ).toHaveText("1");

    // Export JSON page
    const jsonDownloadPromise = page.waitForEvent("download");
    await page.click("#modal-export-json-btn");
    const jsonDownload = await jsonDownloadPromise;
    expect(jsonDownload.suggestedFilename()).toContain(".json");

    await page.click("#close-modal-btn");
    await expect(modal).not.toBeVisible();
  });

  /**
   * Tests live SSE streaming execution of literal SQL via /api/chat/stream.
   */
  test("Live SSE Streaming: streams SQL execution results via /api/chat/stream", async ({
    page,
  }) => {
    await page.click("#streaming-toggle-btn");
    await expect(page.locator("#streaming-toggle-btn")).toHaveAttribute(
      "aria-pressed",
      "true",
    );

    await page.selectOption("#model-select", "sql");
    await page.fill("#chat-input", "SELECT * FROM tiny;");
    await page.click("#send-btn");

    const assistantMsg = page.locator(".message.assistant").first();
    await expect(assistantMsg.locator(".sql-table")).toBeVisible({
      timeout: 10000,
    });
    await expect(assistantMsg.locator(".sql-table tbody tr")).toHaveCount(2);
  });

  /**
   * Tests live server-side full CSV, Excel (.xlsx), and Parquet (.parquet) downloads from the Table Preview Modal
   * as well as live table profiling endpoint (/api/table/patients/profile).
   */
  test("Live Server Exports & Table Profiling: downloads CSV, Excel, and Parquet and verifies /api/table/patients/profile", async ({
    page,
  }) => {
    // 1. Verify live /api/table/patients/profile endpoint returns column completeness metrics
    const profileResp = await page.request.get("/api/table/patients/profile");
    expect(profileResp.ok()).toBe(true);
    const profileJson = (await profileResp.json()) as {
      table_name: string;
      total_rows: number;
      columns: Array<{ column_name: string; null_percentage: number }>;
    };
    expect(profileJson.table_name).toBe("patients");
    expect(profileJson.total_rows).toBeGreaterThan(0);
    expect(profileJson.columns.length).toBeGreaterThan(0);

    // 2. Open Table Preview Modal on tiny and trigger full CSV, Excel, and Parquet server exports
    await page.evaluate(() => {
      const openedUrls: string[] = [];
      (
        window as Window & { __openedExportUrls?: string[] }
      ).__openedExportUrls = openedUrls;
      window.open = (url?: string | URL): WindowProxy | null => {
        if (url) openedUrls.push(String(url));
        return null;
      };
    });

    const schemaTable = page
      .locator(".schema-table")
      .filter({ hasText: "tiny" })
      .first();
    await expect(schemaTable).toBeVisible({ timeout: 10000 });
    await schemaTable.locator(".table-view-btn").click();
    await expect(page.locator("#table-modal")).toBeVisible();
    await expect(page.locator("#modal-export-full-csv-btn")).toBeEnabled();

    await page.click("#modal-export-full-csv-btn");
    await page.click("#modal-export-excel-btn");
    await page.click("#modal-export-parquet-btn");

    const openedUrls = await page.evaluate(
      () =>
        (window as Window & { __openedExportUrls?: string[] })
          .__openedExportUrls || [],
    );
    expect(openedUrls.length).toBe(3);
    expect(openedUrls[0]).toContain("/api/export/csv?table_name=tiny");
    expect(openedUrls[1]).toContain("/api/export/excel?table_name=tiny");
    expect(openedUrls[2]).toContain("/api/export/parquet?table_name=tiny");

    // Verify each export URL returns HTTP 200 with non-empty content from the live FastAPI backend
    for (const exportUrl of openedUrls) {
      const exportResp = await page.request.get(exportUrl);
      expect(exportResp.ok()).toBe(true);
      const bodyBytes = await exportResp.body();
      expect(bodyBytes.byteLength).toBeGreaterThan(0);
    }
  });

  /**
   * Tests persistent server session synchronization (/api/sessions) across browser reloads and session deletion.
   */
  test("Persistent Server Sessions: syncs chat sessions to DuckDB, restores across reload, and deletes cleanly", async ({
    page,
  }) => {
    await page.selectOption("#model-select", "sql");
    await page.fill("#chat-input", "SELECT COUNT(*) AS total FROM patients;");
    await page.click("#send-btn");
    await expect(page.locator(".message.assistant .sql-table")).toBeVisible({
      timeout: 10000,
    });

    // Rename the active chat
    const firstChat = page.locator("#chat-list .chat-item").first();
    await firstChat.locator(".dropdown-btn").click();
    page.once("dialog", async (dialog) => {
      await dialog.accept("Persisted Clinical Session");
    });
    await firstChat.locator('.dropdown-menu [role="menuitem"]').first().click();
    await expect(firstChat.locator(".chat-item-title")).toHaveText(
      "Persisted Clinical Session",
    );

    // Explicitly sync with server via #sync-sessions-btn
    await page.click("#sync-sessions-btn");

    // Verify /api/sessions contains the session on the backend
    const sessionsResp = await page.request.get("/api/sessions");
    expect(sessionsResp.ok()).toBe(true);
    const sessionsPayload = (await sessionsResp.json()) as {
      sessions: Array<{
        session_id: string;
        title: string;
      }>;
    };
    expect(
      sessionsPayload.sessions.some(
        (s) => s.title === "Persisted Clinical Session",
      ),
    ).toBe(true);

    // Reload page and verify session title and result table are restored
    await page.reload();
    await expect(page.locator("#chat-list")).toContainText(
      "Persisted Clinical Session",
    );
    await expect(page.locator(".message.assistant .sql-table")).toBeVisible();
  });

  /**
   * Tests the Visual Cohort Query Builder modal with txgroup filtering, relational joins,
   * and live CGM SVG clinical visualization against the seeded DuckDB backend.
   */
  test("Live Visual Cohort Query Builder & CGM Visualization: executes joined cohort query and renders CGM SVG charts", async ({
    page,
  }) => {
    // Wait for schema tables to populate
    await expect(page.locator(".schema-table").first()).toBeVisible({
      timeout: 10000,
    });

    // Open Visual Cohort Query Builder
    await page.click("#cohort-filter-btn");
    const cohortModal = page.locator("#cohort-modal");
    await expect(cohortModal).toBeVisible();

    // Select patients and join with cgm_readings, filter by Min Age 18, Female, Closed-Loop
    await page.selectOption("#cohort-table-select", "patients");
    await page.selectOption("#cohort-join-table-select", "cgm_readings");
    await page.fill("#cohort-min-age", "18");
    await page.selectOption("#cohort-gender-select", "F");
    await page.fill("#cohort-txgroup", "Closed-Loop");

    const preview = page.locator("#cohort-sql-preview");
    await expect(preview).toContainText('FROM "patients"');
    await expect(preview).toContainText('INNER JOIN "cgm_readings"');
    await expect(preview).toContainText(
      'ON "patients".subject_id = "cgm_readings".subject_id',
    );
    await expect(preview).toContainText(
      "\"patients\".txgroup ILIKE '%Closed-Loop%'",
    );

    // Execute cohort query directly against live DuckDB
    await page.click("#cohort-execute-btn");
    await expect(cohortModal).not.toBeVisible();

    const assistantMsg = page.locator(".message.assistant").first();
    await expect(assistantMsg.locator(".sql-table")).toBeVisible({
      timeout: 10000,
    });
    await expect(assistantMsg.locator('[role="alert"]')).toHaveCount(0);

    // Verify CGM SVG clinical chart (Time-in-Range / AGP / Histogram) renders on live cgm_readings query
    await page.fill(
      "#chat-input",
      "SELECT subject_id, timestamp, glucose_value FROM cgm_readings WHERE subject_id = 1;",
    );
    await page.click("#send-btn");
    const cgmMsg = page.locator(".message.assistant").nth(1);
    const cgmCard = cgmMsg.locator(".cgm-analytics-card");
    await expect(cgmCard).toBeVisible({
      timeout: 10000,
    });
    await expect(cgmCard.locator("svg").first()).toBeVisible();

    // Change observation window dropdown (.agp-window-select) to "0" (All Available Data) and verify card re-renders
    const windowSelect = cgmCard.locator(".agp-window-select");
    await expect(windowSelect).toBeVisible();
    await windowSelect.selectOption("0");
    await expect(
      cgmMsg.locator(".cgm-analytics-card svg").first(),
    ).toBeVisible();
  });

  /**
   * Tests executing all built-in clinical SQL and NLP example chips in the empty state against the live seeded DuckDB backend.
   */
  test("Live Built-in Clinical Example SQL Chips: executes all 5 clinical trial chips cleanly against live DuckDB", async ({
    page,
  }) => {
    for (let chipIdx = 0; chipIdx < 5; chipIdx++) {
      // Create a fresh chat for each chip so the empty state chips are visible
      if (chipIdx > 0) {
        await page.click("#new-chat-btn");
      }
      const chip = page.locator(".empty-state .chip").nth(chipIdx);
      await expect(chip).toBeVisible({ timeout: 10000 });
      await chip.click();

      const inputVal = await page.locator("#chat-input").inputValue();
      if (chipIdx < 4) {
        expect(inputVal.toUpperCase()).toContain("SELECT");
      } else {
        expect(inputVal.toLowerCase()).toContain("demographics");
      }

      await page.click("#send-btn");
      const assistantMsg = page.locator(".message.assistant").first();
      await expect(assistantMsg.locator(".sql-table")).toBeVisible({
        timeout: 10000,
      });
      await expect(assistantMsg.locator('[role="alert"]')).toHaveCount(0);
    }
  });
});
