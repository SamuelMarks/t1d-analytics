/**
 * @file clinical-cohort-and-charts.spec.ts
 * @module tests-e2e/clinical-cohort-and-charts.spec
 * End-to-end Playwright tests for the Visual Cohort Query Builder modal,
 * Continuous Glucose Monitoring (CGM) SVG clinical visualizations (TIR, AGP, histogram),
 * multi-database selector switching, and session synchronization.
 */

import { test, expect } from "./base-test";

test.describe("Visual Cohort Builder, Clinical CGM Charts & Multi-Database Switching", () => {
  /**
   * Tests the Visual Cohort Query Builder modal: table joins, clinical predicates, live SQL preview,
   * Insert into Chat, and Run Cohort Query.
   */
  test("Builds multi-table cohort SQL dynamically in Visual Cohort Query Builder and executes cohort query", async ({
    page,
  }) => {
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
                { name: "gender", type: "VARCHAR" },
                { name: "tx_group", type: "VARCHAR" },
              ],
            },
            {
              name: "hba1c_labs",
              columns: [
                { name: "patient_id", type: "VARCHAR" },
                { name: "hba1c", type: "DOUBLE" },
              ],
            },
            {
              name: "sensor_events",
              columns: [
                { name: "device_id", type: "VARCHAR" },
                { name: "event_type", type: "VARCHAR" },
              ],
            },
            {
              name: "pump_logs",
              columns: [
                { name: "device_id", type: "VARCHAR" },
                { name: "bolus_units", type: "DOUBLE" },
              ],
            },
            {
              name: "audit_trail",
              columns: [{ name: "log_ts", type: "VARCHAR" }],
            },
            {
              name: "patient_notes",
              columns: [{ name: "note_text", type: "VARCHAR" }],
            },
          ],
          database: {
            status_code: "healthy",
            configured_path: "t1d.duckdb",
            exists: true,
            connected: true,
            table_count: 6,
            has_initial_data: true,
          },
        },
      });
    });

    let executedCohortSql = "";
    await page.route("**/api/chat", async (route) => {
      const body = route.request().postDataJSON() as { message: string };
      executedCohortSql = body.message;
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        json: {
          content: "Cohort query executed.",
          sqlQuery: body.message,
          sqlResult: [
            { patient_id: "PT-101", age: 28, gender: "F", hba1c: 8.3 },
            { patient_id: "PT-102", age: 34, gender: "F", hba1c: 8.9 },
          ],
        },
      });
    });

    await page.goto("/");
    await expect(page.locator(".schema-table")).toHaveCount(6);

    // Open Visual Cohort Query Builder
    await page.click("#cohort-filter-btn");
    const cohortModal = page.locator("#cohort-modal");
    await expect(cohortModal).toBeVisible();

    // Test Shift+Tab focus wrap from first focusable element in Cohort Modal
    await page.locator("#close-cohort-modal-btn").focus();
    await page.keyboard.press("Shift+Tab");
    await expect(page.locator("#cohort-execute-btn")).toBeFocused();
    await page.keyboard.press("Tab");
    await expect(page.locator("#close-cohort-modal-btn")).toBeFocused();
    await page.keyboard.press("Tab");
    await expect(page.locator("#cohort-table-select")).toBeFocused();
    await page.keyboard.press("Shift+Tab");
    await expect(page.locator("#close-cohort-modal-btn")).toBeFocused();

    // Exercise reverse ontology lookup (hba1c_labs -> patients)
    await page.selectOption("#cohort-table-select", "hba1c_labs");
    await page.selectOption("#cohort-join-table-select", "patients");
    await expect(page.locator("#cohort-join-on-input")).toHaveValue(
      '"hba1c_labs".subject_id = "patients".subject_id',
    );

    // Exercise shared column discovery (sensor_events <-> pump_logs via device_id)
    await page.selectOption("#cohort-table-select", "sensor_events");
    await page.selectOption("#cohort-join-table-select", "pump_logs");
    await expect(page.locator("#cohort-join-on-input")).toHaveValue(
      '"sensor_events".device_id = "pump_logs".device_id',
    );

    // Exercise patient substring fallback (sensor_events <-> patient_notes)
    await page.selectOption("#cohort-join-table-select", "patient_notes");
    await expect(page.locator("#cohort-join-on-input")).toHaveValue(
      '"sensor_events".patient_id = "patient_notes".patient_id',
    );

    // Exercise unknown table fallback (sensor_events <-> audit_trail) and cleared joinCondition
    await page.selectOption("#cohort-join-table-select", "audit_trail");
    await page.fill("#cohort-join-on-input", "");
    await expect(page.locator("#cohort-sql-preview")).toContainText(
      'ON "sensor_events".patient_id = "audit_trail".patient_id',
    );

    // Select primary and secondary join table
    await page.selectOption("#cohort-table-select", "patients");
    await page.selectOption("#cohort-join-table-select", "hba1c_labs");
    await page.selectOption("#cohort-join-type-select", "LEFT JOIN");

    // Fill clinical criteria
    await page.fill("#cohort-min-age", "18");
    await page.fill("#cohort-max-age", "65");
    await page.selectOption("#cohort-gender-select", "F");
    await page.fill("#cohort-txgroup", "AID");
    await page.fill("#cohort-max-tir", "70");
    await page.fill("#cohort-min-hba1c", "8.0");

    // Verify live SQL preview reflects all predicates and JOIN
    const previewText = await page.locator("#cohort-sql-preview").textContent();
    expect(previewText).toContain('FROM "patients"');
    expect(previewText).toContain('LEFT JOIN "hba1c_labs"');
    expect(previewText).toContain("age >= 18");
    expect(previewText).toContain("age <= 65");
    expect(previewText).toContain("gender = 'F'");
    expect(previewText).toContain("txgroup ILIKE '%AID%'");
    expect(previewText).toContain("tir <= 70");
    expect(previewText).toContain("hba1c >= 8");

    // Click "Insert into Chat" first and verify it populates #chat-input and closes modal
    await page.click("#cohort-insert-chat-btn");
    await expect(cohortModal).toBeHidden();
    const inputVal = await page.locator("#chat-input").inputValue();
    expect(inputVal).toContain('FROM "patients"');

    // Re-open modal and click "Run Cohort Query" to execute directly
    await page.click("#cohort-filter-btn");
    await expect(cohortModal).toBeVisible();
    await page.click("#cohort-execute-btn");
    await expect(cohortModal).toBeHidden();

    await expect(page.locator(".message.assistant .sql-table")).toBeVisible();
    expect(executedCohortSql).toContain('FROM "patients"');
  });

  /**
   * Tests Clinical CGM Analytics card rendering (TIR stacked bar, AGP profile, histogram, GMI/CV metrics)
   * and RTL chart adaptation.
   */
  test("Renders Clinical CGM Analytics dashboard (TIR, AGP, Histogram, GMI, CV) for glucose time-series results", async ({
    page,
  }) => {
    const cgmRows = Array.from({ length: 24 }, (_, idx) => ({
      patient_id: "PT-01",
      timestamp: `2025-01-10T${String(idx).padStart(2, "0")}:00:00Z`,
      glucose:
        idx < 2 ? 50 : idx < 5 ? 65 : idx < 18 ? 125 : idx < 22 ? 210 : 275,
    }));

    await page.route("**/api/chat", async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        json: {
          content: "CGM profile loaded.",
          sqlQuery:
            "SELECT patient_id, timestamp, glucose FROM cgm_readings LIMIT 24;",
          sqlResult: cgmRows,
        },
      });
    });

    await page.goto("/");
    await page.selectOption("#model-select", "sql");
    await page.fill(
      "#chat-input",
      "SELECT patient_id, timestamp, glucose FROM cgm_readings LIMIT 24;",
    );
    await page.click("#send-btn");

    const cgmCard = page.locator(".cgm-analytics-card").first();
    await expect(cgmCard).toBeVisible();

    // Verify SVG visualizations (TIR stacked bar, AGP ribbon, Histogram) are rendered
    const svgCount = await cgmCard.locator("svg").count();
    expect(svgCount).toBeGreaterThanOrEqual(2);

    // Switch language to Arabic (RTL) and verify CGM card adapts cleanly
    await page.selectOption("#lang-select", "ar");
    await expect(page.locator("html")).toHaveAttribute("dir", "rtl");
    await expect(page.locator(".cgm-analytics-card").first()).toBeVisible();

    // Exercise observation window selector and print report button on the rendered CGM card
    await page.evaluate(() => {
      window.print = () => {};
    });
    await page.locator(".cgm-analytics-card .print-report-btn").first().click();
    await page
      .locator(".cgm-analytics-card .agp-window-select")
      .first()
      .selectOption("30");
    await expect(page.locator(".cgm-analytics-card").first()).toBeVisible();
  });

  /**
   * Tests switching active DuckDB databases in the sidebar selector and syncing sessions with the backend.
   */
  test("Switches active DuckDB database in #db-select and synchronizes chat sessions via #sync-sessions-btn", async ({
    page,
  }) => {
    const schemaDbParams: string[] = [];

    await page.route("**/api/databases", async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        json: {
          databases: [
            { name: "t1d.duckdb", path: "t1d.duckdb", size_bytes: 1048576 },
            {
              name: "secondary_cohort.duckdb",
              path: "secondary_cohort.duckdb",
              size_bytes: 2097152,
            },
          ],
          current_db: "t1d.duckdb",
        },
      });
    });

    await page.route("**/api/schema*", async (route) => {
      const url = new URL(route.request().url());
      schemaDbParams.push(url.searchParams.get("db_path") || "default");
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        json: {
          tables: [
            {
              name: "cohort_metrics",
              columns: [{ name: "patient_id", type: "VARCHAR" }],
            },
          ],
          database: {
            status_code: "healthy",
            configured_path: url.searchParams.get("db_path") || "t1d.duckdb",
            exists: true,
            connected: true,
            table_count: 1,
            has_initial_data: true,
          },
        },
      });
    });

    await page.route("**/api/sessions*", async (route) => {
      if (route.request().method() === "GET") {
        await route.fulfill({
          status: 200,
          contentType: "application/json",
          json: [
            {
              session_id: "server-synced-session-1",
              title: "Synced Endocrinology Session",
              model: "sql",
              created_at: "2025-01-01T00:00:00Z",
              updated_at: "2025-01-01T01:00:00Z",
              messages: [
                {
                  role: "user",
                  content: "SELECT COUNT(*) FROM cohort_metrics;",
                  model: "sql",
                },
                {
                  role: "assistant",
                  content: "backend.literalSql",
                  sqlQuery: "SELECT COUNT(*) FROM cohort_metrics;",
                  sqlResult: [{ count: 120 }],
                  model: "sql",
                },
              ],
            },
          ],
        });
      } else {
        await route.fulfill({
          status: 200,
          contentType: "application/json",
          json: { status: "ok" },
        });
      }
    });

    await page.goto("/");

    // Select secondary database
    const dbSelect = page.locator("#db-select");
    await expect(dbSelect.locator("option")).toHaveCount(2);
    await dbSelect.selectOption("secondary_cohort.duckdb");
    await expect
      .poll(() => schemaDbParams)
      .toContain("secondary_cohort.duckdb");

    // Trigger manual session sync and verify synced session appears in sidebar
    await page.click("#sync-sessions-btn");
    await expect(page.locator("#chat-list")).toContainText(
      "Synced Endocrinology Session",
    );
  });
});
