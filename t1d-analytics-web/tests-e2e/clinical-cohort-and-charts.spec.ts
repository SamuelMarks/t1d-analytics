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
    // Tab from first element to middle element (does not wrap)
    await page.keyboard.press("Tab");
    await expect(page.locator("#cohort-table-select")).toBeFocused();
    // Shift+Tab from middle element back to first element (does not wrap)
    await page.keyboard.press("Shift+Tab");
    await expect(page.locator("#close-cohort-modal-btn")).toBeFocused();
    // Dispatch Tab when focus is on modal root itself
    await page.evaluate(() => {
      const m = document.getElementById("cohort-modal")!;
      (document.activeElement as HTMLElement | null)?.blur();
      m.dispatchEvent(
        new KeyboardEvent("keydown", { key: "Tab", bubbles: true }),
      );
      (document.activeElement as HTMLElement | null)?.blur();
      m.dispatchEvent(
        new KeyboardEvent("keydown", {
          key: "Tab",
          shiftKey: true,
          bubbles: true,
        }),
      );
      m.dispatchEvent(
        new KeyboardEvent("keydown", { key: "ArrowDown", bubbles: true }),
      );
      const btns = m.querySelectorAll<HTMLElement>(
        'button:not([disabled]), [href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])',
      );
      btns.forEach((b) => {
        b.setAttribute("disabled", "true");
        if (b.hasAttribute("tabindex")) b.setAttribute("tabindex", "-1");
      });
      m.dispatchEvent(
        new KeyboardEvent("keydown", { key: "Tab", bubbles: true }),
      );
      btns.forEach((b) => {
        b.removeAttribute("disabled");
        if (b.getAttribute("tabindex") === "-1")
          b.setAttribute("tabindex", "0");
      });
    });

    // Exercise empty tableName fallback in buildCohortSql
    await page.evaluate(() => {
      const sel = document.getElementById(
        "cohort-table-select",
      ) as HTMLSelectElement;
      sel.value = "";
      sel.dispatchEvent(new Event("change"));
    });
    await expect(page.locator("#cohort-sql-preview")).toContainText(
      'FROM "patients"',
    );

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

    // Exercise all clinical date formats, generic value column detection, mmol/L scaling,
    // sensor wear compliance thresholds, and chart edge branches in browser runtime
    const edgeCheck = await page.evaluate(async () => {
      const chartMod = await import("/src/chart.ts");

      // 1. detectCgmColumns edge branches: empty rows, generic value/val columns (numeric, string, invalid, out-of-range)
      const detEmpty = chartMod.detectCgmColumns([]);
      const detGenericValid = chartMod.detectCgmColumns([
        { time: "2025-01-01T00:00:00Z", value: 110 },
        { time: "2025-01-01T00:05:00Z", value: "125.5" },
      ]);
      const detGenericLow = chartMod.detectCgmColumns([
        { time: "2025-01-01T00:00:00Z", value: 0.5 },
      ]);
      const detGenericHigh = chartMod.detectCgmColumns([
        { time: "2025-01-01T00:00:00Z", value: 900 },
      ]);
      const detGenericNaN = chartMod.detectCgmColumns([
        { time: "2025-01-01T00:00:00Z", value: "not-a-number" },
      ]);

      // 2. calculateTIR & calculateDayNightTIR: mmol/L scaling, single reading (n=1), string/invalid/negative glucose, invalid time
      const tirSingleMmol = chartMod.calculateTIR([5.5]);
      const dnMmol = chartMod.calculateDayNightTIR(
        [
          { time: "2025-01-01T10:00:00Z", glucose: 5.5 },
          { time: "2025-01-01T11:00:00Z", glucose: "6.2" },
          { time: "2025-01-01T12:00:00Z", glucose: "bad" },
          { time: "2025-01-01T13:00:00Z", glucose: null },
          { time: "2025-01-01T14:00:00Z", glucose: -5 },
          { time: "invalid-date", glucose: 6.0 },
          { time: "2025-01-01T23:00:00Z", glucose: 6.0 },
        ],
        "time",
        "glucose",
      );

      // 3. computePercentile & calculateAGP & calculateActiveWear edge branches
      const pEmpty = chartMod.computePercentile([], 50);
      const agpNull = chartMod.calculateAGP(
        [
          { time: "2025-01-01T10:00:00Z", glucose: "bad" },
          { time: "2025-01-01T10:05:00Z", glucose: null },
          { time: "2025-01-01T10:10:00Z", glucose: -10 },
        ],
        "time",
        "glucose",
      );
      const agpMixed = chartMod.calculateAGP(
        [
          { time: "2025-01-01T10:00:00Z", glucose: "120.5" },
          { time: "bad-time", glucose: 130 },
        ],
        "time",
        "glucose",
      );
      const wearEmpty = chartMod.calculateActiveWear([], "time");
      const wearZeroExpected = chartMod.calculateActiveWear(
        [{ time: 0, glucose: 100 }],
        "time",
        0,
        5,
      );
      const wearZeroEpoch = chartMod.calculateActiveWear(
        [{ time: 0, glucose: 100 }],
        "time",
      );

      // 4. parseDateAndMinuteOfDay & filterObservationWindow & filter14DayWindow
      const dValid = chartMod.parseDateAndMinuteOfDay(
        new Date("2025-01-10T08:30:00Z"),
      );
      const dInvalid = chartMod.parseDateAndMinuteOfDay(new Date("invalid"));
      const isoDateOnly = chartMod.parseDateAndMinuteOfDay("2025-01-10");
      const isoOffsetValid = chartMod.parseDateAndMinuteOfDay(
        "2025-01-10T08:30:00+02:00",
      );
      const isoOffsetInvalid = chartMod.parseDateAndMinuteOfDay(
        "2025-99-99T08:30:00+02:00",
      );
      const cdiscDateOnly = chartMod.parseDateAndMinuteOfDay("10-Jan-2025");
      const cdiscUnknownMonth = chartMod.parseDateAndMinuteOfDay(
        "10-Xyz-2025 08:30:00Z",
      );
      const cdiscOffsetValid = chartMod.parseDateAndMinuteOfDay(
        "10-Jan-2025 08:30:00+02:00",
      );
      const cdiscOffsetInvalid = chartMod.parseDateAndMinuteOfDay(
        "99-Jan-2025 08:30:00+99:99",
      );
      const slashUsDateOnly = chartMod.parseDateAndMinuteOfDay("01/10/2025");
      const slashEuWithTimeZ = chartMod.parseDateAndMinuteOfDay(
        "25/01/2025 08:30:00Z",
      );
      const slashOffsetValid = chartMod.parseDateAndMinuteOfDay(
        "25/01/2025 08:30:00-05:00",
      );
      const slashOffsetInvalid = chartMod.parseDateAndMinuteOfDay(
        "25/99/2025 08:30:00+99:99",
      );
      const natWithTz = chartMod.parseDateAndMinuteOfDay(
        "Wed Jan 15 2025 12:00:00 GMT+0000",
      );
      const natWithZ = chartMod.parseDateAndMinuteOfDay(
        "Wed Jan 15 2025 12:00:00Z",
      );
      const natLocal = chartMod.parseDateAndMinuteOfDay(
        "January 15, 2025 12:00:00",
      );
      const natInvalid = chartMod.parseDateAndMinuteOfDay("not-a-valid-date");
      const numSec = chartMod.parseDateAndMinuteOfDay(1700000000);
      const numMs = chartMod.parseDateAndMinuteOfDay(1700000000000);
      const numNaN = chartMod.parseDateAndMinuteOfDay(NaN);
      const boolInput = chartMod.parseDateAndMinuteOfDay(true);

      const obsDefault = chartMod.filterObservationWindow(
        [{ time: "2025-01-10T00:00:00Z" }, { time: "bad" }],
        "time",
      );
      const obsZeroDays = chartMod.filterObservationWindow(
        [{ time: "2025-01-10T00:00:00Z" }],
        "time",
        0,
      );
      const obsAllInvalid = chartMod.filter14DayWindow(
        [{ time: "bad" }],
        "time",
      );

      // 5. renderHistogramSvg, renderCgmCard valid wear (>=70%), observationDays=0, detached onchange, invalid glucose rows
      const emptyHist = chartMod.renderHistogramSvg([]);
      const highWearRows: Array<{
        time: string;
        glucose: number | string | null;
      }> = Array.from({ length: 250 }, (_, idx) => ({
        time: `2025-01-10T${String(Math.floor(idx / 12)).padStart(2, "0")}:${String((idx % 12) * 5).padStart(2, "0")}:00Z`,
        glucose: idx % 2 === 0 ? 120 : "130.5",
      }));
      highWearRows.push({
        time: "2025-01-10T23:55:00Z",
        glucose: "bad",
      });
      highWearRows.push({
        time: "2025-01-10T23:56:00Z",
        glucose: null,
      });
      const validWearCard = chartMod.renderCgmCard(highWearRows, 0);
      const unattachedSelect =
        validWearCard?.querySelector<HTMLSelectElement>(".agp-window-select");
      if (unattachedSelect && unattachedSelect.onchange) {
        unattachedSelect.value = "14";
        unattachedSelect.onchange.call(unattachedSelect, new Event("change"));
      }
      const nullGlucoseCard = chartMod.renderCgmCard([
        { time: "2025-01-10T00:00:00Z", glucose: "invalid" },
      ]);

      return {
        detEmptyKeys: Object.keys(detEmpty).length,
        detGenericCol: detGenericValid.glucoseCol,
        detGenericLow: detGenericLow.glucoseCol,
        detGenericHigh: detGenericHigh.glucoseCol,
        detGenericNaN: detGenericNaN.glucoseCol,
        tirSingleCount: tirSingleMmol.count,
        dnDayCount: dnMmol.day.count,
        pEmpty,
        agpNull,
        agpMixedCount: agpMixed?.totalReadings,
        wearEmptyPct: wearEmpty.wearPercentage,
        wearZeroExpectedPct: wearZeroExpected.wearPercentage,
        wearZeroEpochDays: wearZeroEpoch.calendarDays,
        dValidMin: dValid?.minuteOfDay,
        dInvalid,
        isoDateOnlyMin: isoDateOnly?.minuteOfDay,
        isoOffsetValid: Boolean(isoOffsetValid),
        isoOffsetInvalid: Boolean(isoOffsetInvalid),
        cdiscDateOnlyMin: cdiscDateOnly?.minuteOfDay,
        cdiscUnknownMonth: Boolean(cdiscUnknownMonth),
        cdiscOffsetValid: Boolean(cdiscOffsetValid),
        cdiscOffsetInvalid: Boolean(cdiscOffsetInvalid),
        slashUsDateOnlyMin: slashUsDateOnly?.minuteOfDay,
        slashEuWithTimeZ: Boolean(slashEuWithTimeZ),
        slashOffsetValid: Boolean(slashOffsetValid),
        slashOffsetInvalid: Boolean(slashOffsetInvalid),
        natWithTz: Boolean(natWithTz),
        natWithZ: Boolean(natWithZ),
        natLocal: Boolean(natLocal),
        natInvalid,
        numSec: Boolean(numSec),
        numMs: Boolean(numMs),
        numNaN,
        boolInput,
        obsDefaultLen: obsDefault.length,
        obsZeroDaysLen: obsZeroDays.length,
        obsAllInvalidLen: obsAllInvalid.length,
        emptyHistTag: emptyHist.tagName.toLowerCase(),
        validWearCardHasSuccess: Boolean(
          validWearCard?.querySelector(".cgm-wear-badge.cgm-badge-success"),
        ),
        nullGlucoseCard,
      };
    });

    expect(edgeCheck.detEmptyKeys).toBe(0);
    expect(edgeCheck.detGenericCol).toBe("value");
    expect(edgeCheck.detGenericLow).toBeUndefined();
    expect(edgeCheck.detGenericHigh).toBeUndefined();
    expect(edgeCheck.detGenericNaN).toBeUndefined();
    expect(edgeCheck.tirSingleCount).toBe(1);
    expect(edgeCheck.dnDayCount).toBe(2);
    expect(edgeCheck.pEmpty).toBe(0);
    expect(edgeCheck.agpNull).toBeNull();
    expect(edgeCheck.agpMixedCount).toBe(1);
    expect(edgeCheck.wearEmptyPct).toBe(0);
    expect(edgeCheck.wearZeroExpectedPct).toBe(0);
    expect(edgeCheck.wearZeroEpochDays).toBe(1);
    expect(edgeCheck.dValidMin).toBe(510);
    expect(edgeCheck.dInvalid).toBeNull();
    expect(edgeCheck.emptyHistTag).toBe("svg");
    expect(edgeCheck.validWearCardHasSuccess).toBe(true);
    expect(edgeCheck.nullGlucoseCard).toBeNull();
  });

  /**
   * Tests multi-database selector (#db-select) switching and manual server session sync (#sync-sessions-btn).
   */
  test("Switches active DuckDB database via #db-select and syncs sessions with server via #sync-sessions-btn", async ({
    page,
  }) => {
    const schemaDbParams: string[] = [];

    await page.route("**/api/databases", async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        json: {
          databases: [
            { name: "t1d.duckdb", path: "t1d.duckdb", size_bytes: 20480 },
            {
              name: "secondary_cohort.duckdb",
              path: "secondary_cohort.duckdb",
              size_bytes: 40960,
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
              columns: [{ name: "id", type: "INTEGER" }],
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

    // Exercise all remaining ChatState persistence, server sync, tombstone, and storage branches in browser runtime
    const stateCoverageCheck = await page.evaluate(async () => {
      const stateMod = await import("/src/state.ts");

      // 1. Legacy HTML entity decoding & parsePersistedChatState
      const decoded = stateMod.decodeLegacyHtmlEntities(
        "&#39;sum&#39; &gt; 0 &amp; &quot;ok&quot; &#x2F;",
      );
      const parsedLegacy = stateMod.parsePersistedChatState(
        JSON.stringify({
          chats: [
            {
              id: "c1",
              title: "Legacy",
              model: "sql",
              messages: [{ role: "assistant", content: "&#39;ok&#39;" }],
            },
            { id: "c2", title: "No messages", model: "sql" },
          ],
        }),
      );

      // 2. Empty object and corrupted JSON in localStorage hydration
      localStorage.setItem("t1d_analytics_chats", JSON.stringify({}));
      const emptyHydrated = new stateMod.ChatState();
      const nullActive = emptyHydrated.getActiveChat();

      localStorage.setItem("t1d_analytics_chats", "{invalid-json");
      const corruptedHydrated = new stateMod.ChatState();

      // 3. setAvailableDbs when currentDb is empty, and setServerSyncEnabled
      emptyHydrated.currentDb = "";
      emptyHydrated.setAvailableDbs(
        [{ name: "db1", path: "db1.duckdb", size_bytes: 512 }],
        "db1.duckdb",
      );
      emptyHydrated.setServerSyncEnabled(true);

      // 4. duplicateChat when original is not found, deleteChat when activeChat falls back to remaining chat
      const dupMissing = emptyHydrated.duplicateChat("non-existent");
      const cA = emptyHydrated.createChat("Chat A");
      const cB = emptyHydrated.createChat("Chat B");
      const origFetch = window.fetch;
      window.fetch = () => Promise.reject(new Error("Delete network error"));
      emptyHydrated.deleteChat(cB.id);
      emptyHydrated.renameChat(cA.id, "Renamed A");
      await new Promise((r) => setTimeout(r, 10));

      // 5. syncWithServer error & merge branches:
      // 5a: HTTP 503 (!resp.ok) -> catch fallback
      window.fetch = async () =>
        new Response(JSON.stringify({ error: "Unavailable" }), { status: 503 });
      const syncFail = await emptyHydrated.syncWithServer();

      // 5b: Object with sessions array, temporary local chat, titleModified=true (both matching and differing),
      // deletedChatIds tombstone with rejected DELETE, remote session without updated_at/messages,
      // existing local chat with fewer messages & remote title update, and null activeChatId
      const syncState = new stateMod.ChatState();
      const tempChat = syncState.createChat("Temp", true);
      const modDiffChat = syncState.createChat("Local Old");
      modDiffChat.id = "mod-diff";
      modDiffChat.titleModified = true;
      modDiffChat.title = "Local Modified";

      const modSameChat = syncState.createChat("Same Title");
      modSameChat.id = "mod-same";
      modSameChat.titleModified = true;

      const mergeTarget = syncState.createChat("Initial Local Title");
      mergeTarget.id = "merge-target";

      const localAhead = syncState.createChat("Ahead Local");
      localAhead.id = "local-ahead";
      syncState.addMessageToActiveChat({ role: "user", content: "m1" });
      syncState.addMessageToActiveChat({ role: "assistant", content: "m2" });

      const localNullRemoteMsgs = syncState.createChat(
        "Null Remote Msgs",
        true,
      );
      localNullRemoteMsgs.id = "local-null-remote-msgs";

      syncState.deletedChatIds.add("tombstone-id");
      syncState.activeChatId = null;

      window.fetch = async (_input: RequestInfo | URL, init?: RequestInit) => {
        if (init?.method === "DELETE") {
          return Promise.reject(new Error("Tombstone delete failed"));
        }
        if (init?.method === "POST") {
          return new Response(JSON.stringify({ status: "ok" }), {
            status: 200,
          });
        }
        return new Response(
          JSON.stringify({
            sessions: [
              null,
              {},
              {
                session_id: "tombstone-id",
                title: "Deleted",
                created_at: "",
                updated_at: "",
                messages: [],
              },
              {
                session_id: "mod-diff",
                title: "Remote Old",
                created_at: "",
                updated_at: "",
                messages: [{ role: "user", content: "m1" }],
              },
              {
                session_id: "mod-same",
                title: "Same Title",
                created_at: "",
                updated_at: "",
                messages: [],
              },
              {
                session_id: "merge-target",
                title: "Updated Remote Title",
                created_at: "",
                updated_at: "2025-01-02T00:00:00Z",
                messages: [{ role: "user", content: "remote msg" }],
              },
              {
                session_id: "local-ahead",
                title: "Ahead Local",
                created_at: "",
                updated_at: "",
                messages: [{ role: "user", content: "m1" }],
              },
              {
                session_id: "local-null-remote-msgs",
                title: "Null Remote Msgs",
                created_at: "",
                updated_at: "",
              },
              {
                session_id: "new-remote-no-msgs",
                title: "New Remote",
                created_at: "",
                updated_at: "",
              },
            ],
          }),
          { status: 200 },
        );
      };
      const syncOk1 = await syncState.syncWithServer();
      await new Promise((r) => setTimeout(r, 10));

      // 5c: Object without sessions array (empty sessionsList) and empty local chats with null activeChatId
      const emptySyncState = new stateMod.ChatState();
      emptySyncState.chats = [];
      emptySyncState.activeChatId = null;
      window.fetch = async () =>
        new Response(JSON.stringify({ unexpected: true }), { status: 200 });
      const syncOk2 = await emptySyncState.syncWithServer();
      window.fetch = origFetch;

      // 6. getAllApiKeys & configured Mozilla any-llm models
      emptyHydrated.setApiKey("openai", "sk-test");
      const allKeys = emptyHydrated.getAllApiKeys();
      emptyHydrated.removeApiKey("openai");
      emptyHydrated.setConfiguredModels("groq/llama-3, mistral/codestral");
      const cfgModels = emptyHydrated.getConfiguredModels();
      emptyHydrated.setConfiguredModels("   ");
      emptyHydrated.removeConfiguredModels();

      // 7. safeStorageGet, safeStorageSet, safeStorageRemove error and undefined localStorage branches
      const origGetItem = Storage.prototype.getItem;
      const origSetItem = Storage.prototype.setItem;
      const origRemoveItem = Storage.prototype.removeItem;
      Storage.prototype.getItem = () => {
        throw new Error("getItem blocked");
      };
      Storage.prototype.setItem = () => {
        throw new Error("setItem blocked");
      };
      Storage.prototype.removeItem = () => {
        throw new Error("removeItem blocked");
      };
      const errGet = stateMod.safeStorageGet("k");
      stateMod.safeStorageSet("k", "v");
      stateMod.safeStorageRemove("k");
      Storage.prototype.getItem = origGetItem;
      Storage.prototype.setItem = origSetItem;
      Storage.prototype.removeItem = origRemoveItem;

      const lsDescriptor = Object.getOwnPropertyDescriptor(
        window,
        "localStorage",
      );
      Object.defineProperty(window, "localStorage", {
        value: undefined,
        configurable: true,
      });
      const undefGet = stateMod.safeStorageGet("k");
      stateMod.safeStorageSet("k", "v");
      stateMod.safeStorageRemove("k");
      if (lsDescriptor) {
        Object.defineProperty(window, "localStorage", lsDescriptor);
      }

      return {
        decoded,
        parsedMsg: parsedLegacy.chats?.[0]?.messages?.[0]?.content,
        nullActive,
        corruptedCount: corruptedHydrated.chats.length,
        dupMissing,
        syncFail,
        syncOk1,
        syncOk2,
        tempStillTemp: tempChat.isTemporary,
        hasOpenAiKey: Boolean(allKeys.openai),
        cfgModelsLen: cfgModels.length,
        errGet,
        undefGet,
      };
    });

    expect(stateCoverageCheck.decoded).toBe(`'sum' > 0 & "ok" /`);
    expect(stateCoverageCheck.parsedMsg).toBe("'ok'");
    expect(stateCoverageCheck.nullActive).toBeNull();
    expect(stateCoverageCheck.corruptedCount).toBe(0);
    expect(stateCoverageCheck.dupMissing).toBeNull();
    expect(stateCoverageCheck.syncFail).toBe(false);
    expect(stateCoverageCheck.syncOk1).toBe(true);
    expect(stateCoverageCheck.syncOk2).toBe(true);
    expect(stateCoverageCheck.tempStillTemp).toBe(true);
    expect(stateCoverageCheck.hasOpenAiKey).toBe(true);
    expect(stateCoverageCheck.cfgModelsLen).toBe(2);
    expect(stateCoverageCheck.errGet).toBeNull();
    expect(stateCoverageCheck.undefGet).toBeNull();
  });
});
