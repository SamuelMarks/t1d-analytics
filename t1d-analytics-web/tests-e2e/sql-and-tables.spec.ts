/**
 * @file sql-and-tables.spec.ts
 * @module tests-e2e/sql-and-tables.spec
 * End-to-end Playwright tests for SQL classification, result rendering, table actions,
 * client-side CSV/JSON exports, markdown rendering, and DOMPurify XSS sanitization.
 */

import { test, expect } from "./base-test";

test.describe("SQL Detection, Result Rendering, Exports & Sanitization", () => {
  /**
   * Tests automatic SQL classification (SELECT, WITH, PRAGMA, comments) versus natural language phrases.
   */
  test("Auto-classifies read-only SQL statements while keeping natural language in LLM mode", async ({
    page,
  }) => {
    const capturedModels: string[] = [];
    await page.route("**/api/chat", async (route) => {
      const postData = route.request().postDataJSON() as {
        message: string;
        model: string;
      };
      capturedModels.push(postData.model);
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        json: {
          content: "Result returned.",
          sqlQuery: postData.message,
          sqlResult: [{ metric: 42 }],
        },
      });
    });

    await page.goto("/");
    await page.selectOption("#model-select", "gemma4");

    // 1. Leading SQL comment + WITH CTE should auto-classify as "sql"
    await page.fill(
      "#chat-input",
      [
        "-- Cohort CTE",
        "((WITH cte AS (SELECT 1 AS metric) SELECT * FROM cte))",
      ].join(String.fromCharCode(10)),
    );
    await page.click("#send-btn");
    await expect(page.locator(".message.assistant")).toHaveCount(1);
    expect(capturedModels[0]).toBe("sql");

    // 2. Natural language starting with "show me" should remain "gemma4"
    await page.selectOption("#model-select", "gemma4");
    await page.fill("#chat-input", "show me the average HbA1c for patients");
    await page.click("#send-btn");
    await expect(page.locator(".message.assistant")).toHaveCount(2);
    expect(capturedModels[1]).toBe("gemma4");
  });

  /**
   * Tests empty SQL result rendering (.sql-empty), multi-row table rendering, NULL cell formatting,
   * and client-side CSV and JSON download triggers.
   */
  test("Renders empty SQL results, multi-row tables with NULLs, and triggers CSV and JSON downloads", async ({
    page,
  }) => {
    let callCount = 0;
    await page.route("**/api/chat", async (route) => {
      callCount++;
      if (callCount === 1) {
        await route.fulfill({
          status: 200,
          contentType: "application/json",
          json: {
            content: "backend.literalSql",
            sqlQuery: "SELECT * FROM empty_cohort;",
            sqlResult: [],
          },
        });
      } else {
        await route.fulfill({
          status: 200,
          contentType: "application/json",
          json: {
            content: "backend.literalSql",
            sqlQuery: "SELECT patient_id, note, hba1c FROM patients;",
            sqlResult: [
              {
                patient_id: "PT-01",
                note: "=SUM(A1:A2)",
                hba1c: 7.2,
                delta: "+1.2",
                meta: { unit: "pct" },
              },
              {
                patient_id: "PT-02",
                note: null,
                hba1c: 6.5,
                delta: "-0.4",
                meta: "+CMD",
              },
            ],
          },
        });
      }
    });

    await page.goto("/");
    await page.selectOption("#model-select", "sql");

    // 1. Empty SQL result
    await page.fill("#chat-input", "SELECT * FROM empty_cohort;");
    await page.click("#send-btn");
    await expect(page.locator(".sql-empty")).toBeVisible();
    await expect(page.locator(".sql-empty")).toContainText("No rows returned");

    // 2. Multi-row SQL result with NULL and formula-injection string
    await page.fill(
      "#chat-input",
      "SELECT patient_id, note, hba1c FROM patients;",
    );
    await page.click("#send-btn");

    const sqlTable = page.locator(".sql-table").first();
    await expect(sqlTable).toBeVisible();
    await expect(sqlTable.locator("th")).toHaveCount(5);
    await expect(sqlTable.locator("tbody tr")).toHaveCount(2);
    await expect(
      sqlTable.locator("tbody tr").nth(1).locator("td").nth(1),
    ).toHaveText("NULL");

    // 3. Trigger CSV download and verify filename
    const csvDownloadPromise = page.waitForEvent("download");
    await page.locator(".export-csv-btn").first().click();
    const csvDownload = await csvDownloadPromise;
    expect(csvDownload.suggestedFilename()).toBe("query_results.csv");

    // 4. Trigger JSON download and verify filename
    const jsonDownloadPromise = page.waitForEvent("download");
    await page.locator(".export-json-btn").first().click();
    const jsonDownload = await jsonDownloadPromise;
    expect(jsonDownload.suggestedFilename()).toBe("query_results.json");
  });

  /**
   * Tests SQL code block copy button, play button (▶), and refresh button (↻) actions.
   */
  test("Executes SQL via Play and Refresh buttons and copies SQL to clipboard", async ({
    page,
    context,
  }) => {
    await context.grantPermissions(["clipboard-read", "clipboard-write"]);

    let executeCalls = 0;
    await page.route("**/api/execute-sql", async (route) => {
      executeCalls++;
      if (executeCalls === 1) {
        // 1st call (Play button): scalar result -> formatted inline in assistant message content
        await route.fulfill({
          status: 200,
          contentType: "application/json",
          json: {
            sqlResult: [{ refreshed_count: 10 }],
          },
        });
      } else {
        // 2nd call (Refresh button): multi-column result -> rendered as .sql-table
        await route.fulfill({
          status: 200,
          contentType: "application/json",
          json: {
            sqlResult: [{ id: 1, refreshed_count: 20 }],
          },
        });
      }
    });

    await page.goto("/");
    await page.selectOption("#model-select", "sql");
    await page.fill("#chat-input", "SELECT 10 AS refreshed_count;");
    await page.click("#send-btn");

    await expect(page.locator(".message.assistant")).toHaveCount(1);

    // Click Copy button on user SQL query header
    const copyBtn = page.locator(".copy-sql-btn").first();
    await copyBtn.click();
    const clipboardText = await page.evaluate(() =>
      navigator.clipboard.readText(),
    );
    expect(clipboardText).toContain("SELECT 10 AS refreshed_count;");

    // Click Play button on user message -> appends a new assistant result message with scalar formatted text
    const playBtn = page.locator(".play-sql-btn").first();
    await playBtn.click();
    await expect(page.locator(".message.assistant")).toHaveCount(2);
    await expect(page.locator(".message.assistant").nth(1)).toContainText("10");

    // Click Refresh button on assistant SQL message -> updates message in-place with multi-column table
    const refreshBtn = page
      .locator('.message.assistant button[aria-label="Refresh Query"]')
      .first();
    await refreshBtn.click();
    await expect(
      page.locator(".message.assistant").nth(0).locator(".sql-table td").nth(1),
    ).toHaveText("20");
  });

  /**
   * Tests markdown rendering and DOMPurify XSS sanitization in assistant messages.
   */
  test("Sanitizes malicious XSS payloads in assistant markdown while preserving safe formatting and SQL blocks", async ({
    page,
  }) => {
    await page.route("**/api/chat", async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        json: {
          content: [
            "### Clinical Summary",
            "Here is **bold** text and an inline `PtID`.",
            "<script>window.__xssExecuted = true;</script>",
            '<img src="x" onerror="window.__xssImg = true" />',
            "```sql",
            "SELECT PtID, HbA1c FROM hba1c_labs LIMIT 5;",
            "```",
          ].join(String.fromCharCode(10, 10)),
          sqlQuery: null,
          sqlResult: null,
        },
      });
    });

    await page.goto("/");
    await page.selectOption("#model-select", "gemma4");
    await page.fill("#chat-input", "Summarize HbA1c data");
    await page.click("#send-btn");

    const assistantMsg = page.locator(".message.assistant").first();
    await expect(assistantMsg).toBeVisible();
    await expect(assistantMsg.locator("h3")).toHaveText("Clinical Summary");
    await expect(assistantMsg.locator("strong")).toHaveText("bold");

    // Verify script tag was stripped and onerror attribute was removed
    await expect(assistantMsg.locator("script")).toHaveCount(0);
    const imgOnerror = await assistantMsg
      .locator("img")
      .getAttribute("onerror");
    expect(imgOnerror).toBeNull();

    // Verify fenced SQL block in markdown received copy + play buttons
    await expect(
      assistantMsg.locator(".sql-query-container .play-sql-btn"),
    ).toBeVisible();
  });
});
