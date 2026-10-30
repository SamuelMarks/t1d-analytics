/**
 * @file streaming-and-errors.spec.ts
 * @module tests-e2e/streaming-and-errors.spec
 * End-to-end Playwright tests for SSE chat streaming, batch fallback,
 * structured backend error translation, network error recovery, and keyboard submission.
 */

import { test, expect } from "./base-test";

test.describe("SSE Streaming, Batch Fallback & Error Recovery", () => {
  /**
   * Tests SSE streaming (/api/chat/stream) with token accumulation, execution progress, and terminal done event.
   */
  test("Streams SSE tokens incrementally and renders terminal SQL results when Streaming Mode is enabled", async ({
    page,
  }) => {
    const sseSep = String.fromCharCode(10, 10);
    await page.route("**/api/chat/stream", async (route) => {
      const sseBody = [
        `data: ${JSON.stringify({ event: "token", token: "SELECT " })}${sseSep}`,
        `data: ${JSON.stringify({ event: "token", token: "COUNT(*) AS total FROM patients;" })}${sseSep}`,
        `data: ${JSON.stringify({ event: "execution_progress", status: "Executing SQL query..." })}${sseSep}`,
        `data: ${JSON.stringify({
          event: "done",
          content: "Found 24 patients in cohort.",
          sqlQuery: "SELECT COUNT(*) AS total FROM patients;",
          sqlResult: [{ total: 24 }],
        })}${sseSep}`,
      ].join("");

      await route.fulfill({
        status: 200,
        headers: {
          "Content-Type": "text/event-stream",
          "Cache-Control": "no-cache",
        },
        body: sseBody,
      });
    });

    await page.goto("/");

    // Enable streaming mode via the header toggle button
    const streamBtn = page.locator("#streaming-toggle-btn");
    await expect(streamBtn).toHaveAttribute("aria-pressed", "false");
    await streamBtn.click();
    await expect(streamBtn).toHaveAttribute("aria-pressed", "true");

    // Verify streaming preference is persisted in localStorage
    const persistedStream = await page.evaluate(() => {
      const raw = localStorage.getItem("t1d_analytics_chats");
      if (!raw) return false;
      const parsed = JSON.parse(raw) as { streamingMode?: boolean };
      return Boolean(parsed.streamingMode);
    });
    expect(persistedStream).toBe(true);

    await page.selectOption("#model-select", "gemma4");
    await page.fill("#chat-input", "How many patients are in the cohort?");
    await page.click("#send-btn");

    const assistantMsg = page.locator(".message.assistant").first();
    await expect(assistantMsg).toBeVisible();
    await expect(assistantMsg).toContainText("Found 24 patients in cohort.");
    await expect(assistantMsg).toContainText("24");
  });

  /**
   * Tests automatic fallback from /api/chat/stream to batch /api/chat when stream endpoint fails.
   */
  test("Falls back gracefully to batch /api/chat when /api/chat/stream returns HTTP 500", async ({
    page,
  }) => {
    let streamCalled = false;
    let batchCalled = false;

    await page.route("**/api/chat/stream", async (route) => {
      streamCalled = true;
      await route.fulfill({
        status: 500,
        contentType: "application/json",
        json: { detail: "Stream unavailable" },
      });
    });

    await page.route("**/api/chat", async (route) => {
      batchCalled = true;
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        json: {
          content: "Batch fallback succeeded.",
          sqlQuery: "SELECT 1 AS fallback_ok;",
          sqlResult: [{ fallback_ok: 1 }, { fallback_ok: 2 }],
        },
      });
    });

    await page.goto("/");
    await page.click("#streaming-toggle-btn");
    await page.fill("#chat-input", "Run query with stream fallback");
    await page.click("#send-btn");

    const assistantMsg = page.locator(".message.assistant").first();
    await expect(assistantMsg).toContainText("Batch fallback succeeded.");
    expect(streamCalled).toBe(true);
    expect(batchCalled).toBe(true);
  });

  /**
   * Tests structured backend error translation and network failure recovery.
   */
  test("Translates structured backend error codes and recovers cleanly after network failures", async ({
    page,
  }) => {
    let attempt = 0;
    await page.route("**/api/chat", async (route) => {
      attempt++;
      if (attempt === 1) {
        await route.fulfill({
          status: 403,
          contentType: "application/json",
          json: {
            detail: {
              error_code: "backend.invalidDatabasePath",
              params: { path: "/etc/passwd" },
            },
          },
        });
      } else if (attempt === 2) {
        await route.fulfill({
          status: 400,
          contentType: "application/json",
          json: {
            detail: "backend.invalidTable",
          },
        });
      } else if (attempt === 3) {
        await route.fulfill({
          status: 400,
          contentType: "application/json",
          json: {
            detail: "Custom literal backend error",
          },
        });
      } else if (attempt === 4) {
        await route.fulfill({
          status: 400,
          contentType: "application/json",
          json: {
            detail: {
              error_code: "custom.unmappedCode",
              params: { reason: "test" },
            },
          },
        });
      } else if (attempt === 5) {
        await route.fulfill({
          status: 422,
          contentType: "application/json",
          json: {
            detail: 422,
          },
        });
      } else if (attempt === 6) {
        await route.fulfill({
          status: 503,
          contentType: "application/json",
          json: {},
        });
      } else if (attempt === 7) {
        await route.abort("failed");
      } else {
        await route.fulfill({
          status: 200,
          contentType: "application/json",
          json: {
            content: "Recovered after error.",
            sqlQuery: "SELECT 1;",
            sqlResult: [{ ok: null }],
          },
        });
      }
    });

    await page.goto("/");

    // 1. Structured 403 invalidDatabasePath error
    await page.fill("#chat-input", "First attempt");
    await page.click("#send-btn");
    const errAlert1 = page.locator('.message.assistant [role="alert"]').first();
    await expect(errAlert1).toBeVisible();
    await expect(errAlert1).toContainText(
      "Invalid or unauthorized database path: /etc/passwd",
    );

    // 2. Translated string detail
    await page.fill("#chat-input", "Second attempt");
    await page.click("#send-btn");
    await expect(
      page.locator('.message.assistant [role="alert"]').nth(1),
    ).toContainText("Invalid table name");

    // 3. Untranslated string detail
    await page.fill("#chat-input", "Third attempt");
    await page.click("#send-btn");
    await expect(
      page.locator('.message.assistant [role="alert"]').nth(2),
    ).toContainText("Custom literal backend error");

    // 4. Unmapped structured error_code
    await page.fill("#chat-input", "Fourth attempt");
    await page.click("#send-btn");
    await expect(
      page.locator('.message.assistant [role="alert"]').nth(3),
    ).toContainText('custom.unmappedCode: {"reason":"test"}');

    // 5. Numeric detail
    await page.fill("#chat-input", "Fifth attempt");
    await page.click("#send-btn");
    await expect(
      page.locator('.message.assistant [role="alert"]').nth(4),
    ).toContainText("422");

    // 6. Missing detail object (HTTP 503 fallback)
    await page.fill("#chat-input", "Sixth attempt");
    await page.click("#send-btn");
    await expect(
      page.locator('.message.assistant [role="alert"]').nth(5),
    ).toContainText("HTTP 503");

    // 7. Network abort error
    await page.fill("#chat-input", "Seventh attempt");
    await page.click("#send-btn");
    const errAlert7 = page.locator('.message.assistant [role="alert"]').nth(6);
    await expect(errAlert7).toBeVisible();

    // Verify input is re-enabled so user can retry
    await expect(page.locator("#chat-input")).toBeEnabled();

    // 8. Successful recovery on eighth attempt (with scalar NULL result)
    await page.fill("#chat-input", "Eighth attempt");
    await page.click("#send-btn");
    await expect(page.locator(".message.assistant").nth(7)).toContainText(
      "Recovered after error.",
    );
  });

  /**
   * Tests Shift+Enter multiline editing versus Enter form submission and example query chip click.
   */
  test("Supports Shift+Enter newline insertion, Enter submission, and example query chips", async ({
    page,
  }) => {
    await page.goto("/");

    // Click an example query chip in the empty state
    const firstChip = page.locator(".empty-state .chip").first();
    await expect(firstChip).toBeVisible();
    await firstChip.click();

    const chatInput = page.locator("#chat-input");
    const chipValue = await chatInput.inputValue();
    expect(chipValue.length).toBeGreaterThan(10);

    // Clear and test Shift+Enter vs Enter
    await chatInput.fill("SELECT 1");
    await chatInput.press("Shift+Enter");
    await chatInput.type("FROM patients;");
    const multilineVal = await chatInput.inputValue();
    expect(multilineVal).toContain(String.fromCharCode(10));
    await expect(page.locator(".message.user")).toHaveCount(0);

    // Press Enter without Shift to submit
    await chatInput.press("Enter");
    await expect(page.locator(".message.user")).toHaveCount(1);
    await expect(page.locator(".message.assistant")).toHaveCount(1);
  });
});
