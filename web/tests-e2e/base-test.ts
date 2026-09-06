/**
 * @fileoverview Custom Playwright test fixture providing NYC code coverage collection
 * and default network endpoint mocking for backend API calls.
 */

import { test as baseTest, expect, type Page } from "@playwright/test";
import fs from "fs";
import path from "path";
import crypto from "crypto";

/**
 * Extended Playwright test runner with automatic coverage output and backend endpoint mocks.
 */
export const test = baseTest.extend<{ page: Page }>({
  page: async ({ page }, use) => {
    // Provide default mocks for backend status and schema endpoints so that
    // mocked E2E tests run reliably without an active FastAPI backend process.
    await page.route("**/api/status", async (route) => {
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
            table_count: 1,
            has_initial_data: true,
          },
          ollama: {
            accessible: true,
          },
        },
      });
    });

    await page.route("**/api/schema", async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        json: {
          tables: [],
          database: {
            status_code: "healthy",
            configured_path: "t1d.duckdb",
            exists: true,
            connected: true,
            table_count: 0,
            has_initial_data: true,
          },
        },
      });
    });

    await page.route("**/api/models", async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        json: {
          models: [{ name: "gemma4" }],
        },
      });
    });

    await page.route("**/api/chat", async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        json: {
          content: "Query executed.",
          sqlQuery: "SELECT 1;",
          sqlResult: [{ count: 1 }],
        },
      });
    });

    await page.route("**/api/execute-sql", async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        json: {
          sqlResult: [{ id: 1 }],
        },
      });
    });

    await page.route("**/api/table/*", async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        json: {
          rows: [],
          total: 0,
        },
      });
    });

    await use(page);

    const coverage = await page.evaluate(() => (window as any).__coverage__);
    if (coverage) {
      const nycPath = path.join(process.cwd(), ".nyc_output");
      if (!fs.existsSync(nycPath)) {
        fs.mkdirSync(nycPath, { recursive: true });
      }
      fs.writeFileSync(
        path.join(
          nycPath,
          `coverage-${crypto.randomBytes(4).toString("hex")}.json`,
        ),
        JSON.stringify(coverage),
      );
    }
  },
});

export { expect };
