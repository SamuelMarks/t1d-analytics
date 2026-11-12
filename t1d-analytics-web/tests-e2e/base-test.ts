/**
 * @file base-test.ts
 * @module tests-e2e/base-test
 * Custom Playwright test fixture providing NYC code coverage collection
 * and compile-time type-checked default network mocks for backend API contracts.
 */

import { test as baseTest, expect, type Page } from "@playwright/test";
import crypto from "crypto";
import fs from "fs";
import path from "path";
import {
  type ChatTurnResponse,
  type ModelsResponsePayload,
  type ProvidersStatusResponsePayload,
  type StatusResponsePayload,
  type TableRowsResponsePayload,
} from "../src/api-types";
import {
  type DatabasesListApiResponse,
  type SchemaExplorerApiResponse,
} from "../src/controllers/schema-explorer-controller";
import { type IstanbulFileCoverage } from "../src/vite-env";

/**
 * Extended Playwright test runner with automatic coverage output and typed backend endpoint mocks.
 */
export const test = baseTest.extend<{ page: Page }>({
  page: async ({ page }, use) => {
    const defaultStatus: StatusResponsePayload = {
      status: "healthy",
      database: {
        status_code: "healthy",
        exists: true,
        accessible: true,
        table_count: 1,
        has_initial_data: true,
        message: "Healthy",
      },
      ollama: {
        accessible: true,
        available_models: ["gemma:2b"],
        message: "Ollama ready",
      },
    };

    await page.route("**/api/status*", async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        json: {
          ...defaultStatus,
          database: {
            ...defaultStatus.database,
            connected: true,
            configured_path: "t1d.duckdb",
          },
        },
      });
    });

    await page.route("**/api/templates*", async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        json: { templates: [] },
      });
    });

    const defaultProvidersStatus: ProvidersStatusResponsePayload = {
      providers: {
        ollama: {
          configured: true,
          source: "local",
        },
        openai: {
          configured: false,
          source: "none",
        },
        anthropic: {
          configured: false,
          source: "none",
        },
        google: {
          configured: false,
          source: "none",
        },
      },
    };

    await page.route("**/api/providers/status*", async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        json: defaultProvidersStatus,
      });
    });

    const defaultSchema: SchemaExplorerApiResponse = {
      tables: [],
      database: {
        status_code: "healthy",
        configured_path: "t1d.duckdb",
        exists: true,
        connected: true,
        table_count: 0,
        has_initial_data: true,
      },
    };

    await page.route("**/api/schema", async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        json: defaultSchema,
      });
    });

    const defaultModels: ModelsResponsePayload = {
      models: [{ name: "gemma:2b" }],
    };

    await page.route("**/api/models", async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        json: defaultModels,
      });
    });

    const defaultDatabases: DatabasesListApiResponse = {
      databases: [{ name: "t1d.duckdb", path: "t1d.duckdb", size_bytes: 1024 }],
      current_db: "t1d.duckdb",
    };

    await page.route("**/api/databases", async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        json: defaultDatabases,
      });
    });

    const defaultChatTurn: ChatTurnResponse = {
      content: "Query executed.",
      sqlQuery: "SELECT 1;",
      sqlResult: [{ count: 1 }],
    };

    await page.route("**/api/chat", async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        json: defaultChatTurn,
      });
    });

    const defaultSqlTurn: ChatTurnResponse = {
      sqlResult: [{ id: 1 }],
    };

    await page.route("**/api/execute-sql", async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        json: defaultSqlTurn,
      });
    });

    const defaultTableRows: TableRowsResponsePayload = {
      rows: [],
      total_pages: 1,
    };

    await page.route("**/api/table/**", async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        json: defaultTableRows,
      });
    });

    await use(page);

    const coverage: Record<string, IstanbulFileCoverage> | undefined =
      await page.evaluate(() => window.__coverage__);
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
