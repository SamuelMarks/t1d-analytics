/**
 * @fileoverview Custom Playwright test fixture for full stack E2E testing
 * providing NYC code coverage collection.
 */

import { test as baseTest, expect, type Page } from "@playwright/test";
import fs from "fs";
import path from "path";
import crypto from "crypto";

/**
 * Extended Playwright test runner for full E2E testing with automatic coverage instrumentation.
 */
export const test = baseTest.extend<{ page: Page }>({
  page: async ({ page }, use) => {
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
