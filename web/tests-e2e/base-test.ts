import { test as baseTest, expect } from "@playwright/test";
import fs from "fs";
import path from "path";
import crypto from "crypto";

export const test = baseTest.extend({
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
