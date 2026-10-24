/**
 * @file playwright.config.ts
 * @module playwright.config
 * Standard Playwright end-to-end test suite configuration.
 */

import { defineConfig, devices } from "@playwright/test";

/**
 * Playwright configuration for mocked end-to-end browser and accessibility testing.
 */
export default defineConfig({
  testDir: "./tests-e2e",
  fullyParallel: true,
  retries: 0,
  workers: 1,
  reporter: "list",
  use: {
    baseURL: "http://localhost:5173",
    trace: "on-first-retry",
  },
  projects: [
    {
      name: "chromium",
      use: {
        ...devices["Desktop Chrome"],
        channel: "chrome",
        colorScheme: "light",
      },
    },
  ],
  webServer: [
    {
      command: "npm run dev",
      url: "http://localhost:5173",
      reuseExistingServer: true,
      timeout: 120000,
    },
  ],
});
