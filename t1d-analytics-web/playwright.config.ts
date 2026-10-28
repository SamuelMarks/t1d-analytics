/**
 * @file playwright.config.ts
 * @module playwright.config
 * Standard Playwright end-to-end test suite configuration.
 */

import {
  defineConfig,
  devices,
  type PlaywrightTestConfig,
} from "@playwright/test";

/**
 * Construct the standard Playwright end-to-end test configuration.
 *
 * @returns Resolved Playwright test configuration.
 */
export function createPlaywrightConfig(): PlaywrightTestConfig {
  return defineConfig({
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
}

/**
 * Playwright configuration for mocked end-to-end browser and accessibility testing.
 */
const playwrightConfig: PlaywrightTestConfig = createPlaywrightConfig();

export default playwrightConfig;
