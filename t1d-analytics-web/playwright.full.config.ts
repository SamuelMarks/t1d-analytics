/**
 * @file playwright.full.config.ts
 * @module playwright.full.config
 * Full end-to-end matrix configuration with active backend instance.
 */

import {
  defineConfig,
  devices,
  type PlaywrightTestConfig,
} from "@playwright/test";

/**
 * Construct the full-stack Playwright end-to-end test configuration.
 *
 * @returns Resolved Playwright full-stack test configuration.
 */
export function createPlaywrightFullConfig(): PlaywrightTestConfig {
  return defineConfig({
    testDir: "./tests-e2e-full",
    fullyParallel: false,
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
        command:
          "python -m uvicorn t1d_analytics.api:app --port 8000 || python3 -m uvicorn t1d_analytics.api:app --port 8000",
        url: "http://127.0.0.1:8000/api/models",
        reuseExistingServer: true,
        timeout: 120000,
      },
      {
        command: "npm run dev",
        url: "http://localhost:5173",
        reuseExistingServer: true,
        timeout: 120000,
        env: {
          VITE_API_PROXY_TARGET: "http://127.0.0.1:8000",
        },
      },
    ],
  });
}

/**
 * Playwright configuration for full-stack end-to-end testing with live Python FastAPI backend and Vite dev server.
 */
const playwrightFullConfig: PlaywrightTestConfig = createPlaywrightFullConfig();

export default playwrightFullConfig;
