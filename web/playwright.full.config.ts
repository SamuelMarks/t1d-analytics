import { defineConfig, devices } from "@playwright/test";

export default defineConfig({
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
      command: "npm run dev",
      url: "http://localhost:5173",
      reuseExistingServer: true,
      timeout: 120000,
    },
    {
      command:
        "cd .. && T1D_DB_PATH=ci_test.duckdb uvicorn t1d_analytics.api:app --port 8000",
      url: "http://localhost:8000/api/models",
      reuseExistingServer: true,
      timeout: 120000,
    },
  ],
});
