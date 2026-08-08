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
      // Start the Python FastAPI backend server required for full E2E tests.
      // Fallback to python3 if python is not available in the environment.
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
    },
  ],
});
