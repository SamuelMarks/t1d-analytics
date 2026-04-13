import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    environment: "jsdom",
    exclude: [
      "node_modules",
      "dist",
      ".idea",
      ".git",
      ".cache",
      "tests-e2e/**",
    ],
    coverage: {
      provider: "v8",
      reporter: ["text", "html"],
      thresholds: {
        lines: 100,
        functions: 100,
        branches: 100,
        statements: 100,
      },
      exclude: [
        "src/main.ts",
        "vitest.config.ts",
        "playwright.config.ts",
        "tests/**",
      ], // Exclude entry point and config from coverage
    },
  },
});
