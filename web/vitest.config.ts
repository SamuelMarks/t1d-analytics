import { defineConfig } from "vitest/config";

export default defineConfig({
  build: {
    target: "esnext",
  },
  test: {
    environment: "jsdom",
    setupFiles: ["./tests/setup.ts"],
    exclude: [
      "node_modules",
      "dist",
      ".idea",
      ".git",
      ".cache",
      "tests-e2e/**",
      "tests-e2e-full/**",
    ],
    coverage: {
      provider: "istanbul",
      reporter: ["text", "html", "lcov", "json"],
      exclude: [
        "vitest.config.ts",
        "playwright.config.ts",
        "playwright.full.config.ts",
        "tests/**",
        "tests-e2e/**",
        "tests-e2e-full/**",
      ],
    },
  },
});
