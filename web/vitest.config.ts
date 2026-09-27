/**
 * @file vitest.config.ts
 * Vitest configuration and code coverage settings.
 */

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
      all: true,
      reporter: ["text", "html", "lcov", "json"],
      include: [
        "src/**/*.ts",
        "vite.config.ts",
        "vitest.config.ts",
        "playwright.config.ts",
        "playwright.full.config.ts",
      ],
      exclude: [
        "docs/**",
        "coverage/**",
        "tests/**",
        "tests-e2e/**",
        "tests-e2e-full/**",
      ],
    },
  },
});
