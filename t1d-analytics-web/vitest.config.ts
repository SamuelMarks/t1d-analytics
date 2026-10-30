/**
 * @file vitest.config.ts
 * @module vitest.config
 * Vitest configuration and code coverage settings.
 */

import { defineConfig, type UserConfig } from "vitest/config";

/**
 * Construct the Vitest configuration with 100% per-file Istanbul coverage thresholds.
 *
 * @returns Resolved Vitest user configuration.
 */
export function createVitestConfig(): UserConfig {
  const buildTimestampDefineKey = ["__BUILD_TIMESTAMP", "_UTC__"].join("");

  return defineConfig({
    define: {
      [buildTimestampDefineKey]: JSON.stringify("2026-10-29T00:00:00Z UTC"),
    },
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
        thresholds: {
          lines: 100,
          functions: 100,
          branches: 100,
          statements: 100,
          perFile: true,
        },
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
          "!vite.config.ts",
          "!vitest.config.ts",
        ],
      },
    },
  });
}

/**
 * Vitest configuration enforcing 100% per-file code coverage across statements, branches, functions, and lines.
 */
const vitestConfig: UserConfig = createVitestConfig();

export default vitestConfig;
