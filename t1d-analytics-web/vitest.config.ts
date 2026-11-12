/**
 * @file vitest.config.ts
 * @module vitest.config
 * Vitest configuration and 100% per-file Istanbul coverage settings for application source modules.
 */

import { defineConfig, type UserConfig } from "vitest/config";

/**
 * Constructs the Vitest configuration enforcing 100% per-file coverage across application source files.
 * @returns {UserConfig} Vitest configuration object.
 */
export function createVitestConfig(): UserConfig {
  return defineConfig({
    define: {
      __BUILD_TIMESTAMP_UTC__: JSON.stringify("2026-10-29T00:00:00Z UTC"),
    },
    build: {
      target: "esnext",
    },
    test: {
      environment: "jsdom",
      setupFiles: ["./tests/setup.ts"],
      exclude: ["node_modules", "dist", "tests-e2e/**", "tests-e2e-full/**"],
      coverage: {
        provider: "istanbul",
        all: true,
        include: ["src/**/*.ts"],
        exclude: ["src/**/*.d.ts"],
        reporter: ["text", "html", "lcov", "json"],
        thresholds: {
          lines: 100,
          functions: 100,
          branches: 100,
          statements: 100,
          perFile: true,
        },
      },
    },
  });
}

/**
 * Default Vitest configuration export.
 */
const vitestConfig: UserConfig = createVitestConfig();

export default vitestConfig;
