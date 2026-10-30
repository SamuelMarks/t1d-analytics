/**
 * @file vite.config.ts
 * @module vite.config
 * Vite development and production build configuration.
 */

import { defineConfig, type UserConfig } from "vite";
import istanbul from "vite-plugin-istanbul";

/**
 * Resolve the default process environment dictionary when running in Node, or an empty object in browser contexts.
 *
 * @returns Default environment variable dictionary.
 */
export function getDefaultViteEnv(): Record<string, string | undefined> {
  return typeof process !== "undefined" && process.env ? process.env : {};
}

/**
 * Construct the Vite development, preview, and build configuration.
 *
 * @param env - Environment variable mapping used to resolve API proxy and preview port settings.
 * @returns Resolved Vite user configuration.
 */
export function createViteConfig(
  env: Record<string, string | undefined> = getDefaultViteEnv(),
): UserConfig {
  const apiProxyTarget =
    env.VITE_API_PROXY_TARGET ||
    `http://127.0.0.1:${env.T1D_API_PORT || "8000"}`;
  const buildTimestampUtc =
    env.BUILD_TIMESTAMP_UTC ||
    new Date().toISOString().replace(/\.\d{3}Z$/, " UTC");

  const buildTimestampDefineKey = ["__BUILD_TIMESTAMP", "_UTC__"].join("");

  return defineConfig({
    define: {
      [buildTimestampDefineKey]: JSON.stringify(buildTimestampUtc),
    },
    optimizeDeps: {
      exclude: ["@playwright/test", "vite", "vite-plugin-istanbul", "vitest"],
    },
    build: {
      target: "esnext",
    },
    server: {
      proxy: {
        "/api": {
          target: apiProxyTarget,
          changeOrigin: true,
        },
      },
    },
    preview: {
      port: Number(env.T1D_WEB_PORT || 3000),
      proxy: {
        "/api": {
          target: apiProxyTarget,
          changeOrigin: true,
        },
      },
    },
    plugins: [
      istanbul({
        include: ["src/*", "*.ts"],
        exclude: ["node_modules", "test/", "tests/"],
        extension: [".js", ".ts"],
        requireEnv: false,
      }),
    ],
  });
}

/**
 * Vite configuration for development, preview server proxying, and Istanbul instrumentation.
 */
const viteConfig: UserConfig = createViteConfig();

export default viteConfig;
