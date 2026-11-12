/**
 * @file vite.config.ts
 * @module vite.config
 * Vite development and production build configuration.
 */

import { defineConfig, type UserConfig } from "vite";
import istanbul from "vite-plugin-istanbul";

/**
 * Constructs the Vite configuration with optional environment overrides.
 * @param {Record<string, string | undefined>} [env=process.env] - Environment dictionary.
 * @returns {UserConfig} Vite configuration object.
 */
export function createViteConfig(
  env: Record<string, string | undefined> = process.env,
): UserConfig {
  const apiPort = env.T1D_API_PORT || "8000";
  const apiProxyTarget =
    env.VITE_API_PROXY_TARGET || `http://127.0.0.1:${apiPort}`;
  const webPreviewPort = Number(env.T1D_WEB_PORT || 3000);
  const buildTimestampUtc =
    env.BUILD_TIMESTAMP_UTC ||
    `${new Date().toISOString().replace(/\.\d{3}Z$/, "Z")} UTC`;

  return defineConfig({
    define: {
      __BUILD_TIMESTAMP_UTC__: JSON.stringify(buildTimestampUtc),
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
      port: webPreviewPort,
      proxy: {
        "/api": {
          target: apiProxyTarget,
          changeOrigin: true,
        },
      },
    },
    plugins: [
      istanbul({
        include: "src/*",
        exclude: ["node_modules", "tests/", "tests-e2e/"],
        extension: [".js", ".ts"],
        requireEnv: true,
      }),
    ],
  });
}

/**
 * Default Vite configuration export.
 */
const viteConfig: UserConfig = createViteConfig();

export default viteConfig;
