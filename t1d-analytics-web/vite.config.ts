/**
 * @file vite.config.ts
 * @module vite.config
 * Vite development and production build configuration.
 */

import { defineConfig, type UserConfig } from "vite";
import istanbul from "vite-plugin-istanbul";

/**
 * Construct the Vite development, preview, and build configuration.
 *
 * @param env - Environment variable mapping used to resolve API proxy and preview port settings.
 * @returns Resolved Vite user configuration.
 */
export function createViteConfig(
  env: Record<string, string | undefined> = process.env,
): UserConfig {
  const apiProxyTarget =
    env.VITE_API_PROXY_TARGET ||
    `http://127.0.0.1:${env.T1D_API_PORT || "8000"}`;

  return defineConfig({
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
