/**
 * @file vite.config.ts
 * @module vite.config
 * Vite development and production build configuration.
 */

import { defineConfig } from "vite";
import istanbul from "vite-plugin-istanbul";

const apiProxyTarget =
  process.env.VITE_API_PROXY_TARGET ||
  `http://127.0.0.1:${process.env.T1D_API_PORT || "8000"}`;

/**
 * Vite configuration for development, preview server proxying, and Istanbul instrumentation.
 */
export default defineConfig({
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
    port: Number(process.env.T1D_WEB_PORT || 3000),
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
