/**
 * @file vite.config.ts
 * Vite development and production build configuration.
 */

import { defineConfig } from "vite";
import istanbul from "vite-plugin-istanbul";

export default defineConfig({
  build: {
    target: "esnext",
  },
  server: {
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
    },
  },
  preview: {
    port: 3000,
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8000",
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
