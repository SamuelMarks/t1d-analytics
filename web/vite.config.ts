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
  plugins: [
    istanbul({
      include: "src/*",
      exclude: ["node_modules", "test/"],
      extension: [".js", ".ts"],
      requireEnv: false,
    }),
  ],
});
