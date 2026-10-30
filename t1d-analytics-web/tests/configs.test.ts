/**
 * @file configs.test.ts
 * @module tests/configs.test
 * Comprehensive unit tests for frontend build, test, and E2E configuration files.
 */

// @vitest-environment node

import fs from "fs";
import path from "path";
import { describe, it, expect } from "vitest";
import viteConfig, {
  createViteConfig,
  getDefaultViteEnv,
} from "../vite.config";
import vitestConfig, { createVitestConfig } from "../vitest.config";
import playwrightConfig, { createPlaywrightConfig } from "../playwright.config";
import playwrightFullConfig, {
  createPlaywrightFullConfig,
} from "../playwright.full.config";

describe("Frontend configuration modules", () => {
  const pkgPath = path.resolve(__dirname, "../package.json");
  const pkgRaw: { scripts?: Record<string, string> } = JSON.parse(
    fs.readFileSync(pkgPath, "utf-8"),
  );

  /**
   * Tests Vite development and build configuration values across default and custom environments.
   */
  it("validates vite.config.ts configuration", () => {
    const expectedTarget =
      process.env.VITE_API_PROXY_TARGET ||
      `http://127.0.0.1:${process.env.T1D_API_PORT || "8000"}`;
    expect(viteConfig).toBeDefined();
    expect(viteConfig.build?.target).toBe("esnext");
    expect(viteConfig.server?.proxy?.["/api"]).toEqual({
      target: expectedTarget,
      changeOrigin: true,
    });
    expect(viteConfig.preview?.port).toBe(
      Number(process.env.T1D_WEB_PORT || 3000),
    );
    expect(viteConfig.preview?.proxy?.["/api"]).toEqual({
      target: expectedTarget,
      changeOrigin: true,
    });
    expect(Array.isArray(viteConfig.plugins)).toBe(true);
    expect(viteConfig.plugins?.length).toBeGreaterThan(0);

    const fallbackCfg = createViteConfig({});
    expect(fallbackCfg.server?.proxy?.["/api"]).toEqual({
      target: "http://127.0.0.1:8000",
      changeOrigin: true,
    });
    expect(fallbackCfg.preview?.port).toBe(3000);

    const portOverrideCfg = createViteConfig({
      T1D_API_PORT: "8080",
      T1D_WEB_PORT: "4173",
    });
    expect(portOverrideCfg.server?.proxy?.["/api"]).toEqual({
      target: "http://127.0.0.1:8080",
      changeOrigin: true,
    });
    expect(portOverrideCfg.preview?.port).toBe(4173);

    const proxyTargetOverrideCfg = createViteConfig({
      VITE_API_PROXY_TARGET: "http://127.0.0.1:9000",
      BUILD_TIMESTAMP_UTC: "2026-10-29T12:00:00Z UTC",
    });
    expect(proxyTargetOverrideCfg.server?.proxy?.["/api"]).toEqual({
      target: "http://127.0.0.1:9000",
      changeOrigin: true,
    });
    expect(proxyTargetOverrideCfg.define?.__BUILD_TIMESTAMP_UTC__).toBe(
      JSON.stringify("2026-10-29T12:00:00Z UTC"),
    );

    expect(getDefaultViteEnv()).toBe(process.env);
    const origProcess = globalThis.process;
    Object.defineProperty(globalThis, "process", {
      value: undefined,
      configurable: true,
    });
    expect(getDefaultViteEnv()).toEqual({});
    Object.defineProperty(globalThis, "process", {
      value: origProcess,
      configurable: true,
    });
  });

  /**
   * Tests Vitest configuration, coverage settings, and exclusions.
   */
  it("validates vitest.config.ts configuration and setup file existence", () => {
    const freshVitestCfg = createVitestConfig();
    expect(freshVitestCfg).toEqual(vitestConfig);
    expect(vitestConfig).toBeDefined();
    expect(vitestConfig.build?.target).toBe("esnext");
    expect(vitestConfig.test?.environment).toBe("jsdom");
    expect(vitestConfig.test?.setupFiles).toEqual(["./tests/setup.ts"]);
    for (const setupFile of vitestConfig.test?.setupFiles || []) {
      expect(fs.existsSync(path.resolve(__dirname, "..", setupFile))).toBe(
        true,
      );
    }
    expect(vitestConfig.test?.exclude).toEqual(
      expect.arrayContaining([
        "node_modules",
        "dist",
        "tests-e2e/**",
        "tests-e2e-full/**",
      ]),
    );
    const coverage = vitestConfig.test?.coverage;
    expect(coverage?.provider).toBe("istanbul");
    if (coverage && coverage.provider === "istanbul") {
      expect(coverage.all).toBe(true);
      expect(coverage.thresholds).toEqual({
        lines: 100,
        functions: 100,
        branches: 100,
        statements: 100,
        perFile: true,
      });
      expect(coverage.reporter).toEqual(
        expect.arrayContaining(["text", "html", "lcov", "json"]),
      );
      expect(coverage.exclude).toEqual(
        expect.arrayContaining([
          "docs/**",
          "coverage/**",
          "tests/**",
          "!vite.config.ts",
          "!vitest.config.ts",
        ]),
      );
    }
  });

  /**
   * Tests standard Playwright end-to-end configuration.
   */
  it("validates playwright.config.ts standard configuration and directory invariants", () => {
    expect(createPlaywrightConfig()).toEqual(playwrightConfig);
    expect(playwrightConfig).toBeDefined();
    expect(playwrightConfig.testDir).toBe("./tests-e2e");
    expect(
      fs.existsSync(path.resolve(__dirname, "..", playwrightConfig.testDir!)),
    ).toBe(true);
    expect(playwrightConfig.fullyParallel).toBe(true);
    expect(playwrightConfig.workers).toBe(1);
    expect(playwrightConfig.retries).toBe(0);
    expect(playwrightConfig.reporter).toBe("list");
    expect(playwrightConfig.use?.baseURL).toBe("http://localhost:5173");
    expect(playwrightConfig.use?.trace).toBe("on-first-retry");
    expect(playwrightConfig.projects?.length).toBe(1);
    expect(playwrightConfig.projects?.[0].name).toBe("chromium");
    expect(playwrightConfig.projects?.[0].use?.channel).toBe("chrome");
    expect(playwrightConfig.projects?.[0].use?.colorScheme).toBe("light");
    expect(Array.isArray(playwrightConfig.webServer)).toBe(true);
    if (Array.isArray(playwrightConfig.webServer)) {
      const webServers = playwrightConfig.webServer;
      expect(webServers[0].command).toBe("npm run dev");
      expect(pkgRaw.scripts?.dev).toBeDefined();
      expect(webServers[0].url).toBe("http://localhost:5173");
      expect(webServers[0].reuseExistingServer).toBe(true);
      expect(webServers[0].timeout).toBe(120000);
    }
  });

  /**
   * Tests full Playwright matrix configuration including backend web server.
   */
  it("validates playwright.full.config.ts full matrix configuration and directory invariants", () => {
    expect(createPlaywrightFullConfig()).toEqual(playwrightFullConfig);
    expect(playwrightFullConfig).toBeDefined();
    expect(playwrightFullConfig.testDir).toBe("./tests-e2e-full");
    expect(
      fs.existsSync(
        path.resolve(__dirname, "..", playwrightFullConfig.testDir!),
      ),
    ).toBe(true);
    expect(playwrightFullConfig.fullyParallel).toBe(false);
    expect(playwrightFullConfig.workers).toBe(1);
    expect(playwrightFullConfig.retries).toBe(0);
    expect(playwrightFullConfig.reporter).toBe("list");
    expect(playwrightFullConfig.use?.baseURL).toBe("http://localhost:5173");
    expect(playwrightFullConfig.use?.trace).toBe("on-first-retry");
    expect(Array.isArray(playwrightFullConfig.webServer)).toBe(true);
    if (Array.isArray(playwrightFullConfig.webServer)) {
      const webServers = playwrightFullConfig.webServer;
      expect(webServers.length).toBe(2);
      expect(webServers[0].command).toBe(
        "python3 -m uvicorn t1d_analytics.api:app --port 8000 || python -m uvicorn t1d_analytics.api:app --port 8000",
      );
      expect(webServers[0].url).toBe("http://127.0.0.1:8000/api/ready");
      expect(webServers[0].reuseExistingServer).toBe(true);
      expect(webServers[0].env).toEqual({
        PYTHONPATH: "../t1d-analytics-backend/src",
        T1D_DB_PATH: "../ci_test.duckdb",
        T1D_SESSIONS_DB_PATH:
          "../t1d-analytics-backend/tests/test_data/sessions.duckdb",
      });
      expect(webServers[1].command).toBe("npm run dev");
      expect(pkgRaw.scripts?.dev).toBeDefined();
      expect(webServers[1].url).toBe("http://localhost:5173");
      expect(webServers[1].reuseExistingServer).toBe(true);
    }
  });

  /**
   * Verifies that every TypeScript implementation module under `src/` is registered in `package.json` `scripts.docs`.
   */
  it("verifies all src/**/*.ts implementation modules are registered in TypeDoc scripts.docs", () => {
    const docsScript = pkgRaw.scripts?.docs || "";
    const srcDir = path.resolve(__dirname, "../src");
    const collectTsFiles = (dir: string, prefix: string = "src"): string[] => {
      const entries = fs.readdirSync(dir, { withFileTypes: true });
      const results: string[] = [];
      for (const entry of entries) {
        if (entry.isDirectory()) {
          results.push(
            ...collectTsFiles(
              path.join(dir, entry.name),
              `${prefix}/${entry.name}`,
            ),
          );
        } else if (
          entry.name.endsWith(".ts") &&
          !entry.name.endsWith(".d.ts")
        ) {
          results.push(`${prefix}/${entry.name}`);
        }
      }
      return results;
    };

    const allSrcModules = collectTsFiles(srcDir);
    expect(allSrcModules.length).toBeGreaterThan(10);
    for (const modPath of allSrcModules) {
      expect(docsScript).toContain(modPath);
    }
  });
});
