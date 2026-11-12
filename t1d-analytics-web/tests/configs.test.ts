/**
 * @file configs.test.ts
 * @module tests/configs.test
 * Behavioral and structural invariant tests for Vite, Vitest, Playwright, and TypeDoc configurations.
 */

// @vitest-environment node

import fs from "fs";
import path from "path";
import { describe, it, expect } from "vitest";
import viteConfig, { createViteConfig } from "../vite.config";
import vitestConfig from "../vitest.config";
import playwrightConfig from "../playwright.config";
import playwrightFullConfig from "../playwright.full.config";

describe("Frontend configuration invariants", () => {
  const pkgPath = path.resolve(__dirname, "../package.json");
  const pkgRaw: { scripts?: Record<string, string> } = JSON.parse(
    fs.readFileSync(pkgPath, "utf-8"),
  );

  /**
   * Validates Vite proxy URL formatting, preview port bounds, and requireEnv Istanbul gating.
   */
  it("validates vite.config.ts proxy URLs, port bounds, and build target", () => {
    expect(viteConfig.build?.target).toBe("esnext");

    const defaultCfg = createViteConfig({});
    const defaultProxy = defaultCfg.server?.proxy?.["/api"];
    expect(defaultProxy).toBeDefined();
    if (
      defaultProxy !== undefined &&
      typeof defaultProxy === "object" &&
      "target" in defaultProxy
    ) {
      const targetUrl = new URL(String(defaultProxy.target));
      expect(targetUrl.protocol).toBe("http:");
      expect(targetUrl.hostname).toBe("127.0.0.1");
      expect(Number(targetUrl.port)).toBe(8000);
    }
    const defaultPort = defaultCfg.preview?.port ?? 0;
    expect(Number.isInteger(defaultPort)).toBe(true);
    expect(defaultPort).toBeGreaterThanOrEqual(1);
    expect(defaultPort).toBeLessThanOrEqual(65535);

    const customCfg = createViteConfig({
      T1D_API_PORT: "8088",
      T1D_WEB_PORT: "4173",
      BUILD_TIMESTAMP_UTC: "2026-11-13T00:00:00Z UTC",
    });
    const customProxy = customCfg.server?.proxy?.["/api"];
    expect(customProxy).toEqual({
      target: "http://127.0.0.1:8088",
      changeOrigin: true,
    });
    expect(customCfg.preview?.port).toBe(4173);
    expect(customCfg.define?.__BUILD_TIMESTAMP_UTC__).toBe(
      JSON.stringify("2026-11-13T00:00:00Z UTC"),
    );

    const explicitProxyCfg = createViteConfig({
      VITE_API_PROXY_TARGET: "http://127.0.0.1:9090",
    });
    expect(explicitProxyCfg.server?.proxy?.["/api"]).toEqual({
      target: "http://127.0.0.1:9090",
      changeOrigin: true,
    });
  });

  /**
   * Validates Vitest coverage scope (src TypeScript files only, 100% per-file thresholds) and setup file existence.
   */
  it("validates vitest.config.ts 100% src coverage scope and setup file existence", () => {
    expect(vitestConfig.test?.environment).toBe("jsdom");
    const setupFiles = vitestConfig.test?.setupFiles;
    const normalizedSetup = Array.isArray(setupFiles)
      ? setupFiles
      : setupFiles
        ? [setupFiles]
        : [];
    expect(normalizedSetup.length).toBeGreaterThan(0);
    for (const setupFile of normalizedSetup) {
      expect(fs.existsSync(path.resolve(__dirname, "..", setupFile))).toBe(
        true,
      );
    }

    const coverage = vitestConfig.test?.coverage;
    expect(coverage?.provider).toBe("istanbul");
    if (coverage && coverage.provider === "istanbul") {
      expect(coverage.include).toEqual(["src/**/*.ts"]);
      expect(coverage.thresholds).toEqual({
        lines: 100,
        functions: 100,
        branches: 100,
        statements: 100,
        perFile: true,
      });
    }
  });

  /**
   * Validates Playwright mocked and full-stack configuration directories, ports, and scripts.
   */
  it("validates playwright.config.ts and playwright.full.config.ts invariants", () => {
    expect(playwrightConfig.testDir).toBe("./tests-e2e");
    expect(
      fs.existsSync(
        path.resolve(__dirname, "..", playwrightConfig.testDir ?? ""),
      ),
    ).toBe(true);
    const baseUrl = new URL(String(playwrightConfig.use?.baseURL));
    expect(Number(baseUrl.port)).toBeGreaterThanOrEqual(1);
    expect(Number(baseUrl.port)).toBeLessThanOrEqual(65535);
    expect(pkgRaw.scripts?.dev).toBeDefined();

    expect(playwrightFullConfig.testDir).toBe("./tests-e2e-full");
    expect(
      fs.existsSync(
        path.resolve(__dirname, "..", playwrightFullConfig.testDir ?? ""),
      ),
    ).toBe(true);
    if (Array.isArray(playwrightFullConfig.webServer)) {
      expect(playwrightFullConfig.webServer.length).toBe(2);
      for (const srv of playwrightFullConfig.webServer) {
        const srvUrl = new URL(String(srv.url));
        const port = Number(srvUrl.port);
        expect(port).toBeGreaterThanOrEqual(1);
        expect(port).toBeLessThanOrEqual(65535);
      }
    }
  });

  /**
   * Verifies that every non-declaration TypeScript source module under `src/` is registered in `package.json` `scripts.docs`.
   */
  it("verifies all src TypeScript implementation modules are registered in TypeDoc scripts.docs", () => {
    const docsScript = pkgRaw.scripts?.docs || "";
    const docsTokens = docsScript.trim().split(/\s+/);
    expect(docsTokens[0]).toBe("typedoc");
    expect(docsTokens).toContain("--treatWarningsAsErrors");

    const registeredEntryPoints = docsTokens
      .filter((token) => token.startsWith("src/"))
      .sort();
    for (const entryPoint of registeredEntryPoints) {
      expect(fs.existsSync(path.resolve(__dirname, "..", entryPoint))).toBe(
        true,
      );
    }

    const srcDir = path.resolve(__dirname, "../src");
    /**
     * Recursively collects non-declaration `.ts` file paths relative to `prefix`.
     * @param {string} dir - Directory path to scan.
     * @param {string} [prefix="src"] - Relative path prefix.
     * @returns {string[]} Collected relative file paths.
     */
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

    const allSrcModules = collectTsFiles(srcDir).sort();
    expect(registeredEntryPoints).toEqual(allSrcModules);
  });
});
