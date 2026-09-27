/**
 * @file configs.test.ts
 * Comprehensive unit tests for frontend build, test, and E2E configuration files.
 */

// @vitest-environment node

import { describe, it, expect } from "vitest";
import viteConfig from "../vite.config";
import vitestConfig from "../vitest.config";
import playwrightConfig from "../playwright.config";
import playwrightFullConfig from "../playwright.full.config";

describe("Frontend configuration modules", () => {
  /**
   * Tests Vite development and build configuration values.
   */
  it("validates vite.config.ts configuration", () => {
    expect(viteConfig).toBeDefined();
    expect(viteConfig.build?.target).toBe("esnext");
    expect(viteConfig.server?.proxy?.["/api"]).toEqual({
      target: "http://127.0.0.1:8000",
      changeOrigin: true,
    });
    expect(viteConfig.preview?.port).toBe(3000);
    expect(viteConfig.preview?.proxy?.["/api"]).toEqual({
      target: "http://127.0.0.1:8000",
      changeOrigin: true,
    });
    expect(Array.isArray(viteConfig.plugins)).toBe(true);
    expect(viteConfig.plugins?.length).toBeGreaterThan(0);
  });

  /**
   * Tests Vitest configuration, coverage settings, and exclusions.
   */
  it("validates vitest.config.ts configuration", () => {
    expect(vitestConfig).toBeDefined();
    expect(vitestConfig.build?.target).toBe("esnext");
    expect(vitestConfig.test?.environment).toBe("jsdom");
    expect(vitestConfig.test?.setupFiles).toEqual(["./tests/setup.ts"]);
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
      expect(coverage.reporter).toEqual(
        expect.arrayContaining(["text", "html", "lcov", "json"]),
      );
      expect(coverage.exclude).toEqual(
        expect.arrayContaining(["docs/**", "coverage/**", "tests/**"]),
      );
    }
  });

  /**
   * Tests standard Playwright end-to-end configuration.
   */
  it("validates playwright.config.ts standard configuration", () => {
    expect(playwrightConfig).toBeDefined();
    expect(playwrightConfig.testDir).toBe("./tests-e2e");
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
    const webServers = playwrightConfig.webServer as Array<{
      command: string;
      url: string;
      reuseExistingServer: boolean;
      timeout: number;
    }>;
    expect(webServers[0].command).toBe("npm run dev");
    expect(webServers[0].url).toBe("http://localhost:5173");
    expect(webServers[0].reuseExistingServer).toBe(true);
    expect(webServers[0].timeout).toBe(120000);
  });

  /**
   * Tests full Playwright matrix configuration including backend web server.
   */
  it("validates playwright.full.config.ts full matrix configuration", () => {
    expect(playwrightFullConfig).toBeDefined();
    expect(playwrightFullConfig.testDir).toBe("./tests-e2e-full");
    expect(playwrightFullConfig.fullyParallel).toBe(false);
    expect(playwrightFullConfig.workers).toBe(1);
    expect(playwrightFullConfig.retries).toBe(0);
    expect(playwrightFullConfig.reporter).toBe("list");
    expect(playwrightFullConfig.use?.baseURL).toBe("http://localhost:5173");
    expect(playwrightFullConfig.use?.trace).toBe("on-first-retry");
    expect(Array.isArray(playwrightFullConfig.webServer)).toBe(true);
    const webServers = playwrightFullConfig.webServer as Array<{
      command: string;
      url: string;
      reuseExistingServer: boolean;
      timeout: number;
    }>;
    expect(webServers.length).toBe(2);
    expect(webServers[0].command).toBe(
      "python -m uvicorn t1d_analytics.api:app --port 8000 || python3 -m uvicorn t1d_analytics.api:app --port 8000",
    );
    expect(webServers[0].url).toBe("http://127.0.0.1:8000/api/models");
    expect(webServers[0].reuseExistingServer).toBe(true);
    expect(webServers[1].command).toBe("npm run dev");
    expect(webServers[1].url).toBe("http://localhost:5173");
    expect(webServers[1].reuseExistingServer).toBe(true);
  });
});
