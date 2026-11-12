/**
 * @file playwright.full.config.ts
 * @module playwright.full.config
 * Full end-to-end matrix configuration with isolated temporary DuckDB backend instance.
 */

import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import {
  defineConfig,
  devices,
  type PlaywrightTestConfig,
} from "@playwright/test";

/**
 * Constructs the full-stack Playwright end-to-end test configuration.
 * @returns {PlaywrightTestConfig} Resolved Playwright full-stack test configuration.
 */
export function createPlaywrightFullConfig(): PlaywrightTestConfig {
  const here = path.dirname(fileURLToPath(import.meta.url));
  const venvPy = path.resolve(here, "../.venv/bin/python");
  const pyCmd = fs.existsSync(venvPy) ? venvPy : "python3";
  const pyBootstrap = [
    "import os, pathlib, subprocess, sys, tempfile",
    "from t1d_analytics.ingestion import load_data_to_duckdb",
    "from t1d_analytics.seed import seed_database",
    "tmp = tempfile.TemporaryDirectory(prefix='t1d_e2e_full_')",
    "d = tmp.name",
    "pathlib.Path(d, 'tiny.csv').write_text('id,name,value\\n1,Test A,100\\n2,Test B,200\\n', encoding='utf-8')",
    "db = os.path.join(d, 'ci_test.duckdb')",
    "sdb = os.path.join(d, 'sessions.duckdb')",
    "os.environ['T1D_EXTRA_DB_ROOTS'] = d",
    "os.environ['T1D_DATA_DIR'] = d",
    "os.environ['T1D_DB_PATH'] = db",
    "os.environ['T1D_SESSIONS_DB_PATH'] = sdb",
    "seed_database(data_dir=d, db_path=db, sessions_db_path=sdb, overwrite_csvs=False)",
    "load_data_to_duckdb(d, db)",
    "sys.exit(subprocess.call([sys.executable, '-m', 'uvicorn', 't1d_analytics.api:app', '--port', '8000']))",
  ].join("; ");

  return defineConfig({
    testDir: "./tests-e2e-full",
    fullyParallel: false,
    retries: 0,
    workers: 1,
    reporter: "list",
    use: {
      baseURL: "http://localhost:5173",
      trace: "on-first-retry",
    },
    projects: [
      {
        name: "chromium",
        use: {
          ...devices["Desktop Chrome"],
          channel: "chrome",
          colorScheme: "light",
        },
      },
    ],
    webServer: [
      {
        command: `"${pyCmd}" -c "${pyBootstrap}"`,
        url: "http://127.0.0.1:8000/api/ready",
        reuseExistingServer: true,
        timeout: 120000,
        env: {
          PYTHONPATH: "../t1d-analytics-backend/src",
        },
      },
      {
        command: "npm run dev",
        url: "http://localhost:5173",
        reuseExistingServer: true,
        timeout: 120000,
        env: {
          VITE_API_PROXY_TARGET: "http://127.0.0.1:8000",
          VITE_COVERAGE: "true",
        },
      },
    ],
  });
}

/**
 * Playwright configuration for full-stack end-to-end testing with live Python FastAPI backend and Vite dev server.
 */
const playwrightFullConfig: PlaywrightTestConfig = createPlaywrightFullConfig();

export default playwrightFullConfig;
