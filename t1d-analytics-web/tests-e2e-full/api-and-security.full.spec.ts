/**
 * @file api-and-security.full.spec.ts
 * @module tests-e2e-full/api-and-security.full.spec
 * Full-stack end-to-end Playwright tests against a live FastAPI + DuckDB backend
 * covering initial chat window resilience, natural language queries in default chat,
 * health/readiness/schema/databases API contracts, read-only SQL security guards,
 * DuckDB external file sandboxing, SSE streaming for SQL and NLP, and localized API negotiation.
 */

import { test, expect } from "./base-test";

test.describe("Live Backend API, Security Sandboxing & Default Chat Resilience E2E", () => {
  /**
   * Verifies that the initial chat window loads with zero alerts, healthy status indicators,
   * and executes natural language questions on the default chat cleanly against live DuckDB.
   */
  test("Initial Default Chat: renders with zero alerts and executes natural language questions cleanly", async ({
    page,
  }) => {
    await page.goto("/");

    await expect(page.locator('.message [role="alert"]')).toHaveCount(0);
    await expect(page.locator("#system-status-banner")).toHaveClass(/hidden/);
    await expect(page.locator("#system-status-chip")).toHaveClass(
      /status-(healthy|ok)/,
    );
    await expect(page.locator("#provider-status-dot")).toHaveClass(/active/);
    await expect(page.locator("#model-select")).toHaveValue("sql");

    // 1. Ask a natural language question in the initial default chat
    await page.fill(
      "#chat-input",
      "Show me the first 5 patients in the demographics table.",
    );
    await page.click("#send-btn");

    const assistantMsg1 = page.locator(".message.assistant").first();
    await expect(assistantMsg1.locator(".sql-table")).toBeVisible({
      timeout: 10000,
    });
    await expect(assistantMsg1.locator('[role="alert"]')).toHaveCount(0);
    const rows = assistantMsg1.locator(".sql-table tbody tr");
    await expect(rows).toHaveCount(5);

    // 2. Ask an aggregate count natural language question in the same chat
    await page.fill("#chat-input", "How many patients are in the dataset?");
    await page.click("#send-btn");

    const assistantMsg2 = page.locator(".message.assistant").nth(1);
    await expect(assistantMsg2).toBeVisible({ timeout: 10000 });
    await expect(assistantMsg2.locator('[role="alert"]')).toHaveCount(0);
    await expect(assistantMsg2).toContainText("100");
  });

  /**
   * Verifies live `/api/health`, `/api/ready`, `/api/status`, `/api/databases`, and `/api/schema`
   * contract responses and confirms `sessions.duckdb` is excluded from `/api/databases`.
   */
  test("Live Diagnostics & Schema Endpoints: validates /api/health, /api/ready, /api/status, /api/databases, and /api/schema", async ({
    page,
  }) => {
    await page.goto("/");

    const healthResp = await page.request.get("/api/health");
    expect(healthResp.status()).toBe(200);
    const healthJson = (await healthResp.json()) as {
      status: string;
      database: { status_code: string; has_initial_data: boolean };
    };
    expect(["healthy", "degraded"]).toContain(healthJson.status);
    expect(healthJson.database.status_code).toBe("healthy");
    expect(healthJson.database.has_initial_data).toBe(true);

    const readyResp = await page.request.get("/api/ready");
    expect(readyResp.status()).toBe(200);

    const statusResp = await page.request.get("/api/status");
    expect(statusResp.status()).toBe(200);
    const statusJson = (await statusResp.json()) as {
      database: {
        exists: boolean;
        connected: boolean;
        status_code: string;
        has_initial_data: boolean;
      };
    };
    expect(statusJson.database.exists).toBe(true);
    expect(statusJson.database.connected).toBe(true);
    expect(statusJson.database.status_code).toBe("healthy");
    expect(statusJson.database.has_initial_data).toBe(true);

    const dbsResp = await page.request.get("/api/databases");
    expect(dbsResp.status()).toBe(200);
    const dbsJson = (await dbsResp.json()) as {
      databases: Array<{ name: string; path: string; size_bytes: number }>;
      current_db: string;
    };
    expect(Array.isArray(dbsJson.databases)).toBe(true);
    expect(dbsJson.databases.some((d) => d.name === "sessions.duckdb")).toBe(
      false,
    );

    const schemaResp = await page.request.get("/api/schema");
    expect(schemaResp.status()).toBe(200);
    const schemaJson = (await schemaResp.json()) as {
      tables: Array<{ name: string; columns: Array<{ name: string }> }>;
    };
    const tableNames = schemaJson.tables.map((t) => t.name);
    for (const expectedTable of [
      "patients",
      "cgm_readings",
      "hba1c_labs",
      "dka_adverse_events",
      "demographics",
      "cgm_data",
      "subjects",
      "tblaptsummary",
      "hba1c",
      "adverseevents",
    ]) {
      expect(tableNames).toContain(expectedTable);
    }
  });

  /**
   * Verifies live read-only SQL enforcement and DuckDB external file sandboxing across
   * `/api/chat`, `/api/chat/stream`, and `/api/execute-sql`.
   */
  test("Live Security Sandboxing: blocks mutating DDL/DML, external file reads, and path traversal", async ({
    page,
  }) => {
    await page.goto("/");

    // 1. Mutating statements rejected by /api/execute-sql
    for (const forbiddenSql of [
      "DROP TABLE patients;",
      "DELETE FROM patients WHERE subject_id = 1;",
      "INSERT INTO patients VALUES (999, 30, 'M', 'Control', 'Dexcom', 7.0, 0);",
      "UPDATE patients SET age = 99 WHERE subject_id = 1;",
      "ALTER TABLE patients DROP COLUMN age;",
      "CREATE TABLE hacked (id INT);",
    ]) {
      const execResp = await page.request.post("/api/execute-sql", {
        data: { query: forbiddenSql },
      });
      expect(execResp.status()).toBe(200);
      const execJson = (await execResp.json()) as {
        error?: { error_code: string };
      };
      expect(execJson.error?.error_code).toBe("backend.sqlExecution");
    }

    // 2. External file system read via read_csv('/etc/passwd') rejected
    const extReadResp = await page.request.post("/api/execute-sql", {
      data: { query: "SELECT * FROM read_csv('/etc/passwd');" },
    });
    expect(extReadResp.status()).toBe(200);
    const extReadJson = (await extReadResp.json()) as {
      error?: { error_code: string };
    };
    expect(extReadJson.error?.error_code).toBe("backend.sqlExecution");

    // 3. Path traversal in db_path rejected with 400
    const traversalResp = await page.request.post("/api/chat", {
      data: {
        message: "SELECT 1;",
        model: "sql",
        db_path: "../../etc/passwd",
      },
    });
    expect(traversalResp.status()).toBe(400);
  });

  /**
   * Verifies live SSE streaming (`/api/chat/stream`) for both literal SQL queries
   * and natural language prompts when Streaming Mode is toggled on.
   */
  test("Live SSE Streaming for SQL and Natural Language: streams results cleanly with zero alerts", async ({
    page,
  }) => {
    await page.goto("/");

    const streamBtn = page.locator("#streaming-toggle-btn");
    await streamBtn.click();
    await expect(streamBtn).toHaveAttribute("aria-pressed", "true");

    // 1. Stream a natural language query in default chat
    await page.fill(
      "#chat-input",
      "Show me the first 3 patients in the demographics table.",
    );
    await page.click("#send-btn");

    const assistantMsg1 = page.locator(".message.assistant").first();
    await expect(assistantMsg1.locator(".sql-table")).toBeVisible({
      timeout: 10000,
    });
    await expect(assistantMsg1.locator('[role="alert"]')).toHaveCount(0);
    await expect(assistantMsg1.locator(".sql-table tbody tr")).toHaveCount(3);

    // 2. Stream a literal SQL query with CTE
    await page.fill(
      "#chat-input",
      "WITH top_patients AS (SELECT subject_id, age FROM patients ORDER BY age DESC LIMIT 4) SELECT * FROM top_patients;",
    );
    await page.click("#send-btn");

    const assistantMsg2 = page.locator(".message.assistant").nth(1);
    await expect(assistantMsg2.locator(".sql-table")).toBeVisible({
      timeout: 10000,
    });
    await expect(assistantMsg2.locator('[role="alert"]')).toHaveCount(0);
    await expect(assistantMsg2.locator(".sql-table tbody tr")).toHaveCount(4);
  });

  /**
   * Verifies live internationalization switching across multiple locales (`ar`, `he`, `ja`, `en`)
   * including RTL layout switching and localized backend status negotiation.
   */
  test("Live Internationalization & RTL Negotiation: switches locales and preserves live query execution", async ({
    page,
  }) => {
    await page.goto("/");

    const langSelect = page.locator("#lang-select");
    await langSelect.selectOption("ar");
    await expect(page.locator("html")).toHaveAttribute("lang", "ar");
    await expect(page.locator("html")).toHaveAttribute("dir", "rtl");

    await langSelect.selectOption("he");
    await expect(page.locator("html")).toHaveAttribute("lang", "he");
    await expect(page.locator("html")).toHaveAttribute("dir", "rtl");

    await langSelect.selectOption("ja");
    await expect(page.locator("html")).toHaveAttribute("lang", "ja");
    await expect(page.locator("html")).toHaveAttribute("dir", "ltr");

    const jaStatusResp = await page.request.get("/api/status", {
      headers: { "Accept-Language": "ja" },
    });
    expect(jaStatusResp.status()).toBe(200);

    await langSelect.selectOption("en");
    await expect(page.locator("html")).toHaveAttribute("lang", "en");
    await expect(page.locator("html")).toHaveAttribute("dir", "ltr");
  });

  /**
   * Verifies live `/api/table/{name}/profile` completeness telemetry across all 4 seeded clinical tables
   * and verifies `/api/export/csv`, `/api/export/excel`, and `/api/export/parquet` server-side exports.
   */
  test("Live Clinical Table Profiling & Server Exports: validates completeness telemetry and CSV/Excel/Parquet exports", async ({
    page,
  }) => {
    await page.goto("/");

    for (const tableName of [
      "patients",
      "cgm_readings",
      "hba1c_labs",
      "dka_adverse_events",
    ]) {
      const profileResp = await page.request.get(
        `/api/table/${tableName}/profile`,
      );
      expect(profileResp.status()).toBe(200);
      const profileJson = (await profileResp.json()) as {
        table_name: string;
        total_rows: number;
        columns: Array<{
          name: string;
          null_count: number;
          completeness_pct: number;
        }>;
      };
      expect(profileJson.table_name).toBe(tableName);
      expect(profileJson.total_rows).toBeGreaterThan(0);
      expect(profileJson.columns.length).toBeGreaterThan(0);
    }

    for (const fmt of ["csv", "json", "excel", "parquet"]) {
      const exportResp = await page.request.get(
        `/api/export/${fmt}?table_name=patients`,
      );
      expect(exportResp.status()).toBe(200);
      const disposition = exportResp.headers()["content-disposition"] ?? "";
      expect(disposition).toContain("attachment");
      const bodyBytes = await exportResp.body();
      expect(bodyBytes.byteLength).toBeGreaterThan(30);
    }
  });

  /**
   * Verifies live `/api/training/jobs`, `/api/training/jobs/{id}`, `/api/providers/status`, and `/api/models` endpoints.
   */
  test("Live Jobs & Provider Readiness Endpoints: validates /api/training/jobs, /api/providers/status, and /api/models", async ({
    page,
  }) => {
    await page.goto("/");

    const jobsResp = await page.request.get("/api/training/jobs");
    expect(jobsResp.status()).toBe(200);
    const jobsJson = (await jobsResp.json()) as {
      jobs: Array<{ job_id: string; status: string }>;
    };
    expect(Array.isArray(jobsJson.jobs)).toBe(true);

    const missingJobResp = await page.request.get(
      "/api/training/jobs/nonexistent-e2e-job-id",
    );
    expect(missingJobResp.status()).toBe(404);

    const provStatusResp = await page.request.get("/api/providers/status");
    expect(provStatusResp.status()).toBe(200);
    const provStatusJson = (await provStatusResp.json()) as {
      providers: Array<{
        provider: string;
        configured: boolean;
        reachable: boolean;
      }>;
    };
    const providerNames = provStatusJson.providers.map((p) => p.provider);
    expect(providerNames).toEqual(
      expect.arrayContaining(["ollama", "openai", "anthropic", "google"]),
    );

    const modelsResp = await page.request.get("/api/models");
    expect(modelsResp.status()).toBe(200);
    const modelsJson = (await modelsResp.json()) as {
      models: Array<{
        name: string;
        provider: string;
        configured: boolean;
        requires_key: boolean;
      }>;
    };
    expect(modelsJson.models.length).toBeGreaterThan(0);
  });
});
