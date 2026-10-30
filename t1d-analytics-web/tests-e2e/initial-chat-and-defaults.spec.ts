/**
 * @file initial-chat-and-defaults.spec.ts
 * @module tests-e2e/initial-chat-and-defaults.spec
 * End-to-end Playwright tests for initial chat window state, default chat model routing,
 * provider status badge synchronization, temporary/new chat fallback, and all 5 built-in clinical chips.
 */

import { test, expect } from "./base-test";

test.describe("Initial Chat Window & Default Chat Lifecycle", () => {
  /**
   * Verifies that the initial chat window loads with zero alerts, healthy status indicators,
   * an active local engine badge, and all 5 clinical example query chips.
   */
  test("Renders initial chat window with zero errors, healthy status indicators, and 5 example chips", async ({
    page,
  }) => {
    await page.goto("/");

    await expect(page.locator('.message [role="alert"]')).toHaveCount(0);
    await expect(page.locator("#system-status-banner")).toHaveClass(/hidden/);
    await expect(page.locator("#system-status-chip")).toHaveClass(
      /status-(healthy|ok)/,
    );
    await expect(page.locator("#provider-status-dot")).toHaveClass(/active/);
    await expect(page.locator(".empty-state")).toBeVisible();
    await expect(page.locator(".empty-state .chip")).toHaveCount(5);
    await expect(page.locator("#model-select")).toHaveValue("gemma4");
  });

  /**
   * Verifies that when Ollama is offline and `/api/models` returns only unconfigured cloud models,
   * the default chat falls back to 'sql' mode without selecting an unconfigured cloud provider.
   */
  test("Falls back to 'sql' mode on initial default chat when only unconfigured cloud models are returned", async ({
    page,
  }) => {
    await page.route("**/api/models*", async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          models: [
            {
              name: "openai/gpt-4o",
              provider: "openai",
              configured: false,
              requires_key: true,
              available: false,
              reachable: false,
              error_code: "MISSING_API_KEY",
            },
            {
              name: "anthropic/claude-3-5-sonnet",
              provider: "anthropic",
              configured: false,
              requires_key: true,
              available: false,
              reachable: false,
              error_code: "MISSING_API_KEY",
            },
          ],
        }),
      });
    });

    await page.goto("/");
    await expect(page.locator("#model-select")).toHaveValue("sql");
    await expect(page.locator("#provider-status-dot")).toHaveClass(/active/);
    await expect(page.locator('.message [role="alert"]')).toHaveCount(0);
  });

  /**
   * Verifies that a server-env-configured cloud model (e.g., ANTHROPIC_API_KEY set on host)
   * does NOT hijack the untouched default chat when no client API key is stored in localStorage,
   * while still showing an active green dot if the user manually selects that server-configured model.
   */
  test("Does not auto-select server-env cloud model on default chat when Ollama is offline, and shows green dot when selected", async ({
    page,
  }) => {
    await page.route("**/api/models*", async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          models: [
            {
              name: "anthropic/claude-3-5-sonnet",
              provider: "anthropic",
              configured: true,
              requires_key: true,
              available: true,
              reachable: true,
            },
            {
              name: "openai/gpt-4o",
              provider: "openai",
              configured: false,
              requires_key: true,
              available: false,
              reachable: false,
              error_code: "MISSING_API_KEY",
            },
          ],
        }),
      });
    });

    await page.goto("/");

    // Default untouched chat stays on 'sql' instead of hijacking to anthropic/claude-3-5-sonnet
    await expect(page.locator("#model-select")).toHaveValue("sql");
    await expect(page.locator("#provider-status-dot")).toHaveClass(/active/);

    // Manually selecting the server-configured model shows active green dot (via data-configured="true")
    await page.selectOption("#model-select", "anthropic/claude-3-5-sonnet");
    await expect(page.locator("#provider-status-dot")).toHaveClass(/active/);

    // Manually selecting the unconfigured model shows warning yellow dot
    await page.selectOption("#model-select", "openai/gpt-4o");
    await expect(page.locator("#provider-status-dot")).toHaveClass(/warning/);
  });

  /**
   * Verifies that when a client API key is stored in localStorage, the default chat can select
   * that ready client-configured cloud model when local Ollama is offline.
   */
  test("Selects client-configured cloud model when API key is present in localStorage and Ollama is offline", async ({
    page,
  }) => {
    await page.addInitScript(() => {
      localStorage.setItem("t1d_api_key_openai", "sk-client-configured-key");
    });

    await page.route("**/api/models*", async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          models: [
            {
              name: "openai/gpt-4o",
              provider: "openai",
              configured: false,
              requires_key: true,
              available: true,
              reachable: true,
            },
          ],
        }),
      });
    });

    await page.goto("/");
    await expect(page.locator("#model-select")).toHaveValue("openai/gpt-4o");
    await expect(page.locator("#provider-status-dot")).toHaveClass(/active/);
  });

  /**
   * Verifies that clicking `+ New Chat`, deleting the last chat into `Temporary chat`,
   * and reloading with a URL hash all preserve a valid `#model-select` value ('sql')
   * and never leave `#model-select` blank ('') when Ollama is offline.
   */
  test("Preserves valid 'sql' model selection across + New Chat, Temporary chat fallback, and hash reload when Ollama is offline", async ({
    page,
  }) => {
    let lastRequestedModel: string | undefined;
    await page.route("**/api/models*", async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          models: [
            {
              name: "openai/gpt-4o",
              provider: "openai",
              configured: false,
              requires_key: true,
              available: false,
              reachable: false,
            },
          ],
        }),
      });
    });

    await page.route("**/api/chat", async (route) => {
      const payload = route.request().postDataJSON() as { model?: string };
      lastRequestedModel = payload.model;
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          content: "backend.literalSql",
          sqlQuery: 'SELECT * FROM "patients" LIMIT 10;',
          sqlResult: [{ id: 1, name: "Alice" }],
        }),
      });
    });

    await page.goto("/");
    await expect(page.locator("#model-select")).toHaveValue("sql");

    // 1. Click + New Chat when Ollama is offline
    await page.click("#new-chat-btn");
    await expect(page.locator("#model-select")).toHaveValue("sql");

    // 2. Delete both chats so Temporary chat is created
    page.on("dialog", (dialog) => dialog.accept());
    await page.locator(".dropdown-btn").first().click();
    await page.locator(".dropdown-item.danger").first().click();

    await page.locator(".dropdown-btn").first().click();
    await page.locator(".dropdown-item.danger").first().click();

    await expect(page.locator(".chat-item-title").first()).toHaveText(
      "Temporary chat",
    );
    await expect(page.locator("#model-select")).toHaveValue("sql");

    // 3. Send a message in Temporary chat and verify model is 'sql' (never '')
    await page.fill("#chat-input", "Hello");
    await page.click("#send-btn");
    await expect(page.locator(".message.assistant")).toBeVisible();
    expect(lastRequestedModel).toBe("sql");

    // 4. Reload with URL hash and verify modelSelect remains 'sql'
    await page.reload();
    await expect(page.locator("#model-select")).toHaveValue("sql");
  });

  /**
   * Verifies that all 5 built-in clinical example query chips populate the chat input,
   * set a valid model in `#model-select` (both when Ollama is online and when offline),
   * and execute cleanly without errors.
   */
  test("Executes all 5 built-in clinical example query chips cleanly when Ollama is online and when offline", async ({
    page,
  }) => {
    await page.goto("/");

    // Verify all 5 chips when gemma4 is available
    for (let i = 0; i < 5; i++) {
      if (i > 0) {
        await page.click("#new-chat-btn");
      }
      const chip = page.locator(".empty-state .chip").nth(i);
      await expect(chip).toBeVisible();
      await chip.click();

      const expectedMode = i < 4 ? "sql" : "gemma4";
      await expect(page.locator("#model-select")).toHaveValue(expectedMode);

      const inputVal = await page.locator("#chat-input").inputValue();
      expect(inputVal.length).toBeGreaterThan(15);

      await page.click("#send-btn");
      await expect(page.locator(".message.assistant").first()).toBeVisible();
      await expect(page.locator('.message [role="alert"]')).toHaveCount(0);
    }

    // Now simulate Ollama offline (no gemma4 option in #model-select) and verify Chip 4 (NLP chip) falls back to 'sql'
    await page.route("**/api/models*", async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({ models: [] }),
      });
    });

    await page.reload();
    await page.click("#new-chat-btn");
    const nlpChip = page.locator(".empty-state .chip").nth(4);
    await nlpChip.click();
    await expect(page.locator("#model-select")).toHaveValue("sql");
    await page.click("#send-btn");
    await expect(page.locator(".message.assistant").first()).toBeVisible();
    await expect(page.locator('.message [role="alert"]')).toHaveCount(0);
  });

  /**
   * Verifies switching between multiple chats with distinct models (`sql`, `gemma4`,
   * server-configured `anthropic/claude-3-5-sonnet`, and unconfigured `openai/gpt-4o`)
   * updates `#model-select` and `#provider-status-dot` (`active` vs `warning`) on every switch,
   * and verifies block-comment and metadata SQL statements auto-route to `'sql'`.
   */
  test("Synchronizes #model-select and #provider-status-dot across multi-chat switches and auto-routes block-comment/metadata SQL", async ({
    page,
  }) => {
    const requestedModels: string[] = [];
    await page.route("**/api/models*", async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          models: [
            {
              name: "gemma4",
              provider: "ollama",
              configured: true,
              requires_key: false,
              available: true,
              reachable: true,
            },
            {
              name: "anthropic/claude-3-5-sonnet",
              provider: "anthropic",
              configured: true,
              requires_key: true,
              available: true,
              reachable: true,
            },
            {
              name: "openai/gpt-4o",
              provider: "openai",
              configured: false,
              requires_key: true,
              available: false,
              reachable: false,
              error_code: "MISSING_API_KEY",
            },
          ],
        }),
      });
    });

    await page.route("**/api/chat", async (route) => {
      const body = route.request().postDataJSON() as { model: string };
      requestedModels.push(body.model);
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          content: "backend.literalSql",
          sqlQuery: "SELECT 1;",
          sqlResult: [{ val: 1 }],
        }),
      });
    });

    await page.goto("/");
    await expect(page.locator("#model-select")).toHaveValue("gemma4");

    // Chat 1: set to unconfigured openai/gpt-4o -> warning dot
    await page.selectOption("#model-select", "openai/gpt-4o");
    await expect(page.locator("#provider-status-dot")).toHaveClass(/warning/);

    // Chat 2: create new chat and set to server-configured anthropic/claude-3-5-sonnet -> active dot
    await page.click("#new-chat-btn");
    await page.selectOption("#model-select", "anthropic/claude-3-5-sonnet");
    await expect(page.locator("#provider-status-dot")).toHaveClass(/active/);

    // Switch back to Chat 1 (index 0 in sidebar) -> modelSelect is openai/gpt-4o and dot is warning
    await page.locator("#chat-list .chat-item").nth(0).click();
    await expect(page.locator("#model-select")).toHaveValue("openai/gpt-4o");
    await expect(page.locator("#provider-status-dot")).toHaveClass(/warning/);

    // Switch back to Chat 2 (index 1 in sidebar) -> modelSelect is anthropic/claude-3-5-sonnet and dot is active
    await page.locator("#chat-list .chat-item").nth(1).click();
    await expect(page.locator("#model-select")).toHaveValue(
      "anthropic/claude-3-5-sonnet",
    );
    await expect(page.locator("#provider-status-dot")).toHaveClass(/active/);

    // Auto-route block-comment SQL, SHOW TABLES, DESCRIBE, PRAGMA, EXPLAIN, SUMMARIZE, and VALUES to 'sql'
    await page.selectOption("#model-select", "gemma4");
    for (const stmt of [
      "/* Clinical Cohort Block Comment */\nSELECT * FROM patients LIMIT 1;",
      "SHOW TABLES;",
      "DESCRIBE patients;",
      "PRAGMA table_info('patients');",
      "EXPLAIN SELECT 1;",
      "SUMMARIZE patients;",
      "VALUES (1, 'CGM');",
    ]) {
      await page.fill("#chat-input", stmt);
      await page.click("#send-btn");
    }
    await expect(page.locator(".message.assistant")).toHaveCount(7);
    expect(requestedModels).toEqual([
      "sql",
      "sql",
      "sql",
      "sql",
      "sql",
      "sql",
      "sql",
    ]);
  });
});
