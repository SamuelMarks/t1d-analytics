/**
 * @file chat.spec.ts
 * @module tests-e2e/chat.spec
 * End-to-end Playwright tests for core chat UI interactions and accessibility.
 */

import { test, expect } from "./base-test";
import AxeBuilder from "@axe-core/playwright";

test.describe("Chat UI E2E", () => {
  test.beforeEach(async ({ page }) => {
    // Add models endpoint mock for all tests
    await page.route("**/api/models", async (route) => {
      await route.fulfill({
        json: { models: [{ name: "gemma4" }] },
      });
    });
    // We will mock the API calls
    await page.route("**/api/chat", async (route) => {
      const request = route.request();
      const postData = JSON.parse(request.postData() || "{}");

      if (postData.model === "sql") {
        if (postData.message.includes("empty")) {
          await route.fulfill({
            json: { content: "Query executed.", sqlResult: [] },
          });
        } else {
          await route.fulfill({
            json: {
              content: "Query executed.",
              sqlResult: [{ count: 42 }],
            },
          });
        }
      } else {
        await route.fulfill({
          json: {
            content: "Here is the result:",
            sqlQuery: "SELECT * FROM test;",
            sqlResult: [{ id: 1, name: "Test" }],
          },
        });
      }
    });

    await page.goto("/");
  });

  test("Ask a question in SQL and show results", async ({ page }) => {
    await page.selectOption("#model-select", "sql");
    await page.fill("#chat-input", "SELECT count(*) as count FROM users");
    await page.click("#send-btn");

    await expect(page.locator(".message.assistant")).toContainText(
      "Query executed.",
    );
    await expect(page.locator(".sql-table")).toBeVisible();
    await expect(page.locator(".sql-table td").first()).toHaveText("42");
  });

  test("Ask a question in natural-language, see the SQL it generates", async ({
    page,
  }) => {
    await page.selectOption("#model-select", "gemma4");
    await page.fill("#chat-input", "Show me all tests");
    await page.click("#send-btn");

    await expect(page.locator(".message.assistant")).toContainText(
      "Here is the result:",
    );
    await expect(page.locator(".sql-query")).toHaveText("SELECT * FROM test;");
    await expect(page.locator(".sql-table td").first()).toHaveText("1");
    await expect(page.locator(".sql-table td").nth(1)).toHaveText("Test");
  });

  test('Show "No rows returned" message', async ({ page }) => {
    await page.selectOption("#model-select", "sql");
    await page.fill("#chat-input", "empty");
    await page.click("#send-btn");

    await expect(page.locator(".sql-empty")).toHaveText("No rows returned");
  });

  test("Show chat ID in URL", async ({ page }) => {
    const chatTitle = page.locator(".chat-item-title").first();
    await expect(chatTitle).toBeVisible();

    // Check hash is present
    await page.waitForFunction(() => window.location.hash.includes("chat-"));
    const hash = await page.evaluate(() => window.location.hash);
    expect(hash).toMatch(/^#chat-/);
  });

  test("Temporary chat fallback", async ({ page }) => {
    // Delete the default chat
    await page.click(".dropdown-btn");

    // Accept dialog
    page.on("dialog", (dialog) => dialog.accept());
    await page.click(".dropdown-item.danger"); // Delete

    // Should create a temporary chat
    await expect(page.locator(".chat-item-title").first()).toHaveText(
      "Temporary chat",
    );

    // Send message to convert it
    await page.fill("#chat-input", "Hello");
    await page.click("#send-btn");

    // Title should change
    await expect(page.locator(".chat-item-title").first()).toHaveText(
      /Chat #\d+/,
    );
  });

  test("Responsive behavior", async ({ page }) => {
    // Mobile view
    await page.setViewportSize({ width: 375, height: 667 });
    await expect(page.locator("#open-sidebar-btn")).toBeVisible();
    await page.click("#open-sidebar-btn");
    await expect(page.locator(".sidebar")).toBeVisible();
    await page.click("#close-sidebar-btn");

    // Tablet view
    await page.setViewportSize({ width: 768, height: 1024 });
    // Still mobile-like or up to breakpoint

    // Desktop view
    await page.setViewportSize({ width: 1280, height: 800 });
    await expect(page.locator("#open-sidebar-btn")).not.toBeVisible();

    // TV view
    await page.setViewportSize({ width: 3840, height: 2160 });
    await expect(page.locator(".chat-form")).toBeVisible();
  });

  test("Accessibility: --text-muted color contrast in dark mode meets WCAG AA", async ({
    page,
  }) => {
    // The empty state text uses --text-muted.
    const emptyState = page.locator(".empty-state").first();
    await expect(emptyState).toBeVisible();

    // Verify the computed color and background color to ensure contrast
    const styles = await emptyState.evaluate((el) => {
      const computed = window.getComputedStyle(el);
      // We also need the background of the body/parent to compute contrast
      const bg = window.getComputedStyle(document.body).backgroundColor;
      return {
        color: computed.color,
        bg: bg,
      };
    });

    // #bac2de is rgb(186, 194, 222)
    expect(styles.color).toBe("rgb(186, 194, 222)");
    // #1e1e2e is rgb(30, 30, 46)
    expect(styles.bg).toBe("rgb(30, 30, 46)");
  });

  test("Accessibility: Chat input has a persistent, visible label", async ({
    page,
  }) => {
    const label = page.locator('label[for="chat-input"]');
    await expect(label).toBeVisible();
    await expect(label).toHaveText("Type a message...");
    // Check it doesn't have the sr-only class
    await expect(label).not.toHaveClass(/sr-only/);
  });

  test("Accessibility: Empty state has role=status and aria-live=polite", async ({
    page,
  }) => {
    // Delete the chat to trigger empty state (no chats)
    await page.click(".dropdown-btn");
    page.on("dialog", (dialog) => dialog.accept());
    await page.click(".dropdown-item.danger");

    // The temporary chat is created automatically, which has 0 messages.
    // It should have the role="status"
    const emptyState = page.locator(".empty-state").first();
    await expect(emptyState).toBeVisible();
    await expect(emptyState).toHaveAttribute("role", "status");
    await expect(emptyState).toHaveAttribute("aria-live", "polite");
  });

  test("Accessibility: Chat input uses accessible transparent text hack", async ({
    page,
  }) => {
    const chatInput = page.locator("#chat-input");
    await expect(chatInput).toBeVisible();

    // Verify textarea color is not transparent
    const color = await chatInput.evaluate(
      (el) => window.getComputedStyle(el).color,
    );
    // Transparent color in computed style is usually rgba(0, 0, 0, 0)
    expect(color).not.toBe("rgba(0, 0, 0, 0)");
    expect(color).not.toBe("transparent");

    // Verify the highlight block is present
    await expect(page.locator("#chat-input-highlight")).toHaveCount(1);
  });
  test("Accessibility: complete accessibility audit", async ({ page }) => {
    // Send a message and show a table first so more UI is present
    await page.selectOption("#model-select", "sql");
    await page.fill("#chat-input", "SELECT * FROM test");
    await page.click("#send-btn");
    await expect(page.locator(".sql-table")).toBeVisible();

    const accessibilityScanResults = await new AxeBuilder({ page }).analyze();
    expect(accessibilityScanResults.violations).toEqual([]);
  });
});

test("Temporary chat is undeletable", async ({ page }) => {
  await page.goto("/");

  // Delete the initial chat
  await page.click(".dropdown-btn");
  page.on("dialog", (dialog) => dialog.accept());
  await page.click(".dropdown-item.danger");

  // Temporary chat appears
  await expect(page.locator(".chat-item-title").first()).toHaveText(
    "Temporary chat",
  );

  // Try to delete Temporary chat
  await page.click(".dropdown-btn");
  await page.click(".dropdown-item.danger"); // Delete

  // It should still exist
  await expect(page.locator(".chat-item-title").first()).toHaveText(
    "Temporary chat",
  );
});

test("Query history is saved to local storage and restored on reload", async ({
  page,
}) => {
  // Setup mock route to handle the chat response before navigation
  await page.route("**/api/chat", async (route) => {
    await route.fulfill({
      json: {
        content: "Here is the cached result:",
        sqlQuery: "SELECT * FROM history;",
        sqlResult: [{ cached: true }],
      },
    });
  });

  await page.goto("/");

  // Send a message
  await page.fill("#chat-input", "Save this to history");
  await page.click("#send-btn");

  // Wait for the response to appear
  await expect(page.locator(".message.assistant")).toContainText(
    "Here is the cached result:",
  );
  await expect(page.locator(".sql-query")).toHaveText("SELECT * FROM history;");

  // Reload the page to verify persistence
  await page.reload();

  // The chat should still be there, without sending another request
  await expect(page.locator(".chat-item-title").first()).toHaveText(
    /Chat #\d+/,
  );
  await expect(page.locator(".message.user")).toContainText(
    "Save this to history",
  );
  await expect(page.locator(".message.assistant")).toContainText(
    "Here is the cached result:",
  );
  await expect(page.locator(".sql-query")).toHaveText("SELECT * FROM history;");
});

test("Visually indicates the model used for an assistant message", async ({
  page,
}) => {
  // We will mock the API calls
  await page.route("**/api/chat", async (route) => {
    const request = route.request();
    const postData = JSON.parse(request.postData() || "{}");

    if (postData.model === "sql") {
      await route.fulfill({
        json: { content: "Query executed.", sqlResult: [] },
      });
    } else {
      await route.fulfill({
        json: {
          content: "Here is the result:",
          sqlQuery: "SELECT * FROM test;",
          sqlResult: [{ id: 1, name: "Test" }],
        },
      });
    }
  });

  await page.route("**/api/models", async (route) => {
    await route.fulfill({ json: { models: [{ name: "gemma4" }] } });
  });

  await page.goto("/");

  // Need to ensure the select element has its options
  await page.waitForFunction(() => {
    const select = document.querySelector("#model-select");
    return select && select.children.length > 1; // Wait for models to load
  });

  // Set the model to SQL to trigger the badge
  await page.selectOption("#model-select", "sql");

  // Send a message
  await page.fill("#chat-input", "SELECT * FROM test");
  await page.click("#send-btn");

  // Wait for the response and check the badge
  const badge = page.locator(".message.assistant .model-badge").first();
  await expect(badge).toBeVisible();
  await expect(badge).toHaveText("Raw SQL");

  // Now switch to gemma4 and test it.
  await expect(page.locator("#model-select")).toBeEnabled();

  // Select option value=gemma4
  await page.locator("#model-select").selectOption("gemma4");

  // Send a second message
  await page.fill("#chat-input", "Show me the patients");
  await page.click("#send-btn");

  // Verify the second badge says Model: gemma4
  const secondBadge = page.locator(".message.assistant .model-badge").nth(1);
  await expect(secondBadge).toBeVisible();
  await expect(secondBadge).toHaveText("Model: gemma4");
});

test("Chat input resizes dynamically with multi-line text and resets on send", async ({
  page,
}) => {
  await page.goto("/");

  const chatInput = page.locator("#chat-input");

  // Wait for it to be visible
  await expect(chatInput).toBeVisible();

  // Get initial height
  const initialBox = await chatInput.boundingBox();
  expect(initialBox).not.toBeNull();
  const initialHeight = initialBox!.height;

  // Type multiple lines
  await chatInput.fill("Line 1\nLine 2\nLine 3\nLine 4\nLine 5");

  // Need to dispatch input event or wait for reactivity, but playwright .fill() does this.
  // Wait a small amount for the DOM to update
  await page.waitForTimeout(100);

  // Get new height
  const expandedBox = await chatInput.boundingBox();
  expect(expandedBox).not.toBeNull();
  const expandedHeight = expandedBox!.height;

  // Assert it grew
  expect(expandedHeight).toBeGreaterThan(initialHeight);

  // Now clear it and see if it shrinks
  await chatInput.fill("");
  await page.waitForTimeout(100);

  const clearedBox = await chatInput.boundingBox();
  expect(clearedBox).not.toBeNull();
  expect(clearedBox!.height).toBeCloseTo(initialHeight, 1);

  // Type again, and test that sending resets it
  await chatInput.fill("Line 1\nLine 2");
  await page.waitForTimeout(100);
  const reExpandedBox = await chatInput.boundingBox();
  expect(reExpandedBox!.height).toBeGreaterThan(initialHeight);

  await page.click("#send-btn");
  await page.waitForTimeout(100);

  const sentBox = await chatInput.boundingBox();
  expect(sentBox).not.toBeNull();
  expect(sentBox!.height).toBeCloseTo(initialHeight, 1);
});

test("Cloud provider settings modal and model selection", async ({ page }) => {
  await page.goto("/");
  // 1. Open Provider settings modal
  await page.click("#provider-settings-btn");
  const modal = page.locator("#provider-modal");
  await expect(modal).toBeVisible();

  // 2. Fill in OpenAI key and save
  await page.fill("#provider-key-openai", "sk-e2e-test-key-12345");
  await page.click("#provider-save-btn");
  await expect(modal).toBeHidden();

  // 3. Verify key is preserved in localStorage
  const savedKey = await page.evaluate(() =>
    localStorage.getItem("t1d_api_key_openai"),
  );
  expect(savedKey).toBe("sk-e2e-test-key-12345");

  // 4. Open modal again and clear keys
  await page.click("#provider-settings-btn");
  await expect(modal).toBeVisible();
  await page.click("#provider-clear-btn");
  const clearedKey = await page.evaluate(() =>
    localStorage.getItem("t1d_api_key_openai"),
  );
  expect(clearedKey).toBeNull();
  await page.click("#close-provider-modal-btn");
  await expect(modal).toBeHidden();
});

test("Topbar language chooser and chat bubbles remain strictly within viewport after delayed long model/db/status updates and wide SQL results", async ({
  page,
}) => {
  await page.route("**/api/models", async (route) => {
    await new Promise((r) => setTimeout(r, 150));
    await route.fulfill({
      status: 200,
      json: {
        models: [
          {
            name: "anthropic/claude-3-5-sonnet-20241022-clinical-research-edition",
            provider: "anthropic",
            configured: false,
            requires_key: true,
          },
          { name: "gemma4", provider: "ollama", configured: true },
        ],
      },
    });
  });

  await page.route("**/api/databases", async (route) => {
    await new Promise((r) => setTimeout(r, 150));
    await route.fulfill({
      status: 200,
      json: {
        databases: [
          {
            name: "t1d_longitudinal_clinical_trial_cohort_full.duckdb",
            path: "t1d_longitudinal_clinical_trial_cohort_full.duckdb",
            size_bytes: 158420992,
          },
        ],
        current_db: "t1d_longitudinal_clinical_trial_cohort_full.duckdb",
      },
    });
  });

  await page.route("**/api/chat", async (route) => {
    await new Promise((r) => setTimeout(r, 150));
    const wideSql =
      "SELECT patient_id, tx_group, baseline_hba1c, follow_up_hba1c, mean_glucose_mg_dl, time_in_range_pct, hypoglycemia_events_count FROM longitudinal_clinical_cohort_summary WHERE baseline_hba1c > 7.5 ORDER BY time_in_range_pct ASC;";
    await route.fulfill({
      status: 200,
      json: {
        content: `Here is the wide clinical query:\n\`\`\`sql\n${wideSql}\n\`\`\``,
        sqlQuery: wideSql,
        sqlResult: [
          {
            patient_id: "PT-0001-STANFORD-COHORT-A",
            tx_group: "Automated Insulin Delivery (AID)",
            baseline_hba1c: 8.4,
            follow_up_hba1c: 6.8,
            mean_glucose_mg_dl: 142.5,
            time_in_range_pct: 76.4,
            hypoglycemia_events_count: 0,
          },
        ],
      },
    });
  });

  for (const vp of [
    { width: 1280, height: 800 },
    { width: 768, height: 1024 },
    { width: 375, height: 667 },
  ]) {
    await page.setViewportSize(vp);
    await page.goto("/");

    // Wait for delayed models and databases to populate
    await page.waitForTimeout(300);

    for (const selector of [
      "#lang-select",
      "#theme-toggle-btn",
      "#model-select",
      "#db-select",
      "#system-status-chip",
    ]) {
      const el = page.locator(selector);
      await expect(el).toBeVisible();
      const box = await el.boundingBox();
      expect(box).not.toBeNull();
      expect(box!.x).toBeGreaterThanOrEqual(0);
      expect(box!.x + box!.width).toBeLessThanOrEqual(vp.width + 1);
      expect(box!.width).toBeGreaterThanOrEqual(28);
      expect(box!.height).toBeGreaterThanOrEqual(24);
    }

    // Send a user message and verify chat bubbles do not shift off-screen after wide SQL response
    await page.fill("#chat-input", "Analyze cohort summary metrics");
    await page.click("#send-btn");

    const userBubble = page.locator(".message.user").last();
    const assistantBubble = page.locator(".message.assistant").last();
    await expect(userBubble).toBeVisible();
    await expect(assistantBubble.locator("table")).toBeVisible();

    const userBox = await userBubble.boundingBox();
    const assistantBox = await assistantBubble.boundingBox();
    expect(userBox).not.toBeNull();
    expect(assistantBox).not.toBeNull();
    expect(userBox!.x).toBeGreaterThanOrEqual(0);
    expect(userBox!.x + userBox!.width).toBeLessThanOrEqual(vp.width + 1);
    expect(assistantBox!.x).toBeGreaterThanOrEqual(0);
    expect(assistantBox!.x + assistantBox!.width).toBeLessThanOrEqual(
      vp.width + 1,
    );

    const overflowCheck = await page.evaluate(() => {
      const doc = document.documentElement;
      const mainPane = document.getElementById("main-pane")!;
      const msgContainer = document.getElementById("messages-container")!;
      return {
        docOverflow: doc.scrollWidth > doc.clientWidth,
        mainOverflow: mainPane.scrollWidth > mainPane.clientWidth,
        msgOverflow: msgContainer.scrollWidth > msgContainer.clientWidth,
      };
    });
    expect(overflowCheck.docOverflow).toBe(false);
    expect(overflowCheck.mainOverflow).toBe(false);
    expect(overflowCheck.msgOverflow).toBe(false);
  }
});

test("Initial chat window loads cleanly with zero errors and falls back to sql mode when Ollama is offline", async ({
  page,
}) => {
  await page.route("**/api/status", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      json: {
        status: "healthy",
        database: {
          status_code: "healthy",
          configured_path: "t1d.duckdb",
          exists: true,
          connected: true,
          table_count: 4,
          has_initial_data: true,
        },
        ollama: {
          accessible: true,
        },
      },
    });
  });

  await page.goto("/");

  // Verify zero error messages and hidden status banner on healthy initial load
  await expect(page.locator(".message.error")).toHaveCount(0);
  await expect(page.locator("#system-status-banner")).toHaveClass(/hidden/);
  await expect(page.locator("#system-status-chip")).toHaveClass(
    /status-healthy/,
  );

  // Verify initial default chat (Chat #1) is active with aria-current="page"
  const activeChatTitle = page.locator(
    "#chat-list .chat-item.active .chat-item-title",
  );
  await expect(activeChatTitle).toHaveText("Chat #1");
  await expect(activeChatTitle).toHaveAttribute("aria-current", "page");
  await expect(page.locator(".empty-state")).toBeVisible();
  await expect(page.locator("#chat-input")).toBeEnabled();

  // Now simulate initial load when local Ollama is offline and no cloud keys are configured
  await page.route("**/api/status", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      json: {
        status: "degraded",
        database: {
          status_code: "healthy",
          configured_path: "t1d.duckdb",
          exists: true,
          connected: true,
          table_count: 4,
          has_initial_data: true,
        },
        ollama: {
          accessible: false,
        },
      },
    });
  });
  await page.route("**/api/models", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      json: {
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
        ],
      },
    });
  });

  await page.evaluate(() => localStorage.clear());
  await page.reload();

  // Default untouched chat should automatically fall back to "sql" mode so queries succeed without LLM error
  await expect(page.locator("#model-select")).toHaveValue("sql");
  const sidebarFooter = page.locator("#sidebar #sidebar-footer");
  await expect(sidebarFooter).toBeVisible();
  await expect(sidebarFooter.locator("#build-timestamp")).toContainText(
    "built on ",
  );
  await expect(sidebarFooter.locator("#build-timestamp")).toContainText("UTC");
  await expect(sidebarFooter.locator("#sidebar-copyright")).toContainText(
    "© 2026 Samuel Marks",
  );
  await expect(sidebarFooter.locator("#sidebar-github-link")).toHaveAttribute(
    "href",
    "https://github.com/SamuelMarks/t1d-analytics",
  );
  await page.fill("#chat-input", "SELECT 1 AS ok;");
  await page.click("#send-btn");
  await expect(page.locator(".message.assistant")).toBeVisible();
  await expect(page.locator(".message.error")).toHaveCount(0);
});

test("Chat session lifecycle: create, keyboard/mouse switch, dropdown keyboard nav, rename, duplicate, delete, and hash deep-linking", async ({
  page,
}) => {
  await page.goto("/");

  // Create Chat #2 and Chat #3
  await page.click("#new-chat-btn");
  await page.click("#new-chat-btn");
  const chatItems = page.locator("#chat-list .chat-item");
  await expect(chatItems).toHaveCount(3);

  // Switch to Chat #1 via mouse click
  await chatItems.nth(0).locator(".chat-item-title").click();
  await expect(chatItems.nth(0)).toHaveClass(/active/);

  // Switch to Chat #2 via keyboard Enter and Chat #3 via Space
  const chat2Btn = chatItems.nth(1).locator(".chat-item-title");
  await chat2Btn.focus();
  await page.keyboard.press("Enter");
  await expect(chatItems.nth(1)).toHaveClass(/active/);

  const chat3Btn = chatItems.nth(2).locator(".chat-item-title");
  await chat3Btn.focus();
  await page.keyboard.press(" ");
  await expect(chatItems.nth(2)).toHaveClass(/active/);

  // Test dropdown keyboard navigation (ArrowDown, ArrowUp, Home, End, Escape)
  const dropdownBtn = chatItems.nth(2).locator(".dropdown-btn");
  await dropdownBtn.focus();
  await page.keyboard.press("ArrowDown");
  const menu = chatItems.nth(2).locator(".dropdown-menu");
  await expect(menu).toHaveClass(/show/);

  const menuItems = menu.locator('[role="menuitem"]');
  await expect(menuItems.nth(0)).toBeFocused();
  await page.keyboard.press("ArrowDown");
  await expect(menuItems.nth(1)).toBeFocused();
  await page.keyboard.press("End");
  await expect(menuItems.nth(2)).toBeFocused();
  await page.keyboard.press("Home");
  await expect(menuItems.nth(0)).toBeFocused();
  await page.keyboard.press("ArrowUp");
  await expect(menuItems.nth(2)).toBeFocused();
  await page.keyboard.press("Escape");
  await expect(menu).not.toHaveClass(/show/);
  await expect(dropdownBtn).toBeFocused();

  // Rename Chat #3 -> cancelled prompt keeps title, accepted prompt renames
  await dropdownBtn.click();
  page.once("dialog", async (dialog) => {
    await dialog.dismiss();
  });
  await menuItems.nth(0).click();
  await expect(chat3Btn).toHaveText("Chat #3");

  await dropdownBtn.click();
  page.once("dialog", async (dialog) => {
    await dialog.accept("Clinical Cohort Review");
  });
  await menuItems.nth(0).click();
  await expect(chat3Btn).toHaveText("Clinical Cohort Review");
  await expect(page).toHaveTitle(/Clinical Cohort Review/);

  // Duplicate Chat #3
  await dropdownBtn.click();
  await menuItems.nth(1).click();
  await expect(chatItems).toHaveCount(4);
  await expect(chatItems.nth(3).locator(".chat-item-title")).toHaveText(
    "Clinical Cohort Review (Copy)",
  );

  // Delete chats: dismiss confirm keeps chat, accept confirm deletes chat
  const copyDropdown = chatItems.nth(3).locator(".dropdown-btn");
  await copyDropdown.click();
  page.once("dialog", async (dialog) => {
    await dialog.dismiss();
  });
  await chatItems
    .nth(3)
    .locator('.dropdown-menu [role="menuitem"]')
    .nth(2)
    .click();
  await expect(chatItems).toHaveCount(4);

  // Delete all 4 chats to verify Temporary chat fallback and conversion on message send
  for (let remaining = 4; remaining >= 1; remaining--) {
    const targetItem = chatItems.nth(0);
    await targetItem.locator(".dropdown-btn").click();
    page.once("dialog", async (dialog) => {
      await dialog.accept();
    });
    await targetItem.locator('.dropdown-menu [role="menuitem"]').nth(2).click();
  }

  await expect(chatItems).toHaveCount(1);
  await expect(chatItems.nth(0).locator(".chat-item-title")).toHaveText(
    "Temporary chat",
  );

  // Sending a message in Temporary chat converts it to a regular numbered chat
  await page.fill("#chat-input", "SELECT 1;");
  await page.click("#send-btn");
  await expect(chatItems.nth(0).locator(".chat-item-title")).not.toHaveText(
    "Temporary chat",
  );

  // Test URL hash navigation between chats
  await page.click("#new-chat-btn");
  const ids = await page.evaluate(() => {
    const raw = localStorage.getItem("t1d_analytics_chats");
    if (!raw) return [] as string[];
    const parsed = JSON.parse(raw) as { chats: Array<{ id: string }> };
    return parsed.chats.map((c) => c.id);
  });
  expect(ids.length).toBe(2);

  await page.goto(`/#${ids[0]}`);
  await expect(chatItems.nth(0)).toHaveClass(/active/);
});
