import { test, expect } from "@playwright/test";

test.describe("Chat UI E2E", () => {
  test.beforeEach(async ({ page }) => {
    // Add models endpoint mock for all tests
    await page.route("http://localhost:8000/api/models", async (route) => {
      await route.fulfill({
        json: { models: [{ name: "gemma4" }] },
      });
    });
    // We will mock the API calls
    await page.route("http://localhost:8000/api/chat", async (route) => {
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
  await page.route("http://localhost:8000/api/chat", async (route) => {
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
  await expect(page.locator(".message.user")).toHaveText(
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
  await page.route("http://localhost:8000/api/chat", async (route) => {
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

  await page.route("http://localhost:8000/api/models", async (route) => {
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
