import { test, expect } from "@playwright/test";

test.describe("App Workflows E2E", () => {
  test.beforeEach(async ({ page }) => {
    // Mock the models API
    await page.route("**/api/models", async (route) => {
      await route.fulfill({
        json: { models: [{ name: "gemma4" }] },
      });
    });

    // Mock the schema API
    await page.route("**/api/schema", async (route) => {
      await route.fulfill({
        json: {
          tables: [
            {
              name: "users",
              columns: [
                { name: "id", type: "integer" },
                { name: "username", type: "varchar" },
              ],
            },
          ],
        },
      });
    });

    // Mock chat API
    await page.route("**/api/chat", async (route) => {
      await route.fulfill({
        json: {
          content: "Here is the SQL query:",
          sqlQuery: "SELECT * FROM users;",
          sqlResult: [], // no results executed yet or maybe empty
        },
      });
    });

    // Mock execute-sql API
    await page.route("**/api/execute-sql", async (route) => {
      await route.fulfill({
        json: {
          sqlResult: [{ id: 1, username: "admin" }],
        },
      });
    });

    // Mock table data API for Schema Explorer
    await page.route("**/api/table/*", async (route) => {
      const url = new URL(route.request().url());
      const offset = parseInt(url.searchParams.get("offset") || "0", 10);

      let rows: any[] = [];
      if (offset === 0) {
        rows = Array.from({ length: 25 }, (_, i) => ({
          id: i + 1,
          username: `user${i + 1}`,
        }));
      } else {
        rows = Array.from({ length: 5 }, (_, i) => ({
          id: offset + i + 1,
          username: `user${offset + i + 1}`,
        }));
      }

      await route.fulfill({
        json: {
          rows,
          total: 30,
        },
      });
    });

    await page.goto("/");
  });

  /**
   * Tests the Schema Explorer UI behavior including toggling and viewing table data.
   */
  test("Schema Explorer: view schema and open table data modal", async ({
    page,
  }) => {
    // Verify the schema table is rendered
    const schemaTable = page.locator(".schema-table").first();
    await expect(schemaTable).toBeVisible();
    await expect(schemaTable.locator(".schema-table-header span")).toHaveText(
      "users",
    );

    // Expand the table columns view
    await schemaTable.locator(".schema-table-header-toggle").click();
    await expect(schemaTable.locator(".schema-columns")).toBeVisible();
    await expect(schemaTable.locator(".schema-column").first()).toContainText(
      "id",
    );

    // Open table data modal
    await schemaTable.locator(".table-view-btn").click();
    const modal = page.locator("#table-modal");
    await expect(modal).toBeVisible();

    // Verify modal content
    const modalTitle = page.locator("#modal-title");
    await expect(modalTitle).toHaveText("Table: users");

    // Verify pagination controls
    const nextPageBtn = page.locator("#next-page-btn");
    await expect(nextPageBtn).toBeEnabled();

    await nextPageBtn.click();
    await expect(page.locator("#page-indicator")).toHaveText("Page 2");

    // Close modal
    await page.locator("#close-modal-btn").click();
    await expect(modal).not.toBeVisible();
  });

  /**
   * Tests chat management features such as creating, renaming, and duplicating chats.
   */
  test("Chat Management: rename and duplicate chat", async ({ page }) => {
    // Send a message so we have a non-temporary chat
    await page.fill("#chat-input", "Hello");
    await page.click("#send-btn");

    // Wait for chat item to appear in the list
    const chatItem = page.locator(".chat-item").first();
    await expect(chatItem).toBeVisible();

    // The chat list gets populated asynchronously
    await page.waitForFunction(() => {
      const items = document.querySelectorAll(".chat-item");
      return (
        items.length > 0 && items[0].querySelector(".chat-item-title") !== null
      );
    });

    // Open dropdown menu
    await chatItem.locator(".dropdown-btn").click();

    // Prepare for prompt dialog
    page.once("dialog", (dialog) => {
      dialog.accept("My Custom Chat");
    });

    // Rename
    await chatItem
      .locator(".dropdown-item")
      .filter({ hasText: "Rename" })
      .click();
    await expect(chatItem.locator(".chat-item-title")).toHaveText(
      "My Custom Chat",
    );

    // Duplicate
    await chatItem.locator(".dropdown-btn").click();
    await chatItem
      .locator(".dropdown-item")
      .filter({ hasText: "Duplicate" })
      .click();

    // Now there should be two chats
    const chats = page.locator(".chat-item");
    await expect(chats).toHaveCount(2);
    await expect(chats.nth(1).locator(".chat-item-title")).toHaveText(
      "My Custom Chat (Copy)",
    );
  });

  /**
   * Tests executing SQL manually via the Play button on an assistant message.
   */
  test("SQL Execution: execute query manually from assistant message", async ({
    page,
  }) => {
    await page.fill("#chat-input", "Show me users");
    await page.click("#send-btn");

    // Wait for response containing SQL
    const assistantMessage = page.locator(".message.assistant").first();
    await expect(assistantMessage).toBeVisible();
    await expect(assistantMessage.locator(".sql-query")).toHaveText(
      "SELECT * FROM users;",
    );

    // Click the play button
    const playBtn = assistantMessage
      .locator("button")
      .filter({ hasText: "Play" });
    await expect(playBtn).toBeVisible();
    await playBtn.click();

    // Verify the query result table appears in the NEW assistant message
    const newAssistantMessage = page.locator(".message.assistant").nth(1);
    await expect(newAssistantMessage).toBeVisible();

    const sqlTable = newAssistantMessage.locator(".sql-table");
    await expect(sqlTable).toBeVisible();
    await expect(sqlTable.locator("td").first()).toHaveText("1");
    await expect(sqlTable.locator("td").nth(1)).toHaveText("admin");
  });

  /**
   * Tests the theme toggling functionality (Dark/Light mode).
   */
  test("Theme Toggle: switches between light and dark mode", async ({
    page,
  }) => {
    const themeBtn = page.locator("#theme-toggle-btn");
    const body = page.locator("body");

    await expect(body).not.toHaveClass(/light-mode/);
    await themeBtn.click();
    await expect(body).toHaveClass(/light-mode/);
    await themeBtn.click();
    await expect(body).not.toHaveClass(/light-mode/);
  });

  /**
   * Tests language selection updates the UI text asynchronously.
   */
  test("Language Selection: changing language updates i18n texts", async ({
    page,
  }) => {
    const langSelect = page.locator("#lang-select");

    // Switch to Japanese
    await langSelect.selectOption("ja");

    // The new chat button text should change away from "+ New Chat"
    const newChatBtn = page.locator("#new-chat-btn");
    await expect(newChatBtn).not.toHaveText("+ New Chat", { timeout: 5000 });
  });

  /**
   * Tests keyboard shortcut Shift+Enter for newline in chat input.
   */
  test("Chat Input: Shift+Enter adds newline, Enter sends", async ({
    page,
  }) => {
    const chatInput = page.locator("#chat-input");

    // Wait for models to load and enable input
    await expect(chatInput).toBeEnabled();

    // Use page.keyboard to simulate Shift+Enter properly
    await chatInput.click();
    await page.keyboard.type("Line 1");
    await page.keyboard.down("Shift");
    await page.keyboard.press("Enter");
    await page.keyboard.up("Shift");
    await page.keyboard.type("Line 2");

    // The value should contain a newline
    const value = await chatInput.inputValue();
    expect(value).toBe("Line 1\nLine 2");

    // Now press Enter without shift to send
    await page.keyboard.press("Enter");

    // Should have sent the message
    await expect(page.locator(".message.user").first()).toContainText(
      "Line 1\nLine 2",
    );
  });

  test("Modal Close: restores focus correctly or falls back to chat input", async ({
    page,
  }) => {
    // Open the schema explorer and click the table view button
    const schemaTable = page.locator(".schema-table").first();
    await expect(schemaTable).toBeVisible();

    const tableViewBtn = schemaTable.locator(".table-view-btn");
    await tableViewBtn.focus();
    await tableViewBtn.click();

    const modal = page.locator("#table-modal");
    await expect(modal).toBeVisible();

    // Now let's dynamically remove the button that opened the modal to simulate it disappearing from DOM
    await page.evaluate(() => {
      document.querySelector(".table-view-btn")?.remove();
    });

    // Close the modal
    await page.locator("#close-modal-btn").click();
    await expect(modal).not.toBeVisible();

    // Since previousFocus is gone, it should fallback to chat input
    const chatInput = page.locator("#chat-input");
    await expect(chatInput).toBeFocused();
  });
});
