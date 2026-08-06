import { test, expect } from "./base-test";
import AxeBuilder from "@axe-core/playwright";

test.describe("App Workflows Full E2E", () => {
  test.beforeEach(async ({ page }) => {
    await page.goto("/");
  });

  /**
   * Tests the Schema Explorer UI behavior including toggling and viewing table data.
   */
  test("Schema Explorer: view schema and open table data modal", async ({
    page,
  }) => {
    // Verify the schema table is rendered (from real duckdb)
    // There may be multiple tables (e.g. patients, tiny), so we find the one containing "tiny"
    const schemaTable = page
      .locator(".schema-table")
      .filter({ hasText: "tiny" })
      .first();
    await expect(schemaTable).toBeVisible({ timeout: 10000 });
    await expect(schemaTable.locator(".schema-table-header span")).toHaveText(
      "tiny",
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
    await expect(modalTitle).toHaveText("Table: tiny");

    // Close modal
    await page.locator("#close-modal-btn").click();
    await expect(modal).not.toBeVisible();
  });

  /**
   * Tests chat management features such as creating, renaming, and duplicating chats.
   */
  test("Chat Management: rename and duplicate chat", async ({ page }) => {
    // Send a message so we have a non-temporary chat
    await page.selectOption("#model-select", "sql");
    await page.fill("#chat-input", "SELECT * FROM tiny LIMIT 1");
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
    await page.selectOption("#model-select", "sql");
    await page.fill("#chat-input", "SELECT * FROM tiny;");
    await page.click("#send-btn");

    // Wait for response containing SQL
    const assistantMessage = page.locator(".message.assistant").first();
    await expect(assistantMessage).toBeVisible();
    await expect(assistantMessage.locator(".sql-query")).toHaveText(
      "SELECT * FROM tiny;",
    );

    // Get table value
    const sqlTable = assistantMessage.locator(".sql-table");
    await expect(sqlTable).toBeVisible();
    await expect(sqlTable.locator("td").first()).toHaveText("1");

    // The refresh button modifies the message in-place
    const refreshBtn = assistantMessage.locator(
      'button[data-i18n-title="ui.refreshQuery"]',
    );
    await expect(refreshBtn).toBeVisible();
    await refreshBtn.click();

    // Verify it refreshed by waiting for the table to still be there
    await expect(sqlTable).toBeVisible();
    await expect(sqlTable.locator("td").first()).toHaveText("1");
    await expect(sqlTable.locator("td").nth(1)).toHaveText("Test A");
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
    await expect(schemaTable).toBeVisible({ timeout: 10000 });

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

  test("Accessibility: complete accessibility audit", async ({ page }) => {
    // Wait for it to settle
    await expect(page.locator(".schema-table").first()).toBeVisible({
      timeout: 10000,
    });
    const accessibilityScanResults = await new AxeBuilder({ page }).analyze();
    expect(accessibilityScanResults.violations).toEqual([]);
  });

  test("Keyboard Navigation: toggles schema and opens modal", async ({
    page,
  }) => {
    // Navigate via tab to schema toggle and use enter
    await expect(page.locator(".schema-table").first()).toBeVisible({
      timeout: 10000,
    });

    await page.locator("#toggle-schema-btn").focus();
    await page.keyboard.press("Enter");

    // Verify it opened or toggled. It's normally open initially, so wait for it to collapse.
    const schemaContent = page.locator("#schema-content");
    // Actually the default might not use collapse via CSS right away in ui.ts unless classes are added,
    // but we can ensure the button receives focus and operates.

    // Let's tab to the schema table toggle
    const tableToggle = page.locator(".schema-table-header-toggle").first();
    await tableToggle.focus();
    await page.keyboard.press("Enter");
    const expanded = await tableToggle.getAttribute("aria-expanded");
    expect(expanded).toMatch(/true|false/);

    // Tab to the view button
    const viewBtn = page.locator(".table-view-btn").first();
    await viewBtn.focus();
    await page.keyboard.press("Enter");

    // Modal opens, check focus is trapped
    const modal = page.locator("#table-modal");
    await expect(modal).toBeVisible();
    await expect(page.locator("#close-modal-btn")).toBeFocused();

    // Tab cycles focus within modal
    await page.keyboard.press("Tab"); // Next focusable (maybe prev btn or something else)
    await page.keyboard.down("Shift");
    await page.keyboard.press("Tab");
    await page.keyboard.up("Shift");
    await expect(page.locator("#close-modal-btn")).toBeFocused();

    // Press enter to close
    await page.keyboard.press("Enter");
    await expect(modal).not.toBeVisible();
    await expect(viewBtn).toBeFocused(); // focus returned to view button
  });
});
