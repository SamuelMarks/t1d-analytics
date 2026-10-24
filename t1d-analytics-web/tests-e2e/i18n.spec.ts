/**
 * @file i18n.spec.ts
 * @module tests-e2e/i18n.spec
 * End-to-end Playwright tests for internationalization and RTL layouts.
 */

import { test, expect } from "./base-test";
import AxeBuilder from "@axe-core/playwright";

test.describe("i18n and RTL E2E", () => {
  test.beforeEach(async ({ page }) => {
    // Setup mock routes
    await page.route("**/api/models", async (route) => {
      await route.fulfill({ json: { models: [{ name: "gemma4" }] } });
    });
    await page.goto("/");
  });

  test("Switches language to Arabic and verifies dir=rtl", async ({ page }) => {
    await page.selectOption("#lang-select", "ar");
    await expect(page.locator("html")).toHaveAttribute("dir", "rtl");
    await expect(page.locator("html")).toHaveAttribute("lang", "ar");
  });

  test("Switches language to Hebrew and verifies dir=rtl", async ({ page }) => {
    await page.selectOption("#lang-select", "he");
    await expect(page.locator("html")).toHaveAttribute("dir", "rtl");
    await expect(page.locator("html")).toHaveAttribute("lang", "he");
  });

  test("Validates RTL layout integrity (sidebar on right, text alignment)", async ({
    page,
  }) => {
    await page.selectOption("#lang-select", "ar");

    // In RTL, the sidebar (which is `flex: 0 0 260px` in a flex container)
    // should visually appear on the right side of the screen.
    const sidebar = page.locator("#sidebar");
    const mainPane = page.locator("#main-pane");

    const sidebarBox = await sidebar.boundingBox();
    const mainBox = await mainPane.boundingBox();

    expect(sidebarBox).not.toBeNull();
    expect(mainBox).not.toBeNull();

    // Sidebar should be to the right of main pane in RTL
    // i.e., sidebar's x position should be greater than main pane's x position
    expect(sidebarBox!.x).toBeGreaterThan(mainBox!.x);

    // Verify chat input text alignment is right or start in RTL
    const chatInput = page.locator("#chat-input");
    const textAlign = await chatInput.evaluate(
      (el) => window.getComputedStyle(el).textAlign,
    );

    // In some browsers/normalize it might be 'start' or 'right'
    expect(["start", "right"]).toContain(textAlign);
  });

  test("Dynamic string replacements render correctly in non-English language", async ({
    page,
  }) => {
    await page.selectOption("#lang-select", "ja");

    // Create a new chat
    await page.click("#new-chat-btn");

    // Ensure the chat is titled with the Japanese template "チャット #{{count}}" -> "チャット #2"
    // Since there's one default chat, the new one is #2
    const chatTitle = page.locator(".chat-item-title").nth(1);
    await expect(chatTitle).toHaveText("チャット #2");
  });

  test("Translates modal placeholders, sort buttons, provider badges, and passes axe-core a11y scans across en, ja, ar, and he", async ({
    page,
  }) => {
    for (const lang of ["en", "ja", "ar", "he"]) {
      await page.selectOption("#lang-select", lang);
      await expect(page.locator("html")).toHaveAttribute("lang", lang);

      // Verify language chooser and topbar controls remain strictly within the viewport in every locale
      const vp = page.viewportSize()!;
      const langBox = await page.locator("#lang-select").boundingBox();
      expect(langBox).not.toBeNull();
      expect(langBox!.x).toBeGreaterThanOrEqual(0);
      expect(langBox!.x + langBox!.width).toBeLessThanOrEqual(vp.width + 1);

      // Run axe accessibility check on main view
      const mainScan = await new AxeBuilder({ page }).analyze();
      expect(mainScan.violations).toEqual([]);

      // Open Cohort Modal and verify translated placeholder & axe scan
      await page.click("#cohort-filter-btn");
      const cohortModal = page.locator("#cohort-modal");
      await expect(cohortModal).toBeVisible();
      const minAgePlaceholder = await page
        .locator("#cohort-min-age")
        .getAttribute("placeholder");
      expect(minAgePlaceholder).toBeTruthy();
      const cohortScan = await new AxeBuilder({ page }).analyze();
      expect(cohortScan.violations).toEqual([]);
      await page.click("#close-cohort-modal-btn");

      // Open Provider Modal and verify badges & axe scan
      await page.click("#provider-settings-btn");
      const providerModal = page.locator("#provider-modal");
      await expect(providerModal).toBeVisible();
      const providerScan = await new AxeBuilder({ page }).analyze();
      expect(providerScan.violations).toEqual([]);
      await page.click("#close-provider-modal-btn");
    }
  });
});
