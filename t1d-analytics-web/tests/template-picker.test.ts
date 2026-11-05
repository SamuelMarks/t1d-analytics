/**
 * @file template-picker.test.ts
 * @module tests/template-picker.test
 * Exhaustive unit tests for TemplatePickerController, mid-conversation template switching,
 * Natural-Language-to-SQL mode, Gold SQL insertion, LLM vs Gold benchmarking, and keyboard shortcuts.
 */

import { beforeEach, describe, expect, it, vi } from "vitest";
import { TemplatePickerController } from "../src/controllers/template-picker-controller";
import { resolvePreferredLlmModel } from "../src/empty-state-builder";
import { ChatState, RAW_SQL_MODEL } from "../src/state";
import { ChatUI } from "../src/ui";
import {
  flushPromises,
  getTestElement,
  mountChatUiTestDom,
  queryAllTestElements,
  queryTestElement,
} from "./setup";

describe("TemplatePickerController & Empty-State Template Mode Toggle", () => {
  let state: ChatState;
  let ui: ChatUI;

  beforeEach(() => {
    localStorage.clear();
    mountChatUiTestDom();
    global.fetch = vi.fn().mockImplementation(async (url: string) => {
      if (url.includes("/api/models")) {
        return {
          ok: true,
          json: async () => ({
            models: [{ name: "gemma:2b", provider: "ollama" }],
          }),
        };
      }
      if (url.includes("/api/schema")) {
        return {
          ok: true,
          json: async () => ({
            database: { status_code: "ok", exists: true, connected: true },
            tables: [
              {
                name: "tblaptsummary",
                columns: [{ name: "PtID", type: "INT" }],
              },
            ],
          }),
        };
      }
      return {
        ok: true,
        json: async () => ({
          content: "SELECT * FROM tblaptsummary LIMIT 5;",
          sqlQuery: "SELECT * FROM tblaptsummary LIMIT 5;",
          sqlResult: [{ PtID: 1 }],
        }),
      };
    });
    state = new ChatState();
    state.createChat();
    ui = new ChatUI(state);
  });

  it("opens and closes TemplatePickerModal via button, backdrop click, Escape, and Ctrl+/", () => {
    const openBtn = getTestElement<HTMLButtonElement>("template-picker-btn");
    const modal = getTestElement<HTMLElement>("template-picker-modal");
    const closeBtn = getTestElement<HTMLButtonElement>(
      "close-template-modal-btn",
    );

    expect(modal.classList.contains("hidden")).toBe(true);
    openBtn.focus();
    openBtn.click();
    expect(modal.classList.contains("hidden")).toBe(false);
    expect(modal.getAttribute("aria-hidden")).toBe("false");

    const cards = queryAllTestElements<HTMLElement>(
      ".template-picker-card",
      modal,
    );
    expect(cards.length).toBe(12); // 5 starters + 7 Table 1

    // Close via close button restores focus
    closeBtn.click();
    expect(modal.classList.contains("hidden")).toBe(true);
    expect(document.activeElement).toBe(openBtn);

    // Open via Ctrl+/ keyboard shortcut and close via Ctrl+/
    document.dispatchEvent(
      new KeyboardEvent("keydown", { key: "/", ctrlKey: true, bubbles: true }),
    );
    expect(modal.classList.contains("hidden")).toBe(false);

    // Escape key inside modal closes it
    modal.dispatchEvent(
      new KeyboardEvent("keydown", { key: "Escape", bubbles: true }),
    );
    expect(modal.classList.contains("hidden")).toBe(true);

    // Keydown while modal is hidden is ignored
    modal.dispatchEvent(
      new KeyboardEvent("keydown", { key: "Tab", bubbles: true }),
    );

    // Non-matching global keydown is ignored
    document.dispatchEvent(
      new KeyboardEvent("keydown", { key: "k", ctrlKey: true, bubbles: true }),
    );

    // Click inside modal content (not backdrop) does not close modal
    if (document.activeElement instanceof HTMLElement) {
      document.activeElement.blur();
    }
    ui.templatePickerController.openModal();
    const firstCard = queryTestElement<HTMLElement>(
      ".template-picker-card",
      modal,
    );
    firstCard.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    expect(modal.classList.contains("hidden")).toBe(false);

    // Backdrop click closes modal and focuses chatInput when previousFocusEl was null (body was focused)
    modal.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    expect(modal.classList.contains("hidden")).toBe(true);
    expect(document.activeElement).toBe(ui.chatInput);

    // Test updateAutoExecuteToggleUi when #auto-execute-toggle-btn is absent from DOM
    const autoBtnEl = document.getElementById("auto-execute-toggle-btn");
    autoBtnEl?.remove();
    const uiWithoutAutoBtn = new ChatUI(state);
    expect(uiWithoutAutoBtn.autoExecuteToggleBtn).toBeNull();
    uiWithoutAutoBtn.updateAutoExecuteToggleUi();
    if (autoBtnEl) {
      document.querySelector(".header-actions")?.prepend(autoBtnEl);
    }
    ui = new ChatUI(state);

    // Cmd+/ toggles open and closed
    document.dispatchEvent(
      new KeyboardEvent("keydown", { key: "/", metaKey: true, bubbles: true }),
    );
    expect(modal.classList.contains("hidden")).toBe(false);
    document.dispatchEvent(
      new KeyboardEvent("keydown", { key: "/", metaKey: true, bubbles: true }),
    );
    expect(modal.classList.contains("hidden")).toBe(true);
  });

  it("filters templates by category tabs and executes Try NL, Insert Gold SQL, and Benchmark LLM actions mid-conversation", async () => {
    // Add an existing message so we are mid-conversation (empty state is gone)
    state.addMessageToActiveChat({
      role: "user",
      content: "Initial question",
      model: "gemma:2b",
    });
    ui.render();
    expect(document.querySelector(".empty-state")).toBeNull();

    // Open Template Picker mid-conversation
    const openBtn = getTestElement<HTMLButtonElement>("template-picker-btn");
    openBtn.click();
    const modal = getTestElement<HTMLElement>("template-picker-modal");

    // Filter by Table 1 tab
    const table1Tab = queryTestElement<HTMLButtonElement>(
      '[data-template-tab="table1"]',
      modal,
    );
    table1Tab.click();
    expect(ui.templatePickerController.activeCategory).toBe("table1");
    expect(
      queryAllTestElements<HTMLElement>(".template-picker-card", modal).length,
    ).toBe(7);

    // Filter by Starter tab
    const starterTab = queryTestElement<HTMLButtonElement>(
      '[data-template-tab="starter"]',
      modal,
    );
    starterTab.click();
    expect(ui.templatePickerController.activeCategory).toBe("starter");
    expect(
      queryAllTestElements<HTMLElement>(".template-picker-card", modal).length,
    ).toBe(5);

    // Filter back to All tab
    const allTab = queryTestElement<HTMLButtonElement>(
      '[data-template-tab="all"]',
      modal,
    );
    allTab.click();
    expect(
      queryAllTestElements<HTMLElement>(".template-picker-card", modal).length,
    ).toBe(12);

    // 1. Click "Try NL -> SQL" on a card while modelSelect is "sql"
    ui.modelSelect.value = RAW_SQL_MODEL;
    const firstTryNlBtn = queryTestElement<HTMLButtonElement>(
      ".template-try-nl-btn",
      modal,
    );
    firstTryNlBtn.click();
    expect(modal.classList.contains("hidden")).toBe(true);
    expect(ui.modelSelect.value).toBe("gemma:2b");
    expect(state.activeGoldSql).toBeTruthy();
    expect(ui.chatInput.value).toContain("Show patient count");

    // 2. Re-open and click "Insert Gold SQL"
    openBtn.click();
    const firstInsertSqlBtn = queryTestElement<HTMLButtonElement>(
      ".template-insert-sql-btn",
      modal,
    );
    firstInsertSqlBtn.click();
    expect(ui.modelSelect.value).toBe(RAW_SQL_MODEL);
    expect(state.activeGoldSql).toBeNull();
    expect(ui.chatInput.value).toContain("SELECT");

    // 3. Re-open and click "Benchmark LLM vs Gold"
    openBtn.click();
    const firstBenchmarkBtn = queryTestElement<HTMLButtonElement>(
      ".template-benchmark-btn",
      modal,
    );
    firstBenchmarkBtn.click();
    await flushPromises();
    const activeMsgs = state.getActiveChat()?.messages ?? [];
    expect(activeMsgs.length).toBe(3);
    expect(activeMsgs[1]?.goldSql).toBeTruthy();
  });

  it("handles resolvePreferredLlmModel fallbacks and missing modal DOM gracefully", () => {
    const select = document.createElement("select");
    const sqlOpt = document.createElement("option");
    sqlOpt.value = RAW_SQL_MODEL;
    select.appendChild(sqlOpt);
    expect(resolvePreferredLlmModel(select)).toBe(RAW_SQL_MODEL);

    const customOpt = document.createElement("option");
    customOpt.value = "llama3.2:1b";
    select.appendChild(customOpt);
    select.value = RAW_SQL_MODEL;
    expect(resolvePreferredLlmModel(select)).toBe("llama3.2:1b");

    select.value = "llama3.2:1b";
    expect(resolvePreferredLlmModel(select)).toBe("llama3.2:1b");

    // Controller when modal elements are absent from DOM
    document.getElementById("template-picker-modal")?.remove();
    document.getElementById("template-picker-btn")?.remove();
    document.getElementById("close-template-modal-btn")?.remove();
    document.getElementById("template-picker-list")?.remove();
    const headlessCtrl = new TemplatePickerController(ui);
    headlessCtrl.openModal();
    headlessCtrl.closeModal();
    headlessCtrl.setCategoryFilter("table1");
    headlessCtrl.renderTemplateList();
    headlessCtrl.dispose();
  });

  it("toggles empty-state Table 1 chips between Gold SQL mode and Natural-Language-to-SQL mode", () => {
    const nlPill = queryTestElement<HTMLButtonElement>('[data-chip-mode="nl"]');
    const sqlPill = queryTestElement<HTMLButtonElement>(
      '[data-chip-mode="sql"]',
    );
    const firstTable1Chip = queryTestElement<HTMLButtonElement>(".table1-chip");

    // Switch to NL mode and click Table 1 chip
    nlPill.click();
    expect(nlPill.getAttribute("aria-pressed")).toBe("true");
    ui.modelSelect.value = RAW_SQL_MODEL;
    firstTable1Chip.click();
    expect(ui.modelSelect.value).toBe("gemma:2b");
    expect(state.activeGoldSql).toContain("SELECT");
    expect(ui.chatInput.value).not.toContain("SELECT");

    // Switch back to Gold SQL mode and click Table 1 chip
    sqlPill.click();
    expect(sqlPill.getAttribute("aria-pressed")).toBe("true");
    firstTable1Chip.click();
    expect(ui.modelSelect.value).toBe(RAW_SQL_MODEL);
    expect(state.activeGoldSql).toBeNull();
    expect(ui.chatInput.value).toContain("SELECT");
  });
});
