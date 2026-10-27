/**
 * @file provider-modal.test.ts
 * @module tests/provider-modal.test
 * Unit tests for ProviderSettingsModalController, API key persistence, model grouping, and provider status indicators.
 */

import { describe, it, expect, beforeEach, vi, afterEach } from "vitest";
import i18next from "i18next";
import { ChatState } from "../src/state";
import { ChatUI, ProviderSettingsModalController } from "../src/ui";
import { flushPromises, getTestElement, mountChatUiTestDom } from "./setup";

describe("ProviderSettingsModalController & Model Loading", () => {
  let state: ChatState;
  let ui: ChatUI;
  let mockFetch: ReturnType<typeof vi.fn>;

  beforeEach(async () => {
    mountChatUiTestDom();
    mockFetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ models: [], schema: [] }),
    });
    vi.stubGlobal("fetch", mockFetch);

    state = new ChatState();
    i18next.changeLanguage("en");
    document.documentElement.lang = "en";
    ui = new ChatUI(state);
    await flushPromises();
  });

  afterEach(() => {
    ui.destroy();
    localStorage.clear();
    document.body.innerHTML = "";
    vi.restoreAllMocks();
  });

  it("handles provider settings modal open, key updates, and clear", async () => {
    expect(ui.providerModalController).toBeInstanceOf(
      ProviderSettingsModalController,
    );
    await ui.providerModalController.open();
    ui.providerModalController.close();

    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        providers: [
          { provider: "ollama", configured: true, healthy: true },
          { provider: "openai", configured: true, healthy: true },
          { provider: "anthropic", configured: false, healthy: false },
          { provider: "google", configured: false, healthy: false },
        ],
      }),
    });

    const settingsBtn = getTestElement<HTMLButtonElement>(
      "provider-settings-btn",
    );
    const modal = getTestElement("provider-modal");
    const closeBtn = getTestElement<HTMLButtonElement>(
      "close-provider-modal-btn",
    );
    const saveBtn = getTestElement<HTMLButtonElement>("provider-save-btn");
    const clearBtn = getTestElement<HTMLButtonElement>("provider-clear-btn");
    const keyOpenAI = getTestElement<HTMLInputElement>("provider-key-openai");

    expect(modal.classList.contains("hidden")).toBe(true);

    settingsBtn.click();
    await flushPromises();
    expect(modal.classList.contains("hidden")).toBe(false);

    const consent = getTestElement<HTMLInputElement>("provider-store-consent");
    consent.checked = false;
    keyOpenAI.value = "sk-custom-openai-123";
    saveBtn.click();
    await flushPromises();
    expect(state.getApiKey("openai")).toBeNull();

    consent.checked = true;
    keyOpenAI.value = "sk-custom-openai-123";
    saveBtn.click();
    await flushPromises();
    expect(state.getApiKey("openai")).toBe("sk-custom-openai-123");
    expect(modal.classList.contains("hidden")).toBe(true);

    settingsBtn.click();
    await flushPromises();
    expect(keyOpenAI.value).toBe("sk-custom-openai-123");

    clearBtn.click();
    await flushPromises();
    expect(state.getApiKey("openai")).toBeNull();
    expect(keyOpenAI.value).toBe("");

    closeBtn.click();
    expect(modal.classList.contains("hidden")).toBe(true);

    settingsBtn.click();
    await flushPromises();
    expect(modal.classList.contains("hidden")).toBe(false);
    modal.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    expect(modal.classList.contains("hidden")).toBe(true);

    // Saving anthropic and google keys and local badge fallbacks
    const keyAnthropic = getTestElement<HTMLInputElement>(
      "provider-key-anthropic",
    );
    const keyGoogle = getTestElement<HTMLInputElement>("provider-key-google");
    keyAnthropic.value = "anthropic-secret-key";
    keyGoogle.value = "google-secret-key";

    await ui.saveProviderSettings();
    expect(state.getApiKey("anthropic")).toBe("anthropic-secret-key");
    expect(state.getApiKey("google")).toBe("google-secret-key");

    mockFetch.mockResolvedValueOnce({ ok: false, status: 500 });
    await ui.openProviderModal();
    const anthropicBadge = document.getElementById("provider-badge-anthropic");
    expect(anthropicBadge?.textContent).toContain("Configured 🟢");

    mockFetch.mockRejectedValueOnce(new Error("Network disconnect"));
    await ui.openProviderModal();
  });

  it("renders grouped models, custom configured models, and updates provider status dot", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        models: [
          { name: "gemma4", provider: "ollama", configured: true },
          {
            name: "openai/gpt-4o",
            provider: "openai",
            configured: false,
            requires_key: true,
          },
          {
            name: "anthropic/claude-3-5-sonnet",
            provider: "anthropic",
            configured: false,
            requires_key: true,
          },
        ],
      }),
    });

    await ui.loadModels();

    const modelSelect = getTestElement<HTMLSelectElement>("model-select");
    const optgroups = modelSelect.querySelectorAll("optgroup");
    expect(optgroups.length).toBeGreaterThan(0);

    const dot = getTestElement("provider-status-dot");

    modelSelect.value = "gemma4";
    modelSelect.dispatchEvent(new Event("change"));
    expect(dot.classList.contains("active")).toBe(true);

    modelSelect.value = "openai/gpt-4o";
    modelSelect.dispatchEvent(new Event("change"));
    expect(dot.classList.contains("warning")).toBe(true);

    state.setApiKey("openai", "sk-valid-key");
    ui.updateProviderBadgeStatus();
    expect(dot.classList.contains("active")).toBe(true);
    state.removeApiKey("openai");

    // Unreachable model and model with slash
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        models: [
          {
            name: "custom_unreachable",
            reachable: false,
            available: false,
            error_code: "CONN_ERR",
          },
          {
            name: "custom_prov/model-x",
            available: true,
          },
        ],
      }),
    });
    await ui.loadModels();
    const unreachableOpt = Array.from(modelSelect.options).find(
      (o) => o.value === "custom_unreachable",
    );
    expect(unreachableOpt?.textContent).toContain("🔴");
    expect(unreachableOpt?.title).toBe("custom_unreachable (CONN_ERR)");

    modelSelect.innerHTML = `<option value="openai/custom-gpt">custom-gpt</option>`;
    modelSelect.value = "openai/custom-gpt";
    ui.updateProviderBadgeStatus();
    expect(dot.classList.contains("warning")).toBe(true);

    // Custom configured any-llm models
    state.setConfiguredModels(
      "groq:llama-3.3-70b-versatile, deepseek/deepseek-chat, custom-local-ollama, openai/gpt-4o",
    );
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        models: [
          { name: "openai/gpt-4o", provider: "openai", configured: true },
        ],
      }),
    });
    await ui.loadModels();
    const optionValues = Array.from(modelSelect.options).map((o) => o.value);
    expect(optionValues).toContain("sql");
    expect(optionValues).toContain("groq/llama-3.3-70b-versatile");
    expect(optionValues).toContain("deepseek/deepseek-chat");
    expect(optionValues).toContain("custom-local-ollama");
    state.removeConfiguredModels();

    // Test without provider-status-dot
    dot.remove();
    const uiWithoutDot = new ChatUI(state);
    expect(() => uiWithoutDot.updateProviderBadgeStatus()).not.toThrow();
    uiWithoutDot.destroy();
  });

  it("handles provider modal Escape key dismissal, focus restoration, and missing elements", async () => {
    document.body.focus();
    await ui.openProviderModal();
    ui.closeProviderModal();

    const trigger = getTestElement<HTMLButtonElement>("provider-settings-btn");
    trigger.focus();

    await ui.openProviderModal();
    const modal = getTestElement("provider-modal");
    expect(modal.classList.contains("hidden")).toBe(false);
    expect(modal.getAttribute("aria-hidden")).toBeNull();

    const escEvent = new KeyboardEvent("keydown", {
      key: "Escape",
      bubbles: true,
      cancelable: true,
    });
    modal.dispatchEvent(escEvent);

    expect(modal.classList.contains("hidden")).toBe(true);
    expect(modal.getAttribute("aria-hidden")).toBe("true");
    expect(document.activeElement).toBe(trigger);

    const tempTrigger = document.createElement("button");
    document.body.appendChild(tempTrigger);
    tempTrigger.focus();
    await ui.openProviderModal();
    tempTrigger.remove();
    ui.closeProviderModal();
    expect(document.activeElement).toBe(trigger);

    await ui.openProviderModal();
    await ui.openProviderModal();
    ui.closeProviderModal();
    ui.closeProviderModal();

    // Test when providerModal exists but key inputs are absent
    const removedKeyNodes: Array<{
      el: HTMLElement;
      parent: HTMLElement;
      next: ChildNode | null;
    }> = [];
    for (const id of [
      "provider-key-openai",
      "provider-key-anthropic",
      "provider-key-google",
      "provider-store-consent",
    ]) {
      const el = document.getElementById(id);
      if (el && el.parentElement) {
        removedKeyNodes.push({
          el,
          parent: el.parentElement,
          next: el.nextSibling,
        });
        el.remove();
      }
    }
    const modalWithoutInputsUi = new ChatUI(state);
    await flushPromises();
    mockFetch.mockResolvedValueOnce({ ok: false, status: 500 });
    await modalWithoutInputsUi.openProviderModal();
    modalWithoutInputsUi.destroy();
    for (const item of removedKeyNodes.reverse()) {
      item.parent.insertBefore(item.el, item.next);
    }

    // Test when provider-modal itself and optional cohort/provider/streaming/db/sync inputs are absent
    for (const id of [
      "streaming-toggle-btn",
      "db-select",
      "sync-sessions-btn",
      "cohort-filter-btn",
      "close-cohort-modal-btn",
      "cohort-modal",
      "cohort-table-select",
      "cohort-join-table-select",
      "cohort-join-type-select",
      "cohort-join-on-input",
      "cohort-min-age",
      "cohort-max-age",
      "cohort-gender-select",
      "cohort-txgroup",
      "cohort-max-tir",
      "cohort-min-hba1c",
      "cohort-insert-chat-btn",
      "cohort-execute-btn",
      "cohort-sql-preview",
      "provider-key-openai",
      "provider-key-anthropic",
      "provider-key-google",
      "provider-store-consent",
      "provider-modal",
    ]) {
      document.getElementById(id)?.remove();
    }
    const minimalUi = new ChatUI(state);
    await flushPromises();
    minimalUi.updateStreamingToggleUi();
    minimalUi.openCohortModal();
    minimalUi.closeCohortModal();
    minimalUi.updateCohortSqlPreview();
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({}),
    });
    await minimalUi.loadModels();
    await expect(minimalUi.openProviderModal()).resolves.toBeUndefined();
    expect(() => minimalUi.closeProviderModal()).not.toThrow();
    await minimalUi.saveProviderSettings();
    await minimalUi.clearProviderSettings();
    minimalUi.destroy();
  });
});
