/**
 * @file provider-modal.test.ts
 * @module tests/provider-modal.test
 * Unit tests for ProviderSettingsModalController, ModelSelectorController, API key persistence, and provider status indicators.
 */

import { describe, it, expect, beforeEach, vi, afterEach } from "vitest";
import i18next from "i18next";
import { ChatState } from "../src/state";
import {
  ChatUI,
  ModelSelectorController,
  ProviderSettingsModalController,
} from "../src/ui";
import { flushPromises, getTestElement, mountChatUiTestDom } from "./setup";

describe("ProviderSettingsModalController & ModelSelectorController", () => {
  let state: ChatState;
  let ui: ChatUI;
  let mockFetch: ReturnType<typeof vi.fn>;

  beforeEach(async () => {
    mountChatUiTestDom();
    mockFetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ models: [], tables: [] }),
    });
    vi.stubGlobal("fetch", mockFetch);

    state = new ChatState();
    await i18next.changeLanguage("en");
    document.documentElement.lang = "en";
    ui = new ChatUI(state);
    await flushPromises();
  });

  afterEach(() => {
    ui.dispose();
    localStorage.clear();
    document.body.innerHTML = "";
    vi.restoreAllMocks();
  });

  it("handles provider settings modal open, key updates, and clear", async () => {
    expect(ui.providerModalController).toBeInstanceOf(
      ProviderSettingsModalController,
    );
    expect(ui.modelSelectorController).toBeInstanceOf(ModelSelectorController);

    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        providers: {
          ollama: { configured: true, available: true },
          openai: { configured: true, available: true },
          anthropic: { configured: false, available: false },
          google: { configured: false, available: false },
        },
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

    const keyAnthropic = getTestElement<HTMLInputElement>(
      "provider-key-anthropic",
    );
    const keyGoogle = getTestElement<HTMLInputElement>("provider-key-google");
    keyAnthropic.value = "anthropic-secret-key";
    keyGoogle.value = "google-secret-key";

    await ui.providerModalController.saveSettings();
    expect(state.getApiKey("anthropic")).toBe("anthropic-secret-key");
    expect(state.getApiKey("google")).toBe("google-secret-key");

    keyAnthropic.value = "";
    keyGoogle.value = "";
    await ui.providerModalController.saveSettings();
    expect(state.getApiKey("anthropic")).toBeNull();
    expect(state.getApiKey("google")).toBeNull();

    state.setApiKey("anthropic", "anthropic-secret-key");
    mockFetch.mockResolvedValueOnce({ ok: false, status: 500 });
    await ui.providerModalController.open();
    const anthropicBadge = document.getElementById("provider-badge-anthropic");
    expect(anthropicBadge?.className).toContain("badge-configured");

    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({}),
    });
    await ui.providerModalController.refreshModalBadges();

    ui.providerModalController.updateModalBadge("non-existent-provider", true);
  });

  it("renders grouped models, custom configured models, and updates provider status dot", async () => {
    state.createChat();
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

    await ui.modelSelectorController.loadModels();

    const modelSelect = getTestElement<HTMLSelectElement>("model-select");
    const optgroups = modelSelect.querySelectorAll("optgroup");
    expect(optgroups.length).toBeGreaterThan(0);

    const dot = getTestElement("provider-status-dot");

    modelSelect.value = "sql";
    modelSelect.dispatchEvent(new Event("change"));
    expect(dot.classList.contains("active")).toBe(true);

    modelSelect.value = "gemma4";
    modelSelect.dispatchEvent(new Event("change"));
    expect(dot.classList.contains("active")).toBe(true);

    modelSelect.value = "openai/gpt-4o";
    modelSelect.dispatchEvent(new Event("change"));
    expect(dot.classList.contains("unconfigured")).toBe(true);

    state.setApiKey("openai", "sk-valid-key");
    ui.providerModalController.updateProviderBadgeStatus();
    expect(dot.classList.contains("configured")).toBe(true);
    state.removeApiKey("openai");

    // Unreachable model and dynamic provider group
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
    await ui.modelSelectorController.loadModels();
    const unreachableOpt = Array.from(modelSelect.options).find(
      (o) => o.value === "custom_unreachable",
    );
    expect(unreachableOpt?.textContent).toContain("🔴");
    expect(unreachableOpt?.title).toBe("custom_unreachable (CONN_ERR)");

    modelSelect.innerHTML = `<option value="openai/custom-gpt">custom-gpt</option><option value="local_plain">local_plain</option>`;
    modelSelect.value = "openai/custom-gpt";
    ui.providerModalController.updateProviderBadgeStatus();
    expect(dot.classList.contains("unconfigured")).toBe(true);
    modelSelect.value = "local_plain";
    ui.providerModalController.updateProviderBadgeStatus();
    expect(dot.classList.contains("active")).toBe(true);

    // Custom configured any-llm models and activeChat fallback resolution
    state.setConfiguredModels(
      "sql, groq:llama-3.3-70b-versatile, deepseek/deepseek-chat, custom-local-ollama, openai/gpt-4o",
    );
    const activeChat = state.getActiveChat();
    if (activeChat) {
      activeChat.model = "missing-model";
    }
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        models: [
          { name: "sql", provider: "duckdb", configured: true },
          {
            name: "hf.co/google/gemma-4-E4B",
            provider: "ollama",
            configured: true,
          },
          { name: "openai/gpt-4o", provider: "openai", configured: true },
        ],
      }),
    });
    await ui.modelSelectorController.loadModels();
    const optionValues = Array.from(modelSelect.options).map((o) => o.value);
    expect(optionValues).toContain("sql");
    expect(optionValues).toContain("groq/llama-3.3-70b-versatile");
    expect(optionValues).toContain("deepseek/deepseek-chat");
    expect(optionValues).toContain("custom-local-ollama");

    // Multi-select custom selection save
    const multiSelect = getTestElement<HTMLSelectElement>(
      "provider-models-multiselect",
    );
    for (const opt of Array.from(multiSelect.options)) {
      opt.selected =
        opt.value === "sql" || opt.value === "groq/llama-3.3-70b-versatile";
    }
    await ui.providerModalController.saveSettings();
    expect(state.getConfiguredModels()).toContain(
      "groq/llama-3.3-70b-versatile",
    );
    state.removeConfiguredModels();

    // Fallback when models list is empty and activeChat has a non-sql model
    if (activeChat) {
      activeChat.model = "nonexistent";
    }
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({}),
    });
    await ui.modelSelectorController.loadModels();
    expect(modelSelect.value).toBe("sql");

    // Fallback when no model is ready and activeChat has messages vs no messages
    if (activeChat) {
      activeChat.model = "nonexistent";
      activeChat.messages = [{ role: "user", content: "hi" }];
    }
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        models: [
          {
            name: "openai/gpt-4o",
            provider: "openai",
            configured: false,
            available: false,
            reachable: false,
          },
        ],
      }),
    });
    await ui.modelSelectorController.loadModels();
    expect(modelSelect.value).toBe("openai/gpt-4o");

    if (activeChat) {
      activeChat.model = "nonexistent";
      activeChat.messages = [];
    }
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        models: [
          {
            name: "openai/gpt-4o",
            provider: "openai",
            configured: false,
            available: false,
            reachable: false,
          },
        ],
      }),
    });
    await ui.modelSelectorController.loadModels();
    expect(modelSelect.value).toBe("sql");

    // Fallback when activeChat is null
    state.chats = [];
    state.activeChatId = null;
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ models: [{ name: "gemma4", configured: true }] }),
    });
    await ui.modelSelectorController.loadModels();

    // :latest tag normalization and small-model fallback preference over 26b/27b models
    const normChat = state.createChat();
    normChat.model = "ollama/gemma4";
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        models: [
          { name: "gemma2:27b", provider: "ollama", configured: true },
          { name: "gemma4:latest", provider: "ollama", configured: true },
        ],
      }),
    });
    await ui.modelSelectorController.loadModels();
    expect(modelSelect.value).toBe("gemma4:latest");
    expect(normChat.model).toBe("gemma4:latest");
    expect(state.selectedModel).toBe("gemma4:latest");

    normChat.model = "unmatched-custom-model";
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        models: [
          { name: "gemma2:27b", provider: "ollama", configured: true },
          { name: "gemma4:26b", provider: "ollama", configured: true },
          { name: "ollama/gemma:2b", provider: "ollama", configured: true },
        ],
      }),
    });
    await ui.modelSelectorController.loadModels();
    expect(modelSelect.value).toBe("ollama/gemma:2b");
    expect(normChat.model).toBe("ollama/gemma:2b");
  });

  it("handles provider modal Escape key dismissal, focus restoration, and dispose", async () => {
    ui.providerModalController.close();
    const trigger = getTestElement<HTMLButtonElement>("provider-settings-btn");
    trigger.focus();

    await ui.providerModalController.open();
    await ui.providerModalController.open();
    const modal = getTestElement("provider-modal");
    expect(modal.classList.contains("hidden")).toBe(false);

    modal.dispatchEvent(
      new KeyboardEvent("keydown", {
        key: "Escape",
        bubbles: true,
        cancelable: true,
      }),
    );
    expect(modal.classList.contains("hidden")).toBe(true);
    expect(document.activeElement).toBe(trigger);

    await ui.providerModalController.open();
    ui.providerModalController.dispose();
  });
});
