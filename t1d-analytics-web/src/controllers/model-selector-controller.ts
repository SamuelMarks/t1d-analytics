/**
 * @file model-selector-controller.ts
 * @module controllers/model-selector-controller
 * Composable sub-controller managing the AI/SQL Model Selector dropdown,
 * provider optgroup organization, custom model merging, and active model fallback resolution.
 */

import i18next from "i18next";
import { fetchWithBackendError, parseResponseJson } from "../api-client";
import { requireElementById } from "../modal-utils";
import { type ChatState } from "../state";
import { type ProviderSettingsModalController } from "./provider-modal-controller";

/**
 * Model metadata entry returned by `/api/models`.
 */
export interface ModelApiItem {
  /** Model identifier string. */
  name: string;
  /** Model size in bytes (for local Ollama models). */
  size?: number;
  /** Provider identifier ('ollama', 'openai', 'anthropic', 'google', etc.). */
  provider?: string;
  /** Whether credentials are configured on the server. */
  configured?: boolean;
  /** Whether an API key is required. */
  requires_key?: boolean;
  /** Whether the model is available for selection. */
  available?: boolean;
  /** Whether the provider is reachable. */
  reachable?: boolean;
  /** Optional error code when unconfigured or offline. */
  error_code?: string;
}

/**
 * Response payload returned by `/api/models`.
 */
export interface ModelsApiResponse {
  /** Available models across local and cloud providers. */
  models?: ModelApiItem[];
}

/**
 * Host contract required by `ModelSelectorController`.
 */
export interface ModelSelectorHost {
  /** Global application state store. */
  state: ChatState;
  /** Provider settings modal sub-controller. */
  providerModalController: ProviderSettingsModalController;
  /** Synchronizes syntax highlighting in the chat input. */
  syncHighlight(): void;
}

/**
 * Controller managing `/api/models` discovery, optgroup rendering, and model selection events.
 */
export class ModelSelectorController {
  /** Host UI reference. */
  private readonly ui: ModelSelectorHost;
  /** Dropdown select element for choosing the active LLM or SQL engine. */
  public readonly modelSelect: HTMLSelectElement;

  /**
   * Creates a new `ModelSelectorController`.
   * @param {ModelSelectorHost} ui - Host UI instance.
   */
  constructor(ui: ModelSelectorHost) {
    this.ui = ui;
    this.modelSelect = requireElementById<HTMLSelectElement>("model-select");
    this.bindEvents();
  }

  /**
   * Binds the `change` listener on the model selector dropdown.
   * @returns {void}
   */
  private bindEvents(): void {
    this.modelSelect.addEventListener("change", () => {
      this.ui.state.setActiveChatModel(this.modelSelect.value);
      this.ui.providerModalController.updateProviderBadgeStatus();
      this.ui.syncHighlight();
    });
  }

  /**
   * Merges backend models with user-configured models from `ChatState`.
   * @param {ModelApiItem[] | undefined} apiModels - Models returned by `/api/models`.
   * @param {string[]} customModels - Custom model names configured in state.
   * @returns {ModelApiItem[]} Combined model list.
   */
  private buildModelsList(
    apiModels: ModelApiItem[] | undefined,
    customModels: string[],
  ): ModelApiItem[] {
    const modelsList: ModelApiItem[] = Array.isArray(apiModels)
      ? [...apiModels]
      : [];

    for (const customModel of customModels) {
      const normalized = customModel.includes(":")
        ? customModel.replace(":", "/")
        : customModel;
      if (!modelsList.some((m) => m.name === normalized)) {
        const prov = normalized.includes("/")
          ? normalized.split("/")[0]
          : "ollama";
        modelsList.push({
          name: normalized,
          provider: prov,
          configured: true,
          requires_key: prov !== "ollama",
          available: true,
          reachable: true,
        });
      }
    }

    return modelsList;
  }

  /**
   * Populates `modelSelect` optgroups from the merged model list.
   * @param {ModelApiItem[]} modelsList - Combined model metadata entries.
   * @returns {void}
   */
  private populateModelOptions(modelsList: ModelApiItem[]): void {
    const sqlOption = document.createElement("option");
    sqlOption.value = "sql";
    sqlOption.setAttribute("data-i18n", "app.literalSql");
    sqlOption.textContent = i18next.t("app.literalSql");
    this.modelSelect.replaceChildren(sqlOption);

    const groups: Record<string, HTMLOptGroupElement> = {
      ollama: document.createElement("optgroup"),
      openai: document.createElement("optgroup"),
      anthropic: document.createElement("optgroup"),
      google: document.createElement("optgroup"),
    };
    groups.ollama.label = i18next.t("provider.groupOllama", "Ollama (Local)");
    groups.openai.label = i18next.t("provider.groupOpenAI", "OpenAI");
    groups.anthropic.label = i18next.t("provider.groupAnthropic", "Anthropic");
    groups.google.label = i18next.t("provider.groupGoogle", "Google");

    for (const model of modelsList) {
      const prov =
        model.provider ||
        (model.name.includes("/") ? model.name.split("/")[0] : "ollama");
      const hasKey = Boolean(this.ui.state.getApiKey(prov));
      const isConfigured = Boolean(model.configured || hasKey);
      const isReachable = model.reachable !== false;
      const isAvailable = model.available !== false && isConfigured;

      const badge = !isReachable ? "🔴" : !isConfigured ? "🟡" : "🟢";
      const option = document.createElement("option");
      option.value = model.name;
      option.setAttribute("data-provider", prov);
      option.setAttribute("data-configured", isConfigured ? "true" : "false");
      option.textContent = `${model.name} ${badge}`;
      if (!isAvailable && model.error_code) {
        option.title = `${model.name} (${model.error_code})`;
      }

      if (!groups[prov]) {
        const dynamicGroup = document.createElement("optgroup");
        dynamicGroup.label = `${prov.charAt(0).toUpperCase() + prov.slice(1)} (any-llm)`;
        groups[prov] = dynamicGroup;
      }
      groups[prov].appendChild(option);
    }

    for (const grp of Object.values(groups)) {
      if (grp.children.length > 0) {
        this.modelSelect.appendChild(grp);
      }
    }
  }

  /**
   * Resolves and synchronizes the active chat's selected model against available options.
   * @param {ModelApiItem[]} modelsList - Combined model metadata entries.
   * @returns {void}
   */
  private resolveActiveChatModel(modelsList: ModelApiItem[]): void {
    const activeChat = this.ui.state.getActiveChat();
    if (!activeChat) return;

    const exists = Array.from(this.modelSelect.options).some(
      (opt) => opt.value === activeChat.model,
    );
    if (exists) {
      this.modelSelect.value = activeChat.model;
      return;
    }

    if (modelsList.length === 0) {
      this.modelSelect.value = "sql";
      this.ui.state.setActiveChatModel("sql");
      return;
    }

    const firstReady = modelsList.find((m) => {
      const prov =
        m.provider || (m.name.includes("/") ? m.name.split("/")[0] : "ollama");
      const readyKey =
        (prov === "ollama" && m.configured !== false) ||
        Boolean(this.ui.state.getApiKey(prov));
      return m.available !== false && m.reachable !== false && readyKey;
    });
    const fallbackModel = firstReady
      ? firstReady.name
      : activeChat.messages.length === 0
        ? "sql"
        : modelsList[0].name;
    this.modelSelect.value = fallbackModel;
    this.ui.state.setActiveChatModel(fallbackModel);
  }

  /**
   * Fetches available models from the backend and populates the dropdown grouped by provider.
   * @returns {Promise<void>}
   */
  public async loadModels(): Promise<void> {
    try {
      const customModels = this.ui.state.getConfiguredModels();
      const modelsUrl =
        customModels.length > 0
          ? `/api/models?configured_models=${encodeURIComponent(customModels.join(","))}`
          : "/api/models";
      const response = await fetchWithBackendError(modelsUrl);
      const data = await parseResponseJson<ModelsApiResponse>(response);

      const modelsList = this.buildModelsList(data.models, customModels);
      this.populateModelOptions(modelsList);

      const hasReachableOllama = modelsList.some((m) => {
        const prov =
          m.provider ||
          (m.name.includes("/") ? m.name.split("/")[0] : "ollama");
        return (
          prov === "ollama" &&
          m.available !== false &&
          m.reachable !== false &&
          m.configured !== false
        );
      });
      this.ui.state.setSystemStatus({ ollamaOnline: hasReachableOllama });
      this.resolveActiveChatModel(modelsList);
      this.ui.providerModalController.updateProviderBadgeStatus();
    } catch (error) {
      console.warn("Failed to fetch models from API:", error);
    }
  }
}
