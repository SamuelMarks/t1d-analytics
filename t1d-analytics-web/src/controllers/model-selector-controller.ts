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

/**
 * Default choice Ollama / Hugging Face model identifiers.
 */
export const DEFAULT_CHOICE_OLLAMA_MODELS: readonly string[] = [
  "gemma:2b",
  "llama3.2:1b",
  "phi4-mini",
];

/**
 * Preferred lightweight local Ollama model identifiers ranked ahead of heavy 26B/27B models.
 */
const PREFERRED_FAST_LOCAL_MODELS: readonly string[] = [
  "gemma:2b",
  "llama3.2:1b",
  "phi4-mini",
];

/**
 * Strips an optional `ollama/` prefix from a model identifier for normalized tag comparison.
 * @param {string} modelId - Raw model identifier.
 * @returns {string} Model tag without `ollama/` prefix.
 */
function stripOllamaPrefix(modelId: string): string {
  return modelId.toLowerCase().startsWith("ollama/")
    ? modelId.slice(7)
    : modelId;
}

/**
 * Checks whether an option/API model identifier matches a requested model identifier, normalizing `:latest` suffixes.
 * @param {string} candidateId - Candidate model ID from `/api/models` or `<option>`.
 * @param {string} currentModel - Current model ID stored on the chat or state.
 * @returns {boolean} True if the identifiers match directly or via `:latest` normalization.
 */
export function matchesModelTagWithLatest(
  candidateId: string,
  currentModel: string,
): boolean {
  const c = stripOllamaPrefix(candidateId.trim()).toLowerCase();
  const t = stripOllamaPrefix(currentModel.trim()).toLowerCase();
  return c === t || c === `${t}:latest` || `${c}:latest` === t;
}

/**
 * Computes a priority rank for a candidate fallback model so small/fast local models are preferred and `:26b`/`:27b` models are ranked last.
 * @param {string} modelName - Candidate model name.
 * @returns {number} Numeric rank (lower is preferred).
 */
export function getModelFallbackRank(modelName: string): number {
  const cleaned = stripOllamaPrefix(modelName.trim()).toLowerCase();
  if (
    cleaned.includes("26b") ||
    cleaned.includes("27b") ||
    cleaned.includes("70b")
  ) {
    return 200;
  }
  const idx = PREFERRED_FAST_LOCAL_MODELS.findIndex(
    (pref) => cleaned === pref || cleaned.startsWith(`${pref}:`),
  );
  return idx >= 0 ? idx : 50;
}

/**
 * Resolves the provider identifier for a model name or API entry.
 * @param {string} modelName - Model name string.
 * @param {string} [explicitProvider] - Optional explicit provider string.
 * @returns {string} Resolved provider identifier.
 */
export function resolveModelProvider(
  modelName: string,
  explicitProvider?: string,
): string {
  if (explicitProvider) {
    return explicitProvider;
  }
  if (
    modelName.startsWith("hf.co/") ||
    DEFAULT_CHOICE_OLLAMA_MODELS.includes(modelName) ||
    !modelName.includes("/")
  ) {
    return "ollama";
  }
  return modelName.split("/")[0];
}
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
  /** Synchronizes syntax highlighting in the chat input. */
  syncHighlight(): void;
  /** Updates the provider badge status indicator for the active model. */
  updateProviderBadgeStatus(): void;
  /** Populates the multi-select model configuration list in the provider modal. */
  populateMultiSelectModels(modelsList: ModelApiItem[]): void;
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
      this.ui.updateProviderBadgeStatus();
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
      ? apiModels.filter((m) => m.name !== "sql")
      : [];

    for (const customModel of customModels) {
      if (customModel.trim().toLowerCase() === "sql") {
        continue;
      }
      const normalized = customModel.includes(":")
        ? customModel.replace(":", "/")
        : customModel;
      if (!modelsList.some((m) => m.name === normalized)) {
        const prov = resolveModelProvider(normalized);
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
      const prov = resolveModelProvider(model.name, model.provider);
      const hasKey = Boolean(this.ui.state.getApiKey(prov));
      const isConfigured =
        prov === "ollama"
          ? model.configured !== false
          : Boolean(model.configured || hasKey);
      const isReachable = model.reachable !== false;
      const isAvailable = model.available !== false && isConfigured;

      const badge = !isReachable ? "🔴" : !isConfigured ? "🟡" : "🟢";
      const option = document.createElement("option");
      option.value = model.name;
      option.setAttribute("data-provider", prov);
      option.setAttribute("data-configured", isConfigured ? "true" : "false");
      option.setAttribute("data-reachable", isReachable ? "true" : "false");
      option.setAttribute("data-available", isAvailable ? "true" : "false");
      option.textContent = `${model.name} ${badge}`;
      if (prov === "ollama" && (!isAvailable || !isReachable)) {
        option.disabled = true;
      }
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

    const currentModel = activeChat.model;
    const matchedOpt = Array.from(this.modelSelect.options).find(
      (opt) =>
        !opt.disabled && matchesModelTagWithLatest(opt.value, currentModel),
    );
    if (matchedOpt) {
      this.modelSelect.value = matchedOpt.value;
      if (matchedOpt.value !== currentModel) {
        this.ui.state.setSelectedModel(matchedOpt.value);
      }
      return;
    }

    if (modelsList.length === 0) {
      this.modelSelect.value = "sql";
      this.ui.state.setSelectedModel("sql");
      return;
    }

    const readyCandidates = modelsList
      .filter((m) => {
        const prov = resolveModelProvider(m.name, m.provider);
        const readyKey =
          (prov === "ollama" && m.configured !== false) ||
          Boolean(this.ui.state.getApiKey(prov));
        return m.available !== false && m.reachable !== false && readyKey;
      })
      .sort(
        (a, b) => getModelFallbackRank(a.name) - getModelFallbackRank(b.name),
      );
    const firstReady = readyCandidates[0];
    const fallbackModel = firstReady
      ? firstReady.name
      : activeChat.messages.length === 0
        ? "sql"
        : modelsList[0].name;
    this.modelSelect.value = fallbackModel;
    this.ui.state.setSelectedModel(fallbackModel);
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
        const prov = resolveModelProvider(m.name, m.provider);
        return (
          prov === "ollama" &&
          m.available !== false &&
          m.reachable !== false &&
          m.configured !== false
        );
      });
      this.ui.state.setSystemStatus({ ollamaOnline: hasReachableOllama });
      this.resolveActiveChatModel(modelsList);
      this.ui.populateMultiSelectModels(modelsList);
      this.ui.updateProviderBadgeStatus();
    } catch (error) {
      console.warn("Failed to fetch models from API:", error);
    }
  }
}
