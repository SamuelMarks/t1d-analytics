/**
 * @file provider-modal-controller.ts
 * @module controllers/provider-modal-controller
 * Composable sub-controller managing the Cloud Provider & API Key Settings modal,
 * local and remote credential status badges, and API key persistence.
 */

import i18next from "i18next";
import { fetchWithBackendError, parseResponseJson } from "../api-client";
import { requireElementById, trapModalFocus } from "../modal-utils";
import { type ChatState } from "../state";
import {
  DEFAULT_CHOICE_OLLAMA_MODELS,
  resolveModelProvider,
  type ModelApiItem,
} from "./model-selector-controller";

/**
 * Status metadata for a single LLM provider in `ProvidersStatusPayload`.
 */
export interface ProviderStatusEntry {
  /** Whether API credentials are configured on the server. */
  configured?: boolean;
  /** Whether the provider is available for inference. */
  available?: boolean;
  /** Whether the provider endpoint is reachable. */
  reachable?: boolean;
  /** Optional machine-readable error code. */
  error_code?: string;
}

/**
 * Response payload returned by `/api/providers/status`.
 */
export interface ProvidersStatusPayload {
  /** Map of provider identifiers to configuration/availability status. */
  providers?: Record<string, ProviderStatusEntry>;
}

/**
 * Host contract required by `ProviderSettingsModalController`.
 */
export interface ProviderSettingsHost {
  /** Global application state store. */
  state: ChatState;
  /** Dropdown select element for choosing the active LLM or SQL engine. */
  modelSelect: HTMLSelectElement;
  /** Reloads available models in the model selector. */
  reloadModels(): Promise<void>;
  /** Announces a message to screen readers. */
  announce(message: string, isAssertive?: boolean): void;
}

/**
 * Controller managing the Cloud LLM Providers & API Keys configuration modal.
 */
export class ProviderSettingsModalController {
  /** Host UI reference. */
  private readonly ui: ProviderSettingsHost;
  /** Header button to open the Cloud LLM Providers & API Keys modal. */
  public readonly providerSettingsBtn: HTMLButtonElement;
  /** Indicator dot displaying the configuration status of the active provider. */
  public readonly providerStatusDot: HTMLElement;
  /** Modal dialog container for Cloud LLM Providers & API Keys settings. */
  public readonly providerModal: HTMLElement;
  /** Button to close the Cloud LLM Providers modal. */
  public readonly closeProviderModalBtn: HTMLButtonElement;
  /** Password input for the OpenAI API key. */
  public readonly providerKeyOpenAI: HTMLInputElement;
  /** Password input for the Anthropic API key. */
  public readonly providerKeyAnthropic: HTMLInputElement;
  /** Password input for the Google Gemini API key. */
  public readonly providerKeyGoogle: HTMLInputElement;
  /** Multi-select element for choosing enabled any-llm and SQL models. */
  public readonly providerModelsMultiSelect: HTMLSelectElement;
  /** Checkbox input for user consent to store API keys in localStorage. */
  public readonly providerStoreConsent: HTMLInputElement;
  /** Button to save configured provider API keys. */
  public readonly providerSaveBtn: HTMLButtonElement;
  /** Button to clear all stored provider API keys. */
  public readonly providerClearBtn: HTMLButtonElement;
  /** Element that held focus prior to opening the provider settings modal. */
  private previousFocusProvider: Element | null = null;
  /** Keydown listener for focus trapping inside the provider modal. */
  private onProviderModalKeyDown: ((e: KeyboardEvent) => void) | null = null;

  /**
   * Creates a new `ProviderSettingsModalController`.
   * @param {ProviderSettingsHost} ui - Host UI instance.
   */
  constructor(ui: ProviderSettingsHost) {
    this.ui = ui;
    this.providerSettingsBtn = requireElementById<HTMLButtonElement>(
      "provider-settings-btn",
    );
    this.providerStatusDot = requireElementById<HTMLElement>(
      "provider-status-dot",
    );
    this.providerModal = requireElementById<HTMLElement>("provider-modal");
    this.closeProviderModalBtn = requireElementById<HTMLButtonElement>(
      "close-provider-modal-btn",
    );
    this.providerKeyOpenAI = requireElementById<HTMLInputElement>(
      "provider-key-openai",
    );
    this.providerKeyAnthropic = requireElementById<HTMLInputElement>(
      "provider-key-anthropic",
    );
    this.providerKeyGoogle = requireElementById<HTMLInputElement>(
      "provider-key-google",
    );
    this.providerModelsMultiSelect = requireElementById<HTMLSelectElement>(
      "provider-models-multiselect",
    );
    this.providerStoreConsent = requireElementById<HTMLInputElement>(
      "provider-store-consent",
    );
    this.providerSaveBtn =
      requireElementById<HTMLButtonElement>("provider-save-btn");
    this.providerClearBtn =
      requireElementById<HTMLButtonElement>("provider-clear-btn");

    this.bindEvents();
  }

  /**
   * Binds DOM event listeners for the Cloud LLM Providers & API Keys modal.
   * @returns {void}
   */
  private bindEvents(): void {
    this.providerSettingsBtn.addEventListener("click", () => {
      void this.open();
    });

    this.closeProviderModalBtn.addEventListener("click", () => {
      this.close();
    });

    this.providerSaveBtn.addEventListener("click", () => {
      void this.saveSettings();
    });

    this.providerClearBtn.addEventListener("click", () => {
      void this.clearSettings();
    });

    this.providerModal.addEventListener("click", (e) => {
      if (e.target === this.providerModal) {
        this.close();
      }
    });
  }

  /**
   * Updates the provider status dot badge based on the currently selected model.
   * @returns {void}
   */
  public updateProviderBadgeStatus(): void {
    const val = this.ui.modelSelect.value;
    if (val === "sql") {
      this.providerStatusDot.className = "provider-status-dot active";
      this.providerStatusDot.title = i18next.t(
        "provider.duckdbReady",
        "DuckDB SQL Engine Ready",
      );
      this.providerStatusDot.setAttribute(
        "aria-label",
        this.providerStatusDot.title,
      );
      return;
    }

    const selectedOpt = this.ui.modelSelect.selectedOptions?.[0];
    const prov = resolveModelProvider(
      val,
      selectedOpt?.getAttribute("data-provider") || undefined,
    );
    const isConfigured =
      selectedOpt?.getAttribute("data-configured") === "true" ||
      Boolean(this.ui.state.getApiKey(prov));

    if (prov === "ollama") {
      const hasExplicitReachableAttr =
        selectedOpt?.hasAttribute("data-reachable") === true;
      const optionExplicitlyOffline =
        selectedOpt?.getAttribute("data-reachable") === "false" ||
        Boolean(selectedOpt?.disabled);
      const isOllamaHealthy =
        !optionExplicitlyOffline &&
        (!hasExplicitReachableAttr ||
          this.ui.state.systemStatus.ollamaOnline !== false);
      if (isOllamaHealthy) {
        this.providerStatusDot.className = "provider-status-dot active";
        this.providerStatusDot.title = i18next.t(
          "provider.localActive",
          "Local Ollama Active",
        );
      } else {
        this.providerStatusDot.className =
          "provider-status-dot warning unconfigured";
        this.providerStatusDot.title = i18next.t(
          "provider.localOffline",
          "Local Ollama Offline (Fallback SQL will be used)",
        );
      }
      this.providerStatusDot.setAttribute(
        "aria-label",
        this.providerStatusDot.title,
      );
    } else if (isConfigured) {
      this.providerStatusDot.className =
        "provider-status-dot active configured";
      this.providerStatusDot.title = i18next.t("provider.cloudKeyReady", {
        provider: prov,
        defaultValue: `${prov} API Key Configured`,
      });
      this.providerStatusDot.setAttribute(
        "aria-label",
        this.providerStatusDot.title,
      );
    } else {
      this.providerStatusDot.className =
        "provider-status-dot warning unconfigured";
      this.providerStatusDot.title = i18next.t("provider.cloudKeyMissing", {
        provider: prov,
        defaultValue: `${prov} API Key Missing (Click ⚙️ to configure)`,
      });
      this.providerStatusDot.setAttribute(
        "aria-label",
        this.providerStatusDot.title,
      );
    }
  }

  /**
   * Populates the multi-select model selector with `'sql'`, the 3 default choice models, and discovered `any-llm` models.
   * @param {ModelApiItem[]} modelsList - Discovered and configured models list.
   * @returns {void}
   */
  public populateMultiSelectModels(modelsList: ModelApiItem[]): void {
    const configuredSet = new Set(this.ui.state.getConfiguredModels());
    const sqlOpt = document.createElement("option");
    sqlOpt.value = "sql";
    sqlOpt.setAttribute("data-i18n", "app.literalSql");
    sqlOpt.textContent = i18next.t("app.literalSql");
    sqlOpt.selected = true;
    this.providerModelsMultiSelect.replaceChildren(sqlOpt);

    const seen = new Set<string>(["sql"]);
    const allNames: string[] = [
      ...DEFAULT_CHOICE_OLLAMA_MODELS,
      ...modelsList.map((m) => m.name),
    ];

    for (const name of allNames) {
      if (seen.has(name)) continue;
      seen.add(name);
      const opt = document.createElement("option");
      opt.value = name;
      opt.textContent = name;
      opt.selected =
        configuredSet.has(name) ||
        (configuredSet.size === 0 &&
          DEFAULT_CHOICE_OLLAMA_MODELS.includes(name));
      this.providerModelsMultiSelect.appendChild(opt);
    }
  }

  /**
   * Opens the Cloud LLM Providers & API Keys configuration modal.
   * @returns {Promise<void>}
   */
  public async open(): Promise<void> {
    if (this.providerModal.classList.contains("hidden")) {
      this.previousFocusProvider = document.activeElement;
    }
    this.providerKeyOpenAI.value = this.ui.state.getApiKey("openai") || "";
    this.providerKeyAnthropic.value =
      this.ui.state.getApiKey("anthropic") || "";
    this.providerKeyGoogle.value = this.ui.state.getApiKey("google") || "";

    await this.refreshModalBadges();

    this.providerModal.classList.remove("hidden");
    this.providerModal.setAttribute("aria-hidden", "false");

    if (this.onProviderModalKeyDown) {
      this.providerModal.removeEventListener(
        "keydown",
        this.onProviderModalKeyDown,
      );
    }
    this.onProviderModalKeyDown = (e: KeyboardEvent): void => {
      trapModalFocus(e, this.providerModal, () => this.close());
    };
    this.providerModal.addEventListener("keydown", this.onProviderModalKeyDown);

    this.closeProviderModalBtn.focus();
  }

  /**
   * Queries `/api/providers/status` and updates modal badge indicators.
   * @returns {Promise<void>}
   */
  public async refreshModalBadges(): Promise<void> {
    try {
      const resp = await fetchWithBackendError("/api/providers/status");
      const data = await parseResponseJson<ProvidersStatusPayload>(resp);
      const provs = data.providers || {};

      for (const p of ["openai", "anthropic", "google"]) {
        const localKey = Boolean(this.ui.state.getApiKey(p));
        const remoteConfig = Boolean(provs[p]?.configured);
        this.updateModalBadge(p, localKey || remoteConfig);
      }
    } catch {
      this.updateModalBadgesFromLocal();
    }
  }

  /**
   * Updates badges in the provider modal using locally stored keys.
   * @returns {void}
   */
  public updateModalBadgesFromLocal(): void {
    for (const p of ["openai", "anthropic", "google"]) {
      this.updateModalBadge(p, Boolean(this.ui.state.getApiKey(p)));
    }
  }

  /**
   * Updates an individual provider status badge element.
   * @param {string} provider - Provider identifier.
   * @param {boolean} isConfigured - Whether provider has active credentials.
   * @returns {void}
   */
  public updateModalBadge(provider: string, isConfigured: boolean): void {
    const badge = document.getElementById(`provider-badge-${provider}`);
    if (!badge) return;
    if (isConfigured) {
      badge.className = "provider-status-badge badge-configured";
      badge.textContent = i18next.t("provider.configured", "Configured");
    } else {
      badge.className = "provider-status-badge badge-missing";
      badge.textContent = i18next.t("provider.missingKey", "Missing Key");
    }
  }

  /**
   * Closes the Cloud LLM Providers & API Keys configuration modal.
   * @returns {void}
   */
  public close(): void {
    this.providerModal.classList.add("hidden");
    this.providerModal.setAttribute("aria-hidden", "true");
    if (this.onProviderModalKeyDown) {
      this.providerModal.removeEventListener(
        "keydown",
        this.onProviderModalKeyDown,
      );
      this.onProviderModalKeyDown = null;
    }
    if (this.previousFocusProvider instanceof HTMLElement) {
      this.previousFocusProvider.focus();
    }
  }

  /**
   * Saves provider API keys and multi-select model configuration from the modal into localStorage.
   * @returns {Promise<void>}
   */
  public async saveSettings(): Promise<void> {
    const consent = this.providerStoreConsent.checked;
    if (consent) {
      const oa = this.providerKeyOpenAI.value.trim();
      const an = this.providerKeyAnthropic.value.trim();
      const go = this.providerKeyGoogle.value.trim();

      if (oa) this.ui.state.setApiKey("openai", oa);
      else this.ui.state.removeApiKey("openai");

      if (an) this.ui.state.setApiKey("anthropic", an);
      else this.ui.state.removeApiKey("anthropic");

      if (go) this.ui.state.setApiKey("google", go);
      else this.ui.state.removeApiKey("google");

      const selectedModels = Array.from(this.providerModelsMultiSelect.options)
        .filter((opt) => opt.selected && opt.value !== "sql")
        .map((opt) => opt.value);
      const isDefaultSelection =
        selectedModels.length === DEFAULT_CHOICE_OLLAMA_MODELS.length &&
        selectedModels.every((m) => DEFAULT_CHOICE_OLLAMA_MODELS.includes(m));
      if (selectedModels.length > 0 && !isDefaultSelection) {
        this.ui.state.setConfiguredModels(selectedModels.join(","));
      }
    }

    this.updateModalBadgesFromLocal();
    await this.ui.reloadModels();
    this.updateProviderBadgeStatus();
    this.close();
    this.ui.announce(i18next.t("provider.saved", "Provider settings saved"));
  }

  /**
   * Clears stored custom provider API keys from localStorage.
   * @returns {Promise<void>}
   */
  public async clearSettings(): Promise<void> {
    for (const p of ["openai", "anthropic", "google"]) {
      this.ui.state.removeApiKey(p);
    }
    this.ui.state.removeConfiguredModels();
    this.providerKeyOpenAI.value = "";
    this.providerKeyAnthropic.value = "";
    this.providerKeyGoogle.value = "";
    this.updateModalBadgesFromLocal();
    await this.ui.reloadModels();
    this.updateProviderBadgeStatus();
    this.ui.announce(i18next.t("provider.cleared", "Stored API keys cleared"));
  }

  /**
   * Removes any active modal keydown listeners when disposing the UI.
   * @returns {void}
   */
  public dispose(): void {
    if (this.onProviderModalKeyDown) {
      this.providerModal.removeEventListener(
        "keydown",
        this.onProviderModalKeyDown,
      );
      this.onProviderModalKeyDown = null;
    }
  }
}
