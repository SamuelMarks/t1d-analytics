/**
 * @file provider-modal-controller.ts
 * @module controllers/provider-modal-controller
 * Composable sub-controller managing the Cloud Provider & API Key Settings modal,
 * local and remote credential status badges, and API key persistence via `ModalDialogController`.
 */

import i18next from "i18next";
import { fetchWithBackendError, parseResponseJson } from "../api-client";
import {
  type ModelCatalogEntryPayload,
  type ProvidersStatusResponsePayload,
} from "../api-types";
import { type AppEventBus } from "../event-bus";
import { ModalDialogController, requireElementById } from "../modal-utils";
import { type ChatState } from "../state";
import {
  DEFAULT_CHOICE_OLLAMA_MODELS,
  resolveModelProvider,
} from "./model-selector-controller";

export { type ProvidersStatusResponsePayload as ProvidersStatusPayload };

/**
 * Narrow injected dependencies for `ProviderSettingsModalController`.
 */
export interface ProviderSettingsModalDeps {
  /** Shared application state store. */
  state: ChatState;
  /** Unidirectional domain event bus. */
  events: AppEventBus;
  /** Model selector dropdown inspected to compute the active provider badge. */
  modelSelect: HTMLSelectElement;
  /** Async callback reloading the `/api/models` catalog after saving/clearing keys. */
  reloadModels: () => Promise<void>;
}

/**
 * Controller managing `#provider-modal` and active provider badge status.
 */
export class ProviderSettingsModalController {
  /** Shared application state store. */
  private readonly state: ChatState;
  /** Domain event bus. */
  private readonly events: AppEventBus;
  /** Model selector dropdown. */
  private readonly modelSelect: HTMLSelectElement;
  /** Callback reloading `/api/models`. */
  private readonly reloadModelsFn: () => Promise<void>;
  /** Composable modal dialog lifecycle controller. */
  private readonly dialog: ModalDialogController;
  /** Persistent AbortController for modal action buttons and event bus subscriptions. */
  private readonly listenerController = new AbortController();
  /** Button to open the Cloud LLM Providers & API Keys modal. */
  public readonly providerSettingsBtn: HTMLButtonElement;
  /** Status dot indicator next to the model selector showing active provider readiness. */
  public readonly providerStatusDot: HTMLElement;
  /** Modal overlay container for Cloud LLM Providers & API Keys. */
  public readonly providerModal: HTMLElement;
  /** Button to close the provider settings modal. */
  public readonly closeProviderModalBtn: HTMLButtonElement;
  /** Input for OpenAI API Key. */
  public readonly providerKeyOpenAI: HTMLInputElement;
  /** Input for Anthropic API Key. */
  public readonly providerKeyAnthropic: HTMLInputElement;
  /** Input for Google Gemini API Key. */
  public readonly providerKeyGoogle: HTMLInputElement;
  /** Multi-select box for configured models. */
  public readonly providerModelsMultiSelect: HTMLSelectElement;
  /** Checkbox consenting to localStorage storage of API keys. */
  public readonly providerStoreConsent: HTMLInputElement;
  /** Button to save provider API keys and model configuration. */
  public readonly providerSaveBtn: HTMLButtonElement;
  /** Button to clear all stored provider API keys. */
  public readonly providerClearBtn: HTMLButtonElement;

  /**
   * Creates a new `ProviderSettingsModalController`.
   * @param {ProviderSettingsModalDeps} deps - Narrow state, event bus, and model reload dependencies.
   */
  constructor(deps: ProviderSettingsModalDeps) {
    this.state = deps.state;
    this.events = deps.events;
    this.modelSelect = deps.modelSelect;
    this.reloadModelsFn = deps.reloadModels;

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

    this.dialog = new ModalDialogController({
      modalEl: this.providerModal,
      closeBtn: this.closeProviderModalBtn,
      initialFocusEl: this.closeProviderModalBtn,
    });

    this.bindEvents();
  }

  /**
   * Binds DOM event listeners and domain event bus subscriptions.
   * @returns {void}
   */
  private bindEvents(): void {
    const signal = this.listenerController.signal;

    this.providerSettingsBtn.addEventListener(
      "click",
      () => {
        void this.open();
      },
      { signal },
    );

    this.providerSaveBtn.addEventListener(
      "click",
      () => {
        void this.saveSettings();
      },
      { signal },
    );

    this.providerClearBtn.addEventListener(
      "click",
      () => {
        void this.clearSettings();
      },
      { signal },
    );

    this.events.on(
      "openProviderSettings",
      () => {
        void this.open();
      },
      signal,
    );

    this.events.on(
      "providerBadgeUpdateRequested",
      () => {
        this.updateProviderBadgeStatus();
      },
      signal,
    );

    this.events.on(
      "modelsLoaded",
      ({ models }) => {
        this.populateMultiSelectModels(models);
      },
      signal,
    );
  }

  /**
   * Applies CSS class, tooltip, and `aria-label` to `providerStatusDot`.
   * @param {string} className - CSS class string.
   * @param {string} label - Localized tooltip and ARIA label.
   * @returns {void}
   */
  private applyBadgeDot(className: string, label: string): void {
    this.providerStatusDot.className = className;
    this.providerStatusDot.title = label;
    this.providerStatusDot.setAttribute("aria-label", label);
  }

  /**
   * Updates `providerStatusDot` when the active provider is local Ollama.
   * @param {HTMLOptionElement | undefined} selectedOpt - Currently selected `<option>`.
   * @returns {void}
   */
  private applyOllamaBadgeStatus(
    selectedOpt: HTMLOptionElement | undefined,
  ): void {
    const hasExplicitReachable =
      selectedOpt?.hasAttribute("data-reachable") === true;
    const explicitlyOffline =
      selectedOpt?.getAttribute("data-reachable") === "false" ||
      Boolean(selectedOpt?.disabled);
    const isHealthy =
      !explicitlyOffline &&
      (!hasExplicitReachable || this.state.systemStatus.ollamaOnline !== false);
    this.applyBadgeDot(
      isHealthy
        ? "provider-status-dot active"
        : "provider-status-dot warning unconfigured",
      isHealthy
        ? i18next.t("provider.localActive", "Local Ollama Active")
        : i18next.t(
            "provider.localOffline",
            "Local Ollama Offline (Fallback SQL will be used)",
          ),
    );
  }

  /**
   * Updates the provider status dot badge based on the currently selected model.
   * @returns {void}
   */
  public updateProviderBadgeStatus(): void {
    const val = this.modelSelect.value;
    if (val === "sql") {
      this.applyBadgeDot(
        "provider-status-dot active",
        i18next.t("provider.duckdbReady", "DuckDB SQL Engine Ready"),
      );
      return;
    }

    const selectedOpt = this.modelSelect.selectedOptions?.[0];
    const prov = resolveModelProvider(
      val,
      selectedOpt?.getAttribute("data-provider") || undefined,
    );

    if (prov === "ollama") {
      this.applyOllamaBadgeStatus(selectedOpt);
      return;
    }

    const isConfigured =
      selectedOpt?.getAttribute("data-configured") === "true" ||
      Boolean(this.state.getApiKey(prov));
    this.applyBadgeDot(
      isConfigured
        ? "provider-status-dot active configured"
        : "provider-status-dot warning unconfigured",
      isConfigured
        ? i18next.t("provider.cloudKeyReady", {
            provider: prov,
            defaultValue: `${prov} API Key Configured`,
          })
        : i18next.t("provider.cloudKeyMissing", {
            provider: prov,
            defaultValue: `${prov} API Key Missing (Click ⚙️ to configure)`,
          }),
    );
  }

  /**
   * Populates the multi-select model selector with `'sql'`, the 3 default choice models, and discovered `any-llm` models.
   * @param {ModelCatalogEntryPayload[]} modelsList - Discovered and configured models list.
   * @returns {void}
   */
  public populateMultiSelectModels(
    modelsList: ModelCatalogEntryPayload[],
  ): void {
    const configuredSet = new Set(this.state.getConfiguredModels());
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
    this.dialog.captureOpenerFocus();
    this.providerKeyOpenAI.value = this.state.getApiKey("openai") || "";
    this.providerKeyAnthropic.value = this.state.getApiKey("anthropic") || "";
    this.providerKeyGoogle.value = this.state.getApiKey("google") || "";

    await this.refreshModalBadges();
    this.dialog.open(this.closeProviderModalBtn);
  }

  /**
   * Queries `/api/providers/status` and updates modal badge indicators.
   * @returns {Promise<void>}
   */
  public async refreshModalBadges(): Promise<void> {
    try {
      const resp = await fetchWithBackendError("/api/providers/status");
      const data =
        await parseResponseJson<ProvidersStatusResponsePayload>(resp);
      const provs = data.providers ?? {};

      for (const p of ["openai", "anthropic", "google"]) {
        const localKey = Boolean(this.state.getApiKey(p));
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
      this.updateModalBadge(p, Boolean(this.state.getApiKey(p)));
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
    this.dialog.close();
  }

  /**
   * Saves provider API keys and multi-select model configuration from the modal into storage.
   * @returns {Promise<void>}
   */
  public async saveSettings(): Promise<void> {
    const consent = this.providerStoreConsent.checked;
    if (consent) {
      const oa = this.providerKeyOpenAI.value.trim();
      const an = this.providerKeyAnthropic.value.trim();
      const go = this.providerKeyGoogle.value.trim();

      if (oa) this.state.setApiKey("openai", oa);
      else this.state.removeApiKey("openai");

      if (an) this.state.setApiKey("anthropic", an);
      else this.state.removeApiKey("anthropic");

      if (go) this.state.setApiKey("google", go);
      else this.state.removeApiKey("google");

      const selectedModels = Array.from(this.providerModelsMultiSelect.options)
        .filter((opt) => opt.selected && opt.value !== "sql")
        .map((opt) => opt.value);
      const isDefaultSelection =
        selectedModels.length === DEFAULT_CHOICE_OLLAMA_MODELS.length &&
        selectedModels.every((m) => DEFAULT_CHOICE_OLLAMA_MODELS.includes(m));
      if (selectedModels.length > 0 && !isDefaultSelection) {
        this.state.setConfiguredModels(selectedModels.join(","));
      }
    }

    this.updateModalBadgesFromLocal();
    await this.reloadModelsFn();
    this.updateProviderBadgeStatus();
    this.close();
    this.events.emit("announce", {
      message: i18next.t("provider.saved", "Provider settings saved"),
    });
  }

  /**
   * Clears stored custom provider API keys from storage.
   * @returns {Promise<void>}
   */
  public async clearSettings(): Promise<void> {
    for (const p of ["openai", "anthropic", "google"]) {
      this.state.removeApiKey(p);
    }
    this.state.removeConfiguredModels();
    this.providerKeyOpenAI.value = "";
    this.providerKeyAnthropic.value = "";
    this.providerKeyGoogle.value = "";
    this.updateModalBadgesFromLocal();
    await this.reloadModelsFn();
    this.updateProviderBadgeStatus();
    this.events.emit("announce", {
      message: i18next.t("provider.cleared", "Stored API keys cleared"),
    });
  }

  /**
   * Disposes modal and bus listeners when tearing down the UI.
   * @returns {void}
   */
  public dispose(): void {
    this.dialog.dispose();
    this.listenerController.abort();
  }
}
