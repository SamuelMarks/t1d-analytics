/**
 * @file provider-modal-controller.ts
 * @module controllers/provider-modal-controller
 * Composable sub-controller managing the Cloud Provider & API Key Settings modal,
 * local and remote credential status badges, and API key persistence.
 */

import i18next from "i18next";
import { parseResponseJson } from "../api-client";
import { trapModalFocus } from "../modal-utils";
import { type ChatState } from "../state";

/**
 * Response payload returned by `/api/providers/status`.
 */
export interface ProvidersStatusPayload {
  /** Provider status list. */
  providers?: Array<{
    /** Provider identifier. */
    provider: string;
    /** Whether credentials are configured. */
    configured: boolean;
  }>;
}

/**
 * Host contract required by `ProviderSettingsModalController`.
 */
export interface ProviderSettingsHost {
  /** Global application state store. */
  state: ChatState;
  /** Header button to open the Cloud LLM Providers & API Keys modal. */
  providerSettingsBtn: HTMLButtonElement | null;
  /** Indicator dot displaying the configuration status of the active provider. */
  providerStatusDot: HTMLElement | null;
  /** Dropdown select element for choosing the active LLM or SQL engine. */
  modelSelect: HTMLSelectElement;
  /** Modal dialog container for Cloud LLM Providers & API Keys settings. */
  providerModal: HTMLElement | null;
  /** Password input for the OpenAI API key. */
  providerKeyOpenAI: HTMLInputElement | null;
  /** Password input for the Anthropic API key. */
  providerKeyAnthropic: HTMLInputElement | null;
  /** Password input for the Google Gemini API key. */
  providerKeyGoogle: HTMLInputElement | null;
  /** Checkbox input for user consent to store API keys in localStorage. */
  providerStoreConsent: HTMLInputElement | null;
  /** Element that held focus prior to opening the provider settings modal. */
  previousFocusProvider: Element | null;
  /** Keydown listener for focus trapping inside the provider modal. */
  onProviderModalKeyDown: ((e: KeyboardEvent) => void) | null;
  /** Reloads available models from `/api/models`. */
  loadModels(): Promise<void>;
  /** Announces a status message to assistive technology. */
  announce(message: string, isAssertive?: boolean): void;
}

/**
 * Composable sub-controller for the Provider Settings Modal.
 */
export class ProviderSettingsModalController {
  /** Host UI reference. */
  private readonly ui: ProviderSettingsHost;

  /**
   * Creates a new `ProviderSettingsModalController`.
   * @param {ProviderSettingsHost} ui - Host UI instance.
   */
  constructor(ui: ProviderSettingsHost) {
    this.ui = ui;
  }

  /**
   * Updates the provider status dot badge based on the currently selected model.
   * @returns {void}
   */
  public updateProviderBadgeStatus(): void {
    if (!this.ui.providerStatusDot) return;
    const selectedOption = this.ui.modelSelect.selectedOptions?.[0];
    const prov =
      selectedOption?.getAttribute("data-provider") ||
      (this.ui.modelSelect.value.includes("/")
        ? this.ui.modelSelect.value.split("/")[0]
        : "ollama");

    this.ui.state.setActiveChatProvider(prov);
    if (prov === "ollama" || this.ui.modelSelect.value === "sql") {
      const statusText = i18next.t(
        "provider.localEngineActive",
        "Local Engine Active",
      );
      this.ui.providerStatusDot.className = "provider-status-dot active";
      this.ui.providerStatusDot.title = statusText;
      this.ui.providerStatusDot.setAttribute("aria-label", statusText);
    } else {
      const hasKey =
        Boolean(this.ui.state.getApiKey(prov)) ||
        selectedOption?.getAttribute("data-configured") === "true";
      if (hasKey) {
        const statusText = i18next.t("provider.providerActive", {
          provider: prov,
          defaultValue: `${prov} Active (Configured)`,
        });
        this.ui.providerStatusDot.className = "provider-status-dot active";
        this.ui.providerStatusDot.title = statusText;
        this.ui.providerStatusDot.setAttribute("aria-label", statusText);
      } else {
        const statusText = i18next.t("provider.providerMissingKey", {
          provider: prov,
          defaultValue: `${prov} Missing API Key (Click ⚙️ to configure)`,
        });
        this.ui.providerStatusDot.className = "provider-status-dot warning";
        this.ui.providerStatusDot.title = statusText;
        this.ui.providerStatusDot.setAttribute("aria-label", statusText);
      }
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
    const el = document.getElementById(`provider-badge-${provider}`);
    if (!el) return;
    if (isConfigured) {
      el.className = "provider-status-badge badge-active";
      el.setAttribute("data-i18n", "provider.configuredBadge");
      el.textContent = i18next.t("provider.configuredBadge", "Configured 🟢");
    } else {
      el.className = "provider-status-badge badge-missing";
      el.setAttribute("data-i18n", "provider.missingKeyBadge");
      el.textContent = i18next.t("provider.missingKeyBadge", "Missing Key 🟡");
    }
  }

  /**
   * Opens the Provider Settings modal and fetches provider status badges.
   * @returns {Promise<void>}
   */
  public async open(): Promise<void> {
    const providerModal = this.ui.providerModal;
    if (!providerModal) return;

    if (this.ui.providerKeyOpenAI) {
      this.ui.providerKeyOpenAI.value = this.ui.state.getApiKey("openai") || "";
    }
    if (this.ui.providerKeyAnthropic) {
      this.ui.providerKeyAnthropic.value =
        this.ui.state.getApiKey("anthropic") || "";
    }
    if (this.ui.providerKeyGoogle) {
      this.ui.providerKeyGoogle.value = this.ui.state.getApiKey("google") || "";
    }

    const activeProvider = document.activeElement;
    this.ui.previousFocusProvider =
      activeProvider && activeProvider !== document.body
        ? activeProvider
        : this.ui.providerSettingsBtn;
    this.updateModalBadgesFromLocal();
    providerModal.classList.remove("hidden");
    providerModal.removeAttribute("aria-hidden");
    if (this.ui.onProviderModalKeyDown) {
      providerModal.removeEventListener(
        "keydown",
        this.ui.onProviderModalKeyDown,
      );
    }
    this.ui.onProviderModalKeyDown = (e: KeyboardEvent) => {
      trapModalFocus(e, providerModal, () => this.close());
    };
    providerModal.addEventListener("keydown", this.ui.onProviderModalKeyDown);
    this.ui.providerKeyOpenAI?.focus();

    try {
      const resp = await fetch("/api/providers/status");
      if (resp.ok) {
        const data = await parseResponseJson<ProvidersStatusPayload>(resp);
        for (const p of data.providers || []) {
          this.updateModalBadge(
            p.provider,
            p.configured || Boolean(this.ui.state.getApiKey(p.provider)),
          );
        }
      } else {
        this.updateModalBadgesFromLocal();
      }
    } catch {
      this.updateModalBadgesFromLocal();
    }
  }

  /**
   * Closes the Provider Settings modal.
   * @returns {void}
   */
  public close(): void {
    if (!this.ui.providerModal) return;
    this.ui.providerModal.classList.add("hidden");
    this.ui.providerModal.setAttribute("aria-hidden", "true");
    if (this.ui.onProviderModalKeyDown) {
      this.ui.providerModal.removeEventListener(
        "keydown",
        this.ui.onProviderModalKeyDown,
      );
      this.ui.onProviderModalKeyDown = null;
    }
    if (
      this.ui.previousFocusProvider instanceof HTMLElement &&
      document.body.contains(this.ui.previousFocusProvider)
    ) {
      this.ui.previousFocusProvider.focus();
    } else {
      this.ui.providerSettingsBtn?.focus();
    }
  }

  /**
   * Saves provider API keys from the modal into localStorage.
   * @returns {Promise<void>}
   */
  public async saveSettings(): Promise<void> {
    const consent = this.ui.providerStoreConsent
      ? this.ui.providerStoreConsent.checked
      : true;
    if (consent) {
      if (this.ui.providerKeyOpenAI) {
        const val = this.ui.providerKeyOpenAI.value.trim();
        if (val) this.ui.state.setApiKey("openai", val);
      }
      if (this.ui.providerKeyAnthropic) {
        const val = this.ui.providerKeyAnthropic.value.trim();
        if (val) this.ui.state.setApiKey("anthropic", val);
      }
      if (this.ui.providerKeyGoogle) {
        const val = this.ui.providerKeyGoogle.value.trim();
        if (val) this.ui.state.setApiKey("google", val);
      }
    }
    await this.ui.loadModels();
    this.close();
    this.ui.announce(i18next.t("ui.settingsSaved", "Provider settings saved"));
  }

  /**
   * Clears stored custom provider API keys from localStorage.
   * @returns {Promise<void>}
   */
  public async clearSettings(): Promise<void> {
    this.ui.state.removeApiKey("openai");
    this.ui.state.removeApiKey("anthropic");
    this.ui.state.removeApiKey("google");
    if (this.ui.providerKeyOpenAI) this.ui.providerKeyOpenAI.value = "";
    if (this.ui.providerKeyAnthropic) this.ui.providerKeyAnthropic.value = "";
    if (this.ui.providerKeyGoogle) this.ui.providerKeyGoogle.value = "";
    this.updateModalBadgesFromLocal();
    await this.ui.loadModels();
    this.ui.announce(i18next.t("ui.settingsCleared", "Provider keys cleared"));
  }
}
