/**
 * @file provider-modal-controller.ts
 * @module controllers/provider-modal-controller
 * Composable sub-controller managing the Cloud Provider & API Key Settings modal.
 */

/**
 * Host contract required by `ProviderSettingsModalController`.
 */
export interface ProviderSettingsHost {
  /**
   * Opens the Provider Settings modal.
   * @returns {Promise<void>}
   */
  openProviderModal(): Promise<void>;
  /**
   * Closes the Provider Settings modal.
   * @returns {void}
   */
  closeProviderModal(): void;
}

/**
 * Composable sub-controller managing the Cloud Provider & API Key Settings modal.
 */
export class ProviderSettingsModalController {
  /** Bound host callbacks. */
  private readonly ui: ProviderSettingsHost;

  /**
   * Creates a new ProviderSettingsModalController.
   * @param {ProviderSettingsHost} ui - Host implementing ProviderSettingsHost.
   */
  constructor(ui: ProviderSettingsHost) {
    this.ui = ui;
  }

  /**
   * Opens the Provider Settings modal.
   * @returns {Promise<void>}
   */
  public async open(): Promise<void> {
    await this.ui.openProviderModal();
  }

  /**
   * Closes the Provider Settings modal.
   * @returns {void}
   */
  public close(): void {
    this.ui.closeProviderModal();
  }
}
