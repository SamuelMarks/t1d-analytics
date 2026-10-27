/**
 * @file cohort-modal-controller.ts
 * @module controllers/cohort-modal-controller
 * Composable sub-controller managing the Visual Cohort Query Builder modal.
 */

/**
 * Host contract required by `CohortBuilderModalController`.
 */
export interface CohortBuilderHost {
  /**
   * Opens the Visual Cohort Query Builder modal.
   * @returns {void}
   */
  openCohortModal(): void;
  /**
   * Closes the Visual Cohort Query Builder modal.
   * @returns {void}
   */
  closeCohortModal(): void;
}

/**
 * Composable sub-controller managing the Visual Cohort Query Builder modal.
 */
export class CohortBuilderModalController {
  /** Bound host callbacks. */
  private readonly ui: CohortBuilderHost;

  /**
   * Creates a new CohortBuilderModalController.
   * @param {CohortBuilderHost} ui - Host implementing CohortBuilderHost.
   */
  constructor(ui: CohortBuilderHost) {
    this.ui = ui;
  }

  /**
   * Opens the Visual Cohort Query Builder modal.
   * @returns {void}
   */
  public open(): void {
    this.ui.openCohortModal();
  }

  /**
   * Closes the Visual Cohort Query Builder modal.
   * @returns {void}
   */
  public close(): void {
    this.ui.closeCohortModal();
  }
}
