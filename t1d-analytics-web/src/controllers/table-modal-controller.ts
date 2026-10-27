/**
 * @file table-modal-controller.ts
 * @module controllers/table-modal-controller
 * Composable sub-controller managing the Table Preview Modal state and pagination.
 */

/**
 * Host contract required by `TablePreviewModalController`.
 */
export interface TablePreviewHost {
  /**
   * Opens the table modal for the specified table name.
   * @param {string} tableName - Table identifier to preview.
   * @returns {Promise<void>}
   */
  openTableModal(tableName: string): Promise<void>;
}

/**
 * Composable sub-controller managing the Table Preview Modal state and pagination.
 */
export class TablePreviewModalController {
  /** Bound host callbacks. */
  private readonly ui: TablePreviewHost;

  /**
   * Creates a new TablePreviewModalController.
   * @param {TablePreviewHost} ui - Host implementing TablePreviewHost.
   */
  constructor(ui: TablePreviewHost) {
    this.ui = ui;
  }

  /**
   * Opens the table modal for the specified table name.
   * @param {string} tableName - Table identifier to preview.
   * @returns {Promise<void>}
   */
  public async open(tableName: string): Promise<void> {
    await this.ui.openTableModal(tableName);
  }
}
