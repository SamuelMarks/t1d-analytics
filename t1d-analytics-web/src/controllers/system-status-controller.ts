/**
 * @file system-status-controller.ts
 * @module controllers/system-status-controller
 * Composable sub-controller and diagnostic status types for System Health polling and banners.
 */

/**
 * Known database diagnostic status codes returned by the backend `/api/status` and `/api/schema` endpoints.
 */
export type DbStatusCode =
  | "healthy"
  | "missing_file"
  | "empty_db"
  | "empty_tables"
  | "missing_initial_data"
  | "unreadable"
  | "corrupted"
  | "unknown";

const KNOWN_DB_STATUS_CODES: ReadonlySet<string> = new Set<DbStatusCode>([
  "healthy",
  "missing_file",
  "empty_db",
  "empty_tables",
  "missing_initial_data",
  "unreadable",
  "corrupted",
  "unknown",
]);

/**
 * Type guard verifying whether a candidate string is a known `DbStatusCode`.
 * @param {string | undefined} val - Candidate status code string.
 * @returns {val is DbStatusCode} True if `val` is a valid `DbStatusCode`.
 */
export function isDbStatusCode(val: string | undefined): val is DbStatusCode {
  return typeof val === "string" && KNOWN_DB_STATUS_CODES.has(val.trim());
}

/**
 * Host contract required by `SystemStatusController`.
 */
export interface SystemStatusHost {
  /**
   * Checks system health and updates status indicators.
   * @param {boolean} [isManualRetry] - Whether triggered by manual user retry.
   * @param {string} [dbPath] - Optional custom database path.
   * @returns {Promise<void>}
   */
  checkSystemStatus(isManualRetry?: boolean, dbPath?: string): Promise<void>;
}

/**
 * Composable sub-controller managing System Health polling and diagnostic banners.
 */
export class SystemStatusController {
  /** Bound host callbacks. */
  private readonly ui: SystemStatusHost;

  /**
   * Creates a new SystemStatusController.
   * @param {SystemStatusHost} ui - Host implementing SystemStatusHost.
   */
  constructor(ui: SystemStatusHost) {
    this.ui = ui;
  }

  /**
   * Checks system health and updates status indicators.
   * @param {boolean} [isManualRetry=false] - Whether triggered by manual user retry.
   * @param {string} [dbPath] - Optional custom database path.
   * @returns {Promise<void>}
   */
  public async check(
    isManualRetry: boolean = false,
    dbPath?: string,
  ): Promise<void> {
    await this.ui.checkSystemStatus(isManualRetry, dbPath);
  }
}
