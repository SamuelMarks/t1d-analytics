/**
 * @file sql-highlight.ts
 * Pure helper utilities for modal focus trapping, clinical cohort SQL generation, and API error translation.
 */

import i18next from "./i18n";

/**
 * Parses a backend error string or detail object into a translated string.
 * @param {unknown} errorObj - The raw backend error string or structured detail object.
 * @returns {string} Localized or formatted error message string.
 */
export function parseApiError(errorObj: unknown): string {
  if (typeof errorObj === "string") {
    return i18next.exists(errorObj)
      ? (i18next.t(errorObj) as string)
      : errorObj;
  }
  if (errorObj && typeof errorObj === "object" && "error_code" in errorObj) {
    const key = (errorObj as { error_code: string }).error_code;
    const params = (errorObj as { params?: Record<string, unknown> }).params;
    return i18next.exists(key)
      ? (i18next.t(key, params as Record<string, unknown>) as string)
      : `${key}: ${JSON.stringify(params)}`;
  }
  return String(errorObj);
}

/**
 * Helper to fetch and automatically throw parsed backend errors.
 * @param {string} url - The URL to fetch.
 * @param {RequestInit} [options] - Fetch options.
 * @returns {Promise<Response>} The fetch response.
 */
export async function fetchWithBackendError(
  url: string,
  options?: RequestInit,
): Promise<Response> {
  const response = await fetch(url, options);
  if (!response.ok) {
    let errorMsg = `HTTP ${response.status}`;
    try {
      const errData = await response.json();
      if (errData.detail) {
        errorMsg = parseApiError(errData.detail);
      }
    } catch {
      // Ignore JSON parse errors
    }
    throw new Error(errorMsg);
  }
  return response;
}

/**
 * Clinical cohort filter parameters for query generation.
 */
export interface CohortFilterParams {
  /** Target primary table name. */
  tableName: string;
  /** Optional secondary table name for relational join. */
  joinTableName?: string;
  /** Relational join condition type ('INNER JOIN' or 'LEFT JOIN'). */
  joinType?: "INNER JOIN" | "LEFT JOIN";
  /** Explicit ON condition for table joining. */
  joinCondition?: string;
  /** Minimum patient age filter. */
  minAge?: number;
  /** Maximum patient age filter. */
  maxAge?: number;
  /** Biological sex or gender. */
  gender?: string;
  /** Clinical treatment group (e.g. Closed-Loop, MDI). */
  txGroup?: string;
  /** Time in Range maximum threshold percentage. */
  maxTIR?: number;
  /** Minimum glycated hemoglobin (HbA1c) threshold. */
  minHbA1c?: number;
}

/**
 * Construct an optimized DuckDB SQL query from clinician cohort filter parameters, supporting relational joins.
 * @param {CohortFilterParams} params - The selected filter criteria.
 * @returns {string} The generated DuckDB SQL query.
 */
export function buildCohortSql(params: CohortFilterParams): string {
  const table1 = params.tableName || "patients";
  const table2 = params.joinTableName?.trim();
  const joinType = params.joinType || "INNER JOIN";
  const joinOn = params.joinCondition?.trim();

  const isJoin = Boolean(table2 && table2 !== table1);
  const prefix = isJoin ? `"${table1.replace(/"/g, '""')}".` : "";
  const predicates: string[] = [];

  if (params.minAge !== undefined && Number.isFinite(params.minAge)) {
    predicates.push(`${prefix}age >= ${params.minAge}`);
  }
  if (params.maxAge !== undefined && Number.isFinite(params.maxAge)) {
    predicates.push(`${prefix}age <= ${params.maxAge}`);
  }
  if (params.gender) {
    const safeGender = params.gender.replace(/'/g, "''");
    predicates.push(`${prefix}gender = '${safeGender}'`);
  }
  if (params.txGroup) {
    const safeTx = params.txGroup.replace(/'/g, "''");
    predicates.push(`${prefix}txgroup ILIKE '%${safeTx}%'`);
  }
  if (params.maxTIR !== undefined && Number.isFinite(params.maxTIR)) {
    predicates.push(`${prefix}tir <= ${params.maxTIR}`);
  }
  if (params.minHbA1c !== undefined && Number.isFinite(params.minHbA1c)) {
    predicates.push(`${prefix}hba1c >= ${params.minHbA1c}`);
  }

  let fromClause = `FROM "${table1.replace(/"/g, '""')}"`;
  if (isJoin && table2) {
    const defaultOn = `"${table1.replace(/"/g, '""')}".patient_id = "${table2.replace(/"/g, '""')}".patient_id`;
    fromClause += `
${joinType} "${table2.replace(/"/g, '""')}"
  ON ${joinOn || defaultOn}`;
  }

  const whereClause =
    predicates.length > 0
      ? `\nWHERE ${predicates.join(
          `
  AND `,
        )}`
      : "";
  return `SELECT *
${fromClause}${whereClause};`;
}

/**
 * Traps keyboard focus within a modal dialog and handles the Escape key to close.
 * @param {KeyboardEvent} e - The keyboard event.
 * @param {HTMLElement} modal - The container element to trap focus within.
 * @param {() => void} onClose - The callback function to invoke when Escape is pressed.
 */
export function trapModalFocus(
  e: KeyboardEvent,
  modal: HTMLElement,
  onClose: () => void,
): void {
  if (e.key === "Escape") {
    e.preventDefault();
    onClose();
    return;
  }
  if (e.key !== "Tab") return;
  const focusableElements = modal.querySelectorAll<HTMLElement>(
    'button:not([disabled]), [href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])',
  );
  if (focusableElements.length === 0) return;
  const firstElement = focusableElements[0];
  const lastElement = focusableElements[focusableElements.length - 1];

  if (e.shiftKey) {
    if (document.activeElement === firstElement) {
      e.preventDefault();
      lastElement.focus();
    }
  } else {
    if (document.activeElement === lastElement) {
      e.preventDefault();
      firstElement.focus();
    }
  }
}
