/**
 * @file formatters.ts
 * @module formatters
 * Clinical and numerical formatting utilities using the ECMA-402 Internationalization API.
 */

import i18next from "i18next";

/**
 * Returns the currently active application locale identifier.
 * @returns {string} The active locale tag (e.g. 'en', 'ja', 'ar', 'he').
 */
export function getCurrentLocale(): string {
  return i18next.language;
}

/**
 * Formats a numeric value according to the active or specified locale.
 * @param {number} value - The number to format.
 * @param {Intl.NumberFormatOptions} [options] - Optional formatting options.
 * @param {string} [locale] - Optional explicit BCP 47 language tag override.
 * @returns {string} The localized number string.
 */
export function formatNumber(
  value: number,
  options?: Intl.NumberFormatOptions,
  locale?: string,
): string {
  const targetLocale = locale || getCurrentLocale();
  return new Intl.NumberFormat(targetLocale, options).format(value);
}

/**
 * Formats a fractional or whole percentage according to the active locale.
 * @param {number} value - The percentage value (e.g. 70.5 for 70.5%).
 * @param {number} [decimals=1] - Maximum fractional digits.
 * @param {string} [locale] - Optional explicit BCP 47 language tag override.
 * @returns {string} The localized formatted percentage with symbol.
 */
export function formatPercent(
  value: number,
  decimals: number = 1,
  locale?: string,
): string {
  const targetLocale = locale || getCurrentLocale();
  return new Intl.NumberFormat(targetLocale, {
    style: "percent",
    minimumFractionDigits: 0,
    maximumFractionDigits: decimals,
  }).format(value / 100);
}

/**
 * Primitive cell value returned by a DuckDB SQL query.
 */
export type SqlCellValue = string | number | boolean | null;

/**
 * Classification for an empty SQL result set.
 */
export interface ClassifiedSqlEmpty {
  /** Discriminator kind for empty results. */
  kind: "empty";
}

/**
 * Classification for a single-row, single-column (1x1) scalar SQL result.
 */
export interface ClassifiedSqlScalar {
  /** Discriminator kind for scalar results. */
  kind: "scalar";
  /** Column name of the scalar value. */
  column: string;
  /** Raw scalar cell value. */
  value: SqlCellValue;
  /** String-formatted representation ("NULL" when null). */
  formattedValue: string;
}

/**
 * Classification for a multi-column or multi-row tabular SQL result.
 */
export interface ClassifiedSqlTable {
  /** Discriminator kind for tabular results. */
  kind: "table";
  /** Column names in display order. */
  columns: string[];
  /** Result rows. */
  rows: ReadonlyArray<Record<string, SqlCellValue>>;
}

/**
 * Discriminated union representing a classified SQL query result.
 */
export type ClassifiedSqlResult =
  | ClassifiedSqlEmpty
  | ClassifiedSqlScalar
  | ClassifiedSqlTable;

/**
 * Classifies a SQL query result array into empty, scalar (1x1), or table variants.
 * @param {ReadonlyArray<Record<string, SqlCellValue>> | undefined} rows - The SQL result rows to classify.
 * @returns {ClassifiedSqlResult} The strongly-typed classification object.
 */
export function classifySqlResult(
  rows: ReadonlyArray<Record<string, SqlCellValue>> | undefined,
): ClassifiedSqlResult {
  if (!rows || rows.length === 0) {
    return { kind: "empty" };
  }
  const keys = Object.keys(rows[0]);
  if (rows.length === 1 && keys.length === 1) {
    const column = keys[0];
    const value = rows[0][column] ?? null;
    const formattedValue = value === null ? "NULL" : String(value);
    return {
      kind: "scalar",
      column,
      value,
      formattedValue,
    };
  }
  return {
    kind: "table",
    columns: keys,
    rows,
  };
}

/**
 * Resolves the UTC build timestamp injected at compile time.
 * @param {string} [injectedTimestamp] - Optional injected timestamp override.
 * @returns {string} Formatted UTC timestamp string (e.g., '2026-10-29T08:00:00Z UTC').
 */
export function getBuildTimestampUtc(
  injectedTimestamp: string = __BUILD_TIMESTAMP_UTC__,
): string {
  return injectedTimestamp.trim();
}
