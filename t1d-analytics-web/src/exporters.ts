/**
 * @file exporters.ts
 * @module exporters
 * Client-side CSV and JSON table export utilities with spreadsheet formula-injection sanitization.
 */

import { type SqlCellValue } from "./formatters";

/**
 * Cell value or nested record permitted in exported clinical rows.
 */
export type ExportableCellValue =
  SqlCellValue | Record<string, SqlCellValue> | undefined;

const CRLF = "\r\n";

const SAFE_SIGNED_DECIMAL_RE = /^[+-](?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$/;

/**
 * Neutralizes spreadsheet formula injection prefixes (`=`, `+`, `-`, `@`, TAB, CR, LF)
 * while preserving finite signed decimal literals (`-12.5`, `+3.2`).
 * @param {string} strVal - Raw string representation of the cell value.
 * @returns {string} Sanitized string safe for spreadsheet opening.
 */
export function sanitizeSpreadsheetCell(strVal: string): string {
  if (strVal.length === 0) {
    return strVal;
  }
  const firstCode = strVal.charCodeAt(0);
  const strippedLeading = strVal.replace(/^[ \u00A0]+/, "");
  const triggerChar = strippedLeading[0] ?? "";
  const triggerCode =
    strippedLeading.length > 0 ? strippedLeading.charCodeAt(0) : 32;
  if (
    firstCode < 32 ||
    triggerCode < 32 ||
    triggerChar === "=" ||
    triggerChar === "+" ||
    triggerChar === "-" ||
    triggerChar === "@"
  ) {
    let isNumeric = false;
    if (
      (triggerChar === "+" || triggerChar === "-") &&
      SAFE_SIGNED_DECIMAL_RE.test(strippedLeading)
    ) {
      isNumeric = Number.isFinite(Number(strippedLeading));
    }
    if (!isNumeric) {
      return `'${strVal}`;
    }
  }
  return strVal;
}

/**
 * Triggers a browser file download for a constructed Blob.
 * @param {Blob} blob - File content blob.
 * @param {string} filename - Target download filename.
 * @returns {void}
 */
export function triggerBlobDownload(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.setAttribute("href", url);
  link.setAttribute("download", filename);
  link.style.display = "none";
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}

/**
 * Export row objects as a downloadable UTF-8 BOM CSV file with spreadsheet formula defense.
 * @param {Array<Record<string, ExportableCellValue>>} rows - The data rows to export.
 * @param {string} filename - The filename for the downloaded file.
 * @returns {void}
 */
export function exportRowsToCsv(
  rows: Array<Record<string, ExportableCellValue>>,
  filename: string,
): void {
  const columnSet = new Set<string>();
  for (const r of rows) {
    for (const k of Object.keys(r)) {
      columnSet.add(k);
    }
  }
  const columns = Array.from(columnSet);

  const headerLine = columns
    .map((col) => `"${col.replace(/"/g, '""')}"`)
    .join(",");
  const dataLines = rows.map((row) =>
    columns
      .map((col) => {
        const val = row[col];
        if (val === null || val === undefined) return '""';
        const rawStr =
          typeof val === "object" ? JSON.stringify(val) : String(val);
        const safeStr = sanitizeSpreadsheetCell(rawStr);
        return `"${safeStr.replace(/"/g, '""')}"`;
      })
      .join(","),
  );
  const csvContent = "\uFEFF" + [headerLine, ...dataLines].join(CRLF);
  const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
  triggerBlobDownload(blob, filename);
}

/**
 * Export row objects as a downloadable formatted JSON file.
 * @param {Array<Record<string, ExportableCellValue>>} rows - The data rows to export.
 * @param {string} filename - The filename for the downloaded file.
 * @returns {void}
 */
export function exportRowsToJson(
  rows: Array<Record<string, ExportableCellValue>>,
  filename: string,
): void {
  const jsonContent = JSON.stringify(rows, null, 2);
  const blob = new Blob([jsonContent], {
    type: "application/json;charset=utf-8;",
  });
  triggerBlobDownload(blob, filename);
}
