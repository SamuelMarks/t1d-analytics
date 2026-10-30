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
  | SqlCellValue
  | Record<string, SqlCellValue>
  | undefined;

const TAB_CHAR = String.fromCharCode(9);
const CR_CHAR = String.fromCharCode(13);
const LF_CHAR = String.fromCharCode(10);
const CRLF = CR_CHAR + LF_CHAR;

/**
 * Neutralizes spreadsheet formula injection prefixes (`=`, `+`, `-`, `@`, TAB, CR)
 * while preserving signed numeric literals (`-12.5`, `+3.2`).
 * @param {string} strVal - Raw string representation of the cell value.
 * @returns {string} Sanitized string safe for spreadsheet opening.
 */
export function sanitizeSpreadsheetCell(strVal: string): string {
  if (strVal && ["=", "+", "-", "@", TAB_CHAR, CR_CHAR].includes(strVal[0])) {
    let isNumeric = false;
    if (strVal[0] === "+" || strVal[0] === "-") {
      const num = Number(strVal);
      isNumeric = !isNaN(num) && isFinite(num);
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
