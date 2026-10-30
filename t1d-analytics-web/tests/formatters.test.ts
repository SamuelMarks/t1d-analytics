/**
 * @file formatters.test.ts
 * @module tests/formatters.test
 * Tests for ECMA-402 Intl formatting utilities.
 */

import { describe, it, expect, beforeEach } from "vitest";
import {
  classifySqlResult,
  getBuildTimestampUtc,
  getCurrentLocale,
  formatNumber,
  formatPercent,
} from "../src/formatters";
import i18next from "../src/i18n";

describe("formatters.ts", () => {
  beforeEach(() => {
    document.documentElement.lang = "en";
  });

  it("resolves current locale from i18next", () => {
    expect(getCurrentLocale()).toBe(i18next.language);
  });

  it("formats numbers with locale or override", () => {
    const formattedEn = formatNumber(123456.78, undefined, "en");
    expect(formattedEn).toContain("123,456.78");

    const formattedCustom = formatNumber(
      1234.56,
      { maximumFractionDigits: 1 },
      "en",
    );
    expect(formattedCustom).toBe("1,234.6");

    const formattedDefault = formatNumber(42);
    expect(formattedDefault).toBe("42");
  });

  it("formats percentages with decimal precision across en, ja, ar, and he locales", () => {
    const defaultPct = formatPercent(70.52);
    expect(defaultPct).toBe("70.5%");

    const roundedPct = formatPercent(70.56, 0, "en");
    expect(roundedPct).toBe("71%");

    const exactPct = formatPercent(100, 1, "en");
    expect(exactPct).toBe("100%");

    const jaPct = formatPercent(75.4, 1, "ja");
    expect(jaPct).toContain("75.4");

    const arPct = formatPercent(82.5, 1, "ar");
    expect(arPct).toBeTruthy();

    const hePct = formatPercent(68.2, 1, "he");
    expect(hePct).toContain("68.2");
  });

  it("classifies SQL results into empty, scalar, and table variants", () => {
    expect(classifySqlResult(undefined)).toEqual({ kind: "empty" });
    expect(classifySqlResult([])).toEqual({ kind: "empty" });

    expect(classifySqlResult([{ total_patients: 42 }])).toEqual({
      kind: "scalar",
      column: "total_patients",
      value: 42,
      formattedValue: "42",
    });

    expect(classifySqlResult([{ max_glucose: null }])).toEqual({
      kind: "scalar",
      column: "max_glucose",
      value: null,
      formattedValue: "NULL",
    });

    const singleRowMultiCol = [{ id: 1, gender: "F" }];
    expect(classifySqlResult(singleRowMultiCol)).toEqual({
      kind: "table",
      columns: ["id", "gender"],
      rows: singleRowMultiCol,
    });

    const multiRowSingleCol = [{ id: 1 }, { id: 2 }];
    expect(classifySqlResult(multiRowSingleCol)).toEqual({
      kind: "table",
      columns: ["id"],
      rows: multiRowSingleCol,
    });
  });

  it("resolves UTC build timestamp from injected constant or override", () => {
    expect(getBuildTimestampUtc()).toContain("UTC");
    expect(getBuildTimestampUtc("  2026-10-29T12:34:56Z UTC  ")).toBe(
      "2026-10-29T12:34:56Z UTC",
    );
  });
});
