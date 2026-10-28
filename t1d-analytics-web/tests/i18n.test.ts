/**
 * @file i18n.test.ts
 * @module tests/i18n.test
 * Unit tests for internationalization (i18next), RTL directionality, and DOM translation.
 */

import { describe, it, expect, beforeEach, vi } from "vitest";
import fs from "fs";
import path from "path";
import i18next, {
  translateDocument,
  setLanguage,
  getInitialLang,
  setDocumentDir,
  isRtlLanguage,
  enTranslations,
  jaTranslations,
  arTranslations,
  heTranslations,
  type TranslationResourceMap,
} from "../src/i18n";

describe("i18n.ts", () => {
  beforeEach(() => {
    document.body.innerHTML = "";
    document.documentElement.lang = "en";
    document.documentElement.dir = "ltr";
  });

  it("translates document elements correctly", () => {
    // Setup DOM
    document.body.innerHTML = `
      <div id="test-text" data-i18n="app.newChat"></div>
      <input id="test-input" data-i18n-placeholder="app.typeMessage" />
      <button id="test-button" data-i18n-aria-label="aria.startNewChat"></button>
      <div id="test-title" data-i18n-title="app.title"></div>
      
      <!-- Elements with attributes but no matching keys just to test branches safely -->
      <div data-i18n=""></div>
      <input data-i18n-placeholder="" />
      <button data-i18n-aria-label=""></button>
      <div data-i18n-title=""></div>
    `;

    // Ensure i18next is ready
    expect(i18next.isInitialized).toBe(true);

    // Act
    translateDocument();

    // Assert
    expect(document.getElementById("test-text")?.textContent).toBe(
      "+ New Chat",
    );
    expect(
      document.getElementById("test-input")?.getAttribute("placeholder"),
    ).toBe("Type a message...");
    expect(
      document.getElementById("test-button")?.getAttribute("aria-label"),
    ).toBe("Start a new chat");
    expect(document.getElementById("test-title")?.getAttribute("title")).toBe(
      "t1d-analytics",
    );
  });

  it("sets language, document lang, dir, and calls translateDocument", async () => {
    document.body.innerHTML = `<div id="test-text" data-i18n="app.newChat"></div>`;

    await setLanguage("ja");
    expect(localStorage.getItem("app-lang")).toBe("ja");
    expect(i18next.language).toBe("ja");
    expect(document.documentElement.lang).toBe("ja");
    expect(document.documentElement.dir).toBe("ltr");
    expect(document.getElementById("test-text")?.textContent).toBe(
      "+ 新しいチャット",
    );

    await setLanguage("ar");
    expect(i18next.language).toBe("ar");
    expect(document.documentElement.lang).toBe("ar");
    expect(document.documentElement.dir).toBe("rtl");
    expect(document.getElementById("test-text")?.textContent).toBe(
      "+ دردشة جديدة",
    );

    await setLanguage("he");
    expect(i18next.language).toBe("he");
    expect(document.documentElement.lang).toBe("he");
    expect(document.documentElement.dir).toBe("rtl");
    expect(document.getElementById("test-text")?.textContent).toBe(
      "+ צ'אט חדש",
    );
  });

  it("determines initial language correctly", () => {
    localStorage.clear();
    // jsdom navigator language is en-US usually
    expect(getInitialLang()).toBe("en");

    localStorage.setItem("app-lang", "ja");
    expect(getInitialLang()).toBe("ja");
  });

  it("determines initial language correctly when falling back to navigator", () => {
    localStorage.clear();

    const originalNavigator = window.navigator;
    vi.stubGlobal("navigator", { language: "ar-AE" });

    expect(getInitialLang()).toBe("ar");

    vi.stubGlobal("navigator", { language: "fr-FR" });

    expect(getInitialLang()).toBe("en");

    // restore
    vi.stubGlobal("navigator", originalNavigator);
  });

  it("determines initial language correctly when falling back to navigator, no navigator available", () => {
    localStorage.clear();

    const originalNavigator = window.navigator;
    vi.stubGlobal("navigator", undefined);

    expect(getInitialLang()).toBe("en");

    // restore
    vi.stubGlobal("navigator", originalNavigator);
  });

  it("determines initial language correctly when localStorage is undefined", () => {
    const originalLocalStorage = window.localStorage;
    vi.stubGlobal("localStorage", undefined);

    expect(getInitialLang()).toBe("en");

    vi.stubGlobal("localStorage", originalLocalStorage);
  });

  it("sets document dir based on initial language correctly", () => {
    setDocumentDir("ar");
    expect(document.documentElement.dir).toBe("rtl");
    setDocumentDir("he");
    expect(document.documentElement.dir).toBe("rtl");
    setDocumentDir("en");
    expect(document.documentElement.dir).toBe("ltr");
  });

  it("verifies translation key parity and namespaces across all supported languages", () => {
    const langs = ["en", "ja", "ar", "he"];
    for (const lang of langs) {
      expect(i18next.t("provider.title", { lng: lang })).toBeTruthy();
      expect(i18next.t("provider.description", { lng: lang })).toBeTruthy();
      expect(i18next.t("provider.consent", { lng: lang })).toBeTruthy();
      expect(i18next.t("provider.clear", { lng: lang })).toBeTruthy();
      expect(i18next.t("provider.save", { lng: lang })).toBeTruthy();

      expect(i18next.t("cohort.joinTable", { lng: lang })).toBeTruthy();
      expect(i18next.t("cohort.joinType", { lng: lang })).toBeTruthy();
      expect(i18next.t("cohort.joinOn", { lng: lang })).toBeTruthy();

      expect(i18next.t("cgm.title", { lng: lang })).toBeTruthy();
      expect(i18next.t("cgm.reportPdf", { lng: lang })).toBeTruthy();
      expect(i18next.t("cgm.timeOfDay", { lng: lang })).toBeTruthy();
      expect(i18next.t("cgm.glucoseMgDl", { lng: lang })).toBeTruthy();
      expect(i18next.t("cgm.targetRange", { lng: lang })).toBeTruthy();

      expect(i18next.t("table.searchPlaceholder", { lng: lang })).toBeTruthy();
      expect(i18next.t("table.exportCsv", { lng: lang })).toBeTruthy();
      expect(i18next.t("table.sortDefault", { lng: lang })).toBeTruthy();

      expect(i18next.t("aria.messageReceived", { lng: lang })).toBeTruthy();
      expect(i18next.t("ui.settingsSaved", { lng: lang })).toBeTruthy();
      expect(i18next.t("ui.settingsCleared", { lng: lang })).toBeTruthy();
    }
  });

  it("formats plural chat strings correctly according to language pluralization rules", () => {
    expect(i18next.t("app.chatNumber", { count: 1, lng: "en" })).toBe(
      "Chat #1",
    );
    expect(i18next.t("app.chatNumber", { count: 5, lng: "en" })).toBe(
      "Chat #5",
    );

    expect(i18next.t("app.chatNumber", { count: 1, lng: "ar" })).toContain("1");
    expect(i18next.t("app.chatNumber", { count: 2, lng: "ar" })).toContain("2");
    expect(i18next.t("app.chatNumber", { count: 5, lng: "ar" })).toContain("5");

    expect(i18next.t("app.chatNumber", { count: 1, lng: "he" })).toContain("1");
    expect(i18next.t("app.chatNumber", { count: 2, lng: "he" })).toContain("2");
    expect(i18next.t("app.chatNumber", { count: 5, lng: "he" })).toContain("5");
  });

  it("verifies recursive key and interpolation placeholder parity across en, ja, ar, and he", async () => {
    /**
     * Flattens a nested translation resource map into dot-separated key paths.
     * @param {TranslationResourceMap} obj - Resource map to flatten.
     * @param {string} [prefix=""] - Key prefix.
     * @returns {Record<string, string>} Flat dictionary of key paths to string values.
     */
    const flattenKeys = (
      obj: TranslationResourceMap,
      prefix = "",
    ): Record<string, string> => {
      const result: Record<string, string> = {};
      for (const [k, v] of Object.entries(obj)) {
        // Normalize plural suffixes (_zero, _one, _two, _few, _many, _other) for base parity check
        const fullKey = prefix ? `${prefix}.${k}` : k;
        if (typeof v === "string") {
          result[fullKey] = v;
        } else {
          Object.assign(result, flattenKeys(v, fullKey));
        }
      }
      return result;
    };

    const enFlat = flattenKeys(enTranslations);
    const locales: Array<{ name: string; map: Record<string, string> }> = [
      { name: "ja", map: flattenKeys(jaTranslations) },
      { name: "ar", map: flattenKeys(arTranslations) },
      { name: "he", map: flattenKeys(heTranslations) },
    ];

    const extractPlaceholders = (str: string): string[] =>
      Array.from(str.matchAll(/\{\{(\w+)\}\}/g))
        .map((m) => m[1])
        .sort();

    for (const [key, enVal] of Object.entries(enFlat)) {
      if (key.includes("chatNumber_")) continue;
      const expectedVars = extractPlaceholders(enVal);
      for (const loc of locales) {
        expect(
          loc.map[key],
          `Missing translation key '${key}' in locale '${loc.name}'`,
        ).toBeDefined();
        expect(
          extractPlaceholders(loc.map[key]),
          `Placeholder mismatch for '${key}' in locale '${loc.name}'`,
        ).toEqual(expectedVars);
      }
    }

    expect(isRtlLanguage("ar")).toBe(true);
    expect(isRtlLanguage("he")).toBe(true);
    expect(isRtlLanguage("en")).toBe(false);
    expect(isRtlLanguage("ja")).toBe(false);
    await setLanguage("en");
    expect(isRtlLanguage()).toBe(false);
  });

  it("translates all data-i18n* attributes in the full index.html DOM across all 4 locales", async () => {
    const indexHtml = fs.readFileSync(
      path.resolve(process.cwd(), "index.html"),
      "utf-8",
    );
    document.body.innerHTML = indexHtml;

    for (const lang of ["en", "ja", "ar", "he"]) {
      await setLanguage(lang);

      document.querySelectorAll("[data-i18n]").forEach((el) => {
        const key = el.getAttribute("data-i18n")!;
        expect(el.textContent).toBe(i18next.t(key));
      });

      document.querySelectorAll("[data-i18n-placeholder]").forEach((el) => {
        const key = el.getAttribute("data-i18n-placeholder")!;
        expect(el.getAttribute("placeholder")).toBe(i18next.t(key));
      });

      document.querySelectorAll("[data-i18n-aria-label]").forEach((el) => {
        const key = el.getAttribute("data-i18n-aria-label")!;
        expect(el.getAttribute("aria-label")).toBe(i18next.t(key));
      });

      document.querySelectorAll("[data-i18n-title]").forEach((el) => {
        const key = el.getAttribute("data-i18n-title")!;
        expect(el.getAttribute("title")).toBe(i18next.t(key));
      });

      document.querySelectorAll("#lang-select option").forEach((opt) => {
        expect(opt.getAttribute("lang")).toBe(opt.getAttribute("value"));
      });
    }

    await setLanguage("en");
  });

  it("interpolates error details without HTML-escaping special characters", () => {
    const rawError =
      "SQL execution error: Binder Error: No function matches 'sum(VARCHAR)' -> HUGEINT (arg BOOLEAN, /)";
    const formatted = i18next.t("ui.errorDetails", {
      error: rawError,
      lng: "en",
    });
    expect(formatted).toBe(`Error details: ${rawError}`);
    expect(formatted).not.toContain("&#39;");
    expect(formatted).not.toContain("&gt;");
    expect(formatted).not.toContain("&#x2F;");
  });
});
