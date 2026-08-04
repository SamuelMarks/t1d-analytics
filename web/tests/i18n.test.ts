import { describe, it, expect, beforeEach } from "vitest";
import i18next, {
  translateDocument,
  setLanguage,
  getInitialLang,
  setDocumentDir,
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

    // We need to mock navigator to trigger the language check.
    // jsdom allows us to set language via Object.defineProperty
    const originalNavigator = window.navigator;
    Object.defineProperty(window, "navigator", {
      value: { language: "ar-AE" },
      configurable: true,
    });

    expect(getInitialLang()).toBe("ar");

    Object.defineProperty(window, "navigator", {
      value: { language: "fr-FR" },
      configurable: true,
    });

    expect(getInitialLang()).toBe("en");

    // restore
    Object.defineProperty(window, "navigator", {
      value: originalNavigator,
      configurable: true,
    });
  });

  it("determines initial language correctly when falling back to navigator, no navigator available", () => {
    localStorage.clear();

    // We need to mock navigator to undefined
    const originalNavigator = window.navigator;
    Object.defineProperty(window, "navigator", {
      value: undefined,
      configurable: true,
    });

    expect(getInitialLang()).toBe("en");

    // restore
    Object.defineProperty(window, "navigator", {
      value: originalNavigator,
      configurable: true,
    });
  });

  it("determines initial language correctly when localStorage is undefined", () => {
    const originalLocalStorage = window.localStorage;
    Object.defineProperty(window, "localStorage", {
      value: undefined,
      configurable: true,
    });

    expect(getInitialLang()).toBe("en");

    Object.defineProperty(window, "localStorage", {
      value: originalLocalStorage,
      configurable: true,
    });
  });

  it("sets document dir based on initial language correctly", () => {
    setDocumentDir("ar");
    expect(document.documentElement.dir).toBe("rtl");
    setDocumentDir("he");
    expect(document.documentElement.dir).toBe("rtl");
    setDocumentDir("en");
    expect(document.documentElement.dir).toBe("ltr");
  });
});
