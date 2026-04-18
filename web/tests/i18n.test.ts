import { describe, it, expect, beforeEach } from "vitest";
import i18next, { translateDocument, setLanguage } from "../src/i18n";

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
});
