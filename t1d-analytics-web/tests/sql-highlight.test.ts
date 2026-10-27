/**
 * @file sql-highlight.test.ts
 * @module tests/sql-highlight.test
 * Unit tests for SQL statement classification, live syntax highlighting synchronization, and modal focus utilities.
 */

import { describe, it, expect, beforeEach, vi, afterEach } from "vitest";
import i18next from "i18next";
import { ChatState, type Message } from "../src/state";
import {
  isReadOnlySqlStatement,
  updateInputHighlight,
  trapModalFocus,
} from "../src/sql-highlight";
import { ChatUI, requireElementById } from "../src/ui";
import { flushPromises, getTestElement, mountChatUiTestDom } from "./setup";

describe("SQL Highlight & Modal Focus Utilities", () => {
  let state: ChatState;
  let ui: ChatUI;
  let mockFetch: ReturnType<typeof vi.fn>;

  beforeEach(async () => {
    mountChatUiTestDom();
    mockFetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ models: [], schema: [] }),
    });
    vi.stubGlobal("fetch", mockFetch);

    state = new ChatState();
    i18next.changeLanguage("en");
    document.documentElement.lang = "en";
    ui = new ChatUI(state);
    await flushPromises();
  });

  afterEach(() => {
    ui.destroy();
    localStorage.clear();
    document.body.innerHTML = "";
    vi.restoreAllMocks();
  });

  it("detects read-only SQL statements vs natural language and routes execution accordingly", async () => {
    expect(
      isReadOnlySqlStatement(`   -- comment only
   `),
    ).toBe(false);
    expect(isReadOnlySqlStatement("((SELECT * FROM patients))")).toBe(true);
    expect(isReadOnlySqlStatement("SHOW TABLES;")).toBe(true);
    expect(isReadOnlySqlStatement("SHOW")).toBe(true);
    expect(isReadOnlySqlStatement("Show me the first 5 patients")).toBe(false);
    expect(isReadOnlySqlStatement("Explain how insulin works")).toBe(false);
    expect(isReadOnlySqlStatement("Describe the study cohort")).toBe(false);
    expect(isReadOnlySqlStatement("From our clinical datasets")).toBe(false);
    expect(isReadOnlySqlStatement("What is the average HbA1c?")).toBe(false);

    state.createChat();
    const modelSelect = getTestElement<HTMLSelectElement>("model-select");
    modelSelect.value = "gemma4";
    const chatInput = getTestElement<HTMLTextAreaElement>("chat-input");
    chatInput.value = "(SELECT COUNT(*) FROM patients)";
    chatInput.dispatchEvent(new Event("input"));
    const highlight = document.getElementById("chat-input-highlight");
    expect(highlight?.className).toContain("language-sql");

    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        content: "",
        sqlQuery: "(SELECT COUNT(*) FROM patients)",
        sqlResult: [{ count: 15 }],
        error: null,
      }),
    });
    await ui["handleSendMessage"]();
    const sqlCallBody = JSON.parse(
      mockFetch.mock.calls[mockFetch.mock.calls.length - 1][1].body,
    );
    expect(sqlCallBody.model).toBe("sql");
    expect(sqlCallBody.auto_execute).toBe(true);
  });

  it("syncHighlight is called on chatInput input event and handles trailing newline", () => {
    const input = getTestElement<HTMLTextAreaElement>("chat-input");
    const highlight = getTestElement("chat-input-highlight");
    input.value = `test
lines`;
    input.dispatchEvent(new Event("input"));
    expect(input.style.height).not.toBe("");

    input.value = `test
`;
    updateInputHighlight(input, highlight, "markdown");
    expect(highlight.textContent).toBe(`test
 `);
  });

  it("highlights code blocks in assistant messages and handles empty/non-SQL blocks", () => {
    const fence = "```";
    const messages: Message[] = [
      {
        role: "assistant",
        content: `Here is code: 
${fence}sql
SELECT * FROM test;
${fence}`,
      },
    ];
    state.chats = [
      {
        id: "chat-3",
        title: "Chat 3",
        model: "gemma4",
        messages,
      },
    ];
    state.activeChatId = "chat-3";
    ui["renderMessages"]();

    const msgBlock = document.querySelector(".message.assistant");
    expect(msgBlock).not.toBeNull();
    const codeBlock = msgBlock?.querySelector("pre code");
    expect(codeBlock).not.toBeNull();
    expect(codeBlock?.classList.contains("hljs")).toBe(true);

    const chat = state.createChat();
    chat.model = "gemma4";
    chat.messages.push({
      role: "assistant",
      content: `${fence}sql
${fence}`,
    });
    state.setActiveChat(chat.id);
    ui["renderMessages"]();
    expect(document.querySelector(".sql-query-container")).not.toBeNull();

    chat.messages = [
      {
        role: "assistant",
        content: `${fence}python
print('hello')
${fence}`,
        model: "gemma",
      },
    ];
    ui["renderMessages"]();
    expect(document.querySelector(".language-python")).toBeTruthy();
  });

  it("deduplicates SQL block when markdown content already contains SQL code block", () => {
    const fence = "```";
    state.createChat();
    state.addMessageToActiveChat({
      role: "assistant",
      content: `Here is the query:
${fence}sql
SELECT * FROM users;
${fence}`,
      sqlQuery: "SELECT * FROM users;",
    });
    ui["renderMessages"]();

    const containers = document.querySelectorAll(".sql-query-container");
    expect(containers.length).toBe(1);
  });

  it("thoroughly tests trapModalFocus utility across all keyboard interaction branches", () => {
    const modal = document.createElement("div");
    modal.tabIndex = -1;
    const btn1 = document.createElement("button");
    const btn2 = document.createElement("button");
    const disabledBtn = document.createElement("button");
    disabledBtn.setAttribute("disabled", "true");
    modal.appendChild(btn1);
    modal.appendChild(disabledBtn);
    modal.appendChild(btn2);
    document.body.appendChild(modal);

    const closeSpy = vi.fn();

    const escEvent = new KeyboardEvent("keydown", {
      key: "Escape",
      cancelable: true,
    });
    trapModalFocus(escEvent, modal, closeSpy);
    expect(closeSpy).toHaveBeenCalledTimes(1);
    expect(escEvent.defaultPrevented).toBe(true);

    const enterEvent = new KeyboardEvent("keydown", { key: "Enter" });
    trapModalFocus(enterEvent, modal, closeSpy);
    expect(closeSpy).toHaveBeenCalledTimes(1);

    btn1.focus();
    const shiftTabFirst = new KeyboardEvent("keydown", {
      key: "Tab",
      shiftKey: true,
      cancelable: true,
    });
    trapModalFocus(shiftTabFirst, modal, closeSpy);
    expect(shiftTabFirst.defaultPrevented).toBe(true);
    expect(document.activeElement).toBe(btn2);

    btn2.focus();
    const shiftTabSecond = new KeyboardEvent("keydown", {
      key: "Tab",
      shiftKey: true,
      cancelable: true,
    });
    trapModalFocus(shiftTabSecond, modal, closeSpy);
    expect(shiftTabSecond.defaultPrevented).toBe(false);

    btn2.focus();
    const tabLast = new KeyboardEvent("keydown", {
      key: "Tab",
      shiftKey: false,
      cancelable: true,
    });
    trapModalFocus(tabLast, modal, closeSpy);
    expect(tabLast.defaultPrevented).toBe(true);
    expect(document.activeElement).toBe(btn1);

    btn1.focus();
    const tabFirst = new KeyboardEvent("keydown", {
      key: "Tab",
      shiftKey: false,
      cancelable: true,
    });
    trapModalFocus(tabFirst, modal, closeSpy);
    expect(tabFirst.defaultPrevented).toBe(false);

    const outsideBtn = document.createElement("button");
    document.body.appendChild(outsideBtn);
    outsideBtn.focus();
    const tabFromOutside = new KeyboardEvent("keydown", {
      key: "Tab",
      shiftKey: false,
      cancelable: true,
    });
    trapModalFocus(tabFromOutside, modal, closeSpy);
    expect(tabFromOutside.defaultPrevented).toBe(true);
    expect(document.activeElement).toBe(btn1);

    modal.focus();
    const shiftTabFromRoot = new KeyboardEvent("keydown", {
      key: "Tab",
      shiftKey: true,
      cancelable: true,
    });
    trapModalFocus(shiftTabFromRoot, modal, closeSpy);
    expect(shiftTabFromRoot.defaultPrevented).toBe(true);
    expect(document.activeElement).toBe(btn2);

    const emptyModal = document.createElement("div");
    const tabEmpty = new KeyboardEvent("keydown", {
      key: "Tab",
      cancelable: true,
    });
    trapModalFocus(tabEmpty, emptyModal, closeSpy);
    expect(tabEmpty.defaultPrevented).toBe(false);

    modal.remove();
    outsideBtn.remove();
  });

  it("throws descriptive error when requireElementById cannot find a required element", () => {
    expect(() => requireElementById("nonexistent-required-element")).toThrow(
      "Required DOM element #nonexistent-required-element not found",
    );
  });
});
