/**
 * @file sql-highlight.ts
 * @module sql-highlight
 * SQL statement classification and live syntax highlighting synchronization.
 */

import hljs from "highlight.js";
import sql from "highlight.js/lib/languages/sql";
import markdown from "highlight.js/lib/languages/markdown";

hljs.registerLanguage("sql", sql);
hljs.registerLanguage("markdown", markdown);

/**
 * Checks whether a user input string is a valid read-only SQL query statement.
 * Strips SQL comments and outer parentheses before verifying the leading keyword.
 * @param {string} text - Raw input string from the chat textarea.
 * @returns {boolean} True if the input is a read-only SQL statement.
 */
export function isReadOnlySqlStatement(text: string): boolean {
  let cleaned = text
    .replace(/--.*$/gm, " ")
    .replace(/\/\*[\s\S]*?\*\//g, " ")
    .trim();
  while (cleaned.startsWith("(") && cleaned.endsWith(")")) {
    cleaned = cleaned.slice(1, -1).trim();
  }
  if (!cleaned) return false;
  if (
    /^(?:show\s+(?:me|us|all|the|a|an)|(?:explain|describe)\s+(?:how|why|what|to|the|this|that|me|us|in|briefly|clearly))\b/i.test(
      cleaned,
    )
  ) {
    return false;
  }
  return /^(select|with|pragma|describe|show|explain|summarize|values)\b/i.test(
    cleaned,
  );
}

/**
 * Synchronizes the live syntax-highlighted code backdrop and auto-resizes the chat textarea.
 * @param {HTMLTextAreaElement} chatInput - The user textarea element.
 * @param {HTMLElement} chatInputHighlight - The backdrop code element for highlight.js.
 * @param {string} selectedModel - Currently selected model or 'sql' mode.
 * @returns {void}
 */
export function updateInputHighlight(
  chatInput: HTMLTextAreaElement,
  chatInputHighlight: HTMLElement,
  selectedModel: string,
): void {
  let value = chatInput.value;

  if (value.endsWith("\n")) {
    value += " ";
  }

  chatInputHighlight.textContent = value;

  if (selectedModel === "sql" || isReadOnlySqlStatement(value)) {
    chatInputHighlight.className = "language-sql";
  } else {
    chatInputHighlight.className = "language-markdown";
  }
  delete chatInputHighlight.dataset.highlighted;
  hljs.highlightElement(chatInputHighlight);

  chatInput.style.height = "auto";
  const scrollHeight = chatInput.scrollHeight;
  chatInput.style.height = `${scrollHeight}px`;
}
