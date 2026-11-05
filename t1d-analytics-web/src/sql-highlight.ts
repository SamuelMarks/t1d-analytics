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

/** Maximum pixel height for the auto-resizing chat composer textarea. */
export const MAX_CHAT_INPUT_HEIGHT_PX = 220;

/**
 * Synchronizes the live syntax-highlighted code backdrop and auto-resizes the chat textarea.
 * Only applies `language-sql` highlighting when `selectedModel === "sql"` or `isReadOnlySqlStatement(value)` is true;
 * otherwise renders clean plaintext so English words (`from`, `in`, `select`, `as`) in natural-language prompts
 * are not miscolored as SQL keywords.
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

  const shouldHighlightSql =
    selectedModel === "sql" || isReadOnlySqlStatement(value);
  delete chatInputHighlight.dataset.highlighted;
  if (shouldHighlightSql) {
    chatInputHighlight.className = "language-sql";
    hljs.highlightElement(chatInputHighlight);
  } else {
    chatInputHighlight.className = "language-plaintext";
  }

  chatInput.style.height = "auto";
  const scrollHeight = chatInput.scrollHeight;
  const clampedHeight =
    scrollHeight > MAX_CHAT_INPUT_HEIGHT_PX
      ? MAX_CHAT_INPUT_HEIGHT_PX
      : scrollHeight;
  chatInput.style.height = `${clampedHeight}px`;
  chatInput.style.overflowY =
    scrollHeight > MAX_CHAT_INPUT_HEIGHT_PX ? "auto" : "hidden";
  if (chatInputHighlight.parentElement) {
    chatInputHighlight.parentElement.scrollTop = chatInput.scrollTop;
  }
}
