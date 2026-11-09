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
  if (cleaned.endsWith("?") || cleaned.endsWith(".")) {
    return false;
  }
  if (/^select\b/i.test(cleaned)) {
    if (
      /^select\s+(?:the|a|an|all\s+patients|patients\b|subjects\b|participants\b|which\b|who\b)/i.test(
        cleaned,
      ) &&
      !/\bfrom\b/i.test(cleaned)
    ) {
      return false;
    }
    return /^select\s+.+/i.test(cleaned);
  }
  if (/^with\b/i.test(cleaned)) {
    return /^with\s+(?:recursive\s+)?["a-zA-Z_]["a-zA-Z0-9_]*\s+as\s*\(/i.test(
      cleaned,
    );
  }
  if (/^show\b/i.test(cleaned)) {
    return /^show\s+(?:all\s+)?(?:tables|databases|schemas|variables)\b/i.test(
      cleaned,
    );
  }
  if (/^(?:describe|summarize)\b/i.test(cleaned)) {
    return /^(?:describe|summarize)\s+(?:table\s+|select\b|["a-zA-Z_]["a-zA-Z0-9_.]*\s*;?$)/i.test(
      cleaned,
    );
  }
  if (/^explain\b/i.test(cleaned)) {
    return /^explain\s+(?:analyze\s+)?(?:select|with|from|show|describe|summarize|values|table)\b/i.test(
      cleaned,
    );
  }
  if (/^pragma\b/i.test(cleaned)) {
    return /^pragma\s+[a-zA-Z_]/i.test(cleaned);
  }
  if (/^values\b/i.test(cleaned)) {
    return /^values\s*\(/i.test(cleaned);
  }
  return false;
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
