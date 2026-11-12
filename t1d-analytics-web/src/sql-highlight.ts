/**
 * @file sql-highlight.ts
 * @module sql-highlight
 * Strict quote-aware SQL statement tokenizer/classifier and live syntax highlighting synchronization.
 */

import { hljs } from "./syntax-highlighter";

const SQL_COMMENT_OR_QUOTED_RE =
  /'(?:''|[^'])*'|"(?:""|[^"])*"|--[^\r\n]*|\/\*(?:(?!\*\/|\/\*).)*\*\//gs;

const SQL_QUOTED_LITERAL_RE = /'(?:''|[^'])*'|"(?:""|[^"])*"/g;

const FORBIDDEN_WRITE_KEYWORD_RE =
  /\b(?:insert\s+into|update\s+\S+\s+set|delete\s+from|drop\s+(?:table|view|index|schema)|alter\s+(?:table|view)|create\s+(?:or\s+replace\s+)?(?:temp(?:orary)?\s+)?(?:table|view|index|schema)|truncate\b|copy\b|attach\b|detach\b)\b/i;

/**
 * Replaces single-quoted string literals and double-quoted identifiers with empty placeholders.
 * @param {string} query - SQL candidate string.
 * @returns {string} Masked SQL string with neutral literal/identifier placeholders.
 */
export function maskSqlLiteralsAndIdentifiers(query: string): string {
  return query.replace(SQL_QUOTED_LITERAL_RE, (tok) =>
    tok.startsWith('"') ? '""' : "''",
  );
}

/**
 * Checks whether `text[0] === "("` and `text[text.length - 1] === ")"` form one enclosing balanced pair.
 * Prevents corrupting compound queries such as `(SELECT 1) UNION ALL (SELECT 2)`.
 * @param {string} text - Trimmed SQL candidate string.
 * @returns {boolean} True only when index `0` closes at `text.length - 1`.
 */
export function hasEnclosingOuterParens(text: string): boolean {
  if (text.length < 2 || !text.startsWith("(") || !text.endsWith(")")) {
    return false;
  }
  const masked = maskSqlLiteralsAndIdentifiers(text);
  let depth = 0;
  const lastIdx = masked.length - 1;
  for (let idx = 0; idx <= lastIdx; idx += 1) {
    const ch = masked[idx];
    if (ch === "(") {
      depth += 1;
    } else if (ch === ")") {
      depth -= 1;
      if (depth === 0 && idx < lastIdx) {
        return false;
      }
    }
  }
  return depth === 0;
}

/**
 * Verifies that all parentheses outside quoted SQL literals/identifiers are balanced.
 * @param {string} masked - Literal-masked SQL string.
 * @returns {boolean} True if parenthesis depth never goes negative and ends at zero.
 */
function hasBalancedParens(masked: string): boolean {
  let depth = 0;
  for (let idx = 0; idx < masked.length; idx += 1) {
    const ch = masked[idx];
    if (ch === "(") {
      depth += 1;
    } else if (ch === ")") {
      depth -= 1;
      if (depth < 0) {
        return false;
      }
    }
  }
  return depth === 0;
}

/**
 * Strips SQL line/block comments outside quoted literals and unwraps balanced enclosing parentheses.
 * @param {string} text - Raw user input string.
 * @returns {string} Normalized SQL candidate preserving compound parenthesized queries.
 */
export function stripSqlCommentsAndEnclosingParens(text: string): string {
  let cleaned = text;
  let prev = "";
  while (prev !== cleaned) {
    prev = cleaned;
    cleaned = cleaned.replace(SQL_COMMENT_OR_QUOTED_RE, (tok) =>
      tok.startsWith("'") || tok.startsWith('"') ? tok : " ",
    );
  }
  cleaned = cleaned.trim();
  while (hasEnclosingOuterParens(cleaned)) {
    cleaned = cleaned.slice(1, -1).trim();
  }
  return cleaned;
}

/**
 * Checks whether a user input string is a valid single-statement read-only SQL query.
 * Uses quote-aware comment stripping, balanced outer-parenthesis unwrapping (preserving
 * `(SELECT 1) UNION ALL (SELECT 2)`), parenthesis balance validation, and write-clause rejection.
 * @param {string} text - Raw input string from the chat textarea.
 * @returns {boolean} True if the input is a read-only SQL statement.
 */
/** Non-SELECT read-only SQL statement prefix patterns. */
const READ_ONLY_VERB_PATTERNS: readonly RegExp[] = [
  /^with\s+(?:recursive\s+)?["a-zA-Z_]["a-zA-Z0-9_]*\s+as\s*\(/i,
  /^show\s+(?:all\s+)?(?:tables|databases|schemas|variables|columns)\b/i,
  /^(?:describe|summarize)\s+(?:table\s+|select\b|["a-zA-Z_]["a-zA-Z0-9_.]*\s*;?$)/i,
  /^explain\s+(?:analyze\s+)?(?:select|with|from|show|describe|summarize|values|table)\b/i,
  /^pragma\s+[a-zA-Z_]/i,
  /^values\s*\(/i,
];

/**
 * Classifies whether a user input string is an unambiguous read-only DuckDB SQL query.
 * @param {string} text - Raw input string from the chat textarea.
 * @returns {boolean} True if the input is a read-only SQL statement.
 */
export function isReadOnlySqlStatement(text: string): boolean {
  const cleaned = stripSqlCommentsAndEnclosingParens(text);
  if (!cleaned || cleaned.endsWith("?") || cleaned.endsWith(".")) {
    return false;
  }

  const masked = maskSqlLiteralsAndIdentifiers(cleaned);
  if (!hasBalancedParens(masked)) {
    return false;
  }
  const statements = masked
    .split(";")
    .map((seg) => seg.trim())
    .filter((seg) => seg.length > 0);
  if (statements.length !== 1 || FORBIDDEN_WRITE_KEYWORD_RE.test(masked)) {
    return false;
  }

  const core = cleaned.replace(/^\(+/, "").trim();
  if (/^select\b/i.test(core)) {
    if (
      /^select\s+(?:the|a|an|all\s+patients|patients\b|subjects\b|participants\b|which\b|who\b)/i.test(
        core,
      ) &&
      !/\bfrom\b/i.test(core)
    ) {
      return false;
    }
    return /^select\s+.+/i.test(core);
  }
  return READ_ONLY_VERB_PATTERNS.some((pattern) => pattern.test(core));
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
