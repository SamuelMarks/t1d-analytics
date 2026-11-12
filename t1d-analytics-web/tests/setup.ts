/**
 * @file setup.ts
 * @module tests/setup
 * Vitest global test setup initializing mock storage, i18next, and cached `index.html` DOM mounting.
 */

import fs from "fs";
import path from "path";
import "./mock-storage";
import "../src/i18n";

/** Cached `<body>` inner markup extracted from the real `t1d-analytics-web/index.html`. */
let cachedIndexBodyMarkup: string | null = null;

/**
 * Reads and caches the `<body>` inner markup from `t1d-analytics-web/index.html`,
 * stripping `<script>` module tags so unit tests mount the exact production HTML structure.
 * @returns {string} Sanitized `<body>` markup from `index.html`.
 */
function loadIndexBodyMarkup(): string {
  if (cachedIndexBodyMarkup !== null) {
    return cachedIndexBodyMarkup;
  }
  const indexHtmlPath = path.resolve(__dirname, "../index.html");
  const fullHtml = fs.readFileSync(indexHtmlPath, "utf-8");
  const bodyMatch = /<body[^>]*>([\s\S]*?)<\/body>/i.exec(fullHtml);
  const rawBody = bodyMatch?.[1] ?? fullHtml;
  cachedIndexBodyMarkup = rawBody.replace(
    /<script\b[^>]*>[\s\S]*?<\/script>/gi,
    "",
  );
  return cachedIndexBodyMarkup;
}

/**
 * Looks up a typed HTML element by ID in the test DOM.
 * @template T - Expected HTML element subtype.
 * @param {string} id - Element ID without leading `#`.
 * @returns {T} Matched element.
 * @throws {Error} When the element is not present in the test DOM.
 */
export function getTestElement<T extends HTMLElement = HTMLElement>(
  id: string,
): T {
  const el = document.querySelector<T>(`#${id}`);
  if (!el) {
    throw new Error(`Test DOM element #${id} not found`);
  }
  return el;
}

/**
 * Queries a single typed element by CSS selector in the test DOM.
 * @template T - Expected Element subtype.
 * @param {string} selector - CSS selector string.
 * @param {ParentNode} [root=document] - Optional parent container.
 * @returns {T} Matched element.
 * @throws {Error} When no element matches the selector in the test DOM.
 */
export function queryTestElement<T extends Element = HTMLElement>(
  selector: string,
  root: ParentNode = document,
): T {
  const el = root.querySelector<T>(selector);
  if (!el) {
    throw new Error(`Test DOM element matching '${selector}' not found`);
  }
  return el;
}

/**
 * Queries all typed elements matching a CSS selector in the test DOM.
 * @template T - Expected Element subtype.
 * @param {string} selector - CSS selector string.
 * @param {ParentNode} [root=document] - Optional parent container.
 * @returns {NodeListOf<T>} Matched NodeList.
 */
export function queryAllTestElements<T extends Element = HTMLElement>(
  selector: string,
  root: ParentNode = document,
): NodeListOf<T> {
  return root.querySelectorAll<T>(selector);
}

/**
 * Deterministically flushes pending microtasks in the test event loop without arbitrary real-time sleeps.
 * @param {number} [ticks=30] - Number of microtask turns to flush.
 * @returns {Promise<void>} A promise that resolves once pending microtasks have settled.
 */
export async function flushPromises(ticks: number = 30): Promise<void> {
  for (let i = 0; i < ticks; i++) {
    await Promise.resolve();
  }
}

/**
 * Mounts the real `t1d-analytics-web/index.html` `<body>` DOM into `document.body` for unit testing.
 * @returns {void}
 */
export function mountChatUiTestDom(): void {
  document.body.className = "";
  document.body.innerHTML = loadIndexBodyMarkup();
}
