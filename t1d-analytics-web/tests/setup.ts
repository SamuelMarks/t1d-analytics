/**
 * @file setup.ts
 * @module tests/setup
 * Vitest global test setup initializing mock storage, i18next, and typed DOM query helpers.
 */

import "./mock-storage";
import "../src/i18n";

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
 * Mounts the standard ChatUI DOM fixture into `document.body` for unit testing.
 * @returns {void}
 */
export function mountChatUiTestDom(): void {
  document.body.innerHTML = `
      <nav id="sidebar">
        <button id="new-chat-btn">New Chat</button>
        <button id="close-sidebar-btn">Close</button>
        <ul id="chat-list"></ul>
        <footer id="sidebar-footer" class="sidebar-footer">
          <div id="build-timestamp" class="sidebar-footer-build"></div>
          <div class="sidebar-footer-meta">
            <span id="sidebar-copyright">© 2026 Samuel Marks (Offscale.io)</span>
            <a id="sidebar-github-link" href="https://github.com/SamuelMarks/t1d-analytics" target="_blank" rel="noopener noreferrer">GitHub</a>
          </div>
        </footer>
        <div id="schema-explorer">
          <div class="schema-header">
            <button id="toggle-schema-btn"></button>
          </div>
          <div id="schema-content"></div>
        </div>
      </nav>
      <div id="table-modal" aria-hidden="true">
        <h2 id="modal-title"></h2>
        <button id="close-modal-btn"></button>
        <input type="search" id="modal-table-search" />
        <select id="modal-table-sort-col"></select>
        <button id="modal-table-sort-order">ASC</button>
        <button id="prev-page-btn"></button>
        <button id="next-page-btn"></button>
        <button id="modal-export-csv-btn" disabled></button>
        <button id="modal-export-json-btn" disabled></button>
        <button id="modal-export-full-csv-btn" disabled></button>
        <button id="modal-export-excel-btn" disabled></button>
        <button id="modal-export-parquet-btn" disabled></button>
        <div id="table-loading"></div>
        <table id="modal-table">
          <caption class="sr-only" id="modal-table-caption">Table Data</caption>
          <thead id="modal-table-head"></thead>
          <tbody id="modal-table-body"></tbody>
        </table>
        <span id="page-indicator"></span>
      </div>
      <main id="main-pane">
        <button id="system-status-chip" class="status-chip status-healthy">
          <span class="status-dot"></span>
          <span id="status-chip-text" class="status-text">All systems operational</span>
        </button>
        <button id="open-sidebar-btn">Open</button>
        <button id="theme-toggle-btn">Theme</button>
        <select id="model-select">
          <option value="gemma4">Gemma</option>
          <option value="sql">SQL</option>
        </select>
        <button id="provider-settings-btn">⚙️</button>
        <span id="provider-status-dot" class="active">●</span>
        <select id="db-select">
          <option value="t1d.duckdb">t1d.duckdb</option>
        </select>
        <button id="streaming-toggle-btn">⚡</button>
        <button id="sync-sessions-btn">Sync</button>
        <button id="cohort-filter-btn">Cohort</button>
        <select id="lang-select">
          <option value="en">English</option>
          <option value="ja">Japanese</option>
        </select>
        <div id="cohort-modal" class="modal-overlay hidden">
          <button id="close-cohort-modal-btn">✕</button>
          <select id="cohort-table-select"></select>
          <select id="cohort-join-table-select">
            <option value="">None (Single Table)</option>
          </select>
          <select id="cohort-join-type-select">
            <option value="INNER JOIN">INNER JOIN</option>
            <option value="LEFT JOIN">LEFT JOIN</option>
          </select>
          <input type="text" id="cohort-join-on-input" />
          <input type="number" id="cohort-min-age" />
          <input type="number" id="cohort-max-age" />
          <select id="cohort-gender-select">
            <option value="">Any</option>
            <option value="M">Male</option>
            <option value="F">Female</option>
          </select>
          <input type="text" id="cohort-txgroup" />
          <input type="number" id="cohort-max-tir" />
          <input type="number" id="cohort-min-hba1c" />
          <code id="cohort-sql-preview"></code>
          <button id="cohort-insert-chat-btn">Insert</button>
          <button id="cohort-execute-btn">Execute</button>
        </div>
        <div id="provider-modal" class="modal-overlay hidden">
          <button id="close-provider-modal-btn">✕</button>
          <input type="password" id="provider-key-openai" />
          <input type="password" id="provider-key-anthropic" />
          <input type="password" id="provider-key-google" />
          <input type="checkbox" id="provider-store-consent" checked />
          <span id="provider-badge-openai"></span>
          <span id="provider-badge-anthropic"></span>
          <span id="provider-badge-google"></span>
          <button id="provider-save-btn">Save</button>
          <button id="provider-clear-btn">Clear</button>
        </div>
        <div id="system-status-banner" class="system-status-banner hidden">
          <span id="banner-icon">⚠️</span>
          <strong id="banner-title"></strong>
          <p id="banner-desc"></p>
          <button id="banner-retry-btn">Retry</button>
          <button id="banner-instructions-btn">Instructions</button>
          <div id="banner-instructions-drawer" class="banner-drawer hidden"></div>
          <button id="banner-dismiss-btn">✕</button>
        </div>
        <div id="messages-container"></div>
        <form id="chat-form">
          <div id="chat-input-wrapper" class="chat-input-wrapper">
            <pre aria-hidden="true"><code id="chat-input-highlight"></code></pre>
            <textarea id="chat-input" aria-describedby="chat-input-help"></textarea>
          </div>
          <div id="chat-input-help" class="sr-only">Press Enter to send, Shift+Enter for a new line</div>
          <button id="send-btn" type="submit">Send</button>
        </form>
      </main>
      <div id="overlay"></div>
      <div id="a11y-announcer"></div>
    `;
}
