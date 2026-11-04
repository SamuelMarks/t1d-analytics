/**
 * @file message-renderer.ts
 * @module message-renderer
 * Renders chat message bubbles, sanitized Markdown (`DOMPurify` + `marked`),
 * SQL syntax-highlighted blocks with copy/play/refresh controls, tabular SQL results,
 * and clinical CGM visual analytics cards with incremental DOM memoization.
 */

import DOMPurify from "dompurify";
import hljs from "highlight.js";
import sql from "highlight.js/lib/languages/sql";
import markdown from "highlight.js/lib/languages/markdown";
import python from "highlight.js/lib/languages/python";
import i18next from "i18next";
import { marked } from "marked";
import {
  type ApiErrorInput,
  fetchWithBackendError,
  parseApiError,
  parseResponseJson,
} from "./api-client";
import { renderCgmCard } from "./chart";
import { type ProviderSettingsModalController } from "./controllers/provider-modal-controller";
import { type SchemaExplorerController } from "./controllers/schema-explorer-controller";
import {
  type SchemaDiagnosticCardOptions,
  buildEmptyStateView,
  createSchemaDiagnosticCard,
} from "./empty-state-builder";
import { exportRowsToCsv, exportRowsToJson } from "./exporters";
import { classifySqlResult, type SqlCellValue } from "./formatters";
import {
  DEFAULT_DB_NAME,
  RAW_SQL_MODEL,
  type Chat,
  type ChatState,
  type Message,
} from "./state";
import {
  TABLE_1_EXAMPLE_QUERIES,
  type Table1ExampleSpec,
} from "./table1-queries";

export {
  type SchemaDiagnosticCardOptions,
  type Table1ExampleSpec,
  TABLE_1_EXAMPLE_QUERIES,
  createSchemaDiagnosticCard,
};

/**
 * Response payload returned by `/api/chat` and `/api/execute-sql`.
 */
export interface ChatApiResponse {
  /** Assistant response text or translation key. */
  content?: string;
  /** Generated or executed SQL query string. */
  sqlQuery?: string;
  /** Tabular SQL query result rows. */
  sqlResult?: Record<string, SqlCellValue>[];
  /** Structured or string API error payload. */
  error?: ApiErrorInput;
}

hljs.registerLanguage("sql", sql);
hljs.registerLanguage("markdown", markdown);
hljs.registerLanguage("python", python);

const SVG_NS = "http://www.w3.org/2000/svg";

/**
 * Child element specification for `createSvgIcon`.
 */
export interface SvgChildSpec {
  /** SVG shape tag name. */
  tag: "polyline" | "rect" | "line" | "path" | "polygon";
  /** Attribute dictionary for the SVG shape. */
  attrs: Record<string, string>;
}

/**
 * Creates an SVG icon element using `document.createElementNS`.
 * @param {number} size - Width and height in pixels.
 * @param {SvgChildSpec[]} children - SVG shape children to append.
 * @param {string} [fill="none"] - SVG `fill` attribute value.
 * @param {boolean} [ariaHidden=false] - Whether to set `aria-hidden="true"`.
 * @returns {SVGSVGElement} The constructed SVG element.
 */
export function createSvgIcon(
  size: number,
  children: SvgChildSpec[],
  fill: string = "none",
  ariaHidden: boolean = false,
): SVGSVGElement {
  const svg = document.createElementNS(SVG_NS, "svg");
  svg.setAttribute("width", String(size));
  svg.setAttribute("height", String(size));
  svg.setAttribute("viewBox", "0 0 24 24");
  svg.setAttribute("fill", fill);
  svg.setAttribute("stroke", "currentColor");
  svg.setAttribute("stroke-width", "2");
  svg.setAttribute("stroke-linecap", "round");
  svg.setAttribute("stroke-linejoin", "round");
  if (ariaHidden) {
    svg.setAttribute("aria-hidden", "true");
  }
  for (const childSpec of children) {
    const childEl = document.createElementNS(SVG_NS, childSpec.tag);
    for (const [attrName, attrVal] of Object.entries(childSpec.attrs)) {
      childEl.setAttribute(attrName, attrVal);
    }
    svg.appendChild(childEl);
  }
  return svg;
}

/**
 * Formatted SQL execution outcome containing display content and optional tabular rows.
 */
export interface SqlExecutionOutcome {
  /** Localized or scalar-formatted text content for the assistant message. */
  content: string;
  /** Tabular SQL result rows (undefined when rendered inline as a scalar). */
  sqlResult?: Record<string, SqlCellValue>[];
  /** Optional screen-reader announcement string when rows are empty. */
  announcement?: string;
}

/**
 * Formats a SQL result set (`rows`) and `baseContent` into display text and table rows.
 * @param {Record<string, SqlCellValue>[] | undefined} rows - Tabular SQL result rows.
 * @param {string} baseContent - Base assistant text (when coming from `/api/chat`).
 * @param {boolean} isDirectExecution - True when invoked from direct `/api/execute-sql` play/refresh actions.
 * @returns {SqlExecutionOutcome} Formatted content, normalized `sqlResult`, and optional announcement.
 */
export function formatSqlExecutionOutcome(
  rows: Record<string, SqlCellValue>[] | undefined,
  baseContent: string,
  isDirectExecution: boolean,
): SqlExecutionOutcome {
  const classified = classifySqlResult(rows);
  if (isDirectExecution) {
    if (classified.kind === "empty") {
      const noRowsMsg = i18next.t("ui.noRows", "No rows returned");
      return { content: noRowsMsg, sqlResult: [], announcement: noRowsMsg };
    }
    if (classified.kind === "scalar") {
      return { content: classified.formattedValue, sqlResult: undefined };
    }
    return {
      content: i18next.t("app.tableData", "Table Data"),
      sqlResult: rows,
    };
  }

  if (classified.kind === "scalar") {
    const content = baseContent
      ? `${baseContent} **${classified.column}:** ${classified.formattedValue}`
      : classified.formattedValue;
    return { content, sqlResult: rows };
  }
  return { content: baseContent, sqlResult: rows };
}

/**
 * Host contract required by `MessageRenderer`.
 */
export interface MessageRendererHost {
  /** Global application state store. */
  state: ChatState;
  /** Scrollable container element holding chat messages. */
  messagesContainer: HTMLElement;
  /** Wrapper element for the syntax-highlighted textarea input. */
  chatInputWrapper: HTMLElement;
  /** Textarea element where the user types queries or SQL. */
  chatInput: HTMLTextAreaElement;
  /** Submit button element for sending messages. */
  sendBtn: HTMLButtonElement;
  /** Dropdown select element for choosing the active LLM or SQL engine. */
  modelSelect: HTMLSelectElement;
  /** Provider settings modal sub-controller. */
  providerModalController: ProviderSettingsModalController;
  /** Schema explorer sub-controller. */
  schemaExplorerController: SchemaExplorerController;
  /** Synchronizes syntax highlighting in the chat input. */
  syncHighlight(): void;
  /** Announces a message to screen readers. */
  announce(message: string, isAssertive?: boolean): void;
}

/**
 * Cached DOM entry for a rendered `Message` object.
 */
interface CachedMessageEntry {
  /** Deterministic fingerprint of the message content and locale. */
  fingerprint: string;
  /** Reference to the SQL result array when rendered. */
  sqlResultRef: Record<string, SqlCellValue>[] | undefined;
  /** Constructed `.message` DOM element. */
  element: HTMLDivElement;
  /** Whether this message rendered a CGM analytics card. */
  hadCgmCard: boolean;
}

/**
 * Renders chat message lists, empty-state query chips, SQL action blocks, and result tables.
 */
export class MessageRenderer {
  /** Host UI reference. */
  private readonly ui: MessageRendererHost;
  /** WeakMap cache of rendered message DOM nodes keyed by `Message` object identity. */
  private readonly messageCache = new WeakMap<Message, CachedMessageEntry>();

  /**
   * Creates a new `MessageRenderer`.
   * @param {MessageRendererHost} ui - Host UI instance.
   */
  constructor(ui: MessageRendererHost) {
    this.ui = ui;
  }

  /**
   * Builds the empty-state element with optional database notice and example query chips.
   * @returns {HTMLDivElement} The constructed `.empty-state` element.
   */
  private buildEmptyState(): HTMLDivElement {
    return buildEmptyStateView(this.ui);
  }

  /**
   * Creates a Copy SQL icon button for a code block.
   * @param {string} textToCopy - SQL query text to copy to clipboard.
   * @returns {HTMLButtonElement} The constructed Copy button.
   */
  private createCopyBtn(textToCopy: string): HTMLButtonElement {
    const btn = document.createElement("button");
    btn.className = "copy-sql-btn icon-btn";
    btn.setAttribute("aria-label", i18next.t("aria.copyQuery", "Copy Query"));
    btn.setAttribute("data-i18n-aria-label", "aria.copyQuery");
    btn.title = i18next.t("ui.copyQuery", "Copy Query");
    btn.setAttribute("data-i18n-title", "ui.copyQuery");
    const renderCopyIcon = (): SVGSVGElement =>
      createSvgIcon(14, [
        {
          tag: "rect",
          attrs: {
            x: "9",
            y: "9",
            width: "13",
            height: "13",
            rx: "2",
            ry: "2",
          },
        },
        {
          tag: "path",
          attrs: {
            d: "M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1",
          },
        },
      ]);
    btn.replaceChildren(renderCopyIcon());
    btn.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(textToCopy);
        this.ui.announce(i18next.t("status.commandCopied"));
        btn.replaceChildren(
          createSvgIcon(14, [
            { tag: "polyline", attrs: { points: "20 6 9 17 4 12" } },
          ]),
        );
        setTimeout(() => {
          btn.replaceChildren(renderCopyIcon());
        }, 2000);
      } catch (err) {
        console.error("Failed to copy text: ", err);
      }
    });
    return btn;
  }

  /**
   * Creates a Play SQL button for executing a SQL query.
   * @param {string} queryToPlay - SQL query text to execute.
   * @returns {HTMLButtonElement} The constructed Play button.
   */
  private createPlayBtn(queryToPlay: string): HTMLButtonElement {
    const btn = document.createElement("button");
    btn.className = "play-sql-btn";
    btn.setAttribute(
      "aria-label",
      i18next.t("aria.playQuery", "Execute Query"),
    );
    btn.setAttribute("data-i18n-aria-label", "aria.playQuery");
    const playSvg = createSvgIcon(
      14,
      [{ tag: "polygon", attrs: { points: "5 3 19 12 5 21 5 3" } }],
      "currentColor",
      true,
    );
    const playLabelSpan = document.createElement("span");
    playLabelSpan.setAttribute("data-i18n", "app.play");
    playLabelSpan.textContent = i18next.t("app.play", "Play");
    btn.replaceChildren(playSvg, playLabelSpan);
    btn.addEventListener("click", () => {
      void this.executeDirectSqlAction(queryToPlay, btn);
    });
    return btn;
  }

  /**
   * Creates a Refresh SQL button that re-executes a query and updates the message in place.
   * @param {string} queryToPlay - SQL query text to re-execute.
   * @param {Message} messageToUpdate - Target assistant message to update.
   * @returns {HTMLButtonElement} The constructed Refresh button.
   */
  private createRefreshBtn(
    queryToPlay: string,
    messageToUpdate: Message,
  ): HTMLButtonElement {
    const btn = document.createElement("button");
    btn.className = "copy-sql-btn icon-btn";
    btn.setAttribute(
      "aria-label",
      i18next.t("aria.refreshQuery", "Refresh Query"),
    );
    btn.setAttribute("data-i18n-aria-label", "aria.refreshQuery");
    btn.title = i18next.t("ui.refreshQuery", "Refresh Query");
    btn.setAttribute("data-i18n-title", "ui.refreshQuery");
    btn.replaceChildren(
      createSvgIcon(14, [
        { tag: "polyline", attrs: { points: "23 4 23 10 17 10" } },
        { tag: "polyline", attrs: { points: "1 20 1 14 7 14" } },
        {
          tag: "path",
          attrs: {
            d: "M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15",
          },
        },
      ]),
    );
    btn.addEventListener("click", () => {
      void this.executeDirectSqlAction(queryToPlay, btn, messageToUpdate);
    });
    return btn;
  }

  /**
   * Creates the accessible typing indicator element shown while awaiting a response.
   * @returns {HTMLDivElement} Loading message indicator element.
   */
  public createLoadingIndicator(): HTMLDivElement {
    const loadingDiv = document.createElement("div");
    loadingDiv.className = "message assistant loading";
    loadingDiv.setAttribute("role", "status");
    loadingDiv.setAttribute("aria-live", "polite");
    const queryingLabel = i18next.t("ui.querying");
    const typingIndicator = document.createElement("div");
    typingIndicator.className = "typing-indicator";
    typingIndicator.setAttribute("aria-label", queryingLabel);
    typingIndicator.append(
      document.createElement("span"),
      document.createElement("span"),
      document.createElement("span"),
    );
    const srSpan = document.createElement("span");
    srSpan.className = "sr-only";
    srSpan.textContent = queryingLabel;
    typingIndicator.appendChild(srSpan);
    const loadingTextDiv = document.createElement("div");
    loadingTextDiv.style.fontSize = "0.85em";
    loadingTextDiv.style.marginTop = "0.5rem";
    loadingTextDiv.style.opacity = "0.7";
    loadingTextDiv.textContent = queryingLabel;
    loadingDiv.append(typingIndicator, loadingTextDiv);
    return loadingDiv;
  }

  /**
   * Formats and appends an assistant chat response from either SSE streaming or batch `/api/chat`.
   * @param {ChatApiResponse} data - The terminal SSE or batch response payload.
   * @param {string} model - The active model identifier.
   * @param {string} [fallbackText=""] - Fallback accumulated text when `data.content` is empty.
   * @returns {void}
   */
  public finalizeAssistantResponse(
    data: ChatApiResponse,
    model: string,
    fallbackText?: string,
  ): void {
    let assistantMsg =
      data.content && i18next.exists(data.content)
        ? i18next.t(data.content)
        : data.content || fallbackText || "";
    if (data.error) {
      const parsedError = parseApiError(data.error);
      assistantMsg += `\n${i18next.t("ui.errorDetails", { error: parsedError })}`;
      this.ui.announce(assistantMsg, true);
    } else {
      const outcome = formatSqlExecutionOutcome(
        data.sqlResult,
        assistantMsg,
        false,
      );
      assistantMsg = outcome.content;
    }

    this.ui.state.addMessageToActiveChat({
      role: "assistant",
      content: assistantMsg,
      sqlResult: data.sqlResult,
      sqlQuery: data.sqlQuery,
      isError: !!data.error,
      model,
    });
  }

  /**
   * Applies an assistant message update either in-place (on refresh) or as a new chat turn.
   * @param {Message | undefined} messageToUpdate - Optional target message to update in place.
   * @param {string} content - Message text content.
   * @param {Record<string, SqlCellValue>[] | undefined} sqlResult - Optional SQL result rows.
   * @param {string} queryToPlay - Executed SQL query string.
   * @param {boolean} isError - Whether the execution resulted in an error.
   * @returns {void}
   */
  private applyDirectSqlMessageUpdate(
    messageToUpdate: Message | undefined,
    content: string,
    sqlResult: Record<string, SqlCellValue>[] | undefined,
    queryToPlay: string,
    isError: boolean,
  ): void {
    if (messageToUpdate) {
      this.ui.state.updateMessageInActiveChat(messageToUpdate, {
        content,
        sqlResult,
        isError,
      });
    } else {
      this.ui.state.addMessageToActiveChat({
        role: "assistant",
        content,
        sqlResult,
        sqlQuery: queryToPlay,
        isError,
        model: RAW_SQL_MODEL,
      });
    }
    this.renderMessages();
    this.ui.messagesContainer.scrollTop =
      this.ui.messagesContainer.scrollHeight;
  }

  /**
   * Executes a SQL query against `/api/execute-sql` and either appends a new assistant message or updates an existing message in place.
   * @param {string} queryToPlay - The SQL query string to execute.
   * @param {HTMLButtonElement} triggerBtn - The button element that triggered execution.
   * @param {Message} [messageToUpdate] - Optional existing assistant message to update in place.
   * @returns {Promise<void>}
   */
  public async executeDirectSqlAction(
    queryToPlay: string,
    triggerBtn: HTMLButtonElement,
    messageToUpdate?: Message,
  ): Promise<void> {
    triggerBtn.disabled = true;
    try {
      const dbPathOpt =
        this.ui.state.currentDb && this.ui.state.currentDb !== DEFAULT_DB_NAME
          ? { db_path: this.ui.state.currentDb }
          : {};
      const response = await fetchWithBackendError("/api/execute-sql", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: queryToPlay, ...dbPathOpt }),
      });
      const data = await parseResponseJson<ChatApiResponse>(response);

      let contentStr = "";
      let sqlRes: Record<string, SqlCellValue>[] | undefined = data.sqlResult;

      if (data.error) {
        const parsedError = parseApiError(data.error);
        contentStr = i18next.t("ui.errorDetails", { error: parsedError });
        this.ui.announce(contentStr, true);
      } else {
        const outcome = formatSqlExecutionOutcome(sqlRes, "", true);
        contentStr = outcome.content;
        sqlRes = outcome.sqlResult;
        if (outcome.announcement) {
          this.ui.announce(outcome.announcement, false);
        }
      }

      this.applyDirectSqlMessageUpdate(
        messageToUpdate,
        contentStr,
        sqlRes,
        queryToPlay,
        !!data.error,
      );
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : String(err);
      const announcedErr = i18next.t("ui.errorComm", { error: errorMsg });
      if (!messageToUpdate) {
        this.ui.announce(announcedErr, true);
      }
      this.applyDirectSqlMessageUpdate(
        messageToUpdate,
        announcedErr,
        undefined,
        queryToPlay,
        true,
      );
    } finally {
      triggerBtn.disabled = false;
    }
  }

  /**
   * Renders a tabular SQL result container with CSV/JSON exports and optional CGM analytics card.
   * @param {Array<Record<string, SqlCellValue>>} sqlRows - Non-empty SQL result rows.
   * @param {number} cgmCardIndex - Current index for unique CGM SVG IDs.
   * @returns {{ container: HTMLDivElement; hadCgmCard: boolean }} Rendered table container and CGM card flag.
   */
  private buildSqlTableContainer(
    sqlRows: Array<Record<string, SqlCellValue>>,
    cgmCardIndex: number,
  ): { container: HTMLDivElement; hadCgmCard: boolean } {
    const tableContainer = document.createElement("div");
    tableContainer.className = "sql-table-container";

    const toolbar = document.createElement("div");
    toolbar.className = "sql-table-toolbar";
    const csvBtn = document.createElement("button");
    csvBtn.className = "btn-secondary export-csv-btn";
    csvBtn.setAttribute("data-i18n", "table.exportCsvShort");
    csvBtn.textContent = i18next.t("table.exportCsvShort", "CSV");
    csvBtn.setAttribute("data-i18n-aria-label", "table.exportCsv");
    csvBtn.setAttribute(
      "aria-label",
      i18next.t("table.exportCsv", "Export CSV"),
    );
    csvBtn.addEventListener("click", () =>
      exportRowsToCsv(sqlRows, "query_results.csv"),
    );

    const jsonBtn = document.createElement("button");
    jsonBtn.className = "btn-secondary export-json-btn";
    jsonBtn.setAttribute("data-i18n", "table.exportJsonShort");
    jsonBtn.textContent = i18next.t("table.exportJsonShort", "JSON");
    jsonBtn.setAttribute("data-i18n-aria-label", "table.exportJson");
    jsonBtn.setAttribute(
      "aria-label",
      i18next.t("table.exportJson", "Export JSON"),
    );
    jsonBtn.addEventListener("click", () =>
      exportRowsToJson(sqlRows, "query_results.json"),
    );

    toolbar.append(csvBtn, jsonBtn);
    tableContainer.appendChild(toolbar);

    const table = document.createElement("table");
    table.className = "sql-table";

    const caption = document.createElement("caption");
    caption.className = "sr-only";
    caption.textContent = i18next.t("app.tableData");
    table.appendChild(caption);

    const thead = document.createElement("thead");
    const headerRow = document.createElement("tr");
    const columns = Object.keys(sqlRows[0]);

    columns.forEach((col) => {
      const th = document.createElement("th");
      th.textContent = col;
      th.setAttribute("scope", "col");
      headerRow.appendChild(th);
    });
    thead.appendChild(headerRow);
    table.appendChild(thead);

    const tbody = document.createElement("tbody");
    sqlRows.forEach((row) => {
      const tr = document.createElement("tr");
      columns.forEach((col) => {
        const td = document.createElement("td");
        const cellValue = row[col];
        td.textContent =
          cellValue === null || cellValue === undefined
            ? "NULL"
            : String(cellValue);
        tr.appendChild(td);
      });
      tbody.appendChild(tr);
    });
    table.appendChild(tbody);

    const cgmCardSuffix = cgmCardIndex === 0 ? "" : `card-${cgmCardIndex}`;
    const cgmCard = renderCgmCard(sqlRows, 14, cgmCardSuffix);
    if (cgmCard) {
      tableContainer.appendChild(cgmCard);
    }

    tableContainer.appendChild(table);
    return { container: tableContainer, hadCgmCard: Boolean(cgmCard) };
  }

  /**
   * Builds a single `.message` DOM element for a chat message.
   * @param {Message} msg - The message model to render.
   * @param {Chat} activeChat - The active chat session.
   * @param {number} cgmCardIndex - Current index for unique CGM SVG IDs.
   * @returns {{ element: HTMLDivElement; hadCgmCard: boolean }} Constructed message node and CGM card flag.
   */
  private buildMessageElement(
    msg: Message,
    activeChat: Chat,
    cgmCardIndex: number,
  ): { element: HTMLDivElement; hadCgmCard: boolean } {
    const msgDiv = document.createElement("div");
    msgDiv.className = `message ${msg.role}`;

    if (msg.model) {
      const header = document.createElement("div");
      header.className = "message-header";
      const badge = document.createElement("span");
      badge.className = "model-badge";
      badge.textContent =
        msg.model === RAW_SQL_MODEL
          ? i18next.t("ui.rawSql")
          : i18next.t("ui.modelName", { name: msg.model });

      if (msg.role === "user") {
        header.style.justifyContent = "flex-end";
        badge.style.backgroundColor = "rgba(255, 255, 255, 0.2)";
        badge.style.color = "var(--user-msg-text)";
      }

      header.appendChild(badge);
      msgDiv.appendChild(header);
    }

    const contentDiv = document.createElement("div");
    const isSqlMessage =
      msg.role === "user" &&
      (msg.model === RAW_SQL_MODEL ||
        (!msg.model && activeChat.model === RAW_SQL_MODEL));

    if (isSqlMessage) {
      contentDiv.className = "sql-user-query-wrapper";
      const headerDiv = document.createElement("div");
      headerDiv.className = "sql-query-header";
      headerDiv.append(
        this.createCopyBtn(msg.content),
        this.createPlayBtn(msg.content),
      );

      const preBlock = document.createElement("pre");
      preBlock.style.margin = "0";
      preBlock.style.background = "transparent";
      preBlock.style.padding = "0";
      preBlock.style.whiteSpace = "pre-wrap";
      preBlock.tabIndex = 0;
      preBlock.setAttribute("aria-label", i18next.t("aria.codeBlock"));
      preBlock.setAttribute("data-i18n-aria-label", "aria.codeBlock");

      const codeBlock = document.createElement("code");
      codeBlock.className = "language-sql";
      codeBlock.style.background = "transparent";
      codeBlock.style.padding = "0";
      codeBlock.textContent = msg.content;

      hljs.highlightElement(codeBlock);
      preBlock.appendChild(codeBlock);
      contentDiv.append(headerDiv, preBlock);
    } else {
      const rawHtml = String(marked.parse(msg.content ?? ""));
      contentDiv.innerHTML = DOMPurify.sanitize(rawHtml, {
        ADD_ATTR: ["target"],
      });
      contentDiv.querySelectorAll<HTMLElement>("pre").forEach((preElement) => {
        preElement.querySelectorAll<HTMLElement>("code").forEach((block) => {
          hljs.highlightElement(block);
          if (block.className.includes("language-sql")) {
            const wrapper = document.createElement("div");
            wrapper.className = "sql-query-container";
            const headerDiv = document.createElement("div");
            headerDiv.className = "sql-query-header";
            const queryStr = String(block.textContent);
            headerDiv.append(
              this.createCopyBtn(queryStr),
              this.createPlayBtn(queryStr),
            );
            preElement.replaceWith(wrapper);
            wrapper.append(headerDiv, preElement);
          }
        });
      });
    }

    if (msg.isError) {
      contentDiv.className = "error-text";
      contentDiv.setAttribute("role", "alert");
      contentDiv.setAttribute("aria-live", "assertive");
    }

    msgDiv.appendChild(contentDiv);

    const hasMarkdownSqlBlock =
      contentDiv.querySelector(".sql-query-container") !== null;
    if (msg.sqlQuery && !hasMarkdownSqlBlock) {
      const queryContainer = document.createElement("div");
      queryContainer.className = "sql-query-container";
      const preBlock = document.createElement("pre");
      preBlock.tabIndex = 0;
      preBlock.setAttribute("aria-label", i18next.t("aria.codeBlock"));
      preBlock.setAttribute("data-i18n-aria-label", "aria.codeBlock");
      const codeBlock = document.createElement("code");
      codeBlock.className = "language-sql sql-query";
      codeBlock.textContent = msg.sqlQuery;
      hljs.highlightElement(codeBlock);
      preBlock.appendChild(codeBlock);

      const headerDiv = document.createElement("div");
      headerDiv.className = "sql-query-header";
      headerDiv.appendChild(this.createCopyBtn(msg.sqlQuery));
      if (msg.role === "assistant" && msg.model === RAW_SQL_MODEL) {
        headerDiv.appendChild(this.createRefreshBtn(msg.sqlQuery, msg));
      } else {
        headerDiv.appendChild(this.createPlayBtn(msg.sqlQuery));
      }

      queryContainer.append(headerDiv, preBlock);
      msgDiv.appendChild(queryContainer);
    }

    let hadCgmCard = false;
    const sqlRows = msg.sqlResult;
    if (sqlRows) {
      if (sqlRows.length > 0) {
        const built = this.buildSqlTableContainer(sqlRows, cgmCardIndex);
        hadCgmCard = built.hadCgmCard;
        msgDiv.appendChild(built.container);
      } else {
        const emptyDiv = document.createElement("div");
        emptyDiv.className = "sql-empty";
        emptyDiv.textContent = i18next.t("ui.noRows");
        msgDiv.appendChild(emptyDiv);
      }
    }

    return { element: msgDiv, hadCgmCard };
  }

  /**
   * Renders the messages of the active chat into `messagesContainer`, reusing cached
   * DOM nodes for unchanged messages to preserve scroll positions and avoid redundant SVG/Markdown builds.
   * @returns {void}
   */
  public renderMessages(): void {
    const activeChat = this.ui.state.getActiveChat();
    if (!activeChat) return;

    if (activeChat.messages.length === 0) {
      this.ui.messagesContainer.replaceChildren(this.buildEmptyState());
      return;
    }

    let cgmCardIndex = 0;
    const desiredNodes: HTMLDivElement[] = [];

    for (const msg of activeChat.messages) {
      const effectiveModel =
        msg.model ?? (msg.role === "user" ? activeChat.model : "");
      const fingerprint = `${i18next.language}|${msg.role}|${effectiveModel}|${Boolean(msg.isError)}|${msg.content ?? ""}|${msg.sqlQuery ?? ""}|${msg.sqlResult ? msg.sqlResult.length : -1}|${cgmCardIndex}`;
      const cached = this.messageCache.get(msg);

      if (
        cached &&
        cached.fingerprint === fingerprint &&
        cached.sqlResultRef === msg.sqlResult
      ) {
        desiredNodes.push(cached.element);
        if (cached.hadCgmCard) {
          cgmCardIndex += 1;
        }
      } else {
        const built = this.buildMessageElement(msg, activeChat, cgmCardIndex);
        this.messageCache.set(msg, {
          fingerprint,
          sqlResultRef: msg.sqlResult,
          element: built.element,
          hadCgmCard: built.hadCgmCard,
        });
        desiredNodes.push(built.element);
        if (built.hadCgmCard) {
          cgmCardIndex += 1;
        }
      }
    }

    const desiredSet = new Set<Element>(desiredNodes);
    Array.from(this.ui.messagesContainer.children).forEach((child) => {
      if (!desiredSet.has(child)) {
        child.remove();
      }
    });

    for (let i = 0; i < desiredNodes.length; i++) {
      const desired = desiredNodes[i];
      const current = this.ui.messagesContainer.children[i];
      if (current !== desired) {
        this.ui.messagesContainer.insertBefore(desired, current ?? null);
      }
    }
  }

  /**
   * Renders the active chat pane, updating input controls, model selector, and message nodes.
   * @returns {void}
   */
  public renderActiveChat(): void {
    const activeChat = this.ui.state.getActiveChat();

    if (!activeChat) {
      const emptyDiv = document.createElement("div");
      emptyDiv.className = "empty-state";
      emptyDiv.setAttribute("role", "status");
      emptyDiv.setAttribute("aria-live", "polite");
      emptyDiv.setAttribute("data-i18n", "app.emptyState");
      emptyDiv.textContent = i18next.t("app.emptyState");
      this.ui.messagesContainer.replaceChildren(emptyDiv);
      this.ui.chatInput.disabled = true;
      this.ui.chatInputWrapper.classList.add("disabled");
      this.ui.sendBtn.disabled = true;
      this.ui.modelSelect.disabled = true;
      return;
    }

    const online = this.ui.state.systemStatus.backendOnline;
    this.ui.chatInput.disabled = !online;
    this.ui.chatInputWrapper.classList.toggle("disabled", !online);
    this.ui.chatInput.placeholder = online
      ? i18next.t("app.typeMessage")
      : i18next.t("status.inputOffline");
    this.ui.sendBtn.disabled = !online;

    this.ui.modelSelect.disabled = false;
    const hasActiveModelOption = Array.from(this.ui.modelSelect.options).some(
      (opt) => opt.value === activeChat.model,
    );
    this.ui.modelSelect.value = hasActiveModelOption
      ? activeChat.model
      : RAW_SQL_MODEL;
    this.ui.providerModalController.updateProviderBadgeStatus();
    this.ui.syncHighlight();
    this.renderMessages();
  }
}
