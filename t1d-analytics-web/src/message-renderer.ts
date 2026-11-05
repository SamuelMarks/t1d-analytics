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
  type SqlGoldComparisonPayload,
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
  /** Optional gold-standard SQL evaluation metrics returned by the backend. */
  gold_comparison?: SqlGoldComparisonPayload | null;
  /** Whether deterministic rule-based SQL synthesis was used instead of an LLM. */
  is_rule_based_sql?: boolean;
  /** SQL generation mode ('llm', 'rule_based_fallback', or 'literal_sql'). */
  generation_mode?: string;
  /** Optional human-readable diagnostic reason why LLM inference failed prior to fallback synthesis. */
  llm_error?: string | null;
  /** Optional structured error code for the underlying LLM failure. */
  llm_error_code?: string | null;
}

const FALLBACK_PARENTHETICAL_ERROR_RE =
  /Local LLM inference was unavailable\s*\(([^)]+)\)/i;

/**
 * Extracts an underlying LLM failure reason from either explicit `msg.llmError` metadata
 * or a parenthetical `Note: Local LLM inference was unavailable (<reason>);` in `msg.content`.
 * @param {Message} msg - Assistant message record.
 * @returns {string | null} Extracted human-readable error reason, or null if none.
 */
export function extractFallbackErrorDetail(msg: Message): string | null {
  if (msg.llmError && msg.llmError.trim()) {
    return msg.llmError.trim();
  }
  const match = FALLBACK_PARENTHETICAL_ERROR_RE.exec(msg.content);
  if (match && match[1].trim()) {
    return match[1].trim();
  }
  return null;
}

/**
 * Checks whether an assistant message used rule-based fallback SQL synthesis.
 * @param {Message} msg - Assistant message record.
 * @returns {boolean} True if fallback SQL synthesis was used.
 */
export function isFallbackSynthesisMessage(msg: Message): boolean {
  if (
    Boolean(msg.isRuleBasedSql) ||
    msg.generationMode === "rule_based_fallback" ||
    Boolean(msg.llmError)
  ) {
    return true;
  }
  const text = msg.content;
  return (
    text.includes("Local LLM inference was unavailable") ||
    text.includes("Local LLM inference encountered a memory error")
  );
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
  /** Updates the provider badge status indicator for the active model. */
  updateProviderBadgeStatus(): void;
  /** Binds interaction events to action buttons inside diagnostic cards. */
  bindSchemaDiagnosticButtons(container: HTMLElement): void;
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
    let resetTimerId: ReturnType<typeof setTimeout> | null = null;
    btn.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(textToCopy);
        this.ui.announce(i18next.t("status.commandCopied"));
        if (resetTimerId !== null) {
          clearTimeout(resetTimerId);
        }
        btn.replaceChildren(
          createSvgIcon(14, [
            { tag: "polyline", attrs: { points: "20 6 9 17 4 12" } },
          ]),
        );
        resetTimerId = setTimeout(() => {
          resetTimerId = null;
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

    const resolvedLlmError =
      data.llm_error && data.llm_error.trim()
        ? data.llm_error.trim()
        : undefined;
    const resolvedLlmErrorCode =
      data.llm_error_code && data.llm_error_code.trim()
        ? data.llm_error_code.trim()
        : undefined;

    if (
      !data.error &&
      (data.is_rule_based_sql ||
        data.generation_mode === "rule_based_fallback") &&
      resolvedLlmError
    ) {
      this.ui.announce(
        `${i18next.t("ui.fallbackBannerTitle", "⚠️ Fallback SQL Synthesized")} — ${i18next.t("ui.fallbackErrorPrefix", "Underlying LLM Error:")} ${resolvedLlmError}`,
        false,
      );
    }

    this.ui.state.addMessageToActiveChat({
      role: "assistant",
      content: assistantMsg,
      sqlResult: data.sqlResult,
      sqlQuery: data.sqlQuery,
      isError: !!data.error,
      model,
      goldComparison: data.gold_comparison ?? undefined,
      isRuleBasedSql: data.is_rule_based_sql,
      generationMode: data.generation_mode,
      ...(resolvedLlmError ? { llmError: resolvedLlmError } : {}),
      ...(resolvedLlmErrorCode ? { llmErrorCode: resolvedLlmErrorCode } : {}),
    });
  }

  /**
   * Creates an accessible "Load / Re-run in Composer" button for a user message so users can
   * re-test the same prompt with a different LLM model without copy-pasting.
   * @param {Message} msg - User message to load into the composer.
   * @returns {HTMLButtonElement} The constructed retry/edit button.
   */
  private createEditRetryBtn(msg: Message): HTMLButtonElement {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "edit-retry-msg-btn icon-btn";
    const label = i18next.t(
      "ui.loadPromptInComposer",
      "Load prompt in composer",
    );
    btn.setAttribute("aria-label", label);
    btn.title = label;
    btn.textContent = "↻";
    btn.addEventListener("click", () => {
      this.ui.chatInput.value = msg.content;
      this.ui.state.setActiveGoldSql(msg.goldSql ?? null);
      this.ui.syncHighlight();
      this.ui.chatInput.focus();
    });
    return btn;
  }

  /**
   * Builds an interactive Gold-Standard SQL Evaluation Card comparing the LLM's generated SQL
   * against the reference Gold SQL executed on the read-only DuckDB cursor.
   * @param {SqlGoldComparisonPayload} comp - Gold-standard SQL comparison metrics.
   * @param {Message} msg - Parent assistant message.
   * @returns {HTMLDivElement} Constructed `.gold-sql-comparison-card` element.
   */
  private buildGoldComparisonCard(
    comp: SqlGoldComparisonPayload,
    msg: Message,
  ): HTMLDivElement {
    const card = document.createElement("div");
    card.className = "gold-sql-comparison-card";
    card.setAttribute("role", "region");
    card.setAttribute(
      "aria-label",
      i18next.t("eval.cardTitle", "Gold-Standard SQL Evaluation"),
    );

    const header = document.createElement("div");
    header.className = "gold-sql-card-header";

    const titleEl = document.createElement("strong");
    titleEl.className = "gold-sql-card-title";
    titleEl.textContent = i18next.t(
      "eval.cardTitle",
      "⚖️ NL-to-SQL vs. Gold-Standard Evaluation (DuckDB Read-Only Cursor)",
    );

    const badgesRow = document.createElement("div");
    badgesRow.className = "gold-sql-badges-row";

    const exBadge = document.createElement("span");
    exBadge.className = comp.execution_accuracy
      ? "eval-badge eval-badge-pass"
      : "eval-badge eval-badge-fail";
    exBadge.textContent = comp.execution_accuracy
      ? i18next.t("eval.exPass", "Execution Match (EX): PASS")
      : i18next.t("eval.exFail", "Execution Match (EX): DIFF");

    const emBadge = document.createElement("span");
    emBadge.className = comp.exact_match
      ? "eval-badge eval-badge-pass"
      : "eval-badge eval-badge-neutral";
    emBadge.textContent = comp.exact_match
      ? i18next.t("eval.emPass", "Exact SQL Match: YES")
      : i18next.t("eval.emFail", "Exact SQL Match: NO");

    const approxBadge = document.createElement("span");
    approxBadge.className = "eval-badge eval-badge-score";
    approxBadge.textContent = i18next.t("eval.approxScore", {
      pct: comp.column_overlap_pct.toFixed(1),
      defaultValue: `Value Approximation: ${comp.column_overlap_pct.toFixed(1)}%`,
    });

    const modeBadge = document.createElement("span");
    const isFallback =
      Boolean(msg.isRuleBasedSql) ||
      msg.generationMode === "rule_based_fallback";
    modeBadge.className = isFallback
      ? "eval-badge eval-badge-warn"
      : "eval-badge eval-badge-info";
    modeBadge.textContent = isFallback
      ? i18next.t("eval.modeFallback", "Mode: Rule-Based Fallback")
      : i18next.t("eval.modeLlm", "Mode: Neural LLM");

    badgesRow.append(exBadge, emBadge, approxBadge, modeBadge);
    header.append(titleEl, badgesRow);
    card.appendChild(header);

    const metaRow = document.createElement("div");
    metaRow.className = "gold-sql-meta-row";
    metaRow.textContent = `DuckDB Read-Only Cursor Rows — Predicted: ${comp.pred_row_count} (${comp.pred_columns.join(", ") || "none"}) | Gold Standard: ${comp.gold_row_count} (${comp.gold_columns.join(", ") || "none"})`;
    card.appendChild(metaRow);

    if (comp.syntax_error && comp.error_message) {
      const errBanner = document.createElement("div");
      errBanner.className = "gold-sql-error-banner";
      errBanner.textContent = comp.error_message;
      card.appendChild(errBanner);
    }

    const detailsEl = document.createElement("details");
    detailsEl.className = "gold-sql-diff-details";
    const summaryEl = document.createElement("summary");
    summaryEl.textContent = i18next.t(
      "eval.inspectDiff",
      "Inspect Target LLM SQL vs. Gold-Standard SQL & DuckDB Rows",
    );

    const splitGrid = document.createElement("div");
    splitGrid.className = "gold-sql-split-grid";

    const buildPane = (heading: string, sqlText: string): HTMLDivElement => {
      const pane = document.createElement("div");
      pane.className = "gold-sql-pane";
      const paneHeader = document.createElement("div");
      paneHeader.className = "gold-sql-pane-header";
      const label = document.createElement("span");
      label.textContent = heading;
      const actions = document.createElement("div");
      actions.className = "gold-sql-pane-actions";
      if (sqlText) {
        actions.append(
          this.createCopyBtn(sqlText),
          this.createPlayBtn(sqlText),
        );
      }
      paneHeader.append(label, actions);
      const pre = document.createElement("pre");
      const code = document.createElement("code");
      code.className = "language-sql";
      code.textContent = sqlText || "-- (none)";
      hljs.highlightElement(code);
      pre.appendChild(code);
      pane.append(paneHeader, pre);
      return pane;
    };

    splitGrid.append(
      buildPane(
        i18next.t("eval.predSqlHeading", "Target LLM Generated SQL"),
        comp.pred_sql,
      ),
      buildPane(
        i18next.t("eval.goldSqlHeading", "Gold-Standard Reference SQL"),
        comp.gold_sql,
      ),
    );
    detailsEl.append(summaryEl, splitGrid);

    if (comp.gold_rows_preview && comp.gold_rows_preview.length > 0) {
      const previewHeading = document.createElement("div");
      previewHeading.className = "gold-sql-preview-heading";
      previewHeading.textContent = i18next.t(
        "eval.goldPreviewHeading",
        "Gold-Standard Read-Only DuckDB Result Preview:",
      );
      const builtGoldTable = this.buildSqlTableContainer(
        comp.gold_rows_preview,
        9999,
      );
      detailsEl.append(previewHeading, builtGoldTable.container);
    }

    card.appendChild(detailsEl);
    return card;
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

    const columns = Object.keys(sqlRows[0]);
    const metaBadge = document.createElement("span");
    metaBadge.className = "sql-table-meta-badge";
    metaBadge.textContent = i18next.t("table.rowColSummary", {
      rows: sqlRows.length,
      cols: columns.length,
      defaultValue: `${sqlRows.length} rows × ${columns.length} cols`,
    });

    const actionsGroup = document.createElement("div");
    actionsGroup.className = "sql-table-toolbar-actions";

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

    actionsGroup.append(csvBtn, jsonBtn);
    toolbar.append(metaBadge, actionsGroup);
    tableContainer.appendChild(toolbar);

    const table = document.createElement("table");
    table.className = "sql-table";

    const caption = document.createElement("caption");
    caption.className = "sr-only";
    caption.textContent = i18next.t("app.tableData");
    table.appendChild(caption);

    const thead = document.createElement("thead");
    const headerRow = document.createElement("tr");

    columns.forEach((col) => {
      const th = document.createElement("th");
      th.textContent = col;
      th.title = col;
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
        const cellText =
          cellValue === null || cellValue === undefined
            ? "NULL"
            : String(cellValue);
        td.textContent = cellText;
        if (cellText.length > 40) {
          td.title = cellText;
        }
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

    const scrollWrapper = document.createElement("div");
    scrollWrapper.className = "sql-table-scroll-wrapper";
    scrollWrapper.appendChild(table);
    tableContainer.appendChild(scrollWrapper);
    return { container: tableContainer, hadCgmCard: Boolean(cgmCard) };
  }

  /**
   * Builds a diagnostic banner for assistant messages that used rule-based fallback SQL synthesis,
   * explicitly surfacing the underlying LLM failure reason and a quick provider configuration button.
   * @param {Message} msg - Assistant message record.
   * @returns {HTMLDivElement | null} Constructed `.fallback-synthesis-banner` element, or null if not applicable.
   */
  private buildFallbackDiagnosticBanner(msg: Message): HTMLDivElement | null {
    if (msg.role !== "assistant" || !isFallbackSynthesisMessage(msg)) {
      return null;
    }
    const errDetail = extractFallbackErrorDetail(msg);
    if (!errDetail) {
      return null;
    }

    const banner = document.createElement("div");
    banner.className = "fallback-synthesis-banner";
    banner.setAttribute("role", "status");
    banner.setAttribute("aria-live", "polite");

    const headerRow = document.createElement("div");
    headerRow.className = "fallback-banner-header";

    const titleSpan = document.createElement("strong");
    titleSpan.className = "fallback-banner-title";
    titleSpan.textContent = i18next.t(
      "ui.fallbackBannerTitle",
      "⚠️ Fallback SQL Synthesized",
    );
    headerRow.appendChild(titleSpan);

    if (msg.llmErrorCode) {
      const codeChip = document.createElement("code");
      codeChip.className = "fallback-error-code";
      codeChip.textContent = msg.llmErrorCode;
      headerRow.appendChild(codeChip);
    }

    const detailBlock = document.createElement("div");
    detailBlock.className = "fallback-error-detail";
    const prefixStrong = document.createElement("strong");
    prefixStrong.textContent = `${i18next.t("ui.fallbackErrorPrefix", "Underlying LLM Error:")} `;
    const reasonSpan = document.createElement("span");
    reasonSpan.className = "fallback-error-reason";
    reasonSpan.textContent = errDetail;
    detailBlock.append(prefixStrong, reasonSpan);

    const actionsRow = document.createElement("div");
    actionsRow.className = "fallback-banner-actions";
    const configBtn = document.createElement("button");
    configBtn.type = "button";
    configBtn.className = "btn-secondary btn-xs fallback-configure-btn";
    configBtn.textContent = i18next.t(
      "ui.fallbackActionConfigure",
      "Configure LLM / Keys",
    );
    configBtn.addEventListener("click", () => {
      document.getElementById("provider-settings-btn")?.click();
    });
    actionsRow.appendChild(configBtn);

    banner.append(headerRow, detailBlock, actionsRow);
    return banner;
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

    if (msg.model || msg.role === "user") {
      const header = document.createElement("div");
      header.className = "message-header";
      if (msg.model) {
        const badge = document.createElement("span");
        badge.className = "model-badge";
        badge.textContent =
          msg.model === RAW_SQL_MODEL
            ? i18next.t("ui.rawSql")
            : i18next.t("ui.modelName", { name: msg.model });

        if (msg.role === "user") {
          badge.style.backgroundColor = "rgba(255, 255, 255, 0.2)";
          badge.style.color = "var(--user-msg-text)";
        }
        header.appendChild(badge);
      }

      if (msg.role === "user") {
        header.style.justifyContent = "flex-end";
        header.appendChild(this.createEditRetryBtn(msg));
      }

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
      const rawHtml = String(marked.parse(msg.content));
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
      contentDiv.classList.add("error-text");
      contentDiv.setAttribute("role", "alert");
      contentDiv.setAttribute("aria-live", "assertive");
    } else if (!isSqlMessage) {
      contentDiv.classList.add("message-content");
    }

    const fallbackBanner = this.buildFallbackDiagnosticBanner(msg);
    if (fallbackBanner) {
      msgDiv.appendChild(fallbackBanner);
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

    if (msg.goldComparison) {
      msgDiv.appendChild(this.buildGoldComparisonCard(msg.goldComparison, msg));
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
      const goldKey = msg.goldComparison
        ? `${msg.goldComparison.execution_accuracy}|${msg.goldComparison.exact_match}|${msg.goldComparison.column_overlap_pct}`
        : "";
      const fingerprint = `${i18next.language}|${msg.role}|${effectiveModel}|${Boolean(msg.isError)}|${msg.content}|${msg.sqlQuery ?? ""}|${msg.sqlResult ? msg.sqlResult.length : -1}|${goldKey}|${msg.llmError ?? ""}|${msg.llmErrorCode ?? ""}|${cgmCardIndex}`;
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
    this.ui.updateProviderBadgeStatus();
    this.ui.syncHighlight();
    this.renderMessages();
  }
}
