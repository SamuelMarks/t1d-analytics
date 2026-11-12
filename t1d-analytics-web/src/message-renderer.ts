/**
 * @file message-renderer.ts
 * @module message-renderer
 * Renders chat message bubbles, sanitized Markdown (`DOMPurify` + `marked`),
 * SQL syntax-highlighted blocks with copy/play/refresh controls, paginated tabular SQL results,
 * and clinical CGM visual analytics cards with incremental DOM memoization.
 */

import DOMPurify from "dompurify";
import i18next from "i18next";
import { marked } from "marked";
import { hljs } from "./syntax-highlighter";
import {
  fetchWithBackendError,
  parseApiError,
  parseResponseJson,
} from "./api-client";
import { type ChatTurnResponse } from "./api-types";
import { renderCgmCard } from "./chart";
import {
  type SchemaDiagnosticCardOptions,
  buildEmptyStateView,
  createSchemaDiagnosticCard,
} from "./empty-state-builder";
import { type AppEventBus } from "./event-bus";
import { exportRowsToCsv, exportRowsToJson } from "./exporters";
import { classifySqlResult, type SqlCellValue } from "./formatters";
import {
  DEFAULT_DB_NAME,
  RAW_SQL_MODEL,
  type Chat,
  type ChatState,
  type Message,
  type SchemaTableMetadata,
  type SqlGoldComparisonPayload,
} from "./state";
import {
  TABLE_1_EXAMPLE_QUERIES,
  type Table1ExampleSpec,
} from "./table1-queries";

export {
  type ChatTurnResponse,
  type SchemaDiagnosticCardOptions,
  type Table1ExampleSpec,
  TABLE_1_EXAMPLE_QUERIES,
  createSchemaDiagnosticCard,
};

/** Maximum number of SQL result rows rendered per page in inline chat tables. */
export const INLINE_SQL_TABLE_PAGE_SIZE = 100;

/**
 * Extracts an underlying LLM failure reason strictly from structured `msg.llmError` metadata.
 * @param {Message} msg - Assistant message record.
 * @returns {string | null} Trimmed human-readable error reason, or null if none.
 */
export function extractFallbackErrorDetail(msg: Message): string | null {
  if (msg.llmError && msg.llmError.trim().length > 0) {
    return msg.llmError.trim();
  }
  return null;
}

/**
 * Checks whether an assistant message used rule-based fallback SQL synthesis strictly via
 * structured boolean/enum fields (`isRuleBasedSql` or `generationMode === "rule_based_fallback"`).
 * @param {Message} msg - Assistant message record.
 * @returns {boolean} True if fallback SQL synthesis was used.
 */
export function isFallbackSynthesisMessage(msg: Message): boolean {
  return (
    msg.isRuleBasedSql === true || msg.generationMode === "rule_based_fallback"
  );
}

/**
 * Checks whether an `ApiErrorInput` value represents a non-empty error condition.
 * @param {ChatTurnResponse["error"]} err - Candidate API error field.
 * @returns {boolean} True when an error payload is present.
 */
function hasApiError(err: ChatTurnResponse["error"]): boolean {
  return err !== undefined && err !== null && err !== "" && err !== false;
}

/**
 * Resolves a known schema table referenced inside `sqlQuery`, or returns null if none matches.
 * @param {string | undefined} sqlQuery - Executed SQL query string.
 * @param {SchemaTableMetadata[]} schemaTables - Active schema table metadata.
 * @returns {string | null} Matched table name, or null.
 */
export function resolveReferencedSchemaTable(
  sqlQuery: string | undefined,
  schemaTables: SchemaTableMetadata[],
): string | null {
  if (!sqlQuery || schemaTables.length === 0) {
    return null;
  }
  /**
   * Tests whether a character can form part of an unquoted SQL identifier.
   * @param {string | undefined} ch - Character to inspect.
   * @returns {boolean} True if `ch` is `[a-z0-9_]` (case-insensitive).
   */
  const isIdentChar = (ch: string | undefined): boolean =>
    ch !== undefined && /[a-z0-9_]/i.test(ch);
  const lowerSql = sqlQuery.toLowerCase();
  for (const tbl of schemaTables) {
    const needle = tbl.name.toLowerCase();
    if (!needle) continue;
    let pos = lowerSql.indexOf(needle);
    while (pos !== -1) {
      const prev = lowerSql[pos - 1];
      const next = lowerSql[pos + needle.length];
      if (!isIdentChar(prev) && !isIdentChar(next)) {
        return tbl.name;
      }
      pos = lowerSql.indexOf(needle, pos + 1);
    }
  }
  return null;
}

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
  const rowsSpread = rows !== undefined ? { sqlResult: rows } : {};
  if (isDirectExecution) {
    if (classified.kind === "empty") {
      const noRowsMsg = i18next.t("ui.noRows", "No rows returned");
      return { content: noRowsMsg, sqlResult: [], announcement: noRowsMsg };
    }
    if (classified.kind === "scalar") {
      return { content: classified.formattedValue };
    }
    return {
      content: i18next.t("app.tableData", "Table Data"),
      ...rowsSpread,
    };
  }

  if (classified.kind === "scalar") {
    const content = baseContent
      ? `${baseContent} **${classified.column}:** ${classified.formattedValue}`
      : classified.formattedValue;
    return {
      content,
      ...rowsSpread,
    };
  }
  return {
    content: baseContent,
    ...rowsSpread,
  };
}

/**
 * Narrow injected dependencies for `MessageRenderer`.
 */
export interface MessageRendererDeps {
  /** Global application state store. */
  state: ChatState;
  /** Unidirectional domain event bus. */
  events: AppEventBus;
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
  /** Callback opening the Cloud Provider & API Key Settings modal. */
  onOpenProviderSettings: () => void;
  /** Callback opening a database table in the Table Preview Explorer modal. */
  onOpenTableExplorer: (tableName: string) => void;
  /** Binds interaction events to action buttons inside diagnostic cards. */
  onBindDiagnosticButtons: (container: HTMLElement) => void;
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
 * Renders chat message lists, empty-state query chips, SQL action blocks, and paginated result tables.
 */
export class MessageRenderer {
  /** Injected dependencies. */
  private readonly deps: MessageRendererDeps;
  /** WeakMap cache of rendered message DOM nodes keyed by `Message` object identity. */
  private readonly messageCache = new WeakMap<Message, CachedMessageEntry>();

  /**
   * Creates a new `MessageRenderer`.
   * @param {MessageRendererDeps} deps - Narrow injected state, DOM, and callback dependencies.
   */
  constructor(deps: MessageRendererDeps) {
    this.deps = deps;
  }

  /**
   * Emits a screen-reader announcement via the domain event bus.
   * @param {string} message - Message string to announce.
   * @param {boolean} [isAssertive=false] - Whether to use assertive politeness.
   * @returns {void}
   */
  private announce(message: string, isAssertive: boolean = false): void {
    this.deps.events.emit("announce", { message, isAssertive });
  }

  /**
   * Builds the empty-state element with optional database notice and example query chips.
   * @returns {HTMLDivElement} The constructed `.empty-state` element.
   */
  private buildEmptyState(): HTMLDivElement {
    return buildEmptyStateView({
      state: this.deps.state,
      modelSelect: this.deps.modelSelect,
      onBindDiagnosticButtons: this.deps.onBindDiagnosticButtons,
      onSelectTemplate: (selection) => {
        this.deps.events.emit("templateSelected", selection);
      },
    });
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
    /**
     * Constructs the default clipboard SVG icon.
     * @returns {SVGSVGElement} Clipboard SVG element.
     */
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
    btn.addEventListener("click", () => {
      void (async (): Promise<void> => {
        try {
          await navigator.clipboard.writeText(textToCopy);
          this.announce(i18next.t("status.commandCopied"));
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
      })();
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
   * Builds a `Message` object from a completed `ChatTurnResponse` payload.
   * @param {ChatTurnResponse} data - Terminal response payload.
   * @param {string} model - Active model identifier.
   * @param {string} content - Formatted assistant text.
   * @param {boolean} isError - Whether `data.error` was set.
   * @param {string | undefined} llmError - Trimmed underlying LLM failure reason.
   * @returns {Message} Constructed assistant `Message` record.
   */
  private buildAssistantMessageRecord(
    data: ChatTurnResponse,
    model: string,
    content: string,
    isError: boolean,
    llmError: string | undefined,
  ): Message {
    const llmErrorCode = data.llm_error_code?.trim() || undefined;
    return {
      role: "assistant",
      content,
      isError,
      model,
      ...(data.sqlResult !== undefined ? { sqlResult: data.sqlResult } : {}),
      ...(data.sqlQuery !== undefined ? { sqlQuery: data.sqlQuery } : {}),
      ...(data.gold_comparison ? { goldComparison: data.gold_comparison } : {}),
      ...(data.is_rule_based_sql !== undefined
        ? { isRuleBasedSql: data.is_rule_based_sql }
        : {}),
      ...(data.generation_mode !== undefined
        ? { generationMode: data.generation_mode }
        : {}),
      ...(llmError ? { llmError } : {}),
      ...(llmErrorCode ? { llmErrorCode } : {}),
    };
  }

  /**
   * Formats and appends an assistant chat response from either SSE streaming or batch `/api/chat`.
   * @param {ChatTurnResponse} data - The terminal SSE or batch response payload.
   * @param {string} model - The active model identifier.
   * @param {string} [fallbackText=""] - Fallback accumulated text when `data.content` is empty.
   * @returns {void}
   */
  public finalizeAssistantResponse(
    data: ChatTurnResponse,
    model: string,
    fallbackText?: string,
  ): void {
    let assistantMsg =
      data.content && i18next.exists(data.content)
        ? i18next.t(data.content)
        : data.content || fallbackText || "";
    const isError = hasApiError(data.error);
    if (isError) {
      const parsedError = parseApiError(data.error);
      assistantMsg += `\n${i18next.t("ui.errorDetails", { error: parsedError })}`;
      this.announce(assistantMsg, true);
    } else {
      assistantMsg = formatSqlExecutionOutcome(
        data.sqlResult,
        assistantMsg,
        false,
      ).content;
    }

    const resolvedLlmError = data.llm_error?.trim() || undefined;
    const isFallbackMode =
      data.is_rule_based_sql === true ||
      data.generation_mode === "rule_based_fallback";

    if (!isError && isFallbackMode && resolvedLlmError) {
      this.announce(
        `${i18next.t("ui.fallbackBannerTitle", "⚠️ Fallback SQL Synthesized")} — ${i18next.t("ui.fallbackErrorPrefix", "Underlying LLM Error:")} ${resolvedLlmError}`,
        false,
      );
    }

    this.deps.state.addMessageToActiveChat(
      this.buildAssistantMessageRecord(
        data,
        model,
        assistantMsg,
        isError,
        resolvedLlmError,
      ),
    );
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
      this.deps.chatInput.value = msg.content;
      this.deps.state.setActiveGoldSql(msg.goldSql ?? null);
      this.deps.events.emit("composerSyncRequested", { focus: true });
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
    const isFallback = isFallbackSynthesisMessage(msg);
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

    /**
     * Constructs a syntax-highlighted SQL diff pane inside the Gold comparison card.
     * @param {string} heading - Pane title text.
     * @param {string} sqlText - SQL statement string.
     * @returns {HTMLDivElement} Constructed pane element.
     */
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

    if (comp.gold_rows_preview.length > 0) {
      const previewHeading = document.createElement("div");
      previewHeading.className = "gold-sql-preview-heading";
      previewHeading.textContent = i18next.t(
        "eval.goldPreviewHeading",
        "Gold-Standard Read-Only DuckDB Result Preview:",
      );
      const builtGoldTable = this.buildSqlTableContainer(
        comp.gold_rows_preview,
        9999,
        comp.gold_sql,
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
      const patch: Partial<Message> = {
        content,
        isError,
        ...(sqlResult !== undefined ? { sqlResult } : {}),
      };
      if (sqlResult === undefined) {
        delete messageToUpdate.sqlResult;
      }
      this.deps.state.updateMessageInActiveChat(messageToUpdate, patch);
    } else {
      this.deps.state.addMessageToActiveChat({
        role: "assistant",
        content,
        sqlQuery: queryToPlay,
        isError,
        model: RAW_SQL_MODEL,
        ...(sqlResult !== undefined ? { sqlResult } : {}),
      });
    }
    this.renderMessages();
    this.deps.messagesContainer.scrollTop =
      this.deps.messagesContainer.scrollHeight;
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
        this.deps.state.currentDb &&
        this.deps.state.currentDb !== DEFAULT_DB_NAME
          ? { db_path: this.deps.state.currentDb }
          : {};
      const response = await fetchWithBackendError("/api/execute-sql", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: queryToPlay, ...dbPathOpt }),
      });
      const data = await parseResponseJson<ChatTurnResponse>(response);

      let contentStr = "";
      let sqlRes: Record<string, SqlCellValue>[] | undefined = data.sqlResult;

      const isError = hasApiError(data.error);
      if (isError) {
        const parsedError = parseApiError(data.error);
        contentStr = i18next.t("ui.errorDetails", { error: parsedError });
        this.announce(contentStr, true);
      } else {
        const outcome = formatSqlExecutionOutcome(sqlRes, "", true);
        contentStr = outcome.content;
        sqlRes = outcome.sqlResult;
        if (outcome.announcement) {
          this.announce(outcome.announcement, false);
        }
      }

      this.applyDirectSqlMessageUpdate(
        messageToUpdate,
        contentStr,
        sqlRes,
        queryToPlay,
        isError,
      );
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : String(err);
      const announcedErr = i18next.t("ui.errorComm", { error: errorMsg });
      if (!messageToUpdate) {
        this.announce(announcedErr, true);
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
   * Appends a slice of SQL result rows (`[startIdx, endIdx)`) as `<tr>` elements into `tbody`.
   * @param {HTMLTableSectionElement} tbody - Target `<tbody>` element.
   * @param {Array<Record<string, SqlCellValue>>} sqlRows - Full SQL result array.
   * @param {string[]} columns - Ordered column names.
   * @param {number} startIdx - Inclusive start row index.
   * @param {number} endIdx - Exclusive end row index.
   * @returns {void}
   */
  private appendSqlRowSlice(
    tbody: HTMLTableSectionElement,
    sqlRows: Array<Record<string, SqlCellValue>>,
    columns: string[],
    startIdx: number,
    endIdx: number,
  ): void {
    const fragment = document.createDocumentFragment();
    for (const row of sqlRows.slice(startIdx, endIdx)) {
      const tr = document.createElement("tr");
      for (const col of columns) {
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
      }
      fragment.appendChild(tr);
    }
    tbody.appendChild(fragment);
  }

  /**
   * Renders a paginated tabular SQL result container (100 rows per page) with CSV/JSON exports,
   * "Show next 100 rows" / "Open in Table Explorer" controls, and optional CGM analytics card.
   * @param {Array<Record<string, SqlCellValue>>} sqlRows - Non-empty SQL result rows.
   * @param {number} cgmCardIndex - Current index for unique CGM SVG IDs.
   * @param {string} [sqlQuery] - Optional SQL query string used to resolve a Table Explorer target.
   * @returns {{ container: HTMLDivElement; hadCgmCard: boolean }} Rendered table container and CGM card flag.
   */
  private buildSqlTableContainer(
    sqlRows: Array<Record<string, SqlCellValue>>,
    cgmCardIndex: number,
    sqlQuery?: string,
  ): { container: HTMLDivElement; hadCgmCard: boolean } {
    const tableContainer = document.createElement("div");
    tableContainer.className = "sql-table-container";

    const toolbar = document.createElement("div");
    toolbar.className = "sql-table-toolbar";

    const columns = Object.keys(Object(sqlRows[0]));
    const metaBadge = document.createElement("span");
    metaBadge.className = "sql-table-meta-badge";
    metaBadge.textContent = i18next.t("table.rowColSummary", {
      rows: sqlRows.length,
      cols: columns.length,
      defaultValue: `${sqlRows.length} rows × ${columns.length} cols`,
    });

    const actionsGroup = document.createElement("div");
    actionsGroup.className = "sql-table-toolbar-actions";

    const referencedTable = resolveReferencedSchemaTable(
      sqlQuery,
      this.deps.state.schemaTables,
    );
    if (referencedTable) {
      const explorerBtn = document.createElement("button");
      explorerBtn.type = "button";
      explorerBtn.className = "btn-secondary open-table-explorer-btn";
      explorerBtn.textContent = i18next.t(
        "table.openInExplorer",
        "Open in Table Explorer",
      );
      explorerBtn.addEventListener("click", () => {
        this.deps.onOpenTableExplorer(referencedTable);
      });
      actionsGroup.appendChild(explorerBtn);
    }

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
    let renderedRows = Math.min(sqlRows.length, INLINE_SQL_TABLE_PAGE_SIZE);
    this.appendSqlRowSlice(tbody, sqlRows, columns, 0, renderedRows);
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

    if (sqlRows.length > INLINE_SQL_TABLE_PAGE_SIZE) {
      const paginationBar = document.createElement("div");
      paginationBar.className = "sql-table-pagination-bar";

      const rangeInfo = document.createElement("span");
      rangeInfo.className = "sql-table-pagination-info";

      const loadMoreBtn = document.createElement("button");
      loadMoreBtn.type = "button";
      loadMoreBtn.className = "btn-secondary sql-table-load-more-btn";

      /**
       * Refreshes the pagination range summary and "Show next N rows" button visibility.
       * @returns {void}
       */
      const updatePaginationControls = (): void => {
        rangeInfo.textContent = i18next.t("table.showingRowsRange", {
          shown: renderedRows,
          total: sqlRows.length,
          defaultValue: `Showing 1–${renderedRows} of ${sqlRows.length} rows`,
        });
        const remaining = sqlRows.length - renderedRows;
        if (remaining <= 0) {
          loadMoreBtn.remove();
        } else {
          const nextBatch = Math.min(remaining, INLINE_SQL_TABLE_PAGE_SIZE);
          loadMoreBtn.textContent = i18next.t("table.showNextRows", {
            count: nextBatch,
            defaultValue: `Show next ${nextBatch} rows`,
          });
        }
      };

      loadMoreBtn.addEventListener("click", () => {
        const nextEnd = Math.min(
          renderedRows + INLINE_SQL_TABLE_PAGE_SIZE,
          sqlRows.length,
        );
        this.appendSqlRowSlice(tbody, sqlRows, columns, renderedRows, nextEnd);
        renderedRows = nextEnd;
        updatePaginationControls();
      });

      updatePaginationControls();
      paginationBar.append(rangeInfo, loadMoreBtn);
      tableContainer.appendChild(paginationBar);
    }

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
      this.deps.onOpenProviderSettings();
    });
    actionsRow.appendChild(configBtn);

    banner.append(headerRow, detailBlock, actionsRow);
    return banner;
  }

  /**
   * Builds the optional `.message-header` row containing model badge and user retry button.
   * @param {Message} msg - Message model.
   * @returns {HTMLDivElement | null} Header element or null when omitted.
   */
  private buildMessageHeaderElement(msg: Message): HTMLDivElement | null {
    if (!msg.model && msg.role !== "user") {
      return null;
    }
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
    return header;
  }

  /**
   * Builds the primary content container (`.sql-user-query-wrapper` or sanitized Markdown) for `msg`.
   * @param {Message} msg - Message model.
   * @param {boolean} isSqlMessage - Whether `msg` is a raw user SQL query.
   * @returns {HTMLDivElement} Populated content container.
   */
  private buildMessageBodyContent(
    msg: Message,
    isSqlMessage: boolean,
  ): HTMLDivElement {
    const contentDiv = document.createElement("div");
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

    if (msg.isError === true) {
      contentDiv.classList.add("error-text");
      contentDiv.setAttribute("role", "alert");
      contentDiv.setAttribute("aria-live", "assertive");
    } else if (!isSqlMessage) {
      contentDiv.classList.add("message-content");
    }
    return contentDiv;
  }

  /**
   * Builds a standalone `.sql-query-container` block when `msg.sqlQuery` is not already rendered in Markdown.
   * @param {Message} msg - Message model.
   * @param {string} sqlQuery - SQL query text.
   * @returns {HTMLDivElement} Constructed SQL query container.
   */
  private buildStandaloneSqlBlock(
    msg: Message,
    sqlQuery: string,
  ): HTMLDivElement {
    const queryContainer = document.createElement("div");
    queryContainer.className = "sql-query-container";
    const preBlock = document.createElement("pre");
    preBlock.tabIndex = 0;
    preBlock.setAttribute("aria-label", i18next.t("aria.codeBlock"));
    preBlock.setAttribute("data-i18n-aria-label", "aria.codeBlock");
    const codeBlock = document.createElement("code");
    codeBlock.className = "language-sql sql-query";
    codeBlock.textContent = sqlQuery;
    hljs.highlightElement(codeBlock);
    preBlock.appendChild(codeBlock);

    const headerDiv = document.createElement("div");
    headerDiv.className = "sql-query-header";
    headerDiv.appendChild(this.createCopyBtn(sqlQuery));
    headerDiv.appendChild(
      msg.role === "assistant" && msg.model === RAW_SQL_MODEL
        ? this.createRefreshBtn(sqlQuery, msg)
        : this.createPlayBtn(sqlQuery),
    );

    queryContainer.append(headerDiv, preBlock);
    return queryContainer;
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

    const headerEl = this.buildMessageHeaderElement(msg);
    if (headerEl) {
      msgDiv.appendChild(headerEl);
    }

    const isSqlMessage =
      msg.role === "user" &&
      (msg.model === RAW_SQL_MODEL ||
        (!msg.model && activeChat.model === RAW_SQL_MODEL));
    const contentDiv = this.buildMessageBodyContent(msg, isSqlMessage);

    const fallbackBanner = this.buildFallbackDiagnosticBanner(msg);
    if (fallbackBanner) {
      msgDiv.appendChild(fallbackBanner);
    }
    msgDiv.appendChild(contentDiv);

    if (msg.sqlQuery && !contentDiv.querySelector(".sql-query-container")) {
      msgDiv.appendChild(this.buildStandaloneSqlBlock(msg, msg.sqlQuery));
    }

    let hadCgmCard = false;
    const sqlRows = msg.sqlResult;
    if (sqlRows) {
      if (sqlRows.length > 0) {
        const built = this.buildSqlTableContainer(
          sqlRows,
          cgmCardIndex,
          msg.sqlQuery,
        );
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
   * Computes a deterministic memoization fingerprint for a `Message` and active render context.
   * @param {Message} msg - Message model.
   * @param {Chat} activeChat - Active chat session.
   * @param {number} cgmCardIndex - Current CGM card index.
   * @returns {string} Memoization fingerprint string.
   */
  private computeMessageFingerprint(
    msg: Message,
    activeChat: Chat,
    cgmCardIndex: number,
  ): string {
    const effectiveModel =
      msg.model ?? (msg.role === "user" ? activeChat.model : "");
    const goldKey = msg.goldComparison
      ? `${String(msg.goldComparison.execution_accuracy)}|${String(msg.goldComparison.exact_match)}|${msg.goldComparison.column_overlap_pct}`
      : "";
    const schemaCount = this.deps.state.schemaTables.length;
    const rowLen = msg.sqlResult ? msg.sqlResult.length : -1;
    return `${i18next.language}|${msg.role}|${effectiveModel}|${String(Boolean(msg.isError))}|${msg.content}|${msg.sqlQuery ?? ""}|${rowLen}|${goldKey}|${String(Boolean(msg.isRuleBasedSql))}|${msg.generationMode ?? ""}|${msg.llmError ?? ""}|${msg.llmErrorCode ?? ""}|${schemaCount}|${cgmCardIndex}`;
  }

  /**
   * Reconciles `messagesContainer.children` in place to match `desiredNodes` without disturbing unchanged nodes.
   * @param {HTMLDivElement[]} desiredNodes - Ordered target message DOM elements.
   * @returns {void}
   */
  private reconcileMessageDomNodes(desiredNodes: HTMLDivElement[]): void {
    const desiredSet = new Set<Element>(desiredNodes);
    Array.from(this.deps.messagesContainer.children).forEach((child) => {
      if (!desiredSet.has(child)) {
        child.remove();
      }
    });

    for (const [i, desired] of desiredNodes.entries()) {
      const current = this.deps.messagesContainer.children[i];
      if (current !== desired) {
        this.deps.messagesContainer.insertBefore(desired, current ?? null);
      }
    }
  }

  /**
   * Renders the messages of the active chat into `messagesContainer`, reusing cached
   * DOM nodes for unchanged messages to preserve scroll positions and avoid redundant SVG/Markdown builds.
   * @returns {void}
   */
  public renderMessages(): void {
    const activeChat = this.deps.state.getActiveChat();
    if (!activeChat) return;

    if (activeChat.messages.length === 0) {
      this.deps.messagesContainer.replaceChildren(this.buildEmptyState());
      return;
    }

    let cgmCardIndex = 0;
    const desiredNodes: HTMLDivElement[] = [];

    for (const msg of activeChat.messages) {
      const fingerprint = this.computeMessageFingerprint(
        msg,
        activeChat,
        cgmCardIndex,
      );
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

    this.reconcileMessageDomNodes(desiredNodes);
  }

  /**
   * Renders the active chat pane, updating input controls, model selector, and message nodes.
   * @returns {void}
   */
  public renderActiveChat(): void {
    const activeChat = this.deps.state.getActiveChat();

    if (!activeChat) {
      const emptyDiv = document.createElement("div");
      emptyDiv.className = "empty-state";
      emptyDiv.setAttribute("role", "status");
      emptyDiv.setAttribute("aria-live", "polite");
      emptyDiv.setAttribute("data-i18n", "app.emptyState");
      emptyDiv.textContent = i18next.t("app.emptyState");
      this.deps.messagesContainer.replaceChildren(emptyDiv);
      this.deps.chatInput.disabled = true;
      this.deps.chatInputWrapper.classList.add("disabled");
      this.deps.sendBtn.disabled = true;
      this.deps.modelSelect.disabled = true;
      return;
    }

    const online = this.deps.state.systemStatus.backendOnline;
    this.deps.chatInput.disabled = !online;
    this.deps.chatInputWrapper.classList.toggle("disabled", !online);
    this.deps.chatInput.placeholder = online
      ? i18next.t("app.typeMessage")
      : i18next.t("status.inputOffline");
    this.deps.sendBtn.disabled = !online;

    this.deps.modelSelect.disabled = false;
    const hasActiveModelOption = Array.from(this.deps.modelSelect.options).some(
      (opt) => opt.value === activeChat.model,
    );
    this.deps.modelSelect.value = hasActiveModelOption
      ? activeChat.model
      : RAW_SQL_MODEL;
    this.deps.events.emit("providerBadgeUpdateRequested", undefined);
    this.deps.events.emit("composerSyncRequested", {});
    this.renderMessages();
  }
}
