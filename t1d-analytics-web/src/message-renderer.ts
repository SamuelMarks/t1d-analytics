/**
 * @file message-renderer.ts
 * @module message-renderer
 * Renders chat message bubbles, sanitized Markdown (`DOMPurify` + `marked`),
 * SQL syntax-highlighted blocks with copy/play/refresh controls, tabular SQL results,
 * and clinical CGM visual analytics cards.
 */

import DOMPurify from "dompurify";
import hljs from "highlight.js";
import sql from "highlight.js/lib/languages/sql";
import markdown from "highlight.js/lib/languages/markdown";
import python from "highlight.js/lib/languages/python";
import i18next from "i18next";
import { marked } from "marked";
import { renderCgmCard } from "./chart";
import { type ExportableCellValue } from "./exporters";
import { type ChatState, type Message } from "./state";

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
 * Options for constructing a schema diagnostic card via DOM APIs.
 */
export interface SchemaDiagnosticCardOptions {
  /** Leading icon text. */
  icon: string;
  /** Header title text. */
  title: string;
  /** Optional descriptive paragraph text. */
  description?: string;
  /** Optional CLI command string to render with a copy button. */
  command?: string;
  /** Whether to include a reload schema button. */
  showReload?: boolean;
  /** Whether to apply the `.error-text` styling and `role="alert"`. */
  isError?: boolean;
  /** Whether to set `role="alert"` on the card container. */
  isAlert?: boolean;
}

/**
 * Constructs a `.schema-diagnostic-card` element using safe DOM APIs without `innerHTML`.
 * @param {SchemaDiagnosticCardOptions} options - Card configuration options.
 * @returns {HTMLDivElement} The constructed diagnostic card element.
 */
export function createSchemaDiagnosticCard(
  options: SchemaDiagnosticCardOptions,
): HTMLDivElement {
  const card = document.createElement("div");
  card.className = options.isError
    ? "schema-diagnostic-card error-text"
    : "schema-diagnostic-card";
  if (options.isAlert || options.isError) {
    card.setAttribute("role", "alert");
  }

  const header = document.createElement("div");
  header.className = "schema-diagnostic-header";
  const iconSpan = document.createElement("span");
  iconSpan.textContent = options.icon;
  const titleSpan = document.createElement("span");
  titleSpan.textContent = options.title;
  header.append(iconSpan, titleSpan);
  card.appendChild(header);

  if (options.description) {
    const descP = document.createElement("p");
    descP.className = "schema-diagnostic-desc";
    descP.textContent = options.description;
    card.appendChild(descP);
  }

  if (options.command) {
    const cmdBox = document.createElement("div");
    cmdBox.className = "schema-command-box";
    const codeEl = document.createElement("code");
    codeEl.textContent = options.command;
    const copyBtn = document.createElement("button");
    copyBtn.className = "btn btn-xs btn-secondary copy-cmd-btn";
    copyBtn.setAttribute("data-cmd", options.command);
    copyBtn.textContent = i18next.t("status.copyCommand");
    cmdBox.append(codeEl, copyBtn);
    card.appendChild(cmdBox);
  }

  if (options.showReload) {
    const reloadBtn = document.createElement("button");
    reloadBtn.className = "btn btn-xs btn-primary reload-schema-btn";
    reloadBtn.textContent = i18next.t("status.reloadSchema");
    card.appendChild(reloadBtn);
  }

  return card;
}

/**
 * Host contract required by `MessageRenderer`.
 */
export interface MessageRendererHost {
  /** Global application state store. */
  state: ChatState;
  /** Scrollable container element holding chat messages. */
  messagesContainer: HTMLElement;
  /** Textarea element where the user types queries or SQL. */
  chatInput: HTMLTextAreaElement;
  /** Dropdown select element for choosing the active LLM or SQL engine. */
  modelSelect: HTMLSelectElement;
  /** Synchronizes syntax highlighting in the chat input. */
  syncHighlight(): void;
  /** Binds copy/reload diagnostic buttons on a card. */
  bindSchemaDiagnosticButtons(container: HTMLElement): void;
  /** Executes a SQL query from a Play or Refresh button. */
  executeDirectSqlAction(
    queryToPlay: string,
    triggerBtn: HTMLButtonElement,
    messageToUpdate?: Message,
  ): Promise<void>;
  /** Exports rows to CSV. */
  exportRowsToCsv(
    rows: Array<Record<string, ExportableCellValue>>,
    filename: string,
  ): void;
  /** Exports rows to JSON. */
  exportRowsToJson(
    rows: Array<Record<string, ExportableCellValue>>,
    filename: string,
  ): void;
  /** Announces a message to screen readers. */
  announce(message: string, isAssertive?: boolean): void;
}

/**
 * Renders chat message lists, empty-state query chips, SQL action blocks, and result tables.
 */
export class MessageRenderer {
  /** Host UI reference. */
  private readonly ui: MessageRendererHost;

  /**
   * Creates a new `MessageRenderer`.
   * @param {MessageRendererHost} ui - Host UI instance.
   */
  constructor(ui: MessageRendererHost) {
    this.ui = ui;
  }

  /**
   * Renders the messages of the active chat into `messagesContainer`.
   * @returns {void}
   */
  public renderMessages(): void {
    const activeChat = this.ui.state.getActiveChat();
    if (!activeChat) return;

    this.ui.messagesContainer.replaceChildren();

    if (activeChat.messages.length === 0) {
      const emptyStateDiv = document.createElement("div");
      emptyStateDiv.className = "empty-state";
      emptyStateDiv.setAttribute("role", "status");
      emptyStateDiv.setAttribute("aria-live", "polite");

      if (
        this.ui.state.systemStatus.dbStatusCode === "empty_db" ||
        this.ui.state.systemStatus.dbStatusCode === "missing_initial_data" ||
        this.ui.state.systemStatus.dbStatusCode === "missing_file"
      ) {
        const dbNotice = createSchemaDiagnosticCard({
          icon: "ℹ️",
          title: i18next.t("status.emptyDbPrompt"),
          command: "t1d-analytics load",
        });
        dbNotice.style.marginBottom = "1rem";
        this.ui.bindSchemaDiagnosticButtons(dbNotice);
        emptyStateDiv.appendChild(dbNotice);
      }

      const noMessagesP = document.createElement("p");
      noMessagesP.style.marginBottom = "1rem";
      noMessagesP.textContent = i18next.t("ui.noMessagesYet");
      emptyStateDiv.appendChild(noMessagesP);

      const chipsContainer = document.createElement("div");
      chipsContainer.className = "chips-container";

      const exampleQueries: Array<{
        label: string;
        query: string;
        mode: "sql" | "gemma4";
      }> = [
        {
          label: i18next.t("ui.exampleQueries.patientDemographics"),
          query: `${i18next.t("ui.exampleQueries.patientDemographicsComment", "-- Patient Demographics & Treatment Breakdown")}

SELECT 
    TxGroup, 
    COUNT(*) as total_patients, 
    ROUND(AVG(AgeAsOfRandDt), 1) as avg_age,
    SUM(NumSevHypo) as total_severe_hypo_events
FROM tblaptsummary 
GROUP BY TxGroup;`,
          mode: "sql",
        },
        {
          label: i18next.t("ui.exampleQueries.adverseEvents"),
          query: `${i18next.t("ui.exampleQueries.adverseEventsComment", "-- Frequency of Adverse Events")}

SELECT 
    event as adverse_event_type, 
    COUNT(*) as occurrence_count 
FROM adverseevents 
GROUP BY event 
ORDER BY occurrence_count DESC;`,
          mode: "sql",
        },
        {
          label: i18next.t("ui.exampleQueries.pumpManufacturers"),
          query: `${i18next.t("ui.exampleQueries.pumpManufacturersComment", "-- Patient Pump Manufacturers")}

SELECT 
    Pt_PumpManuf as pump_manufacturer, 
    COUNT(*) as user_count 
FROM subjects 
WHERE Pt_PumpManuf IS NOT NULL 
GROUP BY Pt_PumpManuf 
ORDER BY user_count DESC;`,
          mode: "sql",
        },
        {
          label: i18next.t("ui.exampleQueries.hba1cDemographics"),
          query: `${i18next.t("ui.exampleQueries.hba1cDemographicsComment", "-- Joining Subject Summaries with HbA1c Lab Results")}

SELECT 
    t.TxGroup, 
    t.Gender,
    ROUND(AVG(h.HbA1c), 2) as average_hba1c,
    COUNT(h.HbA1c) as total_tests_run
FROM tblaptsummary t
JOIN hba1c h ON t.PtID = h.PtID
WHERE h.HbA1c IS NOT NULL
GROUP BY t.TxGroup, t.Gender
ORDER BY t.TxGroup, average_hba1c DESC;`,
          mode: "sql",
        },
        {
          label: i18next.t("ui.exampleQueries.nlpFirst5"),
          query: i18next.t(
            "ui.exampleQueries.nlpFirst5Prompt",
            "Show me the first 5 patients in the demographics table.",
          ),
          mode: "gemma4",
        },
      ];

      exampleQueries.forEach((item) => {
        const chip = document.createElement("button");
        chip.className = "chip";
        chip.textContent = item.label;
        chip.addEventListener("click", () => {
          this.ui.chatInput.value = item.query;
          const hasChipModeOption = Array.from(
            this.ui.modelSelect.options,
          ).some((opt) => opt.value === item.mode);
          const targetChipMode = hasChipModeOption ? item.mode : "sql";
          this.ui.modelSelect.value = targetChipMode;
          this.ui.state.setActiveChatModel(targetChipMode);
          this.ui.chatInput.focus();
          this.ui.syncHighlight();
        });
        chipsContainer.appendChild(chip);
      });

      emptyStateDiv.appendChild(chipsContainer);
      this.ui.messagesContainer.appendChild(emptyStateDiv);
      return;
    }

    activeChat.messages.forEach((msg) => {
      const msgDiv = document.createElement("div");
      msgDiv.className = `message ${msg.role}`;

      if (msg.model) {
        const header = document.createElement("div");
        header.className = "message-header";
        const badge = document.createElement("span");
        badge.className = "model-badge";
        badge.textContent =
          msg.model === "sql"
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
        (msg.model === "sql" || (!msg.model && activeChat.model === "sql"));

      const createCopyBtn = (textToCopy: string): HTMLButtonElement => {
        const btn = document.createElement("button");
        btn.className = "copy-sql-btn icon-btn";
        btn.setAttribute(
          "aria-label",
          i18next.t("aria.copyQuery", "Copy Query"),
        );
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
      };

      const createPlayBtn = (queryToPlay: string): HTMLButtonElement => {
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
          void this.ui.executeDirectSqlAction(queryToPlay, btn);
        });
        return btn;
      };

      const createRefreshBtn = (
        queryToPlay: string,
        messageToUpdate: Message,
      ): HTMLButtonElement => {
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
          void this.ui.executeDirectSqlAction(
            queryToPlay,
            btn,
            messageToUpdate,
          );
        });
        return btn;
      };

      if (isSqlMessage) {
        contentDiv.className = "sql-user-query-wrapper";

        const headerDiv = document.createElement("div");
        headerDiv.className = "sql-query-header";
        headerDiv.appendChild(createCopyBtn(msg.content));
        headerDiv.appendChild(createPlayBtn(msg.content));

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

        contentDiv.appendChild(headerDiv);
        contentDiv.appendChild(preBlock);
      } else {
        const rawHtml = String(marked.parse(msg.content ?? ""));
        contentDiv.innerHTML = DOMPurify.sanitize(rawHtml, {
          ADD_ATTR: ["target"],
        });
        contentDiv
          .querySelectorAll<HTMLElement>("pre")
          .forEach((preElement) => {
            preElement
              .querySelectorAll<HTMLElement>("code")
              .forEach((block) => {
                hljs.highlightElement(block);

                if (block.className.includes("language-sql")) {
                  const wrapper = document.createElement("div");
                  wrapper.className = "sql-query-container";

                  const headerDiv = document.createElement("div");
                  headerDiv.className = "sql-query-header";
                  const queryStr = String(block.textContent);
                  headerDiv.appendChild(createCopyBtn(queryStr));
                  headerDiv.appendChild(createPlayBtn(queryStr));

                  preElement.replaceWith(wrapper);
                  wrapper.appendChild(headerDiv);
                  wrapper.appendChild(preElement);
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

        headerDiv.appendChild(createCopyBtn(msg.sqlQuery));

        if (msg.role === "assistant" && msg.model === "sql") {
          headerDiv.appendChild(createRefreshBtn(msg.sqlQuery, msg));
        } else {
          headerDiv.appendChild(createPlayBtn(msg.sqlQuery));
        }

        queryContainer.appendChild(headerDiv);
        queryContainer.appendChild(preBlock);
        msgDiv.appendChild(queryContainer);
      }
      const sqlRows = msg.sqlResult;
      if (sqlRows) {
        if (sqlRows.length > 0) {
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
            this.ui.exportRowsToCsv(sqlRows, "query_results.csv"),
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
            this.ui.exportRowsToJson(sqlRows, "query_results.json"),
          );

          toolbar.appendChild(csvBtn);
          toolbar.appendChild(jsonBtn);
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

          const cgmCard = renderCgmCard(sqlRows);
          if (cgmCard) {
            tableContainer.appendChild(cgmCard);
          }

          tableContainer.appendChild(table);
          msgDiv.appendChild(tableContainer);
        } else {
          const emptyDiv = document.createElement("div");
          emptyDiv.className = "sql-empty";
          emptyDiv.textContent = i18next.t("ui.noRows");
          msgDiv.appendChild(emptyDiv);
        }
      }

      this.ui.messagesContainer.appendChild(msgDiv);
    });
  }
}
