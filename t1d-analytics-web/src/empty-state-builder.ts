/**
 * @file empty-state-builder.ts
 * @module empty-state-builder
 * Builds the empty-state chat view, starter clinical SQL query chips, and Academic Table 1 chips,
 * emitting typed `onSelectTemplate` callbacks without mutating coordinator DOM inputs directly.
 */

import i18next from "i18next";
import { type TemplateSelectionPayload } from "./event-bus";
import { DEFAULT_CHAT_MODEL, RAW_SQL_MODEL, type ChatState } from "./state";
import {
  getStarterTemplateExamples,
  getTable1ExampleQueries,
  type Table1ExampleSpec,
} from "./table1-queries";

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
 * Narrow dependency options passed to `buildEmptyStateView`.
 */
export interface EmptyStateViewOptions {
  /** Read-only access to application state (`systemStatus.dbStatusCode`). */
  state: ChatState;
  /** Model selector dropdown inspected (read-only) to resolve available LLM options. */
  modelSelect: HTMLSelectElement;
  /** Typed callback emitted when a starter query chip or Academic Table 1 chip is clicked. */
  onSelectTemplate: (selection: TemplateSelectionPayload) => void;
  /** Callback binding copy/reload buttons inside empty-database diagnostic cards. */
  onBindDiagnosticButtons: (container: HTMLElement) => void;
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
  card.className =
    options.isError === true
      ? "schema-diagnostic-card error-text"
      : "schema-diagnostic-card";
  if (options.isAlert === true || options.isError === true) {
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

  if (options.showReload === true) {
    const reloadBtn = document.createElement("button");
    reloadBtn.className = "btn btn-xs btn-primary reload-schema-btn";
    reloadBtn.textContent = i18next.t("status.reloadSchema");
    card.appendChild(reloadBtn);
  }

  return card;
}

/**
 * Specification for a starter example query chip in the empty-state view.
 */
export interface StarterQueryItem {
  /** Stable identifier for the starter template ('starter-1' through 'starter-5'). */
  id: string;
  /** Complexity rating label ('Low', 'Low-Medium', 'Medium', 'Medium-High', 'High', 'Very High'). */
  complexity: Table1ExampleSpec["complexity"];
  /** Localized display label for the starter chip button. */
  label: string;
  /** SQL or natural-language query populated into the input box when clicked. */
  query: string;
  /** Target model mode ('sql' or 'gemma:2b'). */
  mode: string;
  /** Natural-language clinical question associated with this starter template. */
  question: string;
  /** Canonical Gold-Standard SQL query for this starter template. */
  goldSql: string;
}

/**
 * Builds the starter example query specifications for the empty-state view.
 * @returns {StarterQueryItem[]} Starter query items.
 */
export function buildStarterQueryItems(): StarterQueryItem[] {
  const nlpFirst5Prompt = i18next.t(
    "ui.exampleQueries.nlpFirst5Prompt",
    "Show me the first 5 patients in the demographics table.",
  );

  return getStarterTemplateExamples().map((spec) => {
    const isNlpStarter = spec.defaultMode !== "sql";
    const localizedQuestion =
      spec.id === "starter-5" ? nlpFirst5Prompt : spec.question;
    const headerComment =
      spec.commentI18nKey && spec.defaultComment
        ? i18next.t(spec.commentI18nKey, spec.defaultComment)
        : "";
    const queryText = isNlpStarter
      ? localizedQuestion
      : `${headerComment}

${spec.goldSql}`;
    return {
      id: spec.id,
      complexity: spec.complexity,
      label: i18next.t(spec.i18nKey),
      query: queryText,
      mode: isNlpStarter ? DEFAULT_CHAT_MODEL : RAW_SQL_MODEL,
      question: localizedQuestion,
      goldSql: spec.goldSql,
    };
  });
}

/**
 * Resolves the first non-SQL LLM model option available in `#model-select`, or falls back to `DEFAULT_CHAT_MODEL`.
 * @param {HTMLSelectElement} modelSelect - Model selector dropdown element.
 * @returns {string} Resolved LLM model identifier.
 */
export function resolvePreferredLlmModel(
  modelSelect: HTMLSelectElement,
): string {
  if (modelSelect.value && modelSelect.value !== RAW_SQL_MODEL) {
    return modelSelect.value;
  }
  const hasDefaultOption = Array.from(modelSelect.options).some(
    (opt) => opt.value === DEFAULT_CHAT_MODEL,
  );
  if (hasDefaultOption) {
    return DEFAULT_CHAT_MODEL;
  }
  const firstLlmOption = Array.from(modelSelect.options).find(
    (opt) => Boolean(opt.value) && opt.value !== RAW_SQL_MODEL,
  );
  return firstLlmOption ? firstLlmOption.value : RAW_SQL_MODEL;
}

/**
 * Builds the empty-state element with optional database notice, starter chips, and Academic Table 1 chips.
 * @param {EmptyStateViewOptions} options - Narrow state and callback dependencies.
 * @returns {HTMLDivElement} Constructed `.empty-state` element.
 */
export function buildEmptyStateView(
  options: EmptyStateViewOptions,
): HTMLDivElement {
  const emptyStateDiv = document.createElement("div");
  emptyStateDiv.className = "empty-state";
  emptyStateDiv.setAttribute("role", "status");
  emptyStateDiv.setAttribute("aria-live", "polite");

  if (
    options.state.systemStatus.dbStatusCode === "empty_db" ||
    options.state.systemStatus.dbStatusCode === "missing_initial_data" ||
    options.state.systemStatus.dbStatusCode === "missing_file"
  ) {
    const dbNotice = createSchemaDiagnosticCard({
      icon: "ℹ️",
      title: i18next.t("status.emptyDbPrompt"),
      command: "t1d-analytics load",
    });
    dbNotice.style.marginBottom = "1rem";
    options.onBindDiagnosticButtons(dbNotice);
    emptyStateDiv.appendChild(dbNotice);
  }

  const noMessagesP = document.createElement("p");
  noMessagesP.style.marginBottom = "1rem";
  noMessagesP.textContent = i18next.t("ui.noMessagesYet");
  emptyStateDiv.appendChild(noMessagesP);

  let table1ChipActionMode: "sql" | "nl" = "sql";

  const chipsContainer = document.createElement("div");
  chipsContainer.className = "chips-container";

  buildStarterQueryItems().forEach((item) => {
    const chip = document.createElement("button");
    chip.className = "chip";
    chip.setAttribute("data-starter-id", item.id);
    chip.textContent = item.label;
    chip.addEventListener("click", () => {
      if (table1ChipActionMode === "nl") {
        options.onSelectTemplate({
          promptText: item.question,
          targetModel: resolvePreferredLlmModel(options.modelSelect),
          goldSql: item.goldSql,
        });
      } else {
        const hasChipModeOption = Array.from(options.modelSelect.options).some(
          (opt) => opt.value === item.mode,
        );
        const targetChipMode = hasChipModeOption ? item.mode : RAW_SQL_MODEL;
        options.onSelectTemplate({
          promptText: item.query,
          targetModel: targetChipMode,
          goldSql: targetChipMode === RAW_SQL_MODEL ? null : item.goldSql,
        });
      }
    });
    chipsContainer.appendChild(chip);
  });

  emptyStateDiv.appendChild(chipsContainer);

  const table1HeaderRow = document.createElement("div");
  table1HeaderRow.className = "table1-header-row";

  const table1HeaderP = document.createElement("p");
  table1HeaderP.className = "table1-examples-heading";
  table1HeaderP.textContent = i18next.t("ui.exampleQueries.table1SectionTitle");

  const modePillGroup = document.createElement("div");
  modePillGroup.className = "template-mode-pill-group";
  modePillGroup.setAttribute("role", "group");
  modePillGroup.setAttribute(
    "aria-label",
    i18next.t("templates.chipModeAria", "Template chip action mode"),
  );

  const sqlModeBtn = document.createElement("button");
  sqlModeBtn.type = "button";
  sqlModeBtn.className = "template-mode-pill active";
  sqlModeBtn.setAttribute("data-chip-mode", "sql");
  sqlModeBtn.setAttribute("aria-pressed", "true");
  sqlModeBtn.textContent = i18next.t(
    "templates.modeGoldSql",
    "⚡ Insert Gold SQL",
  );

  const nlModeBtn = document.createElement("button");
  nlModeBtn.type = "button";
  nlModeBtn.className = "template-mode-pill";
  nlModeBtn.setAttribute("data-chip-mode", "nl");
  nlModeBtn.setAttribute("aria-pressed", "false");
  nlModeBtn.textContent = i18next.t(
    "templates.modeNaturalLanguage",
    "💬 Test NL -> SQL",
  );

  sqlModeBtn.addEventListener("click", () => {
    table1ChipActionMode = "sql";
    sqlModeBtn.classList.add("active");
    sqlModeBtn.setAttribute("aria-pressed", "true");
    nlModeBtn.classList.remove("active");
    nlModeBtn.setAttribute("aria-pressed", "false");
  });

  nlModeBtn.addEventListener("click", () => {
    table1ChipActionMode = "nl";
    nlModeBtn.classList.add("active");
    nlModeBtn.setAttribute("aria-pressed", "true");
    sqlModeBtn.classList.remove("active");
    sqlModeBtn.setAttribute("aria-pressed", "false");
  });

  modePillGroup.append(sqlModeBtn, nlModeBtn);
  table1HeaderRow.append(table1HeaderP, modePillGroup);
  emptyStateDiv.appendChild(table1HeaderRow);

  const table1Container = document.createElement("div");
  table1Container.className = "table1-chips-container chips-container";

  getTable1ExampleQueries().forEach((spec) => {
    const t1Btn = document.createElement("button");
    t1Btn.type = "button";
    t1Btn.className = "table1-chip";
    t1Btn.setAttribute("data-table1-id", spec.id);
    t1Btn.setAttribute("data-complexity", spec.complexity);
    const labelText = i18next.t(spec.i18nKey);

    const labelSpan = document.createElement("span");
    labelSpan.className = "table1-chip-label";
    labelSpan.textContent = labelText;

    const badgeSpan = document.createElement("span");
    badgeSpan.className = "complexity-badge";
    badgeSpan.textContent = spec.complexity;

    t1Btn.append(labelSpan, badgeSpan);
    t1Btn.title = spec.question;
    t1Btn.addEventListener("click", () => {
      if (table1ChipActionMode === "nl") {
        options.onSelectTemplate({
          promptText: spec.question,
          targetModel: resolvePreferredLlmModel(options.modelSelect),
          goldSql: spec.sql,
        });
      } else {
        options.onSelectTemplate({
          promptText: `-- ${spec.question}
${spec.sql}`,
          targetModel: RAW_SQL_MODEL,
          goldSql: null,
        });
      }
    });
    table1Container.appendChild(t1Btn);
  });

  emptyStateDiv.appendChild(table1Container);
  return emptyStateDiv;
}
