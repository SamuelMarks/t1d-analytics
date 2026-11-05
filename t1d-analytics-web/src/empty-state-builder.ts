/**
 * @file empty-state-builder.ts
 * @module empty-state-builder
 * Builds the empty-state chat view, starter clinical SQL query chips, and Academic Table 1 chips.
 */

import i18next from "i18next";
import { DEFAULT_CHAT_MODEL, RAW_SQL_MODEL, type ChatState } from "./state";
import { TABLE_1_EXAMPLE_QUERIES } from "./table1-queries";

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
 * Host contract required by `buildEmptyStateView`.
 */
export interface EmptyStateHost {
  /** Global application state store. */
  state: ChatState;
  /** Textarea element where the user types queries or SQL. */
  chatInput: HTMLTextAreaElement;
  /** Dropdown select element for choosing the active LLM or SQL engine. */
  modelSelect: HTMLSelectElement;
  /** Binds interaction events to action buttons inside diagnostic cards. */
  bindSchemaDiagnosticButtons(container: HTMLElement): void;
  /** Synchronizes syntax highlighting in the chat input. */
  syncHighlight(): void;
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
 * Specification for a starter example query chip in the empty-state view.
 */
export interface StarterQueryItem {
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
  const demoSql = `SELECT 
    TxGroup, 
    COUNT(*) as total_patients, 
    ROUND(AVG(AgeAsOfRandDt), 1) as avg_age,
    SUM(NumSevHypo) as total_severe_hypo_events
FROM tblaptsummary 
GROUP BY TxGroup;`;
  const adverseSql = `SELECT 
    event as adverse_event_type, 
    COUNT(*) as occurrence_count 
FROM adverseevents 
GROUP BY event 
ORDER BY occurrence_count DESC;`;
  const pumpSql = `SELECT 
    Pt_PumpManuf as pump_manufacturer, 
    COUNT(*) as user_count 
FROM subjects 
WHERE Pt_PumpManuf IS NOT NULL 
GROUP BY Pt_PumpManuf 
ORDER BY user_count DESC;`;
  const hba1cSql = `SELECT 
    t.TxGroup, 
    t.Gender,
    ROUND(AVG(h.HbA1c), 2) as average_hba1c,
    COUNT(h.HbA1c) as total_tests_run
FROM tblaptsummary t
JOIN hba1c h ON t.PtID = h.PtID
WHERE h.HbA1c IS NOT NULL
GROUP BY t.TxGroup, t.Gender
ORDER BY t.TxGroup, average_hba1c DESC;`;
  const nlpFirst5Prompt = i18next.t(
    "ui.exampleQueries.nlpFirst5Prompt",
    "Show me the first 5 patients in the demographics table.",
  );

  return [
    {
      label: i18next.t("ui.exampleQueries.patientDemographics"),
      query: `${i18next.t("ui.exampleQueries.patientDemographicsComment", "-- Patient Demographics & Treatment Breakdown")}

${demoSql}`,
      mode: RAW_SQL_MODEL,
      question:
        "Show patient count, average age, and total severe hypoglycemia events grouped by treatment group.",
      goldSql: demoSql,
    },
    {
      label: i18next.t("ui.exampleQueries.adverseEvents"),
      query: `${i18next.t("ui.exampleQueries.adverseEventsComment", "-- Frequency of Adverse Events")}

${adverseSql}`,
      mode: RAW_SQL_MODEL,
      question:
        "Show the frequency of each adverse event type ordered from highest to lowest occurrence count.",
      goldSql: adverseSql,
    },
    {
      label: i18next.t("ui.exampleQueries.pumpManufacturers"),
      query: `${i18next.t("ui.exampleQueries.pumpManufacturersComment", "-- Patient Pump Manufacturers")}

${pumpSql}`,
      mode: RAW_SQL_MODEL,
      question:
        "Count the number of subjects using each insulin pump manufacturer.",
      goldSql: pumpSql,
    },
    {
      label: i18next.t("ui.exampleQueries.hba1cDemographics"),
      query: `${i18next.t("ui.exampleQueries.hba1cDemographicsComment", "-- Joining Subject Summaries with HbA1c Lab Results")}

${hba1cSql}`,
      mode: RAW_SQL_MODEL,
      question:
        "Join subject summaries with HbA1c lab results and show average HbA1c and total tests by treatment group and gender.",
      goldSql: hba1cSql,
    },
    {
      label: i18next.t("ui.exampleQueries.nlpFirst5"),
      query: nlpFirst5Prompt,
      mode: DEFAULT_CHAT_MODEL,
      question: nlpFirst5Prompt,
      goldSql: "SELECT * FROM tblaptsummary LIMIT 5;",
    },
  ];
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
    (opt) => opt.value && opt.value !== RAW_SQL_MODEL,
  );
  return firstLlmOption ? firstLlmOption.value : RAW_SQL_MODEL;
}

/**
 * Builds the empty-state element with optional database notice, starter chips, and Academic Table 1 chips.
 * @param {EmptyStateHost} ui - Host UI instance.
 * @returns {HTMLDivElement} Constructed `.empty-state` element.
 */
export function buildEmptyStateView(ui: EmptyStateHost): HTMLDivElement {
  const emptyStateDiv = document.createElement("div");
  emptyStateDiv.className = "empty-state";
  emptyStateDiv.setAttribute("role", "status");
  emptyStateDiv.setAttribute("aria-live", "polite");

  if (
    ui.state.systemStatus.dbStatusCode === "empty_db" ||
    ui.state.systemStatus.dbStatusCode === "missing_initial_data" ||
    ui.state.systemStatus.dbStatusCode === "missing_file"
  ) {
    const dbNotice = createSchemaDiagnosticCard({
      icon: "ℹ️",
      title: i18next.t("status.emptyDbPrompt"),
      command: "t1d-analytics load",
    });
    dbNotice.style.marginBottom = "1rem";
    ui.bindSchemaDiagnosticButtons(dbNotice);
    emptyStateDiv.appendChild(dbNotice);
  }

  const noMessagesP = document.createElement("p");
  noMessagesP.style.marginBottom = "1rem";
  noMessagesP.textContent = i18next.t("ui.noMessagesYet");
  emptyStateDiv.appendChild(noMessagesP);

  const chipsContainer = document.createElement("div");
  chipsContainer.className = "chips-container";

  buildStarterQueryItems().forEach((item) => {
    const chip = document.createElement("button");
    chip.className = "chip";
    chip.textContent = item.label;
    chip.addEventListener("click", () => {
      ui.chatInput.value = item.query;
      const hasChipModeOption = Array.from(ui.modelSelect.options).some(
        (opt) => opt.value === item.mode,
      );
      const targetChipMode = hasChipModeOption ? item.mode : RAW_SQL_MODEL;
      ui.modelSelect.value = targetChipMode;
      ui.state.setActiveChatModel(targetChipMode);
      ui.state.setActiveGoldSql(
        targetChipMode === RAW_SQL_MODEL ? null : item.goldSql,
      );
      ui.chatInput.focus();
      ui.syncHighlight();
    });
    chipsContainer.appendChild(chip);
  });

  emptyStateDiv.appendChild(chipsContainer);

  const table1HeaderRow = document.createElement("div");
  table1HeaderRow.className = "table1-header-row";

  const table1HeaderP = document.createElement("p");
  table1HeaderP.className = "table1-examples-heading";
  table1HeaderP.textContent = i18next.t("ui.exampleQueries.table1SectionTitle");

  let table1ChipActionMode: "sql" | "nl" = "sql";
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

  TABLE_1_EXAMPLE_QUERIES.forEach((spec) => {
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
        ui.chatInput.value = spec.question;
        const targetLlm = resolvePreferredLlmModel(ui.modelSelect);
        ui.modelSelect.value = targetLlm;
        ui.state.setActiveChatModel(targetLlm);
        ui.state.setActiveGoldSql(spec.sql);
      } else {
        ui.chatInput.value = `-- ${spec.question}
${spec.sql}`;
        ui.modelSelect.value = RAW_SQL_MODEL;
        ui.state.setActiveChatModel(RAW_SQL_MODEL);
        ui.state.setActiveGoldSql(null);
      }
      ui.chatInput.focus();
      ui.syncHighlight();
    });
    table1Container.appendChild(t1Btn);
  });

  emptyStateDiv.appendChild(table1Container);
  return emptyStateDiv;
}
