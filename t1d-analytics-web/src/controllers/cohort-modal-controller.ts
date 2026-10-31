/**
 * @file cohort-modal-controller.ts
 * @module controllers/cohort-modal-controller
 * Composable sub-controller managing the Visual Cohort Query Builder modal,
 * table relationship join resolution, and live DuckDB SQL generation.
 */

import i18next from "i18next";
import {
  buildCohortSql,
  normalizeJoinType,
  resolveTableJoin,
} from "../cohort-sql";
import { requireElementById, trapModalFocus } from "../modal-utils";
import { type ChatState } from "../state";

/**
 * Host contract required by `CohortBuilderModalController`.
 */
export interface CohortBuilderHost {
  /** Global application state store. */
  state: ChatState;
  /** Textarea element where the user types queries or SQL. */
  chatInput: HTMLTextAreaElement;
  /** Dropdown select element for choosing the active LLM or SQL engine. */
  modelSelect: HTMLSelectElement;
  /** Synchronizes syntax highlighting in the chat input. */
  syncHighlight(): void;
  /** Sends a message or executes a SQL query directly. */
  handleSendMessage(text?: string, modelOverride?: string): Promise<void>;
}

/**
 * Controller managing the Visual Cohort Query Builder modal lifecycle and SQL preview.
 */
export class CohortBuilderModalController {
  /** Host UI reference. */
  private readonly ui: CohortBuilderHost;
  /** Header button to open the Visual Cohort Query Builder modal. */
  public readonly cohortFilterBtn: HTMLButtonElement;
  /** Modal dialog container for the Visual Cohort Query Builder. */
  public readonly cohortModal: HTMLElement;
  /** Button to close the Visual Cohort Query Builder modal. */
  public readonly closeCohortModalBtn: HTMLButtonElement;
  /** Select dropdown for the primary cohort table. */
  public readonly cohortTableSelect: HTMLSelectElement;
  /** Select dropdown for the optional joined secondary table. */
  public readonly cohortJoinTableSelect: HTMLSelectElement;
  /** Select dropdown for the SQL join type (INNER JOIN or LEFT JOIN). */
  public readonly cohortJoinTypeSelect: HTMLSelectElement;
  /** Input field for the SQL join ON condition. */
  public readonly cohortJoinOnInput: HTMLInputElement;
  /** Number input for minimum patient age filter. */
  public readonly cohortMinAge: HTMLInputElement;
  /** Number input for maximum patient age filter. */
  public readonly cohortMaxAge: HTMLInputElement;
  /** Select dropdown for biological sex/gender filter. */
  public readonly cohortGenderSelect: HTMLSelectElement;
  /** Text input for clinical treatment group filter. */
  public readonly cohortTxGroup: HTMLInputElement;
  /** Number input for maximum Time-in-Range (TIR) threshold. */
  public readonly cohortMaxTIR: HTMLInputElement;
  /** Number input for minimum HbA1c threshold. */
  public readonly cohortMinHbA1c: HTMLInputElement;
  /** Preformatted text element showing the live generated DuckDB SQL. */
  public readonly cohortSqlPreview: HTMLElement;
  /** Button to insert the generated cohort SQL into the chat input. */
  public readonly cohortInsertChatBtn: HTMLButtonElement;
  /** Button to immediately execute the generated cohort SQL. */
  public readonly cohortExecuteBtn: HTMLButtonElement;
  /** Element that held focus prior to opening the cohort modal. */
  private previousFocusCohort: Element | null = null;
  /** Keydown listener for focus trapping inside the cohort modal. */
  private onCohortModalKeyDown: ((e: KeyboardEvent) => void) | null = null;

  /**
   * Creates a new `CohortBuilderModalController`.
   * @param {CohortBuilderHost} ui - Host UI instance.
   */
  constructor(ui: CohortBuilderHost) {
    this.ui = ui;
    this.cohortFilterBtn =
      requireElementById<HTMLButtonElement>("cohort-filter-btn");
    this.cohortModal = requireElementById<HTMLElement>("cohort-modal");
    this.closeCohortModalBtn = requireElementById<HTMLButtonElement>(
      "close-cohort-modal-btn",
    );
    this.cohortTableSelect = requireElementById<HTMLSelectElement>(
      "cohort-table-select",
    );
    this.cohortJoinTableSelect = requireElementById<HTMLSelectElement>(
      "cohort-join-table-select",
    );
    this.cohortJoinTypeSelect = requireElementById<HTMLSelectElement>(
      "cohort-join-type-select",
    );
    this.cohortJoinOnInput = requireElementById<HTMLInputElement>(
      "cohort-join-on-input",
    );
    this.cohortMinAge = requireElementById<HTMLInputElement>("cohort-min-age");
    this.cohortMaxAge = requireElementById<HTMLInputElement>("cohort-max-age");
    this.cohortGenderSelect = requireElementById<HTMLSelectElement>(
      "cohort-gender-select",
    );
    this.cohortTxGroup = requireElementById<HTMLInputElement>("cohort-txgroup");
    this.cohortMaxTIR = requireElementById<HTMLInputElement>("cohort-max-tir");
    this.cohortMinHbA1c =
      requireElementById<HTMLInputElement>("cohort-min-hba1c");
    this.cohortSqlPreview =
      requireElementById<HTMLElement>("cohort-sql-preview");
    this.cohortInsertChatBtn = requireElementById<HTMLButtonElement>(
      "cohort-insert-chat-btn",
    );
    this.cohortExecuteBtn =
      requireElementById<HTMLButtonElement>("cohort-execute-btn");

    this.bindEvents();
  }

  /**
   * Binds DOM event listeners for the Visual Cohort Query Builder modal.
   * @returns {void}
   */
  private bindEvents(): void {
    this.cohortFilterBtn.addEventListener("click", () => {
      this.open();
    });

    this.closeCohortModalBtn.addEventListener("click", () => {
      this.close();
    });

    this.cohortModal.addEventListener("click", (e) => {
      if (e.target === this.cohortModal) {
        this.close();
      }
    });

    this.cohortTableSelect.addEventListener("change", () => {
      this.handleJoinTableChange();
      this.updateSqlPreview();
    });

    this.cohortJoinTableSelect.addEventListener("change", () => {
      this.handleJoinTableChange();
      this.updateSqlPreview();
    });

    this.cohortJoinTypeSelect.addEventListener("change", () => {
      this.updateSqlPreview();
    });

    this.cohortJoinOnInput.addEventListener("input", () => {
      this.updateSqlPreview();
    });

    const cohortInputs = [
      this.cohortMinAge,
      this.cohortMaxAge,
      this.cohortGenderSelect,
      this.cohortTxGroup,
      this.cohortMaxTIR,
      this.cohortMinHbA1c,
    ];
    for (const inp of cohortInputs) {
      inp.addEventListener("input", () => this.updateSqlPreview());
      inp.addEventListener("change", () => this.updateSqlPreview());
    }

    this.cohortInsertChatBtn.addEventListener("click", () => {
      const sql = this.getCohortSql();
      this.ui.chatInput.value = sql;
      this.ui.syncHighlight();
      this.close();
      this.ui.chatInput.focus();
    });

    this.cohortExecuteBtn.addEventListener("click", () => {
      const sql = this.getCohortSql();
      this.close();
      this.ui.modelSelect.value = "sql";
      this.ui.state.setActiveChatModel("sql");
      void this.ui.handleSendMessage(sql, "sql");
    });
  }

  /**
   * Generates the cohort filter DuckDB SQL query from active modal inputs.
   * @returns {string} The generated DuckDB SQL statement.
   */
  public getCohortSql(): string {
    const minAgeStr = this.cohortMinAge.value.trim();
    const maxAgeStr = this.cohortMaxAge.value.trim();
    const maxTirStr = this.cohortMaxTIR.value.trim();
    const minHba1cStr = this.cohortMinHbA1c.value.trim();

    return buildCohortSql({
      tableName: this.cohortTableSelect.value || "patients",
      joinTableName: this.cohortJoinTableSelect.value || undefined,
      joinType: normalizeJoinType(this.cohortJoinTypeSelect.value),
      joinCondition: this.cohortJoinOnInput.value.trim() || undefined,
      minAge: minAgeStr !== "" ? Number(minAgeStr) : undefined,
      maxAge: maxAgeStr !== "" ? Number(maxAgeStr) : undefined,
      gender: this.cohortGenderSelect.value || undefined,
      txGroup: this.cohortTxGroup.value.trim() || undefined,
      maxTIR: maxTirStr !== "" ? Number(maxTirStr) : undefined,
      minHbA1c: minHba1cStr !== "" ? Number(minHba1cStr) : undefined,
    });
  }

  /**
   * Updates join condition and types when target or secondary tables change.
   * @returns {void}
   */
  public handleJoinTableChange(): void {
    const primaryTable = this.cohortTableSelect.value || "patients";
    const joinTable = this.cohortJoinTableSelect.value || "";

    if (!joinTable || joinTable === primaryTable) {
      this.cohortJoinOnInput.value = "";
      return;
    }

    const pNode = document
      .querySelector(`.schema-table-name[data-table="${primaryTable}"]`)
      ?.closest(".schema-table");
    const jNode = document
      .querySelector(`.schema-table-name[data-table="${joinTable}"]`)
      ?.closest(".schema-table");
    const pCols = pNode
      ? Array.from(pNode.querySelectorAll(".col-name")).map((el) =>
          String(el.textContent).trim(),
        )
      : undefined;
    const jCols = jNode
      ? Array.from(jNode.querySelectorAll(".col-name")).map((el) =>
          String(el.textContent).trim(),
        )
      : undefined;

    const resolved = resolveTableJoin(primaryTable, joinTable, pCols, jCols);
    const escPrimary = primaryTable.replace(/"/g, '""');
    const escJoin = joinTable.replace(/"/g, '""');
    if (resolved) {
      this.cohortJoinTypeSelect.value = resolved.defaultJoinType;
      this.cohortJoinOnInput.value = `"${escPrimary}".${resolved.fromColumn} = "${escJoin}".${resolved.toColumn}`;
    } else {
      this.cohortJoinOnInput.value = `"${escPrimary}".patient_id = "${escJoin}".patient_id`;
    }
  }

  /**
   * Updates the live DuckDB SQL preview in the cohort modal.
   * @returns {void}
   */
  public updateSqlPreview(): void {
    this.cohortSqlPreview.textContent = this.getCohortSql();
  }

  /**
   * Opens the Visual Cohort Query Builder modal and populates available clinical tables.
   * @returns {void}
   */
  public open(): void {
    if (this.cohortModal.classList.contains("hidden")) {
      this.previousFocusCohort = document.activeElement;
    }
    const schemaTables = Array.from(
      document.querySelectorAll(".schema-table-name"),
    )
      .map((el) => el.getAttribute("data-table"))
      .filter((t): t is string => Boolean(t));

    const tables =
      schemaTables.length > 0
        ? schemaTables
        : ["patients", "cgm_readings", "tblaptsummary", "subjects"];

    this.cohortTableSelect.replaceChildren();
    for (const tbl of tables) {
      const opt = document.createElement("option");
      opt.value = tbl;
      opt.textContent = tbl;
      this.cohortTableSelect.appendChild(opt);
    }

    const noneOpt = document.createElement("option");
    noneOpt.value = "";
    noneOpt.setAttribute("data-i18n", "cohort.noneSingleTable");
    noneOpt.textContent = i18next.t(
      "cohort.noneSingleTable",
      "None (Single Table)",
    );
    this.cohortJoinTableSelect.replaceChildren(noneOpt);
    for (const tbl of tables) {
      const opt = document.createElement("option");
      opt.value = tbl;
      opt.textContent = tbl;
      this.cohortJoinTableSelect.appendChild(opt);
    }

    this.handleJoinTableChange();
    this.updateSqlPreview();
    this.cohortModal.classList.remove("hidden");
    this.cohortModal.setAttribute("aria-hidden", "false");

    if (this.onCohortModalKeyDown) {
      this.cohortModal.removeEventListener(
        "keydown",
        this.onCohortModalKeyDown,
      );
    }
    this.onCohortModalKeyDown = (e: KeyboardEvent): void => {
      trapModalFocus(e, this.cohortModal, () => this.close());
    };
    this.cohortModal.addEventListener("keydown", this.onCohortModalKeyDown);

    this.closeCohortModalBtn.focus();
  }

  /**
   * Closes the Visual Cohort Query Builder modal.
   * @returns {void}
   */
  public close(): void {
    this.cohortModal.classList.add("hidden");
    this.cohortModal.setAttribute("aria-hidden", "true");
    if (this.onCohortModalKeyDown) {
      this.cohortModal.removeEventListener(
        "keydown",
        this.onCohortModalKeyDown,
      );
      this.onCohortModalKeyDown = null;
    }
    if (this.previousFocusCohort instanceof HTMLElement) {
      this.previousFocusCohort.focus();
    }
  }

  /**
   * Removes any active modal keydown listeners when disposing the UI.
   * @returns {void}
   */
  public dispose(): void {
    if (this.onCohortModalKeyDown) {
      this.cohortModal.removeEventListener(
        "keydown",
        this.onCohortModalKeyDown,
      );
      this.onCohortModalKeyDown = null;
    }
  }
}
