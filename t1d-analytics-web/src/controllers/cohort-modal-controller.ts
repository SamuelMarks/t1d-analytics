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
import { trapModalFocus } from "../modal-utils";

/**
 * Host contract required by `CohortBuilderModalController`.
 */
export interface CohortBuilderHost {
  /** Select dropdown for the primary cohort table. */
  cohortTableSelect: HTMLSelectElement | null;
  /** Select dropdown for the optional joined secondary table. */
  cohortJoinTableSelect: HTMLSelectElement | null;
  /** Select dropdown for the SQL join type (INNER JOIN or LEFT JOIN). */
  cohortJoinTypeSelect: HTMLSelectElement | null;
  /** Input field for the SQL join ON condition. */
  cohortJoinOnInput: HTMLInputElement | null;
  /** Number input for minimum patient age filter. */
  cohortMinAge: HTMLInputElement | null;
  /** Number input for maximum patient age filter. */
  cohortMaxAge: HTMLInputElement | null;
  /** Select dropdown for biological sex/gender filter. */
  cohortGenderSelect: HTMLSelectElement | null;
  /** Text input for clinical treatment group filter. */
  cohortTxGroup: HTMLInputElement | null;
  /** Number input for maximum Time-in-Range (TIR) threshold. */
  cohortMaxTIR: HTMLInputElement | null;
  /** Number input for minimum HbA1c threshold. */
  cohortMinHbA1c: HTMLInputElement | null;
  /** Preformatted text element showing the live generated DuckDB SQL. */
  cohortSqlPreview: HTMLElement | null;
  /** Modal dialog container for the Visual Cohort Query Builder. */
  cohortModal: HTMLElement | null;
  /** Header button to open the Visual Cohort Query Builder modal. */
  cohortFilterBtn: HTMLButtonElement | null;
  /** Element that held focus prior to opening the cohort modal. */
  previousFocusCohort: Element | null;
  /** Keydown listener for focus trapping inside the cohort modal. */
  onCohortModalKeyDown: ((e: KeyboardEvent) => void) | null;
}

/**
 * Composable sub-controller for the Visual Cohort Builder Modal.
 */
export class CohortBuilderModalController {
  /** Host UI reference. */
  private readonly ui: CohortBuilderHost;

  /**
   * Creates a new `CohortBuilderModalController`.
   * @param {CohortBuilderHost} ui - Host UI instance.
   */
  constructor(ui: CohortBuilderHost) {
    this.ui = ui;
  }

  /**
   * Generates the cohort filter DuckDB SQL query from active modal inputs.
   * @returns {string} The generated DuckDB SQL statement.
   */
  public getCohortSql(): string {
    const tableName = this.ui.cohortTableSelect?.value || "";
    const joinTableName = this.ui.cohortJoinTableSelect?.value || undefined;
    const joinType = normalizeJoinType(this.ui.cohortJoinTypeSelect?.value);
    const joinCondition = this.ui.cohortJoinOnInput?.value?.trim() || undefined;
    const minAge = this.ui.cohortMinAge?.value
      ? parseFloat(this.ui.cohortMinAge.value)
      : undefined;
    const maxAge = this.ui.cohortMaxAge?.value
      ? parseFloat(this.ui.cohortMaxAge.value)
      : undefined;
    const gender = this.ui.cohortGenderSelect?.value || undefined;
    const txGroup = this.ui.cohortTxGroup?.value?.trim() || undefined;
    const maxTIR = this.ui.cohortMaxTIR?.value
      ? parseFloat(this.ui.cohortMaxTIR.value)
      : undefined;
    const minHbA1c = this.ui.cohortMinHbA1c?.value
      ? parseFloat(this.ui.cohortMinHbA1c.value)
      : undefined;

    return buildCohortSql({
      tableName,
      joinTableName,
      joinType,
      joinCondition,
      minAge,
      maxAge,
      gender,
      txGroup,
      maxTIR,
      minHbA1c,
    });
  }

  /**
   * Updates join condition and types when target or secondary tables change.
   * @returns {void}
   */
  public handleJoinTableChange(): void {
    const table1 = this.ui.cohortTableSelect?.value || "patients";
    const table2 = this.ui.cohortJoinTableSelect?.value;
    if (!table2 || table2 === table1) {
      if (this.ui.cohortJoinOnInput) this.ui.cohortJoinOnInput.value = "";
      return;
    }

    const extractTableCols = (tName: string): string[] => {
      const listEl = document.getElementById(`schema-table-${tName}`);
      return listEl
        ? Array.from(
            listEl.querySelectorAll(".schema-column span:first-child"),
            (s) => s.textContent!,
          )
        : [];
    };

    const rel = resolveTableJoin(
      table1,
      table2,
      extractTableCols(table1),
      extractTableCols(table2),
    );
    if (this.ui.cohortJoinOnInput) {
      this.ui.cohortJoinOnInput.value = rel
        ? `"${table1}".${rel.fromColumn} = "${table2}".${rel.toColumn}`
        : `"${table1}".patient_id = "${table2}".patient_id`;
    }
    if (rel && this.ui.cohortJoinTypeSelect) {
      this.ui.cohortJoinTypeSelect.value = rel.defaultJoinType;
    }
  }

  /**
   * Updates the live DuckDB SQL preview in the cohort modal.
   * @returns {void}
   */
  public updateSqlPreview(): void {
    if (this.ui.cohortSqlPreview) {
      this.ui.cohortSqlPreview.textContent = this.getCohortSql();
    }
  }

  /**
   * Opens the Visual Cohort Query Builder modal and populates available clinical tables.
   * @returns {void}
   */
  public open(): void {
    const cohortModal = this.ui.cohortModal;
    if (!cohortModal) return;
    const tableHeaders = document.querySelectorAll(
      ".schema-table-header-toggle span, .schema-table-header h4",
    );
    const foundTables: string[] = [];
    tableHeaders.forEach((h) => {
      const tName = h.textContent?.trim();
      if (tName) foundTables.push(tName);
    });
    const finalTables =
      foundTables.length > 0
        ? foundTables
        : ["demographics", "cgm_data", "patients"];

    if (this.ui.cohortTableSelect) {
      this.ui.cohortTableSelect.replaceChildren();
      for (const t of finalTables) {
        const opt = document.createElement("option");
        opt.value = t;
        opt.textContent = t;
        this.ui.cohortTableSelect.appendChild(opt);
      }
    }

    if (this.ui.cohortJoinTableSelect) {
      this.ui.cohortJoinTableSelect.replaceChildren();
      const noneOpt = document.createElement("option");
      noneOpt.value = "";
      noneOpt.setAttribute("data-i18n", "cohort.noneSingleTable");
      noneOpt.textContent = i18next.t(
        "cohort.noneSingleTable",
        "None (Single Table)",
      );
      this.ui.cohortJoinTableSelect.appendChild(noneOpt);
      for (const t of finalTables) {
        const opt = document.createElement("option");
        opt.value = t;
        opt.textContent = t;
        this.ui.cohortJoinTableSelect.appendChild(opt);
      }
    }

    this.handleJoinTableChange();
    this.updateSqlPreview();

    const activeCohort = document.activeElement;
    this.ui.previousFocusCohort =
      activeCohort && activeCohort !== document.body
        ? activeCohort
        : this.ui.cohortFilterBtn;
    cohortModal.classList.remove("hidden");
    cohortModal.removeAttribute("aria-hidden");
    if (this.ui.onCohortModalKeyDown) {
      cohortModal.removeEventListener("keydown", this.ui.onCohortModalKeyDown);
    }
    this.ui.onCohortModalKeyDown = (e: KeyboardEvent) => {
      trapModalFocus(e, cohortModal, () => this.close());
    };
    cohortModal.addEventListener("keydown", this.ui.onCohortModalKeyDown);
    this.ui.cohortMinAge?.focus();
  }

  /**
   * Closes the Visual Cohort Query Builder modal.
   * @returns {void}
   */
  public close(): void {
    if (!this.ui.cohortModal) return;
    this.ui.cohortModal.classList.add("hidden");
    this.ui.cohortModal.setAttribute("aria-hidden", "true");
    if (this.ui.onCohortModalKeyDown) {
      this.ui.cohortModal.removeEventListener(
        "keydown",
        this.ui.onCohortModalKeyDown,
      );
      this.ui.onCohortModalKeyDown = null;
    }
    if (
      this.ui.previousFocusCohort instanceof HTMLElement &&
      document.body.contains(this.ui.previousFocusCohort)
    ) {
      this.ui.previousFocusCohort.focus();
    } else {
      this.ui.cohortFilterBtn?.focus();
    }
  }
}
