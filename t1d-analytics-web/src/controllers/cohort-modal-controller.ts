/**
 * @file cohort-modal-controller.ts
 * @module controllers/cohort-modal-controller
 * Composable sub-controller managing the Visual Cohort Query Builder modal,
 * table relationship join resolution, and live DuckDB SQL generation via `ModalDialogController`.
 */

import i18next from "i18next";
import {
  buildCohortSql,
  escapeSqlIdentifier,
  normalizeJoinType,
  resolveTableJoin,
} from "../cohort-sql";
import { type AppEventBus } from "../event-bus";
import { ModalDialogController, requireElementById } from "../modal-utils";
import { type ChatState } from "../state";

/**
 * Narrow injected dependencies for `CohortBuilderModalController`.
 */
export interface CohortBuilderModalDeps {
  /** Shared application state store. */
  state: ChatState;
  /** Unidirectional domain event bus. */
  events: AppEventBus;
}

/**
 * Controller managing the Visual Cohort Query Builder modal (`#cohort-modal`).
 */
export class CohortBuilderModalController {
  /** Shared application state store. */
  private readonly state: ChatState;
  /** Domain event bus. */
  private readonly events: AppEventBus;
  /** Composable modal dialog lifecycle controller. */
  private readonly dialog: ModalDialogController;
  /** Persistent AbortController for form input listeners. */
  private readonly listenerController = new AbortController();
  /** Button to open the Visual Cohort Query Builder modal. */
  public readonly cohortFilterBtn: HTMLButtonElement;
  /** Modal overlay container for the Visual Cohort Query Builder. */
  public readonly cohortModal: HTMLElement;
  /** Button to close the Visual Cohort Query Builder modal. */
  public readonly closeCohortModalBtn: HTMLButtonElement;
  /** Primary target table dropdown inside the cohort modal. */
  public readonly cohortTableSelect: HTMLSelectElement;
  /** Secondary join table dropdown inside the cohort modal. */
  public readonly cohortJoinTableSelect: HTMLSelectElement;
  /** SQL join type dropdown (`INNER JOIN`, `LEFT JOIN`) inside the cohort modal. */
  public readonly cohortJoinTypeSelect: HTMLSelectElement;
  /** Join condition (`ON`) input field inside the cohort modal. */
  public readonly cohortJoinOnInput: HTMLInputElement;
  /** Minimum age filter input inside the cohort modal. */
  public readonly cohortMinAge: HTMLInputElement;
  /** Maximum age filter input inside the cohort modal. */
  public readonly cohortMaxAge: HTMLInputElement;
  /** Sex/gender filter dropdown inside the cohort modal. */
  public readonly cohortGenderSelect: HTMLSelectElement;
  /** Treatment group filter input inside the cohort modal. */
  public readonly cohortTxGroup: HTMLInputElement;
  /** Maximum Time-in-Range (`TIR < %`) filter input inside the cohort modal. */
  public readonly cohortMaxTIR: HTMLInputElement;
  /** Minimum HbA1c (`HbA1c >= %`) filter input inside the cohort modal. */
  public readonly cohortMinHbA1c: HTMLInputElement;
  /** Code element displaying the live generated DuckDB SQL query. */
  public readonly cohortSqlPreview: HTMLElement;
  /** Button to insert the generated cohort SQL into the chat input. */
  public readonly cohortInsertChatBtn: HTMLButtonElement;
  /** Button to immediately execute the generated cohort SQL. */
  public readonly cohortExecuteBtn: HTMLButtonElement;

  /**
   * Creates a new `CohortBuilderModalController`.
   * @param {CohortBuilderModalDeps} deps - Narrow state and event bus dependencies.
   */
  constructor(deps: CohortBuilderModalDeps) {
    this.state = deps.state;
    this.events = deps.events;
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

    this.dialog = new ModalDialogController({
      modalEl: this.cohortModal,
      closeBtn: this.closeCohortModalBtn,
      initialFocusEl: this.closeCohortModalBtn,
    });

    this.bindEvents();
  }

  /**
   * Binds DOM event listeners for the Visual Cohort Query Builder modal.
   * @returns {void}
   */
  private bindEvents(): void {
    const signal = this.listenerController.signal;

    this.cohortFilterBtn.addEventListener(
      "click",
      () => {
        this.open();
      },
      { signal },
    );

    this.cohortTableSelect.addEventListener(
      "change",
      () => {
        this.handleJoinTableChange();
        this.updateSqlPreview();
      },
      { signal },
    );

    this.cohortJoinTableSelect.addEventListener(
      "change",
      () => {
        this.handleJoinTableChange();
        this.updateSqlPreview();
      },
      { signal },
    );

    this.cohortJoinTypeSelect.addEventListener(
      "change",
      () => {
        this.updateSqlPreview();
      },
      { signal },
    );

    this.cohortJoinOnInput.addEventListener(
      "input",
      () => {
        this.updateSqlPreview();
      },
      { signal },
    );

    const cohortInputs = [
      this.cohortMinAge,
      this.cohortMaxAge,
      this.cohortGenderSelect,
      this.cohortTxGroup,
      this.cohortMaxTIR,
      this.cohortMinHbA1c,
    ];
    for (const inp of cohortInputs) {
      inp.addEventListener("input", () => this.updateSqlPreview(), { signal });
      inp.addEventListener("change", () => this.updateSqlPreview(), { signal });
    }

    this.cohortInsertChatBtn.addEventListener(
      "click",
      () => {
        const sql = this.getCohortSql();
        this.close();
        this.events.emit("templateSelected", {
          promptText: sql,
          targetModel: "keep",
          goldSql: null,
        });
      },
      { signal },
    );

    this.cohortExecuteBtn.addEventListener(
      "click",
      () => {
        const sql = this.getCohortSql();
        this.close();
        this.events.emit("templateSelected", {
          promptText: sql,
          targetModel: "sql",
          goldSql: null,
          autoSend: true,
        });
      },
      { signal },
    );
  }

  /**
   * Generates the cohort filter DuckDB SQL query from active modal inputs and typed schema metadata.
   * @returns {string} The generated DuckDB SQL statement.
   */
  public getCohortSql(): string {
    const primaryTable = this.cohortTableSelect.value || "patients";
    const joinTable = this.cohortJoinTableSelect.value || undefined;
    const minAgeStr = this.cohortMinAge.value.trim();
    const maxAgeStr = this.cohortMaxAge.value.trim();
    const maxTirStr = this.cohortMaxTIR.value.trim();
    const minHba1cStr = this.cohortMinHbA1c.value.trim();
    const joinCond = this.cohortJoinOnInput.value.trim();
    const genderVal = this.cohortGenderSelect.value;
    const txVal = this.cohortTxGroup.value.trim();

    const primaryCols = this.state.getTableColumns(primaryTable);
    const joinCols = joinTable
      ? this.state.getTableColumns(joinTable)
      : undefined;

    return buildCohortSql({
      tableName: primaryTable,
      joinType: normalizeJoinType(this.cohortJoinTypeSelect.value),
      ...(primaryCols ? { tableColumns: primaryCols } : {}),
      ...(joinTable
        ? {
            joinTableName: joinTable,
            ...(joinCols ? { joinTableColumns: joinCols } : {}),
          }
        : {}),
      ...(joinCond ? { joinCondition: joinCond } : {}),
      ...(minAgeStr !== "" ? { minAge: Number(minAgeStr) } : {}),
      ...(maxAgeStr !== "" ? { maxAge: Number(maxAgeStr) } : {}),
      ...(genderVal ? { gender: genderVal } : {}),
      ...(txVal ? { txGroup: txVal } : {}),
      ...(maxTirStr !== "" ? { maxTIR: Number(maxTirStr) } : {}),
      ...(minHba1cStr !== "" ? { minHbA1c: Number(minHba1cStr) } : {}),
    });
  }

  /**
   * Updates join condition and types when target or secondary tables change using typed schema state.
   * @returns {void}
   */
  public handleJoinTableChange(): void {
    const primaryTable = this.cohortTableSelect.value || "patients";
    const joinTable = this.cohortJoinTableSelect.value || "";

    if (!joinTable || joinTable === primaryTable) {
      this.cohortJoinOnInput.value = "";
      return;
    }

    const pCols = this.state.getTableColumns(primaryTable);
    const jCols = this.state.getTableColumns(joinTable);

    const resolved = resolveTableJoin(primaryTable, joinTable, pCols, jCols);
    const quotedPrimary = escapeSqlIdentifier(primaryTable);
    const quotedJoin = escapeSqlIdentifier(joinTable);
    if (resolved) {
      this.cohortJoinTypeSelect.value = resolved.defaultJoinType;
      this.cohortJoinOnInput.value = `${quotedPrimary}.${escapeSqlIdentifier(resolved.fromColumn)} = ${quotedJoin}.${escapeSqlIdentifier(resolved.toColumn)}`;
    } else {
      this.cohortJoinOnInput.value = `${quotedPrimary}."patient_id" = ${quotedJoin}."patient_id"`;
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
   * Opens the Visual Cohort Query Builder modal and populates available clinical tables from typed schema state.
   * @returns {void}
   */
  public open(): void {
    this.dialog.captureOpenerFocus();
    const schemaTables = this.state.schemaTables
      .map((t) => t.name)
      .filter((t) => t.length > 0);

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
    this.dialog.open(this.closeCohortModalBtn);
  }

  /**
   * Closes the Visual Cohort Query Builder modal.
   * @returns {void}
   */
  public close(): void {
    this.dialog.close();
  }

  /**
   * Disposes modal and form listeners when tearing down the UI.
   * @returns {void}
   */
  public dispose(): void {
    this.dialog.dispose();
    this.listenerController.abort();
  }
}
