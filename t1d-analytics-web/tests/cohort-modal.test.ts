/**
 * @file cohort-modal.test.ts
 * @module tests/cohort-modal.test
 * Unit tests for CohortBuilderModalController, clinical cohort SQL construction, and relational join resolution.
 */

import { describe, it, expect, beforeEach, vi, afterEach } from "vitest";
import i18next from "i18next";
import { ChatState } from "../src/state";
import {
  ChatUI,
  CohortBuilderModalController,
  buildCohortSql,
  normalizeJoinType,
} from "../src/ui";
import { flushPromises, getTestElement, mountChatUiTestDom } from "./setup";

describe("CohortBuilderModalController & Cohort SQL Generation", () => {
  let state: ChatState;
  let ui: ChatUI;
  let mockFetch: ReturnType<typeof vi.fn>;

  beforeEach(async () => {
    mountChatUiTestDom();
    mockFetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ models: [], tables: [] }),
    });
    vi.stubGlobal("fetch", mockFetch);

    state = new ChatState();
    await i18next.changeLanguage("en");
    document.documentElement.lang = "en";
    ui = new ChatUI(state);
    await flushPromises();
  });

  afterEach(() => {
    ui.dispose();
    localStorage.clear();
    document.body.innerHTML = "";
    vi.restoreAllMocks();
  });

  it("buildCohortSql generates SQL with various filter predicates and join types", () => {
    expect(ui.cohortModalController).toBeInstanceOf(
      CohortBuilderModalController,
    );
    expect(normalizeJoinType("LEFT JOIN")).toBe("LEFT JOIN");
    expect(normalizeJoinType("INNER JOIN")).toBe("INNER JOIN");
    expect(normalizeJoinType(undefined)).toBe("INNER JOIN");

    expect(buildCohortSql({ tableName: "patients" })).toBe(
      `SELECT *
FROM "patients";`,
    );
    expect(buildCohortSql({ tableName: "" })).toBe(
      `SELECT *
FROM "patients";`,
    );

    const sql = buildCohortSql({
      tableName: 'demographics"test',
      minAge: 18,
      maxAge: 65,
      gender: "F",
      txGroup: "Closed'Loop",
      maxTIR: 70,
      minHbA1c: 8.5,
    });

    expect(sql).toContain('FROM "demographics""test"');
    expect(sql).toContain("age >= 18");
    expect(sql).toContain("age <= 65");
    expect(sql).toContain("gender = 'F'");
    expect(sql).toContain("txgroup ILIKE '%Closed''Loop%'");
    expect(sql).toContain("tir <= 70");
    expect(sql).toContain("hba1c >= 8.5");

    const nonFiniteSql = buildCohortSql({
      tableName: "patients",
      minAge: Infinity,
      maxAge: NaN,
      maxTIR: -Infinity,
      minHbA1c: NaN,
    });
    expect(nonFiniteSql).toBe(`SELECT *
FROM "patients";`);

    const joinSql = buildCohortSql({
      tableName: "patients",
      joinTableName: "cgms",
      joinType: "INNER JOIN",
      minAge: 18,
    });
    expect(joinSql).toContain('FROM "patients"');
    expect(joinSql).toContain('INNER JOIN "cgms"');
    expect(joinSql).toContain('ON "patients".patient_id = "cgms".patient_id');

    const leftJoinSql = buildCohortSql({
      tableName: "patients",
      joinTableName: "visits",
      joinType: "LEFT JOIN",
      joinCondition: '"patients".patient_id = "visits".subject_id',
    });
    expect(leftJoinSql).toContain('LEFT JOIN "visits"');
    expect(leftJoinSql).toContain(
      'ON "patients".patient_id = "visits".subject_id',
    );

    const jaebSql = buildCohortSql({
      tableName: "tblaptsummary",
      joinTableName: "hba1c",
      minAge: 12,
      maxAge: 35,
      gender: "M",
      txGroup: "Control",
      maxTIR: 65,
      minHbA1c: 7.2,
      tableColumns: [
        "PtID",
        "AgeAsOfRandDt",
        "Gender",
        "TxGroup",
        "time_in_range",
        "baseline_hba1c",
      ],
      joinTableColumns: ["PtID", "HbA1c"],
    });
    expect(jaebSql).toContain('"tblaptsummary".AgeAsOfRandDt >= 12');
    expect(jaebSql).toContain('"tblaptsummary".AgeAsOfRandDt <= 35');
    expect(jaebSql).toContain("\"tblaptsummary\".Gender = 'M'");
    expect(jaebSql).toContain("\"tblaptsummary\".TxGroup ILIKE '%Control%'");
    expect(jaebSql).toContain('"tblaptsummary".time_in_range <= 65');
    expect(jaebSql).toContain('"tblaptsummary".baseline_hba1c >= 7.2');
    expect(jaebSql).toContain('ON "tblaptsummary".PtID = "hba1c".PtID');

    const fallbackJoinSql = buildCohortSql({
      tableName: "custom_alpha",
      joinTableName: "custom_beta",
      tableColumns: ["unmatched_col"],
      joinTableColumns: ["other_col"],
    });
    expect(fallbackJoinSql).toContain(
      'ON "custom_alpha".patient_id = "custom_beta".patient_id',
    );
  });

  it("handles cohort modal open, input changes, insert, execute, and closing", async () => {
    const filterBtn = getTestElement<HTMLButtonElement>("cohort-filter-btn");
    const modal = getTestElement("cohort-modal");
    const closeBtn = getTestElement<HTMLButtonElement>(
      "close-cohort-modal-btn",
    );

    expect(modal.classList.contains("hidden")).toBe(true);

    filterBtn.click();
    expect(modal.classList.contains("hidden")).toBe(false);

    const minAgeInput = getTestElement<HTMLInputElement>("cohort-min-age");
    minAgeInput.value = "21";
    minAgeInput.dispatchEvent(new Event("input"));

    const preview = getTestElement("cohort-sql-preview");
    expect(preview.textContent).toContain("age >= 21");

    const joinTableSelect = getTestElement<HTMLSelectElement>(
      "cohort-join-table-select",
    );
    const optCgm = document.createElement("option");
    optCgm.value = "cgm_readings";
    optCgm.textContent = "cgm_readings";
    joinTableSelect.appendChild(optCgm);
    joinTableSelect.value = "cgm_readings";
    joinTableSelect.dispatchEvent(new Event("change"));
    expect(preview.textContent).toContain('JOIN "cgm_readings"');

    const tableSelect = getTestElement<HTMLSelectElement>(
      "cohort-table-select",
    );
    tableSelect.dispatchEvent(new Event("change"));

    const joinTypeSelect = getTestElement<HTMLSelectElement>(
      "cohort-join-type-select",
    );
    joinTypeSelect.value = "LEFT JOIN";
    joinTypeSelect.dispatchEvent(new Event("change"));
    expect(preview.textContent).toContain('LEFT JOIN "cgm_readings"');

    const joinOnInput = getTestElement<HTMLInputElement>(
      "cohort-join-on-input",
    );
    joinOnInput.value = '"patients".subject_id = "cgm_readings".subject_id';
    joinOnInput.dispatchEvent(new Event("input"));

    const genderSelect = getTestElement<HTMLSelectElement>(
      "cohort-gender-select",
    );
    genderSelect.value = "F";
    genderSelect.dispatchEvent(new Event("change"));

    const maxAgeInput = getTestElement<HTMLInputElement>("cohort-max-age");
    maxAgeInput.value = "60";
    maxAgeInput.dispatchEvent(new Event("change"));
    const txInput = getTestElement<HTMLInputElement>("cohort-txgroup");
    txInput.value = "Closed";
    txInput.dispatchEvent(new Event("input"));
    const maxTirInput = getTestElement<HTMLInputElement>("cohort-max-tir");
    maxTirInput.value = "75";
    maxTirInput.dispatchEvent(new Event("input"));
    const minHba1cInput = getTestElement<HTMLInputElement>("cohort-min-hba1c");
    minHba1cInput.value = "7.0";
    minHba1cInput.dispatchEvent(new Event("input"));

    joinTableSelect.value = "";
    joinTableSelect.dispatchEvent(new Event("change"));
    expect(preview.textContent).not.toContain("JOIN");

    closeBtn.click();
    expect(modal.classList.contains("hidden")).toBe(true);

    filterBtn.click();
    expect(modal.classList.contains("hidden")).toBe(false);
    modal.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    expect(modal.classList.contains("hidden")).toBe(true);

    filterBtn.click();
    minAgeInput.value = "25";
    minAgeInput.dispatchEvent(new Event("input"));
    const insertBtn = getTestElement<HTMLButtonElement>(
      "cohort-insert-chat-btn",
    );
    insertBtn.click();
    const input = getTestElement<HTMLTextAreaElement>("chat-input");
    expect(input.value).toContain("age >= 25");
    expect(modal.classList.contains("hidden")).toBe(true);

    state.createChat();
    ui.render();
    filterBtn.click();
    minAgeInput.value = "30";
    minAgeInput.dispatchEvent(new Event("input"));
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ content: "Ran cohort query", sqlResult: [] }),
    });
    const execBtn = getTestElement<HTMLButtonElement>("cohort-execute-btn");
    execBtn.click();
    await flushPromises();
    const activeChat = state.getActiveChat();
    expect(activeChat?.model).toBe("sql");
    expect(activeChat?.messages.length).toBe(2);
    expect(activeChat?.messages[0]?.content).toContain("age >= 30");
  });

  it("populates tables from DOM schema headers and resolves relational join columns", () => {
    const schemaContent = getTestElement("schema-content");
    const tblA = document.createElement("div");
    tblA.className = "schema-table";
    const nameA = document.createElement("span");
    nameA.className = "schema-table-name";
    nameA.setAttribute("data-table", "subjects_custom");
    nameA.textContent = "subjects_custom";
    const colA = document.createElement("span");
    colA.className = "col-name";
    colA.textContent = "subject_id";
    tblA.append(nameA, colA);

    const tblB = document.createElement("div");
    tblB.className = "schema-table";
    const nameB = document.createElement("span");
    nameB.className = "schema-table-name";
    nameB.setAttribute("data-table", "labs_custom");
    nameB.textContent = "labs_custom";
    const colB = document.createElement("span");
    colB.className = "col-name";
    colB.textContent = "subject_id";
    tblB.append(nameB, colB);

    const tblC = document.createElement("div");
    tblC.className = "schema-table";
    const nameC = document.createElement("span");
    nameC.className = "schema-table-name";
    nameC.setAttribute("data-table", "patient_notes");
    nameC.textContent = "patient_notes";
    const colC = document.createElement("span");
    colC.className = "col-name";
    colC.textContent = "note_text";
    tblC.append(nameC, colC);

    schemaContent.replaceChildren(tblA, tblB, tblC);

    ui.cohortModalController.open();
    const tableSelect = ui.cohortModalController.cohortTableSelect;
    const joinSelect = ui.cohortModalController.cohortJoinTableSelect;
    const joinOnInput = ui.cohortModalController.cohortJoinOnInput;

    expect(tableSelect.options[0]?.value).toBe("subjects_custom");

    tableSelect.value = "subjects_custom";
    joinSelect.value = "labs_custom";
    ui.cohortModalController.handleJoinTableChange();
    expect(joinOnInput.value).toBe(
      '"subjects_custom".subject_id = "labs_custom".subject_id',
    );

    tableSelect.value = "subjects_custom";
    joinSelect.value = "patient_notes";
    ui.cohortModalController.handleJoinTableChange();
    expect(joinOnInput.value).toBe(
      '"subjects_custom".patient_id = "patient_notes".patient_id',
    );

    // Reverse known relation lookup (cgm_readings -> patients) and unrelated tables fallback
    schemaContent.replaceChildren();
    ui.cohortModalController.open();
    tableSelect.value = "cgm_readings";
    joinSelect.value = "patients";
    ui.cohortModalController.handleJoinTableChange();
    expect(joinOnInput.value).toBe(
      '"cgm_readings".subject_id = "patients".subject_id',
    );

    tableSelect.value = "tblaptsummary";
    joinSelect.value = "cgm_readings";
    ui.cohortModalController.handleJoinTableChange();
    expect(joinOnInput.value).toBe(
      '"tblaptsummary".patient_id = "cgm_readings".patient_id',
    );

    // Fallback when tableSelect/joinSelect values are empty strings
    tableSelect.innerHTML = `<option value="">Empty</option>`;
    tableSelect.value = "";
    expect(ui.cohortModalController.getCohortSql()).toContain(
      'FROM "patients"',
    );
    ui.cohortModalController.handleJoinTableChange();

    // Close before open when previousFocusCohort is null
    const freshController = new CohortBuilderModalController(ui);
    freshController.close();
    freshController.dispose();

    // Escape key dismissal and focus restoration
    ui.cohortModalController.close();
    const trigger = getTestElement<HTMLButtonElement>("cohort-filter-btn");
    trigger.focus();
    ui.cohortModalController.open();
    ui.cohortModalController.open();
    const modal = ui.cohortModalController.cohortModal;
    modal.dispatchEvent(
      new KeyboardEvent("keydown", {
        key: "Escape",
        bubbles: true,
        cancelable: true,
      }),
    );
    expect(modal.classList.contains("hidden")).toBe(true);
    expect(document.activeElement).toBe(trigger);

    ui.cohortModalController.open();
    expect(modal.classList.contains("hidden")).toBe(false);
    ui.cohortModalController.dispose();
    modal.dispatchEvent(
      new KeyboardEvent("keydown", {
        key: "Escape",
        bubbles: true,
        cancelable: true,
      }),
    );
    expect(modal.classList.contains("hidden")).toBe(false);
  });
});
