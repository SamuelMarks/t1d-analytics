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
      json: async () => ({ models: [], schema: [] }),
    });
    vi.stubGlobal("fetch", mockFetch);

    state = new ChatState();
    i18next.changeLanguage("en");
    document.documentElement.lang = "en";
    ui = new ChatUI(state);
    await flushPromises();
  });

  afterEach(() => {
    ui.destroy();
    localStorage.clear();
    document.body.innerHTML = "";
    vi.restoreAllMocks();
  });

  it("buildCohortSql generates SQL with various filter predicates", () => {
    expect(ui.cohortModalController).toBeInstanceOf(
      CohortBuilderModalController,
    );
    ui.cohortModalController.open();
    ui.cohortModalController.close();

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
      txGroup: "Closed-Loop",
      maxTIR: 70,
      minHbA1c: 8.5,
    });

    expect(sql).toContain('FROM "demographics""test"');
    expect(sql).toContain("WHERE age >= 18");
    expect(sql).toContain("AND age <= 65");
    expect(sql).toContain("AND gender = 'F'");
    expect(sql).toContain("AND txgroup ILIKE '%Closed-Loop%'");
    expect(sql).toContain("AND tir <= 70");
    expect(sql).toContain("AND hba1c >= 8.5");

    const nonFiniteSql = buildCohortSql({
      tableName: "patients",
      minAge: Infinity,
      maxAge: NaN,
      maxTIR: -Infinity,
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
    expect(joinSql).toContain('"patients".age >= 18');

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
    joinTableSelect.value = "cgm_data";
    joinTableSelect.dispatchEvent(new Event("change"));
    expect(preview.textContent).toContain('JOIN "cgm_data"');

    joinTableSelect.value = "";
    joinTableSelect.dispatchEvent(new Event("change"));
    expect(preview.textContent).not.toContain("JOIN");

    closeBtn.click();
    expect(modal.classList.contains("hidden")).toBe(true);

    filterBtn.click();
    expect(modal.classList.contains("hidden")).toBe(false);
    modal.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    expect(modal.classList.contains("hidden")).toBe(true);

    // Insert into chat button
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

    // Execute button directly running query
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

  it("handles openCohortModal populating tables from DOM schema headers and fallback tables", () => {
    const header = document.createElement("div");
    header.className = "schema-table-header";
    header.innerHTML = "<h4>clinical_cohort_table</h4><h4>   </h4>";
    document.body.appendChild(header);

    ui.openCohortModal();
    const select = getTestElement<HTMLSelectElement>("cohort-table-select");
    expect(select.options[0]?.value).toBe("clinical_cohort_table");
    header.remove();

    const toggleBtn = document.createElement("button");
    toggleBtn.className = "schema-table-header-toggle";
    toggleBtn.innerHTML = "<span>discovered_cgm_table</span>";
    document.body.appendChild(toggleBtn);
    ui.openCohortModal();
    expect(select.options[0]?.value).toBe("discovered_cgm_table");
    ui.closeCohortModal();
    toggleBtn.remove();

    const origTableSelect = ui["cohortTableSelect"];
    const origJoinSelect = ui["cohortJoinTableSelect"];
    ui["cohortTableSelect"] = null;
    ui["cohortJoinTableSelect"] = null;
    ui.openCohortModal();
    ui["cohortTableSelect"] = origTableSelect;
    ui["cohortJoinTableSelect"] = origJoinSelect;

    // Fallback tables when no schema headers exist
    ui.openCohortModal();
    expect(select.options.length).toBe(3);
    ui.closeCohortModal();
  });

  it("handles cohort relational join resolution and Escape key focus restoration", async () => {
    const tableSelect = getTestElement<HTMLSelectElement>(
      "cohort-table-select",
    );
    const joinTableSelect = getTestElement<HTMLSelectElement>(
      "cohort-join-table-select",
    );
    const joinTypeSelect = getTestElement<HTMLSelectElement>(
      "cohort-join-type-select",
    );
    const joinOnInput = getTestElement<HTMLInputElement>(
      "cohort-join-on-input",
    );

    tableSelect.innerHTML = `<option value="patients">patients</option><option value="cgms">cgms</option>`;
    joinTableSelect.innerHTML = `<option value="">None (Single Table)</option><option value="cgms">cgms</option><option value="visits">visits</option>`;

    tableSelect.dispatchEvent(new Event("change"));

    joinTableSelect.value = "cgms";
    ui.handleCohortJoinTableChange();
    expect(joinOnInput.value).toBe('"patients".patient_id = "cgms".patient_id');
    expect(joinTypeSelect.value).toBe("INNER JOIN");

    joinTableSelect.value = "visits";
    ui.handleCohortJoinTableChange();
    expect(joinTypeSelect.value).toBe("LEFT JOIN");

    tableSelect.innerHTML += `<option value="sensors">sensors</option>`;
    tableSelect.value = "sensors";
    joinTableSelect.innerHTML += `<option value="unknown_table">unknown_table</option>`;
    joinTableSelect.value = "unknown_table";
    ui.handleCohortJoinTableChange();
    expect(joinOnInput.value).toBe(
      '"sensors".patient_id = "unknown_table".patient_id',
    );
    tableSelect.value = "patients";

    joinTypeSelect.value = "";
    expect(ui.getCohortSql()).toContain("INNER JOIN");
    joinTypeSelect.value = "LEFT JOIN";

    joinTableSelect.value = "patients";
    joinTableSelect.dispatchEvent(new Event("change"));
    expect(joinOnInput.value).toBe("");

    tableSelect.innerHTML = `<option value="patients">patients</option>`;
    tableSelect.value = "patients";
    joinTableSelect.innerHTML = `<option value="visits">visits</option>`;
    joinTableSelect.value = "visits";
    joinTypeSelect.value = "LEFT JOIN";
    joinOnInput.value = '"patients".patient_id = "visits".patient_id';

    const minAgeInput = getTestElement<HTMLInputElement>("cohort-min-age");
    minAgeInput.value = "20";
    minAgeInput.dispatchEvent(new Event("change"));

    const maxAgeInput = getTestElement<HTMLInputElement>("cohort-max-age");
    maxAgeInput.value = "60";
    maxAgeInput.dispatchEvent(new Event("change"));

    const maxTIRInput = getTestElement<HTMLInputElement>("cohort-max-tir");
    maxTIRInput.value = "70";
    maxTIRInput.dispatchEvent(new Event("change"));

    const minHbA1cInput = getTestElement<HTMLInputElement>("cohort-min-hba1c");
    minHbA1cInput.value = "6.5";
    minHbA1cInput.dispatchEvent(new Event("change"));

    joinTypeSelect.dispatchEvent(new Event("change"));
    joinOnInput.dispatchEvent(new Event("input"));

    const preview = getTestElement("cohort-sql-preview");
    expect(preview.textContent).toContain('LEFT JOIN "visits"');
    expect(preview.textContent).toContain("age >= 20");
    expect(preview.textContent).toContain("age <= 60");
    expect(preview.textContent).toContain("tir <= 70");
    expect(preview.textContent).toContain("hba1c >= 6.5");

    // Escape key dismissal and focus restoration
    document.body.focus();
    ui.openCohortModal();
    ui.closeCohortModal();

    const trigger = getTestElement<HTMLButtonElement>("cohort-filter-btn");
    trigger.focus();

    ui.openCohortModal();
    const modal = getTestElement("cohort-modal");
    expect(modal.classList.contains("hidden")).toBe(false);
    expect(modal.getAttribute("aria-hidden")).toBeNull();

    const escEvent = new KeyboardEvent("keydown", {
      key: "Escape",
      bubbles: true,
      cancelable: true,
    });
    modal.dispatchEvent(escEvent);

    expect(modal.classList.contains("hidden")).toBe(true);
    expect(modal.getAttribute("aria-hidden")).toBe("true");
    expect(document.activeElement).toBe(trigger);

    const tempTrigger = document.createElement("button");
    document.body.appendChild(tempTrigger);
    tempTrigger.focus();
    ui.openCohortModal();
    tempTrigger.remove();
    ui.closeCohortModal();
    expect(document.activeElement).toBe(trigger);

    ui.openCohortModal();
    ui.openCohortModal();
    ui.closeCohortModal();
    ui.closeCohortModal();

    // Test resilience when optional cohort elements are absent
    const joinOnParent = joinOnInput.parentElement;
    const joinOnNext = joinOnInput.nextSibling;
    joinOnInput.remove();
    const noJoinOnUi = new ChatUI(state);
    await flushPromises();
    tableSelect.innerHTML = `<option value="patients">patients</option>`;
    joinTableSelect.innerHTML = `<option value="patients">patients</option><option value="cgms">cgms</option>`;
    tableSelect.value = "patients";
    joinTableSelect.value = "patients";
    noJoinOnUi.handleCohortJoinTableChange();
    joinTableSelect.value = "cgms";
    noJoinOnUi.handleCohortJoinTableChange();
    noJoinOnUi.destroy();
    if (joinOnParent) {
      joinOnParent.insertBefore(joinOnInput, joinOnNext);
    }
  });
});
