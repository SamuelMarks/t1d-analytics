/**
 * @file template-picker-controller.ts
 * @module controllers/template-picker-controller
 * Composable sub-controller managing the Persistent Prompt & SQL Template Library modal,
 * mid-conversation template switching, Natural-Language-to-SQL mode, and Gold SQL comparison via `ModalDialogController`.
 */

import i18next from "i18next";
import {
  buildStarterQueryItems,
  resolvePreferredLlmModel,
} from "../empty-state-builder";
import { type AppEventBus } from "../event-bus";
import { ModalDialogController } from "../modal-utils";
import { RAW_SQL_MODEL, type ChatState } from "../state";
import {
  fetchClinicalTemplateCatalog,
  getTable1ExampleQueries,
  type Table1ExampleSpec,
} from "../table1-queries";

/**
 * Unified template specification rendered in the Template Picker modal.
 */
export interface TemplateLibraryItem {
  /** Stable identifier ('starter-1'..'starter-5' or 'table1-1'..'table1-8'). */
  id: string;
  /** Category bucket ('starter' or 'table1'). */
  category: "starter" | "table1";
  /** Localized display title. */
  title: string;
  /** Complexity rating label. */
  complexity: Table1ExampleSpec["complexity"];
  /** Plain-English clinical question for NL-to-SQL evaluation. */
  question: string;
  /** Canonical Gold-Standard DuckDB SQL query. */
  goldSql: string;
}

/**
 * Narrow injected dependencies for `TemplatePickerController`.
 */
export interface TemplatePickerDeps {
  /** Shared application state store. */
  state: ChatState;
  /** Unidirectional domain event bus. */
  events: AppEventBus;
  /** Model selector dropdown inspected (read-only) to pick preferred LLM models. */
  modelSelect: HTMLSelectElement;
  /** Callback executing an immediate NL-to-SQL vs Gold benchmark turn. */
  onSendBenchmark: (
    question: string,
    model: string,
    goldSql: string,
  ) => Promise<void>;
}

/**
 * Composable sub-controller for the Persistent Prompt & SQL Template Library modal (`#template-picker-modal`).
 */
export class TemplatePickerController {
  /** Injected dependencies. */
  private readonly deps: TemplatePickerDeps;
  /** Optional composable modal dialog controller when `#template-picker-modal` exists in DOM. */
  private readonly dialog: ModalDialogController | null;
  /** Persistent AbortController for button and `Cmd+/` / `Ctrl+/` shortcut listeners. */
  private readonly listenerController = new AbortController();
  /** Trigger button inside the chat composer toolbar (`#template-picker-btn`). */
  public readonly openBtn: HTMLElement | null;
  /** Modal overlay container (`#template-picker-modal`). */
  public readonly modalEl: HTMLElement | null;
  /** Modal close button (`#close-template-modal-btn`). */
  public readonly closeBtn: HTMLElement | null;
  /** Container holding rendered template cards (`#template-picker-list`). */
  public readonly listContainer: HTMLElement | null;
  /** Active category filter ('all', 'table1', or 'starter'). */
  public activeCategory: "all" | "table1" | "starter" = "all";

  /**
   * Creates a new `TemplatePickerController`.
   * @param {TemplatePickerDeps} deps - Narrow state, event bus, and benchmark callback dependencies.
   */
  constructor(deps: TemplatePickerDeps) {
    this.deps = deps;
    this.openBtn = document.getElementById("template-picker-btn");
    this.modalEl = document.getElementById("template-picker-modal");
    this.closeBtn = document.getElementById("close-template-modal-btn");
    this.listContainer = document.getElementById("template-picker-list");

    this.dialog = this.modalEl
      ? new ModalDialogController({
          modalEl: this.modalEl,
          closeBtn: this.closeBtn,
          initialFocusEl: this.closeBtn,
        })
      : null;

    this.bindEvents();
  }

  /**
   * Builds the full list of starter and Academic Table 1 template items.
   * @returns {TemplateLibraryItem[]} Array of template items.
   */
  public buildTemplateItems(): TemplateLibraryItem[] {
    const items: TemplateLibraryItem[] = [];
    buildStarterQueryItems().forEach((starter) => {
      items.push({
        id: starter.id,
        category: "starter",
        title: starter.label,
        complexity: starter.complexity,
        question: starter.question,
        goldSql: starter.goldSql,
      });
    });

    getTable1ExampleQueries().forEach((spec) => {
      items.push({
        id: spec.id,
        category: "table1",
        title: i18next.t(spec.i18nKey),
        complexity: spec.complexity,
        question: spec.question,
        goldSql: spec.sql,
      });
    });
    return items;
  }

  /**
   * Synchronizes the clinical template catalog from `GET /api/templates` and re-renders if open.
   * @returns {Promise<boolean>} True if backend templates were applied.
   */
  public async refreshCatalog(): Promise<boolean> {
    const updated = await fetchClinicalTemplateCatalog();
    if (updated && this.dialog?.isOpen() === true) {
      this.renderTemplateList();
    }
    return updated;
  }

  /**
   * Binds click and keyboard listeners for opening, filtering, and closing the template modal.
   * @returns {void}
   */
  private bindEvents(): void {
    const signal = this.listenerController.signal;

    this.openBtn?.addEventListener(
      "click",
      () => {
        this.openModal();
      },
      { signal },
    );

    if (this.modalEl) {
      const filterBtns = this.modalEl.querySelectorAll<HTMLButtonElement>(
        "[data-template-tab]",
      );
      filterBtns.forEach((btn) => {
        btn.addEventListener(
          "click",
          () => {
            const tab = btn.getAttribute("data-template-tab");
            const nextTab: "all" | "table1" | "starter" =
              tab === "starter" || tab === "table1" ? tab : "all";
            this.setCategoryFilter(nextTab);
          },
          { signal },
        );
      });
    }

    document.addEventListener(
      "keydown",
      (e: KeyboardEvent) => {
        if ((e.ctrlKey || e.metaKey) && e.key === "/") {
          e.preventDefault();
          if (this.dialog?.isOpen() === true) {
            this.closeModal();
          } else {
            this.openModal();
          }
        }
      },
      { signal },
    );
  }

  /**
   * Updates the active template category filter and re-renders the template list.
   * @param {"all" | "table1" | "starter"} category - Selected category filter.
   * @returns {void}
   */
  public setCategoryFilter(category: "all" | "table1" | "starter"): void {
    this.activeCategory = category;
    if (this.modalEl) {
      const filterBtns = this.modalEl.querySelectorAll<HTMLButtonElement>(
        "[data-template-tab]",
      );
      filterBtns.forEach((btn) => {
        const isMatch = btn.getAttribute("data-template-tab") === category;
        btn.classList.toggle("active", isMatch);
        btn.setAttribute("aria-selected", isMatch ? "true" : "false");
      });
    }
    this.renderTemplateList();
  }

  /**
   * Opens the Template Picker modal and renders all template cards.
   * @returns {void}
   */
  public openModal(): void {
    if (!this.dialog) return;
    this.dialog.captureOpenerFocus();
    this.renderTemplateList();
    this.dialog.open(this.closeBtn);
  }

  /**
   * Closes the Template Picker modal and restores previous keyboard focus.
   * @returns {void}
   */
  public closeModal(): void {
    if (!this.dialog) return;
    this.dialog.close();
  }

  /**
   * Applies a template in Natural-Language-to-SQL mode, emitting `templateSelected`
   * with the natural-language question, preferred LLM model, and `goldSql`.
   * @param {TemplateLibraryItem} item - Selected template specification.
   * @returns {void}
   */
  public applyTemplateAsNaturalLanguage(item: TemplateLibraryItem): void {
    const targetLlm = resolvePreferredLlmModel(this.deps.modelSelect);
    this.closeModal();
    this.deps.events.emit("templateSelected", {
      promptText: item.question,
      targetModel: targetLlm,
      goldSql: item.goldSql,
    });
  }

  /**
   * Applies a template in Literal Gold SQL mode, emitting `templateSelected`
   * with `goldSql` and `RAW_SQL_MODEL`.
   * @param {TemplateLibraryItem} item - Selected template specification.
   * @returns {void}
   */
  public applyTemplateAsGoldSql(item: TemplateLibraryItem): void {
    this.closeModal();
    this.deps.events.emit("templateSelected", {
      promptText: `-- ${item.question}
${item.goldSql}`,
      targetModel: RAW_SQL_MODEL,
      goldSql: null,
    });
  }

  /**
   * Immediately dispatches a benchmark turn sending the template's natural-language question
   * and `goldSql` to the selected target LLM and executing both on the DuckDB read-only cursor.
   * @param {TemplateLibraryItem} item - Selected template specification.
   * @returns {Promise<void>}
   */
  public async runTemplateBenchmark(item: TemplateLibraryItem): Promise<void> {
    const targetLlm = resolvePreferredLlmModel(this.deps.modelSelect);
    this.deps.modelSelect.value = targetLlm;
    this.deps.state.setActiveChatModel(targetLlm);
    this.closeModal();
    await this.deps.onSendBenchmark(item.question, targetLlm, item.goldSql);
  }

  /**
   * Renders the filtered template cards inside `#template-picker-list`.
   * @returns {void}
   */
  public renderTemplateList(): void {
    if (!this.listContainer) return;
    const allItems = this.buildTemplateItems();
    const filtered = allItems.filter(
      (item) =>
        this.activeCategory === "all" || item.category === this.activeCategory,
    );

    const fragment = document.createDocumentFragment();
    for (const item of filtered) {
      const card = document.createElement("div");
      card.className = "template-picker-card";
      card.setAttribute("data-template-id", item.id);
      card.setAttribute("data-complexity", item.complexity);

      const header = document.createElement("div");
      header.className = "template-picker-card-header";

      const titleEl = document.createElement("strong");
      titleEl.className = "template-picker-card-title";
      titleEl.textContent = item.title;

      const badge = document.createElement("span");
      badge.className = "complexity-badge";
      badge.textContent = item.complexity;

      header.append(titleEl, badge);

      const questionP = document.createElement("p");
      questionP.className = "template-picker-card-question";
      questionP.textContent = item.question;

      const detailsEl = document.createElement("details");
      detailsEl.className = "template-picker-sql-details";
      const summaryEl = document.createElement("summary");
      summaryEl.textContent = i18next.t(
        "templates.viewGoldSql",
        "View Gold-Standard DuckDB SQL",
      );
      const sqlPre = document.createElement("pre");
      const sqlCode = document.createElement("code");
      sqlCode.className = "language-sql";
      sqlCode.textContent = item.goldSql;
      sqlPre.appendChild(sqlCode);
      detailsEl.append(summaryEl, sqlPre);

      const actionsRow = document.createElement("div");
      actionsRow.className = "template-picker-card-actions";

      const tryNlBtn = document.createElement("button");
      tryNlBtn.type = "button";
      tryNlBtn.className = "btn-secondary btn-sm template-try-nl-btn";
      tryNlBtn.textContent = i18next.t(
        "templates.tryNaturalLanguage",
        "💬 Try NL -> SQL",
      );
      tryNlBtn.addEventListener("click", () => {
        this.applyTemplateAsNaturalLanguage(item);
      });

      const runGoldBtn = document.createElement("button");
      runGoldBtn.type = "button";
      runGoldBtn.className = "btn-secondary btn-sm template-insert-sql-btn";
      runGoldBtn.textContent = i18next.t(
        "templates.insertGoldSql",
        "⚡ Insert Gold SQL",
      );
      runGoldBtn.addEventListener("click", () => {
        this.applyTemplateAsGoldSql(item);
      });

      const benchmarkBtn = document.createElement("button");
      benchmarkBtn.type = "button";
      benchmarkBtn.className = "btn-primary btn-sm template-benchmark-btn";
      benchmarkBtn.textContent = i18next.t(
        "templates.benchmarkLlm",
        "⚖️ Benchmark LLM vs Gold",
      );
      benchmarkBtn.addEventListener("click", () => {
        void this.runTemplateBenchmark(item);
      });

      actionsRow.append(tryNlBtn, runGoldBtn, benchmarkBtn);
      card.append(header, questionP, detailsEl, actionsRow);
      fragment.appendChild(card);
    }

    this.listContainer.replaceChildren(fragment);
  }

  /**
   * Removes document and modal event listeners registered by this controller.
   * @returns {void}
   */
  public dispose(): void {
    this.dialog?.dispose();
    this.listenerController.abort();
  }
}
