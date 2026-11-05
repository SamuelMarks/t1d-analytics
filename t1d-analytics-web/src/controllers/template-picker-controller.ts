/**
 * @file template-picker-controller.ts
 * @module controllers/template-picker-controller
 * Composable sub-controller managing the Persistent Prompt & SQL Template Library modal,
 * mid-conversation template switching, Natural-Language-to-SQL prompt insertion,
 * direct Gold-Standard SQL insertion, and one-click LLM vs. Gold SQL benchmarking.
 */

import i18next from "i18next";
import {
  buildStarterQueryItems,
  resolvePreferredLlmModel,
} from "../empty-state-builder";
import { trapModalFocus } from "../modal-utils";
import { RAW_SQL_MODEL, type ChatState } from "../state";
import { TABLE_1_EXAMPLE_QUERIES } from "../table1-queries";

/**
 * Host contract required by `TemplatePickerController`.
 */
export interface TemplatePickerHost {
  /** Global application state store. */
  state: ChatState;
  /** Textarea element where the user types queries or SQL. */
  chatInput: HTMLTextAreaElement;
  /** Dropdown select element for choosing the active LLM or SQL engine. */
  modelSelect: HTMLSelectElement;
  /** Synchronizes syntax highlighting in the chat input. */
  syncHighlight(): void;
  /** Sends a chat message with optional model and gold-standard SQL override. */
  handleSendMessage(
    overrideMessage?: string,
    overrideModel?: string,
    overrideGoldSql?: string,
  ): Promise<void>;
}

/**
 * Unified template specification rendered inside the Template Picker modal.
 */
export interface TemplateLibraryItem {
  /** Stable identifier for the template item. */
  id: string;
  /** Category tab ('starter' or 'table1'). */
  category: "starter" | "table1";
  /** Localized display title. */
  title: string;
  /** Complexity tier label. */
  complexity: string;
  /** Natural-language clinical question for NL-to-SQL testing. */
  question: string;
  /** Canonical Gold-Standard DuckDB SQL query. */
  goldSql: string;
}

/**
 * Composable sub-controller for the Persistent Prompt & SQL Template Library modal (`#template-picker-modal`).
 */
export class TemplatePickerController {
  /** Host UI coordinator. */
  private readonly ui: TemplatePickerHost;
  /** Persistent button in the header/composer that opens the template modal (`#template-picker-btn`). */
  public readonly openBtn: HTMLElement | null;
  /** Modal overlay container (`#template-picker-modal`). */
  public readonly modalEl: HTMLElement | null;
  /** Modal close button (`#close-template-modal-btn`). */
  public readonly closeBtn: HTMLElement | null;
  /** Container holding the rendered template cards (`#template-picker-list`). */
  public readonly listContainer: HTMLElement | null;
  /** Currently selected category filter ('all' | 'table1' | 'starter'). */
  public activeCategory: "all" | "table1" | "starter" = "all";
  /** Element focused immediately before opening the modal. */
  private previousFocusEl: HTMLElement | null = null;
  /** Bound keydown listener for focus trapping and Escape handling inside the modal. */
  private onModalKeyDown: ((e: KeyboardEvent) => void) | null = null;
  /** Bound global shortcut listener (`Ctrl+/` or `Cmd+/`). */
  private onGlobalKeyDown: ((e: KeyboardEvent) => void) | null = null;

  /**
   * Initializes the TemplatePickerController and binds trigger/modal listeners when present in the DOM.
   * @param {TemplatePickerHost} ui - Host UI instance.
   */
  constructor(ui: TemplatePickerHost) {
    this.ui = ui;
    this.openBtn = document.getElementById("template-picker-btn");
    this.modalEl = document.getElementById("template-picker-modal");
    this.closeBtn = document.getElementById("close-template-modal-btn");
    this.listContainer = document.getElementById("template-picker-list");
    this.bindEvents();
  }

  /**
   * Builds the full list of starter and Academic Table 1 template items.
   * @returns {TemplateLibraryItem[]} Array of template items.
   */
  public buildTemplateItems(): TemplateLibraryItem[] {
    const items: TemplateLibraryItem[] = [];
    buildStarterQueryItems().forEach((starter, idx) => {
      items.push({
        id: `starter-${idx + 1}`,
        category: "starter",
        title: starter.label,
        complexity: "Low",
        question: starter.question,
        goldSql: starter.goldSql,
      });
    });

    TABLE_1_EXAMPLE_QUERIES.forEach((spec) => {
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
   * Binds click and keyboard listeners for opening, filtering, and closing the template modal.
   * @returns {void}
   */
  private bindEvents(): void {
    this.openBtn?.addEventListener("click", () => {
      this.openModal();
    });

    this.closeBtn?.addEventListener("click", () => {
      this.closeModal();
    });

    this.modalEl?.addEventListener("click", (e) => {
      if (e.target === this.modalEl) {
        this.closeModal();
      }
    });

    if (this.modalEl) {
      const filterBtns = this.modalEl.querySelectorAll<HTMLButtonElement>(
        "[data-template-tab]",
      );
      filterBtns.forEach((btn) => {
        btn.addEventListener("click", () => {
          const tab = btn.getAttribute("data-template-tab");
          const nextTab: "all" | "table1" | "starter" =
            tab === "starter" || tab === "table1" ? tab : "all";
          this.setCategoryFilter(nextTab);
        });
      });

      this.onModalKeyDown = (e: KeyboardEvent) => {
        if (this.modalEl && !this.modalEl.classList.contains("hidden")) {
          trapModalFocus(e, this.modalEl, () => this.closeModal());
        }
      };
      this.modalEl.addEventListener("keydown", this.onModalKeyDown);
    }

    this.onGlobalKeyDown = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key === "/") {
        e.preventDefault();
        if (this.modalEl && !this.modalEl.classList.contains("hidden")) {
          this.closeModal();
        } else {
          this.openModal();
        }
      }
    };
    document.addEventListener("keydown", this.onGlobalKeyDown);
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
    if (!this.modalEl) return;
    this.previousFocusEl =
      document.activeElement instanceof HTMLElement &&
      document.activeElement !== document.body
        ? document.activeElement
        : null;
    this.renderTemplateList();
    this.modalEl.classList.remove("hidden");
    this.modalEl.setAttribute("aria-hidden", "false");
    this.closeBtn?.focus();
  }

  /**
   * Closes the Template Picker modal and restores previous keyboard focus.
   * @returns {void}
   */
  public closeModal(): void {
    if (!this.modalEl) return;
    this.modalEl.classList.add("hidden");
    this.modalEl.setAttribute("aria-hidden", "true");
    if (this.previousFocusEl) {
      this.previousFocusEl.focus();
      this.previousFocusEl = null;
    } else {
      this.ui.chatInput.focus();
    }
  }

  /**
   * Applies a template in Natural-Language-to-SQL mode, populating the natural-language question
   * into `#chat-input`, selecting an LLM model, and attaching `goldSql` for comparison.
   * @param {TemplateLibraryItem} item - Selected template specification.
   * @returns {void}
   */
  public applyTemplateAsNaturalLanguage(item: TemplateLibraryItem): void {
    const targetLlm = resolvePreferredLlmModel(this.ui.modelSelect);
    this.ui.modelSelect.value = targetLlm;
    this.ui.state.setActiveChatModel(targetLlm);
    this.ui.state.setActiveGoldSql(item.goldSql);
    this.ui.chatInput.value = item.question;
    this.ui.syncHighlight();
    this.closeModal();
    this.ui.chatInput.focus();
  }

  /**
   * Applies a template in Literal Gold SQL mode, populating `goldSql` into `#chat-input`
   * and switching the model selector to `RAW_SQL_MODEL`.
   * @param {TemplateLibraryItem} item - Selected template specification.
   * @returns {void}
   */
  public applyTemplateAsGoldSql(item: TemplateLibraryItem): void {
    this.ui.modelSelect.value = RAW_SQL_MODEL;
    this.ui.state.setActiveChatModel(RAW_SQL_MODEL);
    this.ui.state.setActiveGoldSql(null);
    this.ui.chatInput.value = `-- ${item.question}
${item.goldSql}`;
    this.ui.syncHighlight();
    this.closeModal();
    this.ui.chatInput.focus();
  }

  /**
   * Immediately dispatches a benchmark turn sending the template's natural-language question
   * and `goldSql` to the selected target LLM and executing both on the DuckDB read-only cursor.
   * @param {TemplateLibraryItem} item - Selected template specification.
   * @returns {Promise<void>}
   */
  public async runTemplateBenchmark(item: TemplateLibraryItem): Promise<void> {
    const targetLlm = resolvePreferredLlmModel(this.ui.modelSelect);
    this.ui.modelSelect.value = targetLlm;
    this.ui.state.setActiveChatModel(targetLlm);
    this.closeModal();
    await this.ui.handleSendMessage(item.question, targetLlm, item.goldSql);
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
    if (this.onModalKeyDown && this.modalEl) {
      this.modalEl.removeEventListener("keydown", this.onModalKeyDown);
      this.onModalKeyDown = null;
    }
    if (this.onGlobalKeyDown) {
      document.removeEventListener("keydown", this.onGlobalKeyDown);
      this.onGlobalKeyDown = null;
    }
  }
}
