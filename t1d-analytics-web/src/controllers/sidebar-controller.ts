/**
 * @file sidebar-controller.ts
 * @module controllers/sidebar-controller
 * Composable sub-controller managing the Sidebar chat history list, accessible dropdown
 * action menus, keyboard focus navigation, mobile drawer overlay, and chat rename/delete dialogs.
 */

import i18next from "i18next";
import { requireElementById, trapModalFocus } from "../modal-utils";
import { type Chat, type ChatState } from "../state";

/**
 * Accessible dialog abstraction for chat rename prompts and delete confirmations.
 */
export interface SidebarDialogService {
  /**
   * Prompts the user for a new chat title.
   * @param {string} message - Prompt label.
   * @param {string} defaultValue - Current chat title.
   * @returns {string | null} New title or null if cancelled.
   */
  promptRename(message: string, defaultValue: string): string | null;
  /**
   * Asks the user to confirm deleting a chat session.
   * @param {string} message - Confirmation prompt text.
   * @returns {boolean} True when deletion is confirmed.
   */
  confirmDelete(message: string): boolean;
}

/**
 * Default browser dialog implementation backed by `window.prompt` and `window.confirm`.
 */
export const defaultSidebarDialogService: SidebarDialogService = {
  /**
   * Prompts for a new chat title via `window.prompt`.
   * @param {string} message - Prompt label.
   * @param {string} defaultValue - Current chat title.
   * @returns {string | null} New title or null if cancelled.
   */
  promptRename(message: string, defaultValue: string): string | null {
    return window.prompt(message, defaultValue);
  },
  /**
   * Confirms deletion via `window.confirm`.
   * @param {string} message - Confirmation message.
   * @returns {boolean} True when confirmed.
   */
  confirmDelete(message: string): boolean {
    return window.confirm(message);
  },
};

/**
 * Host contract required by `SidebarController`.
 */
export interface SidebarHost {
  /** Global application state store. */
  state: ChatState;
  /** Textarea element where the user types queries or SQL. */
  chatInput: HTMLTextAreaElement;
  /** Re-renders the entire UI. */
  render(): void;
}

/**
 * Controller managing the sidebar chat list, dropdown menus, and mobile sidebar drawer.
 */
export class SidebarController {
  /** Host UI reference. */
  private readonly ui: SidebarHost;
  /** Sidebar navigation element. */
  public readonly sidebarEl: HTMLElement;
  /** Sidebar list element containing chat sessions. */
  public readonly chatListEl: HTMLElement;
  /** Button element to create a new chat session. */
  public readonly newChatBtn: HTMLButtonElement;
  /** Mobile header button element to open the navigation sidebar. */
  public readonly openSidebarBtn: HTMLButtonElement;
  /** Mobile sidebar button element to close the navigation sidebar. */
  public readonly closeSidebarBtn: HTMLButtonElement;
  /** Backdrop overlay element for the mobile navigation sidebar. */
  public readonly overlay: HTMLElement;
  /** Dialog service used for rename and delete interactions. */
  public dialogService: SidebarDialogService;

  /**
   * Creates a new `SidebarController`.
   * @param {SidebarHost} ui - Host UI instance.
   * @param {AbortSignal} signal - AbortSignal for global document listeners.
   * @param {SidebarDialogService} [dialogService=defaultSidebarDialogService] - Dialog service for rename/delete.
   */
  constructor(
    ui: SidebarHost,
    signal: AbortSignal,
    dialogService: SidebarDialogService = defaultSidebarDialogService,
  ) {
    this.ui = ui;
    this.dialogService = dialogService;
    this.sidebarEl = requireElementById<HTMLElement>("sidebar");
    this.chatListEl = requireElementById<HTMLElement>("chat-list");
    this.newChatBtn = requireElementById<HTMLButtonElement>("new-chat-btn");
    this.openSidebarBtn =
      requireElementById<HTMLButtonElement>("open-sidebar-btn");
    this.closeSidebarBtn =
      requireElementById<HTMLButtonElement>("close-sidebar-btn");
    this.overlay = requireElementById<HTMLElement>("overlay");

    this.bindEvents(signal);
  }

  /**
   * Binds sidebar drawer, new-chat, and outside-click dropdown listeners.
   * @param {AbortSignal} signal - AbortSignal for global document listeners.
   * @returns {void}
   */
  private bindEvents(signal: AbortSignal): void {
    this.newChatBtn.addEventListener("click", () => {
      this.ui.state.createChat();
      const hasCloudKeys = Boolean(
        this.ui.state.getApiKey("openai") ||
        this.ui.state.getApiKey("anthropic") ||
        this.ui.state.getApiKey("google") ||
        this.ui.state.getConfiguredModels().length > 0,
      );
      if (!this.ui.state.systemStatus.ollamaOnline && !hasCloudKeys) {
        this.ui.state.setActiveChatModel("sql");
      }
      this.ui.render();
      this.closeMobileSidebar();
      this.ui.chatInput.focus();
    });

    this.openSidebarBtn.addEventListener("click", () => {
      document.body.classList.add("sidebar-open");
      this.openSidebarBtn.setAttribute("aria-expanded", "true");
      this.closeSidebarBtn.focus();
    });

    const closeSidebar = (): void => {
      this.closeMobileSidebar();
      this.openSidebarBtn.focus();
    };

    this.closeSidebarBtn.addEventListener("click", closeSidebar);
    this.overlay.addEventListener("click", closeSidebar);
    this.sidebarEl.addEventListener("keydown", (e: KeyboardEvent) => {
      if (document.body.classList.contains("sidebar-open")) {
        trapModalFocus(e, this.sidebarEl, closeSidebar);
      }
    });

    document.addEventListener(
      "click",
      (e) => {
        const target = e.target;
        if (target instanceof Element) {
          if (!target.closest(".dropdown-container")) {
            this.closeAllDropdowns();
          }
        } else {
          this.closeAllDropdowns();
        }
      },
      { signal },
    );
  }

  /**
   * Closes the mobile sidebar if it is open.
   * @returns {void}
   */
  public closeMobileSidebar(): void {
    document.body.classList.remove("sidebar-open");
    this.openSidebarBtn.setAttribute("aria-expanded", "false");
  }

  /**
   * Closes all currently open dropdown menus in the UI.
   * @returns {void}
   */
  public closeAllDropdowns(): void {
    document.querySelectorAll(".dropdown-menu.show").forEach((menu) => {
      menu.classList.remove("show");
      const btn = menu.previousElementSibling;
      if (btn && btn.classList.contains("dropdown-btn")) {
        btn.setAttribute("aria-expanded", "false");
      }
    });
  }

  /**
   * Builds a single `.chat-item` list item with title button and accessible action menu.
   * @param {Chat} chat - Chat session model.
   * @returns {HTMLLIElement} Constructed list item element.
   */
  private buildChatListItem(chat: Chat): HTMLLIElement {
    const isActive = chat.id === this.ui.state.activeChatId;
    const li = document.createElement("li");
    li.className = `chat-item ${isActive ? "active" : ""}`;

    const titleBtn = document.createElement("button");
    titleBtn.type = "button";
    titleBtn.className = "chat-item-title";
    titleBtn.textContent = chat.title;
    if (isActive) {
      titleBtn.setAttribute("aria-current", "page");
    }

    const activateChat = (): void => {
      this.ui.state.setActiveChat(chat.id);
      this.ui.render();
      this.closeMobileSidebar();
      this.ui.chatInput.focus();
    };

    titleBtn.addEventListener("click", activateChat);
    titleBtn.addEventListener("dblclick", (e) => {
      e.stopPropagation();
      this.startInlineRename(li, titleBtn, chat);
    });
    titleBtn.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        activateChat();
      }
    });

    const dropdownContainer = document.createElement("div");
    dropdownContainer.className = "dropdown-container";

    const dropdownBtn = document.createElement("button");
    dropdownBtn.className = "dropdown-btn";
    dropdownBtn.textContent = "\u22EE";
    dropdownBtn.setAttribute("aria-label", i18next.t("aria.chatOptions"));
    dropdownBtn.setAttribute("data-i18n-aria-label", "aria.chatOptions");
    dropdownBtn.setAttribute("aria-haspopup", "menu");
    dropdownBtn.setAttribute("aria-expanded", "false");
    dropdownBtn.setAttribute("aria-controls", `dropdown-menu-${chat.id}`);

    const dropdownMenu = document.createElement("div");
    dropdownMenu.id = `dropdown-menu-${chat.id}`;
    dropdownMenu.className = "dropdown-menu";
    dropdownMenu.setAttribute("role", "menu");
    dropdownMenu.setAttribute(
      "aria-label",
      i18next.t("aria.chatActions", "Chat actions"),
    );
    dropdownMenu.setAttribute("data-i18n-aria-label", "aria.chatActions");

    const renameBtn = document.createElement("button");
    renameBtn.className = "dropdown-item";
    renameBtn.textContent = i18next.t("ui.rename");
    renameBtn.setAttribute(
      "aria-label",
      i18next.t("aria.editChatTitle", "Edit chat title"),
    );
    renameBtn.setAttribute("data-i18n-aria-label", "aria.editChatTitle");
    renameBtn.setAttribute("role", "menuitem");
    renameBtn.setAttribute("tabindex", "-1");
    renameBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      this.closeAllDropdowns();
      dropdownBtn.focus();
      const newTitle = this.dialogService.promptRename(
        i18next.t("ui.newChatTitle"),
        chat.title,
      );
      if (newTitle !== null) {
        this.ui.state.renameChat(chat.id, newTitle);
        this.ui.render();
      }
    });

    const duplicateBtn = document.createElement("button");
    duplicateBtn.className = "dropdown-item";
    duplicateBtn.textContent = i18next.t("ui.duplicate");
    duplicateBtn.setAttribute(
      "aria-label",
      i18next.t("aria.duplicateChat", "Duplicate chat"),
    );
    duplicateBtn.setAttribute("data-i18n-aria-label", "aria.duplicateChat");
    duplicateBtn.setAttribute("role", "menuitem");
    duplicateBtn.setAttribute("tabindex", "-1");
    duplicateBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      this.closeAllDropdowns();
      dropdownBtn.focus();
      this.ui.state.duplicateChat(chat.id);
      this.ui.render();
    });

    const deleteBtn = document.createElement("button");
    deleteBtn.className = "dropdown-item danger";
    deleteBtn.textContent = i18next.t("ui.delete");
    deleteBtn.setAttribute(
      "aria-label",
      i18next.t("aria.deleteChat", "Delete chat"),
    );
    deleteBtn.setAttribute("data-i18n-aria-label", "aria.deleteChat");
    deleteBtn.setAttribute("role", "menuitem");
    deleteBtn.setAttribute("tabindex", "-1");
    deleteBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      this.closeAllDropdowns();
      dropdownBtn.focus();
      if (
        this.dialogService.confirmDelete(
          i18next.t("ui.deleteConfirm", { title: chat.title }),
        )
      ) {
        this.ui.state.deleteChat(chat.id);
        this.ui.render();
        this.ui.chatInput.focus();
      }
    });

    const menuItems = [renameBtn, duplicateBtn, deleteBtn];

    dropdownBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      const isShowing = dropdownMenu.classList.contains("show");
      this.closeAllDropdowns();
      if (!isShowing) {
        dropdownMenu.classList.add("show");
        dropdownBtn.setAttribute("aria-expanded", "true");
        renameBtn.focus();
      }
    });

    dropdownBtn.addEventListener("keydown", (e) => {
      if (e.key === "ArrowDown" || e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        dropdownBtn.click();
      } else if (e.key === "ArrowUp") {
        e.preventDefault();
        this.closeAllDropdowns();
        dropdownMenu.classList.add("show");
        dropdownBtn.setAttribute("aria-expanded", "true");
        deleteBtn.focus();
      }
    });

    dropdownMenu.addEventListener("keydown", (e) => {
      const index = menuItems.findIndex(
        (item) => item === document.activeElement,
      );
      if (e.key === "ArrowDown") {
        e.preventDefault();
        menuItems[(index + 1) % menuItems.length].focus();
      } else if (e.key === "ArrowUp") {
        e.preventDefault();
        menuItems[(index - 1 + menuItems.length) % menuItems.length].focus();
      } else if (e.key === "Home") {
        e.preventDefault();
        menuItems[0].focus();
      } else if (e.key === "End") {
        e.preventDefault();
        menuItems[menuItems.length - 1].focus();
      } else if (e.key === "Escape") {
        e.preventDefault();
        this.closeAllDropdowns();
        dropdownBtn.focus();
      }
    });

    dropdownMenu.append(renameBtn, duplicateBtn, deleteBtn);
    dropdownContainer.append(dropdownBtn, dropdownMenu);
    li.append(titleBtn, dropdownContainer);
    return li;
  }

  /**
   * Starts an accessible non-blocking inline rename input inside a chat list item.
   * @param {HTMLLIElement} li - Chat list item element.
   * @param {HTMLButtonElement} titleBtn - Chat title button element.
   * @param {Chat} chat - Chat model to rename.
   * @returns {void}
   */
  public startInlineRename(
    li: HTMLLIElement,
    titleBtn: HTMLButtonElement,
    chat: Chat,
  ): void {
    const input = document.createElement("input");
    input.type = "text";
    input.className = "chat-inline-rename-input";
    input.value = chat.title;
    input.setAttribute(
      "aria-label",
      i18next.t("aria.editChatTitle", "Edit chat title"),
    );

    let committed = false;
    const commitRename = (save: boolean): void => {
      if (committed) return;
      committed = true;
      if (save && input.value.trim()) {
        this.ui.state.renameChat(chat.id, input.value.trim());
      }
      this.ui.render();
    };

    input.addEventListener("keydown", (e: KeyboardEvent) => {
      if (e.key === "Enter") {
        e.preventDefault();
        commitRename(true);
      } else if (e.key === "Escape") {
        e.preventDefault();
        commitRename(false);
      }
    });
    input.addEventListener("blur", () => {
      commitRename(true);
    });

    titleBtn.replaceWith(input);
    input.focus();
    input.select();
    void li;
  }

  /**
   * Renders the sidebar chat list.
   * @returns {void}
   */
  public renderSidebar(): void {
    this.chatListEl.replaceChildren();
    for (const chat of this.ui.state.chats) {
      this.chatListEl.appendChild(this.buildChatListItem(chat));
    }
  }
}
