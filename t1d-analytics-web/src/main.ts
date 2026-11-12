/**
 * @file main.ts
 * @module main
 * Entry point for the Chat UI application.
 * Initializes localized document strings, `ChatState`, and `ChatUI` with explicit pagehide disposal.
 */

import { translateDocument } from "./i18n";
import { ChatState } from "./state";
import { ChatUI } from "./ui";

/**
 * Bootstraps the application against the active document, creating an initial chat if none exist
 * and registering explicit `ui.dispose()` cleanup on `pagehide`.
 * @returns {ChatUI} Initialized `ChatUI` coordinator instance.
 */
export function bootstrapApp(): ChatUI {
  translateDocument();

  const appState = new ChatState();
  if (appState.chats.length === 0) {
    appState.createChat();
  }

  const ui = new ChatUI(appState);
  window.addEventListener(
    "pagehide",
    () => {
      ui.dispose();
    },
    { once: true },
  );
  void ui.schemaExplorerController.fetchDatabases();
  void ui.templatePickerController.refreshCatalog();
  return ui;
}

bootstrapApp();
