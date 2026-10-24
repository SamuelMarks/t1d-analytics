/**
 * @file main.ts
 * @module main
 * Entry point for the Chat UI application.
 * Initializes the state and instantiates the main UI controller.
 */

import { ChatState } from "./state";
import { ChatUI } from "./ui";
import { translateDocument } from "./i18n";

// Initialize translations
translateDocument();

/** Global application state instance. */
const appState = new ChatState();

// Create an initial chat if there are no existing chats
if (appState.chats.length === 0) {
  appState.createChat();
}

/** Main application UI controller instance. */
const ui = new ChatUI(appState);
void ui.fetchDatabases?.();
