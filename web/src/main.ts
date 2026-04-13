/**
 * @file main.ts
 * Entry point for the Chat UI application.
 * Initializes the state and instantiates the main UI controller.
 */

import { ChatState } from "./state";
import { ChatUI } from "./ui";

// Initialize the state and UI
const appState = new ChatState();

// Create an initial chat for convenience
appState.createChat();

// Initialize UI
new ChatUI(appState);
