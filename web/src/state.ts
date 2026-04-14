/**
 * @file state.ts
 * Manages the application state for the chat UI.
 */

/**
 * Represents a single chat message.
 */
export interface Message {
  /** The role of the message sender. */
  role: "user" | "assistant";
  /** The textual content of the message. */
  content: string;
  /** Optional tabular data returned from a SQL query execution. */
  sqlResult?: Record<string, string | number | boolean | null>[];
  /** Optional SQL query string generated or executed. */
  sqlQuery?: string;
  /** Optional flag indicating this message represents an error. */
  isError?: boolean;
  /** Optional model identifier used to process this message. */
  model?: string;
}

/**
 * Represents a chat session.
 */
export interface Chat {
  /** Unique identifier for the chat. */
  id: string;
  /** Display title for the chat. */
  title: string;
  /** The sequence of messages in the chat. */
  messages: Message[];
  /** The model identifier used in this chat. */
  model: string;
  /** Whether the chat is a temporary placeholder. */
  isTemporary?: boolean;
}

/**
 * State manager for the Chat Application.
 */
export class ChatState {
  /** Collection of all chats. */
  public chats: Chat[] = [];
  /** Identifier of the currently active chat, or null if none. */
  public activeChatId: string | null = null;
  /** Counter used for auto-generating chat titles. */
  public chatCounter: number = 1;

  constructor() {
    this.loadFromLocalStorage();
  }

  /**
   * Saves the current state to localStorage.
   */
  private saveToLocalStorage(): void {
    try {
      if (typeof localStorage === "undefined") return;
      const data = {
        chats: this.chats,
        activeChatId: this.activeChatId,
        chatCounter: this.chatCounter,
      };
      localStorage.setItem("t1d_analytics_chats", JSON.stringify(data));
    } catch (e) {
      console.warn("Could not save to localStorage", e);
    }
  }

  /**
   * Loads state from localStorage.
   */
  private loadFromLocalStorage(): void {
    try {
      if (typeof localStorage === "undefined") return;
      const data = localStorage.getItem("t1d_analytics_chats");
      if (data) {
        const parsed = JSON.parse(data);
        this.chats = parsed.chats || [];
        this.activeChatId = parsed.activeChatId || null;
        this.chatCounter = parsed.chatCounter || 1;
      }
    } catch (e) {
      console.warn("Could not load from localStorage", e);
    }
  }

  /**
   * Creates a new chat and sets it as active.

   * @param {string} [title] Optional title for the chat.
   * @param {boolean} [isTemporary=false] Whether the chat should be treated as temporary.
   * @returns {Chat} The newly created chat.
   */
  createChat(title?: string, isTemporary: boolean = false): Chat {
    const chat: Chat = {
      id: `chat-${Date.now()}-${Math.floor(Math.random() * 1000)}`,
      title: title || `Chat #${this.chatCounter++}`,
      messages: [],
      model: "gemma4",
      isTemporary,
    };
    this.chats.push(chat);
    this.activeChatId = chat.id;
    this.saveToLocalStorage();
    return chat;
  }

  /**
   * Retrieves the currently active chat.
   * @returns {Chat | null} The active chat or null if none is active.
   */
  getActiveChat(): Chat | null {
    return this.chats.find((c) => c.id === this.activeChatId) || null;
  }

  /**
   * Sets the active chat by ID.
   * @param {string} id The ID of the chat to make active.
   */
  setActiveChat(id: string): void {
    if (this.chats.some((c) => c.id === id)) {
      this.activeChatId = id;
      this.saveToLocalStorage();
    }
  }

  /**
   * Deletes a chat by ID. If the active chat is deleted, clears the active chat.
   * @param {string} id The ID of the chat to delete.
   */
  deleteChat(id: string): void {
    const chat = this.chats.find((c) => c.id === id);
    if (chat && chat.isTemporary) {
      return;
    }
    this.chats = this.chats.filter((c) => c.id !== id);
    if (this.activeChatId === id) {
      this.activeChatId =
        this.chats.length > 0 ? this.chats[this.chats.length - 1].id : null;
    }

    if (this.chats.length === 0) {
      this.createChat("Temporary chat", true);
    }
    this.saveToLocalStorage();
  }

  /**
   * Renames a chat.
   * @param {string} id The ID of the chat to rename.
   * @param {string} newTitle The new title.
   */
  renameChat(id: string, newTitle: string): void {
    const chat = this.chats.find((c) => c.id === id);
    if (chat && newTitle.trim()) {
      chat.title = newTitle.trim();
      this.saveToLocalStorage();
    }
  }

  /**
   * Duplicates a chat, copying its messages and model, and appends " (Copy)" to the title.
   * @param {string} id The ID of the chat to duplicate.
   * @returns {Chat | null} The duplicated chat, or null if the original wasn't found.
   */
  duplicateChat(id: string): Chat | null {
    const original = this.chats.find((c) => c.id === id);
    if (!original) {
      return null;
    }

    const duplicate: Chat = {
      id: `chat-${Date.now()}-${Math.floor(Math.random() * 1000)}`,
      title: `${original.title} (Copy)`,
      messages: JSON.parse(JSON.stringify(original.messages)), // Deep copy
      model: original.model,
      isTemporary: false,
    };
    this.chats.push(duplicate);
    this.activeChatId = duplicate.id;
    this.saveToLocalStorage();
    return duplicate;
  }

  /**
   * Adds a message to the active chat.
   * @param {Message} message The message to add.
   */
  addMessageToActiveChat(message: Message): void {
    const chat = this.getActiveChat();
    if (chat) {
      chat.messages.push(message);
      if (chat.isTemporary) {
        chat.isTemporary = false;
        chat.title = `Chat #${this.chatCounter++}`;
      }
      this.saveToLocalStorage();
    }
  }

  /**
   * Sets the model for the active chat.
   * @param {string} model The model identifier.
   */
  setActiveChatModel(model: string): void {
    const chat = this.getActiveChat();
    if (chat) {
      chat.model = model;
      this.saveToLocalStorage();
    }
  }
}
