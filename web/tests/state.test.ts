import { describe, it, expect, beforeEach } from "vitest";
import { ChatState } from "../src/state";

describe("ChatState", () => {
  let state: ChatState;

  beforeEach(() => {
    state = new ChatState();
  });

  it("initializes empty state", () => {
    expect(state.chats).toEqual([]);
    expect(state.activeChatId).toBeNull();
    expect(state.getActiveChat()).toBeNull();
  });

  it("creates a new chat and sets it as active", () => {
    const chat = state.createChat();
    expect(state.chats.length).toBe(1);
    expect(state.activeChatId).toBe(chat.id);
    expect(chat.title).toBe("Chat #1");
    expect(chat.model).toBe("gemma4");
  });

  it("retrieves the active chat", () => {
    state.createChat();
    const chat2 = state.createChat();
    expect(state.getActiveChat()).toEqual(chat2);
  });

  it("sets the active chat by id", () => {
    const chat1 = state.createChat();
    state.createChat();
    state.setActiveChat(chat1.id);
    expect(state.activeChatId).toBe(chat1.id);
    expect(state.getActiveChat()).toEqual(chat1);
  });

  it("ignores setting active chat to invalid id", () => {
    const chat = state.createChat();
    state.setActiveChat("invalid-id");
    expect(state.activeChatId).toBe(chat.id);
  });

  it("deletes a chat and updates active chat if necessary", () => {
    const chat1 = state.createChat();
    const chat2 = state.createChat();

    // Delete non-active chat
    state.deleteChat(chat1.id);
    expect(state.chats.length).toBe(1);
    expect(state.activeChatId).toBe(chat2.id);

    // Delete active chat with no remaining chats
    state.deleteChat(chat2.id);
    expect(state.chats.length).toBe(1); // Temporary chat created!
    expect(state.chats[0].title).toBe("Temporary chat");
  });

  it("deletes active chat and sets next available chat as active", () => {
    state.createChat();
    const chat2 = state.createChat();
    const chat3 = state.createChat();
    // currently chat3 is active

    state.deleteChat(chat3.id);
    expect(state.chats.length).toBe(2);
    // Should fallback to chat2 (the last one remaining)
    expect(state.activeChatId).toBe(chat2.id);
  });

  it("renames a chat", () => {
    const chat = state.createChat();
    state.renameChat(chat.id, "My Custom Title");
    expect(state.chats[0].title).toBe("My Custom Title");
  });

  it("does not rename chat if title is empty or chat not found", () => {
    const chat = state.createChat();
    state.renameChat(chat.id, "   ");
    expect(state.chats[0].title).toBe("Chat #1");

    state.renameChat("invalid", "New Title");
    expect(state.chats[0].title).toBe("Chat #1");
  });

  it("duplicates a chat", () => {
    const chat = state.createChat();
    state.addMessageToActiveChat({ role: "user", content: "Hello" });

    const duplicate = state.duplicateChat(chat.id);
    expect(duplicate).not.toBeNull();
    expect(duplicate?.title).toBe("Chat #1 (Copy)");
    expect(duplicate?.messages.length).toBe(1);
    expect(state.chats.length).toBe(2);
    expect(state.activeChatId).toBe(duplicate?.id);

    // Ensure deep copy
    state.addMessageToActiveChat({ role: "assistant", content: "Hi" });
    expect(duplicate?.messages.length).toBe(2);
    expect(chat.messages.length).toBe(1);
  });

  it("returns null when duplicating non-existent chat", () => {
    expect(state.duplicateChat("invalid")).toBeNull();
  });

  it("adds message to active chat", () => {
    state.createChat();
    state.addMessageToActiveChat({ role: "user", content: "Test message" });
    const chat = state.getActiveChat();
    expect(chat?.messages[0]).toEqual({
      role: "user",
      content: "Test message",
    });
  });

  it("ignores adding message if no active chat", () => {
    state.addMessageToActiveChat({ role: "user", content: "Test" });
    expect(state.chats.length).toBe(0);
  });

  it("sets model for active chat", () => {
    state.createChat();
    state.setActiveChatModel("sql");
    expect(state.getActiveChat()?.model).toBe("sql");
  });

  it("ignores setting model if no active chat", () => {
    state.setActiveChatModel("sql"); // should not throw
  });

  it("ignores deleting a temporary chat", () => {
    const chat = state.createChat("Temporary chat", true);
    state.deleteChat(chat.id);
    expect(state.chats.length).toBe(1);
  });

  it("adding message to temporary chat sets isTemporary to false and updates title", () => {
    const chat = state.createChat("Temporary chat", true);
    state.addMessageToActiveChat({ role: "user", content: "hello" });
    expect(chat.isTemporary).toBe(false);
    expect(chat.title).toBe("Chat #1");
  });
});
