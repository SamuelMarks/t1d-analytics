/**
 * @file main.test.ts
 * @module tests/main.test
 * Unit tests for application entrypoint initialization.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";

describe("main.ts", () => {
  beforeEach(() => {
    vi.resetModules();
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  it("initializes ChatState and ChatUI, and creates a chat if none exist", async () => {
    const createChatSpy = vi.fn();
    const stateObj = {
      chats: [],
      createChat: createChatSpy,
    };
    vi.doMock("../src/state", () => {
      return {
        ChatState: vi.fn().mockImplementation(() => stateObj),
      };
    });
    vi.doMock("../src/ui", () => ({ ChatUI: vi.fn() }));
    vi.doMock("../src/i18n", () => ({ translateDocument: vi.fn() }));

    const { ChatState } = await import("../src/state");
    const { ChatUI } = await import("../src/ui");
    await import("../src/main");

    expect(ChatState).toHaveBeenCalledTimes(1);
    expect(createChatSpy).toHaveBeenCalledTimes(1);
    expect(ChatUI).toHaveBeenCalledTimes(1);
    expect(ChatUI).toHaveBeenCalledWith(stateObj);
  });

  it("does not create a chat if one already exists", async () => {
    const createChatSpy = vi.fn();
    vi.doMock("../src/state", () => {
      return {
        ChatState: vi.fn().mockImplementation(() => {
          return {
            chats: [{ id: "existing" }],
            createChat: createChatSpy,
          };
        }),
      };
    });
    vi.doMock("../src/ui", () => ({ ChatUI: vi.fn() }));
    vi.doMock("../src/i18n", () => ({ translateDocument: vi.fn() }));

    await import("../src/main");

    expect(createChatSpy).not.toHaveBeenCalled();
  });

  it("integrates real ChatState and ChatUI against index.html DOM fixture on startup", async () => {
    const fs = await import("fs");
    const path = await import("path");
    const html = fs.readFileSync(
      path.resolve(__dirname, "../index.html"),
      "utf-8",
    );
    document.documentElement.innerHTML = html;
    localStorage.clear();

    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify({ models: [], tables: [], databases: [] }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    );

    vi.doUnmock("../src/state");
    vi.doUnmock("../src/ui");
    vi.doUnmock("../src/i18n");

    await import("../src/main");

    const chatItems = document.querySelectorAll("#chat-list .chat-item");
    expect(chatItems.length).toBe(1);
  }, 20000);
});
