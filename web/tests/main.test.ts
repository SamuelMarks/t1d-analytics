import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";

describe("main.ts", () => {
  beforeEach(() => {
    vi.resetModules();
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  it("initializes ChatState and ChatUI, and creates a chat if none exist", async () => {
    vi.doMock("../src/state", () => {
      return {
        ChatState: vi.fn().mockImplementation(() => {
          return {
            chats: [],
            createChat: vi.fn(),
          };
        }),
      };
    });
    vi.doMock("../src/ui", () => ({ ChatUI: vi.fn() }));
    vi.doMock("../src/i18n", () => ({ translateDocument: vi.fn() }));

    const { ChatState } = await import("../src/state");
    const { ChatUI } = await import("../src/ui");
    await import("../src/main");

    expect(ChatState).toHaveBeenCalledTimes(1);

    const stateMock = (ChatState as import("vitest").Mock).mock.results[0]
      .value as { createChat: import("vitest").Mock };

    expect(stateMock.createChat).toHaveBeenCalledTimes(1);
    expect(ChatUI).toHaveBeenCalledTimes(1);
    expect(ChatUI).toHaveBeenCalledWith(stateMock);
  });

  it("does not create a chat if one already exists", async () => {
    vi.doMock("../src/state", () => {
      return {
        ChatState: vi.fn().mockImplementation(() => {
          return {
            chats: [{ id: "existing" }],
            createChat: vi.fn(),
          };
        }),
      };
    });
    vi.doMock("../src/ui", () => ({ ChatUI: vi.fn() }));
    vi.doMock("../src/i18n", () => ({ translateDocument: vi.fn() }));

    const { ChatState } = await import("../src/state");
    await import("../src/main");

    const stateMock = (ChatState as import("vitest").Mock).mock.results[0]
      .value as { createChat: import("vitest").Mock };

    expect(stateMock.createChat).not.toHaveBeenCalled();
  });
});
