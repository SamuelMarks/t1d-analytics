import { describe, it, expect, vi, beforeEach } from "vitest";
import { ChatState } from "../src/state";
import { ChatUI } from "../src/ui";

vi.mock("../src/state", () => {
  return {
    ChatState: vi.fn().mockImplementation(() => {
      return {
        chats: [],
        createChat: vi.fn(),
      };
    }),
  };
});

vi.mock("../src/ui", () => {
  return {
    ChatUI: vi.fn(),
  };
});

describe("main.ts", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.resetModules();
  });

  it("initializes ChatState and ChatUI, and creates a chat if none exist", async () => {
    // Import main to run it
    await import("../src/main");

    expect(ChatState).toHaveBeenCalledTimes(1);

    // Get the mocked instance
    const stateMock = (ChatState as import("vitest").Mock).mock.results[0]
      .value as { createChat: import("vitest").Mock };

    expect(stateMock.createChat).toHaveBeenCalledTimes(1);
    expect(ChatUI).toHaveBeenCalledTimes(1);
    expect(ChatUI).toHaveBeenCalledWith(stateMock);
  });
});
