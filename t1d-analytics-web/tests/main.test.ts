/**
 * @file main.test.ts
 * @module tests/main.test
 * Behavioral tests for `main.ts` against the real `index.html` DOM and mocked `fetch` boundary.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { flushPromises, mountChatUiTestDom } from "./setup";

describe("main.ts", () => {
  const fetchCalls: string[] = [];

  beforeEach(() => {
    vi.resetModules();
    fetchCalls.length = 0;
    localStorage.clear();
    sessionStorage.clear();
    mountChatUiTestDom();

    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url =
        typeof input === "string"
          ? input
          : input instanceof URL
            ? input.toString()
            : input.url;
      fetchCalls.push(url);
      if (url.includes("/api/databases")) {
        return new Response(
          JSON.stringify({
            databases: [
              { name: "t1d.duckdb", path: "t1d.duckdb", size_bytes: 4096 },
            ],
            current_db: "t1d.duckdb",
          }),
          { status: 200, headers: { "Content-Type": "application/json" } },
        );
      }
      if (url.includes("/api/status")) {
        return new Response(
          JSON.stringify({
            status: "healthy",
            database: {
              exists: true,
              connected: true,
              status_code: "healthy",
              table_count: 1,
              has_initial_data: true,
            },
            ollama: { accessible: true },
          }),
          { status: 200, headers: { "Content-Type": "application/json" } },
        );
      }
      return new Response(
        JSON.stringify({
          models: [{ name: "gemma:2b" }],
          tables: [],
          templates: [],
        }),
        { status: 200, headers: { "Content-Type": "application/json" } },
      );
    });
  });

  afterEach(() => {
    window.dispatchEvent(new Event("pagehide"));
    vi.restoreAllMocks();
  });

  it("creates an initial chat on empty storage, invokes fetchDatabases(), and disposes on pagehide", async () => {
    await import("../src/main");
    await flushPromises();

    const chatItems = document.querySelectorAll("#chat-list .chat-item");
    expect(chatItems.length).toBe(1);
    expect(fetchCalls.some((u) => u.includes("/api/databases"))).toBe(true);
    expect(document.querySelectorAll("#db-select option").length).toBe(1);

    window.dispatchEvent(new Event("pagehide"));
  });

  it("preserves existing localStorage chats without creating a duplicate chat on startup", async () => {
    localStorage.setItem(
      "t1d_analytics_chats",
      JSON.stringify({
        activeChatId: "persisted-chat-1",
        chats: [
          {
            id: "persisted-chat-1",
            title: "Saved Clinical Cohort Review",
            model: "sql",
            messages: [
              {
                role: "user",
                content: "SELECT 1;",
                model: "sql",
              },
            ],
            createdAt: 1700000000000,
            updatedAt: 1700000000000,
          },
        ],
      }),
    );

    const { bootstrapApp } = await import("../src/main");
    await flushPromises();

    const chatTitles = Array.from(
      document.querySelectorAll("#chat-list .chat-item-title"),
    ).map((el) => el.textContent);
    expect(chatTitles).toEqual(["Saved Clinical Cohort Review"]);

    window.dispatchEvent(new Event("pagehide"));

    // Explicit bootstrapApp() call also works cleanly on already-populated storage
    const ui2 = bootstrapApp();
    await flushPromises();
    expect(ui2.state.chats.length).toBe(1);
    ui2.dispose();
  });
});
