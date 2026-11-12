/**
 * @file sse-stream.test.ts
 * @module tests/sse-stream.test
 * Unit tests for async chat job submission, polling, timeout/abort handling, and SSE stream dispatch.
 */

import { describe, it, expect, beforeEach, vi, afterEach } from "vitest";
import { pollChatJobUntilComplete, submitChatJob } from "../src/api-client";
import { consumeChatSseStream, dispatchChatRequest } from "../src/sse-stream";

describe("Async Chat Job Polling & SSE Stream Dispatch", () => {
  let mockFetch: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    mockFetch = vi.fn();
    vi.stubGlobal("fetch", mockFetch);
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("submits chat jobs and polls through pending states to completion", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        job_id: "chatjob-100",
        status: "pending",
      }),
    });
    const ticket = await submitChatJob(
      JSON.stringify({ message: "test", model: "gemma:2b" }),
      { "Content-Type": "application/json" },
    );
    expect(ticket.job_id).toBe("chatjob-100");
    expect(ticket.status).toBe("pending");

    const progressUpdates: string[] = [];
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        job_id: "chatjob-100",
        status: "pending",
        partial_content: "Thinking...",
      }),
    });
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        job_id: "chatjob-100",
        status: "pending",
        content: "SELECT * FROM patients;",
      }),
    });
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        job_id: "chatjob-100",
        status: "complete",
        content: "Done!",
        sqlQuery: "SELECT * FROM patients;",
        sqlResult: [{ id: 1 }],
      }),
    });

    const completed = await pollChatJobUntilComplete("chatjob-100", {
      intervalMs: 1,
      timeoutMs: 5000,
      onProgress: (text) => progressUpdates.push(text),
    });
    expect(completed.status).toBe("complete");
    expect(completed.content).toBe("Done!");
    expect(progressUpdates).toEqual(["Thinking...", "SELECT * FROM patients;"]);

    // Default options call on pollChatJobUntilComplete
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        job_id: "chatjob-default-opts",
        status: "complete",
        content: "Default opts complete",
      }),
    });
    const defCompleted = await pollChatJobUntilComplete("chatjob-default-opts");
    expect(defCompleted.content).toBe("Default opts complete");
  });

  it("handles polling timeout, pre-aborted signals, and immediate completion from submitChatJob", async () => {
    expect(consumeChatSseStream).toBeDefined();

    // Pre-aborted signal on pollChatJobUntilComplete
    const preAborted = new AbortController();
    preAborted.abort();
    await expect(
      pollChatJobUntilComplete("chatjob-preabort", {
        signal: preAborted.signal,
      }),
    ).rejects.toThrow("Request aborted by user");

    // Pre-aborted signal during delayWithAbort
    const midAbort = new AbortController();
    mockFetch.mockImplementationOnce(async () => {
      midAbort.abort();
      return {
        ok: true,
        json: async () => ({
          job_id: "chatjob-midabort",
          status: "pending",
        }),
      };
    });
    await expect(
      pollChatJobUntilComplete("chatjob-midabort", {
        intervalMs: 50,
        signal: midAbort.signal,
      }),
    ).rejects.toThrow("Request aborted by user");

    // Polling timeout when job remains pending past timeoutMs
    mockFetch.mockResolvedValue({
      ok: true,
      json: async () => ({
        job_id: "chatjob-timeout",
        status: "pending",
      }),
    });
    await expect(
      pollChatJobUntilComplete("chatjob-timeout", {
        intervalMs: 1,
        timeoutMs: 0,
      }),
    ).rejects.toThrow("Chat job polling timed out");

    // dispatchChatRequest when submitChatJob returns immediate complete record (with and without partial_content)
    mockFetch.mockReset();
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        job_id: "chatjob-instant",
        status: "complete",
        content: "Instant completion",
        partial_content: "Instant completion",
      }),
    });
    const loadingDiv = document.createElement("div");
    const instantOutcome = await dispatchChatRequest({
      bodyJson: "{}",
      headers: {},
      streamingEnabled: false,
      useAsyncJobEndpoint: true,
      loadingDiv,
      onTokenScroll: () => {},
    });
    expect(instantOutcome.data.content).toBe("Instant completion");
    expect(instantOutcome.fallbackText).toBe("Instant completion");

    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        job_id: "chatjob-instant-empty-partial",
        status: "complete",
        content: "Instant without partial",
      }),
    });
    const instantNoPartial = await dispatchChatRequest({
      bodyJson: "{}",
      headers: {},
      streamingEnabled: false,
      useAsyncJobEndpoint: true,
      loadingDiv,
      onTokenScroll: () => {},
    });
    expect(instantNoPartial.fallbackText).toBe("");

    // dispatchChatRequest with multiple pending progress updates to exercise contentEl reuse
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        job_id: "chatjob-multi-prog",
        status: "pending",
      }),
    });
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        job_id: "chatjob-multi-prog",
        status: "pending",
        partial_content: "Chunk 1",
      }),
    });
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        job_id: "chatjob-multi-prog",
        status: "pending",
        partial_content: "Chunk 1 + Chunk 2",
      }),
    });
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        job_id: "chatjob-multi-prog",
        status: "complete",
        content: "Final!",
        partial_content: "Final!",
      }),
    });
    const multiOutcome = await dispatchChatRequest({
      bodyJson: "{}",
      headers: {},
      streamingEnabled: false,
      useAsyncJobEndpoint: true,
      pollIntervalMs: 1,
      loadingDiv,
      onTokenScroll: () => {},
    });
    expect(multiOutcome.data.content).toBe("Final!");
    expect(multiOutcome.fallbackText).toBe("Final!");

    // dispatchChatRequest when submitChatJob is aborted rethrows immediately
    const jobAbort = new AbortController();
    mockFetch.mockImplementationOnce(async () => {
      jobAbort.abort();
      throw new Error("Aborted submitChatJob");
    });
    await expect(
      dispatchChatRequest({
        bodyJson: "{}",
        headers: {},
        streamingEnabled: false,
        useAsyncJobEndpoint: true,
        pollTimeoutMs: 5000,
        loadingDiv,
        onTokenScroll: () => {},
        signal: jobAbort.signal,
      }),
    ).rejects.toThrow("Aborted submitChatJob");

    // Pending job resolution with explicit pollTimeoutMs and empty partial_content
    mockFetch
      .mockResolvedValueOnce(
        new Response(
          JSON.stringify({ status: "pending", job_id: "chatjob-timeout-opt" }),
          { status: 200, headers: { "Content-Type": "application/json" } },
        ),
      )
      .mockResolvedValueOnce(
        new Response(JSON.stringify({ status: "completed", content: "Done" }), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        }),
      );
    const timeoutOptRes = await dispatchChatRequest({
      bodyJson: "{}",
      headers: {},
      streamingEnabled: false,
      useAsyncJobEndpoint: true,
      pollTimeoutMs: 5000,
      loadingDiv,
      onTokenScroll: () => {},
    });
    expect(timeoutOptRes.data.content).toBe("Done");

    // Batch /api/chat without signal and with signal covers both branches of buildPostInit
    mockFetch
      .mockResolvedValueOnce(
        new Response(JSON.stringify({ content: "Batch without signal" }), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        }),
      )
      .mockResolvedValueOnce(
        new Response(JSON.stringify({ content: "Batch with signal" }), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        }),
      );
    const batchNoSignalRes = await dispatchChatRequest({
      bodyJson: "{}",
      headers: {},
      streamingEnabled: false,
      loadingDiv,
      onTokenScroll: () => {},
    });
    expect(batchNoSignalRes.data.content).toBe("Batch without signal");
    const batchSignalRes = await dispatchChatRequest({
      bodyJson: "{}",
      headers: {},
      streamingEnabled: false,
      loadingDiv,
      onTokenScroll: () => {},
      signal: new AbortController().signal,
    });
    expect(batchSignalRes.data.content).toBe("Batch with signal");
  });
});
