/**
 * @file sse-stream.ts
 * @module sse-stream
 * Typed Server-Sent Events (SSE) stream reader and token accumulator for `/api/chat/stream`.
 */

import {
  fetchWithBackendError,
  parseResponseJson,
  pollChatJobUntilComplete,
  submitChatJob,
  type PollChatJobOptions,
} from "./api-client";
import { type ChatTurnResponse } from "./api-types";

export { type ChatTurnResponse };

/**
 * Outcome returned after consuming an SSE chat response stream.
 */
export interface SseStreamResult {
  /** Whether a terminal `done` or `result` SSE event was received. */
  streamUsed: boolean;
  /** Accumulated token text across all `token` events. */
  streamText: string;
  /** Terminal event payload when `streamUsed` is true. */
  finalData: ChatTurnResponse | null;
}

const SSE_FRAME_DELIM_RE = new RegExp("\\r?\\n\\r?\\n");

/**
 * Parses a raw JSON string from an SSE `data:` frame into a structured `ChatTurnResponse` object.
 * @param {string} rawJson - Raw JSON payload string following `data: `.
 * @returns {ChatTurnResponse} Parsed SSE event payload object.
 */
export function parseSseEventPayload(rawJson: string): ChatTurnResponse {
  const parsed: ChatTurnResponse = JSON.parse(rawJson);
  return parsed;
}

/**
 * Consumes a `ReadableStreamDefaultReader<Uint8Array>` from `/api/chat/stream`,
 * invoking `onToken` as tokens arrive and returning the terminal payload.
 * @param {ReadableStreamDefaultReader<Uint8Array>} reader - Stream reader from `Response.body.getReader()`.
 * @param {(accumulatedText: string) => void} onToken - Callback invoked with accumulated text on each token event.
 * @returns {Promise<SseStreamResult>} Accumulated text and terminal result payload.
 */
export async function consumeSseChatStream(
  reader: ReadableStreamDefaultReader<Uint8Array>,
  onToken: (accumulatedText: string) => void,
): Promise<SseStreamResult> {
  const decoder = new TextDecoder();
  let buffer = "";
  let streamText = "";
  let streamUsed = false;
  let finalData: ChatTurnResponse | null = null;

  /**
   * Parses a single SSE frame and accumulates `token` or terminal `result`/`done`/`error` events.
   * @param {string} rawFrame - Raw SSE frame block.
   * @returns {void}
   */
  const processFrame = (rawFrame: string): void => {
    const line = rawFrame.trim();
    if (!line.startsWith("data: ")) return;
    try {
      const eventData = parseSseEventPayload(line.slice(6));
      if (eventData.event === "token" && eventData.token) {
        streamText += eventData.token;
        onToken(streamText);
      } else if (
        eventData.event === "result" ||
        eventData.event === "done" ||
        eventData.event === "error"
      ) {
        finalData = eventData;
        streamUsed = true;
      }
    } catch {
      // ignore JSON parse errors on partial chunks
    }
  };

  while (true) {
    const { done, value } = await reader.read();
    if (done) {
      buffer += decoder.decode();
      if (buffer.trim()) {
        processFrame(buffer);
      }
      break;
    }
    buffer += decoder.decode(value, { stream: true });
    const parts = buffer.split(SSE_FRAME_DELIM_RE);
    buffer = String(parts.pop());
    for (const part of parts) {
      processFrame(part);
    }
  }

  return { streamUsed, streamText, finalData };
}

/**
 * Alias for `consumeSseChatStream` reading `/api/chat/stream` SSE frames and accumulating token deltas.
 */
export const consumeChatSseStream = consumeSseChatStream;

/**
 * Options for dispatching a chat request via SSE (`/api/chat/stream`), async polling (`/api/chat/jobs`), or batch (`/api/chat`).
 */
export interface DispatchChatRequestOptions {
  /** Serialized JSON request body string. */
  bodyJson: string;
  /** HTTP request headers including optional provider credentials. */
  headers: Record<string, string>;
  /** Whether SSE token streaming is enabled. */
  streamingEnabled: boolean;
  /** Loading indicator DOM node updated with live tokens during SSE streaming or job polling. */
  loadingDiv: HTMLDivElement;
  /** Callback invoked on each streamed token to scroll the message pane. */
  onTokenScroll: () => void;
  /** Optional AbortSignal to cancel in-flight SSE, polling, or batch requests. */
  signal?: AbortSignal;
  /** Optional polling interval in milliseconds (defaults to `800`). */
  pollIntervalMs?: number;
  /** Optional maximum polling timeout in milliseconds (defaults to `120000`). */
  pollTimeoutMs?: number;
  /** Whether to dispatch via `POST /api/chat/jobs` before `/api/chat`. */
  useAsyncJobEndpoint?: boolean;
}

/**
 * Outcome returned by `dispatchChatRequest`.
 */
export interface DispatchChatOutcome {
  /** Terminal chat response payload from SSE, async polling, or batch `/api/chat`. */
  data: ChatTurnResponse;
  /** Accumulated token text fallback when `data.content` is empty. */
  fallbackText: string;
}

/**
 * Polls a pending chat job ticket until completion while updating `loadingDiv` with live `partial_content`.
 * @param {string} jobId - Pending job ID (`chatjob-...`).
 * @param {DispatchChatRequestOptions} options - Dispatch options with `loadingDiv` and `signal`.
 * @returns {Promise<DispatchChatOutcome>} Completed chat response outcome.
 */
async function resolvePendingChatJob(
  jobId: string,
  options: DispatchChatRequestOptions,
): Promise<DispatchChatOutcome> {
  let latestPartial = "";
  let contentEl: HTMLDivElement | null = null;
  const pollOpts: PollChatJobOptions = {
    ...(options.pollIntervalMs !== undefined
      ? { intervalMs: options.pollIntervalMs }
      : {}),
    ...(options.pollTimeoutMs !== undefined
      ? { timeoutMs: options.pollTimeoutMs }
      : {}),
    ...(options.signal ? { signal: options.signal } : {}),
    onProgress: (partialText: string): void => {
      latestPartial = partialText;
      if (!contentEl) {
        contentEl = document.createElement("div");
        contentEl.className = "message-content streaming-content";
        options.loadingDiv.replaceChildren(contentEl);
      }
      contentEl.textContent = partialText;
      options.onTokenScroll();
    },
  };
  const completed = await pollChatJobUntilComplete(jobId, pollOpts);
  return {
    data: completed,
    fallbackText: completed.partial_content || latestPartial,
  };
}

/**
 * Builds a POST `RequestInit` object with optional `AbortSignal`.
 * @param {DispatchChatRequestOptions} options - Dispatch options.
 * @returns {RequestInit} Fetch request init.
 */
function buildPostInit(options: DispatchChatRequestOptions): RequestInit {
  return {
    method: "POST",
    headers: options.headers,
    body: options.bodyJson,
    ...(options.signal ? { signal: options.signal } : {}),
  };
}

/**
 * Attempts to fulfill a chat request via SSE (`/api/chat/stream`), returning null to fall back.
 * @param {DispatchChatRequestOptions} options - Dispatch configuration.
 * @returns {Promise<DispatchChatOutcome | null>} SSE outcome when completed, otherwise null.
 */
async function tryDispatchSseStream(
  options: DispatchChatRequestOptions,
): Promise<DispatchChatOutcome | null> {
  if (!options.streamingEnabled) {
    return null;
  }
  try {
    const streamResp = await fetch("/api/chat/stream", buildPostInit(options));

    if (
      streamResp.ok &&
      streamResp.headers?.get("content-type")?.includes("text/event-stream") ===
        true &&
      streamResp.body?.getReader
    ) {
      const reader = streamResp.body.getReader();
      const contentEl = document.createElement("div");
      contentEl.className = "message-content streaming-content";
      options.loadingDiv.replaceChildren(contentEl);

      const sseOutcome = await consumeSseChatStream(reader, (accText) => {
        contentEl.textContent = accText;
        options.onTokenScroll();
      });
      if (sseOutcome.streamUsed && sseOutcome.finalData) {
        return {
          data: sseOutcome.finalData,
          fallbackText: sseOutcome.streamText,
        };
      }
    }
  } catch (err) {
    if (options.signal?.aborted === true) {
      throw err;
    }
  }
  return null;
}

/**
 * Attempts to fulfill a chat request via `POST /api/chat/jobs`, returning null to fall back.
 * @param {DispatchChatRequestOptions} options - Dispatch configuration.
 * @returns {Promise<DispatchChatOutcome | null>} Async job outcome when completed, otherwise null.
 */
async function tryDispatchAsyncJob(
  options: DispatchChatRequestOptions,
): Promise<DispatchChatOutcome | null> {
  if (options.useAsyncJobEndpoint !== true) {
    return null;
  }
  try {
    const ticket = await submitChatJob(
      options.bodyJson,
      options.headers,
      options.signal,
    );
    if (ticket.status === "pending" && ticket.job_id) {
      return resolvePendingChatJob(ticket.job_id, options);
    }
    return { data: ticket, fallbackText: ticket.partial_content || "" };
  } catch (err) {
    if (options.signal?.aborted === true) {
      throw err;
    }
  }
  return null;
}

/**
 * Dispatches a chat message via `/api/chat/stream` when `streamingEnabled` is true,
 * or via `/api/chat/jobs` / `/api/chat` with automatic async job polling when `status === "pending"`.
 * @param {DispatchChatRequestOptions} options - Dispatch configuration.
 * @returns {Promise<DispatchChatOutcome>} Resolved chat response payload and fallback text.
 */
export async function dispatchChatRequest(
  options: DispatchChatRequestOptions,
): Promise<DispatchChatOutcome> {
  const sseOutcome = await tryDispatchSseStream(options);
  if (sseOutcome) {
    return sseOutcome;
  }

  const jobOutcome = await tryDispatchAsyncJob(options);
  if (jobOutcome) {
    return jobOutcome;
  }

  const response = await fetchWithBackendError(
    "/api/chat",
    buildPostInit(options),
  );
  const data = await parseResponseJson<ChatTurnResponse>(response);
  if (data.status === "pending" && data.job_id) {
    return resolvePendingChatJob(data.job_id, options);
  }
  return { data, fallbackText: data.partial_content || "" };
}
