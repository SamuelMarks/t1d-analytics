/**
 * @file sse-stream.ts
 * @module sse-stream
 * Typed Server-Sent Events (SSE) stream reader and token accumulator for `/api/chat/stream`.
 */

import { type SqlCellValue } from "./formatters";
import {
  type ApiErrorInput,
  fetchWithBackendError,
  parseResponseJson,
  pollChatJobUntilComplete,
  submitChatJob,
} from "./api-client";
import { type SqlGoldComparisonPayload } from "./state";

/**
 * Structured payload emitted by the `/api/chat/stream` `done` or `result` event or `/api/chat/jobs/{jobId}` poll.
 */
export interface SseFinalChatData {
  /** Event discriminator (`token`, `done`, `result`, `sql_generated`, `execution_progress`, `error`). */
  event?: string;
  /** Streamed token delta string when `event === "token"`. */
  token?: string;
  /** Async job identifier (`chatjob-...`) when dispatched via `/api/chat/jobs` or `async_poll`. */
  job_id?: string | null;
  /** Lifecycle status (`"pending"`, `"complete"`, `"failed"`). */
  status?: "pending" | "complete" | "failed" | string;
  /** Live accumulated partial text while `"pending"`. */
  partial_content?: string;
  /** Final assistant text or i18n key. */
  content?: string;
  /** Executed SQL result rows. */
  sqlResult?: Record<string, SqlCellValue>[];
  /** Generated or executed SQL query string. */
  sqlQuery?: string;
  /** Structured API error detail when execution or translation fails. */
  error?: ApiErrorInput;
  /** Whether the queried DuckDB database uses synthetic seed data rather than authentic Jaeb trial data. */
  is_synthetic_data?: boolean;
  /** Whether the SQL was generated via deterministic rule-based templates instead of an LLM. */
  is_rule_based_sql?: boolean;
  /** SQL generation mode ('llm', 'rule_based_fallback', 'literal_sql'). */
  generation_mode?: string;
  /** Optional human-readable diagnostic reason why LLM inference failed prior to fallback synthesis. */
  llm_error?: string | null;
  /** Optional structured error code for the underlying LLM failure. */
  llm_error_code?: string | null;
  /** Database provenance classification ('jaeb_authentic', 'synthetic_seed', 'custom'). */
  data_provenance?: string;
  /** Optional gold-standard SQL comparison metrics returned by the backend. */
  gold_comparison?: SqlGoldComparisonPayload | null;
}

/**
 * Outcome returned after consuming an SSE chat response stream.
 */
export interface SseStreamResult {
  /** Whether a terminal `done` or `result` SSE event was received. */
  streamUsed: boolean;
  /** Accumulated token text across all `token` events. */
  streamText: string;
  /** Terminal event payload when `streamUsed` is true. */
  finalData: SseFinalChatData | null;
}

const SSE_FRAME_DELIM_RE = /\r?\n\r?\n/;

/**
 * Parses a raw JSON string from an SSE `data:` frame into a structured `SseFinalChatData` object.
 * @param {string} rawJson - Raw JSON payload string following `data: `.
 * @returns {SseFinalChatData} Parsed SSE event payload object.
 */
export function parseSseEventPayload(rawJson: string): SseFinalChatData {
  const parsed: SseFinalChatData = JSON.parse(rawJson);
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
  let finalData: SseFinalChatData | null = null;

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
    buffer = parts.pop() || "";
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
  data: SseFinalChatData;
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
  const completed = await pollChatJobUntilComplete(jobId, {
    intervalMs: options.pollIntervalMs,
    timeoutMs: options.pollTimeoutMs,
    signal: options.signal,
    onProgress: (partialText) => {
      latestPartial = partialText;
      if (!contentEl) {
        contentEl = document.createElement("div");
        contentEl.className = "message-content streaming-content";
        options.loadingDiv.replaceChildren(contentEl);
      }
      contentEl.textContent = partialText;
      options.onTokenScroll();
    },
  });
  return {
    data: completed,
    fallbackText: completed.partial_content || latestPartial,
  };
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
  if (options.streamingEnabled) {
    try {
      const streamResp = await fetch("/api/chat/stream", {
        method: "POST",
        headers: options.headers,
        body: options.bodyJson,
        signal: options.signal,
      });

      if (
        streamResp.ok &&
        streamResp.headers
          ?.get("content-type")
          ?.includes("text/event-stream") &&
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
      if (options.signal?.aborted) {
        throw err;
      }
      // Fall back to async job or batch /api/chat below
    }
  }

  if (options.useAsyncJobEndpoint) {
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
      if (options.signal?.aborted) {
        throw err;
      }
      // Fall back to /api/chat if /api/chat/jobs is unmocked or returns 404
    }
  }

  const response = await fetchWithBackendError("/api/chat", {
    method: "POST",
    headers: options.headers,
    body: options.bodyJson,
    signal: options.signal,
  });
  const data = await parseResponseJson<SseFinalChatData>(response);
  if (data.status === "pending" && data.job_id) {
    return resolvePendingChatJob(data.job_id, options);
  }
  return { data, fallbackText: data.partial_content || "" };
}
