/**
 * @file sse-stream.ts
 * @module sse-stream
 * Typed Server-Sent Events (SSE) stream reader and token accumulator for `/api/chat/stream`.
 */

import { type SqlCellValue } from "./formatters";
import { type ApiErrorInput } from "./sql-highlight";

/**
 * Structured payload emitted by the `/api/chat/stream` `done` or `result` event.
 */
export interface SseFinalChatData {
  /** Event discriminator (`token`, `done`, `result`, `sql_generated`, `execution_progress`, `error`). */
  event?: string;
  /** Streamed token delta string when `event === "token"`. */
  token?: string;
  /** Final assistant text or i18n key. */
  content?: string;
  /** Executed SQL result rows. */
  sqlResult?: Record<string, SqlCellValue>[];
  /** Generated or executed SQL query string. */
  sqlQuery?: string;
  /** Structured API error detail when execution or translation fails. */
  error?: ApiErrorInput;
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

const DOUBLE_LF = String.fromCharCode(10) + String.fromCharCode(10);

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

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const parts = buffer.split(DOUBLE_LF);
    buffer = parts.pop() || "";
    for (const part of parts) {
      const line = part.trim();
      if (line.startsWith("data: ")) {
        try {
          const eventData = parseSseEventPayload(line.slice(6));
          if (eventData.event === "token" && eventData.token) {
            streamText += eventData.token;
            onToken(streamText);
          } else if (
            eventData.event === "result" ||
            eventData.event === "done"
          ) {
            finalData = eventData;
            streamUsed = true;
          }
        } catch {
          // ignore JSON parse errors on partial chunks
        }
      }
    }
  }

  return { streamUsed, streamText, finalData };
}
