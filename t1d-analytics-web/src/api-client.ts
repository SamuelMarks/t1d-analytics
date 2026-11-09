/**
 * @file api-client.ts
 * @module api-client
 * Typed HTTP fetch wrapper, JSON response parser, and localized backend error translator.
 */

import i18next from "./i18n";
import { type SqlCellValue } from "./formatters";
import { type SqlGoldComparisonPayload } from "./state";

/**
 * Structured API error detail object returned by the backend.
 */
export interface ApiErrorPayload {
  /** Internationalization error code key. */
  error_code: string;
  /** Optional interpolation parameters. */
  params?: Record<string, SqlCellValue>;
}

/**
 * Input value accepted by parseApiError.
 */
export type ApiErrorInput =
  | string
  | number
  | boolean
  | null
  | undefined
  | ApiErrorPayload
  | Record<string, never>;

/**
 * Collapses repeated `"Internal server error: "` prefixes in formatted error messages.
 * @param {string} message - Raw formatted error message.
 * @returns {string} Error message with duplicate `"Internal server error: "` prefixes collapsed.
 */
export function deduplicateErrorPrefix(message: string): string {
  const prefix = "Internal server error: ";
  const doublePrefix = `${prefix}${prefix}`;
  let cleaned = message;
  while (cleaned.toLowerCase().startsWith(doublePrefix.toLowerCase())) {
    cleaned = `${prefix}${cleaned.slice(doublePrefix.length).trimStart()}`;
  }
  return cleaned;
}

/**
 * Parses a backend error string or detail object into a translated string.
 * @param {ApiErrorInput} errorObj - The raw backend error string or structured detail object.
 * @returns {string} Localized or formatted error message string.
 */
export function parseApiError(errorObj: ApiErrorInput): string {
  if (typeof errorObj === "string") {
    const resolved = i18next.exists(errorObj)
      ? String(i18next.t(errorObj))
      : errorObj;
    return deduplicateErrorPrefix(resolved);
  }
  if (errorObj && typeof errorObj === "object" && "error_code" in errorObj) {
    const key = errorObj.error_code;
    const params = errorObj.params;
    const resolved = i18next.exists(key)
      ? String(i18next.t(key, params))
      : `${key}: ${JSON.stringify(params)}`;
    return deduplicateErrorPrefix(resolved);
  }
  return deduplicateErrorPrefix(String(errorObj));
}

/**
 * Backend HTTP error JSON response payload structure.
 */
export interface BackendErrorResponse {
  /** Structured or string error detail returned by FastAPI. */
  detail?: ApiErrorInput;
}

/**
 * Parses a Fetch API Response body as a typed JSON payload.
 * @template T - Expected JSON response structure.
 * @param {Response} response - Fetch API Response object.
 * @returns {Promise<T>} Parsed JSON response payload.
 */
export async function parseResponseJson<T>(response: Response): Promise<T> {
  const data: T = await response.json();
  return data;
}

/**
 * Remote chat session record returned by `/api/sessions`.
 * @template TMsg - Message item type.
 */
export interface RemoteSessionPayload<TMsg = unknown> {
  /** Session identifier. */
  session_id: string;
  /** Session display title. */
  title: string;
  /** Creation ISO timestamp. */
  created_at: string;
  /** Last update ISO timestamp. */
  updated_at: string;
  /** Array of chat messages. */
  messages: TMsg[];
}

/**
 * Fetches persisted chat sessions from `/api/sessions`.
 * @template TMsg - Message item type.
 * @returns {Promise<Array<RemoteSessionPayload<TMsg>>>} Normalized list of remote sessions.
 */
export async function fetchRemoteSessions<TMsg>(): Promise<
  Array<RemoteSessionPayload<TMsg>>
> {
  const resp = await fetch("/api/sessions");
  if (!resp.ok) {
    throw new Error(`Server returned status ${resp.status}`);
  }
  const rawData:
    | Array<RemoteSessionPayload<TMsg>>
    | { sessions?: Array<RemoteSessionPayload<TMsg>> }
    | null = await resp.json();
  return Array.isArray(rawData)
    ? rawData
    : Array.isArray(rawData?.sessions)
      ? rawData.sessions
      : [];
}

/**
 * Fetches a single persisted chat session by ID from `GET /api/sessions/{sessionId}`.
 * @template TMsg - Message item type.
 * @param {string} sessionId - Chat session identifier to retrieve.
 * @returns {Promise<RemoteSessionPayload<TMsg> | null>} Remote session payload, or null if not found.
 */
export async function fetchRemoteSessionById<TMsg>(
  sessionId: string,
): Promise<RemoteSessionPayload<TMsg> | null> {
  const resp = await fetch(`/api/sessions/${encodeURIComponent(sessionId)}`);
  if (!resp.ok) {
    return null;
  }
  const data: RemoteSessionPayload<TMsg> = await resp.json();
  return data;
}

/**
 * Persists a chat session to `/api/sessions` via HTTP POST.
 * @template TMsg - Message item type.
 * @param {string} sessionId - Chat session identifier.
 * @param {string} title - Chat session title.
 * @param {TMsg[]} messages - Ordered array of session messages.
 * @returns {Promise<Response>} Fetch response promise.
 */
export async function pushRemoteSession<TMsg>(
  sessionId: string,
  title: string,
  messages: TMsg[],
): Promise<Response> {
  return fetch("/api/sessions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      session_id: sessionId,
      title,
      messages,
    }),
  });
}

/**
 * Helper to fetch and automatically throw parsed backend errors.
 * @param {string} url - The URL to fetch.
 * @param {RequestInit} [options] - Fetch options.
 * @returns {Promise<Response>} The fetch response.
 */
export async function fetchWithBackendError(
  url: string,
  options?: RequestInit,
): Promise<Response> {
  const response = await fetch(url, options);
  if (!response.ok) {
    let errorMsg = `HTTP ${response.status}`;
    try {
      const errData = await parseResponseJson<BackendErrorResponse>(response);
      if (errData.detail) {
        errorMsg = parseApiError(errData.detail);
      }
    } catch {
      // Ignore JSON parse errors
    }
    throw new Error(errorMsg);
  }
  return response;
}

/**
 * Deletes a persisted chat session on the server via `DELETE /api/sessions/{id}`.
 * @param {string} sessionId - Chat session identifier to delete.
 * @returns {Promise<Response>} Fetch response promise.
 */
export async function deleteRemoteSession(
  sessionId: string,
): Promise<Response> {
  return fetch(`/api/sessions/${encodeURIComponent(sessionId)}`, {
    method: "DELETE",
  });
}

/**
 * Payload returned by `POST /api/chat/jobs` or `GET /api/chat/jobs/{jobId}`.
 */
export interface PollableChatJobData {
  /** Optional SSE event discriminator. */
  event?: string;
  /** Optional streamed token delta. */
  token?: string;
  /** Async job identifier (`chatjob-...`). */
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
 * Options for polling an asynchronous chat job via `pollChatJobUntilComplete`.
 */
export interface PollChatJobOptions {
  /** Polling interval in milliseconds (defaults to `800`). */
  intervalMs?: number;
  /** Maximum total polling duration in milliseconds (defaults to `120000`). */
  timeoutMs?: number;
  /** Optional callback invoked with live `partial_content` while job is `"pending"`. */
  onProgress?: (partialText: string) => void;
  /** Optional `AbortSignal` to cancel polling. */
  signal?: AbortSignal;
}

/**
 * Waits for `ms` milliseconds or rejects immediately if `signal` is aborted.
 * @param {number} ms - Delay duration in milliseconds.
 * @param {AbortSignal} [signal] - Optional abort signal.
 * @returns {Promise<void>} Promise resolving after `ms` milliseconds.
 */
function delayWithAbort(ms: number, signal?: AbortSignal): Promise<void> {
  if (signal?.aborted) {
    return Promise.reject(new Error("Request aborted by user"));
  }
  return new Promise<void>((resolve, reject) => {
    const onAbort = (): void => {
      clearTimeout(timer);
      reject(new Error("Request aborted by user"));
    };
    const timer = setTimeout(() => {
      signal?.removeEventListener("abort", onAbort);
      resolve();
    }, ms);
    signal?.addEventListener("abort", onAbort, { once: true });
  });
}

/**
 * Submits an asynchronous chat job via `POST /api/chat/jobs`.
 * @param {string} bodyJson - Serialized JSON request payload.
 * @param {Record<string, string>} headers - Request headers.
 * @param {AbortSignal} [signal] - Optional abort signal.
 * @returns {Promise<PollableChatJobData>} Initial job ticket or immediate response.
 */
export async function submitChatJob(
  bodyJson: string,
  headers: Record<string, string>,
  signal?: AbortSignal,
): Promise<PollableChatJobData> {
  const response = await fetchWithBackendError("/api/chat/jobs", {
    method: "POST",
    headers,
    body: bodyJson,
    signal,
  });
  return parseResponseJson<PollableChatJobData>(response);
}

/**
 * Polls `GET /api/chat/jobs/{jobId}` until `status !== "pending"` and returns the completed record.
 * @param {string} jobId - Job identifier returned by `submitChatJob` or `POST /api/chat`.
 * @param {PollChatJobOptions} [options] - Polling interval, timeout, progress callback, and abort signal.
 * @returns {Promise<PollableChatJobData>} Completed chat response record.
 */
export async function pollChatJobUntilComplete(
  jobId: string,
  options: PollChatJobOptions = {},
): Promise<PollableChatJobData> {
  const intervalMs = options.intervalMs ?? 800;
  const timeoutMs = options.timeoutMs ?? 120000;
  const startMs = Date.now();
  const encodedId = encodeURIComponent(jobId);

  while (true) {
    if (options.signal?.aborted) {
      throw new Error("Request aborted by user");
    }
    const resp = await fetchWithBackendError(`/api/chat/jobs/${encodedId}`, {
      method: "GET",
      signal: options.signal,
    });
    const record = await parseResponseJson<PollableChatJobData>(resp);
    if (record.status === "pending") {
      const partial = record.partial_content || record.content || "";
      if (partial && options.onProgress) {
        options.onProgress(partial);
      }
      if (Date.now() - startMs >= timeoutMs) {
        throw new Error("Chat job polling timed out");
      }
      await delayWithAbort(intervalMs, options.signal);
      continue;
    }
    return record;
  }
}
