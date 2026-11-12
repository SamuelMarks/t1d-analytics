/**
 * @file api-client.ts
 * @module api-client
 * Typed HTTP fetch wrapper, JSON response parser, and localized backend error translator.
 */

import i18next from "./i18n";
import {
  type ApiErrorInput,
  type ApiErrorPayload,
  type ChatTurnResponse,
} from "./api-types";

export { type ApiErrorInput, type ApiErrorPayload, type ChatTurnResponse };

/**
 * Parses a backend error string or detail object into a translated string.
 * @param {ApiErrorInput} errorObj - The raw backend error string or structured detail object.
 * @returns {string} Localized or formatted error message string.
 */
export function parseApiError(errorObj: ApiErrorInput): string {
  if (typeof errorObj === "string") {
    return i18next.exists(errorObj) ? String(i18next.t(errorObj)) : errorObj;
  }
  if (
    errorObj !== null &&
    typeof errorObj === "object" &&
    "error_code" in errorObj
  ) {
    const key = errorObj.error_code;
    const params = errorObj.params;
    if (!i18next.exists(key)) {
      return `${key}: ${JSON.stringify(params)}`;
    }
    return params ? String(i18next.t(key, params)) : String(i18next.t(key));
  }
  return String(errorObj);
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
export interface RemoteSessionPayload<TMsg> {
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
 * Reads any configured `T1D_API_KEY` Bearer token from `sessionStorage` or `localStorage`.
 * @returns {string | null} Trimmed Bearer token if configured, otherwise null.
 */
export function getStoredBackendApiKey(): string | null {
  try {
    if (typeof sessionStorage !== "undefined") {
      const sessVal = sessionStorage.getItem("t1d_backend_api_key")?.trim();
      if (sessVal) {
        return sessVal;
      }
    }
    if (typeof localStorage !== "undefined") {
      const localVal = localStorage.getItem("t1d_backend_api_key")?.trim();
      if (localVal) {
        return localVal;
      }
    }
  } catch {
    // Ignore storage access errors in restricted contexts
  }
  return null;
}

/**
 * Merges an `Authorization: Bearer <token>` header into `options` when a backend API key is configured.
 * Leaves `options` untouched when no backend API key is configured.
 * @param {RequestInit} [options] - Optional Fetch API RequestInit options.
 * @returns {RequestInit | undefined} RequestInit augmented with Authorization header when configured.
 */
export function withBearerAuth(options?: RequestInit): RequestInit | undefined {
  const token = getStoredBackendApiKey();
  if (!token) {
    return options;
  }
  const mergedHeaders: Record<string, string> = {};
  if (options?.headers) {
    if (options.headers instanceof Headers) {
      options.headers.forEach((val, key) => {
        mergedHeaders[key] = val;
      });
    } else if (Array.isArray(options.headers)) {
      for (const [key, val] of options.headers) {
        mergedHeaders[key] = val;
      }
    } else {
      Object.assign(mergedHeaders, options.headers);
    }
  }
  if (!mergedHeaders.Authorization && !mergedHeaders.authorization) {
    mergedHeaders.Authorization = `Bearer ${token}`;
  }
  return {
    ...(options ?? {}),
    headers: mergedHeaders,
  };
}

/**
 * Fetches persisted chat sessions from `/api/sessions`.
 * @template TMsg - Message item type.
 * @returns {Promise<Array<RemoteSessionPayload<TMsg>>>} Normalized list of remote sessions.
 */
export async function fetchRemoteSessions<TMsg>(): Promise<
  Array<RemoteSessionPayload<TMsg>>
> {
  const authInit = withBearerAuth();
  const resp = authInit
    ? await fetch("/api/sessions", authInit)
    : await fetch("/api/sessions");
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
  const authInit = withBearerAuth();
  const url = `/api/sessions/${encodeURIComponent(sessionId)}`;
  const resp = authInit ? await fetch(url, authInit) : await fetch(url);
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
  return fetch(
    "/api/sessions",
    withBearerAuth({
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: sessionId,
        title,
        messages,
      }),
    }),
  );
}

/**
 * Helper to fetch and automatically throw parsed backend errors.
 * Automatically attaches `Authorization: Bearer <token>` when `t1d_backend_api_key` is configured.
 * @param {string} url - The URL to fetch.
 * @param {RequestInit} [options] - Fetch options.
 * @returns {Promise<Response>} The fetch response.
 */
export async function fetchWithBackendError(
  url: string,
  options?: RequestInit,
): Promise<Response> {
  const response = await fetch(url, withBearerAuth(options));
  if (!response.ok) {
    let errorMsg = `HTTP ${response.status}`;
    try {
      const errData = await parseResponseJson<BackendErrorResponse>(response);
      if (
        errData.detail !== undefined &&
        errData.detail !== null &&
        errData.detail !== ""
      ) {
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
  return fetch(
    `/api/sessions/${encodeURIComponent(sessionId)}`,
    withBearerAuth({
      method: "DELETE",
    }),
  );
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
  if (signal?.aborted === true) {
    return Promise.reject(new Error("Request aborted by user"));
  }
  return new Promise<void>((resolve, reject) => {
    /**
     * Clears the pending timeout and rejects when `signal` aborts.
     * @returns {void}
     */
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
 * @returns {Promise<ChatTurnResponse>} Initial job ticket or immediate response.
 */
export async function submitChatJob(
  bodyJson: string,
  headers: Record<string, string>,
  signal?: AbortSignal,
): Promise<ChatTurnResponse> {
  const reqInit: RequestInit = {
    method: "POST",
    headers,
    body: bodyJson,
    ...(signal ? { signal } : {}),
  };
  const response = await fetchWithBackendError("/api/chat/jobs", reqInit);
  return parseResponseJson<ChatTurnResponse>(response);
}

/**
 * Polls `GET /api/chat/jobs/{jobId}` until `status !== "pending"` and returns the completed record.
 * @param {string} jobId - Job identifier returned by `submitChatJob` or `POST /api/chat`.
 * @param {PollChatJobOptions} [options] - Polling interval, timeout, progress callback, and abort signal.
 * @returns {Promise<ChatTurnResponse>} Completed chat response record.
 */
export async function pollChatJobUntilComplete(
  jobId: string,
  options: PollChatJobOptions = {},
): Promise<ChatTurnResponse> {
  const intervalMs = options.intervalMs ?? 800;
  const timeoutMs = options.timeoutMs ?? 120000;
  const startMs = Date.now();
  const encodedId = encodeURIComponent(jobId);

  while (true) {
    if (options.signal?.aborted === true) {
      throw new Error("Request aborted by user");
    }
    const reqInit: RequestInit = {
      method: "GET",
      ...(options.signal ? { signal: options.signal } : {}),
    };
    const resp = await fetchWithBackendError(
      `/api/chat/jobs/${encodedId}`,
      reqInit,
    );
    const record = await parseResponseJson<ChatTurnResponse>(resp);
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
