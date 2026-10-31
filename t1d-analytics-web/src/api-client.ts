/**
 * @file api-client.ts
 * @module api-client
 * Typed HTTP fetch wrapper, JSON response parser, and localized backend error translator.
 */

import i18next from "./i18n";
import { type SqlCellValue } from "./formatters";

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
 * Parses a backend error string or detail object into a translated string.
 * @param {ApiErrorInput} errorObj - The raw backend error string or structured detail object.
 * @returns {string} Localized or formatted error message string.
 */
export function parseApiError(errorObj: ApiErrorInput): string {
  if (typeof errorObj === "string") {
    return i18next.exists(errorObj) ? String(i18next.t(errorObj)) : errorObj;
  }
  if (errorObj && typeof errorObj === "object" && "error_code" in errorObj) {
    const key = errorObj.error_code;
    const params = errorObj.params;
    return i18next.exists(key)
      ? String(i18next.t(key, params))
      : `${key}: ${JSON.stringify(params)}`;
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
