/**
 * @file mock-storage.ts
 * @module tests/mock-storage
 * In-memory localStorage and minimal document mock for deterministic unit testing in JSDOM and Node environments.
 */

import { vi } from "vitest";

if (typeof document === "undefined") {
  /** Minimal document stub for tests running in the `@vitest-environment node` context. */
  const mockDoc = {
    /** Root html element stub. */
    documentElement: { lang: "en", dir: "ltr" },
    /**
     * Stubbed querySelectorAll returning an empty list.
     * @returns {never[]} Empty array.
     */
    querySelectorAll: (): never[] => [],
  };
  vi.stubGlobal("document", mockDoc);
}

/**
 * In-memory Web Storage API implementation for testing.
 */
export const localStorageMock: Storage = (function (): Storage {
  let store: Record<string, string> = {};
  return {
    /**
     * Returns the number of key/value pairs currently present in the storage.
     * @returns {number} Count of stored keys.
     */
    get length(): number {
      return Object.keys(store).length;
    },
    /**
     * Returns the name of the nth key in the storage, or null if out of bounds.
     * @param {number} index - Zero-based key index.
     * @returns {string | null} Key name or null.
     */
    key: function (index: number): string | null {
      const keys = Object.keys(store);
      return index >= 0 && index < keys.length ? keys[index] : null;
    },
    /**
     * Retrieves a stored string value by key, preserving empty strings (`""`).
     * @param {string} key - The storage key to look up.
     * @returns {string | null} The stored string or null if not found.
     */
    getItem: function (key: string): string | null {
      return Object.prototype.hasOwnProperty.call(store, key)
        ? store[key]
        : null;
    },
    /**
     * Stores a string value under the specified key.
     * @param {string} key - The storage key to set.
     * @param {string} value - The value to store.
     * @returns {void}
     */
    setItem: function (key: string, value: string): void {
      store[key] = value.toString();
    },
    /**
     * Clears all stored entries in the mock storage.
     * @returns {void}
     */
    clear: function (): void {
      store = {};
    },
    /**
     * Removes a single key from the mock storage.
     * @param {string} key - The storage key to delete.
     * @returns {void}
     */
    removeItem: function (key: string): void {
      delete store[key];
    },
  };
})();

vi.stubGlobal("localStorage", localStorageMock);
