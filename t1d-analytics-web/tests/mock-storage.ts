/**
 * @file mock-storage.ts
 * @module tests/mock-storage
 * In-memory localStorage and minimal document mock for deterministic unit testing in JSDOM and Node environments.
 */

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
  Object.defineProperty(globalThis, "document", {
    value: mockDoc,
    writable: true,
    configurable: true,
  });
}

/**
 * In-memory Web Storage API implementation for testing.
 */
export const localStorageMock = (function () {
  let store: Record<string, string> = {};
  return {
    /**
     * Retrieves a stored string value by key.
     * @param {string} key - The storage key to look up.
     * @returns {string | null} The stored string or null if not found.
     */
    getItem: function (key: string): string | null {
      return store[key] || null;
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

Object.defineProperty(globalThis, "localStorage", {
  value: localStorageMock,
  writable: true,
  configurable: true,
});

if (typeof window !== "undefined") {
  Object.defineProperty(window, "localStorage", {
    value: localStorageMock,
    writable: true,
    configurable: true,
  });
}
