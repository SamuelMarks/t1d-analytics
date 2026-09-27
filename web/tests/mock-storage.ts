if (typeof document === "undefined") {
  const mockDoc = {
    documentElement: { lang: "en", dir: "ltr" },
    querySelectorAll: () => [],
    addEventListener: () => {},
    removeEventListener: () => {},
  };
  Object.defineProperty(globalThis, "document", {
    value: mockDoc,
    writable: true,
    configurable: true,
  });
}

export const localStorageMock = (function () {
  let store: Record<string, string> = {};
  return {
    getItem: function (key: string) {
      return store[key] || null;
    },
    setItem: function (key: string, value: string) {
      store[key] = value.toString();
    },
    clear: function () {
      store = {};
    },
    removeItem: function (key: string) {
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
