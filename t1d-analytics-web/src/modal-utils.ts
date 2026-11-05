/**
 * @file modal-utils.ts
 * @module modal-utils
 * Accessible keyboard focus-trapping, DOM element lookup, and modal dialog interaction utilities.
 */

/**
 * Looks up a required typed element by ID and throws a descriptive Error if missing.
 * @template T - Target HTML element subtype.
 * @param {string} id - The DOM element ID.
 * @returns {T} The matched element.
 * @throws {Error} When the required mount element is not found in the document.
 */
export function requireElementById<T extends HTMLElement>(id: string): T {
  const el = document.querySelector<T>(`#${id}`);
  if (!el) {
    throw new Error(`Required DOM element #${id} not found`);
  }
  return el;
}

/**
 * Looks up an optional typed element by ID using runtime `instanceof` type narrowing.
 * @template T - Target HTML element subtype.
 * @param {string} id - The DOM element ID.
 * @param {new (...args: never[]) => T} ctor - Element constructor used for runtime narrowing.
 * @returns {T | null} The matched typed element or null if absent or of a different element type.
 */
export function optionalElementById<T extends HTMLElement>(
  id: string,
  ctor: new (...args: never[]) => T,
): T | null {
  const el = document.getElementById(id);
  return el instanceof ctor ? el : null;
}

/**
 * Traps keyboard focus within a modal dialog and handles the Escape key to close.
 * @param {KeyboardEvent} e - The keyboard event.
 * @param {HTMLElement} modal - The container element to trap focus within.
 * @param {() => void} onClose - The callback function to invoke when Escape is pressed.
 * @returns {void}
 */
export function trapModalFocus(
  e: KeyboardEvent,
  modal: HTMLElement,
  onClose: () => void,
): void {
  if (e.key === "Escape") {
    e.preventDefault();
    onClose();
    return;
  }
  if (e.key !== "Tab") return;
  const focusableElements = modal.querySelectorAll<HTMLElement>(
    'button:not([disabled]), [href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])',
  );
  if (focusableElements.length === 0) return;
  const firstElement = focusableElements[0];
  const lastElement = focusableElements[focusableElements.length - 1];
  const activeEl = document.activeElement;
  const isOutsideOrRoot = !modal.contains(activeEl) || activeEl === modal;

  if (e.shiftKey) {
    if (activeEl === firstElement || isOutsideOrRoot) {
      e.preventDefault();
      lastElement.focus();
    }
  } else {
    if (activeEl === lastElement || isOutsideOrRoot) {
      e.preventDefault();
      firstElement.focus();
    }
  }
}
