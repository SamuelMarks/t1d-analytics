/**
 * @file modal-utils.ts
 * @module modal-utils
 * Accessible keyboard focus-trapping and modal dialog interaction utilities.
 */

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
  const isOutsideOrRoot =
    !activeEl || !modal.contains(activeEl) || activeEl === modal;

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
