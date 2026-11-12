/**
 * @file modal-utils.ts
 * @module modal-utils
 * Accessible keyboard focus-trapping, DOM element lookup, and composable `ModalDialogController`.
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
  const firstElement = focusableElements[0];
  const lastElement = focusableElements[focusableElements.length - 1];
  if (!firstElement || !lastElement) return;
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

/**
 * Configuration options for `ModalDialogController`.
 */
export interface ModalDialogOptions {
  /** Root modal overlay element (e.g. `#table-modal`, `#cohort-modal`, `#provider-modal`). */
  modalEl: HTMLElement;
  /** Optional close button inside the modal header. */
  closeBtn?: HTMLElement | null;
  /** Optional default element to focus when the modal opens (defaults to `closeBtn`). */
  initialFocusEl?: HTMLElement | null;
  /** Fallback element ID focused on close when the opener element is absent or detached. */
  fallbackFocusId?: string;
  /** Optional hook invoked whenever the modal transitions to closed. */
  onClose?: () => void;
}

/**
 * Shared composable controller managing modal visibility (`.hidden`, `aria-hidden`, `aria-modal`),
 * backdrop click dismissal, Escape handling, Tab focus trapping, opener focus restoration,
 * and `AbortController`-scoped listener lifecycle.
 */
export class ModalDialogController {
  /** Managed modal root element. */
  public readonly modalEl: HTMLElement;
  /** Optional header close button. */
  private readonly closeBtn: HTMLElement | null;
  /** Optional default element focused upon opening. */
  private readonly initialFocusEl: HTMLElement | null;
  /** Fallback DOM ID focused when opener focus cannot be restored. */
  private readonly fallbackFocusId: string;
  /** Optional cleanup hook invoked when closing. */
  private readonly onCloseHook: (() => void) | undefined;
  /** Persistent AbortController for close button and backdrop click listeners. */
  private readonly baseAbortController = new AbortController();
  /** Per-open-session AbortController for keyboard focus-trap and Escape listeners. */
  private sessionAbortController: AbortController | null = null;
  /** Element that held keyboard focus before the modal opened. */
  private openerFocusEl: HTMLElement | null = null;

  /**
   * Initializes a composable `ModalDialogController` for `options.modalEl`.
   * @param {ModalDialogOptions} options - Modal DOM elements and lifecycle hooks.
   */
  constructor(options: ModalDialogOptions) {
    this.modalEl = options.modalEl;
    this.closeBtn = options.closeBtn ?? null;
    this.initialFocusEl = options.initialFocusEl ?? null;
    this.fallbackFocusId = options.fallbackFocusId ?? "chat-input";
    this.onCloseHook = options.onClose;

    const signal = this.baseAbortController.signal;
    this.closeBtn?.addEventListener(
      "click",
      () => {
        this.close();
      },
      { signal },
    );
    this.modalEl.addEventListener(
      "click",
      (e: MouseEvent) => {
        if (e.target === this.modalEl) {
          this.close();
        }
      },
      { signal },
    );
  }

  /**
   * Returns true when the modal is currently visible and not `aria-hidden="true"`.
   * @returns {boolean} True when open.
   */
  public isOpen(): boolean {
    return (
      !this.modalEl.classList.contains("hidden") &&
      this.modalEl.getAttribute("aria-hidden") !== "true"
    );
  }

  /**
   * Records `document.activeElement` prior to async pre-open work if not already open.
   * @returns {void}
   */
  public captureOpenerFocus(): void {
    if (!this.isOpen() && this.openerFocusEl === null) {
      const active = document.activeElement;
      this.openerFocusEl =
        active instanceof HTMLElement && active !== document.body
          ? active
          : null;
    }
  }

  /**
   * Opens the modal, wires focus-trapping / Escape listeners, and moves focus inside the dialog.
   * @param {HTMLElement | null} [focusTarget] - Optional override element to focus on open.
   * @returns {AbortSignal} Active open-session `AbortSignal` that aborts automatically when closed.
   */
  public open(focusTarget?: HTMLElement | null): AbortSignal {
    this.captureOpenerFocus();
    this.sessionAbortController?.abort();
    this.sessionAbortController = new AbortController();
    const signal = this.sessionAbortController.signal;

    this.modalEl.classList.remove("hidden");
    this.modalEl.setAttribute("aria-hidden", "false");
    this.modalEl.setAttribute("aria-modal", "true");

    this.modalEl.addEventListener(
      "keydown",
      (e: KeyboardEvent) => {
        trapModalFocus(e, this.modalEl, () => this.close());
      },
      { signal },
    );

    const target = focusTarget ?? this.initialFocusEl ?? this.closeBtn;
    target?.focus();
    return signal;
  }

  /**
   * Closes the modal, aborts session listeners, invokes `onClose`, and restores opener focus.
   * @returns {void}
   */
  public close(): void {
    this.sessionAbortController?.abort();
    this.sessionAbortController = null;
    this.modalEl.classList.add("hidden");
    this.modalEl.setAttribute("aria-hidden", "true");
    this.modalEl.removeAttribute("aria-modal");
    this.onCloseHook?.();

    const opener = this.openerFocusEl;
    this.openerFocusEl = null;
    if (opener && document.body.contains(opener)) {
      opener.focus();
    } else {
      document.getElementById(this.fallbackFocusId)?.focus();
    }
  }

  /**
   * Aborts active open-session listeners without changing modal visibility.
   * @returns {void}
   */
  public abortSessionListeners(): void {
    this.sessionAbortController?.abort();
    this.sessionAbortController = null;
  }

  /**
   * Disposes all open-session and base modal event listeners.
   * @returns {void}
   */
  public dispose(): void {
    this.abortSessionListeners();
    this.baseAbortController.abort();
  }
}
