/**
 * @file event-bus.ts
 * @module event-bus
 * Strongly typed unidirectional domain event bus (`TypedEventTarget<AppEvents>`)
 * decoupling UI sub-controllers from the top-level `ChatUI` coordinator.
 */

import { type ModelCatalogEntryPayload } from "./api-types";

/**
 * Payload emitted when a starter query chip, Academic Table 1 chip, Template Picker card,
 * or Cohort Builder query is selected.
 */
export interface TemplateSelectionPayload {
  /** Prompt or SQL query text to load into the chat composer. */
  promptText: string;
  /** Desired model value (`"sql"`, explicit model ID, or `"keep"` to preserve current model). */
  targetModel: string;
  /** Optional gold-standard SQL reference query for live evaluation. */
  goldSql: string | null;
  /** When true, immediately submits the prompt after updating composer/model state. */
  autoSend?: boolean;
}

/**
 * Payload emitted to display the top-level diagnostic status banner.
 */
export interface StatusBannerPayload {
  /** Visual severity level (`danger`, `warning`, or `info`). */
  severity: "danger" | "warning" | "info";
  /** Localized banner title text. */
  title: string;
  /** Localized banner description text. */
  desc: string;
  /** Icon character or glyph displayed beside the title. */
  icon: string;
  /** Whether to show the retry action button. */
  showRetry: boolean;
}

/**
 * Payload emitted to update the header system-health status chip.
 */
export interface StatusChipPayload {
  /** Visual severity class (`healthy`, `degraded`, `error`, or `offline`). */
  statusClass: "healthy" | "degraded" | "error" | "offline";
  /** Localized status chip label. */
  text: string;
}

/**
 * Screen-reader announcement event payload.
 */
export interface AnnounceEventPayload {
  /** Text announced to assistive technology. */
  message: string;
  /** Whether to use an assertive `aria-live` region. */
  isAssertive?: boolean;
}

/**
 * Composer highlight synchronization event payload.
 */
export interface ComposerSyncEventPayload {
  /** Whether to move keyboard focus to `#chat-input`. */
  focus?: boolean;
}

/**
 * Table Preview open request event payload.
 */
export interface OpenTablePreviewEventPayload {
  /** Target DuckDB table identifier. */
  tableName: string;
}

/**
 * Model catalog loaded notification payload.
 */
export interface ModelsLoadedEventPayload {
  /** Discovered and configured model entries. */
  models: ModelCatalogEntryPayload[];
}

/**
 * Schema reload request event payload.
 */
export interface SchemaReloadEventPayload {
  /** Optional explicit DuckDB path override. */
  dbPath?: string;
}

/**
 * System status health-check request payload.
 */
export interface SystemStatusCheckEventPayload {
  /** Whether triggered by manual user retry interaction. */
  isManualRetry?: boolean;
  /** Optional explicit DuckDB path override. */
  dbPath?: string;
}

/**
 * Domain event map routed through `AppEventBus`.
 */
export interface AppEvents {
  /** Announce a status or alert message to assistive screen readers. */
  announce: AnnounceEventPayload;
  /** Request re-rendering of the sidebar chat list and active chat viewport. */
  renderRequested: undefined;
  /** Synchronize composer syntax highlighting and optionally focus the input box. */
  composerSyncRequested: ComposerSyncEventPayload;
  /** Replace composer text and model from a template or cohort builder action, optionally sending immediately. */
  templateSelected: TemplateSelectionPayload;
  /** Abort any active in-flight LLM/SQL request. */
  cancelActiveRequest: undefined;
  /** Open the Cloud Provider & API Key Settings modal. */
  openProviderSettings: undefined;
  /** Open the Table Preview modal for a specific database table. */
  openTablePreview: OpenTablePreviewEventPayload;
  /** Reload the model catalog from `GET /api/models`. */
  modelsReloadRequested: undefined;
  /** Notify subscribers that a fresh model catalog list has been loaded. */
  modelsLoaded: ModelsLoadedEventPayload;
  /** Refresh the active provider status dot badge in the composer toolbar. */
  providerBadgeUpdateRequested: undefined;
  /** Reload the schema tree from `GET /api/schema`. */
  schemaReloadRequested: SchemaReloadEventPayload;
  /** Trigger a `/api/status` health check. */
  systemStatusCheckRequested: SystemStatusCheckEventPayload;
  /** Display the system diagnostic banner. */
  statusBannerShow: StatusBannerPayload;
  /** Hide the system diagnostic banner. */
  statusBannerHide: undefined;
  /** Update the system status chip indicator. */
  statusChipUpdate: StatusChipPayload;
}

/**
 * Strongly typed subclass of the standard DOM `EventTarget` supporting
 * `AbortSignal`-scoped subscriptions with zero type assertions.
 * @template EventMap - Record mapping event names to payload types.
 */
export class TypedEventTarget<
  EventMap extends { readonly [K in keyof EventMap]: EventMap[K] } = AppEvents,
> extends EventTarget {
  /** Mapped subscriber sets keyed by event name. */
  private readonly listeners: {
    [K in keyof EventMap]?: Set<(payload: EventMap[K]) => void>;
  } = {};

  /**
   * Subscribes a typed listener to `eventName`, returning an unsubscribe function.
   * @template K - Event key in `EventMap`.
   * @param {K} eventName - Domain event name to subscribe to.
   * @param {(payload: EventMap[K]) => void} handler - Callback receiving the typed event payload.
   * @param {AbortSignal} [signal] - Optional `AbortSignal` for automatic cleanup on disposal.
   * @returns {() => void} Function that removes the registered event listener.
   */
  public on<K extends keyof EventMap & string>(
    eventName: K,
    handler: (payload: EventMap[K]) => void,
    signal?: AbortSignal,
  ): () => void {
    if (signal?.aborted === true) {
      return (): void => {};
    }
    const existing = this.listeners[eventName];
    const targetBucket: Set<(payload: EventMap[K]) => void> =
      existing ?? new Set<(payload: EventMap[K]) => void>();
    if (!existing) {
      this.listeners[eventName] = targetBucket;
    }
    targetBucket.add(handler);
    /**
     * Removes `handler` from the subscriber bucket.
     * @returns {void}
     */
    const unsubscribe = (): void => {
      targetBucket.delete(handler);
    };
    signal?.addEventListener("abort", unsubscribe, { once: true });
    return unsubscribe;
  }

  /**
   * Emits a typed domain event synchronously to all active subscribers and dispatches a `CustomEvent`.
   * @template K - Event key in `EventMap`.
   * @param {K} eventName - Domain event name to emit.
   * @param {EventMap[K]} payload - Typed payload delivered to subscribers.
   * @returns {void}
   */
  public emit<K extends keyof EventMap & string>(
    eventName: K,
    payload: EventMap[K],
  ): void {
    this.dispatchEvent(
      new CustomEvent<EventMap[K]>(eventName, { detail: payload }),
    );
    const bucket = this.listeners[eventName];
    if (!bucket) {
      return;
    }
    for (const fn of Array.from(bucket)) {
      fn(payload);
    }
  }
}

/**
 * Application-wide typed event bus alias.
 */
export type AppEventBus = TypedEventTarget<AppEvents>;
