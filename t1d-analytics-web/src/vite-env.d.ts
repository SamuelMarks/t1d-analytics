/// <reference types="vite/client" />

/**
 * @file vite-env.d.ts
 * @module vite-env
 * Ambient TypeScript declarations for Vite client assets and CSS modules.
 */

/** Stylesheet module declaration. */
declare module "*.css" {
  /** Raw CSS stylesheet content. */
  const content: string;
  export default content;
}

/** Highlight.js theme stylesheet module declaration. */
declare module "highlight.js/styles/*.css" {
  /** Raw Highlight.js CSS content. */
  const content: string;
  export default content;
}

/**
 * Per-file Istanbul code coverage instrumentation payload.
 */
export interface IstanbulFileCoverage {
  /** Source file path. */
  path: string;
  /** Statement hit counts indexed by statement ID. */
  s: Record<string, number>;
  /** Function hit counts indexed by function ID. */
  f: Record<string, number>;
  /** Branch hit counts indexed by branch ID. */
  b: Record<string, number[]>;
}

declare global {
  /** Global browser Window extensions for code coverage instrumentation and XSS test assertions. */
  interface Window {
    /** Istanbul/NYC coverage map injected during instrumented builds. */
    __coverage__?: Record<string, IstanbulFileCoverage>;
    /** Optional XSS test sentinel property for markdown sanitization tests. */
    pwned?: boolean;
    /** Optional XSS test sentinel property for schema explorer sanitization tests. */
    hacked?: boolean;
  }
}
