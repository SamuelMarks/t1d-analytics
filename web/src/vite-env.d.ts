/// <reference types="vite/client" />

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
