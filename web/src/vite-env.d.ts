/// <reference types="vite/client" />

declare module "*.css" {
  const content: string;
  export default content;
}

declare module "highlight.js/styles/*.css" {
  const content: string;
  export default content;
}
