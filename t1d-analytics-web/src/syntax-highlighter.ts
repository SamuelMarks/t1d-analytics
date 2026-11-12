/**
 * @file syntax-highlighter.ts
 * @module syntax-highlighter
 * Shared Highlight.js core initializer registering only `sql`, `markdown`, `python`, and `json`.
 */

import hljs from "highlight.js/lib/core";
import json from "highlight.js/lib/languages/json";
import markdown from "highlight.js/lib/languages/markdown";
import python from "highlight.js/lib/languages/python";
import sql from "highlight.js/lib/languages/sql";

hljs.registerLanguage("sql", sql);
hljs.registerLanguage("markdown", markdown);
hljs.registerLanguage("python", python);
hljs.registerLanguage("json", json);

export { hljs };
