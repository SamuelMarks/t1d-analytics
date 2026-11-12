/**
 * @file eslint.config.mjs
 * @module eslint.config
 * Flat ESLint configuration enforcing strict `@typescript-eslint` type-checked rules,
 * cyclomatic complexity <= 15, and 100% JSDoc documentation across all TypeScript files.
 */

import path from "node:path";
import { fileURLToPath } from "node:url";
import tsParser from "@typescript-eslint/parser";
import tsPlugin from "@typescript-eslint/eslint-plugin";
import jsdocPlugin from "eslint-plugin-jsdoc";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

export default [
  {
    ignores: [
      "dist/**",
      "node_modules/**",
      "coverage/**",
      ".nyc_output/**",
      "playwright-report/**",
      "test-results/**",
      "eslint.config.mjs",
      "src/vite-env.d.ts",
    ],
  },
  {
    files: [
      "src/**/*.ts",
      "tests/**/*.ts",
      "tests-e2e/**/*.ts",
      "tests-e2e-full/**/*.ts",
      "*.config.ts",
    ],
    languageOptions: {
      parser: tsParser,
      parserOptions: {
        project: "./tsconfig.json",
        tsconfigRootDir: __dirname,
        ecmaVersion: "latest",
        sourceType: "module",
      },
    },
    plugins: {
      "@typescript-eslint": tsPlugin,
      jsdoc: jsdocPlugin,
    },
    settings: {
      jsdoc: {
        mode: "typescript",
        tagNamePreference: {
          file: "file",
          augments: "extends",
        },
      },
    },
    rules: {
      "@typescript-eslint/no-explicit-any": "error",
      "@typescript-eslint/no-non-null-assertion": "error",
      "@typescript-eslint/no-floating-promises": "error",
      "@typescript-eslint/prefer-promise-reject-errors": "error",
      "@typescript-eslint/strict-boolean-expressions": [
        "error",
        {
          allowString: true,
          allowNumber: true,
          allowNullableObject: true,
          allowNullableBoolean: true,
          allowNullableString: true,
          allowNullableNumber: false,
          allowAny: false,
        },
      ],
      "no-restricted-syntax": [
        "error",
        {
          selector: "TSAsExpression > TSUnknownKeyword",
          message: "Type assertion 'as unknown' is forbidden.",
        },
        {
          selector: "TSAsExpression > TSAnyKeyword",
          message: "Type assertion 'as any' is forbidden.",
        },
      ],
      "jsdoc/require-file-overview": [
        "error",
        {
          tags: {
            file: {
              initialCommentsOnly: true,
              mustExist: true,
              preventDuplicates: true,
            },
          },
        },
      ],
      "jsdoc/require-jsdoc": [
        "error",
        {
          publicOnly: false,
          checkConstructors: true,
          checkGetters: true,
          checkSetters: true,
          require: {
            ArrowFunctionExpression: false,
            ClassDeclaration: true,
            ClassExpression: true,
            FunctionDeclaration: true,
            FunctionExpression: true,
            MethodDefinition: true,
          },
          contexts: [
            "TSInterfaceDeclaration",
            "TSTypeAliasDeclaration",
            "VariableDeclarator > ArrowFunctionExpression",
            "PropertyDefinition > ArrowFunctionExpression",
            "ExportDefaultDeclaration > ArrowFunctionExpression",
          ],
        },
      ],
      "jsdoc/require-param": [
        "error",
        {
          checkDestructured: false,
        },
      ],
      "jsdoc/require-param-description": "error",
      "jsdoc/require-returns": [
        "error",
        {
          forceRequireReturn: false,
        },
      ],
      "jsdoc/check-param-names": [
        "error",
        {
          checkDestructured: false,
        },
      ],
    },
  },
  {
    files: ["src/**/*.ts"],
    rules: {
      complexity: ["error", 15],
    },
  },
];
