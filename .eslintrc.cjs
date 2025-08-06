/** @type {import("eslint").Linter.Config} */
const config = {
  "parser": "@typescript-eslint/parser",
  "parserOptions": {
    "project": "tsconfig.json"
  },
  "plugins": [
    "@typescript-eslint",
    "prettier"
  ],
  "extends": [
    "next/core-web-vitals",
    "plugin:@typescript-eslint/recommended-type-checked",
    "plugin:@typescript-eslint/stylistic-type-checked",
    "prettier"
  ],
  "rules": {
    "prettier/prettier": ["error"],
    "@typescript-eslint/array-type": "off",
    "@typescript-eslint/consistent-type-definitions": "off",
    "@typescript-eslint/consistent-type-imports": [
      "warn",
      {
        "prefer": "type-imports",
        "fixStyle": "inline-type-imports"
      }
    ],
    "@typescript-eslint/no-unused-vars": [
      "error",
      {
        "argsIgnorePattern": "^_",
        "varsIgnorePattern": "^_"
      }
    ],
    "@typescript-eslint/require-await": "off",
    "@typescript-eslint/no-misused-promises": [
      "error",
      {
        "checksVoidReturn": {
          "attributes": false
        }
      }
    ]
  },
  "overrides": [
    {
      "files": ["*.js", "*.mjs", "*.cjs"],
      "extends": ["plugin:@typescript-eslint/disable-type-checked"],
      "parserOptions": {
        "project": null
      }
    }
  ]
}
module.exports = config;