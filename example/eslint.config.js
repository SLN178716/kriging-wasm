import globals from 'globals'
import jseslint from '@eslint/js'
import tseslint from 'typescript-eslint'
import eslintConfigPrettier from "eslint-config-prettier";

export default [
  { files: ["**/*.{js,mjs,cjs,ts}"] },
  {
    ignores: [
      "node_modules",
      "dist",
      "public",
      "src/kriging-wasm/**/*"
    ],
  },
  jseslint.configs.recommended,
  ...tseslint.configs.recommended,
  eslintConfigPrettier,
  {
    languageOptions: {
      globals: globals.browser,
    },
  }
]
