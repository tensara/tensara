import { Monaco } from "@monaco-editor/react";

export function setupMonaco(monaco: Monaco) {
  monaco.editor.defineTheme("tensara-dark", {
    base: "vs-dark",
    inherit: true,
    rules: [
      // Base colors
      { token: "", foreground: "cccccc" },
      { token: "comment", foreground: "6A9955", fontStyle: "italic" },

      // Keywords and control flow (common to all languages)
      { token: "keyword", foreground: "569cd6" },
      { token: "keyword.control", foreground: "569cd6" },

      // Types and variables (C++/CUDA)
      { token: "type", foreground: "4EC9B0" },
      { token: "variable", foreground: "cccccc" },
      { token: "variable.parameter", foreground: "9CDCFE" },

      // Classes, structs, and interfaces
      { token: "class", foreground: "4EC9B0" },
      { token: "struct", foreground: "4EC9B0" },
      { token: "interface", foreground: "4EC9B0" },

      // Functions and methods
      { token: "function", foreground: "4FC1FF" },
      { token: "function.declaration", foreground: "4FC1FF" },

      // Strings and numbers
      { token: "string", foreground: "CE9178" },
      { token: "number", foreground: "B5CEA8" },

      // Preprocessor directives (C++/CUDA)
      { token: "delimiter.directive", foreground: "569cd6" },
      { token: "keyword.directive", foreground: "569cd6" },

      // Constants
      { token: "constant", foreground: "569cd6" },

      // Operators
      { token: "operator", foreground: "D4D4D4" },

      // C++/CUDA specific tokens
      { token: "identifier.cpp", foreground: "D4D4D4" },

      // CUDA specific tokens
      { token: "keyword.cuda", foreground: "C586C0" }, // CUDA specific keywords
      { token: "identifier.cuda", foreground: "DCDCAA" }, // CUDA specific identifiers

      // Python specific tokens
      { token: "keyword.python", foreground: "569cd6" },
      { token: "function.python", foreground: "4FC1FF" },
      { token: "class.python", foreground: "4EC9B0" },
      { token: "decorator.python", foreground: "DCDCAA" },

      // Triton specific tokens
      { token: "keyword.triton", foreground: "C586C0" },
      { token: "function.triton", foreground: "4FC1FF" },
      { token: "decorator.triton", foreground: "DCDCAA" },

      // Mojo specific tokens
      { token: "keyword.mojo", foreground: "569cd6" },
      { token: "type.mojo", foreground: "4EC9B0" },
      { token: "function.mojo", foreground: "C586C0" },
      { token: "decorator.mojo", foreground: "DCDCAA" },
      { token: "struct.mojo", foreground: "4EC9B0" },
    ],
    colors: {
      // Editor UI colors - darker black background
      "editor.background": "#111111",
      "editor.foreground": "#D4D4D4",
      "editorCursor.foreground": "#FFFFFF",
      "editor.lineHighlightBackground": "#1A1A1A",
      "editorLineNumber.foreground": "#858585",
      "editor.selectionBackground": "#264F78",
      "editor.inactiveSelectionBackground": "#3A3D41",
      "editorIndentGuide.background": "#303030",

      // Syntax highlighting
      "editor.wordHighlightBackground": "#575757B8",
      "editor.wordHighlightStrongBackground": "#004972B8",

      // UI elements
      "editorGroupHeader.tabsBackground": "#141414",
      "tab.activeBackground": "#181818",
      "tab.inactiveBackground": "#1C1C1C",
      "tab.activeForeground": "#FFFFFF",
      "tab.inactiveForeground": "#AAAAAA",

      // Borders and dividers
      "editorGroup.border": "#303030",
      "tab.border": "#141414",

      // Status bar
      "statusBar.background": "#141414",
      "statusBar.foreground": "#D4D4D4",

      // Activity bar
      "activityBar.background": "#181818",
      "activityBar.foreground": "#D4D4D4",

      // Panel
      "panel.background": "#111111",
      "panel.border": "#303030",

      // Terminal
      "terminal.background": "#111111",
      "terminal.foreground": "#D4D4D4",

      // Scrollbar
      "scrollbarSlider.background": "#383838AA",
      "scrollbarSlider.hoverBackground": "#454545AA",
      "scrollbarSlider.activeBackground": "#505050AA",
    },
  });

  // Register custom tokenizer for CUDA C++ specifics
  monaco.languages.setMonarchTokensProvider("cpp", {
    defaultToken: "",

    // Common tokens
    tokenizer: {
      root: [
        // CUDA specific keywords
        [
          /\b(threadIdx|blockIdx|blockDim|gridDim|warpSize|__global__|__device__|__host__|__shared__)\b/,
          "keyword.cuda",
        ],

        // Preprocessor directives
        [/#\s*include/, "keyword.directive"],
        [/<.*>/, "string"],

        // Comments
        [/\/\/.*$/, "comment"],
        [/\/\*/, "comment", "@comment"],

        // Keywords
        [
          /\b(if|else|for|while|do|switch|case|default|break|continue|return|void|int|float|double|char|unsigned|const|static|extern|struct|union|enum|typedef|class|template|namespace|using|new|delete|true|false|nullptr)\b/,
          "keyword",
        ],

        // Types
        [/\b(int|float|double|char|bool|void|size_t)\b/, "type"],
        [/\b(float[1-4]|int[1-4]|uint[1-4]|dim3)\b/, "type"],

        // Functions
        [/\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(/, "function"],

        // External keyword
        [/\bextern\b/, "keyword.extern"],

        // Strings
        [/"([^"\\]|\\.)*$/, "string.invalid"],
        [/"/, "string", "@string"],

        // Numbers
        [/\b\d+\.\d+([eE][\-+]?\d+)?\b/, "number.float"],
        [/\b0[xX][0-9a-fA-F]+\b/, "number.hex"],
        [/\b\d+\b/, "number"],

        // Operators
        [/[{}()\[\]]/, "@brackets"],
        [/[<>=%&|+\-*/~^]+/, "operator"],

        // Identifiers
        [/[a-zA-Z_]\w*/, "identifier"],
      ],

      comment: [
        [/[^/*]+/, "comment"],
        [/\/\*/, "comment", "@push"],
        [/\*\//, "comment", "@pop"],
        [/[/*]/, "comment"],
      ],

      string: [
        [/[^\\"]+/, "string"],
        [/\\./, "string.escape"],
        [/"/, "string", "@pop"],
      ],
    },
  });

  // Register Mojo language tokenizer
  monaco.languages.register({ id: "mojo" });
  monaco.languages.setMonarchTokensProvider("mojo", {
    defaultToken: "",
    tokenizer: {
      root: [
        // Comments
        [/#.*$/, "comment"],
        [/'''/, "comment", "@multilineString"],
        [/"""/, "comment", "@multilineDocstring"],
        [/\/\*/, "comment", "@comment"],

        // Decorators
        [/@[a-zA-Z_][\w$]*/, "decorator"],

        // Keywords
        [
          /\b(fn|struct|def|var|let|alias|trait|impl|for|while|if|else|elif|return|break|continue|match|and|or|not|in|is|as|from|import|with|as|try|except|finally|raise|assert|await|async|del|global|nonlocal|lambda|pass|yield|None|True|False|Self|self|owned|inout|mutates|borrowed|raises)\b/,
          "keyword",
        ],

        // Types
        [
          /\b(Int|UInt|Int8|Int16|Int32|Int64|UInt8|UInt16|UInt32|UInt64|Float16|Float32|Float64|Bool|String|SIMD|DType|Scalar|StringLiteral|AnyType|NoneType)\b/,
          "type",
        ],

        // Special control keywords
        [/\b(return|yield|raise|break|continue|pass)\b/, "keyword.control"],

        // Functions
        [/\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(/, "function.mojo"],

        // Strings
        [/"([^"\\]|\\.)*$/, "string.invalid"],
        [/'([^'\\]|\\.)*$/, "string.invalid"],
        [/"/, "string", "@string_double"],
        [/'/, "string", "@string_single"],

        // Numbers
        [/\b(0[xX][0-9a-fA-F]+)\b/, "number.hex"],
        [/\b(0[oO][0-7]+)\b/, "number.octal"],
        [/\b(0[bB][01]+)\b/, "number.binary"],
        [/\b(\d+\.\d+([eE][\-+]?\d+)?)\b/, "number.float"],
        [/\b(\d+([eE][\-+]?\d+)?)\b/, "number"],

        // Operators
        [/[{}()\[\]]/, "@brackets"],
        [/[<>=%&|+\-*/~^]+/, "operator"],

        // Identifiers
        [/[a-zA-Z_]\w*/, "identifier"],
      ],

      comment: [
        [/[^/*]+/, "comment"],
        [/\/\*/, "comment", "@push"],
        [/\*\//, "comment", "@pop"],
        [/[/*]/, "comment"],
      ],

      multilineString: [
        [/[^']+/, "comment"],
        [/'''/, "comment", "@pop"],
        [/'/, "comment"],
      ],

      multilineDocstring: [
        [/[^"]+/, "comment"],
        [/"""/, "comment", "@pop"],
        [/"/, "comment"],
      ],

      string_double: [
        [/[^"\\]+/, "string"],
        [/\\./, "string.escape"],
        [/"/, "string", "@pop"],
      ],

      string_single: [
        [/[^'\\]+/, "string"],
        [/\\./, "string.escape"],
        [/'/, "string", "@pop"],
      ],
    },
  });

  monaco.languages.setMonarchTokensProvider("cpp", {
    defaultToken: "",

    // Common tokens
    tokenizer: {
      root: [
        // CUDA specific keywords
        [
          /\b(threadIdx|blockIdx|blockDim|gridDim|warpSize|__global__|__device__|__host__|__shared__)\b/,
          "keyword.cuda",
        ],

        // Preprocessor directives
        [/#\s*include/, "keyword.directive"],
        [/<.*>/, "string"],

        // Comments
        [/\/\/.*$/, "comment"],
        [/\/\*/, "comment", "@comment"],

        // Keywords
        [
          /\b(if|else|for|while|do|switch|case|default|break|continue|return|void|int|float|double|char|unsigned|const|static|extern|struct|union|enum|typedef|class|template|namespace|using|new|delete|true|false|nullptr)\b/,
          "keyword",
        ],

        // Types
        [/\b(int|float|double|char|bool|void|size_t)\b/, "type"],
        [/\b(float[1-4]|int[1-4]|uint[1-4]|dim3)\b/, "type"],

        // Functions
        [/\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(/, "function"],

        // External keyword
        [/\bextern\b/, "keyword.extern"],

        // Strings
        [/"([^"\\]|\\.)*$/, "string.invalid"],
        [/"/, "string", "@string"],

        // Numbers
        [/\b\d+\.\d+([eE][\-+]?\d+)?\b/, "number.float"],
        [/\b0[xX][0-9a-fA-F]+\b/, "number.hex"],
        [/\b\d+\b/, "number"],

        // Operators
        [/[{}()\[\]]/, "@brackets"],
        [/[<>=%&|+\-*/~^]+/, "operator"],

        // Identifiers
        [/[a-zA-Z_]\w*/, "identifier"],
      ],

      comment: [
        [/[^/*]+/, "comment"],
        [/\/\*/, "comment", "@push"],
        [/\*\//, "comment", "@pop"],
        [/[/*]/, "comment"],
      ],

      string: [
        [/[^\\"]+/, "string"],
        [/\\./, "string.escape"],
        [/"/, "string", "@pop"],
      ],
    },
  });
}