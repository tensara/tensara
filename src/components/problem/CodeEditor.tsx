import { Box } from "@chakra-ui/react";
import Editor, { type Monaco } from "@monaco-editor/react";
import { type ProgrammingLanguage } from "~/types/misc";

interface CodeEditorProps {
  code: string;
  setCode: (code: string) => void;
  selectedLanguage: ProgrammingLanguage;
}

function setupMonaco(monaco: Monaco) {
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

const CodeEditor = ({ code, setCode, selectedLanguage }: CodeEditorProps) => {
  return (
    <Box w="100%" h="100%" bg="gray.800" borderRadius="xl" overflow="hidden">
      <Editor
        height="100%"
        theme="tensara-dark"
        value={code}
        onChange={(value) => setCode(value ?? "")}
        language={selectedLanguage === "cuda" ? "cpp" : "python"}
        beforeMount={setupMonaco}
        options={{
          minimap: { enabled: false },
          fontSize: 14,
          lineNumbers: "on",
          scrollBeyondLastLine: false,
          automaticLayout: true,
          padding: { top: 16, bottom: 16 },
          fontFamily: "JetBrains Mono, monospace",
        }}
      />
    </Box>
  );
};

export default CodeEditor;
