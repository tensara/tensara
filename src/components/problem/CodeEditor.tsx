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

      // Keywords and control flow
      { token: "keyword", foreground: "569cd6" },
      { token: "keyword.control", foreground: "569cd6" },

      // Types and variables - using the green color from the reference
      { token: "type", foreground: "4EC9B0" },
      { token: "variable", foreground: "cccccc" },
      { token: "variable.parameter", foreground: "9CDCFE" },

      // Classes and interfaces
      { token: "class", foreground: "4EC9B0" },
      { token: "interface", foreground: "4EC9B0" },

      // Functions and methods - using the cyan color from the reference
      { token: "function", foreground: "4FC1FF" },
      { token: "function.declaration", foreground: "4FC1FF" },

      // Strings and numbers
      { token: "string", foreground: "CE9178" },
      { token: "number", foreground: "B5CEA8" },

      // Preprocessor directives
      { token: "delimiter.directive", foreground: "569cd6" },
      { token: "keyword.directive", foreground: "569cd6" },

      // Constants
      { token: "constant", foreground: "569cd6" },

      // Operators
      { token: "operator", foreground: "D4D4D4" },

      // Special C++ tokens
      { token: "identifier.cpp", foreground: "D4D4D4" },
    ],
    colors: {
      // Editor UI colors - darker gray background from the reference
      "editor.background": "#1E1E1E",
      "editor.foreground": "#D4D4D4",
      "editorCursor.foreground": "#FFFFFF",
      "editor.lineHighlightBackground": "#282828",
      "editorLineNumber.foreground": "#858585",
      "editor.selectionBackground": "#264F78",
      "editor.inactiveSelectionBackground": "#3A3D41",
      "editorIndentGuide.background": "#404040",

      // Syntax highlighting
      "editor.wordHighlightBackground": "#575757B8",
      "editor.wordHighlightStrongBackground": "#004972B8",

      // UI elements
      "editorGroupHeader.tabsBackground": "#252526",
      "tab.activeBackground": "#1E1E1E",
      "tab.inactiveBackground": "#2D2D2D",
      "tab.activeForeground": "#FFFFFF",
      "tab.inactiveForeground": "#AAAAAA",

      // Borders and dividers
      "editorGroup.border": "#444444",
      "tab.border": "#252526",

      // Status bar
      "statusBar.background": "#252526",
      "statusBar.foreground": "#D4D4D4",

      // Activity bar
      "activityBar.background": "#333333",
      "activityBar.foreground": "#D4D4D4",

      // Panel
      "panel.background": "#1E1E1E",
      "panel.border": "#444444",

      // Terminal
      "terminal.background": "#1E1E1E",
      "terminal.foreground": "#D4D4D4",

      // Scrollbar
      "scrollbarSlider.background": "#424242AA",
      "scrollbarSlider.hoverBackground": "#525252AA",
      "scrollbarSlider.activeBackground": "#626262AA",
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
