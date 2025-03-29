import { Box } from "@chakra-ui/react";
import Editor from "@monaco-editor/react";
import { type ProgrammingLanguage } from "~/types/misc";

interface CodeEditorProps {
  code: string;
  setCode: (code: string) => void;
  selectedLanguage: ProgrammingLanguage;
}

const CodeEditor = ({ code, setCode, selectedLanguage }: CodeEditorProps) => {
  return (
    <Box w="100%" h="100%" bg="gray.800" borderRadius="xl" overflow="hidden">
      <Editor
        height="100%"
        theme="vs-dark"
        value={code}
        onChange={(value) => setCode(value ?? "")}
        language={selectedLanguage === "cuda" ? "cpp" : "python"}
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
