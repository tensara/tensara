// sandbox/index.tsx
import { useState } from "react";
import Split from "react-split";
import { Box, Button, HStack, VStack, Text, IconButton } from "@chakra-ui/react";
import { FiPlay, FiPlus, FiTrash } from "react-icons/fi";
import { FaFileAlt } from "react-icons/fa";
import dynamic from "next/dynamic";
import SandboxConsole from "~/components/sandbox/console";
import { FileExplorer } from "./FileExplorer";
import { useRef } from "react";


const MonacoEditor = dynamic(() => import("@monaco-editor/react"), { ssr: false });

const defaultFile = {
  name: "main.cu",
  content: `#include <stdio.h>\n
__global__ void hello() {
  printf("Hello, world!\\n");
}
int main() {
  hello<<<1, 1>>>();
  cudaDeviceSynchronize();
  return 0;
}`,
};

export default function Sandbox() {
  const [files, setFiles] = useState([{ ...defaultFile }]);
  const [activeIndex, setActiveIndex] = useState(0);
  const [output, setOutput] = useState(null);

  const activeFile = files[activeIndex];

  const runCode = async () => {
    const res = await fetch("/api/sandbox/run", {
      method: "POST",
      body: JSON.stringify({ files, main: activeFile.name }),
    });
    const data = await res.json();
    setOutput(data);
  };

  const updateFile = (content: string) => {
    const updated = [...files];
    updated[activeIndex].content = content;
    setFiles(updated);
  };

  const uploadRef = useRef<HTMLInputElement>(null);

const downloadFile = (index: number) => {
  const file = files[index];
  const blob = new Blob([file.content], { type: "text/plain" });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = file.name;
  link.click();
};


  return (
    <HStack h="100vh" spacing={0} align="start">
      {/* Sidebar */}
    <Box w="240px" h="100%" bg="#1e1e1e" px={4} py={3}>
  <HStack justify="space-between" mb={3}>
    <Text color="white" fontWeight="bold">Files</Text>
    <IconButton
      icon={<FiPlus />}
      size="sm"
      onClick={() => {
        const name = `file${files.length}.cu`;
        setFiles([...files, { name, content: "" }]);
        setActiveIndex(files.length);
      }}
      aria-label="Add File"
    />
  </HStack>

  <Button
    size="sm"
    mb={2}
    w="100%"
    onClick={() => uploadRef.current?.click()}
    colorScheme="teal"
  >
    Upload
  </Button>

  <input
    type="file"
    accept=".cu"
    ref={uploadRef}
    style={{ display: "none" }}
    onChange={(e) => {
      const file = e.target.files?.[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = (event) => {
        const content = event.target?.result as string;
        setFiles([...files, { name: file.name, content }]);
        setActiveIndex(files.length);
      };
      reader.readAsText(file);
    }}
  />

  <FileExplorer
    files={files}
    active={activeIndex}
    onOpen={setActiveIndex}
    onRename={(i, name) => {
      const updated = [...files];
      updated[i].name = name;
      setFiles(updated);
    }}
    onDelete={(i) => {
      if (files.length === 1) return;
      const updated = files.filter((_, idx) => idx !== i);
      setFiles(updated);
      setActiveIndex(0);
    }}
    onDownload={downloadFile}
  />
</Box>


      {/* Editor + Console */}
      <VStack h="100%" flex={1} spacing={0}>
        <HStack justify="space-between" px={4} py={2} w="100%" bg="#1f1f1f">
          <Text color="#4EC9B0" fontWeight="bold">Tensara Sandbox</Text>
          <Button leftIcon={<FiPlay />} colorScheme="green" onClick={runCode}>
            Run
          </Button>
        </HStack>

        <Split
          className="split"
          direction="vertical"
          sizes={[75, 25]}
          minSize={100}
          gutterSize={6}
          style={{ height: "100%", width: "100%" }}
        >
          <Box w="100%" h="100%">
            <MonacoEditor
              theme="vs-dark"
              language="cpp"
              value={activeFile.content}
              onChange={(val) => updateFile(val ?? "")}
              options={{
                fontSize: 14,
                minimap: { enabled: false },
                tabSize: 2,
                automaticLayout: true,
              }}
            />
          </Box>
          <Box w="100%" h="100%">
            <SandboxConsole output={output} status={output?.success ? "passed" : "failed"} isRunning={false} files={files} />
          </Box>
        </Split>
      </VStack>
    </HStack>
  );
}
