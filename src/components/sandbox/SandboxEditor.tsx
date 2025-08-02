// sandbox/index.tsx
import { useState, useRef, useEffect } from "react";
import Split from "react-split";
import {
  Box,
  Button,
  HStack,
  VStack,
  Text,
  IconButton,
} from "@chakra-ui/react";
import { FiPlay, FiPlus, FiTrash, FiSquare } from "react-icons/fi";
import dynamic from "next/dynamic";
import { FileExplorer } from "./FileExplorer";
import { setupMonaco } from "~/components/sandbox/setupmonaco";
import type { SandboxFile } from "~/types/misc";
import { ChevronLeftIcon } from "@chakra-ui/icons";

// Type definitions for API responses
interface ErrorResponse {
  error?: string;
  message?: string;
}

interface SSEMessage {
  status?: string;
  stream?: "stdout" | "stderr";
  line?: string;
  return_code?: number;
  message?: string;
  details?: string;
  error?: string;
}

const MonacoEditor = dynamic(() => import("@monaco-editor/react"), {
  ssr: false,
});

interface TerminalLine {
  id: string;
  type: "stdout" | "stderr" | "info" | "success" | "error" | "compiling";
  content: string;
  timestamp: number;
}

export default function Sandbox({
  files,
  setFiles,
  onManualSave,
  workspaceName,
}: {
  files: SandboxFile[];
  setFiles: (f: SandboxFile[]) => void;
  main: string;
  setMain: (m: string) => void;
  onSave: () => Promise<void>;
  onManualSave: () => void;
  workspaceName: string;
  onDelete: () => void;
  onRename: (newName: string) => void;
}) {
  const [activeIndex, setActiveIndex] = useState(0);
  const [isRunning, setIsRunning] = useState(false);
  const [terminalLines, setTerminalLines] = useState<TerminalLine[]>([]);
  const [terminalStatus, setTerminalStatus] = useState<string>("");
  // const [gpuType, setGpuType] = useState("T4");
  const activeFile = files[activeIndex] ?? files[0];
  const terminalRef = useRef<HTMLDivElement>(null);
  const abortControllerRef = useRef<AbortController | null>(null);

  useEffect(() => {
    // Auto-scroll terminal to bottom when new lines are added
    if (terminalRef.current) {
      terminalRef.current.scrollTop = terminalRef.current.scrollHeight;
    }
  }, [terminalLines]);

  useEffect(() => {
    // Cleanup on unmount
    return () => {
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
    };
  }, []);

  const addTerminalLine = (type: TerminalLine["type"], content: string) => {
    const newLine: TerminalLine = {
      id: `${Date.now()}-${Math.random()}`,
      type,
      content,
      timestamp: Date.now(),
    };
    setTerminalLines((prev) => [...prev, newLine]);
  };

  const runCode = async () => {
    if (!activeFile || isRunning) return;

    setIsRunning(true);
    setTerminalLines([]);
    setTerminalStatus("Starting...");

    // Create a new AbortController for this request
    abortControllerRef.current = new AbortController();

    try {
      const response = await fetch("/api/submissions/sandbox", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          code: activeFile.content,
        }),
        signal: abortControllerRef.current.signal,
      });

      if (!response.ok) {
        const errorData = (await response.json()) as ErrorResponse;
        addTerminalLine(
          "error",
          `❌ Error: ${errorData.error ?? "Failed to start execution"}`
        );
        setIsRunning(false);
        return;
      }

      const reader = response.body?.getReader();
      if (!reader) {
        addTerminalLine("error", "❌ No response stream available");
        setIsRunning(false);
        return;
      }

      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";

        for (const line of lines) {
          if (line.startsWith("event:")) {
            continue;
          }

          if (line.startsWith("data:")) {
            try {
              const data = JSON.parse(line.slice(5)) as SSEMessage;
              handleSSEMessage(data.status ?? "", data);
            } catch (e) {
              console.error("Failed to parse SSE data:", e);
            }
          }
        }
      }
    } catch (error: unknown) {
      if (error instanceof Error && error.name !== "AbortError") {
        console.error("Execution error:", error);
        addTerminalLine("error", `❌ Connection error: ${error.message}`);
      }
    } finally {
      setIsRunning(false);
      abortControllerRef.current = null;
    }
  };

  const stopExecution = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      addTerminalLine("info", "🛑 Execution stopped by user");
      setIsRunning(false);
    }
  };

  const handleSSEMessage = (event: string, data: SSEMessage) => {
    switch (event) {
      case "IN_QUEUE":
        setTerminalStatus("In Queue");
        addTerminalLine("info", "⏳ Submission queued...");
        break;

      case "COMPILING":
        setTerminalStatus("Compiling");
        addTerminalLine("compiling", "🔨 Compiling CUDA code...");
        break;

      case "SANDBOX_RUNNING":
        setTerminalStatus("Running");
        addTerminalLine("info", "▶️  Executing program...");
        break;

      case "SANDBOX_OUTPUT":
        if (data.stream === "stdout") {
          addTerminalLine("stdout", data.line ?? "");
        } else if (data.stream === "stderr") {
          addTerminalLine("stderr", data.line ?? "");
        }
        break;

      case "SANDBOX_SUCCESS":
        setTerminalStatus("Success");
        addTerminalLine(
          "success",
          `✅ Program completed successfully (exit code: ${data.return_code})`
        );
        break;

      case "SANDBOX_ERROR":
        setTerminalStatus("Error");
        addTerminalLine(
          "error",
          `❌ Error: ${data.message ?? "Unknown error"}`
        );
        if (data.details) {
          addTerminalLine("error", data.details);
        }
        break;

      case "COMPILE_ERROR":
        setTerminalStatus("Compile Error");
        addTerminalLine(
          "error",
          `❌ Compile Error: ${data.message ?? "Unknown compile error"}`
        );
        if (data.details) {
          addTerminalLine("error", data.details);
        }
        break;

      case "SANDBOX_TIMEOUT":
        setTerminalStatus("Timeout");
        addTerminalLine(
          "error",
          `⏱️ Execution timeout: ${data.message ?? "Unknown timeout"}`
        );
        break;

      default:
        if (event.includes("ERROR")) {
          setTerminalStatus("Error");
          addTerminalLine(
            "error",
            `❌ ${data.error ?? data.message ?? "Unknown error"}`
          );
        }
    }
  };

  const updateFile = (content: string) => {
    const updated = [...files];
    if (updated[activeIndex]) {
      updated[activeIndex].content = content;
      setFiles(updated);
    }
  };

  const uploadRef = useRef<HTMLInputElement>(null);

  const downloadFile = (index: number) => {
    const file = files[index];
    if (!file) return;
    const blob = new Blob([file.content], { type: "text/plain" });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = file.name;
    link.click();
  };

  const getTerminalLineColor = (type: TerminalLine["type"]) => {
    switch (type) {
      case "stdout":
        return "gray.300";
      case "stderr":
        return "red.400";
      case "info":
        return "blue.400";
      case "success":
        return "green.400";
      case "error":
        return "red.500";
      case "compiling":
        return "yellow.400";
      default:
        return "gray.400";
    }
  };
  return (
    <Box
      h="100%"
      border="1px solid"
      borderColor="brand.dark"
      borderRadius="lg"
      overflow="hidden"
      bg="brand.dark"
    >
      <HStack h="100%" spacing={0} align="stretch">
        {/* Sidebar */}
        <Box
          w="240px"
          h="100%"
          bg="brand.dark"
          borderRight="1px solid"
          borderColor="brand.dark"
        >
          <VStack spacing={0} p={4}>
            <HStack justify="space-between" w="100%" mb={4}>
              <IconButton
                icon={<ChevronLeftIcon />}
                aria-label="Back"
                variant="ghost"
                onClick={() => (window.location.href = "/sandbox/")}
              />

              <Text color="white" fontWeight="600" fontSize="sm">
                {workspaceName}
              </Text>
              <IconButton
                icon={<FiPlus />}
                bg="green.500"
                color="white"
                size="sm"
                borderRadius="md"
                _hover={{ bg: "green.600" }}
                onClick={() => {
                  const name = `file${files.length}.cu`;
                  setFiles([...files, { name, content: "" }]);
                  setActiveIndex(files.length);
                }}
                aria-label="Add File"
                transition="all 0.2s"
                _active={{ transform: "scale(0.95)" }}
              />
            </HStack>
            <Box px={2} w="100%" flex={1} overflowY="auto">
              <Button
                w="100%"
                onClick={() => uploadRef.current?.click()}
                bg="gray.700"
                color="white"
                size="sm"
                borderRadius="md"
                _hover={{ bg: "gray.600", transform: "translateY(-1px)" }}
                _active={{ transform: "translateY(0)" }}
                mb={4}
                transition="all 0.2s"
              >
                Upload
              </Button>
            </Box>
          </VStack>
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
          <Box px={4} pb={4} flex={1} overflowY="auto">
            <FileExplorer
              files={files}
              active={activeIndex}
              onOpen={setActiveIndex}
              onRename={(i: number, name: string) => {
                const updated = [...files];
                if (updated[i]) {
                  updated[i].name = name;
                  setFiles(updated);
                }
              }}
              onDelete={(i: number) => {
                if (files.length === 1) return;
                const updated = files.filter((_, idx) => idx !== i);
                setFiles(updated);
                setActiveIndex((_) => (i === 0 ? 0 : i - 1));
              }}
              onDownload={downloadFile}
            />
          </Box>
        </Box>
        {/* Main Content */}
        <VStack h="100%" flex={1} spacing={0}>
          {/* Editor + Terminal */}
          <Split
            className="split"
            direction="vertical"
            sizes={[65, 35]}
            minSize={100}
            gutterSize={6}
            style={{ height: "100%", width: "100%" }}
          >
            {/* Editor */}
            <Box w="100%" h="100%" bg="gray.900" position="relative">
              {activeFile ? (
                <Box
                  key={activeFile.name}
                  position="absolute"
                  top={0}
                  left={0}
                  right={0}
                  bottom={0}
                  opacity={1}
                  animation="fadeIn 0.2s ease-out"
                >
                  <MonacoEditor
                    theme="tensara-dark"
                    language="cpp"
                    value={activeFile.content}
                    onChange={(val) => updateFile(val ?? "")}
                    beforeMount={setupMonaco}
                    options={{
                      fontSize: 14,
                      minimap: { enabled: false },
                      tabSize: 2,
                      automaticLayout: true,
                      scrollBeyondLastLine: false,
                      padding: { top: 16, bottom: 16 },
                      fontFamily: "JetBrains Mono, monospace",
                    }}
                  />
                </Box>
              ) : (
                <Box
                  h="100%"
                  display="flex"
                  alignItems="center"
                  justifyContent="center"
                >
                  <Text color="gray.400">No file selected</Text>
                </Box>
              )}
            </Box>
            {/* Terminal */}
            <Box
              w="100%"
              h="100%"
              bg="#111111"
              borderTop="1px solid"
              borderColor="brand.dark"
            >
              <VStack h="100%" w="100%" spacing={0}>
                {/* Terminal Header */}
                <HStack
                  w="100%"
                  px={4}
                  py={2}
                  bg="#111111"
                  borderBottom="1px solid"
                  borderColor="brand.dark"
                  justify="space-between"
                >
                  <Text color="white" fontSize="sm" fontWeight="500">
                    Terminal
                  </Text>
                  <HStack spacing={3}>
                    <Button
                      onClick={onManualSave}
                      bg="blue.500"
                      color="white"
                      size="sm"
                      _hover={{ bg: "blue.600", transform: "translateY(-1px)" }}
                      _active={{ transform: "translateY(0)" }}
                      transition="all 0.15s ease"
                    >
                      Save
                    </Button>
                    <Button
                      leftIcon={isRunning ? <FiSquare /> : <FiPlay />}
                      onClick={isRunning ? stopExecution : runCode}
                      bg={isRunning ? "red.500" : "green.500"}
                      color="white"
                      size="sm"
                      _hover={{
                        bg: isRunning ? "red.600" : "green.600",
                        transform: "translateY(-1px)",
                      }}
                      _active={{ transform: "translateY(0)" }}
                      isLoading={isRunning}
                      transition="all 0.15s ease"
                      position="relative"
                      _before={
                        isRunning
                          ? {
                              content: '""',
                              position: "absolute",
                              top: 0,
                              left: 0,
                              right: 0,
                              bottom: 0,
                              borderRadius: "md",
                              bg: "red.400",
                              opacity: 0.3,
                              animation: "pulse 2s infinite",
                            }
                          : {}
                      }
                    >
                      {isRunning ? "Stop" : "Run"}
                    </Button>
                    <Button
                      leftIcon={<FiTrash />}
                      aria-label="Clear terminal"
                      size="sm"
                      variant="ghost"
                      color="gray.400"
                      _hover={{ color: "white", bg: "whiteAlpha.100" }}
                      _active={{ transform: "scale(0.95)" }}
                      onClick={() => {
                        setTerminalLines([]);
                        setTerminalStatus("");
                      }}
                      transition="all 0.15s ease"
                    >
                      Clear
                    </Button>
                  </HStack>
                </HStack>
                {/* Terminal Content */}
                <Box
                  ref={terminalRef}
                  flex={1}
                  w="100%"
                  overflowY="auto"
                  px={4}
                  py={3}
                  fontFamily="JetBrains Mono, monospace"
                  fontSize="13px"
                >
                  {terminalLines.length === 0 ? (
                    <Text color="gray.500" fontStyle="italic">
                      user~
                    </Text>
                  ) : (
                    <VStack align="start" spacing={0.5} w="100%">
                      {terminalLines.map((line) => (
                        <Box
                          key={line.id}
                          w="100%"
                          fontFamily="JetBrains Mono, monospace"
                          whiteSpace="pre-wrap"
                          wordBreak="break-word"
                          animation="slideIn 0.15s ease-out"
                        >
                          <Text
                            color={getTerminalLineColor(line.type)}
                            fontSize="13px"
                          >
                            {line.content}
                          </Text>
                        </Box>
                      ))}
                    </VStack>
                  )}
                </Box>
              </VStack>
            </Box>
          </Split>
        </VStack>
      </HStack>
      <style jsx global>{`
        @keyframes fadeIn {
          from {
            opacity: 0;
          }
          to {
            opacity: 1;
          }
        }

        @keyframes slideIn {
          from {
            opacity: 0;
            transform: translateX(-10px);
          }
          to {
            opacity: 1;
            transform: translateX(0);
          }
        }

        @keyframes pulse {
          0% {
            opacity: 0.3;
          }
          50% {
            opacity: 0.6;
          }
          100% {
            opacity: 0.3;
          }
        }
      `}</style>
    </Box>
  );
}
