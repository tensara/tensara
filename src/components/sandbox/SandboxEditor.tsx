// sandbox/index.tsx
import { useState, useRef, useEffect } from "react";
import {
  Box,
  Button,
  HStack,
  VStack,
  Text,
  IconButton,
  Icon,
  Heading,
} from "@chakra-ui/react";
import {
  FiPlus,
  FiShare2,
  FiArrowLeft,
  FiChevronRight,
  FiChevronLeft,
  FiFile,
  FiChevronDown,
} from "react-icons/fi";
import { FaExclamationCircle } from "react-icons/fa";
import { FileExplorer } from "./FileExplorer";
import type { SandboxFile } from "~/types/misc";
import { type ProgrammingLanguage } from "~/types/misc";
import CodeEditor from "~/components/problem/CodeEditor";
import VerticalSplitPanel from "~/components/problem/VerticalSplitPanel";
import { Menu, MenuButton, MenuList, MenuItem } from "@chakra-ui/react";
import { GPU_DISPLAY_NAMES } from "~/constants/gpu";
import { LANGUAGE_DISPLAY_NAMES } from "~/constants/language";
import { useToast } from "@chakra-ui/react";
import { useHotkey } from "~/hooks/useHotKey";
import { SandboxStatus } from "~/types/submission";
import {
  loadVimModePreference,
  saveVimModePreference,
} from "~/utils/localStorage";

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
  content?: string;
}

interface TerminalLine {
  id: string;
  type:
    | "stdout"
    | "stderr"
    | "info"
    | "success"
    | "error"
    | "compiling"
    | "warning";
  content: string;
  timestamp: number;
}

export default function Sandbox({
  files,
  setFiles,
  workspaceName,
  readOnly,
  onBeforeRun,
  onBeforeShare,
}: {
  files: SandboxFile[];
  setFiles: (f: SandboxFile[]) => void;
  main: string;
  setMain: (m: string) => void;
  onSave: () => Promise<void>;
  workspaceName: string;
  onDelete: () => void;
  onRename: (newName: string) => void;
  readOnly: boolean;
  onBeforeRun?: () => Promise<void>;
  onBeforeShare?: () => Promise<void>;
}) {
  const toast = useToast();
  const [activeIndex, setActiveIndex] = useState(0);
  const [isRunning, setIsRunning] = useState(false);
  const [terminalLines, setTerminalLines] = useState<TerminalLine[]>([]);
  const [isFileExplorerCollapsed, setIsFileExplorerCollapsed] = useState(true);
  // const [terminalStatus, setTerminalStatus] = useState<string>("");
  // const [gpuType, setGpuType] = useState("T4");
  const activeFile = files[activeIndex] ?? files[0];
  const terminalRef = useRef<HTMLDivElement>(null);
  const abortControllerRef = useRef<AbortController | null>(null);
  const [ptxContent, setPtxContent] = useState<string | null>(null);
  const [sassContent, setSassContent] = useState<string | null>(null);
  const [ptxDirty, setPtxDirty] = useState(false);
  const [sassDirty, setSassDirty] = useState(false);
  const [isVimModeEnabled, setIsVimModeEnabled] = useState(false);
  const [hasLoadedVimPreference, setHasLoadedVimPreference] = useState(false);
  const [selectedLanguage, setSelectedLanguage] =
    useState<ProgrammingLanguage>("cuda");

  useHotkey("meta+enter", () => {
    if (isRunning) {
      toast({
        title: "Already running",
        description: "Please wait for the execution to complete",
        status: "error",
        duration: 5000,
        isClosable: true,
      });
      return;
    }
    void runCode();
  });
  useHotkey(
    "meta+shift+v",
    () => {
      setIsVimModeEnabled((prev) => !prev);
    },
    { enabled: hasLoadedVimPreference }
  );
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

  useEffect(() => {
    const stored = loadVimModePreference();
    if (stored !== null) {
      setIsVimModeEnabled(stored);
    }
    setHasLoadedVimPreference(true);
  }, []);

  useEffect(() => {
    if (!hasLoadedVimPreference) return;
    saveVimModePreference(isVimModeEnabled);
  }, [isVimModeEnabled, hasLoadedVimPreference]);

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

    // Save to database before running
    if (onBeforeRun) {
      await onBeforeRun();
    }

    setIsRunning(true);
    setTerminalLines([]);
    setPtxContent(null);
    setSassContent(null);
    setPtxDirty(false);
    setSassDirty(false);
    // setTerminalStatus("Starting...");

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
          language: selectedLanguage,
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

  const getLanguageDisplay = (language: ProgrammingLanguage) =>
    LANGUAGE_DISPLAY_NAMES[language] ?? language.toUpperCase();

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
        // setTerminalStatus("In Queue");
        addTerminalLine("info", "⏳ Submission queued...");
        break;

      case "COMPILING":
        // setTerminalStatus("Compiling");
        addTerminalLine(
          "compiling",
          `🔨 Compiling ${getLanguageDisplay(selectedLanguage)} code...`
        );
        break;

      case "SANDBOX_RUNNING":
        // setTerminalStatus("Running");
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
        // setTerminalStatus("Success");
        addTerminalLine(
          "success",
          `✅ Program completed successfully (exit code: ${data.return_code})`
        );
        break;

      case "SANDBOX_ERROR":
        // setTerminalStatus("Error");
        addTerminalLine(
          "error",
          `❌ Error: ${data.message ?? "Unknown error"}`
        );
        if (data.details) {
          addTerminalLine("error", data.details);
        }
        break;

      case "COMPILE_ERROR":
        // setTerminalStatus("Compile Error");
        addTerminalLine(
          "error",
          `❌ Compile Error: ${data.message ?? "Unknown compile error"}`
        );
        if (data.details) {
          addTerminalLine("error", data.details);
        }
        break;

      case "SANDBOX_TIMEOUT":
        // setTerminalStatus("Timeout");
        addTerminalLine(
          "error",
          `⏱️ Execution timeout: ${data.message ?? "Unknown timeout"}`
        );
        break;
      case SandboxStatus.PTX:
        setPtxContent(data.content ?? null);
        setPtxDirty(false);
        addTerminalLine("info", "📦 PTX generated");
        break;
      case SandboxStatus.SASS:
        setSassContent(data.content ?? null);
        setSassDirty(false);
        addTerminalLine("info", "🧱 SASS generated");
        break;
      case SandboxStatus.WARNING:
        addTerminalLine(
          "warning",
          `⚠️ ${data.message ?? data.details ?? "Warning received"}`
        );
        break;

      default:
        if (event.includes("ERROR")) {
          // setTerminalStatus("Error");
          addTerminalLine(
            "error",
            `❌ ${data.error ?? data.message ?? "Unknown error"}`
          );
        }
    }
  };

  const getNewFileName = (index: number) => {
    const ext = selectedLanguage === "mojo" ? "mojo" : "cu";
    return `file${index}.${ext}`;
  };

  const updateFile = (content: string) => {
    const updated = [...files];
    if (updated[activeIndex]) {
      updated[activeIndex].content = content;
      setFiles(updated);
      if (!ptxDirty && ptxContent !== null) {
        setPtxDirty(true);
      }
      if (!sassDirty && sassContent !== null) {
        setSassDirty(true);
      }
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
      case "warning":
        return "yellow.300";
      default:
        return "gray.400";
    }
  };

  return (
    <VStack w="100%" h="100%" spacing={0}>
      {/* Mobile warning - only show on mobile */}
      <Box
        display={{ base: "flex", md: "none" }}
        w="100%"
        h="100%"
        alignItems="center"
        justifyContent="center"
        p={6}
      >
        <Box w="100%" maxW="400px" p={6} bg="whiteAlpha.50" borderRadius="xl">
          <VStack spacing={4} align="center">
            <Icon as={FaExclamationCircle} boxSize={10} color="yellow.400" />
            <Heading size="md" textAlign="center">
              Desktop Required for Sandbox
            </Heading>
            <Text textAlign="center" color="whiteAlpha.800">
              For the best coding experience, please switch to a desktop device
              to write and execute your code.
            </Text>
          </VStack>
        </Box>
      </Box>

      {/* Main editor - only show on desktop */}
      <Box
        display={{ base: "none", md: "block" }}
        h="100%"
        w="100%"
        border="1px solid"
        borderColor="brand.dark"
        borderRadius="lg"
        overflow="hidden"
        bg="brand.dark"
      >
        <HStack h="100%" spacing={0} align="stretch">
          {/* Sidebar */}
          <Box
            w={isFileExplorerCollapsed ? "50px" : "240px"}
            h="100%"
            bg="brand.dark"
            borderRight="1px solid"
            borderColor="brand.dark"
            transition="width 0.3s ease"
          >
            <VStack spacing={0} p={4} h="100%">
              {/* Toggle button */}
              <HStack
                justify={isFileExplorerCollapsed ? "center" : "space-between"}
                w="100%"
                mb={2}
              >
                <IconButton
                  icon={
                    isFileExplorerCollapsed ? (
                      <Icon as={FiChevronRight} />
                    ) : (
                      <Icon as={FiChevronLeft} />
                    )
                  }
                  aria-label={
                    isFileExplorerCollapsed
                      ? "Expand File Explorer"
                      : "Collapse File Explorer"
                  }
                  variant="ghost"
                  onClick={() =>
                    setIsFileExplorerCollapsed(!isFileExplorerCollapsed)
                  }
                  size="sm"
                  color="gray.400"
                  _hover={{
                    color: "white",
                    bg: "whiteAlpha.100",
                    transition: "all 0.7s ease",
                  }}
                  _focus={{ color: "gray.400", boxShadow: "none" }}
                />

                {!isFileExplorerCollapsed && (
                  <>
                    <Text color="gray.400" fontSize="sm" fontWeight="medium">
                      Files
                    </Text>
                    {!readOnly && (
                      <Menu>
                        <MenuButton
                          as={IconButton}
                          icon={<FiPlus />}
                          size="xs"
                          variant="ghost"
                          color="gray.400"
                          _hover={{ color: "white" }}
                          _active={{ color: "gray.400", boxShadow: "none" }}
                          aria-label="Add File"
                        />
                        <MenuList
                          bg="brand.secondary"
                          border="none"
                          p={0}
                          borderRadius="md"
                          minW="120px"
                        >
                          <MenuItem
                            bg="brand.secondary"
                            fontSize="sm"
                            _hover={{ bg: "whiteAlpha.100" }}
                            borderRadius="md"
                            onClick={() => {
                              const name = getNewFileName(files.length);
                              setFiles([...files, { name, content: "" }]);
                              setActiveIndex(files.length);
                            }}
                          >
                            New File
                          </MenuItem>
                          <MenuItem
                            bg="brand.secondary"
                            fontSize="sm"
                            _hover={{ bg: "whiteAlpha.100" }}
                            borderRadius="md"
                            onClick={() => uploadRef.current?.click()}
                          >
                            Upload File
                          </MenuItem>
                        </MenuList>
                      </Menu>
                    )}
                  </>
                )}
              </HStack>

              {/* Hidden file input */}
              <input
                type="file"
                accept=".cu,.mojo,.cpp,.h,.txt"
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

              {/* File Explorer */}
              {!isFileExplorerCollapsed && (
                <Box w="100%" flex={1} overflowY="auto">
                  <FileExplorer
                    files={files}
                    active={activeIndex}
                    onOpen={setActiveIndex}
                    onRename={(i: number, name: string) => {
                      if (readOnly) return;
                      const updated = [...files];
                      if (updated[i]) {
                        updated[i].name = name;
                        setFiles(updated);
                      }
                    }}
                    onDelete={(i: number) => {
                      if (readOnly || files.length === 1) return;
                      const updated = files.filter((_, idx) => idx !== i);
                      setFiles(updated);
                      setActiveIndex(i === 0 ? 0 : i - 1);
                    }}
                    onDownload={downloadFile}
                    readOnly={readOnly ?? false}
                  />
                </Box>
              )}

              {/* Collapsed state - show three icons */}
              {isFileExplorerCollapsed && (
                <VStack
                  spacing={4}
                  w="100%"
                  align="center"
                  flex={1}
                  justify="start"
                  pt={4}
                >
                  <IconButton
                    icon={<Icon as={FiFile} />}
                    aria-label="Files"
                    variant="ghost"
                    onClick={() => setIsFileExplorerCollapsed(false)}
                    size="sm"
                    color="gray.400"
                    _hover={{ color: "white", bg: "whiteAlpha.100" }}
                  />
                  {!readOnly && (
                    <Menu>
                      <MenuButton
                        as={IconButton}
                        icon={<FiPlus />}
                        aria-label="Add File"
                        variant="ghost"
                        size="sm"
                        color="gray.400"
                        _hover={{ color: "white", bg: "whiteAlpha.100" }}
                        _active={{ color: "gray.400", boxShadow: "none" }}
                      />
                      <MenuList
                        bg="brand.secondary"
                        border="none"
                        p={0}
                        borderRadius="md"
                        minW="120px"
                      >
                        <MenuItem
                          bg="brand.secondary"
                          fontSize="sm"
                          borderRadius="md"
                          _hover={{ bg: "whiteAlpha.100" }}
                          onClick={() => {
                            const name = getNewFileName(files.length);
                            setFiles([...files, { name, content: "" }]);
                            setActiveIndex(files.length);
                          }}
                        >
                          New File
                        </MenuItem>
                        <MenuItem
                          bg="brand.secondary"
                          fontSize="sm"
                          borderRadius="md"
                          _hover={{ bg: "whiteAlpha.100" }}
                          onClick={() => uploadRef.current?.click()}
                        >
                          Upload File
                        </MenuItem>
                      </MenuList>
                    </Menu>
                  )}
                  <IconButton
                    icon={<Icon as={FiArrowLeft} />}
                    aria-label="Back to Workspaces"
                    variant="ghost"
                    onClick={() => (window.location.href = "/sandbox")}
                    size="sm"
                    color="gray.400"
                    _hover={{ color: "white", bg: "whiteAlpha.100" }}
                  />
                </VStack>
              )}
            </VStack>
          </Box>

          {/* Main Content */}
          <VStack h="100%" flex={1} spacing={0}>
            {/* Workspace Title and Controls */}
            <HStack
              w="100%"
              px={4}
              py={3}
              bg="brand.dark"
              borderBottom="1px solid"
              borderColor="brand.dark"
              justify="space-between"
              align="center"
            >
              <HStack spacing={4}>
                <Text color="white" fontWeight="600" fontSize="2xl">
                  {workspaceName}
                </Text>
                {!readOnly && (
                  <Button
                    size="sm"
                    variant="ghost"
                    onClick={async () => {
                      try {
                        if (onBeforeShare) {
                          await onBeforeShare();
                        }

                        const res = await fetch("/api/snapshot/create", {
                          method: "POST",
                          headers: { "Content-Type": "application/json" },
                          body: JSON.stringify({
                            files,
                            main: files[activeIndex]?.name,
                          }),
                        });
                        const { id } = (await res.json()) as { id: string };
                        const url = `${window.location.origin}/snapshot/${id}`;
                        await navigator.clipboard.writeText(url);
                        toast({
                          title: "Snapshot link copied to clipboard!",
                          status: "success",
                          duration: 3000,
                          isClosable: true,
                        });
                      } catch (e) {
                        console.error(e);
                        toast({
                          title: "Failed to create snapshot.",
                          status: "error",
                          duration: 3000,
                          isClosable: true,
                        });
                      }
                    }}
                    leftIcon={<Icon as={FiShare2} />}
                    borderRadius="lg"
                    bg="rgba(234, 179, 8, 0.1)"
                    color="rgb(234, 179, 8)"
                    _hover={{
                      bg: "rgba(234, 179, 8, 0.2)",
                      color: "rgb(234, 179, 8)",
                    }}
                    _active={{
                      bg: "rgba(234, 179, 8, 0.25)",
                    }}
                    transition="all 0.5s ease"
                  >
                    Share
                  </Button>
                )}
              </HStack>
              <HStack spacing={3}>
                <Menu>
                  <MenuButton
                    size="sm"
                    as={Button}
                    rightIcon={<FiChevronDown size={12} color="#a1a1aa" />}
                    bg="whiteAlpha.50"
                    _hover={{ bg: "whiteAlpha.100", borderColor: "gray.600" }}
                    _active={{ bg: "whiteAlpha.150" }}
                    _focus={{ borderColor: "blue.500", boxShadow: "none" }}
                    color="white"
                    w={{ base: "140px", md: "160px" }}
                    fontWeight="normal"
                    textAlign="left"
                    justifyContent="flex-start"
                    borderRadius="lg"
                  >
                    {getLanguageDisplay(selectedLanguage)}
                  </MenuButton>
                  <MenuList
                    bg="brand.secondary"
                    borderColor="gray.800"
                    p={0}
                    borderRadius="lg"
                    minW="160px"
                  >
                    {(["cuda", "mojo"] satisfies ProgrammingLanguage[]).map(
                      (lang) => (
                        <MenuItem
                          key={lang}
                          onClick={() => setSelectedLanguage(lang)}
                          bg="brand.secondary"
                          _hover={{ bg: "gray.700" }}
                          color="white"
                          borderRadius="lg"
                          fontSize="sm"
                        >
                          {getLanguageDisplay(lang)}
                        </MenuItem>
                      )
                    )}
                  </MenuList>
                </Menu>
                <Text
                  color="gray.400"
                  fontSize="sm"
                  fontWeight="medium"
                  px={4}
                  h="32px"
                  display="flex"
                  alignItems="center"
                  bg="whiteAlpha.50"
                  borderRadius="md"
                >
                  {GPU_DISPLAY_NAMES.T4}
                </Text>
                <Button
                  onClick={isRunning ? stopExecution : runCode}
                  bg={isRunning ? "red.500" : "rgba(34, 197, 94, 0.1)"}
                  color={isRunning ? "white" : "rgb(34, 197, 94)"}
                  size="sm"
                  _hover={{
                    bg: isRunning ? "red.600" : "rgba(34, 197, 94, 0.2)",
                    transition: "all 0.5s ease",
                  }}
                  _active={{
                    bg: isRunning ? "red.600" : "rgba(34, 197, 94, 0.25)",
                    transition: "all 0.5s ease",
                  }}
                  px={4}
                  isLoading={isRunning}
                  transition="all 0.5s ease"
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
              </HStack>
            </HStack>

            <Box flex={1} w="100%">
              <VerticalSplitPanel
                topContent={
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
                        <CodeEditor
                          key={`sandbox-editor-${isVimModeEnabled ? "vim" : "std"}`}
                          code={activeFile.content}
                          setCode={updateFile}
                          selectedLanguage={selectedLanguage}
                          isEditable={!readOnly}
                          enablePtxSassView
                          ptxContent={ptxContent}
                          sassContent={sassContent}
                          ptxDirty={ptxDirty}
                          sassDirty={sassDirty}
                          enableVimMode={isVimModeEnabled}
                          onToggleVimMode={setIsVimModeEnabled}
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
                }
                bottomContent={
                  <Box
                    w="100%"
                    h="100%"
                    bg="#111111"
                    borderTop="1px solid"
                    borderColor="brand.dark"
                    borderRadius="lg"
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
                        borderTopRadius="lg"
                      >
                        <Text color="white" fontSize="sm" fontWeight="500">
                          Terminal
                        </Text>
                        <Button
                          onClick={() => {
                            setTerminalLines([]);
                          }}
                          bg="rgba(160, 174, 192, 0.1)"
                          color="rgb(160, 174, 192)"
                          size="sm"
                          _hover={{
                            bg: "rgba(160, 174, 192, 0.2)",
                            transition: "all 0.5s ease",
                          }}
                          _active={{
                            bg: "rgba(160, 174, 192, 0.25)",
                            transition: "all 0.5s ease",
                          }}
                          transition="all 0.5s ease"
                          px={4}
                        >
                          Clear
                        </Button>
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
                }
                initialRatio={65}
                minTopHeight={40}
                minBottomHeight={20}
              />
            </Box>
          </VStack>
        </HStack>
      </Box>
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
    </VStack>
  );
}
