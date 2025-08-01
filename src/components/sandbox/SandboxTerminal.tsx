import { useEffect, useRef, useState } from "react";
import {
  Box,
  HStack,
  Text,
  VStack,
  Spinner,
  IconButton,
} from "@chakra-ui/react";
import { FiTerminal, FiX, FiCheckCircle, FiAlertCircle } from "react-icons/fi";
import { type SandboxOutput } from "~/types/misc";

interface TerminalLine {
  id: string;
  type: "stdout" | "stderr" | "info" | "success" | "error" | "compiling";
  content: string;
  timestamp: number;
}

interface SandboxTerminalProps {
  isRunning: boolean;
  onClear?: () => void;
}

export default function SandboxTerminal({
  isRunning,
  onClear,
}: SandboxTerminalProps) {
  const [lines, setLines] = useState<TerminalLine[]>([]);
  const [status, setStatus] = useState<string>("");
  const terminalRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (terminalRef.current) {
      terminalRef.current.scrollTop = terminalRef.current.scrollHeight;
    }
  }, [lines]);

  const addLine = (type: TerminalLine["type"], content: string) => {
    const newLine: TerminalLine = {
      id: `${Date.now()}-${Math.random()}`,
      type,
      content,
      timestamp: Date.now(),
    };
    setLines((prev) => [...prev, newLine]);
  };

  const handleSSEMessage = (event: string, data: SandboxOutput) => {
    switch (event) {
      case "IN_QUEUE":
        setStatus("In Queue");
        addLine("info", "⏳ Submission queued...");
        break;

      case "COMPILING":
        setStatus("Compiling");
        addLine("compiling", "🔨 Compiling CUDA code...");
        break;

      case "SANDBOX_RUNNING":
        setStatus("Running");
        addLine("info", "▶️  Executing program...");
        break;

      case "SANDBOX_OUTPUT":
        if (data.stdout) {
          addLine("stdout", data.stdout);
        } else if (data.stderr) {
          addLine("stderr", data.stderr);
        }
        break;

      case "SANDBOX_SUCCESS":
        setStatus("Success");
        addLine(
          "success",
          `✅ Program completed successfully (exit code: ${data.return_code})`
        );
        if (data.stdout) {
          data.stdout.split("\n").forEach((line: string) => {
            if (line.trim()) addLine("stdout", line);
          });
        }
        if (data.stderr) {
          data.stderr.split("\n").forEach((line: string) => {
            if (line.trim()) addLine("stderr", line);
          });
        }
        break;

      case "SANDBOX_ERROR":
        setStatus("Error");
        addLine("error", `❌ Error: ${data.stderr ?? "Unknown error"}`);
        break;

      case "SANDBOX_TIMEOUT":
        setStatus("Timeout");
        addLine(
          "error",
          `⏱️ Execution timeout: ${data.stderr ?? "Unknown error"}`
        );
        break;

      default:
        if (event.includes("ERROR")) {
          setStatus("Error");
          addLine("error", `❌ ${data.stderr ?? "Unknown error"}`);
        }
    }
  };

  const clearTerminal = () => {
    setLines([]);
    setStatus("");
    onClear?.();
  };

  const getLineColor = (type: TerminalLine["type"]) => {
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

  const getStatusIcon = () => {
    if (isRunning) {
      return <Spinner size="sm" color="blue.400" />;
    }
    if (status === "Success") {
      return <FiCheckCircle color="rgb(34, 197, 94)" />;
    }
    if (status === "Error" || status === "Timeout") {
      return <FiAlertCircle color="rgb(239, 68, 68)" />;
    }
    return <FiTerminal color="gray.400" />;
  };

  return (
    <VStack h="100%" w="100%" spacing={0} bg="#0d0d0d">
      {/* Terminal Header */}
      <HStack
        w="100%"
        px={4}
        py={2}
        bg="#1a1a1a"
        borderBottom="1px solid"
        borderColor="gray.800"
        justify="space-between"
      >
        <HStack spacing={2}>
          {getStatusIcon()}
          <Text color="gray.300" fontSize="sm" fontWeight="medium">
            Terminal
          </Text>
          {status && (
            <Text color="gray.500" fontSize="xs">
              • {status}
            </Text>
          )}
        </HStack>
        <IconButton
          icon={<FiX />}
          aria-label="Clear terminal"
          size="sm"
          variant="ghost"
          color="gray.400"
          onClick={clearTerminal}
          _hover={{ color: "gray.200", bg: "whiteAlpha.100" }}
        />
      </HStack>

      {/* Terminal Content */}
      <Box
        ref={terminalRef}
        flex={1}
        w="100%"
        overflowY="auto"
        overflowX="hidden"
        px={4}
        py={3}
        fontFamily="'JetBrains Mono', monospace"
        fontSize="13px"
        css={{
          "&::-webkit-scrollbar": {
            width: "8px",
          },
          "&::-webkit-scrollbar-track": {
            background: "transparent",
          },
          "&::-webkit-scrollbar-thumb": {
            background: "#2d2d2d",
            borderRadius: "4px",
          },
          "&::-webkit-scrollbar-thumb:hover": {
            background: "#3d3d3d",
          },
        }}
      >
        {lines.length === 0 ? (
          <Text color="gray.600" fontStyle="italic">
            Terminal output will appear here...
          </Text>
        ) : (
          <VStack align="start" spacing={0.5} w="100%">
            {lines.map((line) => (
              <Box
                key={line.id}
                w="100%"
                fontFamily="'JetBrains Mono', monospace"
                whiteSpace="pre-wrap"
                wordBreak="break-word"
              >
                <Text color={getLineColor(line.type)} fontSize="13px">
                  {line.content}
                </Text>
              </Box>
            ))}
          </VStack>
        )}
      </Box>
    </VStack>
  );
}
