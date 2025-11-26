import { useEffect, useRef } from "react";
import { Box, HStack, Text, VStack, Button } from "@chakra-ui/react";

export interface TerminalLine {
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

interface SandboxTerminalProps {
  isRunning: boolean;
  lines: TerminalLine[];
  status?: string;
  emptyMessage?: string;
  onClear?: () => void;
}

export default function SandboxTerminal({
  lines,
  emptyMessage = "Terminal output will appear here...",
  onClear,
}: SandboxTerminalProps) {
  const terminalRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (terminalRef.current) {
      terminalRef.current.scrollTop = terminalRef.current.scrollHeight;
    }
  }, [lines]);

  const clearTerminal = () => {
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
      case "warning":
        return "yellow.300";
      default:
        return "gray.400";
    }
  };

  return (
    <VStack h="100%" w="100%" spacing={0} borderRadius="md">
      {/* Terminal Header */}
      <HStack
        w="100%"
        px={4}
        py={2}
        borderBottom="1px solid"
        borderColor="whiteAlpha.50"
        justify="space-between"
        borderTopRadius="md"
      >
        <Text color="white" fontSize="sm" fontWeight="500">
          Terminal
        </Text>
        <Button
          onClick={() => {
            clearTerminal();
          }}
          bg="whiteAlpha.100"
          color="gray.400"
          size="sm"
          fontSize="xs"
          _hover={{
            bg: "whiteAlpha.200",
            transition: "all 0.5s ease",
          }}
          _active={{
            bg: "whiteAlpha.200",
            transition: "all 0.5s ease",
          }}
          transition="all 0.5s ease"
          px={4}
          borderRadius="md"
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
            {emptyMessage}
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
