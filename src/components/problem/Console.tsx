import { Box, Text, VStack, HStack } from "@chakra-ui/react";
import { SampleStatus, type SampleStatusType } from "~/types/submission";

type ConsoleTheme = {
  label: string;
  color: string;
  bg: string;
};

type ThemeKey =
  | "ERROR"
  | "PASSED"
  | "FAILED"
  | "INPUT"
  | "OUTPUT"
  | "STDOUT"
  | "STDERR"
  | "IN_QUEUE"
  | "COMPILING"
  | "RUNNING"
  | "LOG";

const CONSOLE_THEMES: Record<ThemeKey, ConsoleTheme> = {
  ERROR: {
    label: "Error",
    color: "#FF5D5D",
    bg: "#351B1B",
  },
  PASSED: {
    label: "Passed",
    color: "#4EC9B0",
    bg: "#1B352B",
  },
  FAILED: {
    label: "Failed",
    color: "#FF5D5D",
    bg: "#351B1B",
  },
  INPUT: {
    label: "Input",
    color: "#569CD6",
    bg: "#1B1B35",
  },
  OUTPUT: {
    label: "Output",
    color: "#569CD6",
    bg: "#1B1B35",
  },
  STDOUT: {
    label: "Stdout",
    color: "#569CD6",
    bg: "#1B1B35",
  },
  STDERR: {
    label: "Stderr",
    color: "#FF5D5D",
    bg: "#351B1B",
  },
  IN_QUEUE: {
    label: "Status",
    color: "#4FC1FF",
    bg: "#1B2535",
  },
  COMPILING: {
    label: "Status",
    color: "#4FC1FF",
    bg: "#1B2535",
  },
  RUNNING: {
    label: "Status",
    color: "#4FC1FF",
    bg: "#1B2535",
  },
  LOG: {
    label: "Log",
    color: "#858585",
    bg: "#1A1A1A",
  },
};

const getConsoleTheme = (line: string): ConsoleTheme => {
  const lowercaseLine = line.toLowerCase();

  // Handle error messages
  if (
    lowercaseLine.includes("error occurred") ||
    lowercaseLine.includes("compilation error") ||
    lowercaseLine.includes("runtime error")
  ) {
    return CONSOLE_THEMES.ERROR;
  }

  // Handle test results
  if (lowercaseLine.includes("test passed")) return CONSOLE_THEMES.PASSED;
  if (lowercaseLine.includes("test failed")) return CONSOLE_THEMES.FAILED;

  // Handle status messages
  if (lowercaseLine.includes("in queue")) return CONSOLE_THEMES.IN_QUEUE;
  if (lowercaseLine.includes("compiling")) return CONSOLE_THEMES.COMPILING;
  if (lowercaseLine.includes("running")) return CONSOLE_THEMES.RUNNING;

  // Handle block headers
  if (line.startsWith("Input")) return CONSOLE_THEMES.INPUT;
  if (line.startsWith("Output")) {
    return CONSOLE_THEMES.OUTPUT;
  }
  if (line.startsWith("Stdout")) return CONSOLE_THEMES.STDOUT;
  if (line.startsWith("Stderr")) return CONSOLE_THEMES.STDERR;

  // Default to LOG theme
  return CONSOLE_THEMES.LOG;
};

type ConsoleProps = {
  output: string[];
};

const ResizableConsole = ({ output }: ConsoleProps) => {
  return (
    <Box
      w="100%"
      h="100%"
      fontFamily="JetBrains Mono, monospace"
      fontSize="sm"
      bg="#111111"
      overflow="hidden"
      borderRadius="xl"
    >
      <Box color="#D4D4D4" p={3} h="100%" overflowY="auto">
        {output.length === 0 ? (
          <Text color="#858585" fontStyle="italic">
            Console output will appear here...
          </Text>
        ) : (
          <VStack align="start" spacing={2}>
            {output.map((line, i) => {
              const theme = getConsoleTheme(line);
              return (
                <HStack key={i} align="start" spacing={2} w="full">
                  <Box
                    px={2}
                    py={0.5}
                    minW="70px"
                    fontWeight="bold"
                    color={theme.color}
                    bg={theme.bg}
                    borderRadius="sm"
                    fontSize="xs"
                    textAlign="center"
                  >
                    {theme.label}
                  </Box>
                  <Text whiteSpace="pre-wrap" color="#D4D4D4">
                    {line}
                  </Text>
                </HStack>
              );
            })}
          </VStack>
        )}
      </Box>
    </Box>
  );
};

export default ResizableConsole;
