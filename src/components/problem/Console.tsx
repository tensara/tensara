import { Box, Text, VStack, HStack, Spinner } from "@chakra-ui/react";
import { keyframes } from "@emotion/react";
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

const pulseAnimation = keyframes`
  0% { opacity: 0.6; }
  50% { opacity: 1; }
  100% { opacity: 0.6; }
`;

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

  if (
    lowercaseLine.includes("error occurred") ||
    lowercaseLine.includes("compilation error") ||
    lowercaseLine.includes("runtime error")
  ) {
    return CONSOLE_THEMES.ERROR;
  }

  if (lowercaseLine.includes("test passed")) return CONSOLE_THEMES.PASSED;
  if (lowercaseLine.includes("test failed")) return CONSOLE_THEMES.FAILED;

  if (lowercaseLine.includes("in queue")) return CONSOLE_THEMES.IN_QUEUE;
  if (lowercaseLine.includes("compiling")) return CONSOLE_THEMES.COMPILING;
  if (lowercaseLine.includes("running")) return CONSOLE_THEMES.RUNNING;

  if (line.startsWith("Input")) return CONSOLE_THEMES.INPUT;
  if (line.startsWith("Output")) return CONSOLE_THEMES.OUTPUT;
  if (line.startsWith("Stdout")) return CONSOLE_THEMES.STDOUT;
  if (line.startsWith("Stderr")) return CONSOLE_THEMES.STDERR;

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
              const lowercaseLine = line.toLowerCase();
              
              const hasOutput = output.some(l => 
                l.startsWith("Output") || 
                l.toLowerCase().includes("test passed") || 
                l.toLowerCase().includes("test failed") ||
                l.toLowerCase().includes("error")
              );
              
              const isAnimated = !hasOutput && (lowercaseLine.includes("compiling") || lowercaseLine.includes("running") || lowercaseLine.includes("in queue"));
              const isLoading = !hasOutput && (lowercaseLine.includes("compiling") || lowercaseLine.includes("running"));

              let displayText = line;
              if (lowercaseLine.includes("in queue")) {
                displayText = "Queued";
              } else if (lowercaseLine.includes("compiling")) {
                displayText = hasOutput ? "Compiled" : "Compiling";
              } else if (lowercaseLine.includes("running")) {
                displayText = hasOutput ? "Complete" : "Running";
              }
              
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
                    animation={isAnimated ? `${pulseAnimation} 2s infinite ease-in-out` : undefined}
                  >
                    {theme.label}
                  </Box>
                  <HStack spacing={2}>
                    {isLoading && (
                      <Spinner size="xs" color={theme.color} speed="0.8s" />
                    )}
                    <Text whiteSpace="pre-wrap" color="#D4D4D4">
                      {displayText}
                    </Text>
                  </HStack>
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
