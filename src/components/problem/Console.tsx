import {
  Box,
  Text,
  VStack,
  HStack,
  Spinner,
  Badge,
  IconButton,
  Tooltip,
  useClipboard,
} from "@chakra-ui/react";
import { keyframes } from "@emotion/react";
import {
  SampleStatus,
  type SampleStatusType,
  type SampleOutput,
} from "~/types/submission";
import { FiCheck, FiCopy } from "react-icons/fi";

const pulseAnimation = keyframes`
  0% { opacity: 0.6; }
  50% { opacity: 1; }
  100% { opacity: 0.6; }
`;

export const ConsoleStatusBadge = ({
  status,
  isRunning,
}: {
  status: SampleStatusType;
  isRunning: boolean;
}) => {
  const getStatusProps = () => {
    switch (status) {
      case SampleStatus.IN_QUEUE:
        return {
          color: "#569cd6",
          bg: "#1B1B35",
          text: "Queued",
          loading: true,
        };
      case SampleStatus.COMPILING:
        return {
          color: "#569cd6",
          bg: "#1B1B35",
          text: "Compiling",
          loading: true,
        };
      case SampleStatus.RUNNING:
        return {
          color: "#569cd6",
          bg: "#1B1B35",
          text: "Running",
          loading: true,
        };
      case SampleStatus.PASSED:
        return {
          color: "#4EC9B0",
          bg: "#1B352B",
          text: "Passed",
          loading: false,
        };
      case SampleStatus.FAILED:
        return {
          color: "#FF5D5D",
          bg: "#351B1B",
          text: "Failed",
          loading: false,
        };
      case SampleStatus.ERROR:
      case SampleStatus.COMPILE_ERROR:
      case SampleStatus.RUNTIME_ERROR:
      case SampleStatus.TIME_LIMIT_EXCEEDED:
        return {
          color: "#FF5D5D",
          bg: "#351B1B",
          text: "Error",
          loading: false,
        };
      default:
        return {
          color: "#858585",
          bg: "#1A1A1A",
          text: "Ready",
          loading: false,
        };
    }
  };

  const props = getStatusProps();

  return (
    <HStack spacing={2}>
      <Badge
        px={3}
        py={1}
        color={props.color}
        bg={props.bg}
        border="1px solid"
        borderColor={`${props.color}40`}
        borderRadius="md"
        fontWeight="600"
        fontSize="xs"
        animation={
          props.loading && isRunning
            ? `${pulseAnimation} 1.5s infinite ease-in-out`
            : undefined
        }
      >
        {props.text}
      </Badge>
      {props.loading && isRunning && (
        <Spinner size="xs" color={props.color} speed="0.8s" thickness="2px" />
      )}
    </HStack>
  );
};

const OutputBox = ({
  title,
  content,
  type = "default",
}: {
  title?: string;
  content?: string;
  type?: "input" | "output" | "expected" | "error" | "default";
}) => {
  const { hasCopied, onCopy } = useClipboard(content ?? "");
  if (!content) return null;

  const getTypeProps = () => {
    switch (type) {
      case "input":
        return {
          color: "#569CD6",
          borderColor: "#2A2A2A",
          label: "Input",
        };
      case "expected":
        return {
          color: "#4FC1FF",
          borderColor: "#2A2A2A",
          label: "Expected Output",
        };
      case "output":
        return {
          color: "#4EC9B0",
          borderColor: "#2A2A2A",
          label: "Your Output",
        };
      case "error":
        return {
          color: "#FF5D5D",
          borderColor: "#2A2A2A",
          label: "Error Output",
        };
      default:
        return {
          color: "#858585",
          borderColor: "#2A2A2A",
          label: title ?? "Output",
        };
    }
  };

  const props = getTypeProps();

  return (
    <Box
      border="1px solid"
      borderColor={props.borderColor}
      borderRadius="md"
      overflow="hidden"
      bg="#111111"
      role="group"
    >
      <HStack px={3} py={1} bg="#111111" justify="space-between">
        <Text fontSize="xs" fontWeight="500" color={props.color}>
          {props.label}
        </Text>
        <Tooltip label={hasCopied ? "Copied" : "Copy"} placement="top" hasArrow>
          <IconButton
            aria-label={hasCopied ? "Copied" : "Copy"}
            icon={hasCopied ? <FiCheck size={14} /> : <FiCopy size={14} />}
            size="xs"
            variant="ghost"
            color="gray.400"
            _hover={{ color: "gray.200", bg: "whiteAlpha.50" }}
            onClick={(e) => {
              e.stopPropagation();
              onCopy();
            }}
            opacity={0}
            _groupHover={{ opacity: 1 }}
            transition="opacity 0.15s ease"
          />
        </Tooltip>
      </HStack>
      <Box px={3} py={2} bg="#111111">
        <Text
          fontFamily="JetBrains Mono, monospace"
          fontSize="sm"
          color="#D4D4D4"
          whiteSpace="pre-wrap"
          wordBreak="break-word"
          lineHeight="1.5"
        >
          {content}
        </Text>
      </Box>
    </Box>
  );
};

type ConsoleProps = {
  output: SampleOutput | null;
  status: SampleStatusType;
  isRunning: boolean;
  embedded?: boolean;
  hideHeader?: boolean;
};

const ResizableConsole = ({
  output,
  status,
  isRunning,
  embedded = false,
  hideHeader = false,
}: ConsoleProps) => {
  return (
    <Box
      w="100%"
      h="100%"
      bg={embedded ? "transparent" : "#111111"}
      borderRadius={embedded ? "0" : "xl"}
      overflow="hidden"
    >
      <Box
        px={4}
        py={3}
        h="100%"
        overflowY="auto"
        css={{
          "&::-webkit-scrollbar": {
            width: "8px",
          },
          "&::-webkit-scrollbar-track": {
            background: "transparent",
          },
          "&::-webkit-scrollbar-thumb": {
            background: "#383838",
            borderRadius: "4px",
          },
          "&::-webkit-scrollbar-thumb:hover": {
            background: "#454545",
          },
        }}
      >
        {!output && status === SampleStatus.IDLE ? (
          <VStack align="center" justify="center" h="100%" spacing={3}>
            <Text color="#858585" fontSize="sm" textAlign="center">
              Hit &#34;Run&#34; to test your code with sample inputs
            </Text>
          </VStack>
        ) : (
          <VStack align="stretch" spacing={4}>
            {!hideHeader && (
              <HStack justify="space-between" align="center">
                <Text color="#D4D4D4" fontSize="md" fontWeight="600">
                  Sample Run Results
                </Text>
                <ConsoleStatusBadge status={status} isRunning={isRunning} />
              </HStack>
            )}

            <VStack align="stretch" spacing={3}>
              <OutputBox content={output?.input} type="input" />
              <OutputBox content={output?.output} type="output" />
              <OutputBox content={output?.expected_output} type="expected" />

              {output?.stdout && (
                <OutputBox
                  title="Standard Output"
                  content={output.stdout}
                  type="default"
                />
              )}

              {output?.stderr && (
                <OutputBox content={output.stderr} type="error" />
              )}

              {output?.message && (
                <OutputBox
                  title="Error Message"
                  content={output.message}
                  type="error"
                />
              )}

              {output?.details && (
                <OutputBox
                  title="Error Details"
                  content={output.details}
                  type="error"
                />
              )}
            </VStack>
          </VStack>
        )}
      </Box>
    </Box>
  );
};

export default ResizableConsole;
