// components/sandbox/console.tsx

import {
  Box,
  Text,
  VStack,
  HStack,
  Spinner,
  Badge,
} from "@chakra-ui/react";
import { keyframes } from "@emotion/react";
import {
  SampleStatus,
  type SampleStatusType,
  type SampleOutput,
} from "~/types/submission";

const pulseAnimation = keyframes`
  0% { opacity: 0.6; }
  50% { opacity: 1; }
  100% { opacity: 0.6; }
`;

const StatusBadge = ({
  status,
  isRunning,
}: {
  status: SampleStatusType;
  isRunning: boolean;
}) => {
  const getStatusProps = () => {
    switch (status) {
      case SampleStatus.IN_QUEUE:
      case SampleStatus.COMPILING:
      case SampleStatus.RUNNING:
        return {
          color: "#569cd6",
          bg: "#1B1B35",
          text: status.charAt(0) + status.slice(1).toLowerCase(),
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
      case SampleStatus.ERROR:
      case SampleStatus.COMPILE_ERROR:
      case SampleStatus.RUNTIME_ERROR:
        return {
          color: "#FF5D5D",
          bg: "#351B1B",
          text: "Failed",
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
  if (!content) return null;

  const getTypeProps = () => {
    switch (type) {
      case "input":
        return { color: "#569CD6", label: "Input" };
      case "expected":
        return { color: "#4FC1FF", label: "Expected Output" };
      case "output":
        return { color: "#4EC9B0", label: "Your Output" };
      case "error":
        return { color: "#FF5D5D", label: "Error Output" };
      default:
        return { color: "#858585", label: title ?? "Output" };
    }
  };

  const props = getTypeProps();

  return (
    <Box border="1px solid #2A2A2A" borderRadius="md" overflow="hidden" bg="#111111">
      <Box px={3} py={1}>
        <Text fontSize="xs" fontWeight="500" color={props.color}>
          {props.label}
        </Text>
      </Box>
      <Box px={3} py={2}>
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

type Props = {
  output: SampleOutput | null;
  status: SampleStatusType;
  isRunning: boolean;
  files: { name: string; content: string }[];
};

export default function SandboxConsole({
  output,
  status,
  isRunning,
}: Props) {
  return (
    <Box w="100%" h="100%" bg="#111111" borderRadius="xl" overflow="hidden">
      <Box px={4} py={3} h="100%" overflowY="auto"
        css={{
          "&::-webkit-scrollbar": { width: "8px" },
          "&::-webkit-scrollbar-thumb": { background: "#383838", borderRadius: "4px" },
          "&::-webkit-scrollbar-thumb:hover": { background: "#454545" },
        }}
      >
        {!output && status === SampleStatus.IDLE ? (
          <VStack align="center" justify="center" h="100%" spacing={3}>
            <Text color="#858585" fontSize="lg">No output yet</Text>
            <Text color="#858585" fontSize="sm" textAlign="center">
              Hit <code>Run</code> to test your code
            </Text>
          </VStack>
        ) : (
          <VStack align="stretch" spacing={4}>
            {/* Header with status */}
            <HStack justify="space-between" align="center">
              <Text color="#D4D4D4" fontSize="md" fontWeight="600">
                Program Output
              </Text>
              <StatusBadge status={status} isRunning={isRunning} />
            </HStack>

            {/* Output blocks */}
            <VStack align="stretch" spacing={3}>
              {output?.stdout && (
                <OutputBox title="Standard Output" content={output.stdout} />
              )}
              {output?.stderr && (
                <OutputBox type="error" content={output.stderr} />
              )}
              {output?.message && (
                <OutputBox title="Message" type="error" content={output.message} />
              )}
              {output?.details && (
                <OutputBox title="Details" type="error" content={output.details} />
              )}
            </VStack>
          </VStack>
        )}
      </Box>
    </Box>
  );
}
