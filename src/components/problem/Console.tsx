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
import { FiCheck, FiChevronDown, FiChevronRight, FiCopy } from "react-icons/fi";
import { useState } from "react";

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
  renderContent,
  copyText,
  type = "default",
}: {
  title?: string;
  content?: string;
  renderContent?: React.ReactNode;
  copyText?: string;
  type?: "input" | "output" | "expected" | "error" | "default";
}) => {
  const copyValue = copyText ?? content ?? "";
  const { hasCopied, onCopy } = useClipboard(copyValue);
  const [isCollapsed, setIsCollapsed] = useState(false);
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
      <HStack
        px={3}
        py={1}
        bg="#111111"
        justify="space-between"
        cursor="pointer"
        onClick={() => setIsCollapsed((prev) => !prev)}
      >
        <HStack spacing={1.5} minW={0}>
          <IconButton
            aria-label={isCollapsed ? "Expand" : "Collapse"}
            icon={
              isCollapsed ? (
                <FiChevronRight size={14} />
              ) : (
                <FiChevronDown size={14} />
              )
            }
            size="xs"
            variant="ghost"
            color="gray.500"
            _hover={{ color: "gray.200", bg: "whiteAlpha.50" }}
            onClick={(e) => {
              e.stopPropagation();
              setIsCollapsed((prev) => !prev);
            }}
          />
          <Text
            fontSize="xs"
            fontWeight="500"
            color={props.color}
            noOfLines={1}
          >
            {props.label}
          </Text>
        </HStack>
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
      {!isCollapsed && (
        <Box px={3} py={2} bg="#111111">
          {renderContent ?? (
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
          )}
        </Box>
      )}
    </Box>
  );
};

const normalizeForCompare = (value: string) =>
  value.replace(/\r\n/g, "\n").replace(/\n+$/g, "");

type DiffLine =
  | { type: "context"; value: string }
  | { type: "add"; value: string }
  | { type: "del"; value: string };

type InlineSegment = { value: string; changed: boolean };

const tokenizeForInlineDiff = (value: string) =>
  value.match(/(\s+|[^\s]+)/g) ?? [];

function createInlineDiffSegments(
  before: string,
  after: string
): {
  before: InlineSegment[];
  after: InlineSegment[];
} {
  const a = tokenizeForInlineDiff(before);
  const b = tokenizeForInlineDiff(after);

  const n = a.length;
  const m = b.length;

  if (n === 0 && m === 0) return { before: [], after: [] };

  // Keep this bounded; worst-case token count is small for expected output.
  const maxTokens = 600;
  if (n > maxTokens || m > maxTokens) {
    return {
      before: [{ value: before, changed: true }],
      after: [{ value: after, changed: true }],
    };
  }

  const cols = m + 1;
  const dp = new Uint16Array((n + 1) * (m + 1));

  for (let i = 1; i <= n; i++) {
    for (let j = 1; j <= m; j++) {
      const idx = i * cols + j;
      if (a[i - 1] === b[j - 1]) {
        dp[idx] = (dp[(i - 1) * cols + (j - 1)] ?? 0) + 1;
      } else {
        const up = dp[(i - 1) * cols + j] ?? 0;
        const left = dp[i * cols + (j - 1)] ?? 0;
        dp[idx] = up >= left ? up : left;
      }
    }
  }

  const beforeSegments: InlineSegment[] = [];
  const afterSegments: InlineSegment[] = [];

  let i = n;
  let j = m;
  while (i > 0 || j > 0) {
    if (i > 0 && j > 0 && a[i - 1] === b[j - 1]) {
      beforeSegments.push({ value: a[i - 1] ?? "", changed: false });
      afterSegments.push({ value: b[j - 1] ?? "", changed: false });
      i--;
      j--;
      continue;
    }

    const up = i > 0 ? (dp[(i - 1) * cols + j] ?? 0) : 0;
    const left = j > 0 ? (dp[i * cols + (j - 1)] ?? 0) : 0;

    if (j > 0 && (i === 0 || left >= up)) {
      afterSegments.push({ value: b[j - 1] ?? "", changed: true });
      j--;
    } else if (i > 0) {
      beforeSegments.push({ value: a[i - 1] ?? "", changed: true });
      i--;
    }
  }

  beforeSegments.reverse();
  afterSegments.reverse();

  return { before: beforeSegments, after: afterSegments };
}

function createLineDiff(expected: string, actual: string): DiffLine[] | null {
  const expectedLines = normalizeForCompare(expected).split("\n");
  const actualLines = normalizeForCompare(actual).split("\n");

  const maxLines = 200;
  if (expectedLines.length > maxLines || actualLines.length > maxLines) {
    return null;
  }

  const n = expectedLines.length;
  const m = actualLines.length;
  const cols = m + 1;
  const dp = new Uint16Array((n + 1) * (m + 1));

  for (let i = 1; i <= n; i++) {
    for (let j = 1; j <= m; j++) {
      const idx = i * cols + j;
      if (expectedLines[i - 1] === actualLines[j - 1]) {
        dp[idx] = (dp[(i - 1) * cols + (j - 1)] ?? 0) + 1;
      } else {
        const up = dp[(i - 1) * cols + j] ?? 0;
        const left = dp[i * cols + (j - 1)] ?? 0;
        dp[idx] = up >= left ? up : left;
      }
    }
  }

  const result: DiffLine[] = [];
  let i = n;
  let j = m;
  while (i > 0 || j > 0) {
    if (i > 0 && j > 0 && expectedLines[i - 1] === actualLines[j - 1]) {
      result.push({ type: "context", value: expectedLines[i - 1] ?? "" });
      i--;
      j--;
      continue;
    }

    const up = i > 0 ? (dp[(i - 1) * cols + j] ?? 0) : 0;
    const left = j > 0 ? (dp[i * cols + (j - 1)] ?? 0) : 0;
    if (j > 0 && (i === 0 || left >= up)) {
      result.push({ type: "add", value: actualLines[j - 1] ?? "" });
      j--;
    } else if (i > 0) {
      result.push({ type: "del", value: expectedLines[i - 1] ?? "" });
      i--;
    }
  }

  result.reverse();
  return result;
}

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
  const diff =
    output?.expected_output && output?.output
      ? normalizeForCompare(output.expected_output) ===
        normalizeForCompare(output.output)
        ? null
        : createLineDiff(output.expected_output, output.output)
      : null;

  const diffCopyText = diff
    ? diff
        .map((line) => {
          const prefix =
            line.type === "add" ? "+ " : line.type === "del" ? "- " : "  ";
          return `${prefix}${line.value}`;
        })
        .join("\n")
    : null;

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
              {diff && (
                <OutputBox
                  title="Diff"
                  content=" "
                  copyText={diffCopyText ?? undefined}
                  renderContent={
                    <Box
                      fontFamily="JetBrains Mono, monospace"
                      fontSize="sm"
                      whiteSpace="pre"
                      lineHeight="1.5"
                    >
                      {(() => {
                        const nodes: React.ReactNode[] = [];
                        for (let idx = 0; idx < diff.length; idx++) {
                          const line = diff[idx];
                          const next = diff[idx + 1];

                          // When a delete is immediately followed by an add, render inline highlights.
                          if (line?.type === "del" && next?.type === "add") {
                            const inline = createInlineDiffSegments(
                              line.value,
                              next.value
                            );
                            nodes.push(
                              <Text
                                key={`del-${idx}`}
                                as="div"
                                color="red.300"
                                whiteSpace="pre"
                              >
                                {"- "}
                                {inline.before.map((seg, i2) => (
                                  <Box
                                    key={i2}
                                    as="span"
                                    bg={
                                      seg.changed
                                        ? "rgba(255, 93, 93, 0.22)"
                                        : "transparent"
                                    }
                                    color={seg.changed ? "red.200" : "red.300"}
                                    px={seg.changed ? 0.5 : 0}
                                    borderRadius={seg.changed ? "sm" : "none"}
                                  >
                                    {seg.value}
                                  </Box>
                                ))}
                              </Text>
                            );
                            nodes.push(
                              <Text
                                key={`add-${idx}`}
                                as="div"
                                color="green.300"
                                whiteSpace="pre"
                              >
                                {"+ "}
                                {inline.after.map((seg, i2) => (
                                  <Box
                                    key={i2}
                                    as="span"
                                    bg={
                                      seg.changed
                                        ? "rgba(78, 201, 176, 0.22)"
                                        : "transparent"
                                    }
                                    color={
                                      seg.changed ? "green.200" : "green.300"
                                    }
                                    px={seg.changed ? 0.5 : 0}
                                    borderRadius={seg.changed ? "sm" : "none"}
                                  >
                                    {seg.value}
                                  </Box>
                                ))}
                              </Text>
                            );
                            idx++; // consume the paired add line
                            continue;
                          }

                          const prefix =
                            line?.type === "add"
                              ? "+ "
                              : line?.type === "del"
                                ? "- "
                                : "  ";
                          const color =
                            line?.type === "add"
                              ? "green.300"
                              : line?.type === "del"
                                ? "red.300"
                                : "gray.300";

                          nodes.push(
                            <Text
                              key={idx}
                              as="div"
                              color={color}
                              whiteSpace="pre"
                            >
                              {prefix}
                              {line?.value ?? ""}
                            </Text>
                          );
                        }
                        return nodes;
                      })()}
                    </Box>
                  }
                />
              )}

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
