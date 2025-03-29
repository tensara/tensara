import {
  Box,
  Heading,
  Text,
  HStack,
  Spinner,
  Code,
  VStack,
  Button,
  Table,
  Thead,
  Tbody,
  Tr,
  Th,
  Td,
  Badge,
  SimpleGrid,
  Collapse,
  IconButton,
  Icon,
} from "@chakra-ui/react";

import {
  isSubmissionError,
  SubmissionError,
  type SubmissionErrorType,
  SubmissionStatus,
  type SubmissionStatusType,
  type TestResultResponse,
  type BenchmarkResultResponse,
  type AcceptedResponse,
  type BenchmarkedResponse,
  type WrongAnswerResponse,
  type ErrorResponse,
  type CheckedResponse,
} from "~/types/submission";

import {
  CheckIcon,
  TimeIcon,
  WarningIcon,
  ChevronDownIcon,
  ChevronUpIcon,
} from "@chakra-ui/icons";

import { FiArrowLeft } from "react-icons/fi";

import { getStatusIcon } from "~/constants/problem";

// Define interface for sample differences in debug info
interface DebugSampleDifference {
  expected: number;
  actual: number;
  diff: number;
}

// Define more specific types for the response data - these match the types in src/types/submission.ts
type ResponseTypeMap = {
  [SubmissionStatus.ACCEPTED]: AcceptedResponse;
  [SubmissionStatus.BENCHMARKED]: BenchmarkedResponse;
  [SubmissionStatus.CHECKED]: CheckedResponse;
  [SubmissionStatus.WRONG_ANSWER]: WrongAnswerResponse;
  [SubmissionError.ERROR]: ErrorResponse;
  [SubmissionError.COMPILE_ERROR]: ErrorResponse;
  [SubmissionError.RUNTIME_ERROR]: ErrorResponse;
  [SubmissionError.TIME_LIMIT_EXCEEDED]: ErrorResponse;
  [SubmissionError.RATE_LIMIT_EXCEEDED]: ErrorResponse;
};

interface DebugInfo {
  message?: string;
  max_difference?: number;
  mean_difference?: number;
  sample_differences?: Record<string, DebugSampleDifference>;
}

interface SubmissionResultsProps {
  metaStatus: SubmissionStatusType | SubmissionErrorType | null;
  metaResponse: ResponseTypeMap[keyof ResponseTypeMap] | null;
  testResults: TestResultResponse[];
  benchmarkResults: BenchmarkResultResponse[];
  isTestCaseTableOpen: boolean;
  setIsTestCaseTableOpen: (isOpen: boolean) => void;
  isBenchmarking: boolean;
  totalTests: number | null;
  getTypedResponse: <T extends SubmissionStatusType | SubmissionErrorType>(
    status: T
  ) => T extends keyof ResponseTypeMap ? ResponseTypeMap[T] | null : null;
  onBackToProblem: () => void;
  onViewSubmissions: () => void;
}

const getStatusMessage = (
  status: SubmissionStatusType | SubmissionErrorType
): string => {
  switch (status) {
    case SubmissionStatus.IN_QUEUE:
      return "In queue";
    case SubmissionStatus.COMPILING:
      return "Compiling...";
    case SubmissionStatus.CHECKING:
      return "Running tests...";
    case SubmissionStatus.CHECKED:
      return "Tests complete!";
    case SubmissionStatus.BENCHMARKING:
      return "Running benchmarks...";
    case SubmissionStatus.BENCHMARKED:
      return "Benchmark complete!";
    case SubmissionStatus.ACCEPTED:
      return "ACCEPTED";
    case SubmissionStatus.WRONG_ANSWER:
      return "Wrong Answer";
    case SubmissionError.ERROR:
      return "Error";
    case SubmissionError.COMPILE_ERROR:
      return "Compile Error";
    case SubmissionError.RUNTIME_ERROR:
      return "Runtime Error";
    case SubmissionError.TIME_LIMIT_EXCEEDED:
      return "Time Limit Exceeded";
    case SubmissionError.RATE_LIMIT_EXCEEDED:
      return "Rate Limit Exceeded";
    default:
      return status;
  }
};

const SubmissionResults = ({
  metaStatus,
  metaResponse,
  testResults,
  benchmarkResults,
  isTestCaseTableOpen,
  setIsTestCaseTableOpen,
  isBenchmarking,
  totalTests,
  getTypedResponse,
  onBackToProblem,
  onViewSubmissions,
}: SubmissionResultsProps) => {
  if (!metaStatus) return null;

  return (
    <VStack spacing={4} align="stretch" p={6}>
      <HStack justify="space-between">
        <Heading size="md">Submission Results</Heading>
        <HStack>
          <Button
            variant="ghost"
            size="sm"
            onClick={onViewSubmissions}
            leftIcon={<TimeIcon />}
            borderRadius="full"
            color="gray.300"
            _hover={{
              bg: "whiteAlpha.50",
              color: "white",
            }}
          >
            My Submissions
          </Button>
          <Button
            size="sm"
            variant="ghost"
            onClick={onBackToProblem}
            leftIcon={<Icon as={FiArrowLeft} />}
            borderRadius="full"
            color="gray.300"
            _hover={{
              bg: "whiteAlpha.50",
              color: "white",
            }}
          >
            Back to Problem
          </Button>
        </HStack>
      </HStack>
      <Box
        bg={
          metaStatus === "ACCEPTED"
            ? "green.900"
            : metaStatus === "IN_QUEUE" ||
                metaStatus === "COMPILING" ||
                metaStatus === "CHECKING" ||
                metaStatus === "CHECKED" ||
                metaStatus === "BENCHMARKING" ||
                metaStatus === "BENCHMARKED"
              ? "blue.900"
              : "red.900"
        }
        p={4}
        borderRadius="xl"
      >
        <HStack spacing={3}>
          {metaStatus === "IN_QUEUE" ||
          metaStatus === "COMPILING" ||
          metaStatus === "CHECKING" ||
          metaStatus === "CHECKED" ||
          metaStatus === "BENCHMARKING" ||
          metaStatus === "BENCHMARKED" ? (
            <Spinner size="sm" color="blue.200" />
          ) : (
            <Icon as={getStatusIcon(metaStatus)} boxSize={6} />
          )}
          <VStack align="start" spacing={0}>
            <Text fontSize="lg" fontWeight="semibold">
              {getStatusMessage(metaStatus)}
            </Text>
          </VStack>
        </HStack>
      </Box>
      {/* Test Case Results Table */}
      {testResults.length > 0 &&
        metaStatus !== SubmissionStatus.WRONG_ANSWER &&
        metaResponse && (
          <Box bg="whiteAlpha.50" borderRadius="xl" overflow="hidden">
            <VStack
              spacing={0}
              align="stretch"
              divider={
                <Box borderBottomWidth={1} borderColor="whiteAlpha.100" />
              }
            >
              {isBenchmarking ? (
                <HStack
                  justify="space-between"
                  px={6}
                  py={4}
                  onClick={() => setIsTestCaseTableOpen(!isTestCaseTableOpen)}
                  cursor="pointer"
                  _hover={{ bg: "whiteAlpha.50" }}
                >
                  <HStack spacing={2} width="100%">
                    <HStack spacing={2}>
                      <Text fontWeight="semibold">Benchmark Results</Text>
                      <IconButton
                        aria-label="Toggle test cases"
                        icon={
                          isTestCaseTableOpen ? (
                            <ChevronUpIcon />
                          ) : (
                            <ChevronDownIcon />
                          )
                        }
                        size="sm"
                        variant="ghost"
                        onClick={(e) => {
                          e.stopPropagation();
                          setIsTestCaseTableOpen(!isTestCaseTableOpen);
                        }}
                        color="gray.300"
                        _hover={{
                          bg: "whiteAlpha.50",
                          color: "white",
                        }}
                      />
                    </HStack>
                  </HStack>
                </HStack>
              ) : (
                <Box
                  w="100%"
                  h="6"
                  bg="whiteAlpha.200"
                  borderRadius="md"
                  overflow="hidden"
                >
                  <Box
                    h="100%"
                    w={`${(testResults.length / (totalTests ?? 10)) * 100}%`}
                    bg="green.500"
                    borderRadius="md"
                    borderRightRadius="xl"
                    transition="width 0.5s ease-in-out"
                  />
                </Box>
              )}

              {/* Test Case Results Table */}
              <Collapse in={isTestCaseTableOpen} animateOpacity>
                <Box>
                  <Table variant="unstyled" size="sm">
                    <Thead bg="whiteAlpha.100">
                      <Tr>
                        <Th color="whiteAlpha.700" py={3}>
                          Test Case
                        </Th>
                        <Th color="whiteAlpha.700" py={3} isNumeric>
                          Runtime
                        </Th>
                        <Th color="whiteAlpha.700" py={3} isNumeric>
                          Performance
                        </Th>
                      </Tr>
                    </Thead>
                    <Tbody>
                      {benchmarkResults.map((result) => (
                        <Tr
                          key={result.result.test_id}
                          _hover={{ bg: "whiteAlpha.100" }}
                        >
                          <Td py={3}>
                            <HStack spacing={2}>
                              <Icon
                                as={CheckIcon}
                                color="green.300"
                                boxSize={4}
                              />
                              <Text>{result.result.name}</Text>
                            </HStack>
                          </Td>
                          <Td py={3} isNumeric>
                            <Text>
                              {result.result.runtime_ms.toFixed(2)} ms
                            </Text>
                          </Td>
                          <Td py={3} isNumeric>
                            <Text>
                              {result.result.gflops.toFixed(2)} GFLOPS
                            </Text>
                          </Td>
                        </Tr>
                      ))}
                      {totalTests !== null &&
                        benchmarkResults &&
                        metaStatus !== SubmissionStatus.BENCHMARKING &&
                        totalTests > benchmarkResults.length &&
                        Array.from(
                          {
                            length: totalTests - benchmarkResults.length,
                          },
                          (_, i) => {
                            const testId =
                              (benchmarkResults?.length ?? 0) + i + 1;
                            return (
                              <Tr
                                key={`failed-${testId}`}
                                _hover={{ bg: "whiteAlpha.100" }}
                              >
                                <Td py={3}>
                                  <HStack spacing={2}>
                                    <Icon
                                      as={WarningIcon}
                                      color="red.300"
                                      boxSize={4}
                                    />
                                    <Text>Test Case {testId}</Text>
                                  </HStack>
                                </Td>
                                <Td py={3} isNumeric>
                                  -
                                </Td>
                                <Td py={3} isNumeric>
                                  -
                                </Td>
                                <Td py={3}>
                                  <Badge colorScheme="red" fontSize="xs">
                                    Failed
                                  </Badge>
                                </Td>
                              </Tr>
                            );
                          }
                        )}
                    </Tbody>
                  </Table>
                </Box>
              </Collapse>
            </VStack>
          </Box>
        )}

      {/* Performance and Runtime Stats (when submission is accepted) */}
      {Boolean(metaStatus) && metaStatus === SubmissionStatus.ACCEPTED && (
        <Box bg="whiteAlpha.50" p={4} borderRadius="xl">
          <Heading size="sm" mb={3}>
            Performance and Runtime Stats
          </Heading>
          <SimpleGrid columns={2} spacing={4}>
            <Box>
              <Text color="whiteAlpha.700" mb={1}>
                Average Performance
              </Text>
              <Heading size="md">
                {(() => {
                  const acceptedResponse = getTypedResponse(
                    SubmissionStatus.ACCEPTED
                  );
                  return acceptedResponse?.avg_gflops?.toFixed(2) ?? "N/A";
                })()}{" "}
                GFLOPS
              </Heading>
            </Box>
            <Box>
              <Text color="whiteAlpha.700" mb={1}>
                Average Runtime
              </Text>
              <Heading size="md">
                {(() => {
                  const acceptedResponse = getTypedResponse(
                    SubmissionStatus.ACCEPTED
                  );
                  return acceptedResponse?.avg_runtime_ms?.toFixed(2) ?? "N/A";
                })()}{" "}
                ms
              </Heading>
            </Box>
          </SimpleGrid>
        </Box>
      )}

      {/* Wrong Answer Debug Info */}
      {Boolean(metaStatus) && metaStatus === SubmissionStatus.WRONG_ANSWER && (
        <Box bg="whiteAlpha.50" p={4} borderRadius="xl">
          <Heading size="sm" mb={3}>
            Wrong Answer Debug Info
          </Heading>
          <Text mb={4}>
            {(() => {
              const wrongAnswerResponse = getTypedResponse(
                SubmissionStatus.WRONG_ANSWER
              );
              return (
                wrongAnswerResponse?.debug_info?.message ??
                "No debug message available"
              );
            })()}
          </Text>
          {(() => {
            const wrongAnswerResponse = getTypedResponse(
              SubmissionStatus.WRONG_ANSWER
            );
            if (!wrongAnswerResponse?.debug_info) return null;

            const debugInfo = wrongAnswerResponse.debug_info;

            return (
              <>
                {debugInfo.max_difference && (
                  <Box>
                    <Text color="red.200" fontSize="sm">
                      Maximum Difference
                    </Text>
                    <Text fontWeight="semibold">
                      {debugInfo.max_difference.toFixed(7)}
                    </Text>
                  </Box>
                )}
                {debugInfo.mean_difference && (
                  <Box>
                    <Text color="red.200" fontSize="sm">
                      Mean Difference
                    </Text>
                    <Text fontWeight="semibold">
                      {debugInfo.mean_difference.toFixed(7)}
                    </Text>
                  </Box>
                )}
                {debugInfo.sample_differences &&
                  Object.keys(debugInfo.sample_differences).length > 0 && (
                    <Box mt={4}>
                      <Text color="red.200" fontSize="sm" mb={2}>
                        Sample Differences
                      </Text>
                      <Table size="sm" variant="simple">
                        <Thead>
                          <Tr>
                            <Th>Sample</Th>
                            <Th isNumeric>Expected</Th>
                            <Th isNumeric>Actual</Th>
                            <Th isNumeric>Diff</Th>
                          </Tr>
                        </Thead>
                        <Tbody>
                          {Object.entries(debugInfo.sample_differences).map(
                            ([key, value]) => {
                              if (
                                typeof value === "object" &&
                                value !== null &&
                                "expected" in value &&
                                "actual" in value &&
                                "diff" in value
                              ) {
                                return (
                                  <Tr key={key}>
                                    <Td>{key}</Td>
                                    <Td isNumeric>
                                      {value.expected.toFixed(7)}
                                    </Td>
                                    <Td isNumeric>{value.actual.toFixed(7)}</Td>
                                    <Td isNumeric>{value.diff.toFixed(7)}</Td>
                                  </Tr>
                                );
                              }
                              return (
                                <Tr key={key}>
                                  <Td>{key}</Td>
                                  <Td isNumeric>-</Td>
                                  <Td isNumeric>-</Td>
                                  <Td isNumeric>-</Td>
                                </Tr>
                              );
                            }
                          )}
                        </Tbody>
                      </Table>
                    </Box>
                  )}
              </>
            );
          })()}
        </Box>
      )}

      {/* Error Details */}
      {Boolean(metaStatus) && isSubmissionError(metaStatus) && (
        <Box bg="red.900" p={4} borderRadius="xl">
          <Heading size="sm" mb={3} color="red.200">
            Error Details
          </Heading>
          {(() => {
            const errorResponse = getTypedResponse(metaStatus);
            const message = errorResponse?.message ?? "Unknown error";
            const details = errorResponse?.details ?? "";

            return (
              <>
                <Text fontWeight="semibold" color="red.200" mb={2}>
                  {message}
                </Text>
                {details && (
                  <Code
                    display="block"
                    whiteSpace="pre-wrap"
                    p={4}
                    bg="red.800"
                    color="red.100"
                    borderRadius="lg"
                    fontSize="sm"
                    fontFamily="mono"
                  >
                    {details}
                  </Code>
                )}
              </>
            );
          })()}
        </Box>
      )}
    </VStack>
  );
};

export default SubmissionResults;
