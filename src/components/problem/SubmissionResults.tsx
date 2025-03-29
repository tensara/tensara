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

// Define types for the response data
type TypedResponse = any; // Replace with actual type when possible

// Define interface for sample differences in debug info
interface DebugSampleDifference {
  expected: number;
  actual: number;
  diff: number;
}

interface SubmissionResultsProps {
  metaStatus: SubmissionStatusType | SubmissionErrorType | null;
  metaResponse: any;
  testResults: TestResultResponse[];
  benchmarkResults: BenchmarkResultResponse[];
  isTestCaseTableOpen: boolean;
  setIsTestCaseTableOpen: (isOpen: boolean) => void;
  isBenchmarking: boolean;
  totalTests: number | null;
  getTypedResponse: (
    status: SubmissionStatusType | SubmissionErrorType
  ) => TypedResponse;
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

      {/* Performance and Runtime Stats */}
      {metaStatus === SubmissionStatus.ACCEPTED && (
        <SimpleGrid columns={2} spacing={4}>
          <Box bg="whiteAlpha.50" p={6} borderRadius="xl">
            <Text color="whiteAlpha.700" mb={1}>
              Performance
            </Text>
            <Text fontSize="2xl" fontWeight="bold">
              {getTypedResponse(SubmissionStatus.ACCEPTED)?.avg_gflops?.toFixed(
                2
              )}{" "}
              GFLOPS
            </Text>
          </Box>
          <Box bg="whiteAlpha.50" p={6} borderRadius="xl">
            <Text color="whiteAlpha.700" mb={1}>
              Runtime
            </Text>
            <Text fontSize="2xl" fontWeight="bold">
              {getTypedResponse(
                SubmissionStatus.ACCEPTED
              )?.avg_runtime_ms?.toFixed(2)}
              ms
            </Text>
          </Box>
        </SimpleGrid>
      )}

      {/* Wrong Answer Debug Info */}
      {metaStatus === SubmissionStatus.WRONG_ANSWER && (
        <Box bg="red.900" p={6} borderRadius="xl">
          {getTypedResponse(SubmissionStatus.WRONG_ANSWER)?.debug_info
            ?.message && (
            <Text color="red.200" fontWeight="semibold" mb={3}>
              {
                getTypedResponse(SubmissionStatus.WRONG_ANSWER)?.debug_info
                  ?.message
              }
            </Text>
          )}
          {getTypedResponse(SubmissionStatus.WRONG_ANSWER)?.debug_info &&
            (() => {
              try {
                const debugInfo = getTypedResponse(
                  SubmissionStatus.WRONG_ANSWER
                )?.debug_info;
                return (
                  <VStack spacing={4} align="stretch">
                    {debugInfo?.message && (
                      <Text color="red.100">{debugInfo.message}</Text>
                    )}
                    {debugInfo?.max_difference && (
                      <Box>
                        <Text color="red.200" fontSize="sm">
                          Maximum Difference:
                        </Text>
                        <Text color="red.100">{debugInfo.max_difference}</Text>
                      </Box>
                    )}
                    {debugInfo?.mean_difference && (
                      <Box>
                        <Text color="red.200" fontSize="sm">
                          Mean Difference:
                        </Text>
                        <Text color="red.100">{debugInfo.mean_difference}</Text>
                      </Box>
                    )}
                    {debugInfo?.sample_differences &&
                      Object.keys(debugInfo.sample_differences).length > 0 && (
                        <Box>
                          <Text color="red.200" fontSize="sm" mb={2}>
                            Sample Differences:
                          </Text>
                          <Box maxH="200px" overflowY="auto">
                            <Table size="sm" variant="unstyled">
                              <Thead position="sticky" top={0}>
                                <Tr>
                                  <Th color="red.200">Index</Th>
                                  <Th color="red.200" isNumeric>
                                    Expected
                                  </Th>
                                  <Th color="red.200" isNumeric>
                                    Actual
                                  </Th>
                                  <Th color="red.200" isNumeric>
                                    Difference
                                  </Th>
                                </Tr>
                              </Thead>
                              <Tbody>
                                {Object.entries(debugInfo.sample_differences)
                                  .slice(0, 50)
                                  .map(([key, value]) => (
                                    <Tr key={key}>
                                      <Td color="red.100">{key}</Td>
                                      <Td color="red.100" isNumeric>
                                        {(
                                          value as DebugSampleDifference
                                        ).expected.toFixed(7)}
                                      </Td>
                                      <Td color="red.100" isNumeric>
                                        {(
                                          value as DebugSampleDifference
                                        ).actual.toFixed(7)}
                                      </Td>
                                      <Td color="red.100" isNumeric>
                                        {(
                                          value as DebugSampleDifference
                                        ).diff.toFixed(7)}
                                      </Td>
                                    </Tr>
                                  ))}
                              </Tbody>
                            </Table>
                          </Box>
                        </Box>
                      )}
                  </VStack>
                );
              } catch (e) {
                console.error("Failed to parse debug info", e);
                return (
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
                    {
                      getTypedResponse(SubmissionStatus.WRONG_ANSWER)
                        ?.debug_info?.message
                    }
                  </Code>
                );
              }
            })()}
        </Box>
      )}

      {metaStatus !== SubmissionStatus.WRONG_ANSWER &&
        isSubmissionError(metaStatus) &&
        getTypedResponse(metaStatus)?.message &&
        getTypedResponse(metaStatus)?.details && (
          <Box bg="red.900" p={6} borderRadius="xl">
            <Text color="red.200" fontWeight="semibold" mb={3}>
              Error Details
            </Text>
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
              {getTypedResponse(metaStatus)?.details ??
                getTypedResponse(metaStatus)?.message}
            </Code>
          </Box>
        )}
    </VStack>
  );
};

export default SubmissionResults;
