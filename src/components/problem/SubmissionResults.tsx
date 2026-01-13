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
  Tooltip,
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
  FaCheck,
  FaExclamationCircle,
  FaClock,
  FaChevronUp,
  FaChevronDown,
  FaThermometerHalf,
  FaMicrochip,
  FaTachometerAlt,
} from "react-icons/fa";

import { FiArrowLeft, FiExternalLink } from "react-icons/fi";
import NextLink from "next/link";

import { getStatusIcon } from "~/constants/problem";
import { useSplitPanel } from "./SplitPanel";
import { GPUMetricInfoPopover } from "~/components/misc/GPUMetricInfoPopover";

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

// Helper to aggregate GPU metrics from benchmark results
interface AggregatedGPUMetrics {
  tempMin: number;
  tempMax: number;
  tempAvg: number;
  clockMin: number;
  clockMax: number;
  clockAvg: number;
  pstateMin: number;
  pstateMax: number;
}

const aggregateGPUMetrics = (
  benchmarkResults: BenchmarkResultResponse[]
): AggregatedGPUMetrics | null => {
  let tempMin = Infinity;
  let tempMax = -Infinity;
  let tempSum = 0;
  let clockMin = Infinity;
  let clockMax = -Infinity;
  let clockSum = 0;
  let pstateMin = Infinity;
  let pstateMax = -Infinity;
  let runCount = 0;

  benchmarkResults.forEach((br) => {
    br.result.runs?.forEach((run) => {
      const metrics = run.gpu_metrics;
      if (metrics && metrics.sample_count > 0) {
        runCount++;
        tempMin = Math.min(tempMin, metrics.temp_c_min);
        tempMax = Math.max(tempMax, metrics.temp_c_max);
        tempSum += metrics.temp_c_mean;
        clockMin = Math.min(clockMin, metrics.sm_clock_mhz_min);
        clockMax = Math.max(clockMax, metrics.sm_clock_mhz_max);
        clockSum += metrics.sm_clock_mhz_mean;
        pstateMin = Math.min(pstateMin, metrics.pstate_min);
        pstateMax = Math.max(pstateMax, metrics.pstate_max);
      }
    });
  });

  if (runCount === 0) return null;

  return {
    tempMin,
    tempMax,
    tempAvg: tempSum / runCount,
    clockMin,
    clockMax,
    clockAvg: clockSum / runCount,
    pstateMin,
    pstateMax,
  };
};

const getTestCaseGPUMetrics = (
  result: BenchmarkResultResponse
): AggregatedGPUMetrics | null => {
  const runs = result.result.runs;
  if (!runs || runs.length === 0) return null;

  let tempMin = Infinity;
  let tempMax = -Infinity;
  let tempSum = 0;
  let clockMin = Infinity;
  let clockMax = -Infinity;
  let clockSum = 0;
  let runCount = 0;

  runs.forEach((run) => {
    const metrics = run.gpu_metrics;
    if (metrics && metrics.sample_count > 0) {
      runCount++;
      tempMin = Math.min(tempMin, metrics.temp_c_min);
      tempMax = Math.max(tempMax, metrics.temp_c_max);
      tempSum += metrics.temp_c_mean;
      clockMin = Math.min(clockMin, metrics.sm_clock_mhz_min);
      clockMax = Math.max(clockMax, metrics.sm_clock_mhz_max);
      clockSum += metrics.sm_clock_mhz_mean;
    }
  });

  if (runCount === 0) return null;

  return {
    tempMin,
    tempMax,
    tempAvg: tempSum / runCount,
    clockMin,
    clockMax,
    clockAvg: clockSum / runCount,
    pstateMin: 0,
    pstateMax: 0,
  };
};

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
  submissionId?: string | null;
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
  submissionId,
}: SubmissionResultsProps) => {
  const { splitRatio } = useSplitPanel();
  const useCompactLabels = splitRatio < 40;
  if (!metaStatus) return null;

  return (
    <VStack spacing={4} align="stretch" p={6}>
      <HStack justify="space-between">
        <Heading size="md">Results</Heading>
        <HStack>
          <Button
            variant="ghost"
            size="sm"
            onClick={onViewSubmissions}
            leftIcon={<FaClock />}
            borderRadius="lg"
            color="gray.300"
            _hover={{
              bg: "whiteAlpha.50",
              color: "white",
            }}
          >
            {useCompactLabels ? "Submissions" : "My Submissions"}
          </Button>
          <Button
            size="sm"
            variant="ghost"
            onClick={onBackToProblem}
            leftIcon={<Icon as={FiArrowLeft} />}
            borderRadius="lg"
            color="gray.300"
            _hover={{
              bg: "whiteAlpha.50",
              color: "white",
            }}
          >
            {useCompactLabels ? "Back" : "Back to Problem"}
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
              ? "transparent"
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
                            <FaChevronUp />
                          ) : (
                            <FaChevronDown />
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
                  {(() => {
                    return (
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
                              GFLOPS
                            </Th>
                          </Tr>
                        </Thead>
                        <Tbody>
                          {benchmarkResults.map((result) => {
                            // Support both old format (runtime_ms, gflops) and new format (avg_runtime_ms, avg_gflops)
                            const runtime =
                              result.result.avg_runtime_ms ??
                              result.result.runtime_ms;
                            const gflops =
                              result.result.avg_gflops ?? result.result.gflops;
                            const gpuMetrics = getTestCaseGPUMetrics(result);

                            const row = (
                              <Tr
                                key={result.result.test_id}
                                _hover={{ bg: "whiteAlpha.100" }}
                              >
                                <Td py={3}>
                                  <HStack spacing={2}>
                                    <Icon
                                      as={FaCheck}
                                      color="green.300"
                                      boxSize={4}
                                    />
                                    <Text>{result.result.name}</Text>
                                  </HStack>
                                </Td>
                                <Td py={3} isNumeric>
                                  <Text>
                                    {runtime !== undefined
                                      ? `${runtime.toFixed(2)} ms`
                                      : "-"}
                                  </Text>
                                </Td>
                                <Td py={3} isNumeric>
                                  <Text>
                                    {gflops !== undefined
                                      ? gflops.toFixed(2)
                                      : "-"}
                                  </Text>
                                </Td>
                              </Tr>
                            );

                            if (gpuMetrics) {
                              return (
                                <Tooltip
                                  key={result.result.test_id}
                                  label={
                                    <HStack spacing={4} py={1} px={2}>
                                      <VStack align="start" spacing={0}>
                                        <Text fontSize="xs" color="whiteAlpha.600">
                                          Temp
                                        </Text>
                                        <Text fontSize="sm" fontWeight="medium">
                                          {gpuMetrics.tempAvg.toFixed(0)}°C
                                        </Text>
                                      </VStack>
                                      <VStack align="start" spacing={0}>
                                        <Text fontSize="xs" color="whiteAlpha.600">
                                          SM Clock
                                        </Text>
                                        <Text fontSize="sm" fontWeight="medium">
                                          {gpuMetrics.clockAvg.toFixed(0)} MHz
                                        </Text>
                                      </VStack>
                                    </HStack>
                                  }
                                  bg="brand.secondary"
                                  borderColor="whiteAlpha.200"
                                  borderWidth={1}
                                  borderRadius="lg"
                                  p={2}
                                  color="white"
                                  placement="right"
                                  openDelay={300}
                                >
                                  {row}
                                </Tooltip>
                              );
                            }

                            return row;
                          })}

                          {totalTests !== null &&
                            benchmarkResults &&
                            metaStatus !== SubmissionStatus.BENCHMARKING &&
                            totalTests > benchmarkResults.length &&
                            Array.from(
                              { length: totalTests - benchmarkResults.length },
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
                                          as={FaExclamationCircle}
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
                                  </Tr>
                                );
                              }
                            )}
                        </Tbody>
                      </Table>
                    );
                  })()}
                </Box>
              </Collapse>
            </VStack>
          </Box>
        )}

      {/* Performance and Runtime Stats (when submission is accepted) */}
      {Boolean(metaStatus) && metaStatus === SubmissionStatus.ACCEPTED && (
        <Box>
          <Text
            fontSize="xs"
            color="whiteAlpha.600"
            fontWeight="medium"
            mb={2}
            letterSpacing="wide"
            textTransform="uppercase"
          >
            Performance
          </Text>
          <Box
            bg="whiteAlpha.50"
            borderRadius="xl"
            overflow="hidden"
          >
            <SimpleGrid columns={2} spacing={0}>
            {/* Runtime */}
            <Box p={4}>
              <Text
                fontSize="xs"
                color="whiteAlpha.500"
                textTransform="uppercase"
                letterSpacing="wide"
                mb={1}
              >
                Runtime
              </Text>
              <Text fontSize="xl" fontWeight="bold" color="white">
                {getTypedResponse(
                  SubmissionStatus.ACCEPTED
                )?.avg_runtime_ms?.toFixed(2) ?? "N/A"}{" "}
                ms
              </Text>
            </Box>
            {getTypedResponse(SubmissionStatus.ACCEPTED)?.avg_gflops !==
              undefined && (
              <Box p={4}>
                <Text
                  fontSize="xs"
                  color="whiteAlpha.500"
                  textTransform="uppercase"
                  letterSpacing="wide"
                  mb={1}
                >
                  Throughput
                </Text>
                <Text fontSize="xl" fontWeight="bold" color="white">
                  {getTypedResponse(
                    SubmissionStatus.ACCEPTED
                  )!.avg_gflops!.toFixed(2)}{" "}
                  GFLOPS
                </Text>
              </Box>
            )}
          </SimpleGrid>
        </Box>
        </Box>
      )}

      {/* GPU Metrics (when benchmarks complete and we have data) */}
      {Boolean(metaStatus) &&
        (metaStatus === SubmissionStatus.ACCEPTED ||
          metaStatus === SubmissionStatus.BENCHMARKED) &&
        benchmarkResults.length > 0 &&
        (() => {
          console.log(benchmarkResults);
          const gpuMetrics = aggregateGPUMetrics(benchmarkResults);
          if (!gpuMetrics) return null;

          return (
            <Box>
              <Text
                fontSize="xs"
                color="whiteAlpha.600"
                fontWeight="medium"
                mb={2}
                letterSpacing="wide"
                textTransform="uppercase"
              >
                GPU Metrics
              </Text>
              <Box
                bg="whiteAlpha.50"
                borderRadius="xl"
                overflow="hidden"
              >
                <SimpleGrid columns={2} spacing={0}>
                {/* Temperature */}
                <Box p={4}>
                  <HStack spacing={1} mb={1}>
                    <Text
                      fontSize="xs"
                      color="whiteAlpha.500"
                      textTransform="uppercase"
                      letterSpacing="wide"
                    >
                      Temperature
                    </Text>
                    <GPUMetricInfoPopover metric="temperature" />
                  </HStack>
                  <Text fontSize="xl" fontWeight="bold" color="white">
                    {gpuMetrics.tempAvg.toFixed(0)}°C
                  </Text>
                </Box>

                {/* SM Clock */}
                <Box p={4}>
                  <HStack spacing={2} mb={1}>
                    <Text
                      fontSize="xs"
                      color="whiteAlpha.500"
                      textTransform="uppercase"
                      letterSpacing="wide"
                    >
                      SM Clock
                    </Text>
                    <GPUMetricInfoPopover metric="smClock" />
                  </HStack>
                  <Text fontSize="xl" fontWeight="bold" color="white">
                    {gpuMetrics.clockAvg.toFixed(0)} MHz
                  </Text>
                </Box>
              </SimpleGrid>
            </Box>
            </Box>
          );
        })()}

      {/* Wrong Answer Debug Info */}
      {Boolean(metaStatus) && metaStatus === SubmissionStatus.WRONG_ANSWER && (
        <Box bg="red.900" p={4} borderRadius="xl">
          {(() => {
            const wrongAnswerResponse = getTypedResponse(
              SubmissionStatus.WRONG_ANSWER
            );
            if (!wrongAnswerResponse?.debug_info) return null;

            const debugInfo = wrongAnswerResponse.debug_info;

            return (
              <VStack spacing={4} align="stretch">
                {debugInfo.message && (
                  <Text color="red.100">{debugInfo.message}</Text>
                )}
                {debugInfo.max_difference && (
                  <Box>
                    <Text color="red.200" fontSize="sm">
                      Maximum Difference:
                    </Text>
                    <Text color="red.100">{debugInfo.max_difference}</Text>
                  </Box>
                )}
                {debugInfo.mean_difference && (
                  <Box>
                    <Text color="red.200" fontSize="sm">
                      Mean Difference:
                    </Text>
                    <Text color="red.100">{debugInfo.mean_difference}</Text>
                  </Box>
                )}
                {debugInfo.sample_differences &&
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
                              {debugInfo.mean_difference && (
                                <Th color="red.200" isNumeric>
                                  Difference
                                </Th>
                              )}
                            </Tr>
                          </Thead>
                          <Tbody>
                            {Object.entries(debugInfo.sample_differences)
                              .slice(0, 50)
                              .map(([key, value]) => (
                                <Tr key={key}>
                                  <Td color="red.100">{key}</Td>
                                  <Td color="red.100" isNumeric>
                                    {value.expected.toFixed(7)}
                                  </Td>
                                  <Td color="red.100" isNumeric>
                                    {value.actual?.toFixed(7) ?? "NaN or inf"}
                                  </Td>
                                  {value.diff && (
                                    <Td color="red.100" isNumeric>
                                      {value.diff?.toFixed(7)}
                                    </Td>
                                  )}
                                </Tr>
                              ))}
                          </Tbody>
                        </Table>
                      </Box>
                    </Box>
                  )}
              </VStack>
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
            const message = errorResponse?.message ?? "";
            const details = errorResponse?.details ?? "";

            return (
              <>
                {message && (
                  <Text color="red.200" mb={2}>
                    {message}
                  </Text>
                )}
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

      {submissionId && (
        <Box pt={2}>
          <Button
            as={NextLink}
            href={`/submissions/${submissionId}`}
            rightIcon={<Icon as={FiExternalLink} boxSize={4} />}
            size="sm"
            variant="ghost"
            borderRadius="lg"
            color="gray.300"
            _hover={{
              bg: "whiteAlpha.50",
              color: "white",
            }}
            px={4}
          >
            View Submission
          </Button>
        </Box>
      )}
    </VStack>
  );
};

export default SubmissionResults;
