import { useRouter } from "next/router";
import { api } from "~/utils/api";
import {
  Box,
  Heading,
  Text,
  HStack,
  Spinner,
  Code,
  VStack,
  Button,
  useToast,
  Alert,
  AlertIcon,
  AlertTitle,
  AlertDescription,
  Tabs,
  TabList,
  TabPanels,
  TabPanel,
  Tab,
  Table,
  Thead,
  Tbody,
  Tr,
  Th,
  Td,
  Badge,
  Stat,
  StatLabel,
  StatNumber,
  StatGroup,
  SimpleGrid,
  Collapse,
  IconButton,
  Select,
  Link,
} from "@chakra-ui/react";
import { useState, useEffect } from "react";
import { Layout } from "~/components/layout";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import rehypeHighlight from "rehype-highlight";
import Editor from "@monaco-editor/react";
import {
  CheckIcon,
  TimeIcon,
  WarningIcon,
  ChevronDownIcon,
  ChevronUpIcon,
} from "@chakra-ui/icons";
import { FiArrowLeft, FiTrendingUp } from "react-icons/fi";
import { Icon } from "@chakra-ui/react";
import { getDifficultyColor } from ".";

type BenchmarkTestResult = {
  test_id: number;
  runtime_ms: number;
  gflops: number;
  name: string;
};

type SubmissionStatus = {
  status:
    | "CHECKING"
    | "BENCHMARKING"
    | "ACCEPTED"
    | "WRONG_ANSWER"
    | "ERROR"
    | "SUBMISSIONS";
  runtime: number | null;
  gflops: number | null;
  passedTests: number | null;
  totalTests: number | null;
  message: string | null;
  errorMessage?: string;
  errorDetails?: string;
  benchmarkResults?: BenchmarkTestResult[];
};

type Submission = {
  id: string;
  status: string | null;
  runtime: number | null;
  gflops: number | null;
  passedTests: number | null;
  totalTests: number | null;
  createdAt: Date;
  problem: {
    title: string;
    slug: string;
  };
};

export default function ProblemPage() {
  const router = useRouter();
  const { slug } = router.query;
  const toast = useToast();
  const [code, setCode] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submissionStatus, setSubmissionStatus] =
    useState<SubmissionStatus | null>(null);
  const [submissionId, setSubmissionId] = useState<string | null>(null);
  const [hasSetInitialCode, setHasSetInitialCode] = useState(false);
  const [isTestCaseTableOpen, setIsTestCaseTableOpen] = useState(false);

  const submissionsQuery = api.problems.getSubmissions.useQuery(
    { problemSlug: slug as string, limit: 10 },
    { enabled: !!slug }
  ) as {
    data?: { submissions: Submission[]; nextCursor: string | null };
    isLoading: boolean;
    refetch: () => void;
  };

  const handleSubmit = () => {
    setIsSubmitting(true);
    setSubmissionStatus({
      status: "CHECKING",
      runtime: null,
      gflops: null,
      passedTests: null,
      totalTests: null,
      message: "Running test cases...",
    });
    
    createSubmissionMutation.mutate({
      problemSlug: slug as string,
      code,
      language: "cuda",
    });
  };

  const { data: problem, isLoading } = api.problems.getById.useQuery(
    { slug: slug as string },
    { enabled: !!slug }
  );

  useEffect(() => {
    if (
      problem?.starterCode &&
      (!hasSetInitialCode || problem.slug !== router.query.slug)
    ) {
      setCode(problem.starterCode);
      setHasSetInitialCode(true);
    }
  }, [problem, hasSetInitialCode, router.query.slug]);

  useEffect(() => {
    if (!submissionId) return;

    let retryCount = 0;
    const maxRetries = 3;
    let retryTimeout: NodeJS.Timeout;

    const setupEventSource = () => {
      console.log("[sse] Setting up EventSource for submission:", submissionId);
      const eventSource = new EventSource(
        `/api/submissions/${submissionId}/status`
      );

      // Handle connection open
      eventSource.onopen = () => {
        console.log("[sse] Connection opened");
        retryCount = 0; // Reset retry count on successful connection
      };

      // Log all raw events for debugging
      const rawEventListener = (e: MessageEvent) => {
        console.log("[sse] Raw event received:", {
          data: e.data,
          lastEventId: e.lastEventId,
          origin: e.origin,
          type: e.type
        });
      };
      eventSource.addEventListener('message', rawEventListener);

      eventSource.onmessage = (event) => {
        try {
          console.log("[sse] Message received:", event.data);
          const data = JSON.parse(event.data);
          console.log("[sse] Parsed data:", data);

          setSubmissionStatus(prevStatus => {
            return {
              status: data.status as SubmissionStatus["status"],
              runtime: data.runtime,
              gflops: data.gflops,
              passedTests: data.passedTests,
              totalTests: data.totalTests,
              message:
                data.status === "BENCHMARKING"
                  ? "All test cases passed! Running performance benchmark..."
                  : data.passedTests !== null
                  ? `${data.passedTests} test cases passed...`
                  : "Running test cases...",
              errorMessage: data.errorMessage,
              errorDetails: data.errorDetails,
              ...(data.benchmarkResults && {
                benchmarkResults: data.benchmarkResults,
              }),
            };
          });

          if (["ACCEPTED", "ERROR", "WRONG_ANSWER"].includes(data.status)) {
            console.log("[sse] Final status reached, closing connection:", data.status);
            eventSource.close();
            setSubmissionId(null);
            setIsSubmitting(false);
            submissionsQuery.refetch();
          }
        } catch (error) {
          console.error("[sse] Error parsing data:", error, "Raw data:", event.data);
        }
      };

      eventSource.onerror = (error) => {
        console.error("[sse] Connection error:", error);
        eventSource.close();

        // If we haven't exceeded max retries and the submission is still in progress
        if (retryCount < maxRetries && submissionStatus?.status !== "ACCEPTED" && 
            submissionStatus?.status !== "ERROR" && submissionStatus?.status !== "WRONG_ANSWER") {
          retryCount++;
          console.log(`[sse] Retrying connection (${retryCount}/${maxRetries})...`);
          retryTimeout = setTimeout(setupEventSource, 1000 * retryCount); // Exponential backoff
        } else {
          console.log("[sse] Max retries exceeded or submission complete, giving up");
          setSubmissionId(null);
          setIsSubmitting(false);
          toast({
            title: "Connection Error",
            description: "Lost connection to submission status updates",
            status: "error",
            duration: 5000,
            isClosable: true,
          });
        }
      };

      return eventSource;
    };

    const eventSource = setupEventSource();

    return () => {
      console.log("[sse] Cleaning up EventSource");
      eventSource.close();
      if (retryTimeout) clearTimeout(retryTimeout);
    };
  }, [submissionId, toast, submissionsQuery]);

  const createSubmissionMutation = api.problems.createSubmission.useMutation({
    onSuccess: (data) => {
      setSubmissionId(data.id);
      // Start the actual submission process
      submitMutation.mutate({ submissionId: data.id });
    },
    onError: (error) => {
      setIsSubmitting(false);
      setSubmissionStatus(null);
      toast({
        title: "Failed to create submission",
        description: error.message,
        status: "error",
        duration: 5000,
        isClosable: true,
      });
    },
  });

  const submitMutation = api.problems.submit.useMutation({
    onError: (error) => {
      setIsSubmitting(false);
      setSubmissionStatus(null);
      setSubmissionId(null); // Clear the submissionId on error
      toast({
        title: "Submission failed",
        description: error.message,
        status: "error",
        duration: 5000,
        isClosable: true,
      });
    },
  });

  if (isLoading) {
    return (
      <Layout title="Loading...">
        <Box
          display="flex"
          justifyContent="center"
          alignItems="center"
          h="100%"
        >
          <Spinner size="xl" />
        </Box>
      </Layout>
    );
  }

  if (!problem) {
    return (
      <Layout title="Not Found">
        <Box p={8}>
          <Text>Problem not found</Text>
        </Box>
      </Layout>
    );
  }

  return (
    <Layout title={problem.title}>
      <HStack align="start" spacing={8} h="100%" maxH="calc(100vh - 120px)">
        {/* Left Panel - Problem Description or Submission Results */}
        <Box w="35%" h="100%" overflowY="auto">
          {submissionStatus ? (
            <VStack spacing={4} align="stretch" p={6}>
              <HStack justify="space-between">
                <Heading size="md">
                  {submissionStatus.status === "SUBMISSIONS"
                    ? "My Submissions"
                    : "Submission Results"}
                </Heading>
                <HStack>
                  {submissionStatus.status !== "SUBMISSIONS" && (
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() =>
                        setSubmissionStatus({
                          status: "SUBMISSIONS",
                          runtime: null,
                          gflops: null,
                          passedTests: null,
                          totalTests: null,
                          message: null,
                        })
                      }
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
                  )}
                  <Button
                    size="sm"
                    variant="ghost"
                    onClick={() => {
                      setSubmissionStatus(null);
                    }}
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

              {submissionStatus.status === "SUBMISSIONS" ? (
                <VStack spacing={4} align="stretch">
                  {submissionsQuery.isLoading ? (
                    <Box display="flex" justifyContent="center" p={4}>
                      <Spinner />
                    </Box>
                  ) : submissionsQuery.data?.submissions.length === 0 ? (
                    <Box p={4} textAlign="center" color="whiteAlpha.700">
                      No submissions yet
                    </Box>
                  ) : (
                    submissionsQuery.data?.submissions.map((submission) => (
                      <Link
                        key={submission.id}
                        href={`/submissions/${submission.id}`}
                        style={{ textDecoration: "none" }}
                      >
                        <Box
                          bg="whiteAlpha.50"
                          p={4}
                          borderRadius="xl"
                          cursor="pointer"
                          _hover={{ bg: "whiteAlpha.100" }}
                        >
                          <HStack justify="space-between" mb={2}>
                            <HStack>
                              <Icon
                                as={
                                  submission.status === "ACCEPTED"
                                    ? CheckIcon
                                    : submission.status === "WRONG_ANSWER"
                                    ? WarningIcon
                                    : TimeIcon
                                }
                                color={
                                  submission.status === "ACCEPTED"
                                    ? "green.400"
                                    : submission.status === "WRONG_ANSWER"
                                    ? "red.400"
                                    : "blue.400"
                                }
                              />
                              <Text fontWeight="semibold">
                                {submission.status === "ACCEPTED"
                                  ? "Accepted"
                                  : submission.status === "WRONG_ANSWER"
                                  ? "Wrong Answer"
                                  : submission.status}
                              </Text>
                            </HStack>
                            <Text color="whiteAlpha.700" fontSize="sm">
                              {new Date(submission.createdAt).toLocaleString()}
                            </Text>
                          </HStack>
                          {submission.gflops !== null &&
                            submission.runtime !== null && (
                              <SimpleGrid columns={2} spacing={4}>
                                <Box>
                                  <Text color="whiteAlpha.600" fontSize="sm">
                                    Performance
                                  </Text>
                                  <Text fontWeight="semibold">
                                    {submission.gflops.toFixed(2)} GFLOPS
                                  </Text>
                                </Box>
                                <Box>
                                  <Text color="whiteAlpha.600" fontSize="sm">
                                    Runtime
                                  </Text>
                                  <Text fontWeight="semibold">
                                    {submission.runtime.toFixed(2)}ms
                                  </Text>
                                </Box>
                              </SimpleGrid>
                            )}
                        </Box>
                      </Link>
                    ))
                  )}
                </VStack>
              ) : (
                <>
                  <Box
                    bg={
                      submissionStatus.status === "ACCEPTED"
                        ? "green.900"
                        : submissionStatus.status === "CHECKING" ||
                          submissionStatus.status === "BENCHMARKING"
                        ? "blue.900"
                        : "red.900"
                    }
                    p={4}
                    borderRadius="xl"
                  >
                    <HStack spacing={3}>
                      {submissionStatus.status === "CHECKING" ||
                      submissionStatus.status === "BENCHMARKING" ? (
                        <Spinner size="sm" color="blue.200" />
                      ) : (
                        <Icon
                          as={
                            submissionStatus.status === "ACCEPTED"
                              ? CheckIcon
                              : WarningIcon
                          }
                          boxSize={6}
                        />
                      )}
                      <VStack align="start" spacing={0}>
                        <Text fontSize="lg" fontWeight="semibold">
                          {submissionStatus.status === "WRONG_ANSWER"
                            ? `Wrong Answer on Test ${
                                submissionStatus.passedTests !== null
                                  ? submissionStatus.passedTests + 1
                                  : 1
                              }`
                            : submissionStatus.status === "CHECKING"
                            ? "Running Tests"
                            : submissionStatus.status === "BENCHMARKING"
                            ? "Running Benchmark"
                            : submissionStatus.status}
                        </Text>
                        {(submissionStatus.status === "CHECKING" ||
                          submissionStatus.status === "BENCHMARKING") && (
                          <Text fontSize="sm" color="whiteAlpha.700">
                            {submissionStatus.status === "BENCHMARKING"
                              ? "All test cases passed! Running performance benchmark..."
                              : submissionStatus.message}
                          </Text>
                        )}
                      </VStack>
                    </HStack>
                  </Box>

                  {submissionStatus.passedTests !== null &&
                    submissionStatus.status !== "WRONG_ANSWER" && 
                    submissionStatus.status !== "BENCHMARKING" && (
                      <Box
                        bg="whiteAlpha.50"
                        borderRadius="xl"
                        overflow="hidden"
                      >
                        <VStack
                          spacing={0}
                          align="stretch"
                          divider={
                            <Box
                              borderBottomWidth={1}
                              borderColor="whiteAlpha.100"
                            />
                          }
                        >
                          <HStack
                            justify="space-between"
                            px={6}
                            py={4}
                            onClick={() =>
                              setIsTestCaseTableOpen(!isTestCaseTableOpen)
                            }
                            cursor="pointer"
                            _hover={{ bg: "whiteAlpha.50" }}
                          >
                            <HStack spacing={2}>
                              <Text fontWeight="semibold">Test Cases</Text>
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
                            <HStack spacing={1}>
                              <Text
                                color={
                                  submissionStatus.status === "ACCEPTED"
                                    ? "green.300"
                                    : "red.300"
                                }
                                fontWeight="bold"
                              >
                                {submissionStatus.passedTests}
                              </Text>
                              <Text color="whiteAlpha.700">/</Text>
                              <Text color="whiteAlpha.700">
                                {submissionStatus.totalTests}
                              </Text>
                              <Text color="whiteAlpha.700">passed</Text>
                            </HStack>
                          </HStack>

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
                                    <Th color="whiteAlpha.700" py={3}>
                                      Status
                                    </Th>
                                  </Tr>
                                </Thead>
                                <Tbody>
                                  {submissionStatus.benchmarkResults?.map(
                                    (result, index) => (
                                      <Tr
                                        key={result.test_id}
                                        _hover={{ bg: "whiteAlpha.100" }}
                                      >
                                        <Td py={3}>
                                          <HStack spacing={2}>
                                            <Icon
                                              as={CheckIcon}
                                              color="green.300"
                                              boxSize={4}
                                            />
                                            <Text>{result.name}</Text>
                                          </HStack>
                                        </Td>
                                        <Td py={3} isNumeric>
                                          <Text>
                                            {result.runtime_ms.toFixed(2)} ms
                                          </Text>
                                        </Td>
                                        <Td py={3} isNumeric>
                                          <Text>
                                            {result.gflops.toFixed(2)} GFLOPS
                                          </Text>
                                        </Td>
                                        <Td py={3}>
                                          <Badge
                                            colorScheme="green"
                                            fontSize="xs"
                                          >
                                            Passed
                                          </Badge>
                                        </Td>
                                      </Tr>
                                    )
                                  )}
                                  {submissionStatus.totalTests !== null &&
                                    submissionStatus.benchmarkResults &&
                                    submissionStatus.totalTests >
                                      submissionStatus.benchmarkResults
                                        .length &&
                                    Array.from(
                                      {
                                        length:
                                          submissionStatus.totalTests -
                                          submissionStatus.benchmarkResults
                                            .length,
                                      },
                                      (_, i) => {
                                        const testId =
                                          (submissionStatus.benchmarkResults
                                            ?.length || 0) +
                                          i +
                                          1;
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
                                              <Badge
                                                colorScheme="red"
                                                fontSize="xs"
                                              >
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

                  {submissionStatus.status === "ACCEPTED" && (
                    <SimpleGrid columns={2} spacing={4}>
                      <Box bg="whiteAlpha.50" p={6} borderRadius="xl">
                        <Text color="whiteAlpha.700" mb={1}>
                          Performance
                        </Text>
                        <Text fontSize="2xl" fontWeight="bold">
                          {submissionStatus.gflops?.toFixed(2)} GFLOPS
                        </Text>
                      </Box>
                      <Box bg="whiteAlpha.50" p={6} borderRadius="xl">
                        <Text color="whiteAlpha.700" mb={1}>
                          Runtime
                        </Text>
                        <Text fontSize="2xl" fontWeight="bold">
                          {submissionStatus.runtime?.toFixed(2)}ms
                        </Text>
                      </Box>
                    </SimpleGrid>
                  )}

                  {submissionStatus.errorMessage && (
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
                        {submissionStatus.errorDetails ||
                          submissionStatus.errorMessage}
                      </Code>
                    </Box>
                  )}
                </>
              )}
            </VStack>
          ) : (
            <Box>
              <HStack
                justify="space-between"
                align="flex-start"
                spacing={4}
                mb={6}
              >
                <VStack align="start" spacing={2}>
                  <Heading as="h1" size="xl">
                    {problem.title}
                  </Heading>
                  <Badge
                    colorScheme={getDifficultyColor(problem.difficulty)}
                    px={2}
                    py={1}
                    borderRadius="full"
                  >
                    {problem.difficulty}
                  </Badge>
                </VStack>

                <VStack spacing={2} align="end">
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() =>
                      setSubmissionStatus({
                        status: "SUBMISSIONS",
                        runtime: null,
                        gflops: null,
                        passedTests: null,
                        totalTests: null,
                        message: null,
                      })
                    }
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
                    variant="ghost"
                    size="sm"
                    onClick={() => {
                      window.location.href = `/leaderboard/${problem.slug}`;
                    }}
                    leftIcon={<Icon as={FiTrendingUp} />}
                    borderRadius="full"
                    color="gray.300"
                    _hover={{
                      bg: "whiteAlpha.50",
                      color: "white",
                    }}
                  >
                    Leaderboard
                  </Button>
                </VStack>
              </HStack>

              <Box className="markdown" color="gray.100">
                <ReactMarkdown
                  remarkPlugins={[remarkGfm, remarkMath]}
                  rehypePlugins={[rehypeKatex, rehypeHighlight]}
                  components={{
                    h1: (props) => (
                      <Heading as="h2" size="lg" mt={8} mb={4} {...props} />
                    ),
                    h2: (props) => (
                      <Heading as="h3" size="md" mt={6} mb={3} {...props} />
                    ),
                    h3: (props) => (
                      <Heading as="h4" size="sm" mt={4} mb={2} {...props} />
                    ),
                    ul: (props) => <Box as="ul" pl={8} mb={4} {...props} />,
                    ol: (props) => <Box as="ol" pl={8} mb={4} {...props} />,
                    li: (props) => <Box as="li" pl={2} mb={2} {...props} />,
                    code: (props) => (
                      <Code
                        px={2}
                        py={1}
                        bg="gray.800"
                        color="gray.100"
                        borderRadius="md"
                        {...props}
                      />
                    ),
                    pre: (props) => (
                      <Box
                        as="pre"
                        p={4}
                        bg="gray.800"
                        borderRadius="md"
                        overflowX="auto"
                        mb={4}
                        {...props}
                      />
                    ),
                  }}
                >
                  {problem.description}
                </ReactMarkdown>
              </Box>
            </Box>
          )}
        </Box>

        {/* Right Panel - Code Editor and Submit Button */}
        <VStack w="65%" h="100%" spacing={4}>
          <HStack w="100%" justify="space-between" spacing={4}>
            <HStack spacing={2}>
              <Box>
                <Text fontSize="sm" color="whiteAlpha.700" mb={1}>
                  GPU Type
                </Text>
                <Select
                  size="sm"
                  bg="whiteAlpha.50"
                  borderColor="whiteAlpha.200"
                  _hover={{ borderColor: "whiteAlpha.300" }}
                  w="160px"
                  defaultValue="t4"
                  borderRadius="full"
                  sx={{
                    "& > option": {
                      bg: "gray.800",
                    },
                  }}
                >
                  <option value="a100">NVIDIA A100</option>
                  <option value="v100">NVIDIA V100</option>
                  <option value="t4">NVIDIA T4</option>
                </Select>
              </Box>
              <Box>
                <Text fontSize="sm" color="whiteAlpha.700" mb={1}>
                  Language
                </Text>
                <Select
                  size="sm"
                  bg="whiteAlpha.50"
                  borderColor="whiteAlpha.200"
                  _hover={{ borderColor: "whiteAlpha.300" }}
                  w="160px"
                  defaultValue="cuda"
                  borderRadius="full"
                  sx={{
                    "& > option": {
                      bg: "gray.800",
                    },
                  }}
                >
                  <option value="cuda">CUDA C++</option>
                  <option value="python">Python (Triton)</option>
                </Select>
              </Box>
              <Box>
                <Text fontSize="sm" color="whiteAlpha.700" mb={1}>
                  Data Type
                </Text>
                <Select
                  size="sm"
                  bg="whiteAlpha.50"
                  borderColor="whiteAlpha.200"
                  _hover={{ borderColor: "whiteAlpha.300" }}
                  w="140px"
                  defaultValue="float32"
                  borderRadius="full"
                  sx={{
                    "& > option": {
                      bg: "gray.800",
                    },
                  }}
                >
                  <option value="float32">float32</option>
                  <option value="float16">float16</option>
                  <option value="int32">int32</option>
                  <option value="int16">int16</option>
                </Select>
              </Box>
            </HStack>

            <HStack spacing={2}>
              <Button
                size="sm"
                variant="ghost"
                onClick={() => {
                  window.location.href = "/problems";
                }}
                leftIcon={<Icon as={FiArrowLeft} />}
                borderRadius="full"
                color="gray.300"
                _hover={{
                  bg: "whiteAlpha.50",
                  color: "white",
                }}
              >
                Back to Problems
              </Button>
              <Button
                bg="rgba(34, 197, 94, 0.1)"
                color="rgb(34, 197, 94)"
                size="md"
                onClick={handleSubmit}
                isLoading={isSubmitting}
                loadingText="Submit"
                spinner={<></>}
                disabled={isSubmitting}
                borderRadius="full"
                height="40px"
                fontSize="sm"
                fontWeight="semibold"
                px={8}
                _hover={{
                  bg: "rgba(34, 197, 94, 0.2)",
                  transform: "translateY(-1px)",
                }}
                _active={{
                  bg: "rgba(34, 197, 94, 0.25)",
                }}
                transition="all 0.2s"
              >
                Submit
              </Button>
            </HStack>
          </HStack>

          <Box
            w="100%"
            h="100%"
            bg="gray.800"
            borderRadius="xl"
            overflow="hidden"
          >
            <Editor
              height="100%"
              defaultLanguage="cpp"
              theme="vs-dark"
              value={code}
              onChange={(value) => setCode(value ?? "")}
              options={{
                minimap: { enabled: false },
                fontSize: 14,
                lineNumbers: "on",
                scrollBeyondLastLine: false,
                automaticLayout: true,
                padding: { top: 16, bottom: 16 },
                fontFamily: "JetBrains Mono, monospace",
              }}
            />
          </Box>
        </VStack>
      </HStack>
    </Layout>
  );
}
