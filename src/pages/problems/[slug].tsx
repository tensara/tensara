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
  Select,
  Link,
  Modal,
  ModalOverlay,
  ModalContent,
  ModalHeader,
  ModalFooter,
  ModalBody,
  ModalCloseButton,
} from "@chakra-ui/react";
import { useState, useEffect, useCallback, useRef } from "react";
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
import { createServerSideHelpers } from "@trpc/react-query/server";
import { appRouter } from "~/server/api/root";
import { createInnerTRPCContext } from "~/server/api/trpc";
import superjson from "superjson";
import type { GetServerSideProps } from "next";
import { useSession } from "next-auth/react";

type BenchmarkTestResult = {
  test_id: number;
  runtime_ms: number;
  gflops: number;
  name: string;
};

type SubmissionStatus = {
  status:
    | "compiling"
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
  gpuType: string;
};

const LOCAL_STORAGE_PREFIX = "problem_solution_";

const getSolutionKey = (slug: string) => `${LOCAL_STORAGE_PREFIX}${slug}`;

const saveSolutionToStorage = (slug: string, code: string) => {
  if (typeof window === "undefined") return;
  localStorage.setItem(getSolutionKey(slug), code);
};

const loadSolutionFromStorage = (slug: string): string | null => {
  if (typeof window === "undefined") return null;
  return localStorage.getItem(getSolutionKey(slug));
};

export const getServerSideProps: GetServerSideProps = async (context) => {
  const helpers = createServerSideHelpers({
    router: appRouter,
    ctx: createInnerTRPCContext({ session: null }),
    transformer: superjson,
  });

  const slug = context.params?.slug as string;

  try {
    // Prefetch the problem data
    await helpers.problems.getById.prefetch({ slug });
    // Prefetch the submissions data (this will only work if user is authenticated)
    await helpers.problems.getSubmissions.prefetch({
      problemSlug: slug,
      limit: 50,
    });

    return {
      props: {
        trpcState: helpers.dehydrate(),
        slug,
      },
    };
  } catch (e) {
    console.error(e);
    return {
      notFound: true,
    };
  }
};

export default function ProblemPage({ slug }: { slug: string }) {
  const { data: session, status } = useSession();
  const toast = useToast();
  const [code, setCode] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submissionStatus, setSubmissionStatus] =
    useState<SubmissionStatus | null>(null);
  const [submissionId, setSubmissionId] = useState<string | null>(null);
  const [hasSetInitialCode, setHasSetInitialCode] = useState(false);
  const [isTestCaseTableOpen, setIsTestCaseTableOpen] = useState(false);
  const [isBenchmarking, setIsBenchmarking] = useState(false);
  const [splitRatio, setSplitRatio] = useState(35);
  const [isDragging, setIsDragging] = useState(false);
  const [selectedGpuType, setSelectedGpuType] = useState("T4");
  const [isCodeDirty, setIsCodeDirty] = useState(false);
  const [isResetModalOpen, setIsResetModalOpen] = useState(false);
  const isProcessing = useRef<boolean>(false);

  const submissionsQuery = api.problems.getSubmissions.useQuery(
    { problemSlug: slug },
    { enabled: !!slug }
  ) as {
    data?: { submissions: Submission[]; nextCursor: string | null };
    isLoading: boolean;
    refetch: () => void;
  };

  const createSubmissionMutation = api.problems.createSubmission.useMutation({
    onSuccess: (data) => {
      setSubmissionId(data.id);
      // No need to call submitMutation here - we'll directly connect to the streaming endpoint
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

  const handleSubmit = () => {
    if (!session?.user) {
      toast({
        title: "Not signed in",
        description: "Please sign in to submit solutions",
        status: "error",
        duration: 5000,
        isClosable: true,
      });
      return;
    }

    setIsTestCaseTableOpen(false);
    setIsBenchmarking(false);
    setIsSubmitting(true);
    setSubmissionStatus({
      status: "CHECKING",
      runtime: null,
      gflops: null,
      passedTests: null,
      totalTests: null,
      message: "status: CHECKING",
    });

    createSubmissionMutation.mutate(
      {
        problemSlug: slug,
        code,
        language: "cuda",
        gpuType: selectedGpuType,
      },
      {
        onSuccess: (data) => {
          // Wrap the async logic in a void function
          void (async () => {
            try {
              const response = await fetch("/api/submissions/direct-submit", {
                method: "POST",
                headers: {
                  "Content-Type": "application/json",
                },
                body: JSON.stringify({ submissionId: data.id }),
                cache: "no-store",
                credentials: "same-origin",
                keepalive: true,
                signal: AbortSignal.timeout(300000),
              });

              if (!response.ok) {
                throw new Error(
                  `Direct submit API returned ${response.status}`
                );
              }

              // Set up reader for streaming response
              const reader = response.body?.getReader();
              if (!reader) {
                throw new Error("No response body from direct-submit");
              }

              // Process the stream
              const decoder = new TextDecoder();
              let buffer = "";

              const MAX_RETRIES = 3;
              let retryCount = 0;

              const attemptConnection = async (): Promise<void> => {
                try {
                  while (true) {
                    const { done, value } = await reader.read();
                    if (done) {
                      console.log("[sse] Stream complete");
                      break;
                    }

                    // Reset retry count on successful read
                    retryCount = 0;

                    const chunk = decoder.decode(value, { stream: true });
                    buffer += chunk;

                    // Process complete events in buffer
                    const events = buffer.split("\n\n");
                    buffer = events.pop() ?? ""; // Use nullish coalescing

                    for (const event of events) {
                      if (!event) continue;

                      const eventLines = event.split("\n");
                      let eventType = "message";
                      let eventData = "";

                      for (const line of eventLines) {
                        if (line.startsWith("event: ")) {
                          eventType = line.slice(7);
                        } else if (line.startsWith("data: ")) {
                          eventData = line.slice(6);
                        }
                      }

                      if (!eventData) continue;

                      try {
                        // Parse and type the data
                        const data = JSON.parse(eventData) as {
                          status?: string;
                          passedTests?: number;
                          totalTests?: number;
                          result?: {
                            status?: string;
                            test_id?: number;
                            runtime_ms?: number;
                            gflops?: number;
                            name?: string;
                          };
                          runtime?: number;
                          gflops?: number;
                          error?: string;
                          details?: string;
                          benchmarkResults?: BenchmarkTestResult[];
                        };

                        console.log(`[sse] ${eventType} event:`, data);

                        // Handle different event types
                        if (eventType === "status") {
                          setSubmissionStatus((prev) => {
                            if (!prev) return prev;
                            return {
                              ...prev,
                              status:
                                (data.status as SubmissionStatus["status"]) ??
                                prev.status,
                              passedTests: data.passedTests ?? prev.passedTests,
                              totalTests: data.totalTests ?? prev.totalTests,
                              message:
                                eventType +
                                ": " +
                                (data.status ?? prev.message),
                            };
                          });
                        } else if (eventType === "checker") {
                          if (data.status === "test_result" && data.result) {
                            setSubmissionStatus((prev) => {
                              if (!prev) {
                                return {
                                  status: "CHECKING",
                                  runtime: null,
                                  gflops: null,
                                  passedTests:
                                    data.result?.status === "PASSED" ? 1 : 0,
                                  totalTests: data.totalTests ?? 1,
                                  message: eventType + ": " + data.status,
                                };
                              }
                              return {
                                ...prev,
                                passedTests:
                                  (prev.passedTests ?? 0) +
                                  (data.result?.status === "PASSED" ? 1 : 0),
                                totalTests: data.totalTests ?? prev.totalTests,
                                message: eventType + ": " + data.status,
                              };
                            });
                          } else if (data.status && data.status !== "error") {
                            // Handle other checker statuses like "compiling"
                            setSubmissionStatus((prev) => {
                              if (!prev) return prev;
                              return {
                                ...prev,
                                message: eventType + ": " + data.status,
                              };
                            });
                          } else if (data.status === "error") {
                            setSubmissionStatus({
                              status: "ERROR",
                              runtime: null,
                              gflops: null,
                              passedTests: null,
                              totalTests: null,
                              message:
                                eventType + ": " + (data.status ?? "ERROR"),
                              errorMessage: data.error ?? undefined,
                              errorDetails: data.details ?? undefined,
                            });

                            setIsSubmitting(false);
                            submissionsQuery.refetch();

                            toast({
                              title: "Submission Error",
                              description:
                                data.error ??
                                "An error occurred during submission",
                              status: "error",
                              duration: 5000,
                              isClosable: true,
                            });
                            return;
                          }
                        } else if (eventType === "benchmark") {
                          if (data.status === "test_result" && data.result) {
                            setIsBenchmarking(true);
                            setIsTestCaseTableOpen(true);
                            setSubmissionStatus((prev) => {
                              if (!prev) return prev;
                              const benchmarkResult: BenchmarkTestResult = {
                                test_id: data.result?.test_id ?? 0,
                                runtime_ms: data.result?.runtime_ms ?? 0,
                                gflops: data.result?.gflops ?? 0,
                                name:
                                  data.result?.name ??
                                  `Test ${data.result?.test_id ?? 0}`,
                              };
                              return {
                                ...prev,
                                message: eventType + ": " + data.status,
                                benchmarkResults: [
                                  ...(prev.benchmarkResults ?? []),
                                  benchmarkResult,
                                ],
                              };
                            });
                          } else if (data.status && data.status !== "error") {
                            // Handle other benchmark statuses
                            setSubmissionStatus((prev) => {
                              if (!prev) return prev;
                              return {
                                ...prev,
                                message: eventType + ": " + data.status,
                              };
                            });
                          }
                        } else if (eventType === "complete") {
                          setSubmissionStatus({
                            status:
                              (data.status as SubmissionStatus["status"]) ??
                              "ERROR",
                            runtime: data.runtime ?? null,
                            gflops: data.gflops ?? null,
                            passedTests: data.passedTests ?? null,
                            totalTests: data.totalTests ?? null,
                            message:
                              eventType + ": " + (data.status ?? "ERROR"),
                            errorMessage: data.error ?? undefined,
                            errorDetails: data.details ?? undefined,
                            benchmarkResults:
                              data.benchmarkResults ?? undefined,
                          });

                          setIsSubmitting(false);
                          submissionsQuery.refetch();
                          return;
                        } else if (eventType === "error") {
                          setSubmissionStatus({
                            status: "ERROR",
                            runtime: null,
                            gflops: null,
                            passedTests: null,
                            totalTests: null,
                            message:
                              eventType + ": " + (data.status ?? "ERROR"),
                            errorMessage: data.error ?? undefined,
                            errorDetails: data.details ?? undefined,
                          });

                          setIsSubmitting(false);
                          submissionsQuery.refetch();

                          toast({
                            title: "Submission Error",
                            description:
                              data.error ??
                              "An error occurred during submission",
                            status: "error",
                            duration: 5000,
                            isClosable: true,
                          });
                          return;
                        }
                      } catch (error) {
                        console.error(
                          "Error parsing event data:",
                          error,
                          "Raw data:",
                          eventData
                        );
                      }
                    }
                  }

                  // If we reach here, the stream ended normally
                  setIsSubmitting(false);
                } catch (error) {
                  console.error("[sse] Stream error:", error);

                  // Attempt retry if we haven't exceeded max retries
                  if (retryCount < MAX_RETRIES) {
                    retryCount++;
                    console.log(
                      `[sse] Retrying connection (${retryCount}/${MAX_RETRIES})...`
                    );

                    // Wait before retrying (exponential backoff)
                    await new Promise((resolve) =>
                      setTimeout(resolve, 1000 * Math.pow(2, retryCount))
                    );

                    return attemptConnection();
                  } else {
                    throw error; // Re-throw if max retries exceeded
                  }
                }
              };

              void attemptConnection();
            } catch (error) {
              console.error("[sse] Error:", error);
              setIsSubmitting(false);

              let errorMessage = "Failed to connect to submission service";
              if (error instanceof Error) {
                if (
                  error.message.includes("QUIC_PROTOCOL_ERROR") ||
                  error.message.includes("network error") ||
                  error.message.includes("failed to fetch")
                ) {
                  errorMessage =
                    "Network protocol error. This may be due to your connection or a server configuration issue. Please try again in a few moments.";
                } else {
                  errorMessage = error.message;
                }
              }

              toast({
                title: "Connection Error",
                description: errorMessage,
                status: "error",
                duration: 5000,
                isClosable: true,
              });
            }
          })();
        },
      }
    );
  };

  const { data: problem, isLoading } = api.problems.getById.useQuery(
    { slug: slug },
    { enabled: !!slug }
  );

  useEffect(() => {
    if (!hasSetInitialCode && slug) {
      const savedSolution = loadSolutionFromStorage(slug);
      if (savedSolution) {
        setCode(savedSolution);
        setHasSetInitialCode(true);
      } else if (problem?.starterCode) {
        setCode(problem.starterCode);
        setHasSetInitialCode(true);
      }
    }
  }, [slug, hasSetInitialCode, problem]);

  useEffect(() => {
    if (code && slug) {
      saveSolutionToStorage(slug, code);
    }
  }, [code, slug]);

  useEffect(() => {
    if (problem?.starterCode) {
      setIsCodeDirty(code !== problem.starterCode);
    }
  }, [code, problem?.starterCode]);

  const handleReset = () => {
    if (problem?.starterCode) {
      setCode(problem.starterCode);
      // Clear localStorage
      if (slug) {
        localStorage.removeItem(getSolutionKey(slug));
      }
    }
  };

  // Move these handlers to useCallback
  const handleMouseDown = useCallback(() => {
    setIsDragging(true);
    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";
  }, []);

  const handleMouseUp = useCallback(() => {
    setIsDragging(false);
    document.body.style.cursor = "default";
    document.body.style.userSelect = "auto";
  }, []);

  useEffect(() => {
    if (isDragging) {
      // Move handleMouseMove inside useEffect
      const handleMouseMove = (e: MouseEvent) => {
        if (!isDragging) return;

        const container = document.getElementById("split-container");
        if (!container) return;

        const containerRect = container.getBoundingClientRect();
        const newRatio =
          ((e.clientX - containerRect.left) / containerRect.width) * 100;

        // Adjust max ratio to 55% to ensure submit button stays visible
        const clampedRatio = Math.min(Math.max(newRatio, 35), 55);
        setSplitRatio(clampedRatio);
      };

      window.addEventListener("mousemove", handleMouseMove);
      window.addEventListener("mouseup", handleMouseUp);

      return () => {
        window.removeEventListener("mousemove", handleMouseMove);
        window.removeEventListener("mouseup", handleMouseUp);
      };
    }
  }, [isDragging, handleMouseUp]);

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
      <Box
        id="split-container"
        display="flex"
        flexDirection={{ base: "column", md: "row" }}
        h="100%"
        maxH="calc(100vh - 120px)"
        position="relative"
      >
        {/* Problem Description or Submission Results */}
        <Box
          w={{ base: "100%", md: `${splitRatio}%` }}
          h={{ base: "auto", md: "100%" }}
          overflowY="auto"
          pr={{ base: 0, md: 4 }}
          mb={{ base: 4, md: 0 }}
          maxH={{ base: "auto", md: "100%" }}
        >
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
                              <Badge
                                ml={2}
                                px={2}
                                rounded="full"
                                variant="outline"
                                borderColor="whiteAlpha.200"
                                color="gray.300"
                                fontSize="xs"
                              >
                                {submission.gpuType}
                              </Badge>
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
                      submissionStatus.status === "ACCEPTED" ||
                      (submissionStatus.message &&
                        submissionStatus.message.startsWith(
                          "complete: ACCEPTED"
                        ))
                        ? "green.900"
                        : submissionStatus.message &&
                          (submissionStatus.message.startsWith(
                            "status: CHECKING"
                          ) ||
                            submissionStatus.message.startsWith(
                              "checker: compiling"
                            ) ||
                            submissionStatus.message.startsWith(
                              "checker: running"
                            ) ||
                            submissionStatus.message.startsWith(
                              "benchmark: compiling"
                            ) ||
                            submissionStatus.message.startsWith(
                              "benchmark: running"
                            ) ||
                            submissionStatus.message.startsWith(
                              "benchmark: test_result"
                            ) ||
                            submissionStatus.status === "CHECKING" ||
                            submissionStatus.status === "BENCHMARKING")
                        ? "blue.900"
                        : "red.900"
                    }
                    p={4}
                    borderRadius="xl"
                  >
                    <HStack spacing={3}>
                      {submissionStatus.message &&
                      (submissionStatus.message.startsWith(
                        "status: CHECKING"
                      ) ||
                        submissionStatus.message.startsWith(
                          "checker: compiling"
                        ) ||
                        submissionStatus.message.startsWith(
                          "checker: running"
                        ) ||
                        submissionStatus.message.startsWith(
                          "benchmark: compiling"
                        ) ||
                        submissionStatus.message.startsWith(
                          "benchmark: running"
                        ) ||
                        submissionStatus.message.startsWith(
                          "benchmark: test_result"
                        ) ||
                        submissionStatus.status === "CHECKING" ||
                        submissionStatus.status === "BENCHMARKING") ? (
                        <Spinner size="sm" color="blue.200" />
                      ) : (
                        <Icon
                          as={
                            submissionStatus.status === "ACCEPTED" ||
                            (submissionStatus.message &&
                              submissionStatus.message.startsWith(
                                "complete: ACCEPTED"
                              ))
                              ? CheckIcon
                              : WarningIcon
                          }
                          boxSize={6}
                        />
                      )}
                      <VStack align="start" spacing={0}>
                        <Text fontSize="lg" fontWeight="semibold">
                          {submissionStatus.message &&
                          submissionStatus.message.startsWith(
                            "status: CHECKING"
                          )
                            ? "Checking..."
                            : submissionStatus.message &&
                              submissionStatus.message.startsWith(
                                "status: BENCHMARKING"
                              )
                            ? "Running benchmarks..."
                            : submissionStatus.message &&
                              submissionStatus.message.startsWith(
                                "checker: compiling"
                              )
                            ? "Compiling..."
                            : submissionStatus.message &&
                              submissionStatus.message.startsWith(
                                "checker: running"
                              )
                            ? "Running tests..."
                            : submissionStatus.message &&
                              submissionStatus.message.startsWith(
                                "checker: complete"
                              )
                            ? "Tests complete"
                            : submissionStatus.message &&
                              submissionStatus.message.startsWith(
                                "benchmark: compiling"
                              )
                            ? "Running benchmarks..."
                            : submissionStatus.message &&
                              submissionStatus.message.startsWith(
                                "benchmark: running"
                              )
                            ? "Running benchmarks..."
                            : submissionStatus.message &&
                              submissionStatus.message.startsWith(
                                "benchmark: test_result"
                              )
                            ? "Running benchmarks..."
                            : submissionStatus.message &&
                              submissionStatus.message.startsWith(
                                "benchmark: success"
                              )
                            ? "Benchmark results"
                            : submissionStatus.message &&
                              submissionStatus.message.startsWith(
                                "complete: ACCEPTED"
                              )
                            ? "ACCEPTED"
                            : submissionStatus.message &&
                              submissionStatus.message.startsWith(
                                "complete: WRONG_ANSWER"
                              )
                            ? "Wrong answer"
                            : submissionStatus.message &&
                              submissionStatus.message.startsWith("error:")
                            ? "Error"
                            : submissionStatus.status === "CHECKING"
                            ? "Checking..."
                            : submissionStatus.status === "BENCHMARKING"
                            ? "Running benchmarks..."
                            : "Status: " + submissionStatus.status}
                        </Text>
                      </VStack>
                    </HStack>
                  </Box>

                  {submissionStatus.passedTests !== null &&
                    submissionStatus.status !== "WRONG_ANSWER" &&
                    !submissionStatus.message?.startsWith(
                      "complete: WRONG_ANSWER"
                    ) && (
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
                          {isBenchmarking ? (
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
                              <HStack spacing={2} width="100%">
                                <HStack spacing={2}>
                                  <Text fontWeight="semibold">
                                    Benchmark Results
                                  </Text>
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
                                      setIsTestCaseTableOpen(
                                        !isTestCaseTableOpen
                                      );
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
                                w={`${
                                  (submissionStatus.passedTests /
                                    (submissionStatus.totalTests ?? 10)) *
                                  100
                                }%`}
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
                                  {submissionStatus.benchmarkResults?.map(
                                    (result) => (
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
                                      </Tr>
                                    )
                                  )}
                                  {submissionStatus.status != "BENCHMARKING" &&
                                    submissionStatus.totalTests !== null &&
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
                                            ?.length ?? 0) +
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

                  {submissionStatus.errorMessage &&
                    submissionStatus.errorDetails && (
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
                          {submissionStatus.errorDetails ??
                            submissionStatus.errorMessage}
                        </Code>
                      </Box>
                    )}
                </>
              )}
            </VStack>
          ) : (
            <Box>
              <Heading as="h1" size="lg" mb={2}>
                {problem.title}
              </Heading>
              <HStack spacing={2} align="center" mb={6}>
                <Badge
                  colorScheme={getDifficultyColor(problem.difficulty)}
                  px={2}
                  py={1}
                  borderRadius="full"
                >
                  {problem.difficulty}
                </Badge>
                <Button
                  variant="outline"
                  height="28px"
                  px={3}
                  py={1}
                  fontSize="xs"
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
                  leftIcon={<TimeIcon boxSize={3} />}
                  borderRadius="full"
                  borderColor="whiteAlpha.200"
                  color="gray.300"
                  cursor="pointer"
                  _hover={{
                    bg: "whiteAlpha.50",
                    color: "white",
                  }}
                >
                  My Submissions
                </Button>
                <Button
                  variant="outline"
                  height="28px"
                  px={3}
                  py={1}
                  fontSize="xs"
                  onClick={() => {
                    window.location.href = `/leaderboard/${problem.slug}`;
                  }}
                  leftIcon={<Icon as={FiTrendingUp} boxSize={3} />}
                  borderRadius="full"
                  borderColor="whiteAlpha.200"
                  color="gray.300"
                  cursor="pointer"
                  _hover={{
                    bg: "whiteAlpha.50",
                    color: "white",
                  }}
                >
                  Leaderboard
                </Button>
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

        {/* Resizer Handle - Only visible on desktop */}
        <Box
          display={{ base: "none", md: "block" }}
          position="absolute"
          left={`${splitRatio}%`}
          transform="translateX(-50%)"
          width="6px"
          height="100%"
          cursor="col-resize"
          zIndex={2}
          onClick={(e) => e.stopPropagation()}
          onMouseDown={handleMouseDown}
          _hover={{
            "& > div": {
              bg: "whiteAlpha.400",
            },
          }}
        >
          <Box
            position="absolute"
            left="50%"
            top="50%"
            transform="translate(-50%, -50%)"
            width="6px"
            height="80px"
            bg="whiteAlpha.200"
            borderRadius="full"
            transition="all 0.2s"
          />
        </Box>

        {/* Mobile Warning - Only visible on mobile */}
        <Box
          display={{ base: "block", md: "none" }}
          w="100%"
          p={6}
          bg="whiteAlpha.50"
          borderRadius="xl"
          mb={4}
        >
          <VStack spacing={4} align="center">
            <Icon as={WarningIcon} boxSize={10} color="yellow.400" />
            <Heading size="md" textAlign="center">
              Desktop Required for Code Submission
            </Heading>
            <Text textAlign="center" color="whiteAlpha.800">
              For the best coding experience, please switch to a desktop device
              to write and submit your solution.
            </Text>
          </VStack>
        </Box>

        {/* Code Editor and Submit Button - Only visible on desktop */}
        <Box
          display={{ base: "none", md: "block" }}
          w={{ base: "100%", md: `${100 - splitRatio}%` }}
          h={{ base: "auto", md: "100%" }}
          minH={{ base: "50vh", md: "auto" }}
          pl={{ base: 0, md: 4 }}
        >
          <VStack w="100%" h="100%" spacing={4}>
            <HStack
              w="100%"
              justify="space-between"
              spacing={4}
              flexDirection={{ base: "column", sm: "row" }}
              alignItems={{ base: "flex-start", sm: "center" }}
            >
              <HStack
                spacing={2}
                flexWrap={{ base: "wrap", lg: "nowrap" }}
                gap={2}
              >
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
                    value={selectedGpuType}
                    onChange={(e) => setSelectedGpuType(e.target.value)}
                    borderRadius="full"
                    sx={{
                      "& > option": {
                        bg: "gray.800",
                      },
                    }}
                  >
                    <option value="T4">NVIDIA T4</option>
                    <option value="H100">NVIDIA H100</option>
                    <option value="A10G">NVIDIA A10G</option>
                    <option value="A100-80GB">NVIDIA A100-80GB</option>
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
                    <option value="python" disabled>
                      Python (Triton)
                    </option>
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
                    <option value="float16" disabled>
                      float16
                    </option>
                    <option value="int32" disabled>
                      int32
                    </option>
                    <option value="int16" disabled>
                      int16
                    </option>
                  </Select>
                </Box>
              </HStack>

              <HStack spacing={2} mt={{ base: 2, sm: 0 }}>
                {splitRatio < 45 && (
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
                )}
                {isCodeDirty && (
                  <Button
                    size="md"
                    variant="ghost"
                    onClick={() => setIsResetModalOpen(true)}
                    borderRadius="full"
                    height="40px"
                    fontSize="sm"
                    fontWeight="semibold"
                    color="gray.300"
                    _hover={{
                      bg: "whiteAlpha.50",
                      color: "white",
                    }}
                  >
                    Reset Code
                  </Button>
                )}
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
              h={{ base: "400px", md: "100%" }}
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
        </Box>
      </Box>

      <Modal
        isOpen={isResetModalOpen}
        onClose={() => setIsResetModalOpen(false)}
        isCentered
      >
        <ModalOverlay bg="blackAlpha.800" backdropFilter="blur(5px)" />
        <ModalContent
          bg="gray.800"
          borderColor="whiteAlpha.100"
          borderWidth={1}
          mx={4}
          maxW="md"
        >
          <ModalHeader color="white">Reset Code</ModalHeader>
          <ModalCloseButton color="gray.400" />
          <ModalBody>
            <Text color="gray.300">
              Are you sure you want to reset to the starter code? Your changes
              will be lost.
            </Text>
          </ModalBody>

          <ModalFooter gap={3}>
            <Button
              variant="ghost"
              onClick={() => setIsResetModalOpen(false)}
              color="gray.300"
              _hover={{ bg: "whiteAlpha.100" }}
            >
              Cancel
            </Button>
            <Button
              bg="rgba(34, 197, 94, 0.1)"
              color="rgb(34, 197, 94)"
              _hover={{
                bg: "rgba(34, 197, 94, 0.2)",
              }}
              onClick={() => {
                handleReset();
                setIsResetModalOpen(false);
              }}
            >
              Reset Code
            </Button>
          </ModalFooter>
        </ModalContent>
      </Modal>
    </Layout>
  );
}
