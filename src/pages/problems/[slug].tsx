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
  Flex,
  MenuList,
  MenuButton,
  MenuItem,
  Menu,
} from "@chakra-ui/react";
import { useState, useCallback } from "react";
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
import { GPU_DISPLAY_NAMES } from "~/constants/gpu";
import { BenchmarkTestResult, SubmissionStatus, Submission, DebugInfo } from "~/types/problem";
import { useCodePersistence } from "~/hooks/useCodePersistence";
import { useSubmissionStream } from "~/hooks/useSubmissionStream";
import { useSplitPanel } from "~/hooks/useSplitPanel";
import { DataType, ProgrammingLanguage } from "~/types/misc";
import { IS_DISABLED_LANGUAGE, LANGUAGE_DISPLAY_NAMES } from "~/constants/language";
import { DATA_TYPE_DISPLAY_NAMES, IS_DISABLED_DATA_TYPE } from "~/constants/datatypes";
import { Problem } from "@prisma/client";


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

const getStatusMessage = (status: SubmissionStatus): string => {
  if (!status.message) return `Status: ${status.status}`;
  
  if (status.message.startsWith("status: CHECKING")) return "Checking...";
  if (status.message.startsWith("complete: ACCEPTED")) return "ACCEPTED";
  
  const messageMap: Record<string, string> = {
    "status: BENCHMARKING": "Running benchmarks...",
    "checker: compiling": "Compiling...",
    "checker: running": "Running tests...",
    "checker: test_result": "Running tests...",
    "checker: complete": "Tests complete",
    "benchmark: compiling": "Running benchmarks...",
    "benchmark: running": "Running benchmarks...",
    "benchmark: test_result": "Running benchmarks...",
    "benchmark: success": "Benchmark results",
    "complete: ACCEPTED": "ACCEPTED",
    "complete: WRONG_ANSWER": "Wrong answer",
    "error:": "Error",
  };
  

  for (const [prefix, message] of Object.entries(messageMap)) {
    if (status.message.startsWith(prefix)) return message;
  }
  return `${status.status}`;
};

export default function ProblemPage({ slug }: { slug: string }) {
    const { data: session } = useSession();
  const toast = useToast();
  const [selectedGpuType, setSelectedGpuType] = useState("T4");
  const [isResetModalOpen, setIsResetModalOpen] = useState(false);

  // Get problem data
  const { data: problem, isLoading } = api.problems.getById.useQuery(
    { slug },
    { enabled: !!slug }
  );

  // Fetch submissions
  const submissionsQuery = api.problems.getSubmissions.useQuery(
    { problemSlug: slug },
    { enabled: !!slug }
  ) as {
    data?: { submissions: Submission[]; nextCursor: string | null };
    isLoading: boolean;
    refetch: () => void;
  };
  // Split panel logic
  const { splitRatio, handleMouseDown } = useSplitPanel();
  // Code persistence logic
  const { 
    code, 
    setCode, 
    selectedLanguage, 
    setSelectedLanguage, 
    selectedDataType, 
    setSelectedDataType, 
    isCodeDirty, 
    handleReset  } = useCodePersistence(slug, problem as Problem);
  
  // Submission stream logic
  const {
    isSubmitting,
    submissionStatus,
    isTestCaseTableOpen,
    isBenchmarking,
    setIsTestCaseTableOpen,
    processSubmission,
    startSubmission,
    setSubmissionStatus
  } = useSubmissionStream(submissionsQuery.refetch);

  // Create submission mutation
  const createSubmissionMutation = api.problems.createSubmission.useMutation({
    onSuccess: (data) => {
      void processSubmission(data.id);
    },
    onError: (error) => { 
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

  // Handle submission
  const handleSubmit = useCallback(() => {
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

    startSubmission();

    createSubmissionMutation.mutate({
      problemSlug: slug,
      code: code,
      language: selectedLanguage,
      gpuType: selectedGpuType,
    });
  }, [
    session?.user, 
    slug, 
    code, 
    selectedLanguage, 
    selectedGpuType, 
    createSubmissionMutation, 
    startSubmission, 
    toast
  ]);

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
                      submissionStatus.message?.startsWith("complete: ACCEPTED")
                        ? "green.900"
                        : submissionStatus.message?.startsWith(
                            "status: CHECKING"
                          ) ||
                          submissionStatus.message?.startsWith(
                            "checker: compiling"
                          ) ||
                          submissionStatus.message?.startsWith(
                            "checker: running"
                          ) ||
                          submissionStatus.message?.startsWith(
                            "benchmark: compiling"
                          ) ||
                          submissionStatus.message?.startsWith(
                            "benchmark: running"
                          ) ||
                          submissionStatus.message?.startsWith(
                            "benchmark: test_result"
                          ) ||
                          submissionStatus.status === "CHECKING" ||
                          submissionStatus.status === "BENCHMARKING"
                        ? "blue.900"
                        : "red.900"
                    }
                    p={4}
                    borderRadius="xl"
                  >
                    <HStack spacing={3}>
                      {submissionStatus.message?.startsWith(
                        "status: CHECKING"
                      ) ||
                      submissionStatus.message?.startsWith(
                        "checker: compiling"
                      ) ||
                      submissionStatus.message?.startsWith(
                        "checker: running"
                      ) ||
                      submissionStatus.message?.startsWith(
                        "benchmark: compiling"
                      ) ||
                      submissionStatus.message?.startsWith(
                        "benchmark: running"
                      ) ||
                      submissionStatus.message?.startsWith(
                        "benchmark: test_result"
                      ) ||
                      submissionStatus.status === "CHECKING" ||
                      submissionStatus.status === "BENCHMARKING" ? (
                        <Spinner size="sm" color="blue.200" />
                      ) : (
                        <Icon
                          as={
                            submissionStatus.status === "ACCEPTED" ||
                            submissionStatus.message?.startsWith(
                              "complete: ACCEPTED"
                            )
                              ? CheckIcon
                              : WarningIcon
                          }
                          boxSize={6}
                        />
                      )}
                      <VStack align="start" spacing={0}>
                        <Text fontSize="lg" fontWeight="semibold">
                          {getStatusMessage(submissionStatus)}
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

                  {submissionStatus.status === "WRONG_ANSWER" && (
                      <Box bg="red.900" p={6} borderRadius="xl">
                        {submissionStatus.errorMessage && (
                          <Text color="red.200" fontWeight="semibold" mb={3}>
                            {submissionStatus.errorMessage}
                          </Text>
                        )}
                        {submissionStatus.errorDetails && (() => {
                          try {
                            const debugInfo = JSON.parse(submissionStatus.errorDetails) as DebugInfo;
                            console.log(debugInfo);
                            return (
                              <VStack spacing={4} align="stretch">
                                {debugInfo.message && (
                                  <Text color="red.100">{debugInfo.message}</Text>
                                )}
                                {debugInfo.max_difference && (
                                  <Box>
                                    <Text color="red.200" fontSize="sm">Maximum Difference:</Text>
                                    <Text color="red.100">{debugInfo.max_difference}</Text>
                                  </Box>
                                )}
                                {debugInfo.mean_difference && (
                                  <Box>
                                    <Text color="red.200" fontSize="sm">Mean Difference:</Text>
                                    <Text color="red.100">{debugInfo.mean_difference}</Text>
                                  </Box>
                                )}
                                {debugInfo.sample_differences && Object.keys(debugInfo.sample_differences).length > 0 && (
                                  <Box>
                                    <Text color="red.200" fontSize="sm" mb={2}>Sample Differences:</Text>
                                    <Box maxH="200px" overflowY="auto">
                                      <Table size="sm" variant="unstyled">
                                        <Thead position="sticky" top={0}>
                                          <Tr>
                                            <Th color="red.200">Index</Th>
                                            <Th color="red.200" isNumeric>Expected</Th>
                                            <Th color="red.200" isNumeric>Actual</Th>
                                            <Th color="red.200" isNumeric>Difference</Th>
                                          </Tr>
                                        </Thead>
                                        <Tbody>
                                          {Object.entries(debugInfo.sample_differences).slice(0, 50).map(([key, value]) => (
                                            <Tr key={key}>
                                              <Td color="red.100">{key}</Td>
                                              <Td color="red.100" isNumeric>{value.expected.toFixed(7)}</Td>
                                              <Td color="red.100" isNumeric>{value.actual.toFixed(7)}</Td>
                                              <Td color="red.100" isNumeric>{value.diff.toFixed(7)}</Td>
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
                                {submissionStatus.errorDetails}
                              </Code>
                            );
                          }
                        })()}
                      </Box>
                    )}

                  {submissionStatus.status !== "WRONG_ANSWER" &&
                    submissionStatus.errorMessage &&
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
                    {
                      Object.entries(GPU_DISPLAY_NAMES)
                        .filter(([key]) => key !== "all")
                        .map(([key, value]) => (
                          <option key={key} value={key}>{value}</option>
                        ))
                    }
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
                    onChange={(e) => setSelectedLanguage(e.target.value as ProgrammingLanguage)}
                    value={selectedLanguage}
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
                    <option value="python">
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
                    value={selectedDataType}
                    onChange={(e) => setSelectedDataType(e.target.value as DataType)}
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
                theme="vs-dark"
                value={code}
                onChange={(value) => setCode(value ?? "")}
                language={selectedLanguage === "cuda" ? "cpp" : "python"}
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