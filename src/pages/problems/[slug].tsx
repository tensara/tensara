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
} from "@chakra-ui/react";
import { useState, useEffect } from "react";
import { Layout } from "~/components/layout";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import rehypeHighlight from "rehype-highlight";
import Editor from "@monaco-editor/react";
import { CheckIcon, TimeIcon, WarningIcon } from "@chakra-ui/icons";

type BenchmarkTestResult = {
  test_id: number;
  runtime_ms: number;
  gflops: number;
};

type SubmissionStatus = {
  status: string | null;
  runtime: number | null;
  gflops: number | null;
  passedTests: number | null;
  totalTests: number | null;
  stage: "CHECKING" | "BENCHMARKING" | "COMPLETED" | null;
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
  const [submissionStatus, setSubmissionStatus] = useState<SubmissionStatus | null>(null);
  const [submissionId, setSubmissionId] = useState<string | null>(null);

  const submissionsQuery = api.problems.getSubmissions.useQuery(
    { problemSlug: slug as string, limit: 10 },
    { enabled: !!slug }
  ) as { data?: { submissions: Submission[]; nextCursor: string | null }; isLoading: boolean; refetch: () => void };

  const handleSubmit = () => {
    setIsSubmitting(true);
    setSubmissionStatus({
      status: "PENDING",
      runtime: null,
      gflops: null,
      passedTests: null,
      totalTests: null,
      stage: "CHECKING",
      message: "Running test cases...",
    });
    submitMutation.mutate({
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
    if (problem?.starterCode) {
      setCode(problem.starterCode);
    }
  }, [problem]);

  useEffect(() => {
    if (!submissionId) return;

    const eventSource = new EventSource(
      `/api/submissions/${submissionId}/status`
    );

    eventSource.onmessage = (event) => {
      const data = JSON.parse(event.data);
      setSubmissionStatus({
        status: data.status,
        runtime: data.runtime,
        gflops: data.gflops,
        passedTests: data.passedTests,
        totalTests: data.totalTests,
        stage:
          data.status === "CHECKING"
            ? "CHECKING"
            : data.status === "BENCHMARKING"
            ? "BENCHMARKING"
            : "COMPLETED",
        message:
          data.status === "CHECKING"
            ? "Running test cases..."
            : data.status === "BENCHMARKING"
            ? "Running performance benchmark..."
            : null,
        errorMessage: data.errorMessage,
        errorDetails: data.errorDetails,
        benchmarkResults: data.benchmarkResults,
      });

      if (
        data.status === "ACCEPTED" ||
        data.status === "ERROR" ||
        data.status === "WRONG_ANSWER" ||
        data.status === "TIMEOUT"
      ) {
        eventSource.close();
        setSubmissionId(null);
        setIsSubmitting(false);
        submissionsQuery.refetch();
      }
    };

    eventSource.onerror = () => {
      eventSource.close();
      setSubmissionId(null);
      setIsSubmitting(false);
    };

    return () => {
      eventSource.close();
    };
  }, [submissionId]);

  const submitMutation = api.problems.submit.useMutation({
    onSuccess: (data) => {
      setSubmissionId(data.id);
    },
    onError: (error) => {
      setIsSubmitting(false);
      setSubmissionStatus(null);
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
        <Box display="flex" justifyContent="center" alignItems="center">
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
        {/* Problem Description */}
        <Box w="50%" h="100%" overflowY="auto" p={6}>
          <Heading size="lg" mb={2}>
            {problem.title}
          </Heading>
          <Text color="gray.400" mb={6}>
            Difficulty: {problem.difficulty}
          </Text>

          <Box className="markdown" color="gray.100">
            <ReactMarkdown
              remarkPlugins={[remarkGfm, remarkMath]}
              rehypePlugins={[rehypeKatex, rehypeHighlight]}
              components={{
                h1: (props) => (
                  <Heading as="h1" size="xl" mt={6} mb={4} {...props} />
                ),
                h2: (props) => (
                  <Heading as="h2" size="lg" mt={5} mb={3} {...props} />
                ),
                h3: (props) => (
                  <Heading as="h3" size="md" mt={4} mb={2} {...props} />
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

        {/* Code Editor and Submission */}
        <VStack w="50%" h="100%" spacing={4}>
          <Box
            w="100%"
            h="calc(100% - 300px)"
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

          <Box w="100%" bg="gray.800" borderRadius="xl" overflow="hidden">
            <Tabs variant="enclosed" colorScheme="blue">
              <TabList bg="gray.900" px={4} pt={2}>
                <Tab
                  _selected={{
                    bg: "gray.800",
                    borderBottomColor: "transparent",
                  }}
                >
                  Submit
                </Tab>
                <Tab
                  _selected={{
                    bg: "gray.800",
                    borderBottomColor: "transparent",
                  }}
                >
                  Submissions
                </Tab>
              </TabList>

              <TabPanels>
                <TabPanel p={6}>
                  <VStack spacing={6} align="stretch">
                    <Button
                      colorScheme="blue"
                      size="lg"
                      width="100%"
                      onClick={handleSubmit}
                      isLoading={isSubmitting}
                      loadingText="Submitting..."
                      borderRadius="lg"
                      height="56px"
                      fontSize="md"
                      fontWeight="semibold"
                      _hover={{
                        transform: "translateY(-1px)",
                        boxShadow: "lg",
                      }}
                      transition="all 0.2s"
                    >
                      Submit Solution
                    </Button>

                    {submissionStatus && (
                      <Box
                        bg="gray.800"
                        borderRadius="xl"
                        overflow="hidden"
                        borderWidth="1px"
                        borderColor="gray.700"
                      >
                        <Box
                          p={4}
                          borderBottomWidth={submissionStatus.stage !== "COMPLETED" ? "1px" : "0"}
                          borderColor="gray.700"
                          bg={
                            submissionStatus.status === "ACCEPTED"
                              ? "green.900"
                              : submissionStatus.status === "PENDING"
                              ? "blue.900"
                              : "red.900"
                          }
                        >
                          <HStack spacing={3}>
                            <Box>
                              {submissionStatus.status === "ACCEPTED" ? (
                                <CheckIcon boxSize={5} />
                              ) : submissionStatus.status === "PENDING" ? (
                                <TimeIcon boxSize={5} />
                              ) : (
                                <WarningIcon boxSize={5} />
                              )}
                            </Box>
                            <VStack align="start" spacing={0}>
                              <Text fontSize="lg" fontWeight="semibold">
                                {submissionStatus.status}
                              </Text>
                              {submissionStatus.stage && submissionStatus.message && (
                                <Text fontSize="sm" color="whiteAlpha.700">
                                  {submissionStatus.message}
                                </Text>
                              )}
                            </VStack>
                          </HStack>
                        </Box>

                        <Box p={6}>
                          <VStack spacing={6} align="stretch">
                            {submissionStatus.passedTests !== null && (
                              <HStack justify="space-between" px={4} py={3} bg="whiteAlpha.50" borderRadius="lg">
                                <Text>Test Cases</Text>
                                <HStack spacing={1}>
                                  <Text fontWeight="semibold">{submissionStatus.passedTests}</Text>
                                  <Text color="whiteAlpha.700">/</Text>
                                  <Text color="whiteAlpha.700">{submissionStatus.totalTests ?? "N/A"}</Text>
                                </HStack>
                              </HStack>
                            )}

                            {submissionStatus.status === "ACCEPTED" && (
                              <SimpleGrid columns={2} spacing={4}>
                                {submissionStatus.runtime && (
                                  <Box p={4} bg="whiteAlpha.50" borderRadius="lg">
                                    <Text fontSize="sm" color="whiteAlpha.700" mb={1}>Runtime</Text>
                                    <Text fontSize="xl" fontWeight="semibold">
                                      {submissionStatus.runtime.toFixed(2)}ms
                                    </Text>
                                  </Box>
                                )}
                                {submissionStatus.gflops && (
                                  <Box p={4} bg="whiteAlpha.50" borderRadius="lg">
                                    <Text fontSize="sm" color="whiteAlpha.700" mb={1}>Performance</Text>
                                    <Text fontSize="xl" fontWeight="semibold">
                                      {submissionStatus.gflops.toFixed(2)} GFLOPS
                                    </Text>
                                  </Box>
                                )}
                              </SimpleGrid>
                            )}

                            {submissionStatus.errorMessage && (
                              <Box bg="red.900" p={4} borderRadius="lg">
                                <Text color="red.200" fontWeight="semibold" mb={2}>
                                  Error Details
                                </Text>
                                <Code
                                  display="block"
                                  whiteSpace="pre-wrap"
                                  p={3}
                                  bg="red.800"
                                  color="red.100"
                                  borderRadius="md"
                                  fontSize="sm"
                                >
                                  {submissionStatus.errorDetails || submissionStatus.errorMessage}
                                </Code>
                              </Box>
                            )}

                            {submissionStatus.benchmarkResults && submissionStatus.benchmarkResults.length > 0 && (
                              <Box>
                                <Text fontWeight="semibold" mb={3}>
                                  Benchmark Results
                                </Text>
                                <Box borderRadius="lg" overflow="hidden" borderWidth="1px" borderColor="gray.700">
                                  <Table size="sm" variant="unstyled">
                                    <Thead bg="whiteAlpha.100">
                                      <Tr>
                                        <Th color="whiteAlpha.700" py={3}>Test</Th>
                                        <Th color="whiteAlpha.700" py={3} isNumeric>Runtime (ms)</Th>
                                        <Th color="whiteAlpha.700" py={3} isNumeric>GFLOPS</Th>
                                      </Tr>
                                    </Thead>
                                    <Tbody>
                                      {submissionStatus.benchmarkResults.map((result) => (
                                        <Tr key={result.test_id} _hover={{ bg: "whiteAlpha.50" }}>
                                          <Td py={3}>Test {result.test_id}</Td>
                                          <Td py={3} isNumeric>{result.runtime_ms.toFixed(2)}</Td>
                                          <Td py={3} isNumeric>{result.gflops.toFixed(2)}</Td>
                                        </Tr>
                                      ))}
                                    </Tbody>
                                  </Table>
                                </Box>
                              </Box>
                            )}
                          </VStack>
                        </Box>
                      </Box>
                    )}
                  </VStack>
                </TabPanel>

                <TabPanel p={4}>
                  {submissionsQuery.isLoading ? (
                    <Box display="flex" justifyContent="center" p={4}>
                      <Spinner />
                    </Box>
                  ) : submissionsQuery.data?.submissions.length === 0 ? (
                    <Text color="gray.400" textAlign="center">
                      No submissions yet
                    </Text>
                  ) : (
                    <Box overflowX="auto">
                      <Table variant="simple" size="sm">
                        <Thead>
                          <Tr>
                            <Th>Status</Th>
                            <Th>Runtime</Th>
                            <Th>GFLOPS</Th>
                            <Th>Passed Tests</Th>
                            <Th>Submitted</Th>
                          </Tr>
                        </Thead>
                        <Tbody>
                          {submissionsQuery.data?.submissions.map(
                            (submission) => (
                              <Tr key={submission.id}>
                                <Td>
                                  <Badge
                                    colorScheme={
                                      submission.status === "ACCEPTED"
                                        ? "green"
                                        : submission.status === "PENDING" ||
                                          submission.status === "CHECKING" ||
                                          submission.status === "BENCHMARKING"
                                        ? "yellow"
                                        : "red"
                                    }
                                  >
                                    {submission.status}
                                  </Badge>
                                </Td>
                                <Td>
                                  {submission.runtime
                                    ? `${submission.runtime.toFixed(2)}ms`
                                    : "-"}
                                </Td>
                                <Td>
                                  {submission.gflops
                                    ? `${submission.gflops.toFixed(2)} GFLOPS`
                                    : "-"}
                                </Td>
                                <Td>
                                  {submission.passedTests
                                    ? `${submission.passedTests} / ${submission.totalTests}`
                                    : "-"}
                                </Td>
                                <Td>
                                  {new Date(
                                    submission.createdAt
                                  ).toLocaleString()}
                                </Td>
                              </Tr>
                            )
                          )}
                        </Tbody>
                      </Table>
                    </Box>
                  )}
                </TabPanel>
              </TabPanels>
            </Tabs>
          </Box>
        </VStack>
      </HStack>
    </Layout>
  );
}
