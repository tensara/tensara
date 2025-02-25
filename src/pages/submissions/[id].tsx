import { type NextPage } from "next";
import { useRouter } from "next/router";
import { api } from "~/utils/api";
import { Layout } from "~/components/layout";
import {
  Box,
  VStack,
  HStack,
  Text,
  Spinner,
  Icon,
  Badge,
  Link as ChakraLink,
} from "@chakra-ui/react";
import { CheckIcon, WarningIcon, TimeIcon } from "@chakra-ui/icons";
import Link from "next/link";
import { Editor } from "@monaco-editor/react";
import { useEffect, useState } from "react";

type BenchmarkTestResult = {
  test_id: number;
  runtime_ms: number;
  gflops: number;
  name: string;
};

const SubmissionPage: NextPage = () => {
  const router = useRouter();
  const { id } = router.query;
  const [code, setCode] = useState("");

  const { data: submission, isLoading } =
    api.problems.getSubmissionStatus.useQuery(
      { submissionId: id as string },
      { enabled: !!id }
    );

  useEffect(() => {
    if (submission?.code) {
      setCode(submission.code);
    }
  }, [submission]);

  if (isLoading) {
    return (
      <Layout title="Loading...">
        <Box
          display="flex"
          justifyContent="center"
          alignItems="center"
          h="100vh"
        >
          <Spinner size="xl" />
        </Box>
      </Layout>
    );
  }

  if (!submission) {
    return (
      <Layout title="Not Found">
        <Box p={8}>
          <Text>Submission not found</Text>
        </Box>
      </Layout>
    );
  }

  const getStatusColor = (status: string | null) => {
    switch (status) {
      case "ACCEPTED":
        return "green";
      case "WRONG_ANSWER":
      case "ERROR":
        return "red";
      case "CHECKING":
      case "BENCHMARKING":
        return "blue";
      default:
        return "gray";
    }
  };

  const formatStatus = (status: string | null) => {
    switch (status) {
      case "ACCEPTED":
        return "Accepted";
      case "WRONG_ANSWER":
        return "Wrong Answer";
      case "ERROR":
        return "Error";
      case "CHECKING":
        return "Checking";
      case "BENCHMARKING":
        return "Benchmarking";
      default:
        return status ?? "Unknown";
    }
  };

  return (
    <Layout title={`Submission ${id as string}`}>
      <Box maxW="7xl" mx="auto" px={4} py={8}>
        <VStack spacing={6} align="stretch">
          {/* Problem Link */}
          <Box>
            <ChakraLink
              as={Link}
              href={`/problems/${submission.problem.slug}`}
              color="blue.400"
            >
              ← Back to {submission.problem.title}
            </ChakraLink>
          </Box>

          {/* Submission Status */}
          <Box bg="whiteAlpha.50" p={6} borderRadius="xl">
            <VStack spacing={4} align="stretch">
              <HStack spacing={3}>
                <Icon
                  as={
                    submission.status === "ACCEPTED"
                      ? CheckIcon
                      : submission.status === "WRONG_ANSWER" ||
                        submission.status === "ERROR"
                      ? WarningIcon
                      : TimeIcon
                  }
                  boxSize={6}
                  color={`${getStatusColor(submission.status)}.400`}
                />
                <Text fontSize="xl" fontWeight="bold">
                  {formatStatus(submission.status)}
                </Text>
                <Badge
                  colorScheme={getStatusColor(submission.status)}
                  fontSize="sm"
                >
                  {submission.language}
                </Badge>
              </HStack>

              {/* Metrics */}
              <HStack spacing={6} wrap="wrap">
                {submission.runtime !== null && (
                  <Box>
                    <Text color="whiteAlpha.700" fontSize="sm">
                      Runtime
                    </Text>
                    <Text fontSize="lg" fontWeight="semibold">
                      {submission.runtime.toFixed(2)} ms
                    </Text>
                  </Box>
                )}
                {submission.gflops !== null && (
                  <Box>
                    <Text color="whiteAlpha.700" fontSize="sm">
                      Performance
                    </Text>
                    <Text fontSize="lg" fontWeight="semibold">
                      {submission.gflops.toFixed(2)} GFLOPS
                    </Text>
                  </Box>
                )}
                {submission.passedTests !== null &&
                  submission.totalTests !== null && (
                    <Box>
                      <Text color="whiteAlpha.700" fontSize="sm">
                        Test Cases
                      </Text>
                      <Text fontSize="lg" fontWeight="semibold">
                        {submission.passedTests}/{submission.totalTests} passed
                      </Text>
                    </Box>
                  )}
              </HStack>

              {/* Error Message */}
              {submission.errorMessage && (
                <Box bg="red.900" p={4} borderRadius="lg">
                  <Text color="red.200" fontWeight="semibold" mb={2}>
                    Error Message:
                  </Text>
                  <Text color="red.100" whiteSpace="pre-wrap">
                    {submission.errorMessage}
                  </Text>
                  {submission.errorDetails && (
                    <>
                      <Text color="red.200" fontWeight="semibold" mt={4} mb={2}>
                        Error Details:
                      </Text>
                      <Text color="red.100" whiteSpace="pre-wrap">
                        {submission.errorDetails}
                      </Text>
                    </>
                  )}
                </Box>
              )}

              {/* Benchmark Results */}
              {submission.benchmarkResults && (
                <Box>
                  <Text color="whiteAlpha.700" fontSize="sm" mb={2}>
                    Benchmark Results
                  </Text>
                  <Box overflowX="auto">
                    <table
                      style={{ width: "100%", borderCollapse: "collapse" }}
                    >
                      <thead>
                        <tr>
                          <th
                            style={{
                              textAlign: "left",
                              padding: "8px",
                              borderBottom: "1px solid rgba(255,255,255,0.16)",
                            }}
                          >
                            Test Case
                          </th>
                          <th
                            style={{
                              textAlign: "left",
                              padding: "8px",
                              borderBottom: "1px solid rgba(255,255,255,0.16)",
                            }}
                          >
                            Runtime (ms)
                          </th>
                          <th
                            style={{
                              textAlign: "left",
                              padding: "8px",
                              borderBottom: "1px solid rgba(255,255,255,0.16)",
                            }}
                          >
                            GFLOPS
                          </th>
                        </tr>
                      </thead>
                      <tbody>
                        {(
                          submission.benchmarkResults as BenchmarkTestResult[]
                        ).map((result, index) => (
                          <tr key={index}>
                            <td
                              style={{
                                padding: "8px",
                                borderBottom:
                                  "1px solid rgba(255,255,255,0.16)",
                              }}
                            >
                              {result.name}
                            </td>
                            <td
                              style={{
                                padding: "8px",
                                borderBottom:
                                  "1px solid rgba(255,255,255,0.16)",
                              }}
                            >
                              {result.runtime_ms.toFixed(2)}
                            </td>
                            <td
                              style={{
                                padding: "8px",
                                borderBottom:
                                  "1px solid rgba(255,255,255,0.16)",
                              }}
                            >
                              {result.gflops.toFixed(2)}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </Box>
                </Box>
              )}
            </VStack>
          </Box>

          {/* Code Editor */}
          <Box bg="whiteAlpha.50" p={6} borderRadius="xl">
            <Text mb={4} fontWeight="semibold">
              Submitted Code
            </Text>
            <Box h="600px" borderRadius="lg" overflow="hidden">
              <Editor
                height="100%"
                defaultLanguage="cpp"
                value={code}
                theme="vs-dark"
                options={{
                  readOnly: true,
                  minimap: { enabled: true },
                  fontSize: 14,
                  lineNumbers: "on",
                  scrollBeyondLastLine: false,
                  automaticLayout: true,
                  padding: { top: 16, bottom: 16 },
                  fontFamily: "JetBrains Mono, monospace",
                }}
              />
            </Box>
          </Box>
        </VStack>
      </Box>
    </Layout>
  );
};

export default SubmissionPage;
