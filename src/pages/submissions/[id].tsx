import { type NextPage } from "next";
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
  Switch,
  FormControl,
  FormLabel,
  useToast,
  Alert,
  AlertIcon,
  AlertDescription,
} from "@chakra-ui/react";
import { CheckIcon, WarningIcon, TimeIcon } from "@chakra-ui/icons";
import Link from "next/link";
import { Editor } from "@monaco-editor/react";
import { useEffect, useState } from "react";
import { useSession } from "next-auth/react";
import { createServerSideHelpers } from "@trpc/react-query/server";
import { appRouter } from "~/server/api/root";
import { createInnerTRPCContext } from "~/server/api/trpc";
import superjson from "superjson";
import type { GetServerSideProps } from "next";
import { auth } from "~/server/auth";
import { TRPCError } from "@trpc/server";

type BenchmarkTestResult = {
  test_id: number;
  runtime_ms: number;
  gflops: number;
  name: string;
};

export const getServerSideProps: GetServerSideProps = async (context) => {
  const helpers = createServerSideHelpers({
    router: appRouter,
    ctx: createInnerTRPCContext({
      session: await auth(context.req, context.res),
    }),
    transformer: superjson,
  });

  const id = context.params?.id as string;

  try {
    await helpers.submissions.getSubmissionById.prefetch({ id });

    return {
      props: {
        trpcState: helpers.dehydrate(),
        id,
      },
    };
  } catch (error) {
    if (error instanceof TRPCError) {
      if (error.code === "FORBIDDEN") {
        return {
          props: {
            error: "You don't have permission to view this submission",
          },
        };
      }
    }
    return { notFound: true };
  }
};

const SubmissionPage: NextPage<{ id: string }> = ({ id }) => {
  // const router = useRouter();
  const [code, setCode] = useState("");
  const { data: session } = useSession();
  const toast = useToast();

  const {
    data: submission,
    isLoading,
    refetch,
  } = api.problems.getSubmissionStatus.useQuery(
    { submissionId: id },
    { enabled: !!id }
  );

  const togglePublicMutation = api.problems.toggleSubmissionPublic.useMutation({
    onSuccess: async () => {
      toast({
        title: "Submission updated",
        description: "Your submission's public status has been updated",
        status: "success",
        duration: 3000,
        isClosable: true,
      });
      // Refetch the submission data to update the UI
      await refetch();
    },
    onError: (error) => {
      toast({
        title: "Error updating submission",
        description: error.message,
        status: "error",
        duration: 5000,
        isClosable: true,
      });
    },
  });

  // Check if the current user is the submission owner
  const isOwner = session?.user?.id === submission?.userId;

  useEffect(() => {
    // Using optional chaining to safely access code which might not be present
    if (submission && "code" in submission) {
      setCode(submission.code as string);
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

  const handleTogglePublic = () => {
    if (submission.id) {
      togglePublicMutation.mutate({
        submissionId: submission.id,
        isPublic: !submission.isPublic,
      });
    }
  };

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

  const hasCode = "code" in submission;

  return (
    <Layout title={`Submission ${id}`}>
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

              {/* Public/Private Toggle (only for submission owner) */}
              {isOwner && (
                <FormControl display="flex" alignItems="center">
                  <FormLabel htmlFor="public-toggle" mb="0">
                    Make submission public
                  </FormLabel>
                  <Switch
                    id="public-toggle"
                    isChecked={submission.isPublic}
                    onChange={handleTogglePublic}
                    colorScheme="blue"
                  />
                </FormControl>
              )}

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
            {!hasCode ? (
              <Alert status="info" variant="solid" mb={4}>
                <AlertIcon />
                <AlertDescription>
                  {session
                    ? "You don't have access to this code. Ask the submission owner to make it public."
                    : "Sign in to view this submission's code."}
                </AlertDescription>
              </Alert>
            ) : (
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
            )}
          </Box>
        </VStack>
      </Box>
    </Layout>
  );
};

export default SubmissionPage;
