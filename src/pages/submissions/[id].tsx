import { type NextPage } from "next";
import { api } from "~/utils/api";
import { Layout } from "~/components/layout";
import {
  Table,
  Thead,
  Tbody,
  Tr,
  Th,
  Code,
  Td,
  Box,
  VStack,
  HStack,
  Text,
  Spinner,
  Icon,
  Link as ChakraLink,
  Switch,
  useToast,
  Alert,
  AlertIcon,
  AlertDescription,
  Button,
} from "@chakra-ui/react";
import Link from "next/link";
import { useEffect, useState } from "react";
import { useSession, signIn } from "next-auth/react";
import { useRouter } from "next/router";
import { createServerSideHelpers } from "@trpc/react-query/server";
import { appRouter } from "~/server/api/root";
import { createInnerTRPCContext } from "~/server/api/trpc";
import superjson from "superjson";
import type { GetServerSideProps } from "next";
import { auth } from "~/server/auth";
import { TRPCError } from "@trpc/server";
import { type DebugInfo } from "~/types/problem";
import {
  formatStatus,
  getStatusColor,
  getStatusIcon,
} from "~/constants/problem";
import { FiCopy } from "react-icons/fi";
import CodeEditor from "~/components/problem/CodeEditor";
import { type ProgrammingLanguage } from "~/types/misc";

type BenchmarkTestResult = {
  test_id: number;
  runtime_ms: number;
  gflops?: number;
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
    // Prefetch submission data
    await helpers.submissions.getSubmissionById.prefetch({ id });

    // Get submission data for metadata
    const submission = await helpers.submissions.getSubmissionById.fetch({
      id,
    });

    // Create an engaging social-friendly title
    let pageTitle = `View submission ${id.substring(0, 8)}`;

    if (submission) {
      const problemName = submission.problem.title ?? "problem";
      pageTitle = `View ${problemName} submission`;

      // Add performance if available
      if (submission.gflops) {
        pageTitle += ` (${submission.gflops.toFixed(2)} GFLOPS)`;
      }
    }

    return {
      props: {
        trpcState: helpers.dehydrate(),
        id,
        pageTitle,
      },
    };
  } catch (error) {
    if (error instanceof TRPCError) {
      if (error.code === "FORBIDDEN") {
        return {
          props: {
            error: "You don't have permission to view this submission",
            pageTitle: `View submission ${id.substring(0, 8)}`,
          },
        };
      }
    }
    return { notFound: true };
  }
};

const SubmissionPage: NextPage<{
  id: string;
  pageTitle: string;
  error?: string;
}> = ({ id, pageTitle }) => {
  const router = useRouter();
  const [code, setCode] = useState("");
  const { data: session } = useSession();
  const toast = useToast();

  const {
    data: submission,
    isLoading,
    refetch,
  } = api.submissions.getSubmissionById.useQuery({ id }, { enabled: !!id });

  const createDraftPost = api.blogpost.createDraft.useMutation();
  const autosavePost = api.blogpost.autosave.useMutation();

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
    if (
      submission &&
      "code" in submission &&
      typeof submission.code === "string"
    ) {
      setCode(submission.code);
    }
  }, [submission]);

  if (isLoading) {
    return (
      <Layout title={pageTitle}>
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
      <Layout title={pageTitle}>
        <Box p={8}>
          <Text>Submission not found</Text>
        </Box>
      </Layout>
    );
  }

  const handleTogglePublic = () => {
    if (submission?.id) {
      togglePublicMutation.mutate({
        submissionId: submission.id,
        isPublic: !submission.isPublic,
      });
    }
  };

  // Check if the submission has code (is public or user is owner)
  const hasCode = "code" in submission;
  const isPrivate = !submission.isPublic;
  const canViewCode = hasCode && (!isPrivate || isOwner);

  const handleCopyCode = () => {
    console.log("submission", submission);
    void navigator.clipboard.writeText(code);
    toast({
      title: "Code copied",
      status: "success",
      duration: 2000,
      isClosable: true,
    });
  };

  return (
    <Layout
      title={pageTitle}
      ogTitle={`View Submission | Tensara`}
      ogDescription={`View submission ${id.substring(0, 8)} on Tensara.`}
      ogImgSubtitle={`${submission.problem.title} | Submissions | Tensara`}
    >
      <Box maxW="7xl" mx="auto" px={4} py={8}>
        <VStack spacing={6} align="stretch">
          {/* Submission Status */}
          <Box
            bg="whiteAlpha.50"
            borderWidth="1px"
            borderColor="whiteAlpha.200"
            p={8}
            borderRadius="2xl"
            position="relative"
            overflow="hidden"
          >
            <VStack spacing={8} align="stretch">
              {/* Header with Problem and User Info */}
              <HStack
                justify="space-between"
                align="flex-start"
                wrap="wrap"
                spacing={6}
              >
                <VStack align="flex-start" spacing={3} flex="1">
                  <VStack align="flex-start" spacing={1}>
                    <ChakraLink
                      as={Link}
                      href={`/problems/${submission.problem.slug}`}
                      _hover={{ textDecoration: "none" }}
                    >
                      <Text
                        fontSize="3xl"
                        fontWeight="bold"
                        letterSpacing="tight"
                      >
                        {submission.problem.title}
                      </Text>
                    </ChakraLink>
                    <HStack spacing={3}>
                      <Text
                        fontSize="sm"
                        color="whiteAlpha.700"
                        letterSpacing="wider"
                        textTransform="uppercase"
                        bg="whiteAlpha.200"
                        px={3}
                        py={1}
                        borderRadius="full"
                      >
                        {submission.language}
                      </Text>
                      <Text
                        fontSize="sm"
                        color="whiteAlpha.700"
                        letterSpacing="wider"
                        textTransform="uppercase"
                        bg="whiteAlpha.200"
                        px={3}
                        py={1}
                        borderRadius="full"
                      >
                        {submission.gpuType}
                      </Text>
                    </HStack>
                  </VStack>
                  <HStack spacing={3} align="center">
                    <Text color="whiteAlpha.700">Submitted by</Text>
                    <ChakraLink
                      as={Link}
                      href={`/user/${submission.user.username}`}
                      _hover={{ textDecoration: "none" }}
                    >
                      <Text
                        color="blue.500"
                        fontWeight="medium"
                        _hover={{ color: "blue.400" }}
                      >
                        {submission.user.username ?? "Unknown User"}
                      </Text>
                    </ChakraLink>
                    <Text color="whiteAlpha.500" fontSize="sm">
                      • {new Date(submission.createdAt).toLocaleString()}
                    </Text>
                  </HStack>
                </VStack>

                {/* Status Badge */}
                <Box>
                  <HStack
                    bg={`${getStatusColor(submission.status)}.900`}
                    borderWidth="1px"
                    borderColor={`${getStatusColor(submission.status)}.700`}
                    px={4}
                    py={3}
                    borderRadius="xl"
                    spacing={3}
                  >
                    <Icon
                      as={getStatusIcon(submission.status)}
                      boxSize={5}
                      color={`${getStatusColor(submission.status)}.200`}
                    />
                    <Text
                      fontSize="md"
                      fontWeight="bold"
                      color={`${getStatusColor(submission.status)}.100`}
                      letterSpacing="wide"
                    >
                      {formatStatus(submission.status)}
                    </Text>
                  </HStack>
                </Box>
              </HStack>

              {/* Metrics Grid */}
              <Box
                bg="whiteAlpha.50"
                borderWidth="1px"
                borderColor="whiteAlpha.100"
                p={6}
                borderRadius="xl"
                position="relative"
                overflow="hidden"
              >
                <HStack spacing={12} wrap="wrap" justify="space-around">
                  {submission.runtime !== null && (
                    <VStack align="center" spacing={1}>
                      <Text
                        color="whiteAlpha.600"
                        fontSize="sm"
                        fontWeight="medium"
                        letterSpacing="wide"
                      >
                        RUNTIME
                      </Text>
                      <Text
                        fontSize="2xl"
                        fontWeight="bold"
                        color={`${getStatusColor(submission.status)}.400`}
                      >
                        {submission.runtime.toFixed(2)}
                        <Text
                          as="span"
                          fontSize="sm"
                          color="whiteAlpha.600"
                          ml={1}
                        >
                          ms
                        </Text>
                      </Text>
                    </VStack>
                  )}
                  {submission.gflops !== null && (
                    <VStack align="center" spacing={1}>
                      <Text
                        color="whiteAlpha.600"
                        fontSize="sm"
                        fontWeight="medium"
                        letterSpacing="wide"
                      >
                        PERFORMANCE
                      </Text>
                      <Text
                        fontSize="2xl"
                        fontWeight="bold"
                        color={`${getStatusColor(submission.status)}.400`}
                      >
                        {submission.gflops.toFixed(2)}
                        <Text
                          as="span"
                          fontSize="sm"
                          color="whiteAlpha.600"
                          ml={1}
                        >
                          GFLOPS
                        </Text>
                      </Text>
                    </VStack>
                  )}
                  {submission.passedTests !== null &&
                    submission.totalTests !== null && (
                      <VStack align="center" spacing={1}>
                        <Text
                          color="whiteAlpha.600"
                          fontSize="sm"
                          fontWeight="medium"
                          letterSpacing="wide"
                        >
                          TEST CASES
                        </Text>
                        <Text
                          fontSize="2xl"
                          fontWeight="bold"
                          color={`${getStatusColor(submission.status)}.400`}
                        >
                          {submission.passedTests}/{submission.totalTests}
                          <Text
                            as="span"
                            fontSize="sm"
                            color="whiteAlpha.600"
                            ml={1}
                          >
                            passed
                          </Text>
                        </Text>
                      </VStack>
                    )}
                </HStack>
              </Box>

              {/* Error Message */}
              {submission.status === "WRONG_ANSWER" && (
                <Box bg="red.900" p={6} borderRadius="xl">
                  {submission.errorMessage && (
                    <Text color="red.200" fontWeight="semibold" mb={3}>
                      {submission.errorMessage}
                    </Text>
                  )}
                  {submission.errorDetails &&
                    (() => {
                      try {
                        const debugInfo = JSON.parse(
                          submission.errorDetails
                        ) as DebugInfo;
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
                                <Text color="red.100">
                                  {debugInfo.max_difference}
                                </Text>
                              </Box>
                            )}
                            {debugInfo.mean_difference && (
                              <Box>
                                <Text color="red.200" fontSize="sm">
                                  Mean Difference:
                                </Text>
                                <Text color="red.100">
                                  {debugInfo.mean_difference}
                                </Text>
                              </Box>
                            )}
                            {debugInfo.sample_differences &&
                              Object.keys(debugInfo.sample_differences).length >
                                0 && (
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
                                        {Object.entries(
                                          debugInfo.sample_differences
                                        )
                                          .slice(0, 50)
                                          .map(([key, value]) => (
                                            <Tr key={key}>
                                              <Td color="red.100">{key}</Td>
                                              <Td color="red.100" isNumeric>
                                                {value.expected.toFixed(10)}
                                              </Td>
                                              <Td color="red.100" isNumeric>
                                                {value.actual.toFixed(10)}
                                              </Td>
                                              <Td color="red.100" isNumeric>
                                                {value.diff.toFixed(10)}
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
                            {submission.errorDetails}
                          </Code>
                        );
                      }
                    })()}
                </Box>
              )}

              {submission.status !== "WRONG_ANSWER" &&
                submission.errorMessage &&
                submission.errorDetails && (
                  <Box bg="red.900" p={6} borderRadius="xl">
                    <Text color="red.200" fontWeight="semibold" mb={3}>
                      Error Details ({submission.errorMessage})
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
                      {submission.errorDetails ?? submission.errorMessage}
                    </Code>
                  </Box>
                )}

              {/* Benchmark Results */}
              {Array.isArray(submission.benchmarkResults) && (
                <Box>
                  <Text color="whiteAlpha.700" fontSize="sm" mb={2}>
                    Benchmark Results
                  </Text>
                  <Box overflowX="auto">
                    {(() => {
                      const results: BenchmarkTestResult[] =
                        (submission.benchmarkResults as unknown as BenchmarkTestResult[]) ??
                        [];
                      const hasGflops = results.some(
                        (r: BenchmarkTestResult) => r.gflops !== undefined
                      );

                      const thStyle: React.CSSProperties = {
                        textAlign: "center",
                        padding: "8px",
                        borderBottom: "1px solid rgba(255,255,255,0.16)",
                      };
                      const tdStyle: React.CSSProperties = {
                        padding: "8px",
                        borderBottom: "1px solid rgba(255,255,255,0.16)",
                        textAlign: "center",
                      };

                      return (
                        <table
                          style={{
                            width: "100%",
                            borderCollapse: "collapse",
                            tableLayout: "fixed",
                          }}
                        >
                          <thead>
                            <tr>
                              <th style={thStyle}>Test Case</th>
                              <th style={thStyle}>Runtime (ms)</th>
                              {hasGflops && <th style={thStyle}>GFLOPS</th>}
                            </tr>
                          </thead>
                          <tbody>
                            {results.map(
                              (result: BenchmarkTestResult, index: number) => (
                                <tr key={index}>
                                  <td style={tdStyle}>{result.name}</td>
                                  <td style={tdStyle}>
                                    {result.runtime_ms.toFixed(2)}
                                  </td>
                                  {hasGflops && (
                                    <td style={tdStyle}>
                                      {result.gflops !== undefined
                                        ? result.gflops.toFixed(2)
                                        : ""}
                                    </td>
                                  )}
                                </tr>
                              )
                            )}
                          </tbody>
                        </table>
                      );
                    })()}
                  </Box>
                </Box>
              )}
            </VStack>
          </Box>

          {/* Code Editor */}
          <Box
            bg="whiteAlpha.50"
            borderWidth="1px"
            borderColor="whiteAlpha.200"
            p={8}
            borderRadius="2xl"
            position="relative"
            overflow="hidden"
          >
            <HStack justify="space-between" align="center" mb={4}>
              <Text fontWeight="semibold">Submitted Code</Text>
              <HStack spacing={4}>
                {canViewCode && (
                  <Button
                    leftIcon={<Icon as={FiCopy} />}
                    size="sm"
                    onClick={handleCopyCode}
                    colorScheme="blue"
                    variant="ghost"
                    _focus={{ bg: "none" }}
                    _hover={{ bg: "whiteAlpha.100" }}
                    transition="background 0.5s"
                  >
                    Copy Code
                  </Button>
                )}
                {canViewCode && (
                  <Button
                    size="sm"
                    colorScheme="green"
                    variant="ghost"
                    onClick={async () => {
                      if (!session) {
                        void signIn();
                        return;
                      }
                      if (!submission) return;

                      try {
                        // 1) Build initial title/content (same as your current localStorage draft)
                        const language = (() => {
                          const l = (submission.language ?? "").toLowerCase();
                          if (l === "cuda") return "cpp";
                          if (l === "triton") return "python";
                          return l;
                        })();

                        const titleDraft = `Solution: ${submission.problem?.title ?? ""}`;
                        const createdStr = new Date(
                          submission.createdAt
                        ).toLocaleString();
                        const runtimeStr =
                          submission.runtime != null
                            ? submission.runtime.toFixed(2)
                            : "N/A";
                        const gflopsStr =
                          submission.gflops != null
                            ? submission.gflops.toFixed(2)
                            : "N/A";
                        const testsStr =
                          submission.passedTests != null &&
                          submission.totalTests != null
                            ? `${submission.passedTests}/${submission.totalTests}`
                            : "N/A";
                        const problemLink = `https://tensara.org/problems/${
                          submission.problem?.slug ?? ""
                        }`;

                        const contentDraft = `# Solution: ${submission.problem?.title ?? ""}

[Problem: ${submission.problem?.title ?? ""}](${problemLink})

**Submitted on:** \`${createdStr}\`

- **Runtime:** \`${runtimeStr} ms\`
- **Performance:** \`${gflopsStr} GFLOPS\`
- **Tests:** \`${testsStr}\`

\`\`\`${language}
${code}
\`\`\``;

                        // 2) Create an empty DRAFT on the server
                        const newPost = await createDraftPost.mutateAsync({
                          title: titleDraft || "Untitled draft",
                        });

                        // 3) Immediately autosave the generated content + tags into that draft
                        await autosavePost.mutateAsync({
                          id: newPost.id,
                          title: titleDraft,
                          content: contentDraft,
                          // optional: seed tag strings; your server resolves strings -> tagIds
                          tagIds: undefined, // if you're using resolve-by-name server-side, omit this
                          // If you wired autosave to accept `tags: string[]`, pass that instead:
                          // tags: ["solution"],
                        });

                        // 4) Jump to the editor for this draft
                        toast({
                          title: "Draft created",
                          description: "Opening blog editor…",
                          status: "success",
                          duration: 1500,
                          isClosable: true,
                        });
                        void router.push(`/blog/edit/${newPost.id}`);
                      } catch (error) {
                        console.error(error);
                        const err = error as { message?: string } | undefined;
                        toast({
                          title: "Could not create draft",
                          description:
                            err?.message ?? String(error) ?? "Unexpected error",
                          status: "error",
                        });
                      }
                    }}
                    isLoading={
                      createDraftPost.isPending || autosavePost.isPending
                    }
                  >
                    Publish
                  </Button>
                )}
                {isOwner && (
                  <HStack spacing={2}>
                    <Text fontSize="sm" color="whiteAlpha.600">
                      Public
                    </Text>
                    <Switch
                      id="public-toggle"
                      isChecked={submission.isPublic}
                      onChange={handleTogglePublic}
                      colorScheme="blue"
                      size="sm"
                    />
                  </HStack>
                )}
              </HStack>
            </HStack>

            {!canViewCode ? (
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
                <CodeEditor
                  code={code}
                  setCode={setCode}
                  selectedLanguage={submission.language as ProgrammingLanguage}
                  isEditable={false}
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
