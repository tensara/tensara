import { useState, useCallback, useEffect } from "react";
import {
  Box,
  Spinner,
  Text,
  VStack,
  useToast,
  Icon,
  Heading,
  Modal,
  ModalOverlay,
  ModalContent,
  ModalHeader,
  ModalBody,
  ModalCloseButton,
  useDisclosure,
} from "@chakra-ui/react";
import { useSession } from "next-auth/react";
import superjson from "superjson";

import type { GetServerSideProps } from "next";
import { type Problem, type Submission } from "@prisma/client";

import { Layout } from "~/components/layout";
import MySubmissions from "~/components/problem/MySubmissions";
import ProblemView from "~/components/problem/ProblemView";
import CodeEditor from "~/components/problem/CodeEditor";
import SubmissionForm from "~/components/problem/SubmissionForm";
import SubmissionResults from "~/components/problem/SubmissionResults";
import ResetCodeModal from "~/components/problem/ResetCodeModal";
import SplitPanel from "~/components/problem/SplitPanel";
import ResizableConsole from "~/components/problem/Console";
import VerticalSplitPanel from "~/components/problem/VerticalSplitPanel";

import { FaExclamationCircle } from "react-icons/fa";

import { useCodePersistence } from "~/hooks/useCodePersistence";
import { useSubmissionStream } from "~/hooks/useSubmissionStream";
import { useSampleStream } from "~/hooks/useSampleStream";

import { validateCode } from "~/utils/starter";

import { createInnerTRPCContext } from "~/server/api/trpc";
import { createServerSideHelpers } from "@trpc/react-query/server";
import { appRouter } from "~/server/api/root";
import { api } from "~/utils/api";
import Editor from "@monaco-editor/react";

type ViewType = "submissions" | "problem" | "result";

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
  } catch (err: unknown) {
    if (err instanceof Error) {
      console.error(err);
    } else {
      console.error(String(err));
    }
    return {
      notFound: true,
    };
  }
};

export default function ProblemPage({ slug }: { slug: string }) {
  const { data: session } = useSession();
  const toast = useToast();
  const [selectedGpuType, setSelectedGpuType] = useState("T4");
  const [isResetModalOpen, setIsResetModalOpen] = useState(false);
  const [viewType, setViewType] = useState<ViewType>("problem");
  const { isOpen, onOpen, onClose } = useDisclosure();
  // const [consoleOutput, setConsoleOutput] = useState<string[]>([]);
  // const [isRunning, setIsRunning] = useState(false);

  const {
    output: consoleOutput,
    status,
    isRunning,
    startSampleRun,
  } = useSampleStream();

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

  // Code persistence logic
  const {
    code,
    setCode,
    selectedLanguage,
    setSelectedLanguage,
    selectedDataType,
    setSelectedDataType,
    isCodeDirty,
    handleReset,
  } = useCodePersistence(slug, problem as Problem);

  // Submission stream logic
  const {
    isSubmitting,
    metaStatus,
    metaResponse,
    testResults,
    benchmarkResults,
    isTestCaseTableOpen,
    isBenchmarking,
    setIsTestCaseTableOpen,
    processSubmission,
    startSubmission,
    totalTests,
    getTypedResponse,
  } = useSubmissionStream(submissionsQuery.refetch);

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

    const { valid, error } = validateCode(code, selectedLanguage);
    if (!valid) {
      toast({
        title: "Invalid code",
        description: error,
        status: "error",
        duration: 5000,
        isClosable: true,
      });
      return;
    }

    startSubmission();
    setViewType("result");

    void processSubmission({
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
    processSubmission,
    startSubmission,
    setViewType,
    toast,
  ]);
  const handleRun = useCallback(async () => {
    if (!session?.user) {
      toast({
        title: "Not signed in",
        description: "Please sign in to run solutions",
        status: "error",
        duration: 5000,
        isClosable: true,
      });
      return;
    }

    const { valid, error } = validateCode(code, selectedLanguage);
    if (!valid) {
      toast({
        title: "Invalid code",
        description: error,
        status: "error",
        duration: 5000,
        isClosable: true,
      });
      return;
    }

    await startSampleRun({
      code,
      language: selectedLanguage,
      gpuType: selectedGpuType,
      problemSlug: slug,
    });
  }, [startSampleRun, code, selectedLanguage, selectedGpuType, slug, toast]);

  // Keyboard shortcuts:
  // - Cmd+Enter => submit
  // - Cmd+' (Quote) => run sample
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      // Only respond to plain Meta (Cmd) + key (no Ctrl/Alt)
      if (!e.metaKey || e.ctrlKey || e.altKey) return;

      // Cmd+Enter -> submit
      if (e.key === "Enter") {
        e.preventDefault();
        void handleSubmit();
        return;
      }

      // Cmd+' (Quote) -> run sample
      // Some keyboards report "'" as the key, some report code === "Quote"
      if (e.key === "'" || e.code === "Quote") {
        e.preventDefault();
        void handleRun();
        return;
      }
    };

    const opts: AddEventListenerOptions = { capture: true };
    window.addEventListener("keydown", onKeyDown, opts);
    return () => window.removeEventListener("keydown", onKeyDown, opts);
  }, [handleSubmit, handleRun]);
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

  // Generate left content for split panel based on view type
  const leftContent = (() => {
    switch (viewType) {
      case "submissions":
        return (
          <MySubmissions
            submissions={submissionsQuery.data?.submissions}
            isLoading={submissionsQuery.isLoading}
            onBackToProblem={() => setViewType("problem")}
          />
        );
      case "result":
        return metaStatus ? (
          <SubmissionResults
            metaStatus={metaStatus}
            metaResponse={metaResponse}
            testResults={testResults}
            benchmarkResults={benchmarkResults}
            isTestCaseTableOpen={isTestCaseTableOpen}
            setIsTestCaseTableOpen={setIsTestCaseTableOpen}
            isBenchmarking={isBenchmarking}
            totalTests={totalTests}
            getTypedResponse={getTypedResponse}
            onBackToProblem={() => setViewType("problem")}
            onViewSubmissions={() => setViewType("submissions")}
          />
        ) : null;
      default:
        return (
          <ProblemView
            problem={problem}
            onViewSubmissions={() => setViewType("submissions")}
            onViewReference={onOpen}
          />
        );
    }
  })();

  // Right panel - Editor and controls
  const rightContent = (
    <VStack w="100%" h="100%" spacing={2}>
      <SubmissionForm
        selectedGpuType={selectedGpuType}
        setSelectedGpuType={setSelectedGpuType}
        selectedLanguage={selectedLanguage}
        setSelectedLanguage={setSelectedLanguage}
        selectedDataType={selectedDataType}
        setSelectedDataType={setSelectedDataType}
        isCodeDirty={isCodeDirty}
        onResetClick={() => setIsResetModalOpen(true)}
        onSubmit={handleSubmit}
        isSubmitting={isSubmitting}
        onRun={handleRun}
        isRunning={isRunning}
      />
      <Box flex={1} w="100%" minH={0}>
        <VerticalSplitPanel
          topContent={
            <CodeEditor
              code={code}
              setCode={setCode}
              selectedLanguage={selectedLanguage}
            />
          }
          bottomContent={
            <ResizableConsole
              output={consoleOutput}
              status={status}
              isRunning={isRunning}
            />
          }
          initialRatio={75}
          minTopHeight={40}
          minBottomHeight={20}
        />
      </Box>
    </VStack>
  );

  // Mobile warning
  const mobileWarning = (
    <Box
      display={{ base: "block", md: "none" }}
      w="100%"
      p={6}
      bg="whiteAlpha.50"
      borderRadius="xl"
      mb={4}
    >
      <VStack spacing={4} align="center">
        <Icon as={FaExclamationCircle} boxSize={10} color="yellow.400" />
        <Heading size="md" textAlign="center">
          Desktop Required for Code Submission
        </Heading>
        <Text textAlign="center" color="whiteAlpha.800">
          For the best coding experience, please switch to a desktop device to
          write and submit your solution.
        </Text>
      </VStack>
    </Box>
  );

  return (
    <Layout
      title={problem.title}
      ogTitle={`${problem.title}`}
      ogImgSubtitle={`${problem.difficulty.charAt(0) + problem.difficulty.toLowerCase().slice(1)} | Problems | Tensara`}
    >
      <Box
        bg="brand.secondary"
        borderRadius="xl"
        border="1px solid"
        borderColor="gray.800"
        h="100%"
        p={{ base: 3, md: 4 }}
        overflow="auto"
        display="flex"
        flexDirection="column"
      >
        <Box flex="1" overflow="auto" mb={2}>
          <SplitPanel leftContent={leftContent} rightContent={rightContent} />
        </Box>
        {mobileWarning}
        <ResetCodeModal
          isOpen={isResetModalOpen}
          onClose={() => setIsResetModalOpen(false)}
          onReset={handleReset}
        />
        <Modal isOpen={isOpen} onClose={onClose} isCentered size="4xl">
          <ModalOverlay bg="blackAlpha.800" backdropFilter="blur(5px)" />
          <ModalContent
            bg="brand.secondary"
            borderColor="whiteAlpha.100"
            borderWidth={1}
          >
            <ModalHeader color="white">Reference Solution</ModalHeader>
            <ModalCloseButton color="gray.400" />
            <ModalBody pb={6}>
              {problem.referenceSolution && (
                <Box
                  borderRadius="md"
                  overflow="hidden"
                  border="1px solid"
                  borderColor="whiteAlpha.200"
                >
                  {(() => {
                    const lines = problem.referenceSolution.split("\n").length;
                    const height = Math.min(lines * 20 + 100, 600);
                    return (
                      <Editor
                        height={height}
                        language="python"
                        value={problem.referenceSolution}
                        theme="vs-dark"
                        options={{
                          readOnly: true,
                          minimap: { enabled: false },
                          fontSize: 14,
                          lineNumbers: "on",
                          scrollBeyondLastLine: false,
                          fontFamily: "JetBrains Mono, monospace",
                        }}
                      />
                    );
                  })()}
                </Box>
              )}
            </ModalBody>
          </ModalContent>
        </Modal>
      </Box>
    </Layout>
  );
}
