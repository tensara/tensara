import { useState, useCallback, useEffect, useRef } from "react";
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
  Link as ChakraLink,
} from "@chakra-ui/react";
import { useSession } from "next-auth/react";
import superjson from "superjson";
import { useHotkey } from "~/hooks/useHotKey";

import type { GetServerSideProps } from "next";
import { type Problem, type Submission } from "@prisma/client";
import NextLink from "next/link";

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

import {
  SampleStatus,
  SubmissionStatus,
  type SampleStatusType,
} from "~/types/submission";
import {
  savePreferences,
  loadVimModePreference,
  saveVimModePreference,
} from "~/utils/localStorage";
import { validateCode } from "~/utils/starter";

import { createInnerTRPCContext } from "~/server/api/trpc";
import { createServerSideHelpers } from "@trpc/react-query/server";
import { appRouter } from "~/server/api/root";
import { api } from "~/utils/api";
import Editor from "@monaco-editor/react";
import { FlopsModal } from "~/components/misc/FlopsModal";

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
  const [isResetModalOpen, setIsResetModalOpen] = useState(false);
  const [viewType, setViewType] = useState<ViewType>("problem");
  const { isOpen, onOpen, onClose } = useDisclosure();
  const {
    isOpen: isFlopsModalOpen,
    onOpen: onFlopsModalOpen,
    onClose: onFlopsModalClose,
  } = useDisclosure();

  const {
    output: consoleOutput,
    status,
    isRunning,
    startSampleRun,
    ptxContent,
    sassContent,
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
    isCodeDirty,
    handleReset,
    savedGpuType,
    hasLoadedPreferences,
  } = useCodePersistence(slug, problem as Problem);

  const [selectedGpuType, setSelectedGpuType] = useState("T4");
  const [isVimModeEnabled, setIsVimModeEnabled] = useState(false);
  const [hasLoadedVimPreference, setHasLoadedVimPreference] = useState(false);

  // Update GPU type when saved preferences are loaded
  useEffect(() => {
    if (savedGpuType) {
      setSelectedGpuType(savedGpuType);
    }
  }, [savedGpuType]);

  useEffect(() => {
    const stored = loadVimModePreference();
    if (stored !== null) {
      setIsVimModeEnabled(stored);
    }
    setHasLoadedVimPreference(true);
  }, []);

  useEffect(() => {
    if (!hasLoadedVimPreference) return;
    saveVimModePreference(isVimModeEnabled);
  }, [isVimModeEnabled, hasLoadedVimPreference]);

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
    ptxContent: submissionPtxContent,
    sassContent: submissionSassContent,
    submissionId,
  } = useSubmissionStream(submissionsQuery.refetch);
  const lastSampleStatusRef = useRef<SampleStatusType>(SampleStatus.IDLE);
  const [wrongSubmissionStreak, setWrongSubmissionStreak] = useState(0);

  const [submissionPtxTimestamp, setSubmissionPtxTimestamp] =
    useState<number>(0);
  const [submissionSassTimestamp, setSubmissionSassTimestamp] =
    useState<number>(0);
  const [samplePtxTimestamp, setSamplePtxTimestamp] = useState<number>(0);
  const [sampleSassTimestamp, setSampleSassTimestamp] = useState<number>(0);
  const [ptxDirty, setPtxDirty] = useState(false);
  const [sassDirty, setSassDirty] = useState(false);

  useEffect(() => {
    if (submissionPtxContent) {
      setSubmissionPtxTimestamp(Date.now());
    }
  }, [submissionPtxContent]);

  useEffect(() => {
    if (submissionSassContent) {
      setSubmissionSassTimestamp(Date.now());
    }
  }, [submissionSassContent]);

  useEffect(() => {
    if (ptxContent) {
      setSamplePtxTimestamp(Date.now());
    }
  }, [ptxContent]);

  useEffect(() => {
    if (sassContent) {
      setSampleSassTimestamp(Date.now());
    }
  }, [sassContent]);

  const showHelpToast = useCallback(() => {
    const toastId = "need-help-toast";
    if (toast.isActive(toastId)) return;
    toast({
      id: toastId,
      title: "Need help?",
      description: (
        <>
          <ChakraLink as={NextLink} href="/blog" textDecoration="underline">
            Post a question on our blog
          </ChakraLink>{" "}
          to get tips from the community.
        </>
      ),
      duration: 5000,
      isClosable: true,
    });
  }, [toast]);

  useEffect(() => {
    if (
      status === SampleStatus.FAILED &&
      lastSampleStatusRef.current !== SampleStatus.FAILED
    ) {
      setWrongSubmissionStreak((prev) => prev + 1);
    }
    lastSampleStatusRef.current = status;
  }, [status, showHelpToast]);

  useEffect(() => {
    const status = metaResponse?.status;
    if (!status) return;

    if (status === SubmissionStatus.WRONG_ANSWER) {
      setWrongSubmissionStreak((prev) => prev + 1);
    } else {
      setWrongSubmissionStreak(0);
    }
  }, [metaResponse]);

  useEffect(() => {
    if (wrongSubmissionStreak >= 3) {
      showHelpToast();
      setWrongSubmissionStreak(0);
    }
  }, [wrongSubmissionStreak, showHelpToast]);

  const handleSetCode = useCallback(
    (newCode: string) => {
      setCode(newCode);
      if (selectedLanguage === "cuda") {
        const currentPtx =
          submissionPtxTimestamp > samplePtxTimestamp
            ? (submissionPtxContent ?? ptxContent)
            : (ptxContent ?? submissionPtxContent);
        const currentSass =
          submissionSassTimestamp > sampleSassTimestamp
            ? (submissionSassContent ?? sassContent)
            : (sassContent ?? submissionSassContent);

        if (currentPtx) {
          setPtxDirty(true);
        }
        if (currentSass) {
          setSassDirty(true);
        }
      }
    },
    [
      setCode,
      selectedLanguage,
      submissionPtxContent,
      ptxContent,
      submissionSassContent,
      sassContent,
      submissionPtxTimestamp,
      samplePtxTimestamp,
      submissionSassTimestamp,
      sampleSassTimestamp,
    ]
  );

  useEffect(() => {
    if (submissionPtxContent || ptxContent) {
      setPtxDirty(false);
    }
  }, [submissionPtxContent, ptxContent]);

  useEffect(() => {
    if (submissionSassContent || sassContent) {
      setSassDirty(false);
    }
  }, [submissionSassContent, sassContent]);

  useEffect(() => {
    if (
      hasLoadedPreferences &&
      slug &&
      selectedLanguage &&
      selectedGpuType
    ) {
      savePreferences(slug, {
        language: selectedLanguage,
        gpuType: selectedGpuType,
      });
    }
  }, [slug, selectedLanguage, selectedGpuType, hasLoadedPreferences]);

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
  }, [
    startSampleRun,
    session?.user,
    code,
    selectedLanguage,
    selectedGpuType,
    slug,
    toast,
  ]);

  // Cmd+Enter to submit
  useHotkey("meta+enter", () => {
    if (isSubmitting) {
      toast({
        title: "Already submitting",
        description: "Please wait for the submission to complete",
        status: "error",
        duration: 5000,
        isClosable: true,
      });
      return;
    }
    void handleSubmit();
  });

  // Cmd+' to run sample
  useHotkey("meta+'", () => {
    if (isRunning) {
      toast({
        title: "Already running",
        description: "Please wait for the sample run to complete",
        status: "error",
        duration: 5000,
        isClosable: true,
      });
      return;
    }
    void handleRun();
  });

  useHotkey("meta+shift+v", () => {
    setIsVimModeEnabled((prev) => !prev);
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
            submissionId={submissionId}
            onViewFlops={onFlopsModalOpen}
            hasFlopsCode={!!(problem as { getFlops?: string | null }).getFlops}
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
              key={`problem-editor-${isVimModeEnabled ? "vim" : "std"}`}
              code={code}
              setCode={handleSetCode}
              selectedLanguage={selectedLanguage}
              enableVimMode={isVimModeEnabled}
              onToggleVimMode={setIsVimModeEnabled}
              ptxContent={
                submissionPtxTimestamp > samplePtxTimestamp
                  ? (submissionPtxContent ?? ptxContent)
                  : (ptxContent ?? submissionPtxContent)
              }
              sassContent={
                submissionSassTimestamp > sampleSassTimestamp
                  ? (submissionSassContent ?? sassContent)
                  : (sassContent ?? submissionSassContent)
              }
              enablePtxSassView={selectedLanguage === "cuda"}
              ptxDirty={ptxDirty}
              sassDirty={sassDirty}
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
                <Box>
                  {(() => {
                    const lines = problem.referenceSolution.split("\n").length;
                    const height = Math.min(lines * 20 + 100, 600);
                    return (
                      <Editor
                        height={height}
                        language="python"
                        value={problem.referenceSolution}
                        theme="tensara-dark"
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
        <FlopsModal
          isOpen={isFlopsModalOpen}
          onClose={onFlopsModalClose}
          problemSlug={problem.slug}
          getFlops={(problem as { getFlops?: string | null }).getFlops}
        />
      </Box>
    </Layout>
  );
}
