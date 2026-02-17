import { useState, useCallback, useEffect } from "react";
import { Box, VStack, Spinner, Text, useToast, HStack } from "@chakra-ui/react";
import { useSession } from "next-auth/react";
import { useRouter } from "next/router";
import Link from "next/link";
import { FiArrowLeft, FiArrowRight } from "react-icons/fi";

import { Layout } from "~/components/layout";
import PuzzleView from "~/components/learn/PuzzleView";
import PuzzleToolbar from "~/components/learn/PuzzleToolbar";
import CodeEditor from "~/components/problem/CodeEditor";
import SplitPanel from "~/components/problem/SplitPanel";
import VerticalSplitPanel from "~/components/problem/VerticalSplitPanel";
import ResizableConsole from "~/components/problem/Console";

import { useSampleStream } from "~/hooks/useSampleStream";
import { useSubmissionStream } from "~/hooks/useSubmissionStream";
import { useHotkey } from "~/hooks/useHotKey";
import { validateCode } from "~/utils/starter";

import { GPU_PUZZLES } from "~/constants/puzzles";
import type { ProgrammingLanguage } from "~/types/misc";

export default function PuzzleSolvePage() {
  const router = useRouter();
  const { slug } = router.query;
  const { data: session } = useSession();
  const toast = useToast();

  const puzzle = GPU_PUZZLES.find((p) => p.slug === slug);
  const puzzleIndex = puzzle ? GPU_PUZZLES.findIndex((p) => p.id === puzzle.id) : -1;
  const prevPuzzle = puzzleIndex > 0 ? GPU_PUZZLES[puzzleIndex - 1] : null;
  const nextPuzzle = puzzleIndex < GPU_PUZZLES.length - 1 ? GPU_PUZZLES[puzzleIndex + 1] : null;

  const [code, setCode] = useState("");
  const [selectedLanguage, setSelectedLanguage] = useState<ProgrammingLanguage>("cuda");
  const [selectedGpuType, setSelectedGpuType] = useState("T4");
  const [isCodeDirty, setIsCodeDirty] = useState(false);

  // Initialize code when puzzle loads
  useEffect(() => {
    if (puzzle) {
      const saved = localStorage.getItem(`puzzle-code-${puzzle.slug}-${selectedLanguage}`);
      if (saved) {
        setCode(saved);
        setIsCodeDirty(true);
      } else {
        setCode(puzzle.starterCode);
        setIsCodeDirty(false);
      }
    }
  }, [puzzle, selectedLanguage]);

  // Save code to localStorage on change
  useEffect(() => {
    if (puzzle && code && code !== puzzle.starterCode) {
      localStorage.setItem(`puzzle-code-${puzzle.slug}-${selectedLanguage}`, code);
    }
  }, [code, puzzle, selectedLanguage]);

  const handleSetCode = useCallback((newCode: string) => {
    setCode(newCode);
    setIsCodeDirty(true);
  }, []);

  const handleReset = useCallback(() => {
    if (puzzle) {
      setCode(puzzle.starterCode);
      setIsCodeDirty(false);
      localStorage.removeItem(`puzzle-code-${puzzle.slug}-${selectedLanguage}`);
    }
  }, [puzzle, selectedLanguage]);

  const {
    output: consoleOutput,
    status: sampleStatus,
    isRunning,
    startSampleRun,
  } = useSampleStream();

  const {
    isSubmitting,
    processSubmission,
    startSubmission,
  } = useSubmissionStream(() => {});

  const handleRun = useCallback(async () => {
    if (!session?.user) {
      toast({ title: "Sign in required", status: "error", duration: 3000 });
      return;
    }
    if (!puzzle) return;
    const { valid, error } = validateCode(code, selectedLanguage);
    if (!valid) {
      toast({ title: "Invalid code", description: error, status: "error", duration: 3000 });
      return;
    }
    await startSampleRun({
      code,
      language: selectedLanguage,
      gpuType: selectedGpuType,
      problemSlug: puzzle.slug,
    });
  }, [session, puzzle, code, selectedLanguage, selectedGpuType, startSampleRun, toast]);

  const handleSubmit = useCallback(() => {
    if (!session?.user) {
      toast({ title: "Sign in required", status: "error", duration: 3000 });
      return;
    }
    if (!puzzle) return;
    const { valid, error } = validateCode(code, selectedLanguage);
    if (!valid) {
      toast({ title: "Invalid code", description: error, status: "error", duration: 3000 });
      return;
    }
    startSubmission();
    void processSubmission({
      problemSlug: puzzle.slug,
      code,
      language: selectedLanguage,
      gpuType: selectedGpuType,
    });
  }, [session, puzzle, code, selectedLanguage, selectedGpuType, processSubmission, startSubmission, toast]);

  useHotkey("meta+enter", () => { if (!isSubmitting) handleSubmit(); });
  useHotkey("meta+'", () => { if (!isRunning) void handleRun(); });

  if (!router.isReady) {
    return (
      <Layout title="Loading...">
        <Box display="flex" justifyContent="center" alignItems="center" h="100%">
          <Spinner size="xl" />
        </Box>
      </Layout>
    );
  }

  if (!puzzle) {
    return (
      <Layout title="Not Found">
        <Box p={8}>
          <Text>Puzzle not found</Text>
        </Box>
      </Layout>
    );
  }

  const leftContent = (
    <Box h="100%" display="flex" flexDirection="column">
      <Box flex={1} overflowY="auto" p={4}>
        <PuzzleView
          puzzle={{
            title: puzzle.title,
            difficulty: puzzle.difficulty,
            description: puzzle.description,
            hint: puzzle.hint,
            order: puzzle.id,
            totalPuzzles: GPU_PUZZLES.length,
          }}
        />
      </Box>
      <HStack
        justify="space-between"
        px={4}
        py={2}
        borderTop="1px solid"
        borderColor="whiteAlpha.100"
        flexShrink={0}
      >
        {prevPuzzle ? (
          <Link href={`/learn/gpu-puzzles/${prevPuzzle.slug}`}>
            <HStack
              spacing={1}
              color="whiteAlpha.500"
              fontSize="sm"
              cursor="pointer"
              _hover={{ color: "brand.primary" }}
            >
              <FiArrowLeft />
              <Text>{prevPuzzle.title}</Text>
            </HStack>
          </Link>
        ) : <Box />}
        {nextPuzzle ? (
          <Link href={`/learn/gpu-puzzles/${nextPuzzle.slug}`}>
            <HStack
              spacing={1}
              color="whiteAlpha.500"
              fontSize="sm"
              cursor="pointer"
              _hover={{ color: "brand.primary" }}
            >
              <Text>{nextPuzzle.title}</Text>
              <FiArrowRight />
            </HStack>
          </Link>
        ) : <Box />}
      </HStack>
    </Box>
  );

  const rightContent = (
    <VStack w="100%" h="100%" spacing={2}>
      <PuzzleToolbar
        selectedLanguage={selectedLanguage}
        setSelectedLanguage={setSelectedLanguage}
        selectedGpuType={selectedGpuType}
        setSelectedGpuType={setSelectedGpuType}
        isCodeDirty={isCodeDirty}
        onResetClick={handleReset}
        onRun={() => void handleRun()}
        isRunning={isRunning}
        onSubmit={handleSubmit}
        isSubmitting={isSubmitting}
      />
      <Box flex={1} w="100%" minH={0}>
        <VerticalSplitPanel
          topContent={
            <CodeEditor
              code={code}
              setCode={handleSetCode}
              selectedLanguage={selectedLanguage}
              enableVimMode={false}
              onToggleVimMode={() => {}}
            />
          }
          bottomContent={
            <ResizableConsole
              output={consoleOutput}
              status={sampleStatus}
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

  return (
    <Layout
      title={`${puzzle.title} | GPU Puzzles`}
      ogTitle={puzzle.title}
      ogImgSubtitle="GPU Puzzles | Learn | Tensara"
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
        <Box flex="1" overflow="auto">
          <SplitPanel leftContent={leftContent} rightContent={rightContent} />
        </Box>
      </Box>
    </Layout>
  );
}
