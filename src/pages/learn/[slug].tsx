import { useState, useCallback, useEffect } from "react";
import {
  Box,
  Spinner,
  Text,
  VStack,
  useToast,
  Icon,
  Heading,
  HStack,
  Button,
  Badge,
  Collapse,
  useDisclosure,
  Card,
  CardBody,
  Flex,
  Link as ChakraLink,
} from "@chakra-ui/react";
import superjson from "superjson";
import type { GetServerSideProps } from "next";
import { Layout } from "~/components/layout";
import CodeEditor from "~/components/problem/CodeEditor";
import SubmissionForm from "~/components/problem/SubmissionForm";
import SplitPanel from "~/components/problem/SplitPanel";
import { createInnerTRPCContext } from "~/server/api/trpc";
import { createServerSideHelpers } from "@trpc/react-query/server";
import { appRouter } from "~/server/api/root";
import { motion } from "framer-motion";
import {
  FiArrowLeft,
  FiHelpCircle,
  FiExternalLink,
  FiArrowRight,
} from "react-icons/fi";
import Link from "next/link";
import { useRouter } from "next/router";
const MotionBox = motion(Box);
import { gpuPuzzles } from "./sample";
import ReactMarkdown from "react-markdown";

// Helper function to get badge color based on difficulty
const getDifficultyColor = (difficulty: string) => {
  switch (difficulty) {
    case "beginner":
      return "green";
    case "intermediate":
      return "blue";
    case "advanced":
      return "purple";
    default:
      return "gray";
  }
};

export const getServerSideProps: GetServerSideProps = async (context) => {
  const helpers = createServerSideHelpers({
    router: appRouter,
    ctx: createInnerTRPCContext({ session: null }),
    transformer: superjson,
  });

  // Changed from id to slug
  const puzzleSlug = context.params?.slug as string;

  // Check if this puzzle exists
  if (!puzzleSlug || !gpuPuzzles[puzzleSlug]) {
    return {
      props: {
        trpcState: helpers.dehydrate(),
        puzzleSlug: null, // Use null instead of undefined
      },
    };
  }

  return {
    props: {
      trpcState: helpers.dehydrate(),
      puzzleSlug,
    },
  };
};

export default function PuzzlePage({
  puzzleSlug,
}: {
  puzzleSlug: string | null;
}) {
  const router = useRouter();
  const toast = useToast();
  const { isOpen: isHintsOpen, onToggle: toggleHints } = useDisclosure();
  const [selectedLanguage, setSelectedLanguage] = useState<"cuda" | "python">(
    "cuda"
  );
  const [code, setCode] = useState("");
  const [revealedHints, setRevealedHints] = useState<number[]>([]);

  const puzzle = puzzleSlug ? gpuPuzzles[puzzleSlug] : undefined;

  // Redirect to puzzles list if puzzle not found
  useEffect(() => {
    if (router.isReady && (puzzleSlug === null || !puzzle)) {
      toast({
        title: "Puzzle not found",
        description: "Redirecting to puzzles list",
        status: "error",
        duration: 3000,
        isClosable: true,
      });
      void router.push("/learn");
    }
  }, [puzzleSlug, puzzle, router, toast, router.isReady]);

  // Set initial language preference from URL query
  useEffect(() => {
    if (router.isReady) {
      const langParam = router.query.lang as string;
      if (langParam === "cuda" || langParam === "python") {
        setSelectedLanguage(langParam);
      }
    }
  }, [router.isReady, router.query]);

  // Set initial code when puzzle or language changes
  useEffect(() => {
    if (puzzle?.startingCode) {
      setCode(puzzle.startingCode[selectedLanguage] ?? "");
    }
  }, [puzzle, selectedLanguage]);

  // Handle submission
  const handleSubmit = useCallback(() => {
    toast({
      title: "Solution submitted",
      description: "Your solution has been submitted for evaluation.",
      status: "success",
      duration: 3000,
      isClosable: true,
    });
  }, [toast]);

  if (!puzzle) {
    return (
      <Layout title="Loading Puzzle...">
        <Box
          display="flex"
          justifyContent="center"
          alignItems="center"
          h="100%"
        >
          <VStack>
            <Spinner size="xl" />
            <Text mt={4}>Puzzle not found or loading...</Text>
          </VStack>
        </Box>
      </Layout>
    );
  }

  // Left content - Puzzle description and instructions
  const leftContent = (
    <MotionBox
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.5 }}
      h="100%"
      overflowY="auto"
      pr={{ base: 0, lg: 2 }}
    >
      <Box mb={4}>
        <Button
          as={Link}
          href="/learn"
          variant="ghost"
          leftIcon={<FiArrowLeft />}
          size="sm"
          color="whiteAlpha.700"
          _hover={{ color: "white", bg: "whiteAlpha.100" }}
        >
          Back to Puzzles
        </Button>
      </Box>
      <Heading as="h1" size="lg" mb={2} fontFamily="Space Grotesk, sans-serif">
        {puzzle.title}
      </Heading>
      {/* Author info */}
      <HStack mb={4}>
        <Text color="whiteAlpha.800" fontSize="sm">
          by {puzzle.author}
        </Text>
        <ChakraLink
          href="https://github.com/srush/GPU-Puzzles"
          isExternal
          color="green.400"
          fontSize="sm"
          display="flex"
          alignItems="center"
        >
          GitHub <Icon as={FiExternalLink} ml={1} boxSize={3} />
        </ChakraLink>
      </HStack>
      <HStack spacing={2} mb={4}>
        <Badge
          colorScheme={getDifficultyColor(puzzle.difficulty)}
          px={2}
          py={1}
          borderRadius="md"
          fontSize="xs"
          fontWeight="semibold"
          textTransform="uppercase"
        >
          {puzzle.difficulty}
        </Badge>
      </HStack>
      {/* Puzzle markdown content */}
      <Box mb={6}>
        <Box
          className="markdown-body"
          color="whiteAlpha.900"
          sx={{
            "& h1": { fontSize: "2xl", fontWeight: "bold", mt: 6, mb: 3 },
            "& h2": {
              fontSize: "xl",
              fontWeight: "bold",
              mt: 5,
              mb: 3,
              color: "green.400",
            },
            "& h3": { fontSize: "lg", fontWeight: "bold", mt: 4, mb: 2 },
            "& ul": { pl: 6, mb: 4 },
            "& li": { mb: 1 },
            "& p": { mb: 4 },
            "& code": {
              bg: "gray.700",
              px: 1,
              borderRadius: "sm",
              fontFamily: "monospace",
            },
            "& pre": {
              bg: "gray.800",
              p: 3,
              borderRadius: "md",
              overflowX: "auto",
              mb: 4,
            },
            "& pre code": { bg: "transparent", p: 0 },
            "& em": { fontStyle: "italic", color: "green.300" },
            "& strong": { fontWeight: "bold", color: "whiteAlpha.900" },
          }}
        >
          <ReactMarkdown>{puzzle.puzzleMd}</ReactMarkdown>
        </Box>
      </Box>
      {/* Hints Section */}
      <Box mb={6}>
        <Button
          onClick={toggleHints}
          leftIcon={<FiHelpCircle />}
          variant="outline"
          colorScheme="blue"
          size="sm"
          mb={2}
        >
          {isHintsOpen ? "Hide Hints" : "Show Hints"}
        </Button>
        <Collapse in={isHintsOpen} animateOpacity>
          <Card bg="blue.900" borderRadius="md">
            <CardBody>
              <Text fontWeight="medium" mb={2} color="whiteAlpha.900">
                Hints:
              </Text>
              <VStack align="start" spacing={2}>
                {puzzle.hints?.map((hint, index) => (
                  <Collapse key={index} in={revealedHints.includes(index)}>
                    <HStack align="start" spacing={2}>
                      <Text color="blue.300" fontWeight="bold">
                        {index + 1}.
                      </Text>
                      <Text color="whiteAlpha.900">{hint}</Text>
                    </HStack>
                  </Collapse>
                ))}
              </VStack>
              {isHintsOpen &&
                puzzle.hints &&
                revealedHints.length < puzzle.hints.length && (
                  <Button
                    size="sm"
                    variant="ghost"
                    colorScheme="blue"
                    mt={4}
                    onClick={() =>
                      setRevealedHints([...revealedHints, revealedHints.length])
                    }
                  >
                    Reveal Next Hint
                  </Button>
                )}
            </CardBody>
          </Card>
        </Collapse>
      </Box>
      {/* Navigation buttons */}
      <Flex justify="space-between" mt={10} mb={6}>
        <Button
          leftIcon={<FiArrowLeft />}
          isDisabled={!puzzle.prevPuzzleId}
          variant="ghost"
        >
          Previous Puzzle
        </Button>
        <Button
          rightIcon={<FiArrowRight />}
          isDisabled={!puzzle.nextPuzzleId}
          variant="ghost"
        >
          Next Puzzle
        </Button>
      </Flex>
    </MotionBox>
  );

  // Right panel - Editor and controls
  const rightContent = (
    <VStack w="100%" h="100%" spacing={4}>
      <SubmissionForm
        selectedGpuType="T4"
        setSelectedGpuType={() => undefined}
        selectedDataType="float32"
        setSelectedDataType={() => undefined}
        selectedLanguage={selectedLanguage === "cuda" ? "cuda" : "python"}
        setSelectedLanguage={(lang) =>
          setSelectedLanguage(lang === "cuda" ? "cuda" : "python")
        }
        isCodeDirty={false}
        onResetClick={() =>
          puzzle.startingCode && setCode(puzzle.startingCode[selectedLanguage])
        }
        onSubmit={handleSubmit}
        isSubmitting={false}
        isGpuSelectionDisabled={true}
        isLanguageSelectionDisabled={false}
        isDataTypeSelectionDisabled={true}
      />
      <CodeEditor
        code={code}
        setCode={setCode}
        selectedLanguage={selectedLanguage === "cuda" ? "cuda" : "python"}
      />
    </VStack>
  );

  return (
    <Layout title={`Learn: ${puzzle.title}`}>
      <SplitPanel leftContent={leftContent} rightContent={rightContent} />
    </Layout>
  );
}
