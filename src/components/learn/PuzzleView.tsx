import {
  Box,
  Heading,
  HStack,
  Badge,
  Text,
  Code,
  Collapse,
  VStack,
} from "@chakra-ui/react";
import { useState } from "react";
import { FiInfo } from "react-icons/fi";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import rehypeHighlight from "rehype-highlight";

const difficultyColor: Record<string, string> = {
  INTRO: "blue",
  EASY: "green",
  MEDIUM: "yellow",
  HARD: "red",
};

interface PuzzleViewProps {
  puzzle: {
    title: string;
    difficulty: string;
    description?: string | null;
    hint?: string | null;
    order: number;
    totalPuzzles: number;
  };
}

const PuzzleView = ({ puzzle }: PuzzleViewProps) => {
  const [showHint, setShowHint] = useState(false);

  return (
    <Box p={2} h="100%" overflowY="auto">
      <HStack spacing={2} mb={1}>
        <Text color="whiteAlpha.400" fontSize="xs">
          {puzzle.order} / {puzzle.totalPuzzles}
        </Text>
      </HStack>

      <Heading as="h1" size="lg" mb={2}>
        {puzzle.title}
      </Heading>

      <HStack spacing={2} mb={4}>
        <Badge
          colorScheme={difficultyColor[puzzle.difficulty.toUpperCase()] ?? "gray"}
          px={2}
          py={1}
          borderRadius="lg"
          textTransform="capitalize"
        >
          {puzzle.difficulty}
        </Badge>
      </HStack>

      {puzzle.hint && (
        <Box mb={4}>
          <HStack
            spacing={1}
            cursor="pointer"
            onClick={() => setShowHint(!showHint)}
            color="brand.primary"
            fontSize="sm"
            _hover={{ opacity: 0.8 }}
          >
            <FiInfo />
            <Text>{showHint ? "Hide hint" : "Show hint"}</Text>
          </HStack>
          <Collapse in={showHint}>
            <Box
              mt={2}
              p={3}
              bg="whiteAlpha.50"
              borderRadius="md"
              borderLeft="3px solid"
              borderColor="brand.primary"
            >
              <Text color="whiteAlpha.700" fontSize="sm">
                {puzzle.hint}
              </Text>
            </Box>
          </Collapse>
        </Box>
      )}

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
              <Text
                as="code"
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
          {puzzle.description}
        </ReactMarkdown>
      </Box>
    </Box>
  );
};

export default PuzzleView;
