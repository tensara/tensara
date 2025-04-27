import {
  Box,
  Heading,
  HStack,
  Badge,
  Button,
  Icon,
  Text,
} from "@chakra-ui/react";
import { IoMdTime } from "react-icons/io";
import { FiTrendingUp } from "react-icons/fi";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import rehypeHighlight from "rehype-highlight";
import { type Problem } from "@prisma/client";
import { getDifficultyColor } from "~/pages/problems";

interface ProblemViewProps {
  problem: Problem;
  onViewSubmissions: () => void;
}

const ProblemView = ({ problem, onViewSubmissions }: ProblemViewProps) => {
  return (
    <Box>
      <Heading as="h1" size="lg" mb={2}>
        {problem.title}
      </Heading>
      <HStack spacing={2} align="center" mb={6}>
        <Badge
          colorScheme={getDifficultyColor(problem.difficulty)}
          px={2}
          py={1}
          borderRadius="lg"
        >
          {problem.difficulty}
        </Badge>
        <Button
          variant="outline"
          height="28px"
          px={2}
          py={1}
          fontSize="xs"
          onClick={onViewSubmissions}
          leftIcon={<IoMdTime size={16} />}
          borderRadius="lg"
          borderColor="whiteAlpha.200"
          color="gray.300"
          cursor="pointer"
          _hover={{
            bg: "whiteAlpha.50",
            color: "white",
          }}
          // iconSpacing={1}
        >
          My Submissions
        </Button>
        <Button
          variant="outline"
          height="28px"
          px={2}
          py={1}
          fontSize="xs"
          onClick={() => {
            window.location.href = `/leaderboard/${problem.slug}`;
          }}
          leftIcon={<Icon as={FiTrendingUp} boxSize={3} />}
          borderRadius="lg"
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
          {problem.description}
        </ReactMarkdown>
      </Box>
    </Box>
  );
};

export default ProblemView;
