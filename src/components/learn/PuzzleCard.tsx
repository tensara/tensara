import {
  Box,
  Badge,
  Text,
  HStack,
} from "@chakra-ui/react";
import Link from "next/link";
import type { GpuPuzzle } from "~/constants/puzzles";

const difficultyColor: Record<string, string> = {
  intro: "blue",
  easy: "green",
  medium: "yellow",
  hard: "red",
};

export function PuzzleCard({ puzzle }: { puzzle: GpuPuzzle }) {
  return (
    <Link href={`/learn/gpu-puzzles/${puzzle.slug}`} passHref legacyBehavior>
      <Box
        as="a"
        display="block"
        bg="brand.secondary"
        border="1px solid"
        borderColor="whiteAlpha.100"
        borderRadius="lg"
        p={4}
        _hover={{
          borderColor: "brand.primary",
          transform: "translateY(-1px)",
          boxShadow: "0 2px 12px rgba(16, 185, 129, 0.1)",
        }}
        transition="all 0.2s"
        cursor="pointer"
      >
        <HStack justify="space-between">
          <HStack spacing={3}>
            <Text color="whiteAlpha.400" fontSize="sm" fontWeight="bold" minW="24px">
              {puzzle.id}
            </Text>
            <Text color="white" fontSize="md" fontWeight="semibold">
              {puzzle.title}
            </Text>
          </HStack>
          <Badge colorScheme={difficultyColor[puzzle.difficulty]} fontSize="xs">
            {puzzle.difficulty}
          </Badge>
        </HStack>
        <Text color="whiteAlpha.600" fontSize="sm" mt={2} noOfLines={1} pl="36px">
          {puzzle.description}
        </Text>
      </Box>
    </Link>
  );
}
