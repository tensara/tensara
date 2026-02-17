import { Box, Text, SimpleGrid, VStack, HStack, Badge } from "@chakra-ui/react";
import { Layout } from "~/components/layout";
import { TopicCard } from "~/components/learn/TopicCard";

export default function LearnPage() {
  return (
    <Layout title="Learn">
      <Box maxW="6xl" mx="auto" py={8} px={4}>
        <VStack align="start" spacing={2} mb={10}>
          <HStack>
            <Text fontSize="3xl" fontWeight="bold" color="white">
              Learn GPU Programming
            </Text>
            <Badge colorScheme="green" fontSize="xs" mt={1}>
              Beta
            </Badge>
          </HStack>
          <Text color="whiteAlpha.600" fontSize="md" maxW="2xl">
            Master GPU programming from the ground up. Work through interactive
            puzzles, read curated guides, and build real intuition for parallel
            computing.
          </Text>
        </VStack>

        <Text
          color="whiteAlpha.500"
          fontSize="xs"
          fontWeight="bold"
          textTransform="uppercase"
          letterSpacing="wider"
          mb={4}
        >
          Topics
        </Text>

        <SimpleGrid columns={{ base: 1, md: 2, lg: 3 }} spacing={5}>
          <TopicCard
            title="GPU Puzzles"
            description="Learn GPU programming interactively by solving 14 progressively harder CUDA kernel puzzles. From basic maps to matrix multiply."
            href="/learn/gpu-puzzles"
            icon="🧩"
            puzzleCount={14}
            difficulty="Beginner → Advanced"
            tag="Popular"
          />
          <TopicCard
            title="CUDA Fundamentals"
            description="Understand the CUDA programming model — threads, blocks, grids, memory hierarchy, and synchronization."
            href="/learn/cuda-fundamentals"
            icon="📚"
            puzzleCount={0}
            difficulty="Beginner"
            tag="Coming Soon"
          />
          <TopicCard
            title="Memory Optimization"
            description="Deep dive into shared memory, coalescing, bank conflicts, and memory access patterns for peak performance."
            href="/learn/memory-optimization"
            icon="⚡"
            puzzleCount={0}
            difficulty="Intermediate"
            tag="Coming Soon"
          />
        </SimpleGrid>
      </Box>
    </Layout>
  );
}
