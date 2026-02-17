import {
  Box,
  Text,
  VStack,
  HStack,
  Badge,
  Tabs,
  TabList,
  Tab,
  Link as ChakraLink,
} from "@chakra-ui/react";
import Link from "next/link";
import { Layout } from "~/components/layout";
import { PuzzleCard } from "~/components/learn/PuzzleCard";
import { GPU_PUZZLES } from "~/constants/puzzles";
import { useState } from "react";
import { FiArrowLeft, FiExternalLink } from "react-icons/fi";

const FILTERS = ["all", "intro", "easy", "medium", "hard"] as const;

export default function GpuPuzzlesPage() {
  const [filter, setFilter] = useState<string>("all");

  const filtered =
    filter === "all"
      ? GPU_PUZZLES
      : GPU_PUZZLES.filter((p) => p.difficulty === filter);

  return (
    <Layout title="GPU Puzzles">
      <Box maxW="4xl" mx="auto" py={8} px={4}>
        <Link href="/learn" passHref legacyBehavior>
          <HStack
            as="a"
            spacing={1}
            color="whiteAlpha.500"
            fontSize="sm"
            mb={6}
            cursor="pointer"
            _hover={{ color: "brand.primary" }}
          >
            <FiArrowLeft />
            <Text>Back to Learn</Text>
          </HStack>
        </Link>

        <VStack align="start" spacing={3} mb={8}>
          <Text fontSize="2xl" fontWeight="bold" color="white">
            🧩 GPU Puzzles
          </Text>
          <Text color="whiteAlpha.600" fontSize="sm" maxW="2xl">
            Learn GPU programming by solving progressively harder kernel
            puzzles. Based on{" "}
            <ChakraLink
              href="https://github.com/srush/GPU-Puzzles"
              isExternal
              color="brand.primary"
            >
              Sasha Rush&apos;s GPU Puzzles <FiExternalLink style={{ display: "inline" }} />
            </ChakraLink>
            . The exercises use NUMBA which directly maps Python code to CUDA
            kernels.
          </Text>

          <HStack spacing={4} mt={2}>
            <HStack spacing={1}>
              <Badge colorScheme="blue" fontSize="2xs">intro</Badge>
              <Text color="whiteAlpha.500" fontSize="xs">3</Text>
            </HStack>
            <HStack spacing={1}>
              <Badge colorScheme="green" fontSize="2xs">easy</Badge>
              <Text color="whiteAlpha.500" fontSize="xs">3</Text>
            </HStack>
            <HStack spacing={1}>
              <Badge colorScheme="yellow" fontSize="2xs">medium</Badge>
              <Text color="whiteAlpha.500" fontSize="xs">4</Text>
            </HStack>
            <HStack spacing={1}>
              <Badge colorScheme="red" fontSize="2xs">hard</Badge>
              <Text color="whiteAlpha.500" fontSize="xs">4</Text>
            </HStack>
          </HStack>
        </VStack>

        <Tabs
          variant="soft-rounded"
          colorScheme="green"
          size="sm"
          mb={6}
          onChange={(i) => setFilter(FILTERS[i] ?? "all")}
        >
          <TabList>
            {FILTERS.map((f) => (
              <Tab
                key={f}
                color="whiteAlpha.600"
                _selected={{ color: "white", bg: "whiteAlpha.200" }}
                fontSize="xs"
                textTransform="capitalize"
              >
                {f} {f === "all" ? `(${GPU_PUZZLES.length})` : ""}
              </Tab>
            ))}
          </TabList>
        </Tabs>

        <VStack spacing={4} align="stretch">
          {filtered.map((puzzle) => (
            <PuzzleCard key={puzzle.id} puzzle={puzzle} />
          ))}
        </VStack>

        {filtered.length === 0 && (
          <Text color="whiteAlpha.400" textAlign="center" py={10}>
            No puzzles match this filter.
          </Text>
        )}
      </Box>
    </Layout>
  );
}
