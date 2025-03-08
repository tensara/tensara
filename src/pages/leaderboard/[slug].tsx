import { type NextPage } from "next";
import {
  Box,
  Table,
  Thead,
  Tbody,
  Tr,
  Th,
  Td,
  Text,
  Badge,
  Link as ChakraLink,
  Tooltip,
  Spinner,
  Flex,
  Heading,
  Select,
  HStack,
} from "@chakra-ui/react";
import { useState, useEffect } from "react";
import { api } from "~/utils/api";
import { Layout } from "~/components/layout";
import Link from "next/link";
import { useRouter } from "next/router";
import { formatDistanceToNow } from "date-fns";
import { createServerSideHelpers } from "@trpc/react-query/server";
import { appRouter } from "~/server/api/root";
import { createInnerTRPCContext } from "~/server/api/trpc";
import superjson from "superjson";
import type { GetServerSideProps } from "next";
import { GPU_DISPLAY_NAMES } from "~/constants/gpu";

type LeaderboardEntry = {
  id: string;
  createdAt: Date;
  runtime: number | null;
  gflops: number | null;
  user: {
    username: string | null;
  };
  problem: {
    slug: string;
    title: string;
  };
  status: string | null;
  gpuType: string | null;
};


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
    // Prefetch all submissions for the leaderboard
    await helpers.submissions.getLeaderboardSubmissions.prefetch();

    return {
      props: {
        trpcState: helpers.dehydrate(),
        slug,
      },
    };
  } catch (e) {
    console.error(e);
    return {
      notFound: true,
    };
  }
};

const LeaderboardPage: NextPage<{ slug: string }> = ({ slug }) => {
  const router = useRouter();
  const { gpu } = router.query;
  const [selectedGpu, setSelectedGpu] = useState<string>("all");

  // Update selectedGpu when URL parameter changes
  useEffect(() => {
    if (router.isReady && gpu) {
      setSelectedGpu(gpu as string);
    }
  }, [router.isReady, gpu]);

  // Update URL when selectedGpu changes (but not on initial mount)
  useEffect(() => {
    if (router.isReady && selectedGpu !== "all") {
      void router.push(
        {
          pathname: router.pathname,
          query: {
            ...router.query,
            gpu: selectedGpu,
          },
        },
        undefined,
        { shallow: true }
      );
    }
  }, [selectedGpu, router]);

  const { data: problem, isLoading: isProblemLoading } =
    api.problems.getById.useQuery({ slug: slug }, { enabled: !!slug });

  const { data: submissions, isLoading: isSubmissionsLoading } =
    api.submissions.getLeaderboardSubmissions.useQuery();

  // Filter submissions for the current problem
  const problemSubmissions = submissions?.filter(
    (submission) => submission.problem.slug === slug
  );

  // Process submissions to get the best submission per user per GPU type
  const getBestSubmissions = (submissions: LeaderboardEntry[] | undefined) => {
    if (!submissions) return [];

    const userGpuBestMap = new Map<string, LeaderboardEntry>();

    submissions.forEach((submission) => {
      if (submission.status !== "ACCEPTED" || !submission.gflops) return;
      
      if (selectedGpu !== "all" && submission.gpuType !== selectedGpu) return;

      const userGpuKey = `${submission.user.username ?? "Anonymous"}-${submission.gpuType}`;
      const currentBest = userGpuBestMap.get(userGpuKey);

      if (!currentBest || submission.gflops > currentBest.gflops!) {
        userGpuBestMap.set(userGpuKey, submission);
      }
    });

    return Array.from(userGpuBestMap.values()).sort(
      (a, b) => (b.gflops ?? 0) - (a.gflops ?? 0)
    );
  };

  const leaderboardEntries = getBestSubmissions(problemSubmissions);

  if (isProblemLoading || isSubmissionsLoading) {
    return (
      <Layout title="Leaderboard">
        <Box
          display="flex"
          justifyContent="center"
          alignItems="center"
          h="50vh"
        >
          <Spinner size="xl" />
        </Box>
      </Layout>
    );
  }

  if (!problem) {
    return (
      <Layout title="Leaderboard">
        <Box maxW="7xl" mx="auto" px={4} py={8}>
          <Text>Problem not found</Text>
        </Box>
      </Layout>
    );
  }

  return (
    <Layout
      title={`Leaderboard: ${problem.title} on ${GPU_DISPLAY_NAMES[selectedGpu]}`}
    >
      <Box maxW="7xl" mx="auto" px={4} py={8}>
        <Flex direction="column" gap={6}>
          <Heading size="lg">Leaderboard</Heading>
          <HStack justify="space-between" align="center">
            <Heading size="md">{problem.title}</Heading>
            <Select
              value={selectedGpu}
              onChange={(e) => setSelectedGpu(e.target.value)}
              w="200px"
              bg="whiteAlpha.50"
              borderColor="transparent"
              _hover={{ borderColor: "gray.600" }}
              _focus={{ borderColor: "gray.500" }}
            >
              {
                Object.entries(GPU_DISPLAY_NAMES).map(([key, value]) => (
                  <option key={key} value={key}>{value}</option>
                ))
              }
            </Select>
          </HStack>
          {leaderboardEntries.length === 0 ? (
            <Box p={4} textAlign="center" color="whiteAlpha.700">
              No submissions yet
              {selectedGpu !== "all"
                ? ` for ${GPU_DISPLAY_NAMES[selectedGpu]}`
                : ""}
              ,{" "}
              <ChakraLink
                as={Link}
                href={`/problems/${problem.slug}`}
                color="blue.400"
                _hover={{ textDecoration: "underline" }}
              >
                be the first to submit!
              </ChakraLink>
            </Box>
          ) : (
            <Box overflowX="auto">
              <Table variant="unstyled">
                <Thead>
                  <Tr>
                    <Th borderBottom="1px solid" borderColor="whiteAlpha.200">
                      Rank
                    </Th>
                    <Th borderBottom="1px solid" borderColor="whiteAlpha.200">
                      User
                    </Th>
                    <Th borderBottom="1px solid" borderColor="whiteAlpha.200">
                      GPU
                    </Th>
                    <Th borderBottom="1px solid" borderColor="whiteAlpha.200">
                      Performance
                    </Th>
                    <Th borderBottom="1px solid" borderColor="whiteAlpha.200">
                      Submitted
                    </Th>
                  </Tr>
                </Thead>
                <Tbody>
                  {leaderboardEntries.map((entry, index) => (
                    <Tr
                      key={entry.id}
                      _hover={{ bg: "whiteAlpha.50" }}
                      transition="background-color 0.2s"
                    >
                      <Td borderBottom="1px solid" borderColor="whiteAlpha.100">
                        <Badge
                          colorScheme={index < 3 ? "yellow" : "gray"}
                          fontSize="md"
                          px={3}
                          py={1}
                          borderRadius="full"
                        >
                          #{index + 1}
                        </Badge>
                      </Td>
                      <Td borderBottom="1px solid" borderColor="whiteAlpha.100">
                        <Text fontWeight={index < 3 ? "bold" : "normal"}>
                          {entry.user.username ?? "Anonymous"}
                        </Text>
                      </Td>
                      <Td borderBottom="1px solid" borderColor="whiteAlpha.100">
                        <Badge colorScheme="gray">
                          {GPU_DISPLAY_NAMES[entry.gpuType ?? "T4"]}
                        </Badge>
                      </Td>
                      <Td borderBottom="1px solid" borderColor="whiteAlpha.100">
                        <ChakraLink as={Link} href={`/submissions/${entry.id}`}>
                          <Tooltip
                            label={`Runtime: ${entry.runtime?.toFixed(2)} ms`}
                          >
                            <Text fontWeight="medium">
                              {entry.gflops?.toFixed(2)} GFLOPS
                            </Text>
                          </Tooltip>
                        </ChakraLink>
                      </Td>
                      <Td borderBottom="1px solid" borderColor="whiteAlpha.100">
                        <Tooltip
                          label={new Date(entry.createdAt).toLocaleString()}
                        >
                          <Text>
                            {formatDistanceToNow(new Date(entry.createdAt))} ago
                          </Text>
                        </Tooltip>
                      </Td>
                    </Tr>
                  ))}
                </Tbody>
              </Table>
            </Box>
          )}
        </Flex>
      </Box>
    </Layout>
  );
};

export default LeaderboardPage;
