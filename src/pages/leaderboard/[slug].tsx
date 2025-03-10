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
  Card,
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

type ProblemLeaderboardEntry = {
  id: string;
  username: string | null;
  gflops: number;
  runtime: number | null;
  createdAt: Date;
  gpuType: string | null;
};

export const getServerSideProps: GetServerSideProps = async (context) => {
  const slug = context.params?.slug as string;
  const gpuType = (context.query.gpu as string) || "all";

  const helpers = createServerSideHelpers({
    router: appRouter,
    ctx: createInnerTRPCContext({ session: null }),
    transformer: superjson,
  });

  // Prefetch problem data
  await helpers.problems.getById.prefetch({ slug });

  // Prefetch all relevant GPU types for this problem
  const gpuTypes = ["all", "A100", "H100", "RTX4090"];

  // Prefetch all GPU types for instant switching
  await Promise.all(
    gpuTypes.map((gt) =>
      helpers.submissions.getProblemLeaderboard.prefetch({
        slug,
        gpuType: gt,
      })
    )
  );

  return {
    props: {
      trpcState: helpers.dehydrate(),
      slug,
    },
  };
};

const LeaderboardPage: NextPage<{ slug: string }> = ({ slug }) => {
  const router = useRouter();
  const [selectedGpu, setSelectedGpu] = useState<string>(
    (router.query.gpu as string) || "all"
  );

  // Handle GPU selection change
  useEffect(() => {
    if (router.query.gpu !== selectedGpu && router.isReady) {
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

  // Get problem data
  const { data: problem, isLoading: isProblemLoading } =
    api.problems.getById.useQuery({ slug }, { enabled: !!slug });

  // Get leaderboard data from the new optimized endpoint
  const { data: leaderboardEntries = [], isLoading: isLeaderboardLoading } =
    api.submissions.getProblemLeaderboard.useQuery<ProblemLeaderboardEntry[]>(
      { slug, gpuType: selectedGpu },
      {
        staleTime: 300000, // 5 minutes
        refetchOnMount: false,
        refetchOnWindowFocus: false,
      }
    );

  if (isProblemLoading || isLeaderboardLoading) {
    return (
      <Layout>
        <Box
          height="50vh"
          display="flex"
          alignItems="center"
          justifyContent="center"
        >
          <Spinner size="xl" />
        </Box>
      </Layout>
    );
  }

  return (
    <Layout
      title={problem?.title ? `Leaderboard: ${problem.title}` : "Leaderboard"}
    >
      <Box maxW="7xl" mx="auto" px={4} py={8}>
        <Flex direction="column" gap={6}>
          <HStack justify="space-between" align="center">
            <Heading size="lg">
              {problem?.title ? (
                <>
                  <Link href="/leaderboard" passHref>
                    <ChakraLink as="span" mr={2}>
                      Leaderboards
                    </ChakraLink>
                  </Link>
                  {" > "}
                  {problem.title}
                </>
              ) : (
                "Leaderboard"
              )}
            </Heading>
            <Select
              value={selectedGpu}
              onChange={(e) => setSelectedGpu(e.target.value)}
              w="200px"
              bg="whiteAlpha.50"
              color="white"
            >
              <option value="all">All GPUs</option>
              {Object.entries(GPU_DISPLAY_NAMES).map(([key, value]) => (
                <option key={key} value={key}>
                  {value}
                </option>
              ))}
            </Select>
          </HStack>

          {leaderboardEntries && leaderboardEntries.length > 0 ? (
            <Box overflowX="auto">
              <Table variant="simple">
                <Thead>
                  <Tr>
                    <Th>Rank</Th>
                    <Th>User</Th>
                    <Th>GPU</Th>
                    <Th isNumeric>GFLOPS</Th>
                    <Th isNumeric>Runtime (ms)</Th>
                    <Th>Date</Th>
                  </Tr>
                </Thead>
                <Tbody>
                  {leaderboardEntries.map((entry, index) => (
                    <Tr
                      key={entry.id}
                      onClick={() => router.push(`/submissions/${entry.id}`)}
                      cursor="pointer"
                      _hover={{ bg: "whiteAlpha.100" }}
                    >
                      <Td>
                        <Badge
                          colorScheme={
                            index === 0
                              ? "yellow"
                              : index === 1
                              ? "gray"
                              : index === 2
                              ? "orange"
                              : "blue"
                          }
                        >
                          #{index + 1}
                        </Badge>
                      </Td>
                      <Td>{entry.username ?? "Anonymous"}</Td>
                      <Td>
                        <Badge colorScheme="purple">{entry.gpuType}</Badge>
                      </Td>
                      <Td isNumeric color="green.300" fontWeight="bold">
                        {(entry.gflops ?? 0).toFixed(2)}
                      </Td>
                      <Td isNumeric>{entry.runtime?.toFixed(2) ?? "N/A"}</Td>
                      <Td>{new Date(entry.createdAt).toLocaleDateString()}</Td>
                    </Tr>
                  ))}
                </Tbody>
              </Table>
            </Box>
          ) : (
            <Card p={6} bg="gray.700">
              <Text>
                No submissions found
                {selectedGpu !== "all" &&
                  ` for ${GPU_DISPLAY_NAMES[selectedGpu]}`}
                .
              </Text>
            </Card>
          )}
        </Flex>
      </Box>
    </Layout>
  );
};

export default LeaderboardPage;
