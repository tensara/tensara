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
import { GPU_DISPLAY_NAMES, gpuTypes } from "~/constants/gpu";
import { LANGUAGE_DISPLAY_NAMES } from "~/constants/language";

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
  language: string | null;
};

export const getServerSideProps: GetServerSideProps = async (context) => {
  const slug = context.params?.slug as string;

  const helpers = createServerSideHelpers({
    router: appRouter,
    ctx: createInnerTRPCContext({ session: null }),
    transformer: superjson,
  });

  // Prefetch problem data
  await helpers.problems.getById.prefetch({ slug });

  // Prefetch all GPU types for instant switching
  await Promise.all(
    gpuTypes.map((gt: string) =>
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

const formatPerformance = (gflops: number | null | undefined): string => {
  if (!gflops) return "N/A";

  if (gflops >= 1000) {
    const tflops = (gflops / 1000).toFixed(2);
    return `${parseFloat(tflops)}T`;
  }
  return `${parseFloat(gflops.toFixed(2))}G`;
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
      ogTitle={`Leaderboard: ${problem?.title ? problem.title : "Leaderboard"}`}
      ogDescription={`Leaderboard for ${problem?.title ? problem.title : "Leaderboard"} on Tensara.`}
      ogImgSubtitle={`Leaderboards | Tensara`}
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
                  {"> "}
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
              borderColor="transparent"
              _hover={{ borderColor: "gray.600" }}
              _focus={{ borderColor: "gray.500" }}
              color="white"
            >
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
                <Thead borderBottom="1px solid" borderColor="whiteAlpha.100">
                  <Tr>
                    <Th borderBottom="none">Rank</Th>
                    <Th borderBottom="none">User</Th>
                    <Th borderBottom="none" isNumeric>
                      GFLOPS
                    </Th>
                    <Th borderBottom="none" isNumeric>
                      Runtime (ms)
                    </Th>
                    {selectedGpu === "all" && <Th borderBottom="none">GPU</Th>}
                    <Th borderBottom="none">Language</Th>
                    <Th borderBottom="none">Date</Th>
                  </Tr>
                </Thead>
                <Tbody>
                  {leaderboardEntries.map((entry, index) => {
                    const medalColor =
                      index <= 2
                        ? index === 0
                          ? "#FFD700"
                          : index === 1
                            ? "#C0C0C0"
                            : "#CD7F32"
                        : undefined;

                    return (
                      <Tr
                        key={entry.id}
                        onClick={() => router.push(`/submissions/${entry.id}`)}
                        cursor="pointer"
                        _hover={{ bg: "whiteAlpha.100" }}
                        borderBottom="1px solid"
                        borderColor="whiteAlpha.100"
                        my={medalColor ? 2 : 0}
                      >
                        <Td borderBottom="none">
                          <Text
                            color={medalColor}
                            fontWeight={medalColor ? "bold" : "normal"}
                          >
                            #{index + 1}
                          </Text>
                        </Td>
                        <Td borderBottom="none">
                          <Text
                            color={medalColor}
                            fontWeight={medalColor ? "bold" : "normal"}
                          >
                            {entry.username ?? "Anonymous"}
                          </Text>
                        </Td>
                        <Td isNumeric borderBottom="none">
                          <Text
                            color={medalColor}
                            fontWeight="bold"
                            fontFamily="mono"
                            fontSize="sm"
                          >
                            {formatPerformance(entry.gflops)}
                          </Text>
                        </Td>
                        <Td isNumeric borderBottom="none">
                          {entry.runtime?.toFixed(2) ?? "N/A"}
                        </Td>
                        {selectedGpu === "all" && (
                          <Td borderBottom="none">
                            <Badge
                              bg={"whiteAlpha.200"}
                              color={"white"}
                              px={2}
                              py={0.5}
                              borderRadius="md"
                              fontSize="xs"
                              fontWeight="medium"
                            >
                              {entry.gpuType}
                            </Badge>
                          </Td>
                        )}
                        <Td borderBottom="none">
                          <Text>
                            {LANGUAGE_DISPLAY_NAMES[entry.language ?? ""] ??
                              "Unknown"}
                          </Text>
                        </Td>
                        <Td borderBottom="none">
                          <Tooltip
                            label={new Date(entry.createdAt).toLocaleString()}
                          >
                            <Text>
                              {formatDistanceToNow(new Date(entry.createdAt))}{" "}
                              ago
                            </Text>
                          </Tooltip>
                        </Td>
                      </Tr>
                    );
                  })}
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
