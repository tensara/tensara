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
  HStack,
  Card,
  MenuButton,
  Button,
  MenuList,
  MenuItem,
  Menu,
  Icon,
  Tag,
} from "@chakra-ui/react";
import { useState, useEffect, useMemo } from "react";
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
import {
  FaChevronDown,
  FaExternalLinkAlt,
  FaEye,
  FaLock,
} from "react-icons/fa";
import { FiArrowLeft } from "react-icons/fi";
import { useSession } from "next-auth/react";

const BASELINE_DISPLAY_NAMES: Record<string, string> = {
  torch_compile: "PyTorch",
  torch_vanilla: "PyTorch",
  tinygrad: "Tinygrad",
};

const BASELINE_USERNAMES: Record<string, string> = {
  torch_compile: "Torch Compile",
  torch_vanilla: "Vanilla Torch",
  tinygrad: "Tinygrad",
};

type ProblemLeaderboardEntry = {
  id: string;
  username: string | null;
  gflops: number;
  runtime: number | null;
  createdAt: Date;
  gpuType: string | null;
  language: string | null;
  isPublic: boolean;
};

type BaselineResult = {
  name: string;
  gflops: number;
  test_id: number;
  runtime_ms: number;
};

type BaselineGPUData = {
  results: BaselineResult[];
  avg_gflops: number;
  total_tests: number;
  avg_runtime_ms: number;
};

type BaselineFrameworkData = Record<string, BaselineGPUData>;

type BaselineBenchmarks = Record<string, BaselineFrameworkData>;

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
  const { data: session } = useSession();
  const [selectedGpu, setSelectedGpu] = useState<string>(
    (router.query.gpu as string) || "all"
  );
  const [showBaselines, setShowBaselines] = useState(false);

  // Add this to get the current user data
  const currentUsername = session?.user?.username;

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

  // Get baseline benchmarks data
  const { data: baselineBenchmarks } =
    api.problems.getBaselineBenchmarks.useQuery(
      { slug },
      {
        enabled: !!slug,
      }
    );

  // Merge baseline benchmarks with user submissions
  const combinedEntries = useMemo(() => {
    if (!baselineBenchmarks || !leaderboardEntries) return leaderboardEntries;

    const baselineEntries = Object.entries(
      baselineBenchmarks as BaselineBenchmarks
    ).flatMap(([framework, gpuData]) => {
      if (selectedGpu === "all") {
        const bestGpuEntry = Object.entries(gpuData).reduce(
          (best, [gpu, data]) => {
            if (!best || data.avg_gflops > best.data.avg_gflops) {
              return { gpu, data };
            }
            return best;
          },
          null as { gpu: string; data: BaselineGPUData } | null
        );

        if (!bestGpuEntry) return [];

        return [
          {
            id: `baseline-${framework}-${bestGpuEntry.gpu}`,
            username: `${BASELINE_USERNAMES[framework]}`,
            gflops: bestGpuEntry.data.avg_gflops,
            runtime: bestGpuEntry.data.avg_runtime_ms,
            createdAt: new Date(0),
            gpuType: bestGpuEntry.gpu,
            language: "python",
            isPublic: true,
            isBaseline: true,
          },
        ];
      } else {
        // For specific GPU selection, keep existing behavior
        return gpuData[selectedGpu]
          ? [
              {
                id: `baseline-${framework}-${selectedGpu}`,
                username: `${BASELINE_USERNAMES[framework]}`,
                gflops: gpuData[selectedGpu].avg_gflops,
                runtime: gpuData[selectedGpu].avg_runtime_ms,
                createdAt: new Date(0),
                gpuType: selectedGpu,
                language: "python",
                isPublic: true,
                isBaseline: true,
              },
            ]
          : [];
      }
    });

    const entries = showBaselines
      ? [...leaderboardEntries, ...baselineEntries]
      : leaderboardEntries;
    return entries.sort((a, b) => b.gflops - a.gflops);
  }, [baselineBenchmarks, leaderboardEntries, selectedGpu, showBaselines]);

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
      ogTitle={`Leaderboard: ${problem?.title ?? "Leaderboard"}`}
      ogDescription={`Leaderboard for ${problem?.title ?? "Leaderboard"} on Tensara.`}
      ogImgSubtitle={`Leaderboards | Tensara`}
    >
      <Box maxW="7xl" mx="auto" px={4} py={8}>
        <Flex direction="column" gap={6}>
          <ChakraLink
            as={Link}
            href="/leaderboard"
            alignSelf="flex-start"
            mb={1}
            display="flex"
            alignItems="center"
            _hover={{
              textDecoration: "none",
            }}
          >
            <Button
              size="sm"
              variant="ghost"
              leftIcon={<Icon as={FiArrowLeft} />}
              borderRadius="lg"
              color="gray.300"
              _hover={{
                bg: "whiteAlpha.50",
                color: "white",
              }}
            >
              Back to Leaderboards
            </Button>
          </ChakraLink>

          <HStack justify="space-between" align="center">
            <Heading size="lg">
              {problem?.title ? (
                <HStack gap={4}>
                  <Text>{problem.title}</Text>
                  <ChakraLink href={`/problems/${problem.slug}`} isExternal>
                    <Icon
                      as={FaExternalLinkAlt}
                      color="gray.400"
                      boxSize={6}
                      _hover={{ color: "blue.400" }}
                      transition="color 0.5s"
                    />
                  </ChakraLink>
                </HStack>
              ) : (
                "Leaderboard"
              )}
            </Heading>
            <HStack spacing={4}>
              <Button
                size="sm"
                bg={showBaselines ? "whiteAlpha.200" : "whiteAlpha.50"}
                _hover={{ bg: "whiteAlpha.100" }}
                _active={{ bg: "whiteAlpha.150" }}
                color="white"
                fontWeight="normal"
                borderRadius="md"
                onClick={() => setShowBaselines(!showBaselines)}
              >
                {showBaselines ? "Hide Baselines" : "Show Baselines"}
              </Button>
              <Menu>
                <MenuButton
                  as={Button}
                  rightIcon={<FaChevronDown color="#d4d4d8" size={10} />}
                  bg="whiteAlpha.50"
                  _hover={{ bg: "whiteAlpha.100", borderColor: "gray.600" }}
                  _active={{ bg: "whiteAlpha.150" }}
                  _focus={{ borderColor: "blue.500", boxShadow: "none" }}
                  color="white"
                  w="200px"
                  fontWeight="normal"
                  textAlign="left"
                  justifyContent="flex-start"
                >
                  {GPU_DISPLAY_NAMES[selectedGpu]}
                </MenuButton>
                <MenuList bg="gray.800" borderColor="gray.800" p={0}>
                  {Object.entries(GPU_DISPLAY_NAMES).map(([key, value]) => (
                    <MenuItem
                      key={key}
                      onClick={() => setSelectedGpu(key)}
                      bg="gray.800"
                      _hover={{ bg: "gray.700" }}
                      color="white"
                      borderRadius="md"
                    >
                      {value}
                    </MenuItem>
                  ))}
                </MenuList>
              </Menu>
            </HStack>
          </HStack>

          {combinedEntries && combinedEntries.length > 0 ? (
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
                    <Th borderBottom="none">Code</Th>
                  </Tr>
                </Thead>
                <Tbody>
                  {combinedEntries.map((entry, index) => {
                    const medalColor =
                      index <= 2
                        ? index === 0
                          ? "#FFD700"
                          : index === 1
                            ? "#C0C0C0"
                            : "#CD7F32"
                        : undefined;

                    const isOwnSubmission =
                      currentUsername && entry.username === currentUsername;
                    const isBaseline = "isBaseline" in entry;

                    return (
                      <Tr
                        key={entry.id}
                        onClick={() =>
                          !isBaseline && router.push(`/submissions/${entry.id}`)
                        }
                        cursor={isBaseline ? "default" : "pointer"}
                        _hover={{
                          bg: isBaseline ? undefined : "whiteAlpha.100",
                        }}
                        borderBottom="1px solid"
                        borderColor="whiteAlpha.100"
                        my={medalColor ? 2 : 0}
                        bg={
                          medalColor
                            ? `rgba(${
                                medalColor
                                  .replace("#", "")
                                  .match(/../g)
                                  ?.map((hex) => parseInt(hex, 16))
                                  .join(",") ?? "0, 0, 0"
                              }, 0.08)`
                            : isBaseline
                              ? "whiteAlpha.200"
                              : undefined
                        }
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
                            {isBaseline && (
                              <Tag
                                size="sm"
                                variant="solid"
                                colorScheme="blue"
                                bg="blue.900"
                                ml={2}
                              >
                                Baseline
                              </Tag>
                            )}
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
                          {isBaseline ? (
                            <Text>
                              {BASELINE_DISPLAY_NAMES[
                                entry.id.split("-")[1] ?? ""
                              ] ?? "Unknown"}
                            </Text>
                          ) : (
                            <Text>
                              {LANGUAGE_DISPLAY_NAMES[entry.language ?? ""] ??
                                "Unknown"}
                            </Text>
                          )}
                        </Td>
                        <Td borderBottom="none">
                          {isBaseline ? (
                            <Text color="gray.400">-</Text>
                          ) : (
                            <Tooltip
                              label={new Date(entry.createdAt).toLocaleString()}
                            >
                              <Text>
                                {formatDistanceToNow(new Date(entry.createdAt))}{" "}
                                ago
                              </Text>
                            </Tooltip>
                          )}
                        </Td>
                        <Td borderBottom="none">
                          {isBaseline ? (
                            <Text color="gray.400">-</Text>
                          ) : entry.isPublic || isOwnSubmission ? (
                            <FaEye color="#4CAF50" />
                          ) : (
                            <FaLock color="#d4d4d8" />
                          )}
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
