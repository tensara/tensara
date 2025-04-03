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
  Link as ChakraLink,
  Tooltip,
  Spinner,
  Flex,
  Heading,
  Select,
  HStack,
  SimpleGrid,
  Card,
  CardHeader,
  CardBody,
  Avatar,
  Badge,
  Tabs,
  TabList,
  Tab,
  TabPanels,
  TabPanel,
  Fade,
} from "@chakra-ui/react";
import { useState } from "react";
import { api } from "~/utils/api";
import { Layout } from "~/components/layout";
import Link from "next/link";
import { useRouter } from "next/router";
import { createServerSideHelpers } from "@trpc/react-query/server";
import { appRouter } from "~/server/api/root";
import { createInnerTRPCContext } from "~/server/api/trpc";
import superjson from "superjson";
import type { GetServerSideProps } from "next";
import { GPU_DISPLAY_NAMES, gpuTypes } from "~/constants/gpu";
import { LANGUAGE_DISPLAY_NAMES } from "~/constants/language";
import { formatDistanceToNow } from "date-fns";

// Helper function to format performance numbers
const formatPerformance = (gflops: number | null | undefined): string => {
  if (!gflops) return "N/A";
  if (gflops >= 1000) {
    const tflops = (gflops / 1000).toFixed(2);
    return `${parseFloat(tflops)}T`;
  }
  return `${parseFloat(gflops.toFixed(2))}G`;
};

// Helper function to format rating numbers
const formatRating = (rating: number | null | undefined): string => {
  if (!rating) return "0";
  return rating.toFixed(0);
};

const getMedalColor = (index: number): string => {
  switch (index) {
    case 0:
      return "#FFD700"; // Gold
    case 1:
      return "#C0C0C0"; // Silver
    case 2:
      return "#CD7F32"; // Bronze
    default:
      return "white.800";
  }
};

export const getServerSideProps: GetServerSideProps = async (_context) => {
  const helpers = createServerSideHelpers({
    router: appRouter,
    ctx: createInnerTRPCContext({ session: null }),
    transformer: superjson,
  });

  // Prefetch the player rankings
  await helpers.users.getTopRankedPlayers.prefetch({ limit: 100 });

  // Prefetch ALL GPU types during server-side render
  await Promise.all(
    gpuTypes.map((gpuType: string) =>
      helpers.submissions.getBestSubmissionsByProblem.prefetch({ gpuType })
    )
  );

  return {
    props: {
      trpcState: helpers.dehydrate(),
    },
  };
};

const LeaderboardPage: NextPage = () => {
  const router = useRouter();
  const [selectedGpu, setSelectedGpu] = useState<string>("all");
  const [selectedTab, setSelectedTab] = useState<string>("users");

  // User rankings data
  const { data: rankedUsers, isLoading: isLoadingUsers } =
    api.users.getTopRankedPlayers.useQuery(
      { limit: 100 },
      {
        staleTime: 300000, // 5 minutes
        refetchOnMount: false,
        refetchOnWindowFocus: false,
      }
    );

  // Problem leaderboard data
  const { data: leaderboardData, isLoading: isLoadingProblems } =
    api.submissions.getBestSubmissionsByProblem.useQuery(
      { gpuType: selectedGpu },
      {
        placeholderData: (prev) => prev,
        refetchOnMount: false,
        refetchOnWindowFocus: false,
        staleTime: 300000,
      }
    );

  if (isLoadingUsers || isLoadingProblems) {
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

  return (
    <Layout title="Leaderboard">
      <Box maxW="7xl" mx="auto" px={4} py={8}>
        <Tabs variant="soft-rounded" colorScheme="blue" mb={6}>
          <Flex justifyContent="space-between" alignItems="center" mb={4}>
            <Heading size="lg">Leaderboard</Heading>
            {selectedTab === "problems" && (
              <Fade in={selectedTab === "problems"}>
                <Select
                  value={selectedGpu}
                  onChange={(e) => setSelectedGpu(e.target.value)}
                  w="200px"
                  bg="whiteAlpha.50"
                  color="white"
                  borderColor="whiteAlpha.200"
                  _hover={{ borderColor: "whiteAlpha.400" }}
                  _focus={{ borderColor: "blue.500" }}
                >
                  {Object.entries(GPU_DISPLAY_NAMES).map(([key, value]) => (
                    <option key={key} value={key}>
                      {value}
                    </option>
                  ))}
                </Select>
              </Fade>
            )}
            <HStack justify="flex-end" align="center">
              <TabList bg="whiteAlpha.100" p={1} borderRadius="full">
                <Tab
                  _selected={{ color: "white", bg: "blue.500" }}
                  onClick={() => setSelectedTab("users")}
                >
                  Users
                </Tab>
                <Tab
                  _selected={{ color: "white", bg: "blue.500" }}
                  onClick={() => setSelectedTab("problems")}
                >
                  Problems
                </Tab>
              </TabList>
            </HStack>
          </Flex>

          <TabPanels>
            {/* User Rankings Tab */}
            <TabPanel px={0}>
              {rankedUsers && rankedUsers.length > 0 ? (
                <Box overflowX="auto">
                  <Table variant="simple">
                    <Thead
                      borderBottom="1px solid"
                      borderColor="whiteAlpha.100"
                    >
                      <Tr>
                        <Th borderBottom="none">Rank</Th>
                        <Th borderBottom="none">User</Th>
                        <Th borderBottom="none" isNumeric>
                          Rating
                        </Th>
                        <Th borderBottom="none" isNumeric>
                          Submissions
                        </Th>
                        <Th borderBottom="none">Best Performance</Th>
                      </Tr>
                    </Thead>
                    <Tbody>
                      {rankedUsers.map((user, index) => {
                        const medalColor = getMedalColor(index);
                        return (
                          <Tr
                            key={user.id}
                            onClick={() =>
                              router.push(`/${user.username ?? "anonymous"}`)
                            }
                            cursor="pointer"
                            _hover={{ bg: "whiteAlpha.100" }}
                            borderBottom="1px solid"
                            borderColor="whiteAlpha.100"
                            my={medalColor ? 1 : 0}
                            bg={
                              medalColor
                                ? `rgba(${
                                    medalColor
                                      .replace("#", "")
                                      .match(/../g)
                                      ?.map((hex) => parseInt(hex, 16))
                                      .join(",") ?? "0, 0, 0"
                                  }, 0.08)`
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
                              <Flex align="center">
                                <Avatar
                                  size="sm"
                                  src={user.image ?? ""}
                                  name={user.username ?? "Anonymous"}
                                  mr={2}
                                />
                                <Text
                                  color={medalColor}
                                  fontWeight={medalColor ? "bold" : "normal"}
                                >
                                  {user.username ?? "Anonymous"}
                                </Text>
                              </Flex>
                            </Td>
                            <Td isNumeric borderBottom="none">
                              <Text
                                color={medalColor}
                                fontWeight="bold"
                                fontFamily="mono"
                                fontSize="sm"
                              >
                                {formatRating(user.rating)}
                              </Text>
                            </Td>
                            <Td isNumeric borderBottom="none">
                              {user.submissionsCount}
                            </Td>
                            <Td borderBottom="none">
                              {user.bestSubmission ? (
                                <Flex direction="column" gap={1}>
                                  <Text fontWeight="medium" fontSize="sm">
                                    <ChakraLink
                                      as={Link}
                                      href={`/problems/${user.bestSubmission.problem.slug}`}
                                      onClick={(e) => {
                                        e.stopPropagation();
                                      }}
                                      color="blue.300"
                                      _hover={{ textDecoration: "underline" }}
                                    >
                                      {user.bestSubmission.problem.title}
                                    </ChakraLink>
                                  </Text>
                                  <HStack spacing={2}>
                                    {user.bestSubmission.gpuType && (
                                      <Badge
                                        bg="whiteAlpha.200"
                                        color="white"
                                        px={2}
                                        py={0.5}
                                        borderRadius="md"
                                        fontSize="xs"
                                        fontWeight="medium"
                                      >
                                        {user.bestSubmission.gpuType}
                                      </Badge>
                                    )}
                                    <Badge
                                      bg="whiteAlpha.200"
                                      color="white"
                                      px={2}
                                      py={0.5}
                                      borderRadius="md"
                                      fontSize="xs"
                                      fontWeight="medium"
                                    >
                                      {user.bestSubmission.gflops?.toFixed(2) ??
                                        "0.00"}{" "}
                                      GFLOPS
                                    </Badge>
                                  </HStack>
                                </Flex>
                              ) : (
                                <Text fontSize="sm" color="whiteAlpha.600">
                                  No submissions
                                </Text>
                              )}
                            </Td>
                          </Tr>
                        );
                      })}
                    </Tbody>
                  </Table>
                </Box>
              ) : (
                <Box
                  p={6}
                  bg="gray.800"
                  borderRadius="md"
                  borderColor="whiteAlpha.200"
                  borderWidth={1}
                >
                  <Text>No users found.</Text>
                </Box>
              )}
            </TabPanel>

            {/* Problems Tab */}
            <TabPanel px={0}>
              <Flex direction="column" gap={6}>
                <SimpleGrid columns={{ base: 1, md: 2, lg: 3 }} spacing={6}>
                  {leaderboardData?.map((problem) => {
                    const topSubmissions = problem.topSubmissions;
                    return (
                      <Card
                        key={problem.slug}
                        bg="gray.800"
                        borderColor="whiteAlpha.200"
                        borderWidth={1}
                      >
                        <CardHeader>
                          <ChakraLink
                            as={Link}
                            href={`/leaderboard/${problem.slug}${
                              selectedGpu !== "all" ? `?gpu=${selectedGpu}` : ""
                            }`}
                          >
                            <Heading
                              size="md"
                              color="white"
                              _hover={{ color: "blue.400" }}
                            >
                              {problem.title}
                            </Heading>
                          </ChakraLink>
                        </CardHeader>
                        <CardBody>
                          {topSubmissions.length === 0 ? (
                            <Text color="whiteAlpha.700">
                              No submissions yet
                              {selectedGpu !== "all"
                                ? ` for ${GPU_DISPLAY_NAMES[selectedGpu]}`
                                : ""}
                            </Text>
                          ) : (
                            <Table variant="unstyled" size="sm">
                              <Thead>
                                <Tr>
                                  <Th pl={2} color="whiteAlpha.600">
                                    Rank
                                  </Th>
                                  <Th color="whiteAlpha.600">User</Th>
                                  <Th isNumeric color="whiteAlpha.600">
                                    FLOPS
                                  </Th>
                                </Tr>
                              </Thead>
                              <Tbody>
                                {topSubmissions.map((submission, index) => (
                                  <Tr
                                    key={submission.id}
                                    onClick={(e) => {
                                      e.preventDefault();
                                      void router.push(
                                        `/submissions/${submission.id}`
                                      );
                                    }}
                                    cursor="pointer"
                                    _hover={{
                                      bg: "whiteAlpha.50",
                                    }}
                                    px={4}
                                  >
                                    <Td pl={2}>
                                      <Text color="whiteAlpha.600">
                                        #{index + 1}
                                      </Text>
                                    </Td>
                                    <Td color="white">
                                      <Tooltip
                                        label={`${
                                          LANGUAGE_DISPLAY_NAMES[
                                            submission.language ?? ""
                                          ] ?? "Unknown"
                                        } | ${
                                          GPU_DISPLAY_NAMES[
                                            submission.gpuType ?? ""
                                          ] ?? "Unknown GPU"
                                        }`}
                                      >
                                        <ChakraLink
                                          as={Link}
                                          href={`/${
                                            submission.username ?? "anonymous"
                                          }`}
                                          onClick={(e) => {
                                            e.stopPropagation();
                                            e.preventDefault();
                                            void router.push(
                                              `/${submission.username ?? "anonymous"}`
                                            );
                                          }}
                                          _hover={{ color: "blue.400" }}
                                        >
                                          {submission.username ?? "Anonymous"}
                                        </ChakraLink>
                                      </Tooltip>
                                    </Td>
                                    <Td isNumeric>
                                      <Tooltip
                                        label={`Runtime: ${submission.runtime?.toFixed(
                                          2
                                        )} ms`}
                                      >
                                        <Text
                                          color={getMedalColor(index)}
                                          fontWeight="bold"
                                          fontFamily="mono"
                                          fontSize="sm"
                                        >
                                          {formatPerformance(submission.gflops)}
                                        </Text>
                                      </Tooltip>
                                    </Td>
                                  </Tr>
                                ))}
                              </Tbody>
                            </Table>
                          )}
                        </CardBody>
                      </Card>
                    );
                  })}
                </SimpleGrid>
              </Flex>
            </TabPanel>
          </TabPanels>
        </Tabs>
      </Box>
    </Layout>
  );
};

export default LeaderboardPage;
