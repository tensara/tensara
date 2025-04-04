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
  Icon,
  useBreakpointValue,
  ListItem,
  List,
  Popover,
  PopoverTrigger,
  PopoverContent,
  PopoverBody,
  Divider,
  IconButton,
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
import { motion, AnimatePresence } from "framer-motion";
import { FaExclamationCircle, FaFilter } from "react-icons/fa";

type UserRanking = {
  id: string;
  username: string;
  name: string;
  image: string;
  rating: number;
  rank: number;
  submissionsCount: number;
  solvedProblemsCount: number;
  bestSubmission: {
    id: string;
    gflops: number | null;
    gpuType: string | null;
    problem: {
      title: string;
      slug: string;
    };
  };
};

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
  const isMobile = useBreakpointValue({ base: true, md: false });

  // User rankings data
  const { data: rankedUsers, isLoading: isLoadingUsers } =
    api.users.getTopRankedPlayers.useQuery<UserRanking[]>(
      { limit: 100 },
      {
        staleTime: 300000, // 5 minutes
        refetchOnMount: false,
        refetchOnWindowFocus: false,
      }
    );

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
    <Layout
      title={`Leaderboards: ${GPU_DISPLAY_NAMES[selectedGpu]}`}
      ogTitle={`Leaderboards | Tensara`}
      ogDescription={`Leaderboards for ${GPU_DISPLAY_NAMES[selectedGpu]} on Tensara.`}
    >
      <Box maxW="7xl" mx="auto" px={4} py={8}>
        <Tabs
          variant="soft-rounded"
          colorScheme="blue"
          mb={6}
          isLazy
          onChange={(index) => {
            setSelectedTab(index === 0 ? "users" : "problems");
          }}
        >
          <Flex
            justifyContent="space-between"
            alignItems="center"
            mb={4}
            direction={{ base: "column", sm: "row" }}
            gap={{ base: 4, sm: 0 }}
          >
            <Heading size="lg">Leaderboard</Heading>

            <Flex align="center" gap={4}>
              <AnimatePresence>
                {selectedTab === "problems" && !isMobile && (
                  <motion.div
                    initial={{ opacity: 0, x: 20 }}
                    animate={{ opacity: 1, x: 0 }}
                    exit={{ opacity: 0, x: 20 }}
                    transition={{ duration: 0.2 }}
                  >
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
                  </motion.div>
                )}
                {selectedTab === "problems" && isMobile && (
                  <motion.div
                    initial={{ opacity: 0, scale: 0.9 }}
                    animate={{ opacity: 1, scale: 1 }}
                    exit={{ opacity: 0, scale: 0.9 }}
                    transition={{ duration: 0.2 }}
                  >
                    <Popover placement="bottom">
                      <PopoverTrigger>
                        <IconButton
                          aria-label="Filter by GPU"
                          icon={<FaFilter />}
                          colorScheme="blue"
                          variant="outline"
                          size="sm"
                        />
                      </PopoverTrigger>
                      <PopoverContent
                        bg="gray.800"
                        borderColor="whiteAlpha.300"
                        w="150px"
                      >
                        <PopoverBody p={0}>
                          <List spacing={0}>
                            {Object.entries(GPU_DISPLAY_NAMES).map(
                              ([key, value], index, arr) => (
                                <>
                                  <ListItem
                                    key={key}
                                    px={3}
                                    fontSize="sm"
                                    py={2}
                                    onClick={() => setSelectedGpu(key)}
                                    cursor="pointer"
                                    bg={
                                      selectedGpu === key
                                        ? "blue.900"
                                        : undefined
                                    }
                                    _hover={{
                                      bg:
                                        selectedGpu === key
                                          ? "blue.800"
                                          : "whiteAlpha.100",
                                    }}
                                    fontWeight={
                                      selectedGpu === key ? "bold" : "normal"
                                    }
                                  >
                                    {value}
                                  </ListItem>
                                  {index < arr.length - 1 && (
                                    <Divider borderColor="whiteAlpha.200" />
                                  )}
                                </>
                              )
                            )}
                          </List>
                        </PopoverBody>
                      </PopoverContent>
                    </Popover>
                  </motion.div>
                )}
              </AnimatePresence>
              {/* Tab Selector */}
              <TabList bg="whiteAlpha.100" p={1} borderRadius="full">
                <Tab
                  _selected={{ color: "white", bg: "blue.800" }}
                  onClick={() => setSelectedTab("users")}
                >
                  Users
                </Tab>
                <Tab
                  _selected={{ color: "white", bg: "blue.800" }}
                  onClick={() => setSelectedTab("problems")}
                >
                  Problems
                </Tab>
              </TabList>
            </Flex>
          </Flex>

          <TabPanels>
            {/* User Rankings Tab */}
            <TabPanel px={0}>
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                transition={{ duration: 0.3 }}
              >
                {rankedUsers && rankedUsers.length > 0 ? (
                  <Box overflowX="auto" borderRadius="md" boxShadow="md">
                    {isMobile ? (
                      /* Mobile Users Leaderboard */
                      <Box>
                        {rankedUsers.map((user, index) => {
                          const medalColor = getMedalColor(index);
                          return (
                            <Flex
                              key={user.id}
                              onClick={() =>
                                router.push(`/${user.username ?? "anonymous"}`)
                              }
                              cursor="pointer"
                              direction="row"
                              align="center"
                              justify="space-between"
                              p={4}
                              borderBottom="1px solid"
                              borderColor="whiteAlpha.100"
                              transition="background 0.2s"
                              _hover={{ bg: "whiteAlpha.100" }}
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
                              <Flex align="center" flex={1}>
                                <Text
                                  color={medalColor}
                                  fontWeight={medalColor ? "bold" : "normal"}
                                  fontSize="sm"
                                  minWidth="40px"
                                >
                                  #{index + 1}
                                </Text>
                                <Avatar
                                  size="sm"
                                  src={user.image ?? ""}
                                  name={user.username ?? "Anonymous"}
                                  mr={2}
                                />
                                <Text
                                  color={medalColor}
                                  fontWeight={medalColor ? "bold" : "normal"}
                                  fontSize="sm"
                                  noOfLines={1}
                                >
                                  {user.username ?? "Anonymous"}
                                </Text>
                              </Flex>

                              <Flex align="center" justifyContent="flex-end">
                                <Badge
                                  bg="blue.900"
                                  color="whiteAlpha.900"
                                  px={2}
                                  py={1}
                                  borderRadius="md"
                                  fontSize="sm"
                                  fontWeight="bold"
                                  fontFamily="mono"
                                >
                                  {formatRating(user.rating)}
                                </Badge>
                              </Flex>
                            </Flex>
                          );
                        })}
                      </Box>
                    ) : (
                      /* Desktop Users Leaderboard */
                      <Table variant="simple" size="md" layout="fixed">
                        <Thead bg="gray.800" borderTopRadius="md">
                          <Tr>
                            <Th
                              borderBottom="none"
                              color="whiteAlpha.700"
                              width="10%"
                            >
                              Rank
                            </Th>
                            <Th
                              borderBottom="none"
                              color="whiteAlpha.700"
                              width="30%"
                            >
                              User
                            </Th>
                            <Th
                              borderBottom="none"
                              isNumeric
                              color="whiteAlpha.700"
                              width="15%"
                            >
                              Rating
                            </Th>
                            <Th
                              borderBottom="none"
                              isNumeric
                              color="whiteAlpha.700"
                              width="15%"
                            >
                              Submissions
                            </Th>
                            <Th
                              borderBottom="none"
                              color="whiteAlpha.700"
                              width="35%"
                              textAlign="center"
                            >
                              Best Performance
                            </Th>
                          </Tr>
                        </Thead>
                        <Tbody>
                          {rankedUsers.map((user, index) => {
                            const medalColor = getMedalColor(index);
                            return (
                              <Tr
                                key={user.id}
                                onClick={() =>
                                  router.push(
                                    `/${user.username ?? "anonymous"}`
                                  )
                                }
                                cursor="pointer"
                                _hover={{ bg: "whiteAlpha.100" }}
                                borderBottom="1px solid"
                                borderColor="whiteAlpha.100"
                                transition="background 0.2s"
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
                                      fontWeight={
                                        medalColor ? "bold" : "normal"
                                      }
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
                                <Td
                                  borderBottom="none"
                                  py={2.5}
                                  justifyItems="center"
                                >
                                  {user.bestSubmission ? (
                                    <Flex
                                      direction="column"
                                      gap={2}
                                      align="center"
                                      maxWidth="100%"
                                    >
                                      <Text
                                        fontWeight="medium"
                                        fontSize="sm"
                                        noOfLines={1}
                                        width="100%"
                                        align="center"
                                      >
                                        <ChakraLink
                                          as={Link}
                                          href={`/problems/${user.bestSubmission.problem.slug}`}
                                          onClick={(e) => {
                                            e.stopPropagation();
                                          }}
                                          color="gray.200"
                                          _hover={{ color: "blue.200" }}
                                        >
                                          {user.bestSubmission.problem.title}
                                        </ChakraLink>
                                      </Text>
                                      <Flex gap={2} wrap="nowrap">
                                        {user.bestSubmission.gpuType && (
                                          <Badge
                                            bg="blackAlpha.400"
                                            color="whiteAlpha.900"
                                            px={2}
                                            py={0.5}
                                            borderRadius="md"
                                            fontSize="xs"
                                            fontWeight="medium"
                                            minWidth="fit-content"
                                          >
                                            {user.bestSubmission.gpuType}
                                          </Badge>
                                        )}
                                        <Badge
                                          bg="blackAlpha.400"
                                          color="white.300"
                                          px={2}
                                          py={0.5}
                                          borderRadius="md"
                                          fontSize="xs"
                                          fontWeight="medium"
                                          minWidth="fit-content"
                                        >
                                          {user.bestSubmission.gflops !== null
                                            ? user.bestSubmission.gflops >= 1000
                                              ? `${(user.bestSubmission.gflops / 1000).toFixed(2)} T`
                                              : `${user.bestSubmission.gflops.toFixed(2)} G`
                                            : "0.00 G"}
                                          {"FLOPS"}
                                        </Badge>
                                      </Flex>
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
                    )}
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
              </motion.div>
            </TabPanel>

            {/* Problems Tab */}
            <TabPanel px={0}>
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                transition={{ duration: 0.3 }}
              >
                <Flex direction="column" gap={6}>
                  <SimpleGrid
                    columns={{ base: 1, md: 2, lg: 3 }}
                    spacing={6}
                    minChildWidth="300px"
                  >
                    {leaderboardData?.map((problem) => {
                      const topSubmissions = problem.topSubmissions;
                      return (
                        <Card
                          key={problem.slug}
                          bg="gray.800"
                          borderColor="whiteAlpha.200"
                          borderWidth={1}
                          transition="transform 0.2s, box-shadow 0.2s"
                          _hover={{
                            transform: "translateY(-2px)",
                            boxShadow: "lg",
                          }}
                        >
                          <CardHeader pb={2}>
                            <ChakraLink
                              as={Link}
                              href={`/leaderboard/${problem.slug}${
                                selectedGpu !== "all"
                                  ? `?gpu=${selectedGpu}`
                                  : ""
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
                          <CardBody pt={2}>
                            {topSubmissions.length === 0 ? (
                              <Flex
                                direction="column"
                                align="center"
                                justify="center"
                                p={4}
                                minH="120px"
                                bg="whiteAlpha.50"
                                borderRadius="md"
                              >
                                <Icon
                                  as={FaExclamationCircle}
                                  color="whiteAlpha.600"
                                  mb={2}
                                />
                                <Text color="whiteAlpha.700" textAlign="center">
                                  No submissions yet
                                  {selectedGpu !== "all"
                                    ? ` for ${GPU_DISPLAY_NAMES[selectedGpu]}`
                                    : ""}
                                </Text>
                              </Flex>
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
                                      borderRadius="md"
                                      transition="background 0.15s"
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
                                          hasArrow
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
                                          hasArrow
                                        >
                                          <Text
                                            color={getMedalColor(index)}
                                            fontWeight="bold"
                                            fontFamily="mono"
                                            fontSize="sm"
                                          >
                                            {formatPerformance(
                                              submission.gflops
                                            )}
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
              </motion.div>
            </TabPanel>
          </TabPanels>
        </Tabs>
      </Box>
    </Layout>
  );
};

export default LeaderboardPage;
