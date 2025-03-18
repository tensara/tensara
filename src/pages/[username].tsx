import React from "react";
import { useRouter } from "next/router";
import { useEffect, useState } from "react";
import { useSession } from "next-auth/react";
import {
  Box,
  Container,
  Heading,
  Text,
  VStack,
  HStack,
  Image,
  Flex,
  Skeleton,
  Badge,
  Divider,
  Icon,
  Grid,
  GridItem,
  Stat,
  StatLabel,
  StatNumber,
  Table,
  Thead,
  Tbody,
  Tr,
  Th,
  Td,
  Link as ChakraLink,
  Button,
  SimpleGrid,
  Tooltip,
  Tag,
  TagLabel,
  TagLeftIcon,
  useToken,
  Alert,
  AlertIcon,
  AlertTitle,
  AlertDescription,
} from "@chakra-ui/react";
import { Layout } from "~/components/layout";
import {
  FiCalendar,
  FiAward,
  FiCode,
  FiList,
  FiClock,
  FiCheck,
  FiStar,
  FiHash,
  FiExternalLink,
  FiX,
} from "react-icons/fi";
import { FaTrophy } from "react-icons/fa";
import NextLink from "next/link";
import { api } from "~/utils/api";

// Define the type for activity data
interface ActivityItem {
  date: string;
  count: number;
}

// Activity calendar heatmap component
function ActivityCalendar({
  data,
  joinedYear,
}: {
  data: ActivityItem[];
  joinedYear: number;
}) {
  const weeks = 52; // Full year (52 weeks)
  const days = 7;

  // Group data by date
  const dateMap: Record<string, number> = {};
  data.forEach((item) => {
    dateMap[item.date] = item.count;
  });

  // Define grid cell type
  type GridCell = { date: string; count: number; bgColor: string };

  // Create calendar grid in a column-major format like GitHub
  const calendarGrid: GridCell[][] = Array(days)
    .fill(null)
    .map(() =>
      Array(weeks)
        .fill(null)
        .map(() => ({ date: "", count: 0, bgColor: "whiteAlpha.100" }))
    );

  const today = new Date();
  const dayNames = ["Mon", "Wed", "Fri", "Sun"]; // Show fewer day labels to save space

  // Get month labels
  const months: string[] = [];
  for (let i = 0; i < 12; i++) {
    const date = new Date();
    date.setDate(1); // First of month
    date.setMonth(today.getMonth() - 11 + i); // Start from 11 months ago
    months.push(date.toLocaleString("default", { month: "short" }));
  }

  // Fill the grid with data for the past year
  for (let w = 0; w < weeks; w++) {
    for (let d = 0; d < days; d++) {
      const date = new Date();
      // Calculate date: most recent contributions on the right side
      // Start from 1 year ago, move forward
      date.setDate(today.getDate() - ((weeks - w - 1) * 7 + (days - d - 1)));

      const dateStr = date.toISOString().split("T")[0]!;
      const count = dateMap[dateStr] ?? 0;

      // Determine color based on count
      let bgColor = "whiteAlpha.100";
      if (count > 0) {
        if (count === 1) bgColor = "blue.200";
        else if (count === 2) bgColor = "blue.300";
        else if (count === 3) bgColor = "blue.400";
        else if (count === 4) bgColor = "blue.500";
        else bgColor = "blue.600";
      }

      // Add to grid - column-major ordering
      if (calendarGrid[d]) {
        calendarGrid[d]![w] = {
          date: dateStr,
          count,
          bgColor,
        };
      }
    }
  }

  // Get total submissions in the date range
  const totalCount = Object.values(dateMap).reduce(
    (sum, count) => sum + count,
    0
  );

  // Calculate available years (from join year to current year)
  const currentYear = today.getFullYear();
  const availableYears = [];
  for (let year = currentYear; year >= joinedYear; year--) {
    availableYears.push(year);
  }

  return (
    <Box>
      <Text fontSize="sm" color="whiteAlpha.800" mb={3}>
        {totalCount} submissions in the last year
      </Text>

      <Flex position="relative">
        {/* Left side with day labels */}
        <Box>
          <Box h="20px"></Box> {/* Empty space to align with graph */}
          <VStack spacing={2} align="flex-start" mt={1}>
            {dayNames.map((day, index) => (
              <Text
                key={day}
                fontSize="xs"
                color="whiteAlpha.600"
                h="10px"
                mt={index > 0 ? "5px" : "0"}
              >
                {day}
              </Text>
            ))}
          </VStack>
        </Box>

        {/* Main calendar area */}
        <Box flex="1" ml={2}>
          {/* Month labels */}
          <Flex mb={1} width="100%">
            {months.map((month, i) => (
              <Text
                key={month + i}
                fontSize="xs"
                color="whiteAlpha.700"
                width={`${
                  85 / months.length
                }%`} /* Use 85% to leave space for year selector */
                textAlign="left"
              >
                {month}
              </Text>
            ))}
          </Flex>

          {/* Grid of contribution cells */}
          <Box>
            {calendarGrid.map((row, rowIndex) => (
              <HStack key={rowIndex} spacing={1} mb={1}>
                {row.map((day, colIndex) => (
                  <Tooltip
                    key={`${rowIndex}-${colIndex}`}
                    label={
                      day.date && day.count > 0
                        ? `${day.count} submission${
                            day.count === 1 ? "" : "s"
                          } on ${new Date(day.date).toLocaleDateString(
                            "en-US",
                            {
                              month: "long",
                              day: "numeric",
                            }
                          )}${getOrdinalSuffix(new Date(day.date).getDate())}.`
                        : day.date
                        ? `No submissions on ${new Date(
                            day.date
                          ).toLocaleDateString("en-US", {
                            month: "long",
                            day: "numeric",
                          })}${getOrdinalSuffix(new Date(day.date).getDate())}.`
                        : "No submissions"
                    }
                    placement="top"
                    hasArrow
                    bg="blue.200"
                    color="gray.800"
                    fontSize="xs"
                    px={2}
                    py={1}
                  >
                    <Box w="10px" h="10px" bg={day.bgColor} borderRadius="sm" />
                  </Tooltip>
                ))}
              </HStack>
            ))}

            {/* Less/More spectrum - aligned to grid */}
            <Flex justifyContent="flex-end" pr="120px" mt={4} width="100%">
              <Flex alignItems="center">
                <Text fontSize="xs" color="whiteAlpha.700" mr={2}>
                  Less
                </Text>
                <HStack spacing={1.5}>
                  <Box
                    w="10px"
                    h="10px"
                    borderRadius="sm"
                    bg="whiteAlpha.100"
                  />
                  <Box w="10px" h="10px" borderRadius="sm" bg="blue.200" />
                  <Box w="10px" h="10px" borderRadius="sm" bg="blue.300" />
                  <Box w="10px" h="10px" borderRadius="sm" bg="blue.400" />
                  <Box w="10px" h="10px" borderRadius="sm" bg="blue.500" />
                  <Box w="10px" h="10px" borderRadius="sm" bg="blue.600" />
                </HStack>
                <Text fontSize="xs" color="whiteAlpha.700" ml={2}>
                  More
                </Text>
              </Flex>
            </Flex>
          </Box>
        </Box>

        {/* Year selector (inside the panel) */}
        <VStack
          align="flex-start"
          spacing={2}
          mt="45px"
          position="absolute"
          right={0}
          top={0}
          minWidth="100px"
        >
          {availableYears.map((year, i) => (
            <Button
              key={year}
              variant="ghost"
              size="xs"
              fontSize="sm"
              p={2}
              height="auto"
              width="100%"
              color={i === 0 ? "blue.400" : "whiteAlpha.700"}
              fontWeight={i === 0 ? "bold" : "normal"}
              _hover={{ color: "blue.400", bg: "blue.900" }}
            >
              {year}
            </Button>
          ))}
        </VStack>
      </Flex>
    </Box>
  );
}

// Helper function to get ordinal suffix for dates (1st, 2nd, 3rd, etc.)
function getOrdinalSuffix(day: number): string {
  if (day > 3 && day < 21) return "th";
  switch (day % 10) {
    case 1:
      return "st";
    case 2:
      return "nd";
    case 3:
      return "rd";
    default:
      return "th";
  }
}

export default function UserProfile() {
  const router = useRouter();
  const { username } = router.query;
  const { data: session } = useSession();

  // Fetch user data with tRPC
  const {
    data: userData,
    isLoading,
    error: apiError,
  } = api.users.getByUsername.useQuery(
    { username: typeof username === "string" ? username : "" },
    {
      enabled: !!username && typeof username === "string",
      retry: false,
      refetchOnWindowFocus: false,
    }
  );

  // Extract join year from userData
  const joinedYear = userData?.joinedAt
    ? new Date(userData.joinedAt).getFullYear()
    : new Date().getFullYear();

  if (!username) {
    return null; // Still loading the username parameter
  }

  return (
    <Layout
      title={`${typeof username === "string" ? username : "User"}'s Profile`}
    >
      <Container maxW="container.xl" py={4}>
        {apiError ? (
          <Alert
            status="error"
            variant="solid"
            borderRadius="xl"
            flexDirection="column"
            alignItems="center"
            justifyContent="center"
            textAlign="center"
            py={6}
          >
            <AlertIcon boxSize="40px" mr={0} />
            <AlertTitle mt={4} mb={1} fontSize="lg">
              User Not Found
            </AlertTitle>
            <AlertDescription maxWidth="sm">
              The user {typeof username === "string" ? username : "User"}{" "}
              doesn&apos;t exist or has been removed.
            </AlertDescription>
            <Button
              mt={4}
              colorScheme="white"
              variant="outline"
              onClick={() => router.push("/")}
            >
              Return to Home
            </Button>
          </Alert>
        ) : (
          <Grid templateColumns={{ base: "1fr", md: "300px 1fr" }} gap={6}>
            {/* Profile Card */}
            <GridItem>
              <Box
                bg="gray.800"
                borderRadius="xl"
                overflow="hidden"
                boxShadow="xl"
                borderWidth="1px"
                borderColor="blue.900"
              >
                <Box p={6} textAlign="center">
                  <Skeleton isLoaded={!isLoading} borderRadius="full" mx="auto">
                    <Image
                      src={userData?.image ?? "https://via.placeholder.com/150"}
                      alt={`${
                        typeof username === "string" ? username : "User"
                      }'s profile`}
                      borderRadius="full"
                      boxSize={{ base: "150px", md: "180px" }}
                      border="4px solid"
                      borderColor="blue.500"
                      mx="auto"
                      boxShadow="lg"
                    />
                  </Skeleton>

                  <VStack mt={6} spacing={3}>
                    <Skeleton
                      isLoaded={!isLoading}
                      width={isLoading ? "200px" : "auto"}
                    >
                      <Heading color="white" size="lg">
                        {userData?.username ?? username}
                      </Heading>
                    </Skeleton>

                    <Skeleton
                      isLoaded={!isLoading}
                      width={isLoading ? "150px" : "auto"}
                    >
                      <HStack spacing={2}>
                        <Icon as={FiCalendar} color="blue.300" />
                        <Text color="whiteAlpha.800" fontSize="sm">
                          Joined{" "}
                          {userData
                            ? (() => {
                                const date = new Date(userData.joinedAt);
                                const day = date.getDate();
                                const month = date.toLocaleString("default", {
                                  month: "short",
                                });
                                const year = date.getFullYear();
                                return `${day}${getOrdinalSuffix(
                                  day
                                )} ${month} ${year}`;
                              })()
                            : "Loading..."}
                        </Text>
                      </HStack>
                    </Skeleton>

                    <Skeleton isLoaded={!isLoading}>
                      <Badge
                        colorScheme="blue"
                        px={3}
                        py={1.5}
                        fontSize="md"
                        borderRadius="full"
                        display="flex"
                        alignItems="center"
                        gap={2}
                        boxShadow="sm"
                      >
                        <Icon as={FiAward} />
                        {userData?.stats?.ranking
                          ? `Rank #${userData.stats.ranking}`
                          : "Unranked"}
                      </Badge>
                    </Skeleton>
                  </VStack>
                </Box>

                <Divider borderColor="blue.900" />

                {/* Stats Section */}
                <Box p={4}>
                  {/* <Heading
                    size="sm"
                    color="white"
                    mb={5}
                    textAlign="left"
                    textTransform="uppercase"
                  >
                    Stats
                  </Heading> */}

                  {/* Stats Boxes */}
                  <Box mb={4}>
                    <SimpleGrid columns={2} spacing={4} width="100%">
                      <Skeleton isLoaded={!isLoading}>
                        <Box
                          bg="blue.900"
                          borderRadius="lg"
                          py={3}
                          px={3}
                          textAlign="center"
                          height="100%"
                        >
                          <Text
                            fontSize="2xl"
                            color="blue.300"
                            fontWeight="bold"
                            mb={1}
                          >
                            {userData?.stats?.submissions ?? 0}
                          </Text>
                          <Flex
                            justifyContent="center"
                            alignItems="center"
                            color="whiteAlpha.700"
                            fontSize="sm"
                          >
                            <Icon as={FiList} mr={1} boxSize={3} />
                            <Text fontWeight={500}>Submissions</Text>
                          </Flex>
                        </Box>
                      </Skeleton>

                      <Skeleton isLoaded={!isLoading}>
                        <Box
                          bg="blue.900"
                          borderRadius="lg"
                          py={3}
                          px={3}
                          textAlign="center"
                          height="100%"
                        >
                          <Text
                            fontSize="2xl"
                            color="blue.300"
                            fontWeight="bold"
                            mb={1}
                          >
                            {userData?.stats?.solvedProblems ?? 0}
                          </Text>
                          <Flex
                            justifyContent="center"
                            alignItems="center"
                            color="whiteAlpha.700"
                            fontSize="sm"
                          >
                            <Icon as={FiCode} mr={1} boxSize={3} />
                            <Text fontWeight={500}>Problems</Text>
                          </Flex>
                        </Box>
                      </Skeleton>
                    </SimpleGrid>
                  </Box>

                  {/* Score */}
                  <Skeleton isLoaded={!isLoading}>
                    <Box
                      bg="blue.800"
                      borderRadius="lg"
                      py={4}
                      px={3}
                      textAlign="center"
                      boxShadow="md"
                    >
                      <Text
                        fontSize="3xl"
                        color="yellow.400"
                        fontWeight="bold"
                        mb={1}
                      >
                        {userData?.stats?.score
                          ? userData.stats.score.toFixed(2)
                          : 0}
                      </Text>
                      <Flex
                        justifyContent="center"
                        alignItems="center"
                        color="whiteAlpha.800"
                        fontSize="sm"
                      >
                        <Icon
                          as={FaTrophy}
                          color="yellow.400"
                          mr={1}
                          boxSize={3}
                        />
                        <Text fontWeight={500}>Tensara Score</Text>
                      </Flex>
                    </Box>
                  </Skeleton>
                </Box>
              </Box>
            </GridItem>

            {/* Right Column */}
            <GridItem display="flex" flexDirection="column" gap={6}>
              {/* Activity Graph */}
              <Box
                bg="gray.800"
                borderRadius="xl"
                overflow="hidden"
                boxShadow="xl"
                p={6}
                borderWidth="1px"
                borderColor="blue.900"
              >
                <Heading size="md" color="white" mb={4}>
                  Activity
                </Heading>

                <Skeleton
                  isLoaded={!isLoading}
                  height={isLoading ? "120px" : "auto"}
                >
                  {userData?.activityData &&
                    userData.activityData.length > 0 && (
                      <ActivityCalendar
                        data={userData.activityData as ActivityItem[]}
                        joinedYear={joinedYear}
                      />
                    )}
                </Skeleton>
              </Box>

              {/* Recent Submissions */}
              <Box
                bg="gray.800"
                borderRadius="xl"
                overflow="hidden"
                boxShadow="xl"
                borderWidth="1px"
                borderColor="blue.900"
                flex="1"
              >
                <Box
                  p={5}
                  bg="blue.900"
                  borderBottom="1px solid"
                  borderColor="blue.700"
                >
                  <Heading size="md" color="white">
                    Recent Submissions
                  </Heading>
                </Box>

                <Box p={4}>
                  {isLoading ? (
                    <VStack spacing={4} align="stretch">
                      {Array(3).map(
                        (_: undefined, i: number): JSX.Element => (
                          <Skeleton key={i} height="60px" borderRadius="md" />
                        )
                      )}
                    </VStack>
                  ) : userData?.recentSubmissions &&
                    userData.recentSubmissions.length > 0 ? (
                    <Table variant="simple" size="sm">
                      <Thead>
                        <Tr>
                          <Th color="whiteAlpha.600">Problem</Th>
                          <Th color="whiteAlpha.600">Date</Th>
                          <Th color="whiteAlpha.600">Status</Th>
                          <Th color="whiteAlpha.600">Runtime</Th>
                          <Th></Th>
                        </Tr>
                      </Thead>
                      <Tbody>
                        {userData.recentSubmissions.map((submission) => (
                          <Tr key={submission.id}>
                            <Td>
                              <NextLink
                                href={`/problems/${submission.problemId}`}
                                passHref
                              >
                                <ChakraLink
                                  color="blue.300"
                                  fontWeight="medium"
                                >
                                  {submission.problemName}
                                </ChakraLink>
                              </NextLink>
                            </Td>
                            <Td color="whiteAlpha.800">{submission.date}</Td>
                            <Td>
                              <Tag
                                size="sm"
                                colorScheme={
                                  submission.status === "accepted"
                                    ? "green"
                                    : "red"
                                }
                              >
                                <TagLeftIcon
                                  as={
                                    submission.status === "accepted"
                                      ? FiCheck
                                      : FiX
                                  }
                                />
                                <TagLabel>
                                  {submission.status === "accepted"
                                    ? "Accepted"
                                    : "Failed"}
                                </TagLabel>
                              </Tag>
                            </Td>
                            <Td color="whiteAlpha.800">{submission.runtime}</Td>
                            <Td>
                              <NextLink
                                href={`/submissions/${submission.id}`}
                                passHref
                              >
                                <Button
                                  as="a"
                                  size="xs"
                                  leftIcon={<FiExternalLink />}
                                  colorScheme="blue"
                                  variant="ghost"
                                >
                                  View
                                </Button>
                              </NextLink>
                            </Td>
                          </Tr>
                        ))}
                      </Tbody>
                    </Table>
                  ) : (
                    <Text color="whiteAlpha.700" textAlign="center" py={4}>
                      No submissions to display.
                    </Text>
                  )}
                </Box>

                {userData?.recentSubmissions &&
                  userData.recentSubmissions.length > 0 && (
                    <Box p={4} textAlign="center">
                      <NextLink
                        href={`/submissions?username=${
                          typeof username === "string" ? username : "User"
                        }`}
                        passHref
                      >
                        <Button
                          as="a"
                          colorScheme="blue"
                          size="sm"
                          variant="outline"
                        >
                          View All Submissions
                        </Button>
                      </NextLink>
                    </Box>
                  )}
              </Box>
            </GridItem>
          </Grid>
        )}
      </Container>
    </Layout>
  );
}
