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
  FiGithub,
  FiBarChart2,
  FiCpu,
  FiUser,
  FiZap,
  FiTrendingUp,
  FiArrowRight,
} from "react-icons/fi";
import { FaTrophy, FaFire, FaMedal } from "react-icons/fa";
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
      <HStack justify="space-between" mb={4}>
        <HStack spacing={3}>
          <Icon as={FaFire} color="blue.300" w={5} h={5} />
          <Text fontSize="sm" color="whiteAlpha.800">
            <Text as="span" fontWeight="bold" fontSize="md" color="white">
              {totalCount}
            </Text>{" "}
            submissions in the last year
          </Text>
        </HStack>
        
        {/* Year Tabs */}
        <HStack spacing={1} bg="gray.700" borderRadius="md" p={1}>
          {availableYears.slice(0, 3).map((year, i) => (
            <Button
              key={year}
              size="xs"
              fontSize="xs"
              colorScheme={i === 0 ? "blue" : "gray"}
              variant={i === 0 ? "solid" : "ghost"}
              height="24px"
            >
              {year}
            </Button>
          ))}
          {availableYears.length > 3 && (
            <Button
              size="xs"
              fontSize="xs"
              variant="ghost"
              colorScheme="gray"
              height="24px"
            >
              More
            </Button>
          )}
        </HStack>
      </HStack>

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
                width={`${100 / months.length}%`}
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

            {/* Less/More spectrum - styled nicely */}
            <Flex justify="flex-end" mt={4} width="100%">
              <Flex alignItems="center" bg="gray.700" py={1} px={3} borderRadius="md">
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
    
  // Mock streak data (in real app would come from API)
  const currentStreak = 5;
  const maxStreak = 18;

  if (!username) {
    return null; // Still loading the username parameter
  }

return (
  <Layout
    title={`${typeof username === "string" ? username : "User"}'s Profile`}
  >
    <Container maxW="container.xl" py={6}>
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
        <Grid templateColumns={{ base: "1fr", md: "320px 1fr" }} gap={8}>
          {/* Left Column */}
          <GridItem>
            <VStack spacing={5} align="stretch" height="100%">
              {/* User Profile Card */}
              <Box
                bg="gray.800"
                borderRadius="xl"
                overflow="hidden"
                boxShadow="xl"
                borderWidth="1px"
                borderColor="blue.900"
                w="100%"
                position="relative"
              >
                <Box p={6} textAlign="center">
                  <Skeleton isLoaded={!isLoading} borderRadius="full" mx="auto">
                    <Image
                      src={userData?.image ?? "https://via.placeholder.com/150"}
                      alt={`${typeof username === "string" ? username : "User"}'s profile`}
                      borderRadius="full"
                      boxSize="100px"
                      border="4px solid"
                      borderColor="blue.500"
                      mx="auto"
                      boxShadow="lg"
                      bg="gray.700"
                    />
                  </Skeleton>

                  <VStack mt={4} spacing={2}>
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
                                return `${day}${getOrdinalSuffix(day)} ${month} ${year}`;
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
                        mt={2}
                      >
                        <Icon as={FiAward} />
                        {userData?.stats?.ranking
                          ? `Rank #${userData.stats.ranking}`
                          : "Unranked"}
                      </Badge>
                    </Skeleton>
                  </VStack>
                </Box>
              </Box>

              {/* Stats Cards */}
              <SimpleGrid columns={2} spacing={4} width="100%">
                <Skeleton isLoaded={!isLoading}>
                  <Box
                    bg="gray.800"
                    borderRadius="xl"
                    py={4}
                    px={4}
                    textAlign="center"
                    height="100%"
                    borderWidth="1px"
                    borderColor="blue.900"
                    boxShadow="md"
                    position="relative"
                    overflow="hidden"
                  >
                    {/* Background subtle pattern */}
                    <Box
                      position="absolute"
                      top={0}
                      left={0}
                      right={0}
                      bottom={0}
                      opacity={0.05}
                      bgGradient="radial(blue.400, transparent 70%)"
                    />
                    
                    <Flex 
                      direction="column" 
                      justify="center" 
                      align="center" 
                      position="relative"
                      height="100%"
                    >
                      <Icon as={FiList} color="blue.300" boxSize={5} mb={1} />
                      <Text
                        fontSize="2xl"
                        color="white"
                        fontWeight="bold"
                      >
                        {userData?.stats?.submissions ?? 0}
                      </Text>
                      <Text
                        color="whiteAlpha.800"
                        fontSize="sm"
                        fontWeight={500}
                      >
                        Submissions
                      </Text>
                    </Flex>
                  </Box>
                </Skeleton>

                <Skeleton isLoaded={!isLoading}>
                  <Box
                    bg="gray.800"
                    borderRadius="xl"
                    py={4}
                    px={4}
                    textAlign="center"
                    height="100%"
                    borderWidth="1px"
                    borderColor="blue.900"
                    boxShadow="md"
                    position="relative"
                    overflow="hidden"
                  >
                    {/* Background subtle pattern */}
                    <Box
                      position="absolute"
                      top={0}
                      left={0}
                      right={0}
                      bottom={0}
                      opacity={0.05}
                      bgGradient="radial(blue.400, transparent 70%)"
                    />
                    
                    <Flex 
                      direction="column" 
                      justify="center" 
                      align="center" 
                      position="relative"
                      height="100%"
                    >
                      <Icon as={FiCode} color="blue.300" boxSize={5} mb={1} />
                      <Text
                        fontSize="2xl"
                        color="white"
                        fontWeight="bold"
                      >
                        {userData?.stats?.solvedProblems ?? 0}
                      </Text>
                      <Text
                        color="whiteAlpha.800"
                        fontSize="sm"
                        fontWeight={500}
                      >
                        Problems
                      </Text>
                    </Flex>
                  </Box>
                </Skeleton>
              </SimpleGrid>

              {/* Score Card */}
              <Skeleton isLoaded={!isLoading} width="100%">
                <Box
                  bg="gray.800"
                  borderRadius="xl"
                  py={5}
                  px={5}
                  position="relative"
                  overflow="hidden"
                  borderWidth="1px"
                  borderColor="blue.900"
                  boxShadow="xl"
                >
                  <Flex justify="center" mb={4}>
                    <Icon 
                      as={FaTrophy} 
                      color="yellow.400" 
                      boxSize={6} 
                      mr={3}
                    />
                    <Heading size="md" color="white">
                      Tensara Score
                    </Heading>
                  </Flex>
                  
                  <Text
                    fontSize="5xl"
                    color="yellow.400"
                    fontWeight="bold"
                    textShadow="0 0 10px rgba(236, 201, 75, 0.3)"
                    textAlign="center"
                  >
                    {userData?.stats?.score
                      ? userData.stats.score.toFixed(2)
                      : 0}
                  </Text>
                </Box>
              </Skeleton>
              
              {/* Languages Card */}
              <Skeleton isLoaded={!isLoading} width="100%">
                <Box
                  bg="gray.800"
                  borderRadius="xl"
                  p={5}
                  borderWidth="1px"
                  borderColor="blue.900"
                  boxShadow="md"
                >
                  <Flex justify="space-between" align="center" mb={4}>
                    <Heading size="sm" color="white">
                      Frameworks Used
                    </Heading>
                    <Icon as={FiBarChart2} color="blue.200" boxSize={5} />
                  </Flex>

                  <HStack spacing={2} flexWrap="wrap">
                    <Tag size="md" borderRadius="full" mb={2}>
                      <Box w="10px" h="10px" bg="purple.700" borderRadius="full" mr={2} />
                      <TagLabel>Triton (48%)</TagLabel>
                    </Tag>
                    <Tag size="md" borderRadius="full" mb={2}>
                      <Box w="10px" h="10px" bg="green.700" borderRadius="full" mr={2} />
                      <TagLabel>Cuda (52%)</TagLabel>
                    </Tag>
                  </HStack>
                </Box>
              </Skeleton>
            </VStack>
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
              position="relative"
            >
              <Flex justify="space-between" align="center" mb={6}>
                <HStack>
                  <Icon as={FiTrendingUp} color="blue.300" boxSize={5} />
                  <Heading size="md" color="white">
                    Activity
                  </Heading>
                </HStack>
                
                <Button size="sm" colorScheme="blue" variant="ghost" leftIcon={<FiBarChart2 />}>
                  View Stats
                </Button>
              </Flex>

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
              bg="gray.900"
              borderRadius="xl"
              overflow="hidden"
              boxShadow="lg"
              borderWidth="1px"
              borderColor="blue.900"
            >
              {/* Header */}
              <Flex
                px={5}
                py={4}
                bg="gray.800"
                borderBottom="1px solid"
                borderColor="blue.800"
                align="center"
                justify="space-between"
              >
                <HStack spacing={3}>
                  <Icon as={FiList} color="blue.400" boxSize={5} />
                  <Heading size="md" color="white" fontWeight="semibold">
                    Recent Submissions
                  </Heading>
                </HStack>
                
                {userData?.recentSubmissions && userData.recentSubmissions.length > 0 && (
                  <NextLink
                    href={`/submissions?username=${typeof username === "string" ? username : "User"}`}
                    passHref
                  >
                    <Button
                      as="a"
                      size="sm"
                      variant="ghost"
                      colorScheme="blue"
                      rightIcon={<Icon as={FiArrowRight} />}
                    >
                      View All
                    </Button>
                  </NextLink>
                )}
              </Flex>

              <Box bg="gray.850" px={0} py={0}>
                {isLoading ? (
                  <VStack spacing={1} align="stretch" p={2}>
                    {Array(3).fill(undefined).map((_: undefined, i: number): JSX.Element => (
                      <Skeleton key={i} height="80px" startColor="gray.700" endColor="gray.800" borderRadius="md" />
                    ))}
                  </VStack>
                ) : userData?.recentSubmissions && userData.recentSubmissions.length > 0 ? (
                  <VStack spacing={0} align="stretch" divider={<Divider borderColor="gray.700" />}>
                    {userData.recentSubmissions.slice(0, 3).map((submission, _) => (
                      <NextLink
                        key={submission.id}
                        href={`/submissions/${submission.id}`}
                        passHref
                      >
                        <Box 
                          as="a"
                          py={3}
                          px={5}
                          position="relative"
                          bg="gray.800"
                          _hover={{ 
                            bg: "gray.700",
                            cursor: "pointer" 
                          }}
                          transition="all 0.2s ease"
                          display="block"
                          borderLeftWidth="3px"
                          borderLeftColor={submission.status === "accepted" ? "green.400" : "red.400"}
                        >
                          <Grid templateColumns="3fr 2fr" gap={4} alignItems="center">
                            {/* Left side: Problem information */}
                            <Box>
                              <Flex align="center" mb={1.5}>
                                <Text
                                  color="white"
                                  fontWeight="medium"
                                  mr={2}
                                >
                                  {submission.problemName}
                                </Text>
                                
                                <Tag
                                  size="sm"
                                  borderRadius="full"
                                  colorScheme={submission.status === "accepted" ? "green" : "red"}
                                  py={0.5}
                                >
                                  {submission.status === "accepted" ? "Accepted" : "Failed"}
                                </Tag>
                              </Flex>
                              
                              <HStack spacing={4}>
                                <HStack spacing={1.5}>
                                  <Icon as={FiCalendar} color="blue.300" boxSize="14px" />
                                  <Text color="whiteAlpha.700" fontSize="sm">{submission.date}</Text>
                                </HStack>
                              </HStack>
                            </Box>
                            
                            {/* Right side: Performance metrics */}
                            <Flex justify="flex-end">
                              <HStack spacing={3} bg="gray.800" borderRadius="lg" p={2} borderWidth="1px" borderColor="gray.700">
                                {/* GPU Type */}
                                <Box 
                                  px={3} 
                                  py={1.5} 
                                  borderRadius="md" 
                                  bg="blue.900"
                                  minW="70px"
                                  textAlign="center"
                                >
                                  <Text color="white" fontSize="sm" fontWeight="semibold">
                                    {submission.gpuType || "N/A"}
                                  </Text>
                                  <Text color="whiteAlpha.700" fontSize="xs" mt={0.5}>
                                    GPU
                                  </Text>
                                </Box>
                                
                                {/* GLOPS info */}
                                <Box 
                                  px={3} 
                                  py={1.5} 
                                  borderRadius="md" 
                                  bg="gray.700"
                                  minW="70px"
                                  textAlign="center"
                                >
                                  <Text color="blue.300" fontSize="sm" fontWeight="semibold">
                                    {(submission.gflops || "").split(" ")[0] || "N/A"}
                                  </Text>
                                  <Text color="whiteAlpha.700" fontSize="xs" mt={0.5}>
                                    GLOPS
                                  </Text>
                                </Box>
                                
                                {/* Runtime info*/}
                                <Box 
                                  px={3} 
                                  py={1.5} 
                                  borderRadius="md" 
                                  bg="gray.700"
                                  minW="70px"
                                  textAlign="center"
                                >
                                  <Text color="white" fontSize="sm" fontWeight="semibold">
                                    {submission.runtime || "N/A"}
                                  </Text>
                                  <Text color="whiteAlpha.700" fontSize="xs" mt={0.5}>
                                    Runtime
                                  </Text>
                                </Box>
                              </HStack>
                            </Flex>
                          </Grid>
                        </Box>
                      </NextLink>
                    ))}
                  </VStack>
                ) : (
                  <Flex 
                    direction="column" 
                    align="center" 
                    justify="center" 
                    py={10}
                    px={5}
                  >
                    <Box
                      p={4}
                      borderRadius="full"
                      bg="gray.800"
                      mb={3}
                    >
                      <Icon as={FiList} color="blue.400" boxSize={6} />
                    </Box>
                    <Text color="whiteAlpha.800" fontSize="md" fontWeight="medium" mb={1}>
                      No submissions yet
                    </Text>
                    <Text color="whiteAlpha.600" fontSize="sm" textAlign="center" maxW="xs">
                      Your recent submission history will appear here
                    </Text>
                  </Flex>
                )}
              </Box>
            </Box>
          </GridItem>
        </Grid>
      )}
    </Container>
  </Layout>
  );
}
