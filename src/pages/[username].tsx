import React from "react";
import { useRouter } from "next/router";
import { useState } from "react";
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
  Button,
  SimpleGrid,
  Tooltip,
  Tag,
  TagLabel,
  Alert,
  AlertIcon,
  AlertTitle,
  AlertDescription,
  Menu,
  MenuButton,
  MenuItem,
  MenuList,
} from "@chakra-ui/react";
import { Layout } from "~/components/layout";
import {
  FiCalendar,
  FiAward,
  FiCode,
  FiList,
  FiBarChart2,
  FiTrendingUp,
  FiArrowRight,
} from "react-icons/fi";
import { FaTrophy, FaFire } from "react-icons/fa";
import NextLink from "next/link";
import { api } from "~/utils/api";
import { GPU_DISPLAY_ON_PROFILE } from "~/constants/gpu";
import { LANGUAGE_PROFILE_DISPLAY_NAMES } from "~/constants/language";
import { CheckIcon, ChevronDownIcon } from "@chakra-ui/icons";

// Define the type for activity data
interface ActivityItem {
  date: string;
  count: number;
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
function ActivityCalendar({
  data,
  joinedYear,
}: {
  data: ActivityItem[];
  joinedYear: number;
}) {
  const weeks = 52; 
  const days = 7;
  const [selectedYear, setSelectedYear] = useState(new Date().getFullYear()); 
  const today = new Date();
  const currentYear = today.getFullYear();

  const dateMap: Record<string, number> = {};
  data.forEach((item) => {
    const itemYear = parseInt(item.date.split('-')[0] ?? "0");
    if (itemYear === selectedYear) {
      dateMap[item.date] = (dateMap[item.date] ?? 0) + item.count;
    }
  });

  type GridCell = { date: string; count: number; bgColor: string };

  const calendarGrid: GridCell[][] = Array(days)
    .fill(null)
    .map(() =>
      Array(weeks)
        .fill(null)
        .map(() => ({ date: "", count: 0, bgColor: "whiteAlpha.100" }))
    );

  
  const dayNames = ["Mon", "Wed", "Fri", "Sun"];

  const months: string[] = [];
  
  if (selectedYear === currentYear) {
    for (let i = 0; i < 12; i++) {
      const date = new Date();
      date.setDate(1);
      date.setMonth(today.getMonth() - 11 + i);
      months.push(date.toLocaleString("default", { month: "short" }));
    }
  } else {
    for (let i = 0; i < 12; i++) {
      const date = new Date(selectedYear, i, 1);
      months.push(date.toLocaleString("default", { month: "short" }));
    }
  }

  const isCurrentYear = selectedYear === currentYear;
  
  for (let w = 0; w < weeks; w++) {
    for (let d = 0; d < days; d++) {
      let date;
      
      if (isCurrentYear) {
        date = new Date();
        date.setDate(today.getDate() - ((weeks - w - 1) * 7 + (days - d - 1)));
      } else {
        date = new Date(selectedYear, 0, 1);
        const dayOfWeek = date.getDay();
        
        date.setDate(date.getDate() - (dayOfWeek === 0 ? 6 : dayOfWeek - 1));
        
        date.setDate(date.getDate() + (w * 7) + d);
      }
      
      const dateStr = date.toISOString().split("T")[0]!;
      const count = dateMap[dateStr] ?? 0;

      let bgColor = "whiteAlpha.100";
      if (count > 0) {
        if (count < 3) bgColor = "green.100";
        else if (count < 6) bgColor = "green.200";
        else if (count < 10) bgColor = "green.400";
        else if (count < 15) bgColor = "green.600";
        else bgColor = "green.700";
      }

      if (calendarGrid[d]) {
        calendarGrid[d]![w] = {
          date: dateStr,
          count,
          bgColor,
        };
      }
    }
  }

  const totalCount = Object.values(dateMap).reduce(
    (sum, count) => sum + count,
    0
  );
  
  const timeDisplayText = selectedYear === currentYear 
    ? "in the last year" 
    : `in ${selectedYear}`;

  
  const availableYears = [];
  for (let year = currentYear; year >= joinedYear; year--) {
    availableYears.push(year);
  }

  const handleYearChange = (year: number) => {
    setSelectedYear(year);
  };

  return (
    <Box>
      <HStack mb={4} justifyContent="space-between">
        <HStack spacing={3}>
          <Icon as={FaFire} color="blue.300" w={5} h={5} />
          <Text fontSize="sm" color="whiteAlpha.800">
            <Text as="span" fontWeight="bold" fontSize="md" color="white">
              {totalCount}
            </Text>{" "}
            submissions {timeDisplayText}
          </Text>
        </HStack>
        
        {/* Year dropdown */}
        <Menu>
          <MenuButton
            as={Button}
            size="sm"
            width="100px"
            bg="gray.700"
            color="white"
            borderRadius="lg"
            borderColor="gray.600"
            borderWidth="1px"
            fontWeight="medium"
            _hover={{ borderColor: "blue.300", bg: "gray.650" }}
            _active={{ bg: "gray.650" }}
            rightIcon={<ChevronDownIcon color="blue.300" />}
          >
            {selectedYear}
          </MenuButton>
          <MenuList
            bg="gray.700"
            borderColor="gray.600"
            borderRadius="lg"
            boxShadow="lg"
            py={1}
            minW="100px"
          >
            {availableYears.map((year) => (
              <MenuItem
                key={year}
                value={year}
                onClick={() => handleYearChange(year)}
                bg={selectedYear === year ? "blue.900" : "gray.700"}
                borderRadius="lg"
                color="white"
                _hover={{ bg: "gray.600" }}
                fontSize="sm"
              >
                <Text>{year}</Text>
                {selectedYear === year && (
                  <Icon as={CheckIcon} ml="auto" boxSize={3} />
                )}
              </MenuItem>
            ))}
          </MenuList>
        </Menu>
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
        <Box flex="1" ml={2} key={selectedYear} transition="opacity 0.9s ease" opacity={1} animation="fadeIn 0.9s">
          {/* Month labels */}
          <Flex mb={1} width="100%" ml={4}>
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
                  <Box w="10px" h="10px" borderRadius="sm" bg="green.200" />
                  <Box w="10px" h="10px" borderRadius="sm" bg="green.400" />
                  <Box w="10px" h="10px" borderRadius="sm" bg="green.600" />
                  <Box w="10px" h="10px" borderRadius="sm" bg="green.700" />
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

export default function UserProfile() {
  const router = useRouter();
  const { username } = router.query;
  useSession();

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
        <Grid templateColumns={{ base: "1fr", md: "320px 1fr" }} gap={8} height="100%">
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
                  <Skeleton isLoaded={!isLoading} mx="auto" startColor="gray.700" endColor="gray.800">
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
                      startColor="gray.700"
                      endColor="gray.800"
                    >
                      <Heading color="white" size="lg">
                        {userData?.username ?? username}
                      </Heading>
                    </Skeleton>

                    <Skeleton
                      isLoaded={!isLoading}
                      width={isLoading ? "150px" : "auto"}
                      startColor="gray.700"
                      endColor="gray.800"
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

                    <Skeleton isLoaded={!isLoading} startColor="gray.700" endColor="gray.800">
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
                <Skeleton isLoaded={!isLoading} startColor="gray.700" endColor="gray.800">
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

                <Skeleton isLoaded={!isLoading} startColor="gray.700" endColor="gray.800">
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
              <Skeleton isLoaded={!isLoading} width="100%" startColor="gray.700" endColor="gray.800">
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
              <Skeleton isLoaded={!isLoading} width="100%" startColor="gray.700" endColor="gray.800">
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
                      <TagLabel>
                        {userData?.languagePercentage && userData.languagePercentage.length > 0 
                          ? `${userData.languagePercentage[0]?.language} (${userData.languagePercentage[0]?.percentage}%)`
                          : "N/A"}
                      </TagLabel>
                    </Tag>
                    <Tag size="md" borderRadius="full" mb={2}>
                      <Box w="10px" h="10px" bg="green.700" borderRadius="full" mr={2} />
                      <TagLabel>
                        {userData?.languagePercentage && userData.languagePercentage.length > 1
                          ? `${userData.languagePercentage[1]?.language} (${userData.languagePercentage[1]?.percentage}%)`
                          : "N/A"}
                      </TagLabel>
                    </Tag>
                  </HStack>
                </Box>
              </Skeleton>
            </VStack>
          </GridItem>

          {/* Right Column */}
          <GridItem display="flex" flexDirection="column" gap={6} flexGrow={1}>
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
              </Flex>

              <Skeleton
                isLoaded={!isLoading}
                height={isLoading ? "200px" : "auto"}
                startColor="gray.700"
                endColor="gray.800"
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
              boxShadow="lg"
              borderWidth="1px"
              borderColor="blue.900"
              flexGrow={1}
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
                  <VStack spacing={0} align="stretch" divider={<Divider borderColor="gray.700" />} paddingTop={2}>
                    {userData.recentSubmissions.slice(0, 3).map((submission, _) => (
                      <NextLink
                        key={submission.id}
                        href={`/submissions/${submission.id}`}
                        passHref
                      >
                        <Box 
                          as="a"
                          py={4}
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
                                {/* Language */}
                                <Box 
                                  px={3} 
                                  py={1.5} 
                                  borderRadius="md" 
                                  bg="blue.900"
                                  minW="80px"
                                  textAlign="center"
                                >
                                  <Text color="white" fontSize="sm" fontWeight="semibold">
                                    {LANGUAGE_PROFILE_DISPLAY_NAMES[submission.language] ?? "N/A"}
                                  </Text>
                                  <Text color="whiteAlpha.700" fontSize="xs" mt={0.5}>
                                    Framework
                                  </Text>
                                </Box>

                                {/* GPU Type */}
                                <Box 
                                  px={3} 
                                  py={1.5} 
                                  borderRadius="md" 
                                  bg="blue.900"
                                  minW="80px"
                                  textAlign="center"
                          
                                >
                                  <Text color="white" fontSize="sm" fontWeight="semibold">
                                    {GPU_DISPLAY_ON_PROFILE[submission.gpuType as keyof typeof GPU_DISPLAY_ON_PROFILE] ?? "N/A"}
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
                                  bg="blue.900"
                                  minW="80px"
                                  textAlign="center"
                                >
                                  <Text color="white" fontSize="sm" fontWeight="semibold">
                                    {submission.gflops ?? "N/A"}
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
                                  bg="blue.900"
                                  minW="80px"
                                  textAlign="center"
                                >
                                  <Text color="white" fontSize="sm" fontWeight="semibold">
                                    {submission.runtime ?? "N/A"}
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
