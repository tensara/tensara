import React from "react";
import {
  Container,
  Grid,
  GridItem,
  VStack,
  Box,
  Skeleton,
  Heading,
  HStack,
  Icon,
  Flex,
  Tabs,
  TabList,
  TabPanels,
  Tab,
  TabPanel,
  Text,
} from "@chakra-ui/react";
import { useRouter } from "next/router";
import { useSession } from "next-auth/react";
import NextLink from "next/link";

import { Layout } from "~/components/layout";

import {
  UserHeaderBox,
  UserStatsBox,
  TensaraScoreBox,
  UserLanguagesBox,
} from "~/components/profile/user";
import ActivityCalendar from "~/components/profile/ActivityCalendar";
import { RecentSubmissionsList } from "~/components/profile/RecentSubmissions";
import UserNotFoundAlert from "~/components/profile/UserNotFoundAlert";

import { FiTrendingUp, FiList } from "react-icons/fi";

import { api } from "~/utils/api";

// Define the type for activity data
interface ActivityItem {
  date: string;
  count: number;
}

type BlogPostSummary = {
  id: string;
  title: string;
  slug: string | null;
  publishedAt: string;
  votes: number;
};

function timeAgo(dateString: string) {
  const seconds = Math.max(
    1,
    Math.floor((Date.now() - new Date(dateString).getTime()) / 1000)
  );
  const intervals: { seconds: number; label: string }[] = [
    { seconds: 31536000, label: "y" },
    { seconds: 2592000, label: "mo" },
    { seconds: 86400, label: "d" },
    { seconds: 3600, label: "h" },
    { seconds: 60, label: "m" },
  ];
  for (const interval of intervals) {
    if (seconds >= interval.seconds) {
      return `${Math.floor(seconds / interval.seconds)}${interval.label}`;
    }
  }
  return `${seconds}s`;
}

const VotePill = ({ count }: { count: number }) => (
  <Box
    minW="52px"
    textAlign="center"
    px={2.5}
    py={1.5}
    rounded="lg"
    border="1px solid"
    borderColor="gray.700"
    bg="gray.800"
  >
    <Text fontSize="sm" fontWeight="700" color="white">
      {typeof count === "number" ? count : 0}
    </Text>
    <Text fontSize="10px" color="gray.400" mt={-0.5}>
      votes
    </Text>
  </Box>
);

const BlogPostsList = ({
  posts,
  isLoading,
  username,
}: {
  posts?: BlogPostSummary[];
  isLoading: boolean;
  username?: string;
}) => {
  if (isLoading) {
    return (
      <VStack spacing={3} align="stretch">
        {Array.from({ length: 3 }).map((_, idx) => (
          <Skeleton
            key={idx}
            height="54px"
            startColor="gray.700"
            endColor="gray.800"
            borderRadius="md"
          />
        ))}
      </VStack>
    );
  }

  if (!posts?.length) {
    return (
      <Box py={6} textAlign="center">
        <Text color="gray.400" fontSize="sm">
          {username
            ? `${username} hasn't published any blog posts yet.`
            : "No blog posts found."}
        </Text>
      </Box>
    );
  }

  return (
    <VStack align="stretch" spacing={0}>
      {posts.map((post, idx) => (
        <Flex
          key={post.id}
          py={3}
          borderBottom={idx === posts.length - 1 ? undefined : "1px solid"}
          borderColor="gray.800"
          align="center"
          gap={3}
          flexWrap="wrap"
        >
          <Box flex="1" minW={0}>
            <NextLink href={`/blog/${post.slug ?? post.id}`} passHref>
              <Text
                color="white"
                fontWeight="600"
                noOfLines={1}
                _hover={{ color: "brand.primary" }}
                cursor="pointer"
              >
                {post.title}
              </Text>
            </NextLink>
            <Text color="gray.500" fontSize="sm" noOfLines={1}>
              {timeAgo(post.publishedAt)} ago
            </Text>
          </Box>
          <VotePill count={post.votes} />
        </Flex>
      ))}
    </VStack>
  );
};

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
      ogTitle={`${typeof username === "string" ? username : "User"}`}
      ogDescription={`View ${typeof username === "string" ? username : "User"}'s profile on Tensara.`}
      ogImgSubtitle={`Profiles | Tensara`}
    >
      <Container maxW="container.xl" py={6}>
        {apiError ? (
          <UserNotFoundAlert
            username={typeof username === "string" ? username : ""}
            onReturnHome={() => router.push("/")}
          />
        ) : (
          <Grid
            templateColumns={{ base: "1fr", md: "320px 1fr" }}
            gap={8}
            height="100%"
          >
            {/* Left Column */}
            <GridItem>
              <VStack spacing={5} align="stretch" height="100%">
                <UserHeaderBox
                  userData={userData}
                  isLoading={isLoading}
                  username={typeof username === "string" ? username : ""}
                />
                <UserStatsBox userData={userData} isLoading={isLoading} />
                <TensaraScoreBox
                  score={userData?.stats?.rating}
                  isLoading={isLoading}
                />
                <UserLanguagesBox
                  languagePercentage={userData?.languagePercentage}
                  isLoading={isLoading}
                  communityStats={userData?.communityStats}
                />
              </VStack>
            </GridItem>

            {/* Right Column */}
            <GridItem
              display="flex"
              flexDirection="column"
              gap={6}
              flexGrow={1}
            >
              {/* Activity Graph */}
              <Box
                bg="brand.secondary"
                borderRadius="xl"
                overflow="hidden"
                boxShadow="xl"
                p={6}
                borderWidth="1px"
                borderColor="brand.dark"
                position="relative"
              >
                <Flex justify="space-between" align="center" mb={4}>
                  <HStack>
                    <Icon as={FiTrendingUp} color="brand.primary" boxSize={5} />
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
                  borderRadius="xl"
                >
                  {userData?.activityData && (
                    <ActivityCalendar
                      data={userData.activityData as ActivityItem[]}
                      joinedYear={joinedYear}
                    />
                  )}
                </Skeleton>
              </Box>

              {/* Recent Submissions & Blog Posts */}
              <Box
                bg="brand.secondary"
                borderRadius="xl"
                overflow="hidden"
                boxShadow="lg"
                borderWidth="1px"
                borderColor="brand.dark"
              >
                <Tabs variant="unstyled">
                  <Flex
                    px={5}
                    py={4}
                    borderBottom="1px solid"
                    borderColor="brand.dark"
                    align="center"
                    justify="space-between"
                    flexWrap="wrap"
                    gap={4}
                  >
                    <HStack spacing={3}>
                      <Icon as={FiList} color="brand.primary" boxSize={5} />
                      <Heading size="md" color="white" fontWeight="semibold">
                        Recent Submissions
                      </Heading>
                    </HStack>
                    <TabList gap={4}>
                      <Tab
                        px={0}
                        pb={2}
                        fontWeight="semibold"
                        color="gray.400"
                        borderBottom="2px solid transparent"
                        _selected={{
                          color: "white",
                          borderColor: "brand.primary",
                        }}
                      >
                        Submissions
                      </Tab>
                      <Tab
                        px={0}
                        pb={2}
                        fontWeight="semibold"
                        color="gray.400"
                        borderBottom="2px solid transparent"
                        _selected={{
                          color: "white",
                          borderColor: "brand.primary",
                        }}
                      >
                        Blog Posts
                      </Tab>
                    </TabList>
                  </Flex>
                  <TabPanels>
                    <TabPanel px={0} pt={0}>
                      <RecentSubmissionsList
                        submissions={userData?.recentSubmissions}
                        isLoading={isLoading}
                      />
                    </TabPanel>
                    <TabPanel px={0} pt={0}>
                      <Box px={5} py={4}>
                        <BlogPostsList
                          posts={userData?.blogPosts}
                          isLoading={isLoading}
                          username={
                            typeof username === "string" ? username : undefined
                          }
                        />
                      </Box>
                    </TabPanel>
                  </TabPanels>
                </Tabs>
              </Box>
            </GridItem>
          </Grid>
        )}
      </Container>
    </Layout>
  );
}
