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
} from "@chakra-ui/react";
import { useRouter } from "next/router";
import { useSession } from "next-auth/react";

import { Layout } from "~/components/layout";

import {
  UserHeaderBox,
  UserStatsBox,
  TensaraScoreBox,
  UserLanguagesBox,
} from "~/components/profile/user";
import ActivityCalendar from "~/components/profile/ActivityCalendar";
import RecentSubmissions from "~/components/profile/RecentSubmissions";
import UserNotFoundAlert from "~/components/profile/UserNotFoundAlert";

import { FiTrendingUp } from "react-icons/fi";

import { api } from "~/utils/api";

// Define the type for activity data
interface ActivityItem {
  date: string;
  count: number;
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
              <RecentSubmissions
                submissions={userData?.recentSubmissions}
                isLoading={isLoading}
              />
            </GridItem>
          </Grid>
        )}
      </Container>
    </Layout>
  );
}
