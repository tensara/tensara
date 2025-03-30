import React from "react";
import {
  Box,
  VStack,
  HStack,
  Image,
  Heading,
  Text,
  Icon,
  Badge,
  Skeleton,
} from "@chakra-ui/react";
import { FiCalendar, FiAward } from "react-icons/fi";

// Helper function to get ordinal suffix for dates (1st, 2nd, 3rd, etc.)
export function getOrdinalSuffix(day: number): string {
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

interface UserHeaderBoxProps {
  userData:
    | {
        username?: string | null;
        image?: string | null;
        joinedAt: Date | string;
        stats?: {
          ranking?: number;
        };
      }
    | undefined;
  isLoading: boolean;
  username: string;
}

const UserHeaderBox: React.FC<UserHeaderBoxProps> = ({
  userData,
  isLoading,
  username,
}) => {
  return (
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
        <Skeleton
          isLoaded={!isLoading}
          mx="auto"
          startColor="gray.700"
          endColor="gray.800"
        >
          <Image
            src={userData?.image ?? "https://via.placeholder.com/150"}
            alt={`${
              typeof username === "string" ? username : "User"
            }'s profile`}
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

          <Skeleton
            isLoaded={!isLoading}
            startColor="gray.700"
            endColor="gray.800"
          >
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
  );
};

export default UserHeaderBox;
