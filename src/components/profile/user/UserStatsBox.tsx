import React from "react";
import { SimpleGrid, Box, Flex, Text, Icon, Skeleton } from "@chakra-ui/react";
import { FiList, FiCode } from "react-icons/fi";

interface UserStatsProps {
  userData:
    | {
        stats?: {
          submissions?: number;
          solvedProblems?: number;
        };
      }
    | undefined;
  isLoading: boolean;
}

const UserStats: React.FC<UserStatsProps> = ({ userData, isLoading }) => {
  return (
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
            <Text fontSize="2xl" color="white" fontWeight="bold">
              {userData?.stats?.submissions ?? 0}
            </Text>
            <Text color="whiteAlpha.800" fontSize="sm" fontWeight={500}>
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
            <Text fontSize="2xl" color="white" fontWeight="bold">
              {userData?.stats?.solvedProblems ?? 0}
            </Text>
            <Text color="whiteAlpha.800" fontSize="sm" fontWeight={500}>
              Problems
            </Text>
          </Flex>
        </Box>
      </Skeleton>
    </SimpleGrid>
  );
};

export default UserStats;
