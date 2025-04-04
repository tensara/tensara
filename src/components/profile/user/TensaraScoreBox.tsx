import React from "react";
import { Box, Flex, Heading, Text, Icon, Skeleton } from "@chakra-ui/react";
import { FaTrophy } from "react-icons/fa";

interface UserScoreProps {
  score?: number;
  isLoading: boolean;
}

const UserScore: React.FC<UserScoreProps> = ({ score, isLoading }) => {
  return (
    <Skeleton
      isLoaded={!isLoading}
      width="100%"
      startColor="gray.700"
      endColor="gray.800"
    >
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
          <Icon as={FaTrophy} color="yellow.400" boxSize={6} mr={3} />
          <Heading size="md" color="white">
            Tensara Rating
          </Heading>
        </Flex>

        <Text
          fontSize="5xl"
          color="yellow.400"
          fontWeight="bold"
          textShadow="0 0 10px rgba(236, 201, 75, 0.3)"
          textAlign="center"
        >
          {score ?? 0}
        </Text>
      </Box>
    </Skeleton>
  );
};

export default UserScore;
