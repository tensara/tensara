import React from "react";
import {
  Box,
  Flex,
  Heading,
  Icon,
  HStack,
  Tag,
  TagLabel,
  Skeleton,
  SimpleGrid,
  Text,
  VStack,
} from "@chakra-ui/react";
import { FiBarChart2, FiHeart, FiFileText } from "react-icons/fi";

interface UserLanguagesBoxProps {
  languagePercentage?: Array<{
    language: string | undefined;
    percentage: number;
  }>;
  isLoading: boolean;
  communityStats?: {
    totalPosts?: number;
    totalLikes?: number;
  };
}

const UserLanguagesBox: React.FC<UserLanguagesBoxProps> = ({
  languagePercentage,
  isLoading,
  communityStats,
}) => {
  const showCommunityStats = (communityStats?.totalPosts ?? 0) > 0;

  return (
    <Skeleton
      isLoaded={!isLoading}
      width="100%"
      startColor="gray.700"
      endColor="gray.800"
    >
      <Box
        bg="brand.secondary"
        borderRadius="xl"
        p={5}
        borderWidth="1px"
        borderColor="brand.dark"
        boxShadow="md"
      >
        <Flex justify="space-between" align="center" mb={4}>
          <Heading size="sm" color="white">
            Frameworks Used
          </Heading>
          <Icon as={FiBarChart2} color="gray.400" boxSize={5} />
        </Flex>

        <HStack spacing={2} flexWrap="wrap">
          {languagePercentage && languagePercentage.length > 0 ? (
            [...languagePercentage]
              .sort((a, b) => b.percentage - a.percentage)
              .map((item, idx) => {
                const colorPalette = [
                  "purple.700",
                  "green.700",
                  "blue.600",
                  "orange.500",
                  "red.600",
                  "cyan.600",
                  "pink.600",
                  "yellow.500",
                  "teal.700",
                  "gray.600",
                ];
                const color = colorPalette[idx % colorPalette.length];
                return (
                  <Tag key={item.language ?? idx} size="md" borderRadius="full">
                    <Box
                      w="10px"
                      h="10px"
                      bg={color}
                      borderRadius="full"
                      mr={2}
                    />
                    <TagLabel>
                      {item.language} ({item.percentage}%)
                    </TagLabel>
                  </Tag>
                );
              })
          ) : (
            <Tag size="md" borderRadius="full" mb={2}>
              <Box w="10px" h="10px" bg="gray.600" borderRadius="full" mr={2} />
              <TagLabel>N/A</TagLabel>
            </Tag>
          )}
        </HStack>
        {showCommunityStats && (
          <Box mt={6}>
            <Flex justify="space-between" align="center" mb={3}>
              <Heading size="sm" color="white">
                Community Stats
              </Heading>
            </Flex>
            <SimpleGrid columns={2} spacing={3}>
              <VStack
                borderRadius="lg"
                p={4}
                borderWidth="1px"
                borderColor="gray.700"
                align="center"
                spacing={2.5}
                transition="all 0.5s"
                _hover={{
                  bg: "whiteAlpha.100",
                  borderColor: "gray.600",
                  transform: "translateY(-2px)",
                }}
              >
                <Icon as={FiFileText} color="gray.400" boxSize={6} />
                <Text
                  color="white"
                  fontSize="2xl"
                  fontWeight="bold"
                  lineHeight={1}
                >
                  {communityStats?.totalPosts ?? 0}
                </Text>
                <Text
                  color="gray.400"
                  fontSize="xs"
                  fontWeight="medium"
                  letterSpacing="wide"
                >
                  Total Posts
                </Text>
              </VStack>
              <VStack
                borderRadius="lg"
                p={4}
                borderWidth="1px"
                borderColor="gray.700"
                align="center"
                spacing={2.5}
                transition="all 0.5s"
                _hover={{
                  bg: "whiteAlpha.100",
                  borderColor: "gray.600",
                  transform: "translateY(-2px)",
                }}
              >
                <Icon as={FiHeart} color="gray.400" boxSize={6} />
                <Text
                  color="white"
                  fontSize="2xl"
                  fontWeight="bold"
                  lineHeight={1}
                >
                  {communityStats?.totalLikes ?? 0}
                </Text>
                <Text
                  color="gray.400"
                  fontSize="xs"
                  fontWeight="medium"
                  letterSpacing="wide"
                >
                  Total Likes
                </Text>
              </VStack>
            </SimpleGrid>
          </Box>
        )}
      </Box>
    </Skeleton>
  );
};

export default UserLanguagesBox;
