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
} from "@chakra-ui/react";
import { FiBarChart2 } from "react-icons/fi";

interface UserLanguagesBoxProps {
  languagePercentage?: Array<{
    language: string | undefined;
    percentage: number;
  }>;
  isLoading: boolean;
}

const UserLanguagesBox: React.FC<UserLanguagesBoxProps> = ({
  languagePercentage,
  isLoading,
}) => {
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
          <Icon as={FiBarChart2} color="brand.primary" boxSize={5} />
        </Flex>

        <HStack spacing={2} flexWrap="wrap">
          <Tag size="md" borderRadius="full" mb={2}>
            <Box w="10px" h="10px" bg="purple.700" borderRadius="full" mr={2} />
            <TagLabel>
              {languagePercentage && languagePercentage.length > 0
                ? `${languagePercentage[0]?.language} (${languagePercentage[0]?.percentage}%)`
                : "N/A"}
            </TagLabel>
          </Tag>
          <Tag size="md" borderRadius="full" mb={2}>
            <Box w="10px" h="10px" bg="green.700" borderRadius="full" mr={2} />
            <TagLabel>
              {languagePercentage && languagePercentage.length > 1
                ? `${languagePercentage[1]?.language} (${languagePercentage[1]?.percentage}%)`
                : "N/A"}
            </TagLabel>
          </Tag>
        </HStack>
      </Box>
    </Skeleton>
  );
};

export default UserLanguagesBox;
