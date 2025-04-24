import React from "react";
import {
  Box,
  Flex,
  Heading,
  Text,
  HStack,
  VStack,
  Icon,
  Grid,
  Divider,
  Tag,
  Skeleton,
} from "@chakra-ui/react";
import NextLink from "next/link";

import { FiList, FiCalendar } from "react-icons/fi";

import { GPU_DISPLAY_ON_PROFILE } from "~/constants/gpu";
import { LANGUAGE_PROFILE_DISPLAY_NAMES } from "~/constants/language";

interface SubmissionData {
  id: string;
  problemId?: string;
  problemName: string;
  status: string;
  date: string | undefined;
  language: string;
  gpuType: string | null;
  gflops?: string;
  runtime?: string;
}

interface RecentSubmissionsProps {
  submissions?: SubmissionData[];
  isLoading: boolean;
}

const RecentSubmissions: React.FC<RecentSubmissionsProps> = ({
  submissions,
  isLoading,
}) => {
  return (
    <Box
      bg="brand.secondary"
      borderRadius="xl"
      overflow="hidden"
      boxShadow="lg"
      borderWidth="1px"
      borderColor="brand.dark"
      flexGrow={1}
    >
      {/* Header */}
      <Flex
        px={5}
        py={4}
        bg="brand.secondary"
        borderBottom="1px solid"
        borderColor="brand.dark"
        align="center"
        justify="space-between"
      >
        <HStack spacing={3}>
          <Icon as={FiList} color="brand.primary" boxSize={5} />
          <Heading size="md" color="white" fontWeight="semibold">
            Recent Submissions
          </Heading>
        </HStack>
      </Flex>

      <Box bg="brand.secondary" px={0} py={0}>
        {isLoading ? (
          <VStack spacing={1} align="stretch" p={2}>
            {Array(4)
              .fill(undefined)
              .map(
                (_: undefined, i: number): JSX.Element => (
                  <Skeleton
                    key={i}
                    height="80px"
                    startColor="gray.700"
                    endColor="gray.800"
                    borderRadius="md"
                  />
                )
              )}
          </VStack>
        ) : submissions && submissions.length > 0 ? (
          <VStack
            spacing={0}
            align="stretch"
            divider={<Divider borderColor="gray.700" />}
            paddingTop={2}
          >
            {submissions.slice(0, 3).map((submission) => (
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
                  bg="brand.secondary"
                  _hover={{
                    bg: "brand.secondary",
                    cursor: "pointer",
                  }}
                  transition="all 0.2s ease"
                  display="block"
                  borderLeftWidth="3px"
                  borderLeftColor={
                    submission.status === "accepted" ? "green.400" : "red.400"
                  }
                >
                  <Grid templateColumns="3fr 2fr" gap={4} alignItems="center">
                    {/* Left side: Problem information */}
                    <Box>
                      <Flex align="center" mb={1.5}>
                        <Text color="white" fontWeight="medium" mr={2}>
                          {submission.problemName}
                        </Text>

                        <Tag
                          size="sm"
                          borderRadius="full"
                          colorScheme={
                            submission.status === "accepted" ? "green" : "red"
                          }
                          py={0.5}
                        >
                          {submission.status === "accepted"
                            ? "Accepted"
                            : "Failed"}
                        </Tag>
                      </Flex>

                      <HStack spacing={4}>
                        <HStack spacing={1.5}>
                          <Icon
                            as={FiCalendar}
                            color="brand.primary"
                            boxSize="14px"
                          />
                          <Text color="whiteAlpha.700" fontSize="sm">
                            {submission.date}
                          </Text>
                        </HStack>
                      </HStack>
                    </Box>

                    {/* Right side: Performance metrics */}
                    <Flex justify="flex-end">
                      <HStack
                        spacing={4}
                        bg="gray.800"
                        borderRadius="lg"
                        p={2}
                        borderWidth="1px"
                        borderColor="gray.700"
                      >
                        {/* Language */}
                        <Box
                          px={3}
                          pr={4}
                          py={1.5}
                          borderRight="1px solid"
                          borderColor="gray.600"
                          bg="gray.800"
                          minW="80px"
                          textAlign="center"
                        >
                          <Text
                            color="white"
                            fontSize="sm"
                            fontWeight="semibold"
                          >
                            {LANGUAGE_PROFILE_DISPLAY_NAMES[
                              submission.language
                            ] ?? "N/A"}
                          </Text>
                          <Text color="whiteAlpha.700" fontSize="xs" mt={0.5}>
                            Framework
                          </Text>
                        </Box>

                        {/* GPU Type */}
                        <Box
                          px={3}
                          pr={8}
                          py={1.5}
                          borderRight="1px solid"
                          borderColor="gray.600"
                          bg="gray.800"
                          minW="80px"
                          textAlign="center"
                        >
                          <Text
                            color="white"
                            fontSize="sm"
                            fontWeight="semibold"
                          >
                            {GPU_DISPLAY_ON_PROFILE[
                              submission.gpuType as keyof typeof GPU_DISPLAY_ON_PROFILE
                            ] ?? "N/A"}
                          </Text>
                          <Text color="whiteAlpha.700" fontSize="xs" mt={0.5}>
                            GPU
                          </Text>
                        </Box>

                        {/* GLOPS info */}
                        <Box
                          px={3}
                          py={1.5}
                          pr={8}
                          borderRight="1px solid"
                          borderColor="gray.600"
                          bg="gray.800"
                          minW="80px"
                          textAlign="center"
                        >
                          <Text
                            color="white"
                            fontSize="sm"
                            fontWeight="semibold"
                          >
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
                          bg="gray.800"
                          minW="80px"
                          textAlign="center"
                        >
                          <Text
                            color="white"
                            fontSize="sm"
                            fontWeight="semibold"
                          >
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
            <Box p={4} borderRadius="full" bg="gray.800" mb={3}>
              <Icon as={FiList} color="blue.400" boxSize={6} />
            </Box>
            <Text
              color="whiteAlpha.800"
              fontSize="md"
              fontWeight="medium"
              mb={1}
            >
              No submissions yet
            </Text>
            <Text
              color="whiteAlpha.600"
              fontSize="sm"
              textAlign="center"
              maxW="xs"
            >
              Your recent submission history will appear here
            </Text>
          </Flex>
        )}
      </Box>
    </Box>
  );
};

export default RecentSubmissions;
