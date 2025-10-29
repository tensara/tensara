import {
  Box,
  VStack,
  HStack,
  Text,
  Badge,
  Button,
  Collapse,
  useDisclosure,
  Skeleton,
  SkeletonText,
  Alert,
  AlertIcon,
  AlertDescription,
  Icon,
  Flex,
} from "@chakra-ui/react";
import { useState, useEffect } from "react";
import { api } from "~/utils/api";
import CodeEditor from "~/components/problem/CodeEditor";
import { type ProgrammingLanguage } from "~/types/misc";
import Link from "next/link";
import {
  FiCode,
  FiChevronDown,
  FiChevronUp,
  FiExternalLink,
  FiCpu,
  FiZap,
  FiCheckCircle,
  FiClock,
} from "react-icons/fi";
import { format } from "date-fns";

interface SubmissionEmbedCardProps {
  submissionId: string;
  description?: string;
}

/**
 * SubmissionEmbedCard displays a rich, interactive embed of a submission
 * within a blog post. It shows submission metrics, code preview (collapsible),
 * and links to the full submission page.
 */
export function SubmissionEmbedCard({
  submissionId,
  description,
}: SubmissionEmbedCardProps) {
  const { isOpen, onToggle } = useDisclosure({ defaultIsOpen: false });
  const [code, setCode] = useState("");

  // Fetch submission data using the existing getById endpoint
  const {
    data: submission,
    isLoading,
    error,
  } = api.submissions.getSubmissionById.useQuery(
    { id: submissionId },
    {
      enabled: !!submissionId,
    }
  );

  // Set code when submission data is loaded
  useEffect(() => {
    if (submission && "code" in submission) {
      setCode(submission.code as string);
    }
  }, [submission]);

  // Loading state with skeleton
  if (isLoading) {
    return (
      <Box
        p={6}
        borderRadius="2xl"
        bg="whiteAlpha.50"
        backdropFilter="blur(10px)"
        borderWidth="1px"
        borderColor="whiteAlpha.100"
        my={6}
      >
        <VStack align="stretch" spacing={4}>
          <Skeleton height="30px" width="70%" />
          <HStack spacing={3}>
            <Skeleton height="24px" width="80px" />
            <Skeleton height="24px" width="80px" />
            <Skeleton height="24px" width="100px" />
          </HStack>
          <SkeletonText mt="4" noOfLines={2} spacing="4" />
        </VStack>
      </Box>
    );
  }

  // Error state - submission not found or not accessible
  if (error || !submission) {
    return (
      <Alert
        status="warning"
        variant="subtle"
        borderRadius="xl"
        my={6}
        bg="orange.900"
        borderWidth="1px"
        borderColor="orange.700"
      >
        <AlertIcon />
        <AlertDescription color="orange.100">
          {error?.message ||
            "Submission not found or you don't have access to view it."}
        </AlertDescription>
      </Alert>
    );
  }

  // Check if code is available
  const hasCode = "code" in submission;
  const difficultyColors: Record<string, string> = {
    EASY: "green",
    MEDIUM: "yellow",
    HARD: "orange",
    EXPERT: "red",
  };

  return (
    <Box
      borderRadius="2xl"
      bg="whiteAlpha.50"
      backdropFilter="blur(10px)"
      borderWidth="1px"
      borderColor="whiteAlpha.100"
      overflow="hidden"
      my={6}
      transition="all 0.3s"
      _hover={{
        borderColor: "green.600",
        transform: "translateY(-2px)",
        shadow: "lg",
      }}
    >
      <VStack align="stretch" spacing={0}>
        {/* Header Section */}
        <Box
          p={6}
          bg="whiteAlpha.50"
          borderBottomWidth="1px"
          borderColor="whiteAlpha.100"
        >
          <VStack align="stretch" spacing={4}>
            {/* Problem Name and Difficulty */}
            <HStack justify="space-between" align="flex-start" wrap="wrap">
              <VStack align="flex-start" spacing={2} flex="1">
                <Text
                  fontSize="2xl"
                  fontWeight="bold"
                  color="white"
                  letterSpacing="tight"
                >
                  {submission.problem.title}
                </Text>
                <HStack spacing={3} flexWrap="wrap">
                  {"difficulty" in submission.problem && (
                    <Badge
                      colorScheme={
                        difficultyColors[
                          (submission.problem as any).difficulty || "MEDIUM"
                        ] || "yellow"
                      }
                      fontSize="xs"
                      px={2}
                      py={1}
                      borderRadius="md"
                    >
                      {(submission.problem as any).difficulty || "MEDIUM"}
                    </Badge>
                  )}
                  <Badge
                    colorScheme="blue"
                    fontSize="xs"
                    px={2}
                    py={1}
                    borderRadius="md"
                  >
                    {submission.language.toUpperCase()}
                  </Badge>
                  {submission.gpuType && (
                    <Badge
                      colorScheme="purple"
                      fontSize="xs"
                      px={2}
                      py={1}
                      borderRadius="md"
                    >
                      <HStack spacing={1}>
                        <Icon as={FiCpu} />
                        <Text>{submission.gpuType}</Text>
                      </HStack>
                    </Badge>
                  )}
                </HStack>
              </VStack>
            </HStack>

            {/* Metrics Grid */}
            <Flex
              bg="whiteAlpha.50"
              borderRadius="xl"
              p={4}
              gap={6}
              flexWrap="wrap"
              justify="space-around"
            >
              {submission.gflops !== null && (
                <VStack spacing={1} minW="100px">
                  <HStack spacing={2} color="green.400">
                    <Icon as={FiZap} boxSize={4} />
                    <Text fontSize="xs" fontWeight="semibold" color="gray.400">
                      PERFORMANCE
                    </Text>
                  </HStack>
                  <Text fontSize="xl" fontWeight="bold" color="green.300">
                    {submission.gflops.toFixed(2)}
                    <Text
                      as="span"
                      fontSize="xs"
                      color="gray.500"
                      ml={1}
                      fontWeight="normal"
                    >
                      GFLOPS
                    </Text>
                  </Text>
                </VStack>
              )}

              {submission.runtime !== null && (
                <VStack spacing={1} minW="100px">
                  <HStack spacing={2} color="blue.400">
                    <Icon as={FiClock} boxSize={4} />
                    <Text fontSize="xs" fontWeight="semibold" color="gray.400">
                      RUNTIME
                    </Text>
                  </HStack>
                  <Text fontSize="xl" fontWeight="bold" color="blue.300">
                    {submission.runtime.toFixed(2)}
                    <Text
                      as="span"
                      fontSize="xs"
                      color="gray.500"
                      ml={1}
                      fontWeight="normal"
                    >
                      ms
                    </Text>
                  </Text>
                </VStack>
              )}

              {submission.passedTests !== null &&
                submission.totalTests !== null && (
                  <VStack spacing={1} minW="100px">
                    <HStack spacing={2} color="green.400">
                      <Icon as={FiCheckCircle} boxSize={4} />
                      <Text
                        fontSize="xs"
                        fontWeight="semibold"
                        color="gray.400"
                      >
                        TESTS PASSED
                      </Text>
                    </HStack>
                    <Text fontSize="xl" fontWeight="bold" color="green.300">
                      {submission.passedTests}/{submission.totalTests}
                    </Text>
                  </VStack>
                )}
            </Flex>

            {/* Optional Description */}
            {description && (
              <Box
                p={4}
                bg="whiteAlpha.50"
                borderRadius="lg"
                borderLeftWidth="3px"
                borderColor="green.600"
              >
                <Text color="gray.300" fontSize="sm" whiteSpace="pre-wrap">
                  {description}
                </Text>
              </Box>
            )}
          </VStack>
        </Box>

        {/* Code Preview Section (Collapsible) */}
        {hasCode && (
          <Box>
            <Button
              w="100%"
              onClick={onToggle}
              variant="ghost"
              justifyContent="space-between"
              rightIcon={
                <Icon as={isOpen ? FiChevronUp : FiChevronDown} boxSize={5} />
              }
              leftIcon={<Icon as={FiCode} boxSize={5} />}
              py={6}
              borderRadius="none"
              _hover={{ bg: "whiteAlpha.100" }}
              color="gray.300"
              fontWeight="semibold"
            >
              {isOpen ? "Hide Code" : "Show Code"}
            </Button>

            <Collapse in={isOpen} animateOpacity>
              <Box p={6} pt={0}>
                <Box
                  h="400px"
                  borderRadius="lg"
                  overflow="hidden"
                  borderWidth="1px"
                  borderColor="whiteAlpha.200"
                >
                  <CodeEditor
                    code={code}
                    setCode={setCode}
                    selectedLanguage={
                      submission.language as ProgrammingLanguage
                    }
                    isEditable={false}
                  />
                </Box>
              </Box>
            </Collapse>
          </Box>
        )}

        {/* Footer with Actions */}
        <Box
          p={4}
          bg="whiteAlpha.50"
          borderTopWidth="1px"
          borderColor="whiteAlpha.100"
        >
          <HStack justify="space-between" flexWrap="wrap" gap={3}>
            <HStack spacing={2} color="gray.400" fontSize="sm">
              <Text>Submitted by</Text>
              <Link href={`/user/${submission.user.username || ""}`}>
                <Text
                  color="blue.400"
                  fontWeight="medium"
                  _hover={{ color: "blue.300", textDecoration: "underline" }}
                >
                  {submission.user.username || "Unknown"}
                </Text>
              </Link>
              <Text>•</Text>
              <Text>
                {format(new Date(submission.createdAt), "MMM d, yyyy")}
              </Text>
            </HStack>

            <Button
              as={Link}
              href={`/submissions/${submission.id}`}
              size="sm"
              colorScheme="green"
              variant="outline"
              rightIcon={<Icon as={FiExternalLink} />}
              _hover={{ bg: "green.900", borderColor: "green.500" }}
            >
              View Full Submission
            </Button>
          </HStack>
        </Box>
      </VStack>
    </Box>
  );
}
