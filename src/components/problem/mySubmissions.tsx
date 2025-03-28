import {
  Box,
  Heading,
  Text,
  HStack,
  Spinner,
  VStack,
  Button,
  SimpleGrid,
  Icon,
  Link,
} from "@chakra-ui/react";
import { FiArrowLeft } from "react-icons/fi";
import { CheckIcon, TimeIcon, WarningIcon } from "@chakra-ui/icons";
import { type Submission } from "@prisma/client";
import { GPU_DISPLAY_NAMES } from "~/constants/gpu";

interface MySubmissionsProps {
  submissions: Submission[] | undefined;
  isLoading: boolean;
  onBackToProblem: () => void;
}

export const MySubmissions = ({
  submissions,
  isLoading,
  onBackToProblem,
}: MySubmissionsProps) => {
  return (
    <VStack spacing={4} align="stretch" p={6}>
      <HStack justify="space-between">
        <Heading size="md">My Submissions</Heading>
        <HStack>
          <Button
            size="sm"
            variant="ghost"
            onClick={onBackToProblem}
            leftIcon={<Icon as={FiArrowLeft} />}
            borderRadius="full"
            color="gray.300"
            _hover={{
              bg: "whiteAlpha.50",
              color: "white",
            }}
          >
            Back to Problem
          </Button>
        </HStack>
      </HStack>

      <VStack spacing={4} align="stretch">
        {isLoading ? (
          <Box display="flex" justifyContent="center" p={4}>
            <Spinner />
          </Box>
        ) : submissions?.length === 0 ? (
          <Box p={4} textAlign="center" color="whiteAlpha.700">
            No submissions yet
          </Box>
        ) : (
          submissions?.map((submission) => (
            <Link
              key={submission.id}
              href={`/submissions/${submission.id}`}
              style={{ textDecoration: "none" }}
            >
              <Box
                bg="whiteAlpha.50"
                p={4}
                borderRadius="xl"
                cursor="pointer"
                _hover={{ bg: "whiteAlpha.100" }}
              >
                <HStack justify="space-between">
                  <HStack>
                    <Icon
                      as={
                        submission.status === "ACCEPTED"
                          ? CheckIcon
                          : submission.status === "WRONG_ANSWER" || submission.status === "ERROR"
                            ? WarningIcon
                            : TimeIcon
                      }
                      color={
                        submission.status === "ACCEPTED"
                          ? "green.400"
                          : submission.status === "WRONG_ANSWER" || submission.status === "ERROR"
                            ? "red.400"
                            : "blue.400"
                      }
                    />
                    <Text fontWeight="semibold">
                      {submission.status === "ACCEPTED"
                        ? "Accepted"
                        : submission.status === "WRONG_ANSWER"
                          ? "Wrong Answer"
                          : submission.status === "ERROR"
                            ? "Error"
                            : submission.status}
                    </Text>
                    <Text color="whiteAlpha.600" fontSize="sm" ml={1}>
                      {submission.language === "cuda" ? "CUDA" : "Python"} • {GPU_DISPLAY_NAMES[submission.gpuType ?? "T4"]}
                    </Text>
                  </HStack>
                  <Text color="whiteAlpha.700" fontSize="sm">
                    {new Date(submission.createdAt).toLocaleString("en-US", {
                      year: "numeric",
                      month: "short",
                      day: "numeric",
                      hour: "numeric",
                      minute: "2-digit",
                      hour12: true,
                    })}
                  </Text>
                </HStack>
                {submission.gflops !== null && submission.runtime !== null && (
                  <SimpleGrid columns={2} spacing={4} mt={2}>
                    <Box>
                      <Text color="whiteAlpha.600" fontSize="sm">
                        Performance
                      </Text>
                      <Text fontWeight="semibold">
                        {submission.gflops.toFixed(2)} GFLOPS
                      </Text>
                    </Box>
                    <Box>
                      <Text color="whiteAlpha.600" fontSize="sm">
                        Runtime
                      </Text>
                      <Text fontWeight="semibold">
                        {submission.runtime.toFixed(2)}ms
                      </Text>
                    </Box>
                  </SimpleGrid>
                )}
              </Box>
            </Link>
          ))
        )}
      </VStack>
    </VStack>
  );
};
