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
  ButtonGroup,
  Menu,
  MenuButton,
  MenuList,
  MenuItem,
} from "@chakra-ui/react";
import { FiArrowLeft, FiFilter } from "react-icons/fi";
import { type Submission } from "@prisma/client";
import { GPU_DISPLAY_ON_PROFILE } from "~/constants/gpu";
import {
  formatStatus,
  getStatusColor,
  getStatusIcon,
} from "~/constants/problem";
import { FaSortAmountDown } from "react-icons/fa";
import { useState, useMemo } from "react";
import { LANGUAGE_PROFILE_DISPLAY_NAMES } from "~/constants/language";
import { useSplitPanel } from "./SplitPanel";

interface MySubmissionsProps {
  submissions: Submission[] | undefined;
  isLoading: boolean;
  onBackToProblem: () => void;
}

const MySubmissions = ({
  submissions,
  isLoading,
  onBackToProblem,
}: MySubmissionsProps) => {
  const [statusFilter, setStatusFilter] = useState<string[]>(["all"]);
  const [sortBy, setSortBy] = useState<"time" | "performance">("time");
  const { splitRatio } = useSplitPanel();

  const useCompactLabels = splitRatio < 40;

  const filteredSubmissions = useMemo(() => {
    if (!submissions) return [];
    const filtered = statusFilter.includes("all")
      ? submissions
      : submissions.filter((sub) => statusFilter.includes(sub.status ?? ""));

    return [...filtered].sort((a, b) => {
      if (sortBy === "time") {
        return (
          new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime()
        );
      } else {
        const aGflops = a.gflops ?? -1;
        const bGflops = b.gflops ?? -1;
        return bGflops - aGflops;
      }
    });
  }, [submissions, statusFilter, sortBy]);

  const filterOptions = [
    { value: ["all"], label: "All", shortLabel: "All" },
    { value: ["ACCEPTED"], label: "Accepted", shortLabel: "AC" },
    { value: ["WRONG_ANSWER"], label: "Wrong Answer", shortLabel: "WA" },
    {
      value: [
        "ERROR",
        "RUNTIME_ERROR",
        "COMPILE_ERROR",
        "TIME_LIMIT_EXCEEDED",
        "MEMORY_LIMIT_EXCEEDED",
      ],
      label: "Errors",
      shortLabel: "Err",
    },
  ];

  return (
    <VStack spacing={4} align="stretch" p={6}>
      <VStack spacing={4} align="stretch">
        <HStack justify="space-between">
          <Heading size="md">My Submissions</Heading>
          <Button
            size="sm"
            variant="ghost"
            onClick={onBackToProblem}
            leftIcon={<Icon as={FiArrowLeft} />}
            borderRadius="lg"
            color="gray.300"
            _hover={{
              bg: "whiteAlpha.50",
              color: "white",
            }}
          >
            Back to Problem
          </Button>
        </HStack>

        <HStack justify="space-between" align="center">
          <Box>
            <ButtonGroup
              size="sm"
              variant="ghost"
              spacing={1}
              display={{ base: "none", md: "flex" }}
              flexWrap="wrap"
            >
              {filterOptions.map((status) => (
                <Button
                  key={status.value.join(",")}
                  onClick={() => setStatusFilter(status.value)}
                  color={
                    statusFilter.some((s) => status.value.includes(s))
                      ? "white"
                      : "whiteAlpha.600"
                  }
                  bg={
                    statusFilter.some((s) => status.value.includes(s))
                      ? "whiteAlpha.100"
                      : "transparent"
                  }
                  leftIcon={
                    status.value.includes("all") ? undefined : (
                      <Icon
                        as={getStatusIcon(status.value[0]!)}
                        color={`${getStatusColor(status.value[0]!)}.400`}
                      />
                    )
                  }
                  _hover={{
                    bg: "whiteAlpha.100",
                    color: "white",
                  }}
                  fontSize="sm"
                  title={status.label}
                >
                  {useCompactLabels ? status.shortLabel : status.label}
                </Button>
              ))}
            </ButtonGroup>

            <Menu>
              <MenuButton
                as={Button}
                display={{ base: "flex", md: "none" }}
                size="sm"
                variant="ghost"
                leftIcon={<Icon as={FiFilter} />}
                color="whiteAlpha.600"
                _hover={{
                  bg: "whiteAlpha.100",
                  color: "white",
                }}
              >
                Filter
              </MenuButton>
              <MenuList bg="gray.800" borderColor="whiteAlpha.200" p={0}>
                {filterOptions.map((status) => (
                  <MenuItem
                    key={status.value.join(",")}
                    onClick={() => setStatusFilter(status.value)}
                    bg={
                      statusFilter.some((s) => status.value.includes(s))
                        ? "whiteAlpha.100"
                        : "transparent"
                    }
                    _hover={{
                      bg: "whiteAlpha.200",
                    }}
                    icon={
                      status.value.includes("all") ? undefined : (
                        <Icon
                          as={getStatusIcon(status.value[0]!)}
                          color={`${getStatusColor(status.value[0]!)}.400`}
                        />
                      )
                    }
                    borderRadius="md"
                    fontSize="sm"
                  >
                    {status.label}
                  </MenuItem>
                ))}
              </MenuList>
            </Menu>
          </Box>

          <Button
            size="sm"
            variant="ghost"
            onClick={() =>
              setSortBy(sortBy === "time" ? "performance" : "time")
            }
            color="gray.300"
            leftIcon={<Icon as={FaSortAmountDown} color="gray.300" />}
            bg="whiteAlpha.50"
            _focus={{
              bg: "whiteAlpha.100",
            }}
            _hover={{
              bg: "whiteAlpha.100",
            }}
            fontSize="sm"
            px={3}
          >
            {sortBy === "time" ? "Newest" : "Fastest"}
          </Button>
        </HStack>
      </VStack>

      <VStack spacing={4} align="stretch">
        {isLoading ? (
          <Box display="flex" justifyContent="center" p={4}>
            <Spinner />
          </Box>
        ) : filteredSubmissions.length === 0 ? (
          <Box p={4} textAlign="center" color="whiteAlpha.700">
            No submissions yet
          </Box>
        ) : (
          filteredSubmissions.map((submission) => (
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
                      as={getStatusIcon(submission.status)}
                      color={`${getStatusColor(submission.status)}.400`}
                    />
                    <Text fontWeight="semibold">
                      {formatStatus(submission.status)}
                    </Text>
                    <Text color="whiteAlpha.600" fontSize="sm" ml={1}>
                      {LANGUAGE_PROFILE_DISPLAY_NAMES[submission.language]} •{" "}
                      {
                        GPU_DISPLAY_ON_PROFILE[
                          (submission.gpuType ??
                            "T4") as keyof typeof GPU_DISPLAY_ON_PROFILE
                        ]
                      }
                    </Text>
                  </HStack>
                  <Text color="whiteAlpha.700" fontSize="sm">
                    {new Date(submission.createdAt).toLocaleString("en-US", {
                      year: useCompactLabels ? "2-digit" : "numeric",
                      month: useCompactLabels ? "numeric" : "short",
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

export default MySubmissions;
