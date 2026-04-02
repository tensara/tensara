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
  Checkbox,
  Tooltip,
  useToast,
} from "@chakra-ui/react";
import { FiArrowLeft, FiFilter } from "react-icons/fi";
import { type Submission } from "@prisma/client";
import { GPU_DISPLAY_ON_PROFILE } from "~/constants/gpu";
import {
  formatStatus,
  getStatusColor,
  getStatusIcon,
} from "~/constants/problem";
import { formatRuntime } from "~/utils/format";
import { FaSortAmountDown } from "react-icons/fa";
import { useState, useMemo } from "react";
import { LANGUAGE_PROFILE_DISPLAY_NAMES } from "~/constants/language";
import { useSplitPanel } from "./SplitPanel";
import { api } from "~/utils/api";
import {
  buildCombinedBenchmarkCsv,
  buildCombinedBenchmarkCsvFilename,
  downloadCsv,
  normalizeStoredBenchmarkResults,
} from "~/utils/benchmarkCsv";

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
  const [isCompareMode, setIsCompareMode] = useState(false);
  const [selectedSubmissionIds, setSelectedSubmissionIds] = useState<string[]>(
    []
  );
  const { splitRatio } = useSplitPanel();
  const toast = useToast();

  const useCompactLabels = splitRatio < 40;
  const sortedSelectedSubmissionIds = useMemo(
    () => [...selectedSubmissionIds].sort(),
    [selectedSubmissionIds]
  );
  const exportQuery = api.submissions.getSubmissionsForExport.useQuery(
    { ids: sortedSelectedSubmissionIds },
    {
      enabled: false,
      retry: false,
    }
  );

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
        const aPerf = a.gflops ?? -1 * (a.runtime ?? 0);
        const bPerf = b.gflops ?? -1 * (b.runtime ?? 0);
        return bPerf - aPerf;
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

  const acceptedSubmissionIds = useMemo(
    () =>
      (submissions ?? [])
        .filter((submission) => submission.status === "ACCEPTED")
        .map((submission) => submission.id),
    [submissions]
  );

  const selectedCount = selectedSubmissionIds.length;

  const toggleSubmissionSelection = (submissionId: string) => {
    setSelectedSubmissionIds((current) =>
      current.includes(submissionId)
        ? current.filter((id) => id !== submissionId)
        : [...current, submissionId]
    );
  };

  const clearSelection = () => {
    setSelectedSubmissionIds([]);
  };

  const exitCompareMode = () => {
    setIsCompareMode(false);
    clearSelection();
  };

  const handleDownloadSelectedCsv = async () => {
    if (selectedSubmissionIds.length === 0) return;

    const result = await exportQuery.refetch();
    const exportSubmissions = result.data;

    if (!exportSubmissions || exportSubmissions.length === 0) {
      toast({
        title: "No accepted submissions selected",
        description: "Select accepted submissions to export a comparison CSV.",
        status: "warning",
        duration: 4000,
        isClosable: true,
      });
      return;
    }

    const hydratedSubmissions = exportSubmissions.filter(
      (
        submission
      ): submission is NonNullable<(typeof exportSubmissions)[number]> =>
        Boolean(submission)
    );

    const csv = buildCombinedBenchmarkCsv(
      hydratedSubmissions.map((submission) => ({
        submission: {
          submissionId: submission.id,
          submissionName: submission.name,
          problemSlug: submission.problem.slug,
          problemTitle: submission.problem.title,
          language: submission.language,
          gpuType: submission.gpuType,
          createdAt: submission.createdAt,
          overallAvgRuntimeMs: submission.runtime,
          overallAvgGflops: submission.gflops,
        },
        testCases: normalizeStoredBenchmarkResults({
          benchmarkResults:
            (submission.benchmarkResults as Array<{
              test_id: number;
              name: string;
              runtime_ms?: number;
              avg_runtime_ms?: number;
              gflops?: number;
              avg_gflops?: number;
            }> | null) ?? [],
          testResults: submission.testResults.map((testResult) => ({
            testId: testResult.testId,
            name: testResult.name,
            avgRuntimeMs: testResult.avgRuntimeMs,
            avgGflops: testResult.avgGflops,
            runs: testResult.runs.map((run) => ({
              runtimeMs: run.runtimeMs,
              gflops: run.gflops,
              gpuMetrics: run.gpuMetrics,
            })),
          })),
        }),
      }))
    );

    downloadCsv(
      buildCombinedBenchmarkCsvFilename({
        problemSlug: hydratedSubmissions[0]?.problem.slug,
      }),
      csv
    );
  };

  return (
    <VStack spacing={4} align="stretch" p={3}>
      <VStack spacing={4} align="stretch">
        <HStack justify="space-between">
          <Heading size="md">My Submissions</Heading>
          <HStack spacing={2}>
            <Button
              size="sm"
              variant={isCompareMode ? "solid" : "ghost"}
              onClick={() =>
                isCompareMode ? exitCompareMode() : setIsCompareMode(true)
              }
              borderRadius="lg"
              bg={isCompareMode ? "blue.500" : "transparent"}
              color={isCompareMode ? "white" : "gray.300"}
              _hover={{
                bg: isCompareMode ? "blue.400" : "whiteAlpha.50",
                color: "white",
              }}
            >
              {isCompareMode ? "Done" : "Compare"}
            </Button>
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
        </HStack>
        {isCompareMode && (
          <Box bg="whiteAlpha.50" borderRadius="xl" p={4}>
            <Text fontSize="sm" color="whiteAlpha.800">
              Pick accepted submissions to compare test-by-test. CSV export will
              include one row per submission and test case, which is the right
              shape for later plotting.
            </Text>
          </Box>
        )}

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

          <HStack spacing={2}>
            {isCompareMode && acceptedSubmissionIds.length > 0 && (
              <Button
                size="sm"
                variant="ghost"
                onClick={() =>
                  setSelectedSubmissionIds((current) =>
                    current.length === acceptedSubmissionIds.length
                      ? []
                      : acceptedSubmissionIds
                  )
                }
                color="gray.300"
                bg="whiteAlpha.50"
                _focus={{ bg: "whiteAlpha.100" }}
                _hover={{ bg: "whiteAlpha.100" }}
                fontSize="sm"
                px={3}
              >
                {selectedCount === acceptedSubmissionIds.length
                  ? "Clear All"
                  : "Select All Accepted"}
              </Button>
            )}
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
          filteredSubmissions.map((submission) => {
            const isAccepted = submission.status === "ACCEPTED";
            const isSelected = selectedSubmissionIds.includes(submission.id);
            const card = (
              <Box
                bg={isSelected ? "whiteAlpha.100" : "whiteAlpha.50"}
                p={4}
                borderRadius="xl"
                cursor={
                  isCompareMode
                    ? isAccepted
                      ? "pointer"
                      : "not-allowed"
                    : "pointer"
                }
                borderWidth="1px"
                borderColor={isSelected ? "blue.400" : "transparent"}
                opacity={isCompareMode && !isAccepted ? 0.7 : 1}
                _hover={{
                  bg:
                    isCompareMode && !isAccepted
                      ? "whiteAlpha.50"
                      : "whiteAlpha.100",
                }}
                onClick={() => {
                  if (!isCompareMode || !isAccepted) return;
                  toggleSubmissionSelection(submission.id);
                }}
              >
                <HStack justify="space-between" align="start">
                  <HStack align="start" spacing={3}>
                    {isCompareMode && (
                      <Tooltip
                        label={
                          isAccepted
                            ? "Add this submission to the comparison CSV"
                            : "Only accepted submissions can be compared"
                        }
                        hasArrow
                      >
                        <Box pt={1}>
                          <Checkbox
                            isChecked={isSelected}
                            isDisabled={!isAccepted}
                            pointerEvents="none"
                            colorScheme="blue"
                          />
                        </Box>
                      </Tooltip>
                    )}
                    <Icon
                      as={getStatusIcon(submission.status)}
                      color={`${getStatusColor(submission.status)}.400`}
                      mt={1}
                    />
                    <VStack align="start" spacing={0}>
                      <HStack spacing={2} wrap="wrap">
                        <Text fontWeight="semibold">
                          {submission.name?.trim() ??
                            formatStatus(submission.status)}
                        </Text>
                        {submission.name?.trim() ? (
                          <Text color="whiteAlpha.500" fontSize="sm">
                            {formatStatus(submission.status)}
                          </Text>
                        ) : null}
                        {isCompareMode && !isAccepted ? (
                          <Text color="whiteAlpha.500" fontSize="sm">
                            accepted only
                          </Text>
                        ) : null}
                      </HStack>
                      <Text color="whiteAlpha.600" fontSize="sm" ml={1}>
                        {LANGUAGE_PROFILE_DISPLAY_NAMES[submission.language]} •{" "}
                        {
                          GPU_DISPLAY_ON_PROFILE[
                            (submission.gpuType ??
                              "T4") as keyof typeof GPU_DISPLAY_ON_PROFILE
                          ]
                        }
                      </Text>
                    </VStack>
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
                {(submission.gflops !== null ||
                  submission.runtime !== null) && (
                  <SimpleGrid columns={2} spacing={4} mt={2}>
                    {submission.gflops !== null && (
                      <Box>
                        <Text color="whiteAlpha.600" fontSize="sm">
                          Performance
                        </Text>
                        <Text fontWeight="semibold">
                          {submission.gflops.toFixed(2)} GFLOPS
                        </Text>
                      </Box>
                    )}
                    {submission.runtime !== null && (
                      <Box>
                        <Text color="whiteAlpha.600" fontSize="sm">
                          Runtime
                        </Text>
                        <Text fontWeight="semibold">
                          {formatRuntime(submission.runtime)}
                        </Text>
                      </Box>
                    )}
                  </SimpleGrid>
                )}
              </Box>
            );

            if (isCompareMode) {
              return <Box key={submission.id}>{card}</Box>;
            }

            return (
              <Link
                key={submission.id}
                href={`/submissions/${submission.id}`}
                style={{ textDecoration: "none" }}
              >
                {card}
              </Link>
            );
          })
        )}
      </VStack>
      {isCompareMode && (
        <Box
          position="sticky"
          bottom={0}
          bg="rgba(9, 13, 24, 0.96)"
          borderWidth="1px"
          borderColor="whiteAlpha.200"
          borderRadius="xl"
          p={4}
          backdropFilter="blur(10px)"
        >
          <HStack
            justify="space-between"
            align="center"
            wrap="wrap"
            spacing={3}
          >
            <Box>
              <Text fontWeight="semibold">
                {selectedCount} submission{selectedCount === 1 ? "" : "s"}{" "}
                selected
              </Text>
              <Text color="whiteAlpha.600" fontSize="sm">
                Export a single CSV with one row per submission and test case.
              </Text>
            </Box>
            <HStack spacing={2}>
              <Button
                size="sm"
                variant="ghost"
                onClick={clearSelection}
                isDisabled={selectedCount === 0}
              >
                Clear
              </Button>
              <Button
                size="sm"
                colorScheme="blue"
                onClick={() => void handleDownloadSelectedCsv()}
                isDisabled={selectedCount === 0}
                isLoading={exportQuery.isFetching}
                loadingText="Preparing CSV"
              >
                Download CSV
              </Button>
            </HStack>
          </HStack>
        </Box>
      )}
    </VStack>
  );
};

export default MySubmissions;
