import { type NextPage } from "next";
import type { GetServerSideProps } from "next";
import { useEffect, useMemo, useState } from "react";
import {
  Box,
  Button,
  Checkbox,
  Divider,
  Flex,
  HStack,
  Icon,
  Spinner,
  Text,
  useToast,
  VStack,
} from "@chakra-ui/react";
import { FiArrowLeft, FiDownload, FiTrendingUp } from "react-icons/fi";
import NextLink from "next/link";
import { format } from "date-fns";
import superjson from "superjson";
import { createServerSideHelpers } from "@trpc/react-query/server";
import { createInnerTRPCContext } from "~/server/api/trpc";
import { appRouter } from "~/server/api/root";
import { auth } from "~/server/auth";
import { Layout } from "~/components/layout";
import { api, type RouterOutputs } from "~/utils/api";
import { formatRuntime } from "~/utils/format";
import {
  buildCombinedBenchmarkCsv,
  buildCombinedBenchmarkCsvFilename,
  downloadCsv,
  normalizeStoredBenchmarkResults,
} from "~/utils/benchmarkCsv";
import { LANGUAGE_PROFILE_DISPLAY_NAMES } from "~/constants/language";
import { GPU_DISPLAY_NAMES } from "~/constants/gpu";

type AnalysisSubmission =
  RouterOutputs["problems"]["getAnalysisSubmissions"][number];

type ChartSeries = {
  id: string;
  label: string;
  color: string;
  points: Array<number | null>;
};

type PlotExportErrorResponse = {
  error?: string;
  details?: string;
};

type MetricMode = "runtime" | "gflops" | "temperature";
type BackgroundMode = "grid" | "plain" | "spotlight";

const SURFACE_BG = "#0f172a";
const SURFACE_CARD = "#181f2a";
const SURFACE_SOFT = "rgba(24, 31, 42, 0.86)";
const SURFACE_ELEVATED = "rgba(15, 23, 42, 0.92)";
const SURFACE_BORDER = "rgba(148, 163, 184, 0.18)";
const SURFACE_GRID = "rgba(148, 163, 184, 0.12)";
const SURFACE_MUTED = "rgba(226, 232, 240, 0.68)";
const SURFACE_TEXT = "#f8fafc";
const ACCENT = "#10B981";
const ACCENT_SOFT = "rgba(16, 185, 129, 0.12)";
const ACCENT_LINE = "rgba(16, 185, 129, 0.28)";
const SERIES_COLORS = [
  "#38bdf8",
  "#f97316",
  "#22c55e",
  "#a78bfa",
  "#f43f5e",
  "#facc15",
];

const buildSubmissionLabel = (submission: AnalysisSubmission): string => {
  const trimmedName = submission.name?.trim();
  const normalizedName = trimmedName === "" ? undefined : trimmedName;

  return (
    normalizedName ??
    `${LANGUAGE_PROFILE_DISPLAY_NAMES[submission.language]} ${format(
      new Date(submission.createdAt),
      "MMM d"
    )}`
  );
};

const getAverageTemperatureForSubmission = (
  submission: AnalysisSubmission
): number | null => {
  const temps = submission.testResults.flatMap((testResult) =>
    testResult.runs
      .map((run) => {
        const metrics = run.gpuMetrics as {
          sample_count?: number;
          temp_c_mean?: number;
        } | null;
        if (!metrics || (metrics.sample_count ?? 0) <= 0) return null;
        return metrics.temp_c_mean ?? null;
      })
      .filter((value): value is number => value != null)
  );

  if (temps.length === 0) return null;

  return temps.reduce((sum, value) => sum + value, 0) / temps.length;
};

const buildTestCaseChart = (
  submissions: AnalysisSubmission[],
  metricMode: MetricMode,
  seriesColors: string[]
): {
  categories: string[];
  series: ChartSeries[];
  yLabel: string;
} => {
  const categories: string[] = [];
  const seen = new Set<string>();

  const normalized = submissions.map((submission) => {
    const testCases = normalizeStoredBenchmarkResults({
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
    });

    for (const testCase of testCases) {
      if (!seen.has(testCase.testName)) {
        seen.add(testCase.testName);
        categories.push(testCase.testName);
      }
    }

    return {
      submission,
      byName: new Map(
        testCases.map((testCase) => [testCase.testName, testCase])
      ),
    };
  });

  return {
    categories,
    yLabel:
      metricMode === "runtime"
        ? "Runtime (ms)"
        : metricMode === "gflops"
          ? "GFLOPS"
          : "Temperature (°C)",
    series: normalized.map(({ submission, byName }, index) => ({
      id: submission.id,
      label: buildSubmissionLabel(submission),
      color: seriesColors[index % seriesColors.length]!,
      points: categories.map((category) => {
        const testCase = byName.get(category);
        if (!testCase) return null;
        if (metricMode === "runtime") return testCase.avgRuntimeMs ?? null;
        if (metricMode === "gflops") return testCase.avgGflops ?? null;

        const runTemps = (testCase.runs ?? [])
          .map((run) => run.gpuMetrics?.temp_c_mean ?? null)
          .filter((value): value is number => value != null);

        if (runTemps.length === 0) return null;

        return (
          runTemps.reduce((sum, value) => sum + value, 0) / runTemps.length
        );
      }),
    })),
  };
};

const buildTimelineChart = (
  submissions: AnalysisSubmission[],
  metricMode: MetricMode,
  seriesColors: string[]
): {
  categories: string[];
  series: ChartSeries[];
  yLabel: string;
} => {
  const ordered = submissions
    .slice()
    .sort(
      (a, b) =>
        new Date(a.createdAt).getTime() - new Date(b.createdAt).getTime()
    );

  return {
    categories: ordered.map((submission) =>
      format(new Date(submission.createdAt), "MMM d")
    ),
    yLabel:
      metricMode === "runtime"
        ? "Average Runtime (ms)"
        : metricMode === "gflops"
          ? "Average GFLOPS"
          : "Average Temperature (°C)",
    series: [
      {
        id: "timeline",
        label: "Selected submissions",
        color: seriesColors[0] ?? SERIES_COLORS[0]!,
        points: ordered.map((submission) => {
          if (metricMode === "runtime") return submission.runtime ?? null;
          if (metricMode === "gflops") return submission.gflops ?? null;
          return getAverageTemperatureForSubmission(submission);
        }),
      },
    ],
  };
};

const DataLineChart = ({
  categories,
  series,
  yLabel,
  metricMode,
  backgroundMode,
}: {
  categories: string[];
  series: ChartSeries[];
  yLabel: string;
  metricMode: MetricMode;
  backgroundMode: BackgroundMode;
}) => {
  const width = 920;
  const height = 420;
  const padding = { top: 26, right: 24, bottom: 86, left: 72 };
  const plotWidth = width - padding.left - padding.right;
  const plotHeight = height - padding.top - padding.bottom;

  const values = series.flatMap((entry) =>
    entry.points.filter((point): point is number => point != null)
  );

  if (categories.length === 0 || values.length === 0) {
    return (
      <Flex
        h="420px"
        align="center"
        justify="center"
        borderRadius="18px"
        border="1px solid"
        borderColor={SURFACE_BORDER}
        bg={SURFACE_ELEVATED}
      >
        <Text color={SURFACE_MUTED}>
          Select accepted submissions with benchmark data to draw the chart.
        </Text>
      </Flex>
    );
  }

  const minValue = Math.min(...values);
  const maxValue = Math.max(...values);
  const range = Math.max(maxValue - minValue, maxValue * 0.1, 1);
  const domainMin = Math.max(0, minValue - range * 0.1);
  const domainMax = maxValue + range * 0.15;
  const xStep =
    categories.length === 1 ? 0 : plotWidth / (categories.length - 1);
  const yTicks = 5;
  const xLabelEvery = Math.max(1, Math.ceil(categories.length / 8));

  const pointX = (index: number) =>
    padding.left + (categories.length === 1 ? plotWidth / 2 : index * xStep);
  const pointY = (value: number) =>
    padding.top +
    plotHeight -
    ((value - domainMin) / Math.max(domainMax - domainMin, 1)) * plotHeight;

  const chartBackground =
    backgroundMode === "plain"
      ? SURFACE_CARD
      : backgroundMode === "spotlight"
        ? "radial-gradient(circle at 50% 40%, rgba(16, 185, 129, 0.10), rgba(15, 23, 42, 0.96) 58%)"
        : SURFACE_ELEVATED;

  const chartOverlay =
    backgroundMode === "grid"
      ? {
          background:
            "linear-gradient(90deg, rgba(255,255,255,0.03) 1px, transparent 1px), linear-gradient(rgba(255,255,255,0.03) 1px, transparent 1px)",
          backgroundSize: "26px 26px",
        }
      : undefined;

  return (
    <Box
      borderRadius="18px"
      border="1px solid"
      borderColor={SURFACE_BORDER}
      bg={chartBackground}
      p={5}
      overflowX="auto"
      boxShadow="inset 0 1px 0 rgba(255,255,255,0.03)"
      position="relative"
      _before={
        chartOverlay
          ? {
              content: '""',
              position: "absolute",
              inset: 0,
              pointerEvents: "none",
              opacity: 1,
              ...chartOverlay,
            }
          : undefined
      }
    >
      <svg
        viewBox={`0 0 ${width} ${height}`}
        width="100%"
        height="420"
        role="img"
        aria-label="Submission comparison chart"
        style={{ position: "relative", zIndex: 1 }}
      >
        <rect x="0" y="0" width={width} height={height} fill="transparent" />

        {Array.from({ length: yTicks }).map((_, index) => {
          const value =
            domainMin +
            ((domainMax - domainMin) * (yTicks - 1 - index)) / (yTicks - 1);
          const y = padding.top + (plotHeight * index) / (yTicks - 1);
          return (
            <g key={`y-${index}`}>
              <line
                x1={padding.left}
                y1={y}
                x2={width - padding.right}
                y2={y}
                stroke={SURFACE_GRID}
                strokeWidth="1"
              />
              <text
                x={padding.left - 12}
                y={y + 4}
                textAnchor="end"
                fontSize="12"
                fill={SURFACE_MUTED}
              >
                {metricMode === "runtime"
                  ? value < 1
                    ? `${(value * 1000).toFixed(0)}μs`
                    : `${value.toFixed(2)}ms`
                  : metricMode === "gflops"
                    ? value.toFixed(1)
                    : `${value.toFixed(1)}°`}
              </text>
            </g>
          );
        })}

        <line
          x1={padding.left}
          y1={padding.top + plotHeight}
          x2={width - padding.right}
          y2={padding.top + plotHeight}
          stroke={SURFACE_TEXT}
          strokeWidth="1.75"
        />
        <line
          x1={padding.left}
          y1={padding.top}
          x2={padding.left}
          y2={padding.top + plotHeight}
          stroke={SURFACE_TEXT}
          strokeWidth="1.75"
        />

        {categories.map((category, index) => {
          if (index % xLabelEvery !== 0 && index !== categories.length - 1) {
            return null;
          }

          const x = pointX(index);
          return (
            <g key={`${category}-${index}`}>
              <line
                x1={x}
                y1={padding.top + plotHeight}
                x2={x}
                y2={padding.top + plotHeight + 6}
                stroke={SURFACE_TEXT}
                strokeWidth="1"
              />
              <text
                x={x}
                y={height - 24}
                textAnchor="end"
                transform={`rotate(-24 ${x} ${height - 24})`}
                fontSize="12"
                fill={SURFACE_MUTED}
              >
                {category}
              </text>
            </g>
          );
        })}

        {series.map((entry) => {
          const commands = entry.points
            .map((point, index) => {
              if (point == null) return null;
              const x = pointX(index);
              const y = pointY(point);
              return `${index === 0 ? "M" : "L"} ${x} ${y}`;
            })
            .filter((command): command is string => Boolean(command))
            .join(" ");

          return (
            <g key={entry.id}>
              {commands ? (
                <path
                  d={commands}
                  fill="none"
                  stroke={entry.color}
                  strokeWidth="3"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              ) : null}
              {entry.points.map((point, index) =>
                point == null ? null : (
                  <circle
                    key={`${entry.id}-${index}`}
                    cx={pointX(index)}
                    cy={pointY(point)}
                    r="4.5"
                    fill={entry.color}
                    stroke={SURFACE_BG}
                    strokeWidth="2"
                  />
                )
              )}
            </g>
          );
        })}

        <text
          x={padding.left - 52}
          y={padding.top + plotHeight / 2}
          textAnchor="middle"
          transform={`rotate(-90 ${padding.left - 52} ${padding.top + plotHeight / 2})`}
          fontSize="13"
          fill={SURFACE_MUTED}
        >
          {yLabel}
        </text>
      </svg>
    </Box>
  );
};

export const getServerSideProps: GetServerSideProps = async (context) => {
  const session = await auth(context.req, context.res);

  if (!session) {
    return {
      redirect: {
        destination: "/api/auth/signin",
        permanent: false,
      },
    };
  }

  const helpers = createServerSideHelpers({
    router: appRouter,
    ctx: createInnerTRPCContext({ session }),
    transformer: superjson,
  });

  const slug = context.params?.slug as string;

  try {
    await Promise.all([
      helpers.problems.getById.prefetch({ slug }),
      helpers.problems.getAnalysisSubmissions.prefetch({
        problemSlug: slug,
        limit: 18,
      }),
    ]);

    return {
      props: {
        trpcState: helpers.dehydrate(),
        slug,
      },
    };
  } catch (error) {
    console.error(error);
    return { notFound: true };
  }
};

const AnalyzePage: NextPage<{ slug: string }> = ({ slug }) => {
  const toast = useToast();
  const { data: problem } = api.problems.getById.useQuery({ slug });
  const { data: submissions, isLoading } =
    api.problems.getAnalysisSubmissions.useQuery({
      problemSlug: slug,
      limit: 18,
    });
  const [selectedSubmissionIds, setSelectedSubmissionIds] = useState<string[]>(
    []
  );
  const [viewMode, setViewMode] = useState<"testCases" | "timeline">(
    "testCases"
  );
  const [metricMode, setMetricMode] = useState<MetricMode>("runtime");
  const [isDownloadingPlot, setIsDownloadingPlot] = useState(false);
  const backgroundMode: BackgroundMode = "grid";
  const seriesColors = SERIES_COLORS;

  useEffect(() => {
    if (!submissions?.length || selectedSubmissionIds.length > 0) return;
    setSelectedSubmissionIds(
      submissions.slice(0, 3).map((submission) => submission.id)
    );
  }, [submissions, selectedSubmissionIds.length]);

  const selectedSubmissions = useMemo(() => {
    const idSet = new Set(selectedSubmissionIds);
    return (submissions ?? []).filter((submission) => idSet.has(submission.id));
  }, [selectedSubmissionIds, submissions]);

  const chartData = useMemo(
    () =>
      viewMode === "testCases"
        ? buildTestCaseChart(selectedSubmissions, metricMode, seriesColors)
        : buildTimelineChart(selectedSubmissions, metricMode, seriesColors),
    [metricMode, selectedSubmissions, seriesColors, viewMode]
  );

  const matplotlibPayload = useMemo(
    () => ({
      problemSlug: problem?.slug ?? slug,
      title: `${problem?.title ?? slug} Analysis`,
      subtitle: `${selectedSubmissions.length} accepted submission${selectedSubmissions.length === 1 ? "" : "s"} • ${
        viewMode === "testCases"
          ? "test-case comparison"
          : "timeline comparison"
      } • ${
        metricMode === "runtime"
          ? "runtime"
          : metricMode === "gflops"
            ? "GFLOPS"
            : "temperature"
      }`,
      xLabel: viewMode === "testCases" ? "Test case" : "Submission date",
      yLabel: chartData.yLabel,
      metricMode,
      categories: chartData.categories,
      series: chartData.series,
    }),
    [
      chartData.categories,
      chartData.series,
      chartData.yLabel,
      metricMode,
      problem?.slug,
      problem?.title,
      selectedSubmissions.length,
      slug,
      viewMode,
    ]
  );

  const handleToggleSubmission = (submissionId: string) => {
    setSelectedSubmissionIds((current) =>
      current.includes(submissionId)
        ? current.filter((id) => id !== submissionId)
        : [...current, submissionId]
    );
  };

  const handleDownloadCsv = () => {
    if (selectedSubmissions.length === 0) return;

    const csv = buildCombinedBenchmarkCsv(
      selectedSubmissions.map((submission) => ({
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
        problemSlug: problem?.slug ?? slug,
      }),
      csv
    );
  };

  const handleDownloadSvg = async () => {
    if (chartData.categories.length === 0 || chartData.series.length === 0) {
      return;
    }

    setIsDownloadingPlot(true);

    try {
      const response = await fetch("/api/plots/matplotlib", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(matplotlibPayload),
      });

      if (!response.ok) {
        const errorBody = (await response
          .json()
          .catch(() => null)) as PlotExportErrorResponse | null;
        throw new Error(
          errorBody?.details ?? errorBody?.error ?? "Plot export failed"
        );
      }

      const svgBlob = await response.blob();
      const downloadUrl = URL.createObjectURL(svgBlob);
      const anchor = document.createElement("a");
      anchor.href = downloadUrl;
      anchor.download = `${problem?.slug ?? slug}-analysis-plot.svg`;
      document.body.appendChild(anchor);
      anchor.click();
      document.body.removeChild(anchor);
      URL.revokeObjectURL(downloadUrl);
    } catch (error) {
      toast({
        title: "Plot export failed",
        description:
          error instanceof Error
            ? error.message
            : "Could not generate the matplotlib plot.",
        status: "error",
        duration: 5000,
        isClosable: true,
      });
    } finally {
      setIsDownloadingPlot(false);
    }
  };

  return (
    <Layout
      title={problem ? `${problem.title} Analysis` : "Submission Analysis"}
    >
      <Box maxW="7xl" mx="auto" px={{ base: 4, md: 6 }} py={{ base: 6, md: 8 }}>
        <VStack align="stretch" spacing={6}>
          <HStack
            justify="space-between"
            align="center"
            wrap="wrap"
            spacing={3}
          >
            <VStack align="start" spacing={1}>
              <Text
                color={ACCENT}
                letterSpacing="0.08em"
                textTransform="uppercase"
                fontSize="xs"
              >
                Research View
              </Text>
              <Text fontSize={{ base: "2xl", md: "4xl" }} fontWeight="bold">
                {problem?.title ?? slug} Analysis
              </Text>
              <Text color="whiteAlpha.700" maxW="2xl">
                Compare accepted submissions test-by-test, then export the same
                slice to CSV.
              </Text>
            </VStack>
            <HStack spacing={2}>
              <Button
                as={NextLink}
                href={`/problems/${slug}`}
                variant="ghost"
                leftIcon={<FiArrowLeft />}
              >
                Back to Problem
              </Button>
              <Button
                onClick={handleDownloadSvg}
                leftIcon={<Icon as={FiDownload} />}
                variant="ghost"
                borderRadius="md"
                h="38px"
                px={4}
                fontWeight="semibold"
                border="1px solid"
                borderColor={SURFACE_BORDER}
                color="whiteAlpha.900"
                _hover={{
                  bg: "whiteAlpha.100",
                }}
                isDisabled={selectedSubmissions.length === 0}
                isLoading={isDownloadingPlot}
                loadingText="Rendering"
              >
                Download SVG
              </Button>
              <Button
                onClick={handleDownloadCsv}
                leftIcon={<Icon as={FiDownload} />}
                bg={ACCENT_SOFT}
                color={ACCENT}
                borderRadius="md"
                h="38px"
                px={4}
                fontWeight="semibold"
                border="1px solid"
                borderColor={ACCENT_LINE}
                _hover={{
                  bg: "rgba(16, 185, 129, 0.18)",
                  transform: "translateY(-1px)",
                }}
                _active={{
                  bg: "rgba(16, 185, 129, 0.22)",
                }}
                isDisabled={selectedSubmissions.length === 0}
              >
                Download CSV
              </Button>
            </HStack>
          </HStack>

          <Box
            borderRadius="20px"
            bg={`linear-gradient(180deg, ${SURFACE_SOFT} 0%, rgba(15, 23, 42, 0.98) 100%)`}
            border="1px solid"
            borderColor={SURFACE_BORDER}
            px={{ base: 4, md: 6 }}
            py={{ base: 4, md: 6 }}
            color={SURFACE_TEXT}
            position="relative"
            overflow="hidden"
            boxShadow="0 18px 60px rgba(2, 6, 23, 0.28)"
            _before={{
              content: '""',
              position: "absolute",
              inset: 0,
              pointerEvents: "none",
              background:
                "linear-gradient(90deg, rgba(255,255,255,0.02) 1px, transparent 1px), linear-gradient(rgba(255,255,255,0.02) 1px, transparent 1px)",
              backgroundSize: "28px 28px",
              opacity: 0.45,
            }}
          >
            {isLoading ? (
              <Flex align="center" justify="center" minH="480px">
                <Spinner color={ACCENT} />
              </Flex>
            ) : (
              <Flex
                direction={{ base: "column", lg: "row" }}
                gap={6}
                position="relative"
                zIndex={1}
              >
                <VStack
                  align="stretch"
                  spacing={4}
                  w={{ base: "100%", lg: "280px" }}
                  flexShrink={0}
                >
                  <VStack
                    align="stretch"
                    spacing={3}
                    border="1px solid"
                    borderColor={SURFACE_BORDER}
                    borderRadius="16px"
                    bg={SURFACE_CARD}
                    p={4}
                    maxH={{ base: "none", lg: "580px" }}
                    overflowY="auto"
                  >
                    <Text fontSize="lg" fontWeight="bold">
                      Filters
                    </Text>
                    <Text fontSize="sm" color={SURFACE_MUTED}>
                      Accepted submissions
                    </Text>
                    <Divider borderColor={SURFACE_BORDER} />
                    {(submissions ?? []).length === 0 ? (
                      <Text fontSize="sm" color={SURFACE_MUTED}>
                        No accepted submissions yet for this problem.
                      </Text>
                    ) : (
                      submissions?.map((submission) => (
                        <Box
                          key={submission.id}
                          border="1px solid"
                          borderColor={
                            selectedSubmissionIds.includes(submission.id)
                              ? ACCENT_LINE
                              : SURFACE_BORDER
                          }
                          borderRadius="12px"
                          p={3}
                          bg={
                            selectedSubmissionIds.includes(submission.id)
                              ? "rgba(16, 185, 129, 0.08)"
                              : "rgba(255,255,255,0.02)"
                          }
                          cursor="pointer"
                          onClick={() => handleToggleSubmission(submission.id)}
                          _hover={{
                            borderColor: "rgba(16, 185, 129, 0.24)",
                            bg: "rgba(255,255,255,0.04)",
                          }}
                        >
                          <HStack align="start" spacing={3}>
                            <Checkbox
                              isChecked={selectedSubmissionIds.includes(
                                submission.id
                              )}
                              pointerEvents="none"
                              colorScheme="green"
                              mt={1}
                            />
                            <VStack align="start" spacing={0} flex={1}>
                              <Text fontWeight="semibold" lineHeight="1.2">
                                {buildSubmissionLabel(submission)}
                              </Text>
                              <Text fontSize="sm" color={SURFACE_MUTED}>
                                {
                                  LANGUAGE_PROFILE_DISPLAY_NAMES[
                                    submission.language
                                  ]
                                }{" "}
                                •{" "}
                                {GPU_DISPLAY_NAMES[submission.gpuType ?? "T4"]}
                              </Text>
                              <Text fontSize="sm" color={SURFACE_MUTED}>
                                {format(
                                  new Date(submission.createdAt),
                                  "MMM d, yyyy"
                                )}
                              </Text>
                              <Text fontSize="sm" color={SURFACE_MUTED}>
                                {formatRuntime(submission.runtime)} avg
                              </Text>
                            </VStack>
                          </HStack>
                        </Box>
                      ))
                    )}
                  </VStack>
                </VStack>

                <VStack align="stretch" spacing={4} flex={1} minW={0}>
                  <HStack
                    justify="space-between"
                    align="center"
                    wrap="wrap"
                    spacing={3}
                  >
                    <VStack align="start" spacing={1}>
                      <HStack spacing={2}>
                        <Icon as={FiTrendingUp} />
                        <Text fontSize="lg" fontWeight="bold">
                          {viewMode === "testCases"
                            ? metricMode === "runtime"
                              ? "Runtime by test case"
                              : metricMode === "gflops"
                                ? "GFLOPS by test case"
                                : "Temperature by test case"
                            : metricMode === "runtime"
                              ? "Average runtime over time"
                              : metricMode === "gflops"
                                ? "Average GFLOPS over time"
                                : "Average temperature over time"}
                        </Text>
                      </HStack>
                      <Text fontSize="sm" color={SURFACE_MUTED}>
                        Default view mirrors a matplotlib-style comparison of
                        the latest accepted submissions.
                      </Text>
                    </VStack>
                    <HStack
                      bg="rgba(255,255,255,0.04)"
                      borderRadius="999px"
                      border="1px solid"
                      borderColor={SURFACE_BORDER}
                      p={1}
                      spacing={1}
                    >
                      <Button
                        size="sm"
                        borderRadius="999px"
                        variant={viewMode === "testCases" ? "solid" : "ghost"}
                        bg={
                          viewMode === "testCases"
                            ? "rgba(255,255,255,0.12)"
                            : "transparent"
                        }
                        color={
                          viewMode === "testCases"
                            ? SURFACE_TEXT
                            : SURFACE_MUTED
                        }
                        _hover={{
                          bg:
                            viewMode === "testCases"
                              ? "rgba(255,255,255,0.12)"
                              : "rgba(255,255,255,0.06)",
                        }}
                        onClick={() => setViewMode("testCases")}
                      >
                        Test Cases
                      </Button>
                      <Button
                        size="sm"
                        borderRadius="999px"
                        variant={viewMode === "timeline" ? "solid" : "ghost"}
                        bg={
                          viewMode === "timeline"
                            ? "rgba(255,255,255,0.12)"
                            : "transparent"
                        }
                        color={
                          viewMode === "timeline" ? SURFACE_TEXT : SURFACE_MUTED
                        }
                        _hover={{
                          bg:
                            viewMode === "timeline"
                              ? "rgba(255,255,255,0.12)"
                              : "rgba(255,255,255,0.06)",
                        }}
                        onClick={() => setViewMode("timeline")}
                      >
                        Timeline
                      </Button>
                    </HStack>
                  </HStack>

                  <HStack
                    bg="rgba(255,255,255,0.04)"
                    borderRadius="999px"
                    border="1px solid"
                    borderColor={SURFACE_BORDER}
                    p={1}
                    spacing={1}
                    wrap="wrap"
                  >
                    <Button
                      size="sm"
                      borderRadius="999px"
                      variant={metricMode === "runtime" ? "solid" : "ghost"}
                      bg={
                        metricMode === "runtime"
                          ? "rgba(16, 185, 129, 0.16)"
                          : "transparent"
                      }
                      color={metricMode === "runtime" ? ACCENT : SURFACE_MUTED}
                      _hover={{
                        bg:
                          metricMode === "runtime"
                            ? "rgba(16, 185, 129, 0.16)"
                            : "rgba(255,255,255,0.06)",
                      }}
                      onClick={() => setMetricMode("runtime")}
                    >
                      Runtime
                    </Button>
                    <Button
                      size="sm"
                      borderRadius="999px"
                      variant={metricMode === "gflops" ? "solid" : "ghost"}
                      bg={
                        metricMode === "gflops"
                          ? "rgba(16, 185, 129, 0.16)"
                          : "transparent"
                      }
                      color={metricMode === "gflops" ? ACCENT : SURFACE_MUTED}
                      _hover={{
                        bg:
                          metricMode === "gflops"
                            ? "rgba(16, 185, 129, 0.16)"
                            : "rgba(255,255,255,0.06)",
                      }}
                      onClick={() => setMetricMode("gflops")}
                    >
                      GFLOPS
                    </Button>
                    <Button
                      size="sm"
                      borderRadius="999px"
                      variant={metricMode === "temperature" ? "solid" : "ghost"}
                      bg={
                        metricMode === "temperature"
                          ? "rgba(16, 185, 129, 0.16)"
                          : "transparent"
                      }
                      color={
                        metricMode === "temperature" ? ACCENT : SURFACE_MUTED
                      }
                      _hover={{
                        bg:
                          metricMode === "temperature"
                            ? "rgba(16, 185, 129, 0.16)"
                            : "rgba(255,255,255,0.06)",
                      }}
                      onClick={() => setMetricMode("temperature")}
                    >
                      Temperature
                    </Button>
                  </HStack>

                  <DataLineChart
                    categories={chartData.categories}
                    series={chartData.series}
                    yLabel={chartData.yLabel}
                    metricMode={metricMode}
                    backgroundMode={backgroundMode}
                  />

                  <Flex wrap="wrap" gap={2}>
                    {chartData.series.map((entry) => (
                      <HStack
                        key={entry.id}
                        spacing={2}
                        px={3}
                        py={1.5}
                        borderRadius="md"
                        border="1px solid"
                        borderColor={SURFACE_BORDER}
                        bg="rgba(255,255,255,0.04)"
                      >
                        <Box
                          boxSize="10px"
                          borderRadius="full"
                          bg={entry.color}
                          border="1px solid rgba(2, 6, 23, 0.45)"
                        />
                        <Text fontSize="sm" color={SURFACE_MUTED}>
                          {entry.label}
                        </Text>
                      </HStack>
                    ))}
                  </Flex>
                </VStack>
              </Flex>
            )}
          </Box>
        </VStack>
      </Box>
    </Layout>
  );
};

export default AnalyzePage;
