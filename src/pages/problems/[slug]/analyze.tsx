import { type NextPage } from "next";
import type { GetServerSideProps } from "next";
import { useEffect, useMemo, useRef, useState } from "react";
import {
  Box,
  Button,
  Checkbox,
  Collapse,
  Divider,
  Flex,
  HStack,
  Icon,
  Image,
  Menu,
  MenuButton,
  MenuItem,
  MenuList,
  Spinner,
  Text,
  Tooltip,
  VStack,
} from "@chakra-ui/react";
import {
  FiArrowLeft,
  FiChevronDown,
  FiChevronLeft,
  FiDownload,
  FiTrendingUp,
} from "react-icons/fi";
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
import {
  LANGUAGE_DISPLAY_NAMES,
  LANGUAGE_PROFILE_DISPLAY_NAMES,
} from "~/constants/language";
import { GPU_DISPLAY_NAMES } from "~/constants/gpu";

type AnalysisSubmission =
  RouterOutputs["problems"]["getAnalysisSubmissions"][number];

type ChartSeries = {
  id: string;
  label: string;
  color: string;
  points: Array<number | null>;
};

type BackgroundMode = "grid" | "plain" | "spotlight";
type AnalysisTab = "compare" | "progress";

const SURFACE_BG = "#0f172a";
const SURFACE_CARD = "#181f2a";
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

const downloadSvgFromElement = (
  svgElement: SVGSVGElement,
  filename: string
) => {
  const serializer = new XMLSerializer();
  const clonedSvg = svgElement.cloneNode(true) as SVGSVGElement;
  clonedSvg.setAttribute("xmlns", "http://www.w3.org/2000/svg");
  clonedSvg.setAttribute("width", "1180");
  clonedSvg.setAttribute("height", "540");

  const svgMarkup = serializer.serializeToString(clonedSvg);
  const blob = new Blob([svgMarkup], {
    type: "image/svg+xml;charset=utf-8",
  });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  document.body.removeChild(anchor);
  URL.revokeObjectURL(url);
};

const LanguageLogo = ({ language }: { language: string | null }) => {
  if (!language) return <Text color={SURFACE_MUTED}>-</Text>;

  const logoMap: Record<
    string,
    { src?: string; emoji?: string; label: string }
  > = {
    cuda: { src: "/cuda-icon.svg", label: "CUDA C++" },
    python: { src: "/triton-logo.png", label: "Triton" },
    mojo: { emoji: "🔥", label: "Mojo" },
    cute: { emoji: "🧩", label: "CuTe DSL" },
    cutile: { emoji: "🧱", label: "CuTile" },
  };

  const logo = logoMap[language];
  if (!logo) {
    return (
      <Tooltip label={LANGUAGE_DISPLAY_NAMES[language] ?? "Unknown"}>
        <Text fontSize="sm" color={SURFACE_MUTED}>
          {language}
        </Text>
      </Tooltip>
    );
  }

  return (
    <Tooltip label={logo.label}>
      <Flex align="center" justify="center">
        {logo.src ? (
          <Image
            src={logo.src}
            alt={logo.label}
            boxSize="20px"
            objectFit="contain"
          />
        ) : (
          <Text fontSize="lg" lineHeight={1}>
            {logo.emoji}
          </Text>
        )}
      </Flex>
    </Tooltip>
  );
};

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

const buildTestCaseChart = (
  submissions: AnalysisSubmission[],
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
    yLabel: "Runtime (ms)",
    series: normalized.map(({ submission, byName }, index) => ({
      id: submission.id,
      label: buildSubmissionLabel(submission),
      color: seriesColors[index % seriesColors.length]!,
      points: categories.map((category) => {
        const testCase = byName.get(category);
        if (!testCase) return null;
        return testCase.avgRuntimeMs ?? null;
      }),
    })),
  };
};

const buildProgressChart = (
  submissions: AnalysisSubmission[],
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

  let runningBest: number | null = null;
  const points = ordered.map((submission) => {
    const currentValue = submission.runtime ?? null;

    if (currentValue == null) return runningBest;

    if (runningBest == null) {
      runningBest = currentValue;
      return runningBest;
    }

    runningBest = Math.min(runningBest, currentValue);

    return runningBest;
  });

  return {
    categories: ordered.map((submission) =>
      format(new Date(submission.createdAt), "MMM d")
    ),
    yLabel: "Cumulative Best Runtime (ms)",
    series: [
      {
        id: "progress",
        label: "Best runtime so far",
        color: seriesColors[0] ?? SERIES_COLORS[0]!,
        points,
      },
    ],
  };
};

const DataLineChart = ({
  categories,
  series,
  yLabel,
  backgroundMode,
  svgRef,
}: {
  categories: string[];
  series: ChartSeries[];
  yLabel: string;
  backgroundMode: BackgroundMode;
  svgRef?: React.Ref<SVGSVGElement>;
}) => {
  const width = 1180;
  const height = 540;
  const padding = { top: 30, right: 28, bottom: 102, left: 76 };
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
      borderRadius="24px"
      bg={chartBackground}
      p={{ base: 4, md: 6 }}
      overflowX="auto"
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
        ref={svgRef}
        viewBox={`0 0 ${width} ${height}`}
        width="100%"
        height="540"
        role="img"
        aria-label="Submission comparison chart"
        style={{ position: "relative", zIndex: 1 }}
      >
        <rect x="0" y="0" width={width} height={height} fill={SURFACE_BG} />

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
                {value < 1
                  ? `${(value * 1000).toFixed(0)}μs`
                  : `${value.toFixed(2)}ms`}
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
  const { data: problem } = api.problems.getById.useQuery({ slug });
  const { data: submissions, isLoading } =
    api.problems.getAnalysisSubmissions.useQuery({
      problemSlug: slug,
      limit: 18,
    });
  const [compareSubmissionIds, setCompareSubmissionIds] = useState<string[]>(
    []
  );
  const [progressSubmissionIds, setProgressSubmissionIds] = useState<string[]>(
    []
  );
  const [activeTab, setActiveTab] = useState<AnalysisTab>("compare");
  const [isSubmissionListOpen, setIsSubmissionListOpen] = useState(false);
  const backgroundMode: BackgroundMode = "grid";
  const seriesColors = SERIES_COLORS;
  const chartSvgRef = useRef<SVGSVGElement | null>(null);

  useEffect(() => {
    if (!submissions?.length) return;

    setCompareSubmissionIds((current) =>
      current.length > 0 ? current : [submissions[0]!.id]
    );
    setProgressSubmissionIds((current) =>
      current.length > 0
        ? current
        : submissions.map((submission) => submission.id)
    );
  }, [submissions]);

  const compareSubmissions = useMemo(() => {
    const idSet = new Set(compareSubmissionIds);
    return (submissions ?? []).filter((submission) => idSet.has(submission.id));
  }, [compareSubmissionIds, submissions]);

  const progressSubmissions = useMemo(() => {
    const idSet = new Set(progressSubmissionIds);
    return (submissions ?? []).filter((submission) => idSet.has(submission.id));
  }, [progressSubmissionIds, submissions]);

  const activeSubmissions = useMemo(
    () => (activeTab === "compare" ? compareSubmissions : progressSubmissions),
    [activeTab, compareSubmissions, progressSubmissions]
  );
  const compareTabTextColor =
    activeTab === "compare" ? SURFACE_TEXT : SURFACE_MUTED;
  const progressTabTextColor =
    activeTab === "progress" ? SURFACE_TEXT : SURFACE_MUTED;
  const highlightedSubmissionIds = useMemo(
    () =>
      new Set(
        activeTab === "compare" ? compareSubmissionIds : progressSubmissionIds
      ),
    [activeTab, compareSubmissionIds, progressSubmissionIds]
  );

  const chartData = useMemo(
    () =>
      activeTab === "compare"
        ? buildTestCaseChart(activeSubmissions, seriesColors)
        : buildProgressChart(activeSubmissions, seriesColors),
    [activeSubmissions, activeTab, seriesColors]
  );

  const handleToggleSubmission = (submissionId: string) => {
    if (activeTab === "compare") {
      setCompareSubmissionIds((current) =>
        current.includes(submissionId)
          ? current.filter((id) => id !== submissionId)
          : [...current, submissionId]
      );
      return;
    }

    setProgressSubmissionIds((current) =>
      current.includes(submissionId)
        ? current.filter((id) => id !== submissionId)
        : [...current, submissionId]
    );
  };

  const handleDownloadCsv = () => {
    if (activeSubmissions.length === 0) return;

    const csv = buildCombinedBenchmarkCsv(
      activeSubmissions.map((submission) => ({
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

  const handleDownloadSvg = () => {
    if (!chartSvgRef.current) return;

    downloadSvgFromElement(
      chartSvgRef.current,
      `${problem?.slug ?? slug}-analysis-plot.svg`
    );
  };

  return (
    <Layout
      title={problem ? `${problem.title} Analysis` : "Submission Analysis"}
    >
      <Box
        maxW="none"
        w="100%"
        px={{ base: 4, md: 6, xl: 10 }}
        py={{ base: 6, md: 8 }}
      >
        <VStack align="stretch" spacing={6}>
          <HStack
            justify="space-between"
            align="center"
            wrap="wrap"
            spacing={3}
          >
            <VStack align="start" spacing={1}>
              <Text fontSize={{ base: "2xl", md: "4xl" }} fontWeight="bold">
                {problem?.title ?? slug} Analysis
              </Text>
              <HStack
                as={NextLink}
                href={`/problems/${slug}`}
                spacing={1.5}
                color="whiteAlpha.700"
                fontSize="sm"
                _hover={{ color: "whiteAlpha.900" }}
              >
                <Icon as={FiArrowLeft} />
                <Text>Back to problem</Text>
              </HStack>
            </VStack>
            <Menu>
              <MenuButton
                as={Button}
                leftIcon={<Icon as={FiDownload} />}
                rightIcon={<FiChevronDown />}
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
                isDisabled={activeSubmissions.length === 0}
              >
                Download
              </MenuButton>
              <MenuList
                bg={SURFACE_CARD}
                borderColor={SURFACE_BORDER}
                color={SURFACE_TEXT}
              >
                <MenuItem
                  bg={SURFACE_CARD}
                  _hover={{ bg: "whiteAlpha.100" }}
                  _focus={{ bg: "whiteAlpha.100" }}
                  onClick={handleDownloadSvg}
                >
                  Download SVG
                </MenuItem>
                <MenuItem
                  bg={SURFACE_CARD}
                  _hover={{ bg: "whiteAlpha.100" }}
                  _focus={{ bg: "whiteAlpha.100" }}
                  onClick={handleDownloadCsv}
                >
                  Download CSV
                </MenuItem>
              </MenuList>
            </Menu>
          </HStack>

          <Box
            borderRadius="0"
            bg="transparent"
            border="none"
            px="0"
            py="0"
            color={SURFACE_TEXT}
            position="relative"
            overflow="visible"
          >
            {isLoading ? (
              <Flex align="center" justify="center" minH="480px">
                <Spinner color={ACCENT} />
              </Flex>
            ) : (
              <Flex
                direction={{ base: "column", lg: "row" }}
                gap={{ base: 4, lg: 5 }}
                position="relative"
                zIndex={1}
                align="stretch"
                minH={{ base: "auto", lg: "calc(100vh - 260px)" }}
              >
                <VStack
                  align="stretch"
                  spacing={4}
                  w={{
                    base: "100%",
                    lg: isSubmissionListOpen ? "292px" : "56px",
                  }}
                  flexShrink={0}
                >
                  <VStack
                    align="stretch"
                    spacing={3}
                    border="1px solid"
                    borderColor={SURFACE_BORDER}
                    borderRadius="16px"
                    bg={SURFACE_CARD}
                    p={isSubmissionListOpen ? 4 : 2}
                    minH={{ base: "auto", lg: "100%" }}
                    justify={isSubmissionListOpen ? "flex-start" : "center"}
                  >
                    {isSubmissionListOpen ? (
                      <HStack justify="space-between" align="center">
                        <VStack align="start" spacing={0}>
                          <Text fontSize="lg" fontWeight="bold">
                            Submissions
                          </Text>
                          <Text fontSize="sm" color={SURFACE_MUTED}>
                            {highlightedSubmissionIds.size} selected
                          </Text>
                        </VStack>
                        <Button
                          size="sm"
                          variant="ghost"
                          borderRadius="md"
                          color={SURFACE_MUTED}
                          rightIcon={<Icon as={FiChevronLeft} />}
                          onClick={() =>
                            setIsSubmissionListOpen((current) => !current)
                          }
                        >
                          Hide
                        </Button>
                      </HStack>
                    ) : (
                      <Button
                        variant="ghost"
                        h="100%"
                        minH={{ base: "auto", lg: "480px" }}
                        px={0}
                        py={4}
                        borderRadius="12px"
                        color={SURFACE_MUTED}
                        _hover={{
                          bg: "rgba(255,255,255,0.04)",
                          color: SURFACE_TEXT,
                        }}
                        onClick={() =>
                          setIsSubmissionListOpen((current) => !current)
                        }
                      >
                        <VStack spacing={4}>
                          <Icon as={FiChevronDown} transform="rotate(-90deg)" />
                          <Text
                            fontSize="xs"
                            fontWeight="semibold"
                            letterSpacing="0.12em"
                            textTransform="uppercase"
                            transform="rotate(180deg)"
                            sx={{ writingMode: "vertical-rl" }}
                          >
                            Submissions
                          </Text>
                          <Box
                            px={2}
                            py={1}
                            borderRadius="full"
                            bg="rgba(16, 185, 129, 0.12)"
                            color={ACCENT}
                            fontSize="xs"
                            fontWeight="semibold"
                          >
                            {highlightedSubmissionIds.size}
                          </Box>
                        </VStack>
                      </Button>
                    )}
                    <Collapse in={isSubmissionListOpen} animateOpacity>
                      <VStack
                        align="stretch"
                        spacing={3}
                        pt={3}
                        maxH={{ base: "none", lg: "calc(100vh - 360px)" }}
                        overflowY="auto"
                      >
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
                                highlightedSubmissionIds.has(submission.id)
                                  ? ACCENT_LINE
                                  : SURFACE_BORDER
                              }
                              borderRadius="12px"
                              p={3}
                              bg={
                                highlightedSubmissionIds.has(submission.id)
                                  ? "rgba(16, 185, 129, 0.08)"
                                  : "rgba(255,255,255,0.02)"
                              }
                              cursor="pointer"
                              onClick={() =>
                                handleToggleSubmission(submission.id)
                              }
                              _hover={{
                                borderColor: "rgba(16, 185, 129, 0.24)",
                                bg: "rgba(255,255,255,0.04)",
                              }}
                            >
                              <HStack align="start" spacing={3}>
                                <Checkbox
                                  isChecked={highlightedSubmissionIds.has(
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
                                  <HStack spacing={2} color={SURFACE_MUTED}>
                                    <LanguageLogo
                                      language={submission.language}
                                    />
                                    <Text fontSize="sm">
                                      {
                                        GPU_DISPLAY_NAMES[
                                          submission.gpuType ?? "T4"
                                        ]
                                      }
                                    </Text>
                                  </HStack>
                                  <Text fontSize="sm" color={SURFACE_MUTED}>
                                    {formatRuntime(submission.runtime)}
                                  </Text>
                                </VStack>
                              </HStack>
                            </Box>
                          ))
                        )}
                      </VStack>
                    </Collapse>
                  </VStack>
                </VStack>

                <VStack
                  align="stretch"
                  spacing={3}
                  flex={1}
                  minW={0}
                  py={{ base: 0, lg: 1 }}
                >
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
                          {activeTab === "compare"
                            ? "Runtime by test case"
                            : "Cumulative best runtime"}
                        </Text>
                      </HStack>
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
                        variant={activeTab === "compare" ? "solid" : "ghost"}
                        bg={
                          activeTab === "compare"
                            ? "rgba(255,255,255,0.12)"
                            : "transparent"
                        }
                        color={compareTabTextColor}
                        _hover={{
                          bg:
                            activeTab === "compare"
                              ? "rgba(255,255,255,0.12)"
                              : "rgba(255,255,255,0.06)",
                        }}
                        onClick={() => setActiveTab("compare")}
                      >
                        Compare
                      </Button>
                      <Button
                        size="sm"
                        borderRadius="999px"
                        variant={activeTab === "progress" ? "solid" : "ghost"}
                        bg={
                          activeTab === "progress"
                            ? "rgba(255,255,255,0.12)"
                            : "transparent"
                        }
                        color={progressTabTextColor}
                        _hover={{
                          bg:
                            activeTab === "progress"
                              ? "rgba(255,255,255,0.12)"
                              : "rgba(255,255,255,0.06)",
                        }}
                        onClick={() => setActiveTab("progress")}
                      >
                        Progress
                      </Button>
                    </HStack>
                  </HStack>

                  <DataLineChart
                    categories={chartData.categories}
                    series={chartData.series}
                    yLabel={chartData.yLabel}
                    backgroundMode={backgroundMode}
                    svgRef={chartSvgRef}
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
                        bg="rgba(255,255,255,0.03)"
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
