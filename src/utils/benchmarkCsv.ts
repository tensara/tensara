import type {
  BenchmarkResultResponse,
  GPUMetricsStats,
} from "~/types/submission";

export interface BenchmarkCsvSubmissionMeta {
  submissionId?: string | null;
  submissionName?: string | null;
  problemSlug?: string | null;
  problemTitle?: string | null;
  language?: string | null;
  gpuType?: string | null;
  createdAt?: string | Date | null;
  overallAvgRuntimeMs?: number | null;
  overallAvgGflops?: number | null;
}

export interface BenchmarkCsvTestCase {
  testId: number;
  testName: string;
  avgRuntimeMs?: number | null;
  avgGflops?: number | null;
  runs?: Array<{
    runtimeMs?: number | null;
    gflops?: number | null;
    gpuMetrics?: GPUMetricsStats | null;
  }>;
}

export interface BenchmarkCsvSubmissionExport {
  submission: BenchmarkCsvSubmissionMeta;
  testCases: BenchmarkCsvTestCase[];
}

const escapeCsvValue = (value: string): string => {
  if (/[",\n]/.test(value)) {
    return `"${value.replace(/"/g, '""')}"`;
  }
  return value;
};

const toCsvCell = (value: string | number | null | undefined): string => {
  if (value == null) return "";
  return escapeCsvValue(String(value));
};

const average = (values: number[]): number | null => {
  if (values.length === 0) return null;
  return values.reduce((sum, value) => sum + value, 0) / values.length;
};

const aggregateGpuMetrics = (
  runs: BenchmarkCsvTestCase["runs"]
): {
  gpuSampleCount: number | null;
  gpuTempMeanC: number | null;
  gpuTempMinC: number | null;
  gpuTempMaxC: number | null;
  gpuSmClockMeanMhz: number | null;
  gpuSmClockMinMhz: number | null;
  gpuSmClockMaxMhz: number | null;
  gpuPstateMin: number | null;
  gpuPstateMax: number | null;
} => {
  const metrics = (runs ?? [])
    .map((run) => run.gpuMetrics)
    .filter((metric): metric is GPUMetricsStats => Boolean(metric));

  if (metrics.length === 0) {
    return {
      gpuSampleCount: null,
      gpuTempMeanC: null,
      gpuTempMinC: null,
      gpuTempMaxC: null,
      gpuSmClockMeanMhz: null,
      gpuSmClockMinMhz: null,
      gpuSmClockMaxMhz: null,
      gpuPstateMin: null,
      gpuPstateMax: null,
    };
  }

  return {
    gpuSampleCount: metrics.reduce(
      (sum, metric) => sum + (metric.sample_count ?? 0),
      0
    ),
    gpuTempMeanC: average(metrics.map((metric) => metric.temp_c_mean)),
    gpuTempMinC: Math.min(...metrics.map((metric) => metric.temp_c_min)),
    gpuTempMaxC: Math.max(...metrics.map((metric) => metric.temp_c_max)),
    gpuSmClockMeanMhz: average(
      metrics.map((metric) => metric.sm_clock_mhz_mean)
    ),
    gpuSmClockMinMhz: Math.min(
      ...metrics.map((metric) => metric.sm_clock_mhz_min)
    ),
    gpuSmClockMaxMhz: Math.max(
      ...metrics.map((metric) => metric.sm_clock_mhz_max)
    ),
    gpuPstateMin: Math.min(...metrics.map((metric) => metric.pstate_min)),
    gpuPstateMax: Math.max(...metrics.map((metric) => metric.pstate_max)),
  };
};

export const normalizeLiveBenchmarkResults = (
  benchmarkResults: BenchmarkResultResponse[]
): BenchmarkCsvTestCase[] =>
  benchmarkResults.map((result) => ({
    testId: result.result.test_id,
    testName: result.result.name,
    avgRuntimeMs: result.result.avg_runtime_ms ?? result.result.runtime_ms,
    avgGflops: result.result.avg_gflops ?? result.result.gflops,
    runs: (result.result.runs ?? []).map((run) => ({
      runtimeMs: run.runtime_ms,
      gflops: run.gflops,
      gpuMetrics: run.gpu_metrics,
    })),
  }));

const BENCHMARK_CSV_HEADER = [
  "submission_id",
  "submission_name",
  "problem_slug",
  "problem_title",
  "language",
  "gpu_type",
  "submitted_at",
  "overall_avg_runtime_ms",
  "overall_avg_gflops",
  "test_id",
  "test_name",
  "test_avg_runtime_ms",
  "test_avg_gflops",
  "run_count",
  "gpu_sample_count",
  "gpu_temp_mean_c",
  "gpu_temp_min_c",
  "gpu_temp_max_c",
  "gpu_sm_clock_mean_mhz",
  "gpu_sm_clock_min_mhz",
  "gpu_sm_clock_max_mhz",
  "gpu_pstate_min",
  "gpu_pstate_max",
];

const buildBenchmarkCsvRows = ({
  submission,
  testCases,
}: BenchmarkCsvSubmissionExport): string[] =>
  testCases
    .slice()
    .sort((a, b) => a.testId - b.testId)
    .map((testCase) => {
      const runs = testCase.runs ?? [];
      const metrics = aggregateGpuMetrics(runs);
      return [
        submission.submissionId,
        submission.submissionName,
        submission.problemSlug,
        submission.problemTitle,
        submission.language,
        submission.gpuType,
        submission.createdAt instanceof Date
          ? submission.createdAt.toISOString()
          : submission.createdAt,
        submission.overallAvgRuntimeMs,
        submission.overallAvgGflops,
        testCase.testId,
        testCase.testName,
        testCase.avgRuntimeMs,
        testCase.avgGflops,
        runs.length,
        metrics.gpuSampleCount,
        metrics.gpuTempMeanC,
        metrics.gpuTempMinC,
        metrics.gpuTempMaxC,
        metrics.gpuSmClockMeanMhz,
        metrics.gpuSmClockMinMhz,
        metrics.gpuSmClockMaxMhz,
        metrics.gpuPstateMin,
        metrics.gpuPstateMax,
      ]
        .map(toCsvCell)
        .join(",");
    });

export const buildBenchmarkCsv = ({
  submission,
  testCases,
}: BenchmarkCsvSubmissionExport): string =>
  [
    BENCHMARK_CSV_HEADER.join(","),
    ...buildBenchmarkCsvRows({ submission, testCases }),
  ].join("\n");

export const buildCombinedBenchmarkCsv = (
  exports: BenchmarkCsvSubmissionExport[]
): string =>
  [
    BENCHMARK_CSV_HEADER.join(","),
    ...exports.flatMap((submissionExport) =>
      buildBenchmarkCsvRows(submissionExport)
    ),
  ].join("\n");

const slugifyFilenamePart = (value: string | null | undefined): string =>
  (value ?? "")
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");

export const buildBenchmarkCsvFilename = (
  submission: BenchmarkCsvSubmissionMeta
): string => {
  const preferredName =
    slugifyFilenamePart(submission.submissionName) ||
    slugifyFilenamePart(submission.problemSlug) ||
    "submission";
  const gpu = slugifyFilenamePart(submission.gpuType) || "gpu";
  const date =
    submission.createdAt instanceof Date
      ? submission.createdAt.toISOString().slice(0, 10)
      : typeof submission.createdAt === "string"
        ? submission.createdAt.slice(0, 10)
        : new Date().toISOString().slice(0, 10);

  return `${preferredName}-${gpu}-benchmarks-${date}.csv`;
};

export const buildCombinedBenchmarkCsvFilename = ({
  problemSlug,
  createdAt,
}: {
  problemSlug?: string | null;
  createdAt?: string | Date | null;
}): string => {
  const slug = slugifyFilenamePart(problemSlug) || "submissions";
  const date =
    createdAt instanceof Date
      ? createdAt.toISOString().slice(0, 10)
      : typeof createdAt === "string"
        ? createdAt.slice(0, 10)
        : new Date().toISOString().slice(0, 10);

  return `${slug}-comparison-${date}.csv`;
};

export const downloadCsv = (filename: string, content: string): void => {
  if (typeof window === "undefined") return;

  const blob = new Blob([content], { type: "text/csv;charset=utf-8;" });
  const url = window.URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.style.display = "none";
  document.body.appendChild(anchor);
  anchor.click();
  document.body.removeChild(anchor);
  window.URL.revokeObjectURL(url);
};

export const normalizeStoredBenchmarkRuns = (
  runs: Array<{
    runtimeMs: number;
    gflops: number | null;
    gpuMetrics: unknown;
  }>
): BenchmarkCsvTestCase["runs"] =>
  runs.map((run) => ({
    runtimeMs: run.runtimeMs,
    gflops: run.gflops ?? undefined,
    gpuMetrics: (run.gpuMetrics as GPUMetricsStats | null) ?? null,
  }));

export const normalizeStoredBenchmarkResults = ({
  benchmarkResults,
  testResults,
}: {
  benchmarkResults: Array<{
    test_id: number;
    name: string;
    runtime_ms?: number;
    avg_runtime_ms?: number;
    gflops?: number;
    avg_gflops?: number;
  }>;
  testResults: Array<{
    testId: number;
    name: string;
    avgRuntimeMs: number;
    avgGflops: number | null;
    runs: Array<{
      runtimeMs: number;
      gflops: number | null;
      gpuMetrics: unknown;
    }>;
  }>;
}): BenchmarkCsvTestCase[] => {
  const runsByTestId = new Map(
    testResults.map((testResult) => [testResult.testId, testResult])
  );

  return benchmarkResults.map((result) => {
    const detailed = runsByTestId.get(result.test_id);

    return {
      testId: result.test_id,
      testName: result.name ?? detailed?.name ?? `Test ${result.test_id}`,
      avgRuntimeMs:
        result.avg_runtime_ms ??
        result.runtime_ms ??
        detailed?.avgRuntimeMs ??
        null,
      avgGflops:
        result.avg_gflops ?? result.gflops ?? detailed?.avgGflops ?? null,
      runs: detailed ? normalizeStoredBenchmarkRuns(detailed.runs) : [],
    };
  });
};
