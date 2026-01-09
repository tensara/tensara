// Submission Status Constants
export const SubmissionStatus = {
  IN_QUEUE: "IN_QUEUE",
  COMPILING: "COMPILING",
  CHECKING: "CHECKING",
  TEST_RESULT: "TEST_RESULT",
  WRONG_ANSWER: "WRONG_ANSWER",
  CHECKED: "CHECKED",
  BENCHMARKING: "BENCHMARKING",
  BENCHMARK_RESULT: "BENCHMARK_RESULT",
  BENCHMARKED: "BENCHMARKED",
  ACCEPTED: "ACCEPTED",
  SANITY_CHECK_PASSED: "SANITY_CHECK_PASSED",
  PTX: "PTX",
  SASS: "SASS",
  WARNING: "WARNING",
  // Sandbox statuses
  SANDBOX_RUNNING: "SANDBOX_RUNNING",
  SANDBOX_OUTPUT: "SANDBOX_OUTPUT",
  SANDBOX_SUCCESS: "SANDBOX_SUCCESS",
  SANDBOX_ERROR: "SANDBOX_ERROR",
  SANDBOX_TIMEOUT: "SANDBOX_TIMEOUT",
  SANDBOX_OUTPUT_LIMIT: "SANDBOX_OUTPUT_LIMIT",
} as const;

// Submission Errors
export const SubmissionError = {
  COMPILE_ERROR: "COMPILE_ERROR",
  RUNTIME_ERROR: "RUNTIME_ERROR",
  TIME_LIMIT_EXCEEDED: "TIME_LIMIT_EXCEEDED",
  ERROR: "ERROR",
  RATE_LIMIT_EXCEEDED: "RATE_LIMIT_EXCEEDED",
} as const;

export type SubmissionStatusType =
  (typeof SubmissionStatus)[keyof typeof SubmissionStatus];
export type SubmissionErrorType =
  (typeof SubmissionError)[keyof typeof SubmissionError];

export function isSubmissionError(
  status: string
): status is SubmissionErrorType {
  return Object.values(SubmissionError).includes(status as SubmissionErrorType);
}

// Test and Benchmark Result Types
export type TestResult = {
  test_id: number;
  name: string;
  status: string;
  debug_info?: string;
};

export type DebugInfo = {
  max_difference?: number;
  mean_difference?: number;
  sample_differences?: Record<
    string,
    {
      expected: number;
      actual: number;
      diff: number;
    }
  >;
  message?: string;
};

// ===== LEGACY BENCHMARK TYPES (for backward compatibility) =====

/** Legacy benchmark result format (used by LegacySubmission) */
export type LegacyBenchmarkResult = {
  name: string;
  test_id: number;
  gflops?: number;
  runtime_ms: number;
};

/** @deprecated Use LegacyBenchmarkResult for old data, or TestResultWithRuns for new */
export type BenchmarkResult = LegacyBenchmarkResult;

// ===== GPU MONITORING TYPES =====

/** GPU clock throttle reason bitmask values */
export const ThrottleReasons = {
  NONE: 0,
  GPU_IDLE: 0x0000000000000001,
  APPLICATIONS_CLOCKS_SETTING: 0x0000000000000002,
  SW_POWER_CAP: 0x0000000000000004,
  HW_SLOWDOWN: 0x0000000000000008,
  SYNC_BOOST: 0x0000000000000010,
  SW_THERMAL_SLOWDOWN: 0x0000000000000020,
  HW_THERMAL_SLOWDOWN: 0x0000000000000040,
  HW_POWER_BRAKE_SLOWDOWN: 0x0000000000000080,
  DISPLAY_CLOCK_SETTING: 0x0000000000000100,
} as const;

/** Single GPU sample collected every 5ms during kernel execution */
export interface GPUSample {
  /** Unix timestamp in seconds */
  timestamp: number;
  /** Streaming multiprocessor clock in MHz */
  sm_clock_mhz: number;
  /** Memory clock in MHz */
  mem_clock_mhz: number;
  /** GPU temperature in Celsius */
  temp_c: number;
  /** Power draw in Watts */
  power_w: number;
  /** GPU compute utilization percentage (0-100) */
  utilization_gpu_pct: number;
  /** Memory bandwidth utilization percentage (0-100) */
  utilization_memory_pct: number;
  /** Performance state (0 = max performance, higher = lower performance) */
  pstate: number;
  /** Bitmask of active throttle reasons (see ThrottleReasons) */
  throttle_reasons: number;
}

/** Aggregated GPU metrics statistics for a single benchmark run */
export interface GPUMetricsStats {
  /** Number of samples collected during the run */
  sample_count: number;

  // Temperature stats
  temp_c_min: number;
  temp_c_max: number;
  temp_c_mean: number;

  // SM Clock stats
  sm_clock_mhz_min: number;
  sm_clock_mhz_max: number;
  sm_clock_mhz_mean: number;

  // Memory Clock stats
  mem_clock_mhz_min: number;
  mem_clock_mhz_max: number;
  mem_clock_mhz_mean: number;

  // Power stats
  power_w_min: number;
  power_w_max: number;
  power_w_mean: number;

  // Utilization averages
  utilization_gpu_pct_mean: number;
  utilization_memory_pct_mean: number;

  /** OR of all throttle reasons seen during the run */
  throttle_reasons_any: number;
}

/** Single benchmark iteration result with GPU monitoring data */
export interface BenchmarkRunResult {
  /** 0-indexed iteration number */
  run_index: number;
  /** Runtime for this specific iteration in milliseconds */
  runtime_ms: number;
  /** GFLOPS for this iteration (calculated if problem provides FLOP count) */
  gflops?: number;
  /** Raw GPU samples collected during this run (every 5ms) */
  gpu_samples: GPUSample[];
  /** Aggregated GPU metrics for this run */
  gpu_metrics: GPUMetricsStats;
}

/** Per-test-case result with all benchmark iterations */
export interface TestResultWithRuns {
  /** 1-indexed test number */
  test_id: number;
  /** Test case name (e.g., "small_128x128") */
  name: string;
  /** Average runtime across all iterations in milliseconds */
  avg_runtime_ms: number;
  /** Average GFLOPS across all iterations (if calculable) */
  avg_gflops?: number;
  /** Individual benchmark iterations */
  runs: BenchmarkRunResult[];
}

// Response Types
interface BaseResponse<T extends SubmissionStatusType | SubmissionErrorType> {
  status: T;
}

export interface ErrorResponse extends BaseResponse<SubmissionErrorType> {
  message: string;
  details: string;
}

export interface TestResultResponse extends BaseResponse<"TEST_RESULT"> {
  result: TestResult;
  total_tests: number;
}

export interface CheckedResponse extends BaseResponse<"CHECKED"> {
  test_results: TestResult[];
  passed_tests: number;
  total_tests: number;
}

export interface WrongAnswerResponse extends BaseResponse<"WRONG_ANSWER"> {
  debug_info: DebugInfo;
  passed_tests: number;
  total_tests: number;
  test_results: TestResult[];
}

export interface BenchmarkResultResponse
  extends BaseResponse<"BENCHMARK_RESULT"> {
  /** New format: full test result with per-run data */
  result: TestResultWithRuns;
  total_tests: number;
}

/** @deprecated Use BenchmarkResultResponse with TestResultWithRuns */
export interface LegacyBenchmarkResultResponse
  extends BaseResponse<"BENCHMARK_RESULT"> {
  result: LegacyBenchmarkResult;
  total_tests: number;
}

export interface BenchmarkedResponse extends BaseResponse<"BENCHMARKED"> {
  /** New format: full test results with per-run data */
  test_results: TestResultWithRuns[];
  avg_gflops?: number;
  avg_runtime_ms: number;
  total_tests: number;
}

/** @deprecated Use BenchmarkedResponse with TestResultWithRuns[] */
export interface LegacyBenchmarkedResponse extends BaseResponse<"BENCHMARKED"> {
  test_results: LegacyBenchmarkResultResponse[];
  avg_gflops?: number;
  avg_runtime_ms: number;
  total_tests: number;
}

export interface AcceptedResponse extends BaseResponse<"ACCEPTED"> {
  /** New format: full benchmark results with per-run data */
  benchmark_results: TestResultWithRuns[];
  avg_gflops?: number;
  avg_runtime_ms: number;
  total_tests: number;
}

/** @deprecated Use AcceptedResponse with TestResultWithRuns[] */
export interface LegacyAcceptedResponse extends BaseResponse<"ACCEPTED"> {
  benchmark_results: LegacyBenchmarkResultResponse[];
  avg_gflops?: number;
  avg_runtime_ms: number;
  total_tests: number;
}

// Sandbox Response Types
export interface SandboxOutputResponse extends BaseResponse<"SANDBOX_OUTPUT"> {
  stream: "stdout" | "stderr";
  line: string;
  timestamp: number;
}

export interface SandboxSuccessResponse
  extends BaseResponse<"SANDBOX_SUCCESS"> {
  stdout: string;
  stderr: string;
  return_code: number;
}

export interface SandboxErrorResponse extends BaseResponse<"SANDBOX_ERROR"> {
  message: string;
  stdout?: string;
  stderr?: string;
  return_code?: number;
  details?: string;
}

export interface SandboxTimeoutResponse
  extends BaseResponse<"SANDBOX_TIMEOUT"> {
  message: string;
  details: string;
}

export interface SandboxOutputLimitResponse
  extends BaseResponse<"SANDBOX_OUTPUT_LIMIT"> {
  message: string;
  stdout?: string;
  stderr?: string;
  details?: string;
}

// Unified Union Type for API Response
export type SubmissionResponse =
  | ErrorResponse
  | TestResultResponse
  | CheckedResponse
  | WrongAnswerResponse
  | BenchmarkResultResponse
  | BenchmarkedResponse
  | AcceptedResponse
  | SandboxOutputResponse
  | SandboxSuccessResponse
  | SandboxErrorResponse
  | SandboxTimeoutResponse
  | SandboxOutputLimitResponse;

// Sample Run Status Constants
export const SampleStatus = {
  IDLE: "IDLE",
  IN_QUEUE: "IN_QUEUE",
  COMPILING: "COMPILING",
  RUNNING: "RUNNING",
  PASSED: "PASSED",
  FAILED: "FAILED",
  ERROR: "ERROR",
  COMPILE_ERROR: "COMPILE_ERROR",
  TIME_LIMIT_EXCEEDED: "TIME_LIMIT_EXCEEDED",
  RUNTIME_ERROR: "RUNTIME_ERROR",
  TOO_MANY_REQUESTS: "TOO_MANY_REQUESTS",
  PTX: "PTX",
  SASS: "SASS",
  WARNING: "WARNING",
} as const;

// Sandbox Status Constants
export const SandboxStatus = {
  IN_QUEUE: "IN_QUEUE",
  COMPILING: "COMPILING",
  PTX: "PTX",
  SASS: "SASS",
  WARNING: "WARNING",
  SANDBOX_RUNNING: "SANDBOX_RUNNING",
  SANDBOX_OUTPUT: "SANDBOX_OUTPUT",
  SANDBOX_SUCCESS: "SANDBOX_SUCCESS",
  SANDBOX_ERROR: "SANDBOX_ERROR",
  SANDBOX_TIMEOUT: "SANDBOX_TIMEOUT",
  SANDBOX_OUTPUT_LIMIT: "SANDBOX_OUTPUT_LIMIT",
} as const;

export type SandboxStatusType =
  (typeof SandboxStatus)[keyof typeof SandboxStatus];

export type SampleStatusType = (typeof SampleStatus)[keyof typeof SampleStatus];

export type SampleEvent = {
  status: SampleStatusType;
  message?: string;
  details?: string;
  input?: unknown;
  output?: unknown;
  debug_info?: unknown;
  stdout?: string;
  stderr?: string;
  expected_output?: unknown;
};

// Sample Run Errors
export const SampleError = {
  COMPILE_ERROR: "COMPILE_ERROR",
  RUNTIME_ERROR: "RUNTIME_ERROR",
  ERROR: "ERROR",
  TOO_MANY_REQUESTS: "TOO_MANY_REQUESTS",
} as const;

export type SampleResult = {
  status:
    | (typeof SampleStatus)[keyof typeof SampleStatus]
    | (typeof SampleError)[keyof typeof SampleError];
  message?: string;
  details?: string;
  input?: unknown;
  actual_output?: unknown;
  debug_info?: unknown;
  stdout?: string;
  stderr?: string;
  expected_output?: unknown;
};

// Added from Console.tsx
export type SampleOutput = {
  status: SampleStatusType;
  input?: string;
  output?: string;
  stdout?: string;
  stderr?: string;
  message?: string;
  details?: string;
  expected_output?: string;
  ptx?: string;
  sass?: string;
};

// ===== LEADERBOARD TYPES =====

/** Leaderboard view modes */
export const LeaderboardMode = {
  /** Show only legacy submissions (GFLOPS-based ranking) */
  LEGACY: "legacy",
  /** Show only new submissions (runtime-based ranking) */
  NEW: "new",
  /** Show combined view of both legacy and new submissions */
  ALL: "all",
} as const;

export type LeaderboardModeType =
  (typeof LeaderboardMode)[keyof typeof LeaderboardMode];

/** Unified submission entry for leaderboard display */
export interface LeaderboardSubmissionEntry {
  id: string;
  /** Whether this is from LegacySubmission or new Submission */
  isLegacy: boolean;
  username: string | null;
  /** Runtime in milliseconds (primary metric for new, secondary for legacy) */
  runtime: number | null;
  /** GFLOPS (primary metric for legacy, secondary for new) */
  gflops: number | null;
  gpuType: string | null;
  language: string | null;
  createdAt: Date;
  isPublic: boolean;
}

/** Helper to determine if a submission uses new schema */
export function isNewSubmission(
  submission: unknown
): submission is { avgRuntimeMs: number } {
  return (
    typeof submission === "object" &&
    submission !== null &&
    "avgRuntimeMs" in submission
  );
}

/** Helper to check if GPU was throttled during a run */
export function wasThrottled(throttleReasons: number): boolean {
  return (
    throttleReasons !== 0 &&
    throttleReasons !== ThrottleReasons.GPU_IDLE &&
    throttleReasons !== ThrottleReasons.APPLICATIONS_CLOCKS_SETTING
  );
}

/** Get human-readable throttle reason descriptions */
export function getThrottleReasonDescriptions(
  throttleReasons: number
): string[] {
  const descriptions: string[] = [];

  if (throttleReasons & ThrottleReasons.SW_POWER_CAP) {
    descriptions.push("Software power cap");
  }
  if (throttleReasons & ThrottleReasons.HW_SLOWDOWN) {
    descriptions.push("Hardware slowdown");
  }
  if (throttleReasons & ThrottleReasons.SW_THERMAL_SLOWDOWN) {
    descriptions.push("Software thermal slowdown");
  }
  if (throttleReasons & ThrottleReasons.HW_THERMAL_SLOWDOWN) {
    descriptions.push("Hardware thermal slowdown");
  }
  if (throttleReasons & ThrottleReasons.HW_POWER_BRAKE_SLOWDOWN) {
    descriptions.push("Hardware power brake");
  }
  if (throttleReasons & ThrottleReasons.SYNC_BOOST) {
    descriptions.push("Sync boost");
  }

  return descriptions;
}
