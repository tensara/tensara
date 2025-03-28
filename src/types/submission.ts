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

export type BenchmarkResult = {
  name: string;
  test_id: number;
  gflops: number;
  runtime_ms: number;
};

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
  result: BenchmarkResult;
  total_tests: number;
}

export interface BenchmarkedResponse extends BaseResponse<"BENCHMARKED"> {
  test_results: BenchmarkResultResponse[];
  avg_gflops: number;
  avg_runtime_ms: number;
  total_tests: number;
}

export interface AcceptedResponse extends BaseResponse<"ACCEPTED"> {
  benchmark_results: BenchmarkResultResponse[];
  avg_gflops: number;
  avg_runtime_ms: number;
  total_tests: number;
}

// Unified Union Type for API Response
export type SubmissionResponse =
  | ErrorResponse
  | TestResultResponse
  | CheckedResponse
  | WrongAnswerResponse
  | BenchmarkResultResponse
  | BenchmarkedResponse
  | AcceptedResponse;
