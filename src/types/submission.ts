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

export const SubmissionError = {
  COMPILE_ERROR: "COMPILE_ERROR", 
  RUNTIME_ERROR: "RUNTIME_ERROR",
  TIME_LIMIT_EXCEEDED: "TIME_LIMIT_EXCEEDED",
  ERROR: "ERROR", 
  RATE_LIMIT_EXCEEDED: "RATE_LIMIT_EXCEEDED",
} as const;

export type SubmissionStatusType = (typeof SubmissionStatus)[keyof typeof SubmissionStatus];
export type SubmissionErrorType = (typeof SubmissionError)[keyof typeof SubmissionError];

export function isSubmissionError(status: string): status is SubmissionErrorType {
  return Object.values(SubmissionError).includes(status as SubmissionErrorType);
}

export type TestResult = {
  test_id: number,
  name: string,
  status: string,
  debug_info?: string,
}

export type ErrorResponse = {
    status: SubmissionErrorType,
    message: string,
    details: string,
}

export type TestResultResponse = {
  status: "TEST_RESULT",
  result: TestResult,
  total_tests: number,
}

export type CheckedResponse = {
  status: "CHECKED",
  test_results: TestResult[],
  passed_tests: number,
  total_tests: number,
}

export type WrongAnswerResponse = {
  status: "WRONG_ANSWER",
  debug_info: {
    max_difference?: number;
    mean_difference?: number;
    sample_differences?: Record<string, {
      expected: number;
      actual: number;
      diff: number;
    }>;
    message?: string;
  },
  passed_tests: number,
  total_tests: number,
  test_results: TestResult[]
}
  
export type BenchmarkResultResponse = {
  status: "BENCHMARK_RESULT",
  result: {
    name: string,
    test_id: number, 
    gflops: number,
    runtime_ms: number
  },
  total_tests: number
}

export type BenchmarkedResponse = {
  status: "BENCHMARKED",
  test_results: BenchmarkResultResponse[],
  avg_gflops: number,
  avg_runtime_ms: number,
  total_tests: number
}

export type AcceptedResponse = {
  status: "ACCEPTED",
  benchmark_results: BenchmarkResultResponse[],
  avg_gflops: number,
  avg_runtime_ms: number,
  total_tests: number,
}