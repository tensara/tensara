export interface Parameter {
  name: string
  type: string
  const: string
  pointer: string
}

export interface Problem {
  id: string;
  title: string;
  slug: string;
  description: string;
  difficulty: string;
  parameters: Parameter[];
}

export interface BenchmarkTestResult {
  test_id: number;
  runtime_ms: number;
  gflops: number;
  name: string;
}

export type DebugInfo = {
  max_difference?: number;
  mean_difference?: number;
  sample_differences?: Record<string, {
    expected: number;
    actual: number;
    diff: number;
  }>;
  message?: string;
};

export type SubmissionStatusType = 
  | "compiling"
  | "CHECKING"
  | "BENCHMARKING"
  | "ACCEPTED"
  | "WRONG_ANSWER"
  | "ERROR"
  | "SUBMISSIONS";

export interface SubmissionStatus {
  status: SubmissionStatusType;
  runtime: number | null;
  gflops: number | null;
  passedTests: number | null;
  totalTests: number | null;
  message: string | null;
  errorMessage?: string;
  errorDetails?: string;
  benchmarkResults?: BenchmarkTestResult[];
}

export interface Submission {
  id: string;
  status: string | null;
  runtime: number | null;
  gflops: number | null;
  passedTests: number | null;
  totalTests: number | null;
  createdAt: Date;
  problem: {
    title: string;
    slug: string;
  };
  gpuType: string;
}

export type SubmissionEventData = {
  status?: string;
  passedTests?: number;
  totalTests?: number;
  result?: {
    status?: string;
    test_id?: number;
    runtime_ms?: number;
    gflops?: number;
    name?: string;
  };
  runtime?: number;
  gflops?: number;
  error?: string;
  details?: string;
  benchmarkResults?: BenchmarkTestResult[];
};
