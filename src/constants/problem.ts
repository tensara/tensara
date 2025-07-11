import { isSubmissionError } from "~/types/submission";
import { FaCheck, FaTimes, FaExclamationCircle, FaClock } from "react-icons/fa";

export const DEFAULT_LANGUAGE = "cuda";
export const DEFAULT_DATA_TYPE = "float32";

export const PROBLEM_DIFFICULTY_MULTIPLIERS = {
  EASY: 1,
  MEDIUM: 1.5,
  HARD: 3,
} as const;

export const FIRST_SOLVE_BONUS = 15;

export const ADJUSTMENT_FACTOR = 32; // No reasoning for this, i just like 2^5

export const START_RATING = 1000;

export const formatStatus = (status: string | null) => {
  switch (status) {
    case "ACCEPTED":
      return "Accepted";
    case "WRONG_ANSWER":
      return "Wrong Answer";
    case "ERROR":
      return "Error";
    case "CHECKING":
      return "Checking";
    case "BENCHMARKING":
      return "Benchmarking";
    case "COMPILE_ERROR":
      return "Compile Error";
    case "RUNTIME_ERROR":
      return "Runtime Error";
    case "MEMORY_LIMIT_EXCEEDED":
      return "Memory Limit Exceeded";
    case "TIME_LIMIT_EXCEEDED":
      return "Time Limit Exceeded";

    default:
      return status ?? "Unknown";
  }
};

export const getStatusColor = (status: string | null) => {
  switch (status) {
    case "ACCEPTED":
      return "green";
    case "WRONG_ANSWER":
      return "red";
    case "CHECKING":
    case "BENCHMARKING":
      return "blue";
    default:
      if (isSubmissionError(status ?? "")) {
        return "red";
      }
      return "gray";
  }
};
export const getStatusIcon = (status: string | null) => {
  if (status === "ACCEPTED") {
    return FaCheck;
  } else if (status === "WRONG_ANSWER") {
    return FaTimes;
  } else if (isSubmissionError(status ?? "")) {
    return FaExclamationCircle;
  } else {
    return FaClock;
  }
};

export const tags = [
  "convolution",
  "pooling",
  "normalization",
  "activation-function",
  "graphics",
  "fused",
  "loss-function",
  "scalar",
  "vector",
  "matmul",
  "scan",
  "reduction",
] as const;

export const tagNames = {
  convolution: "Convolution",
  pooling: "Pooling",
  normalization: "Normalization",
  "activation-function": "Activations",
  graphics: "Graphics",
  fused: "Fused",
  "loss-function": "Loss",
  scalar: "Scalar",
  vector: "Vector",
  matmul: "Matmul",
  scan: "Scan",
  reduction: "Reduction",
} as const;

export const tagAltNames = {
  convolution: "Convolution",
  pooling: "Pooling",
  normalization: "Normalization",
  "activation-function": "Activation Functions",
  graphics: "Graphics",
  fused: "Fused Operations",
  "loss-function": "Loss Functions",
  scalar: "Scalar",
  vector: "Vector",
  matmul: "Matrix Multiplication",
  scan: "Scan",
  reduction: "Reduction",
} as const;
