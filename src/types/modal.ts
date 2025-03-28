const SubmissionStatus = {
  QUEUING: "QUEUING", 
  COMPILING: "COMPILING",
  CHECKING: "CHECKING",
  BENCHMARKING: "BENCHMARKING",
  ACCEPTED: "ACCEPTED",
} as const;

const SubmissionError = {
  COMPILE_ERROR: "Compile Error", 
  WRONG_ANSWER: "Wrong Answer",
  RUNTIME_ERROR: "Runtime Error",
  TIME_LIMIT_EXCEEDED: "Time Limit Exceeded",
  ERROR: "Error", 
} as const;

