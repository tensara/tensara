import { useState, useCallback } from "react";
import { useToast } from "@chakra-ui/react";
import {
  type SubmissionStatusType,
  type SubmissionErrorType,
  type SubmissionResponse,
  type ErrorResponse,
  type WrongAnswerResponse,
  type CheckedResponse,
  type BenchmarkedResponse,
  type TestResultResponse,
  type BenchmarkResultResponse,
  SubmissionError,
  SubmissionStatus,
  type AcceptedResponse,
} from "~/types/submission";

// Define a type mapping from status to response type
type ResponseTypeMap = {
  [SubmissionStatus.ACCEPTED]: AcceptedResponse;
  [SubmissionStatus.BENCHMARKED]: BenchmarkedResponse;
  [SubmissionStatus.CHECKED]: CheckedResponse;
  [SubmissionStatus.WRONG_ANSWER]: WrongAnswerResponse;
  [SubmissionError.ERROR]: ErrorResponse;
  [SubmissionError.COMPILE_ERROR]: ErrorResponse;
  [SubmissionError.RUNTIME_ERROR]: ErrorResponse;
  [SubmissionError.TIME_LIMIT_EXCEEDED]: ErrorResponse;
  [SubmissionError.RATE_LIMIT_EXCEEDED]: ErrorResponse;
};

export function useSubmissionStream(refetchSubmissions: () => void) {
  const toast = useToast();
  // higher level status and response
  const [metaStatus, setMetaStatus] = useState<
    SubmissionStatusType | SubmissionErrorType | null
  >(null);
  const [metaResponse, setMetaResponse] = useState<
    ResponseTypeMap[keyof ResponseTypeMap] | null
  >(null);
  const [totalTests, setTotalTests] = useState<number>(0);
  const [testResults, setTestResults] = useState<TestResultResponse[]>([]);
  const [benchmarkResults, setBenchmarkResults] = useState<
    BenchmarkResultResponse[]
  >([]);
  const [isSubmitting, setIsSubmitting] = useState<boolean>(false);
  const [isTestCaseTableOpen, setIsTestCaseTableOpen] =
    useState<boolean>(false);
  const [isBenchmarking, setIsBenchmarking] = useState<boolean>(false);

  // Type-safe accessor for metaResponse based on current metaStatus
  const getTypedResponse = useCallback(
    <T extends SubmissionStatusType | SubmissionErrorType>(
      status: T
    ): T extends keyof ResponseTypeMap ? ResponseTypeMap[T] | null : null => {
      if (metaStatus === status) {
        return metaResponse as T extends keyof ResponseTypeMap
          ? ResponseTypeMap[T]
          : never;
      }
      return null as T extends keyof ResponseTypeMap
        ? ResponseTypeMap[T] | null
        : null;
    },
    [metaStatus, metaResponse]
  );

  const handleStreamError = useCallback(
    (error: unknown): void => {
      console.error("[sse] Error:", error);
      setIsSubmitting(false);

      let errorMessage = "Failed to connect to submission service";
      if (error instanceof Error) {
        if (
          error.message.includes("QUIC_PROTOCOL_ERROR") ||
          error.message.includes("network error") ||
          error.message.includes("failed to fetch")
        ) {
          errorMessage =
            "Network protocol error. This may be due to your connection or a server configuration issue. Please try again in a few moments.";
        } else {
          errorMessage = error.message;
        }
      }

      toast({
        title: "Connection Error",
        description: errorMessage,
        status: "error",
        duration: 5000,
        isClosable: true,
      });
    },
    [toast]
  );

  const processEvent = useCallback(
    (frame: string): void => {
      const lines = frame.split(/\r?\n/);
      const dataLine = lines.find((l) => l.startsWith("data: "));
      if (!dataLine) return;

      let data: SubmissionResponse | undefined;
      try {
        data = JSON.parse(dataLine.slice(6).trim()) as SubmissionResponse;
      } catch {
        return;
      }

      const status = data?.status as string | undefined;
      if (!status) return;

      // Switch on status (not on "event:")
      switch (status) {
        case "IN_QUEUE":
          setMetaStatus(SubmissionStatus.IN_QUEUE);
          try {
            const remaining = (
              data as Partial<{ remainingSubmissions: number }>
            ).remainingSubmissions;
            if (
              remaining === 50 ||
              remaining === 25 ||
              (remaining !== undefined && remaining <= 10)
            ) {
              toast({
                title: "Submissions Remaining",
                description: `You have ${remaining} submission${remaining !== 1 ? "s" : ""} left today.`,
                status: remaining <= 10 ? "warning" : "info",
                duration: 5000,
                isClosable: true,
              });
            }
          } catch {}
          break;

        case "CHECKING":
        case "BENCHMARKING":
          setMetaStatus(status as SubmissionStatusType);
          break;

        case "TEST_RESULT":
          setTestResults((prev) => [...prev, data as TestResultResponse]);
          setTotalTests((data as TestResultResponse).total_tests);
          break;

        case "CHECKED":
          setMetaStatus(SubmissionStatus.CHECKED);
          setMetaResponse(data as CheckedResponse);
          refetchSubmissions();
          break;

        case "WRONG_ANSWER":
          setIsSubmitting(false);
          setMetaStatus(SubmissionStatus.WRONG_ANSWER);
          setMetaResponse(data as WrongAnswerResponse);
          refetchSubmissions();
          break;

        case "BENCHMARK_RESULT":
          setIsBenchmarking(true);
          setIsTestCaseTableOpen(true);
          setBenchmarkResults((prev) => [
            ...prev,
            data as BenchmarkResultResponse,
          ]);
          setTotalTests((data as BenchmarkResultResponse).total_tests);
          break;

        case "BENCHMARKED": {
          const bench = data as BenchmarkedResponse;

          // Mark stream done
          setIsSubmitting(false);
          setMetaStatus(SubmissionStatus.BENCHMARKED);
          setMetaResponse(bench);
          refetchSubmissions();

          // Synthesize ACCEPTED for UI (works whether or not server sends it)
          const accepted: AcceptedResponse = {
            status: SubmissionStatus.ACCEPTED,
            avg_runtime_ms: bench.avg_runtime_ms ?? undefined,
            avg_gflops: bench.avg_gflops ?? undefined,
            // If your AcceptedResponse supports passing the per-test results:
            benchmark_results: benchmarkResults, // <-- from hook state
            total_tests: benchmarkResults.length,
          };

          setMetaStatus(SubmissionStatus.ACCEPTED);
          setMetaResponse(accepted);
          break;
        }

        case "ACCEPTED":
          setMetaStatus(SubmissionStatus.ACCEPTED);
          setMetaResponse(data as AcceptedResponse);
          refetchSubmissions();
          break;

        case "ERROR":
        case "COMPILE_ERROR":
        case "RUNTIME_ERROR":
        case "TIME_LIMIT_EXCEEDED":
        case "RATE_LIMIT_EXCEEDED":
          setMetaStatus(status as SubmissionErrorType);
          setMetaResponse(data as ErrorResponse);
          toast({
            title: "Submission Error",
            description:
              (data as Partial<ErrorResponse>).message ??
              "An error occurred during submission",
            status: "error",
            duration: 5000,
            isClosable: true,
          });
          refetchSubmissions();
          break;

        default:
          // ignore unknown
          break;
      }
    },
    [refetchSubmissions, toast, benchmarkResults]
  );

  const processEventStream = useCallback(
    async (reader: ReadableStreamDefaultReader<Uint8Array>): Promise<void> => {
      const decoder = new TextDecoder();
      let buffer = "";

      const MAX_RETRIES = 3;
      let retryCount = 0;

      const attemptConnection = async (): Promise<void> => {
        try {
          while (true) {
            const { done, value } = await reader.read();
            if (done) {
              console.log("[sse] Stream complete");
              break;
            }

            // Reset retry count on successful read
            retryCount = 0;

            const chunk = decoder.decode(value, { stream: true });
            buffer += chunk;

            // Process complete events in buffer
            const events = buffer.split(/\r?\n\r?\n/);

            buffer = events.pop() ?? "";

            for (const event of events) {
              if (!event) continue;
              processEvent(event);
            }
          }

          // If we reach here, the stream ended normally
          setIsSubmitting(false);
          refetchSubmissions(); // Ensure we refetch submissions when complete
        } catch (error) {
          console.error("[sse] Stream error:", error);

          // Attempt retry if we haven't exceeded max retries
          if (retryCount < MAX_RETRIES) {
            retryCount++;
            console.log(
              `[sse] Retrying connection (${retryCount}/${MAX_RETRIES})...`
            );

            // Wait before retrying (exponential backoff)
            await new Promise((resolve) =>
              setTimeout(resolve, 1000 * Math.pow(2, retryCount))
            );

            return attemptConnection();
          } else {
            throw error; // Re-throw if max retries exceeded
          }
        }
      };

      await attemptConnection();
    },
    [processEvent, refetchSubmissions]
  );

  const processSubmission = useCallback(
    async (submissionData: {
      code: string;
      language: string;
      gpuType: string;
      problemSlug: string;
    }) => {
      try {
        const response = await fetch("/api/submissions/direct-submit", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(submissionData),
          cache: "no-store",
          credentials: "same-origin",
          keepalive: true,
          signal: AbortSignal.timeout(300000),
        });

        if (!response.ok) {
          if (response.status === 429) {
            const errorMessage = (await response.json()) as ErrorResponse;
            setIsSubmitting(false);
            setMetaStatus(SubmissionError.RATE_LIMIT_EXCEEDED);
            setMetaResponse({
              status: SubmissionError.RATE_LIMIT_EXCEEDED,
              message: errorMessage.message || "Rate limit exceeded",
              details: errorMessage.details || "Rate limit exceeded",
            });

            // Show toast for rate limit error
            toast({
              title: "Submission Error",
              description: errorMessage.message || "Rate limit exceeded",
              status: "error",
              duration: 5000,
              isClosable: true,
            });

            return;
          } else {
            throw new Error(`Direct submit API returned ${response.status}`);
          }
        }

        const reader = response.body?.getReader();
        if (!reader) {
          throw new Error("No response body from direct-submit");
        }

        await processEventStream(reader);
      } catch (error) {
        handleStreamError(error);
      }
    },
    [handleStreamError, processEventStream, toast]
  );

  const startSubmission = useCallback(() => {
    // Reset state for a new submission
    setIsTestCaseTableOpen(false);
    setIsBenchmarking(false);
    setIsSubmitting(true);
    setMetaStatus(SubmissionStatus.IN_QUEUE);
    setTestResults([]);
    setBenchmarkResults([]);
    setTotalTests(0);
  }, []);

  return {
    isSubmitting,
    metaStatus,
    metaResponse,
    testResults,
    benchmarkResults,
    isTestCaseTableOpen,
    isBenchmarking,
    setIsTestCaseTableOpen,
    processSubmission,
    startSubmission,
    setMetaStatus,
    totalTests,
    getTypedResponse,
  };
}
