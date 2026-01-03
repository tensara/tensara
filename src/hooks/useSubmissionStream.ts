/**
 * useSubmissionStream.ts
 *
 * React hook for consuming `/api/submissions/direct-submit` SSE events.
 * - Opens a streaming POST request and incrementally updates client state for
 *   each event (`IN_QUEUE`, `CHECKING`, `TEST_RESULT`, `WRONG_ANSWER`,
 *   `BENCHMARK_RESULT`, `BENCHMARKED`, `ACCEPTED`, or error).
 * - Maintains high-level submission state (`metaStatus`, `metaResponse`,
 *   `isSubmitting`, `isBenchmarking`, `testResults`, `benchmarkResults`).
 * - Provides safe event parsing, retry/backoff, and toast-based notifications.
 * - Exposes helpers: `startSubmission()`, `processSubmission()`,
 *   `getTypedResponse()`, and `setIsTestCaseTableOpen()`.
 *
 * Primary frontend state manager for Tensara’s live submission results.
 */

import { useState, useCallback, useRef, useEffect } from "react";
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
import { saveAmdRunTimestamp } from "~/utils/amdVmStatus";

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
  const [submissionId, setSubmissionId] = useState<string | null>(null);
  const [isTestCaseTableOpen, setIsTestCaseTableOpen] =
    useState<boolean>(false);
  const [isBenchmarking, setIsBenchmarking] = useState<boolean>(false);
  const [ptxContent, setPtxContent] = useState<string | null>(null);
  const [sassContent, setSassContent] = useState<string | null>(null);

  // Track if current submission is AMD (for saving warm VM timestamp)
  const isAmdSubmissionRef = useRef<boolean>(false);

  // AMD provisioning elapsed time tracking
  const [provisioningStartTime, setProvisioningStartTime] = useState<
    number | null
  >(null);
  const [elapsedSeconds, setElapsedSeconds] = useState<number>(0);
  const elapsedIntervalRef = useRef<NodeJS.Timeout | null>(null);

  // Heartbeat tracking for connection status
  const [lastHeartbeat, setLastHeartbeat] = useState<number | null>(null);

  // AbortController for cancellation support
  const abortControllerRef = useRef<AbortController | null>(null);

  // Update elapsed seconds when provisioning
  useEffect(() => {
    if (provisioningStartTime !== null) {
      // Start interval to update elapsed time
      elapsedIntervalRef.current = setInterval(() => {
        const elapsed = Math.floor((Date.now() - provisioningStartTime) / 1000);
        setElapsedSeconds(elapsed);
      }, 1000);

      return () => {
        if (elapsedIntervalRef.current) {
          clearInterval(elapsedIntervalRef.current);
          elapsedIntervalRef.current = null;
        }
      };
    } else {
      // Clear interval when not provisioning
      if (elapsedIntervalRef.current) {
        clearInterval(elapsedIntervalRef.current);
        elapsedIntervalRef.current = null;
      }
      setElapsedSeconds(0);
    }
  }, [provisioningStartTime]);

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
      // Don't show error for intentional abort (user cancelled)
      if (error instanceof Error && error.name === "AbortError") {
        console.log("[SSE] Request aborted (user cancelled or timeout)");
        setIsSubmitting(false);
        return;
      }

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
      const eventLine = lines.find((l) => l.startsWith("event: "));
      if (!dataLine) return;

      let data: SubmissionResponse | undefined;
      try {
        data = JSON.parse(dataLine.slice(6).trim()) as SubmissionResponse;
      } catch {
        console.warn("[SSE] Failed to parse event data");
        return;
      }

      const eventName = eventLine?.slice(7).trim();
      const status = (data as Partial<{ status: string }>)?.status ?? eventName;
      if (!status) return;

      // Track heartbeat events for connection status, but don't log them
      if (eventName === "heartbeat" || status === "heartbeat") {
        setLastHeartbeat(Date.now());
        return;
      }

      console.log(`[SSE] Event received: ${eventName || status}`, {
        status,
        data: JSON.stringify(data).substring(0, 200),
      });

      if (status === "PTX" && "content" in data) {
        setPtxContent((data as { status: string; content: string }).content);
        return;
      }
      if (status === "SASS" && "content" in data) {
        setSassContent((data as { status: string; content: string }).content);
        return;
      }

      switch (status) {
        case "IN_QUEUE":
          console.log(`[SSE] Status changed: null → IN_QUEUE`);
          setMetaStatus(SubmissionStatus.IN_QUEUE);
          try {
            const remaining = (
              data as Partial<{ remainingSubmissions: number }>
            ).remainingSubmissions;
            const newSubmissionId = (data as Partial<{ id: string }>).id;
            if (newSubmissionId) setSubmissionId(newSubmissionId);
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

        case "PROVISIONING":
          console.log(`[SSE] Status changed: ${metaStatus} → PROVISIONING`);
          setMetaStatus("PROVISIONING" as SubmissionStatusType);
          // Start elapsed time tracking for AMD provisioning
          if (provisioningStartTime === null) {
            setProvisioningStartTime(Date.now());
          }
          break;

        case "CHECKING":
        case "BENCHMARKING":
        case "COMPILING":
          console.log(`[SSE] Status changed: ${metaStatus} → ${status}`);
          setMetaStatus(status as SubmissionStatusType);
          // Clear provisioning timer when we move past provisioning
          setProvisioningStartTime(null);
          break;

        case "TEST_RESULT":
          setTestResults((prev) => [...prev, data as TestResultResponse]);
          setTotalTests((data as TestResultResponse).total_tests);
          break;

        case "CHECKED":
          console.log(`[SSE] Status changed: ${metaStatus} → CHECKED`);
          setMetaStatus(SubmissionStatus.CHECKED);
          setMetaResponse(data as CheckedResponse);
          refetchSubmissions();
          break;

        case "WRONG_ANSWER":
          console.log(`[SSE] Status changed: ${metaStatus} → WRONG_ANSWER`);
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
          console.log(`[SSE] Status changed: ${metaStatus} → BENCHMARKED`, {
            avg_runtime_ms: bench.avg_runtime_ms,
            avg_gflops: bench.avg_gflops,
          });

          // Mark stream done
          setIsSubmitting(false);
          setMetaStatus(SubmissionStatus.BENCHMARKED);
          setMetaResponse(bench);
          refetchSubmissions();

          // Save AMD warm VM timestamp on successful completion
          if (isAmdSubmissionRef.current) {
            saveAmdRunTimestamp();
          }

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
          console.log(`[SSE] Status changed: ${metaStatus} → ACCEPTED`);
          setMetaStatus(SubmissionStatus.ACCEPTED);
          setMetaResponse(data as AcceptedResponse);
          refetchSubmissions();
          // Save AMD warm VM timestamp on successful completion
          if (isAmdSubmissionRef.current) {
            saveAmdRunTimestamp();
          }
          break;

        case "ERROR":
        case "COMPILE_ERROR":
        case "RUNTIME_ERROR":
        case "TIME_LIMIT_EXCEEDED":
        case "RATE_LIMIT_EXCEEDED":
          console.error(`[SSE] Error status received: ${status}`, {
            message: (data as ErrorResponse).message,
            details: (data as ErrorResponse).details,
          });
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
          console.log("[SSE] Starting to read event stream");
          while (true) {
            const { done, value } = await reader.read();
            if (done) {
              console.log("[SSE] Stream completed successfully");
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
          console.error("[SSE] Stream error occurred:", error);

          // Attempt retry if we haven't exceeded max retries
          if (retryCount < MAX_RETRIES) {
            retryCount++;
            console.log(
              `[SSE] Retrying connection (${retryCount}/${MAX_RETRIES})...`
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
        // Determine endpoint based on GPU type
        const isAmdGpu =
          submissionData.gpuType === "MI300X" ||
          submissionData.gpuType === "MI210";

        // Track if this is an AMD submission for saving warm VM timestamp
        isAmdSubmissionRef.current = isAmdGpu;

        const endpoint = isAmdGpu
          ? "/api/submissions/benchmark"
          : "/api/submissions/direct-submit";

        console.log(`[SSE] Starting SSE stream for endpoint: ${endpoint}`, {
          problemSlug: submissionData.problemSlug,
          gpuType: submissionData.gpuType,
          language: submissionData.language,
          isAmdGpu,
        });

        // Create AbortController for cancellation support
        abortControllerRef.current = new AbortController();
        const timeoutId = setTimeout(() => {
          abortControllerRef.current?.abort();
        }, 900000); // 15 minute timeout

        const response = await fetch(endpoint, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(submissionData),
          cache: "no-store",
          credentials: "same-origin",
          keepalive: true,
          signal: abortControllerRef.current.signal,
        });

        // Clear timeout once response starts
        clearTimeout(timeoutId);

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
        console.error("[SSE] Error in processSubmission:", error);
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
    setSubmissionId(null);
    // Reset AMD provisioning state
    setProvisioningStartTime(null);
    setElapsedSeconds(0);
    setLastHeartbeat(null);
  }, []);

  /**
   * Cancel the current submission
   * - Aborts the fetch request
   * - Calls the cancel API to clean up backend resources
   * - Resets all state
   * - Shows cancellation toast
   */
  const cancelSubmission = useCallback(async () => {
    console.log("[SSE] Cancelling submission", submissionId);

    // 1. Abort the fetch request (will trigger disconnect handler on server)
    abortControllerRef.current?.abort();

    // 2. Call cancel API if we have a submission ID
    if (submissionId) {
      try {
        await fetch("/api/submissions/cancel", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ submissionId }),
        });
        console.log("[SSE] Cancel API called successfully");
      } catch (e) {
        console.warn("[SSE] Failed to call cancel API:", e);
      }
    }

    // 3. Reset all state
    setIsSubmitting(false);
    setMetaStatus(null);
    setMetaResponse(null);
    setProvisioningStartTime(null);
    setElapsedSeconds(0);
    setLastHeartbeat(null);
    setTestResults([]);
    setBenchmarkResults([]);
    setSubmissionId(null);
    setIsBenchmarking(false);
    setIsTestCaseTableOpen(false);

    // 4. Show cancellation toast
    toast({
      title: "Submission Cancelled",
      description: "Your submission has been cancelled.",
      status: "info",
      duration: 3000,
      isClosable: true,
    });
  }, [submissionId, toast]);

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
    ptxContent,
    sassContent,
    submissionId,
    elapsedSeconds,
    lastHeartbeat,
    cancelSubmission,
  };
}
