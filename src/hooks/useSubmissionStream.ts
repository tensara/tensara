import { useState, useCallback } from 'react';
import { useToast } from '@chakra-ui/react';
import { SubmissionStatus, SubmissionEventData, SubmissionStatusType, BenchmarkTestResult } from '~/types/problem';

export function useSubmissionStream(refetchSubmissions: () => void) {
  const toast = useToast();
  const [isSubmitting, setIsSubmitting] = useState<boolean>(false);
  const [submissionStatus, setSubmissionStatus] = useState<SubmissionStatus | null>(null);
  const [submissionId, setSubmissionId] = useState<string | null>(null);
  const [isTestCaseTableOpen, setIsTestCaseTableOpen] = useState<boolean>(false);
  const [isBenchmarking, setIsBenchmarking] = useState<boolean>(false);

  const processSubmission = useCallback(async (id: string) => {
    setSubmissionId(id);
    
    try {
      const response = await fetch("/api/submissions/direct-submit", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ submissionId: id }),
        cache: "no-store",
        credentials: "same-origin",
        keepalive: true,
        signal: AbortSignal.timeout(300000),
      });

      if (!response.ok) {
        if (response.status === 429) {
          const errorMessage = await response.json();
          setIsSubmitting(false);

          setSubmissionStatus({
            status: "ERROR",
            runtime: null,
            gflops: null,
            passedTests: null,
            totalTests: null,
            message: "Rate limit exceeded",
            errorMessage: errorMessage.error || "Rate limit exceeded",
            errorDetails: errorMessage.error || "Rate limit exceeded",
          });

          return;                  
        } else {
          throw new Error(
            `Direct submit API returned ${response.status}`
          );
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
  }, []);

  const processEventStream = async (reader: ReadableStreamDefaultReader<Uint8Array>): Promise<void> => {
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
          const events = buffer.split("\n\n");
          buffer = events.pop() ?? "";

          for (const event of events) {
            if (!event) continue;
            processEvent(event);
          }
        }

        // If we reach here, the stream ended normally
        setIsSubmitting(false);
      } catch (error) {
        console.error("[sse] Stream error:", error);

        // Attempt retry if we haven't exceeded max retries
        if (retryCount < MAX_RETRIES) {
          retryCount++;
          console.log(`[sse] Retrying connection (${retryCount}/${MAX_RETRIES})...`);

          // Wait before retrying (exponential backoff)
          await new Promise(resolve => 
            setTimeout(resolve, 1000 * Math.pow(2, retryCount))
          );

          return attemptConnection();
        } else {
          throw error; // Re-throw if max retries exceeded
        }
      }
    };

    await attemptConnection();
  };

  const processEvent = (event: string): void => {
    const eventLines = event.split("\n");
    let eventType = "message";
    let eventData = "";

    for (const line of eventLines) {
      if (line.startsWith("event: ")) {
        eventType = line.slice(7);
      } else if (line.startsWith("data: ")) {
        eventData = line.slice(6);
      }
    }

    if (!eventData) return;

    try {
      const data = JSON.parse(eventData) as SubmissionEventData;
      console.log(`[sse] ${eventType} event:`, data);

      // Handle different event types
      if (eventType === "status") {
        handleStatusEvent(data);
      } else if (eventType === "checker") {
        handleCheckerEvent(data);
      } else if (eventType === "benchmark") {
        handleBenchmarkEvent(data);
      } else if (eventType === "complete") {
        handleCompleteEvent(data);
      } else if (eventType === "error") {
        handleErrorEvent(data);
      }
    } catch (error) {
      console.error("Error parsing event data:", error, "Raw data:", eventData);
    }
  };
  
  const handleStatusEvent = (data: SubmissionEventData): void => {
    setSubmissionStatus(prev => {
      if (!prev) return prev;
      return {
        ...prev,
        status: (data.status as SubmissionStatusType) ?? prev.status,
        passedTests: data.passedTests ?? prev.passedTests,
        totalTests: data.totalTests ?? prev.totalTests,
        message: "status: " + (data.status ?? prev.message),
      };
    });
  };

  const handleCheckerEvent = (data: SubmissionEventData): void => {
    if (data.status === "test_result" && data.result) {
      setSubmissionStatus(prev => {
        if (!prev) {
          return {
            status: "CHECKING",
            runtime: null,
            gflops: null,
            passedTests: data.result?.status === "PASSED" ? 1 : 0,
            totalTests: data.totalTests ?? 1,
            message: "checker: " + data.status,
          };
        }
        return {
          ...prev,
          passedTests: (prev.passedTests ?? 0) + (data.result?.status === "PASSED" ? 1 : 0),
          totalTests: data.totalTests ?? prev.totalTests,
          message: "checker: " + data.status,
        };
      });
    } else if (data.status && data.status !== "error") {
      setSubmissionStatus(prev => {
        if (!prev) return prev;
        return {
          ...prev,
          message: "checker: " + data.status,
        };
      });
    } else if (data.status === "error") {
      setSubmissionStatus({
        status: "ERROR",
        runtime: null,
        gflops: null,
        passedTests: null,
        totalTests: null,
        message: "checker: " + (data.status ?? "ERROR"),
        errorMessage: data.error ?? undefined,
        errorDetails: data.details ?? undefined,
      });

      setIsSubmitting(false);
      refetchSubmissions();

      toast({
        title: "Submission Error",
        description: data.error ?? "An error occurred during submission",
        status: "error",
        duration: 5000,
        isClosable: true,
      });
    }
  };

  const handleBenchmarkEvent = (data: SubmissionEventData): void => {
    if (data.status === "test_result" && data.result) {
      setIsBenchmarking(true);
      setIsTestCaseTableOpen(true);
      setSubmissionStatus(prev => {
        if (!prev) return prev;
        const benchmarkResult: BenchmarkTestResult = {
          test_id: data.result?.test_id ?? 0,
          runtime_ms: data.result?.runtime_ms ?? 0,
          gflops: data.result?.gflops ?? 0,
          name: data.result?.name ?? `Test ${data.result?.test_id ?? 0}`,
        };
        return {
          ...prev,
          message: "benchmark: " + data.status,
          benchmarkResults: [...(prev.benchmarkResults ?? []), benchmarkResult],
        };
      });
    } else if (data.status && data.status !== "error") {
      setSubmissionStatus(prev => {
        if (!prev) return prev;
        return {
          ...prev,
          message: "benchmark: " + data.status,
        };
      });
    }
  };

  const handleCompleteEvent = (data: SubmissionEventData): void => {
    setSubmissionStatus({
      status: (data.status as SubmissionStatusType) ?? "ERROR",
      runtime: data.runtime ?? null,
      gflops: data.gflops ?? null,
      passedTests: data.passedTests ?? null,
      totalTests: data.totalTests ?? null,
      message: "complete: " + (data.status ?? "ERROR"),
      errorMessage: data.error ?? undefined,
      errorDetails: data.details ?? undefined,
      benchmarkResults: data.benchmarkResults ?? [],
    });

    setIsSubmitting(false);
    refetchSubmissions();
  };

  const handleErrorEvent = (data: SubmissionEventData): void => {
    setSubmissionStatus({
      status: "ERROR",
      runtime: null,
      gflops: null,
      passedTests: null,
      totalTests: null,
      message: "error: " + (data.status ?? "ERROR"),
      errorMessage: data.error ?? undefined,
      errorDetails: data.details ?? undefined,
    });

    setIsSubmitting(false);
    refetchSubmissions();

    toast({
      title: "Submission Error",
      description: data.error ?? "An error occurred during submission",
      status: "error",
      duration: 5000,
      isClosable: true,
    });
  };

  const handleStreamError = (error: unknown): void => {
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
  };

  const startSubmission = useCallback(() => {
    setIsTestCaseTableOpen(false);
    setIsBenchmarking(false);
    setIsSubmitting(true);
    setSubmissionStatus({
      status: "CHECKING",
      runtime: null,
      gflops: null,
      passedTests: null,
      totalTests: null,
      message: "status: CHECKING",
    });
  }, []);

  return {
    isSubmitting,
    submissionStatus,
    submissionId,
    isTestCaseTableOpen,
    isBenchmarking,
    setIsTestCaseTableOpen,
    processSubmission,
    startSubmission,
    setSubmissionStatus
  };
}