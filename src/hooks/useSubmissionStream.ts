import { useState, useCallback } from 'react';
import { useToast } from '@chakra-ui/react';
import { SubmissionStatusType, SubmissionErrorType, ErrorResponse, WrongAnswerResponse, CheckedResponse, BenchmarkedResponse, TestResultResponse, BenchmarkResultResponse, SubmissionError, SubmissionStatus, AcceptedResponse } from '~/types/submission';

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
  const [metaStatus, setMetaStatus] = useState<SubmissionStatusType | SubmissionErrorType | null>(null);
  const [metaResponse, setMetaResponse] = useState<ResponseTypeMap[keyof ResponseTypeMap] | null>(null);
  const [totalTests, setTotalTests] = useState<number>(0);

  const [testResults, setTestResults] = useState<TestResultResponse[]>([]);
  const [benchmarkResults, setBenchmarkResults] = useState<BenchmarkResultResponse[]>([]);

  const [isSubmitting, setIsSubmitting] = useState<boolean>(false);
  const [submissionId, setSubmissionId] = useState<string | null>(null);
  const [isTestCaseTableOpen, setIsTestCaseTableOpen] = useState<boolean>(false);
  const [isBenchmarking, setIsBenchmarking] = useState<boolean>(false);

  // Type-safe accessor for metaResponse based on current metaStatus
  const getTypedResponse = <T extends SubmissionStatusType | SubmissionErrorType>(
    status: T
  ): T extends keyof ResponseTypeMap ? ResponseTypeMap[T] | null : null => {
    if (metaStatus === status) {
      return metaResponse as any;
    }
    return null as any;
  };

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
          const errorMessage = await response.json() as ErrorResponse;
          setIsSubmitting(false);
          setMetaStatus(SubmissionError.ERROR);
          setMetaResponse({
            status: SubmissionError.RATE_LIMIT_EXCEEDED,
            message: errorMessage.message || "Rate limit exceeded",
            details: errorMessage.details || "Rate limit exceeded",
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

    switch (eventType) {
      case "IN_QUEUE":
      case "COMPILING":
      case "CHECKING":
      case "BENCHMARKING":
        setMetaStatus(eventType as SubmissionStatusType);
        break;
      case "TEST_RESULT":
        const data = JSON.parse(eventData) as TestResultResponse;
        setTestResults(prev => [...prev, data]);
        setTotalTests(data.total_tests);
        break;
      case "WRONG_ANSWER":
        const wrongAnswerData = JSON.parse(eventData) as WrongAnswerResponse;
        setMetaStatus(SubmissionStatus.WRONG_ANSWER);
        setMetaResponse(wrongAnswerData);
        break;
      case "CHECKED":
        const checkedData = JSON.parse(eventData) as CheckedResponse;
        setMetaStatus(SubmissionStatus.CHECKED);
        setMetaResponse(checkedData);
        break;
      case "BENCHMARKED":
        const benchmarkedData = JSON.parse(eventData) as BenchmarkedResponse;
        setMetaStatus(SubmissionStatus.BENCHMARKED);
        setMetaResponse(benchmarkedData);
        break;
      case "BENCHMARK_RESULT":
        const benchmarkResultData = JSON.parse(eventData) as BenchmarkResultResponse;
        setBenchmarkResults(prev => [...prev, benchmarkResultData]);
        setTotalTests(benchmarkResultData.total_tests);
        break;
      case "ACCEPTED":
        setMetaStatus(SubmissionStatus.ACCEPTED);
        setMetaResponse(JSON.parse(eventData) as AcceptedResponse);
        break;
      case "ERROR":
      case "RATE_LIMIT_EXCEEDED":
      case "COMPILE_ERROR": 
      case "RUNTIME_ERROR":
      case "TIME_LIMIT_EXCEEDED":
        setMetaStatus(eventType as SubmissionErrorType);
        setMetaResponse(JSON.parse(eventData) as ErrorResponse);
        break;
      default:
        break;
    }

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
    setMetaStatus("IN_QUEUE");
  }, []);

  return {
    isSubmitting,
    metaStatus,
    metaResponse,
    testResults,
    benchmarkResults,
    submissionId,
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