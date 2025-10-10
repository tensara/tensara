/**
 * useSampleStream.ts
 *
 * React hook that manages the lifecycle of a sample submission run.
 * - Initiates a POST request to `/api/submissions/sample` to start execution.
 * - Streams real-time Server-Sent Events (SSE) updates from the backend.
 * - Updates local state (`status`, `output`, `isRunning`) based on received events.
 * - Handles formatting of input/output vectors for display and error toasts.
 *
 * Used by the Tensara frontend to show live feedback while user code compiles/runs.
 */

import { useState, useCallback } from "react";
import { useToast } from "@chakra-ui/react";
import {
  SampleStatus,
  type SampleStatusType,
  type SampleEvent,
  type SampleOutput,
} from "~/types/submission";

const formatVector = (data: unknown): string => {
  if (Array.isArray(data)) {
    if (Array.isArray(data[0])) {
      return "[" + data.map((row) => formatVector(row)).join("\n") + "]";
    }
    return (
      "[" +
      (data as number[])
        .map((x) => {
          const str = String(x);
          return str.includes(".") ? str.replace(/\.?0+$/, "") : str;
        })
        .join(" ") +
      "]"
    );
  }
  return String(data);
};

const formatParameters = (data: unknown): string =>
  Array.isArray(data)
    ? (data as unknown[]).map(formatVector).join("\n\n")
    : formatVector(data);

export function useSampleStream() {
  const toast = useToast();
  const [output, setOutput] = useState<SampleOutput | null>(null);
  const [status, setStatus] = useState<SampleStatusType>(SampleStatus.IDLE);
  const [isRunning, setIsRunning] = useState(false);

  const startSampleRun = useCallback(
    async (submissionData: {
      code: string;
      language: string;
      gpuType: string;
      problemSlug: string;
    }) => {
      setIsRunning(true);
      setOutput(null);
      setStatus(SampleStatus.IN_QUEUE);

      try {
        const response = await fetch("/api/submissions/sample", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(submissionData),
        });

        if (response.status === 429) {
          setStatus(SampleStatus.TOO_MANY_REQUESTS);
          setIsRunning(false);
          toast({
            title: "Too Many Requests",
            description: "Daily sample limit exceeded.",
            status: "error",
            duration: 5000,
            isClosable: true,
          });
          return;
        }

        const reader = response.body?.getReader();
        if (!reader) throw new Error("No response body");

        const decoder = new TextDecoder();
        let buffer = "";

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });

          // Split SSE frames; tolerate \n\n or \r\n\r\n
          const frames = buffer.split(/\r?\n\r?\n/);
          buffer = frames.pop() ?? "";

          for (const frame of frames) {
            // We only require a "data:" line. Ignore "event:" if absent.
            const dataLine = frame
              .split(/\r?\n/)
              .find((l) => l.startsWith("data: "));
            if (!dataLine) continue;

            let data: SampleEvent | null = null;
            try {
              data = JSON.parse(dataLine.slice(6).trim()) as SampleEvent;
            } catch {
              continue;
            }

            // Heartbeats / comments may have no status            if (!data?.status) continue;

            // Drive UI by status codes only
            switch (data.status) {
              case SampleStatus.IN_QUEUE:
              case SampleStatus.COMPILING:
              case SampleStatus.RUNNING:
                setStatus(data.status);
                break;

              case SampleStatus.ERROR:
              case SampleStatus.COMPILE_ERROR:
              case SampleStatus.RUNTIME_ERROR:
              case SampleStatus.TIME_LIMIT_EXCEEDED:
                setStatus(data.status);
                setOutput({
                  status: data.status,
                  message: data.message,
                  details: data.details,
                  stdout: data.stdout,
                  stderr: data.stderr,
                });
                setIsRunning(false);
                break;

              case SampleStatus.PASSED:
              case SampleStatus.FAILED:
                setStatus(data.status);
                setOutput({
                  status: data.status,
                  input: data.input ? formatParameters(data.input) : undefined,
                  output: data.output ? formatVector(data.output) : undefined,
                  stdout: data.stdout,
                  stderr: data.stderr,
                  expected_output: data.expected_output
                    ? formatVector(data.expected_output)
                    : undefined,
                });
                setIsRunning(false);
                break;

              default:
                // Unknown code: ignore gracefully
                break;
            }
          }
        }
      } catch (err) {
        console.error("Sample run error:", err);
        setStatus(SampleStatus.ERROR);
        setIsRunning(false);
        toast({
          title: "Network Error",
          description: "Could not connect to sample run service.",
          status: "error",
          duration: 5000,
          isClosable: true,
        });
      }
    },
    [toast]
  );

  return { output, status, isRunning, startSampleRun };
}
