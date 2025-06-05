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
    return "[" + data.map((x: number) => {
      const str = x.toString();
      const formatted = str.includes('.') ? str.replace(/\.?0+$/, '') : str;
      return formatted;
    }).join(" ") + "]";
  }
  return String(data);
};

const formatParameters = (data: unknown): string => {
  if (Array.isArray(data)) {
    return data.map((x) => formatVector(x)).join("\n\n");
  }
  return formatVector(data);
};

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
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(submissionData),
        });

        const reader = response.body?.getReader();
        if (!reader) throw new Error("No response body");

        const decoder = new TextDecoder();
        let buffer = "";

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          const chunk = decoder.decode(value, { stream: true });
          buffer += chunk;

          const events = buffer.split("\n\n");
          buffer = events.pop() ?? "";

          for (const event of events) {
            const lines = event.split("\n");
            const typeLine = lines.find((l) => l.startsWith("event: "));
            const dataLine = lines.find((l) => l.startsWith("data: "));
            if (!typeLine || !dataLine) continue;

            const type = typeLine.slice(7).trim();
            const data = JSON.parse(dataLine.slice(6).trim()) as SampleEvent;
            console.log(type, data);

            switch (type) {
              case SampleStatus.IN_QUEUE:
              case SampleStatus.COMPILING:
              case SampleStatus.RUNNING:
                setStatus(type as SampleStatusType);
                break;

              case "SAMPLE_RESULT":
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

              case SampleStatus.ERROR:
              case SampleStatus.COMPILE_ERROR:
              case SampleStatus.RUNTIME_ERROR:
                setStatus(type as SampleStatusType);
                setOutput({
                  status: type as SampleStatusType,
                  message: data.message,
                  details: data.details,
                });
                setIsRunning(false);
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

  return {
    output,
    status,
    isRunning,
    startSampleRun,
  };
}
