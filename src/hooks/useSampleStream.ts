import { useState, useCallback } from "react";
import { useToast } from "@chakra-ui/react";
import {
  SampleStatus,
  type SampleStatusType,
  type SampleEvent,
} from "~/types/submission";

const formatVector = (data: unknown): string => {
  if (Array.isArray(data)) {
    if (Array.isArray(data[0])) {
      return data.map((row) => formatVector(row)).join("\n");
    }
    return (
      "[" + data.map((x: number) => x.toFixed(6).padStart(10)).join(" ") + "]"
    );
  }
  return String(data);
};

export function useSampleStream() {
  const toast = useToast();
  const [output, setOutput] = useState<string[]>([]);
  const [status, setStatus] = useState<SampleStatusType>(SampleStatus.IDLE);
  const [isRunning, setIsRunning] = useState(false);

  const append = useCallback((line: string) => {
    setOutput((prev) => [...prev, line]);
  }, []);

  const formatBlock = (label: string, content: string) => {
    return `${label} \n----------------\n${content}\n`;
  };

  const formatStatus = (status: SampleStatusType) => {
    switch (status) {
      case SampleStatus.IN_QUEUE:
        return "In queue...";
      case SampleStatus.COMPILING:
        return "Compiling...";
      case SampleStatus.RUNNING:
        return "Running...";
      case SampleStatus.PASSED:
        return "Test Passed!";
      case SampleStatus.FAILED:
        return "Test Failed";
      case SampleStatus.ERROR:
        return "Error Occurred";
      case SampleStatus.COMPILE_ERROR:
        return "Compilation Error";
      case SampleStatus.RUNTIME_ERROR:
        return "Runtime Error";
      default:
        return status;
    }
  };

  const startSampleRun = useCallback(
    async (submissionData: {
      code: string;
      language: string;
      gpuType: string;
      problemSlug: string;
    }) => {
      setIsRunning(true);
      setOutput([]);
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
                setStatus(type as SampleStatusType);
                append(formatStatus(type as SampleStatusType));
                break;

              case "SAMPLE_RESULT":
                setStatus(data.status);
                append(formatStatus(data.status));
                if (data.input)
                  append(formatBlock("Input", formatVector(data.input)));
                if (data.output)
                  append(formatBlock("Output", formatVector(data.output)));
                if (data.stdout) append(formatBlock("Stdout", data.stdout));
                if (data.stderr) append(formatBlock("Stderr", data.stderr));
                setIsRunning(false);
                break;

              case SampleStatus.ERROR:
              case SampleStatus.COMPILE_ERROR:
              case SampleStatus.RUNTIME_ERROR:
                setStatus(type as SampleStatusType);
                append(formatStatus(type as SampleStatusType));
                if (data.message) append(`${data.message}`);
                if (data.details) append(`${data.details}`);
                setIsRunning(false);
                break;

              default:
                append(`Log: ${type}`);
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
    [append, toast]
  );

  return {
    output,
    status,
    isRunning,
    startSampleRun,
  };
}
