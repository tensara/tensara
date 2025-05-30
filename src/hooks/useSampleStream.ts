import { useState, useCallback } from "react";
import { useToast } from "@chakra-ui/react";

type SampleStatus =
  | "IDLE"
  | "IN_QUEUE"
  | "COMPILING"
  | "SAMPLE_RESULT"
  | "ERROR";

type SampleEvent = {
  status: SampleStatus;
  message?: string;
  details?: string;
};

export function useSampleStream() {
  const toast = useToast();
  const [output, setOutput] = useState<string[]>([]);
  const [status, setStatus] = useState<SampleStatus>("IDLE");
  const [isRunning, setIsRunning] = useState(false);

  const append = useCallback((line: string) => {
    setOutput((prev) => [...prev, line]);
  }, []);

  const startSampleRun = useCallback(
    async (submissionData: {
      code: string;
      language: string;
      gpuType: string;
      problemSlug: string;
    }) => {
      setIsRunning(true);
      setOutput([]);
      setStatus("IN_QUEUE");

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
          if (done) {
            console.log("✅ Stream complete");
            setIsRunning(false);
            break;
          }
        
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
            const data = JSON.parse(dataLine.slice(6).trim());
        
            switch (type) {
              case "IN_QUEUE":
              case "COMPILING":
                setStatus(type as SampleStatus);
                append(`📦 ${type}`);
                break;
              case "SAMPLE_RESULT":
                setStatus("SAMPLE_RESULT");
                console.log("Sample result:", data);
                // append(`✅ Result: ${data.status}`);
                setIsRunning(false);
                break;
              case "ERROR":
                setStatus("ERROR");
                append(`❌ Error: ${data.message}`);
                toast({
                  title: "Sample Error",
                  description: data.message,
                  status: "error",
                  duration: 5000,
                  isClosable: true,
                });
                setIsRunning(false);
                break;
              default:
                append(`ℹ️ ${type}`);
                break;
            }
          }
        }
        
       } catch (err) {
        console.error("Sample run error:", err);
        setStatus("ERROR");
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
