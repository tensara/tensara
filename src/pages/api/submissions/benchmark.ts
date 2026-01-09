import { type NextApiRequest, type NextApiResponse } from "next";
import { env } from "~/env";
import { combinedAuth } from "~/server/auth";
import { checkRateLimit } from "~/hooks/useRateLimit";
import {
  isSubmissionError,
  SubmissionError,
  SubmissionStatus,
} from "~/types/submission";
import type {
  BenchmarkedResponse,
  BenchmarkResultResponse,
  SubmissionErrorType,
  TestResultWithRuns,
} from "~/types/submission";
import { db } from "~/server/db";

export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse
) {
  if (req.method !== "POST") {
    res.setHeader("Allow", ["POST"]);
    res.status(405).end(`Method ${req.method} Not Allowed`);
    return;
  }

  const session = await combinedAuth(req, res);

  if (!session) {
    res.status(401).json({ error: "Not authenticated" });
    return;
  }

  if (session && "error" in session) {
    res.status(401).json({ error: session.error });
    return;
  }

  const { problemSlug, code, language, gpuType } = req.body as {
    problemSlug: string;
    code: string;
    language: string;
    gpuType: string;
  };

  const requiredFields = { problemSlug, code, language, gpuType };
  const missingFields = Object.entries(requiredFields).filter(
    ([_, value]) => value === undefined
  );

  if (missingFields.length > 0) {
    res.status(400).json({
      error: `Missing required fields: ${missingFields.map(([key]) => key).join(", ")}`,
    });
    return;
  }

  const rateLimit = await checkRateLimit(session.user.id);
  if (!rateLimit.allowed) {
    res.status(rateLimit.statusCode ?? 429).json({
      status: SubmissionError.RATE_LIMIT_EXCEEDED as SubmissionErrorType,
      error: rateLimit.error,
      details: rateLimit.error,
    });
    return;
  }

  const problem = await db.problem.findUnique({
    where: { slug: problemSlug },
  });

  if (!problem) {
    res.status(404).json({ error: "Problem not found" });
    return;
  }

  res.setHeader("Content-Type", "text/event-stream");
  res.setHeader("Cache-Control", "no-cache, no-transform");
  res.setHeader("Connection", "keep-alive");
  res.setHeader("X-Accel-Buffering", "no");

  res.setHeader("Transfer-Encoding", "chunked");
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "POST, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type");

  res.setHeader("Content-Encoding", "identity");

  res.setHeader("Keep-Alive", "timeout=120, max=1000");

  const sendSSE = (event: string, data: unknown) => {
    const payload = `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
    res.write(payload);

    try {
      if (
        "flush" in res &&
        typeof (res as { flush: () => void }).flush === "function"
      ) {
        (res as { flush: () => void }).flush();
      } else if (res.flushHeaders) {
        res.flushHeaders();
      }
    } catch (error) {
      console.warn("Failed to flush response:", error);
    }
  };

  const heartbeat = setInterval(() => {
    try {
      sendSSE("heartbeat", { timestamp: Date.now() });
    } catch (error) {
      console.warn("Failed to send heartbeat:", error);
    }
  }, 30000);

  try {
    const benchmarkResponse = await fetch(
      env.MODAL_ENDPOINT + "/benchmark_cli-" + (gpuType ?? "T4"),
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          solution_code: code,
          problem: problem.slug,
          problem_def: problem.definition,
          gpu_type: gpuType,
          dtype: "float32",
          language: language,
        }),
      }
    );

    if (!benchmarkResponse.ok) {
      const errorText = await benchmarkResponse.text();

      sendSSE(SubmissionError.ERROR, {
        error: `Benchmark API returned ${benchmarkResponse.status}`,
        message: `Benchmark API returned ${benchmarkResponse.status}`,
        details: errorText,
      });

      res.end();
      return;
    }

    const benchmarkReader = benchmarkResponse.body?.getReader();
    if (!benchmarkReader) {
      sendSSE(SubmissionError.ERROR, {
        error: "No response body from benchmark",
        message: "No response body from benchmark",
      });

      res.end();
      return;
    }

    const benchmarkResults: TestResultWithRuns[] = [];
    let partialMessage = "";

    try {
      while (true) {
        const { done, value } = await benchmarkReader.read();
        if (done) break;

        const text = new TextDecoder().decode(value);
        partialMessage += text;

        const messages = partialMessage.split("\n\n");
        partialMessage = messages.pop() ?? "";

        for (const message of messages) {
          if (!message?.startsWith("data: ")) continue;

          try {
            const response_json = message.slice(6).trim();
            const parsed = JSON.parse(response_json) as {
              status: string;
            };
            if (!parsed) {
              continue;
            }
            const response_status = parsed.status;
            console.log(response_status);

            if (response_status === "COMPILING") {
              sendSSE(SubmissionStatus.COMPILING, {
                status: SubmissionStatus.COMPILING,
              });
            } else if (response_status === "SANITY_CHECK_PASSED") {
              sendSSE(SubmissionStatus.SANITY_CHECK_PASSED, {
                status: SubmissionStatus.SANITY_CHECK_PASSED,
              });
            } else if (response_status === "BENCHMARKING") {
              sendSSE(SubmissionStatus.BENCHMARKING, {
                status: SubmissionStatus.BENCHMARKING,
              });
            }
            if (response_status === SubmissionStatus.BENCHMARK_RESULT) {
              const response = JSON.parse(
                response_json
              ) as BenchmarkResultResponse;
              if (response.result) {
                benchmarkResults.push(response.result);
                sendSSE(response_status, response);
              }
            } else if (response_status === SubmissionStatus.BENCHMARKED) {
              const response = JSON.parse(response_json) as BenchmarkedResponse;
              if (response.avg_gflops && response.avg_runtime_ms) {
                const averageGflops = response.avg_gflops;
                const averageRuntime = response.avg_runtime_ms;

                sendSSE(SubmissionStatus.ACCEPTED, {
                  avg_runtime_ms: averageRuntime,
                  avg_gflops: averageGflops,
                  benchmark_results: benchmarkResults,
                  total_tests: benchmarkResults.length,
                });

                res.end();
                return;
              }
              if (!response.avg_gflops && response.avg_runtime_ms) {
                // For Graph problems where gflops is not calculated
                const averageRuntime = response.avg_runtime_ms;

                sendSSE(SubmissionStatus.ACCEPTED, {
                  avg_runtime_ms: averageRuntime,
                  benchmark_results: benchmarkResults,
                  total_tests: benchmarkResults.length,
                });

                res.end();
                return;
              }
            } else if (isSubmissionError(response_status)) {
              const response = JSON.parse(response_json) as {
                status: SubmissionErrorType;
                error: string;
                details: string;
              };

              sendSSE(response_status, response);
              res.end();
              return;
            }
          } catch (e) {
            console.error("Failed to parse benchmark SSE data:", e);
            continue;
          }
        }
      }
    } finally {
      benchmarkReader.releaseLock();
    }

    res.end();
  } catch (error) {
    console.error("Error in benchmark handler:", error);

    sendSSE(SubmissionError.ERROR, {
      error: error instanceof Error ? error.message : "Unknown error",
      message: error instanceof Error ? error.message : "Unknown error",
      details: error instanceof Error ? error.stack : undefined,
    });

    res.end();
  } finally {
    clearInterval(heartbeat);
  }
}
