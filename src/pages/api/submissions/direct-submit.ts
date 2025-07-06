import { type NextApiRequest, type NextApiResponse } from "next";
import { db } from "~/server/db";
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
  CheckedResponse,
  SubmissionErrorType,
  TestResult,
  TestResultResponse,
  WrongAnswerResponse,
} from "~/types/submission";

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

  const submission = await db.submission.create({
    data: {
      code,
      language,
      gpuType: gpuType || "T4",
      status: SubmissionStatus.IN_QUEUE,
      problem: { connect: { id: problem.id } },
      user: { connect: { id: session.user.id } },
    },
    include: {
      problem: true,
    },
  });

  try {
    if (!submission) {
      res.status(404).json({ error: "Submission not found" });
      return;
    }

    if (submission.userId !== session.user.id) {
      res.status(403).json({ error: "Unauthorized" });
      return;
    }

    // TODO:
    await db.submission.update({
      where: { id: submission.id },
      data: {
        status: SubmissionStatus.CHECKING,
      },
    });

    sendSSE(SubmissionStatus.IN_QUEUE, {
      id: submission.id,
    });

    console.log("Starting checker process");
    const checkerResponse = await fetch(
      env.MODAL_ENDPOINT + "/checker-" + (submission.gpuType ?? "t4"),
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          solution_code: submission.code,
          problem: submission.problem.slug,
          problem_def: submission.problem.definition,
          gpu_type: submission.gpuType,
          dtype: "float32",
          language: submission.language,
        }),
      }
    );

    if (!checkerResponse.ok) {
      const errorText = await checkerResponse.text();
      sendSSE(SubmissionError.ERROR, {
        error: `Checker API returned ${checkerResponse.status}`,
        message: `Checker API returned ${checkerResponse.status}`,
        details: errorText,
      });

      await db.submission.update({
        where: { id: submission.id },
        data: {
          status: SubmissionError.ERROR,
          errorMessage: `Checker API returned ${checkerResponse.status}`,
          errorDetails: errorText,
        },
      });

      res.end();
      return;
    }

    const reader = checkerResponse.body?.getReader();
    if (!reader) {
      sendSSE(SubmissionError.ERROR, {
        error: "No response body from checker",
      });

      await db.submission.update({
        where: { id: submission.id },
        data: {
          status: SubmissionError.ERROR,
          errorMessage: "No response body from checker",
        },
      });

      res.end();
      return;
    }

    let passedTests = 0;
    let totalTests = 0;
    let partialMessage = "";

    const processedTestIds = new Set<number>();

    try {
      while (true) {
        const { done, value } = await reader.read();
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

            if (response_status === SubmissionStatus.TEST_RESULT) {
              const response = JSON.parse(response_json) as TestResultResponse;

              if (response.result) {
                if (!processedTestIds.has(response.result.test_id)) {
                  processedTestIds.add(response.result.test_id);
                  totalTests++;
                  if (response.result.status === "PASSED") {
                    passedTests++;
                  }
                  await db.submission.update({
                    where: { id: submission.id },
                    data: {
                      passedTests,
                      totalTests,
                    },
                  });
                  sendSSE(response_status, response);
                }
              }
            } else if (response_status === SubmissionStatus.CHECKED) {
              const response = JSON.parse(response_json) as CheckedResponse;
              const checkerPassed =
                response.passed_tests === response.total_tests;

              if (
                response.total_tests !== undefined &&
                response.passed_tests !== undefined
              ) {
                totalTests = response.total_tests;
                passedTests = response.passed_tests;
              } else {
              }
              if (checkerPassed) {
                await db.submission.update({
                  where: { id: submission.id },
                  data: {
                    passedTests,
                    totalTests,
                  },
                });
              }
              sendSSE(response_status, response);
            } else if (response_status === SubmissionStatus.WRONG_ANSWER) {
              const response = JSON.parse(response_json) as WrongAnswerResponse;
              const failedTest = response.test_results?.find(
                (t: TestResult) => t.status === "FAILED"
              );
              const errorMessage = `Failed on test ${failedTest?.test_id} (${failedTest?.name})`;
              const errorDetails = JSON.stringify(response.debug_info);
              if (
                response.total_tests !== undefined &&
                response.passed_tests !== undefined
              ) {
                totalTests = response.total_tests;
                passedTests = response.passed_tests;
              }
              await db.submission.update({
                where: { id: submission.id },
                data: {
                  status: SubmissionStatus.WRONG_ANSWER,
                  passedTests,
                  totalTests,
                  errorMessage,
                  errorDetails,
                },
              });

              sendSSE(response_status, response);

              res.end();
              return;
            } else if (isSubmissionError(response_status)) {
              const response = JSON.parse(response_json) as {
                status: SubmissionErrorType;
                error: string;
                details: string;
              };

              console.log("Submission error:", response);

              sendSSE(response_status, response);

              await db.submission.update({
                where: { id: submission.id },
                data: {
                  status: response_status,
                  errorMessage: response.error ?? "Unknown error",
                  errorDetails: response.details ?? "",
                  passedTests,
                  totalTests,
                },
              });

              res.end();
              return;
            } else {
              sendSSE(response_status, {});
            }
          } catch (e) {
            console.error("Failed to parse checker SSE data:", e);
            continue;
          }
        }
      }
    } finally {
      reader.releaseLock();
    }

    await db.submission.update({
      where: { id: submission.id },
      data: {
        status: SubmissionStatus.BENCHMARKING,
        passedTests,
        totalTests,
      },
    });

    sendSSE(SubmissionStatus.BENCHMARKING, {
      passedTests,
      totalTests,
    });

    const benchmarkResponse = await fetch(
      env.MODAL_ENDPOINT + "/benchmark-" + (submission.gpuType ?? "t4"),
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          solution_code: submission.code,
          problem: submission.problem.slug,
          problem_def: submission.problem.definition,
          gpu_type: submission.gpuType,
          dtype: "float32",
          language: submission.language,
        }),
      }
    );

    if (!benchmarkResponse.ok) {
      const errorText = await benchmarkResponse.text();

      await db.submission.update({
        where: { id: submission.id },
        data: {
          status: SubmissionError.ERROR,
          errorMessage: `Benchmark API returned ${benchmarkResponse.status}`,
          errorDetails: errorText,
        },
      });

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
      await db.submission.update({
        where: { id: submission.id },
        data: {
          status: SubmissionError.ERROR,
          errorMessage: "No response body from benchmark",
        },
      });

      sendSSE(SubmissionError.ERROR, {
        error: "No response body from benchmark",
        message: "No response body from benchmark",
      });

      res.end();
      return;
    }

    const benchmarkResults: BenchmarkResultResponse["result"][] = [];
    partialMessage = "";

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

            if (response_status === SubmissionStatus.BENCHMARK_RESULT) {
              const response = JSON.parse(
                response_json
              ) as BenchmarkResultResponse;
              if (response.result) {
                benchmarkResults.push(response.result);
                await db.submission.update({
                  where: { id: submission.id },
                  data: {
                    benchmarkResults,
                  },
                });
                sendSSE(response_status, response);
              }
            } else if (response_status === SubmissionStatus.BENCHMARKED) {
              const response = JSON.parse(response_json) as BenchmarkedResponse;
              if (response.avg_gflops && response.avg_runtime_ms) {
                const averageGflops = response.avg_gflops;
                const averageRuntime = response.avg_runtime_ms;

                await db.submission.update({
                  where: { id: submission.id },
                  data: {
                    status: SubmissionStatus.ACCEPTED,
                    runtime: averageRuntime,
                    gflops: averageGflops,
                    benchmarkResults,
                  },
                });

                sendSSE(SubmissionStatus.ACCEPTED, {
                  avg_runtime_ms: averageRuntime,
                  avg_gflops: averageGflops,
                  benchmark_results: benchmarkResults,
                  total_tests: benchmarkResults.length,
                });
              }
            } else if (isSubmissionError(response_status)) {
              const response = JSON.parse(response_json) as {
                status: SubmissionErrorType;
                error: string;
                details: string;
              };
              await db.submission.update({
                where: { id: submission.id },
                data: {
                  status: response_status,
                  errorMessage: response.error ?? "Unknown error",
                  errorDetails: response.details ?? "",
                },
              });

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
    console.error("Error in direct-submit handler:", error);

    if (submission.id) {
      try {
        await db.submission.update({
          where: { id: submission.id },
          data: {
            status: SubmissionError.ERROR,
            errorMessage:
              error instanceof Error ? error.message : "Unknown error occurred",
            errorDetails: error instanceof Error ? (error.stack ?? "") : "",
          },
        });
      } catch (dbError) {
        console.error("Failed to update submission with error:", dbError);
      }
    }

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
