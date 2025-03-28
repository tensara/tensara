import { type NextApiRequest, type NextApiResponse } from "next";
import { db } from "~/server/db";
import { env } from "~/env";
import { auth } from "~/server/auth";
import { checkRateLimit } from "~/hooks/useRateLimit";
import { BenchmarkedResponse, BenchmarkResultResponse, CheckedResponse, SubmissionError, SubmissionErrorType, SubmissionStatus, TestResultResponse, WrongAnswerResponse } from "~/types/submission";
import { SubmissionStatusType } from "~/types/submission";
import { escape } from "querystring";

function isSubmissionError(status: string): status is SubmissionErrorType {
  return Object.values(SubmissionError).includes(status as SubmissionErrorType);
}

export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse
) {
  if (req.method !== "POST") {
    res.setHeader("Allow", ["POST"]);
    res.status(405).end(`Method ${req.method} Not Allowed`);
    return;
  }

  const session = await auth(req, res);
  if (!session) {
    res.status(401).json({ error: "Not authenticated" });
    return;
  }

  const { submissionId } = req.body as { submissionId: string };

  if (!submissionId) {
    res.status(400).json({ error: "Missing submissionId" });
    return;
  }

  const rateLimit = await checkRateLimit(session.user.id);
  if (!rateLimit.allowed) {
    await db.submission.deleteMany({
      where: { id: submissionId }
    });
    res.status(rateLimit.statusCode ?? 429).json({ error: rateLimit.error });
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
    const submission = await db.submission.findUnique({
      where: { id: submissionId },
      include: {
        problem: true,
      },
    });

    console.log("Submission found", submission);
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
      where: { id: submissionId },
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

    // should go under "ERROR"
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
            const response_status = parsed.status;

            console.log("Response status", response_status);

            if (response_status === SubmissionStatus.TEST_RESULT) {
              const response = JSON.parse(response_json) as TestResultResponse;
              sendSSE(response_status, response);
              if (response.result) {
                if (!processedTestIds.has(response.result.test_id)) {
                console.log(
                  `Processing test result for test ID ${response.result.test_id}`
                );
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
                }
              }
            } else if (response_status === SubmissionStatus.WRONG_ANSWER) {
              const response = JSON.parse(response_json) as WrongAnswerResponse;
              sendSSE(response_status, response);
              const failedTest = response.test_results?.find(
                  (t: any) => t.status === "FAILED"
                );
              const errorMessage = `Failed on test ${failedTest?.result.test_id} (${failedTest?.result.name})`;
              const errorDetails = JSON.stringify(failedTest?.result.debug_info);
              await db.submission.update({
                where: { id: submission.id },
                data: {
                  passedTests,
                  totalTests,
                  errorMessage,
                  errorDetails,
                },
              });
            } else if (response_status === SubmissionStatus.CHECKED) {
              const response = JSON.parse(response_json) as CheckedResponse;
              sendSSE(response_status, response);
              let checkerPassed = response.passed_tests === response.total_tests;

              if (
                response.total_tests !== undefined &&
                response.passed_tests !== undefined
              ) {
                console.log(
                  `Using complete message test counts: passed=${response.passed_tests}, total=${response.total_tests}`
                );
                totalTests = response.total_tests;
                passedTests = response.passed_tests;
              } else {
                console.log(
                  `Using local test counts: passed=${passedTests}, total=${totalTests}`
                );
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
          } else if (isSubmissionError(response_status)) {
            const response = JSON.parse(response_json) as {
              status: SubmissionErrorType;
              error: string;
              details: string;
            };

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
      sendSSE(SubmissionError.ERROR, {
        error: `Benchmark API returned ${benchmarkResponse.status}`,
        message: `Benchmark API returned ${benchmarkResponse.status}`,
        details: errorText,
      });

      await db.submission.update({
        where: { id: submission.id },
        data: {
          status: SubmissionError.ERROR,
          errorMessage: `Benchmark API returned ${benchmarkResponse.status}`,
          errorDetails: errorText,
        },
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

      await db.submission.update({
        where: { id: submission.id },
        data: {
          status: SubmissionError.ERROR,
          errorMessage: "No response body from benchmark",
        },
      });

      res.end();
      return;
    }

    let benchmarkResults:BenchmarkResultResponse["result"][] = [];
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
            const response_status = parsed.status;

            if (response_status === SubmissionStatus.BENCHMARK_RESULT) {
              const response = JSON.parse(response_json) as BenchmarkResultResponse;
              if (response.result) {
              sendSSE(response_status, response);
              benchmarkResults.push(response.result);
              await db.submission.update({
                where: { id: submission.id },
                data: {
                  benchmarkResults,
                },
              });
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
                runtime: averageRuntime,
                gflops: averageGflops,
                benchmarkResults,
              });
              }
            }
           
            else if (isSubmissionError(response_status)) {
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

    sendSSE("error", {
      error: error instanceof Error ? error.message : "Unknown error",
      details: error instanceof Error ? error.stack : undefined,
    });

    if (submissionId) {
      try {
        await db.submission.update({
          where: { id: submissionId },
          data: {
            status: SubmissionError.ERROR,
            errorMessage:
              error instanceof Error ? error.message : "Unknown error occurred",
            errorDetails: error instanceof Error ? error.stack ?? "" : "",
          },
        });
      } catch (dbError) {
        console.error("Failed to update submission with error:", dbError);
      }
    }

    res.end();
  } finally {
    clearInterval(heartbeat);
  }
}
