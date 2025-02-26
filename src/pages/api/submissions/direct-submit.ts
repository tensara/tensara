import { type NextApiRequest, type NextApiResponse } from "next";
import { db } from "~/server/db";
import { env } from "~/env";
import { auth } from "~/server/auth";

const SubmissionStatus = {
  CHECKING: "CHECKING",
  BENCHMARKING: "BENCHMARKING",
  ACCEPTED: "ACCEPTED",
  WRONG_ANSWER: "WRONG_ANSWER",
  ERROR: "ERROR",
} as const;

type SubmissionStatus =
  (typeof SubmissionStatus)[keyof typeof SubmissionStatus];

type BenchmarkTestResult = {
  test_id: number;
  name: string;
  runtime_ms: number;
  gflops: number;
};

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

  res.setHeader("Content-Type", "text/event-stream");
  res.setHeader("Cache-Control", "no-cache, no-transform");
  res.setHeader("Connection", "keep-alive");
  res.setHeader("X-Accel-Buffering", "no");

  res.setHeader("Transfer-Encoding", "chunked");
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "POST, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type");

  res.setHeader("Content-Encoding", "identity");

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

  const { submissionId } = req.body as { submissionId: string };

  if (!submissionId) {
    res.status(400).json({ error: "Missing submissionId" });
    return;
  }

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

    await db.submission.update({
      where: { id: submissionId },
      data: {
        status: SubmissionStatus.CHECKING,
      },
    });

    sendSSE("status", {
      id: submission.id,
      status: SubmissionStatus.CHECKING,
    });

    console.log("Starting checker process");
    const checkerResponse = await fetch(
      `https://${env.MODAL_CHECKER_SLUG}-${(
        submission.gpuType ?? "t4"
      ).toLowerCase()}.modal.run`,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          solution_code: submission.code,
          tests_code: submission.problem.tests,
          reference_code: submission.problem.reference,
          gpu_type: submission.gpuType,
        }),
      }
    );

    if (!checkerResponse.ok) {
      const errorText = await checkerResponse.text();
      sendSSE("error", {
        error: `Checker API returned ${checkerResponse.status}`,
        details: errorText,
      });

      await db.submission.update({
        where: { id: submission.id },
        data: {
          status: SubmissionStatus.ERROR,
          errorMessage: `Checker API returned ${checkerResponse.status}`,
          errorDetails: errorText,
        },
      });

      res.end();
      return;
    }

    const reader = checkerResponse.body?.getReader();
    if (!reader) {
      sendSSE("error", { error: "No response body from checker" });

      await db.submission.update({
        where: { id: submission.id },
        data: {
          status: SubmissionStatus.ERROR,
          errorMessage: "No response body from checker",
        },
      });

      res.end();
      return;
    }

    let passedTests = 0;
    let totalTests = 0;
    let partialMessage = "";
    let checkerPassed = false;

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
            const response = JSON.parse(message.slice(6).trim()) as {
              status: string;
              result?: {
                test_id: number;
                name: string;
                status: string;
              };
              passed?: boolean;
              error?: string;
              details?: string;
              test_results?: Array<{
                test_id: number;
                name: string;
                status: string;
              }>;
              passed_tests?: number;
              total_tests?: number;
            };

            sendSSE("checker", response);

            if (response.status === "test_result" && response.result) {
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
              } else {
                console.log(
                  `Skipping duplicate test result for test ID ${response.result.test_id}`
                );
              }
            } else if (response.status === "complete") {
              checkerPassed = response.passed ?? false;

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

              await db.submission.update({
                where: { id: submission.id },
                data: {
                  passedTests,
                  totalTests,
                },
              });
            } else if (response.status === "error") {
              await db.submission.update({
                where: { id: submission.id },
                data: {
                  status: SubmissionStatus.ERROR,
                  errorMessage: response.error ?? "Unknown error",
                  errorDetails: response.details ?? "",
                  passedTests,
                  totalTests,
                },
              });

              res.end();
              return;
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

    if (!checkerPassed) {
      await db.submission.update({
        where: { id: submission.id },
        data: {
          status: SubmissionStatus.WRONG_ANSWER,
          passedTests,
          totalTests,
        },
      });

      sendSSE("complete", {
        status: SubmissionStatus.WRONG_ANSWER,
        passedTests,
        totalTests,
      });

      res.end();
      return;
    }

    await db.submission.update({
      where: { id: submission.id },
      data: {
        status: SubmissionStatus.BENCHMARKING,
        passedTests,
        totalTests,
      },
    });

    sendSSE("status", {
      status: SubmissionStatus.BENCHMARKING,
      passedTests,
      totalTests,
    });

    const benchmarkResponse = await fetch(
      `https://${env.MODAL_BENCHMARK_SLUG}-${(
        submission.gpuType ?? "t4"
      ).toLowerCase()}.modal.run`,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          solution_code: submission.code,
          tests_code: submission.problem.tests,
          gpu_type: submission.gpuType,
        }),
      }
    );

    if (!benchmarkResponse.ok) {
      const errorText = await benchmarkResponse.text();
      sendSSE("error", {
        error: `Benchmark API returned ${benchmarkResponse.status}`,
        details: errorText,
      });

      await db.submission.update({
        where: { id: submission.id },
        data: {
          status: SubmissionStatus.ERROR,
          errorMessage: `Benchmark API returned ${benchmarkResponse.status}`,
          errorDetails: errorText,
        },
      });

      res.end();
      return;
    }

    const benchmarkReader = benchmarkResponse.body?.getReader();
    if (!benchmarkReader) {
      sendSSE("error", { error: "No response body from benchmark" });

      await db.submission.update({
        where: { id: submission.id },
        data: {
          status: SubmissionStatus.ERROR,
          errorMessage: "No response body from benchmark",
        },
      });

      res.end();
      return;
    }

    let benchmarkResults: BenchmarkTestResult[] = [];
    let averageGflops = 0;
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
            const response = JSON.parse(message.slice(6).trim()) as {
              status: string;
              result?: {
                test_id: number;
                name: string;
                runtime_ms: number;
                gflops: number;
              };
              test_results?: BenchmarkTestResult[];
              average_gflops?: number;
              error?: string;
              details?: string;
            };

            // Forward event to client
            sendSSE("benchmark", response);

            // Update database based on event type
            if (response.status === "test_result" && response.result) {
              benchmarkResults.push(response.result);

              await db.submission.update({
                where: { id: submission.id },
                data: {
                  benchmarkResults,
                },
              });
            } else if (
              response.status === "success" &&
              response.test_results &&
              response.average_gflops
            ) {
              benchmarkResults = response.test_results;
              averageGflops = response.average_gflops;

              const runtime =
                benchmarkResults.reduce((acc, r) => acc + r.runtime_ms, 0) /
                benchmarkResults.length;

              await db.submission.update({
                where: { id: submission.id },
                data: {
                  status: SubmissionStatus.ACCEPTED,
                  runtime,
                  gflops: averageGflops,
                  benchmarkResults,
                },
              });

              sendSSE("complete", {
                status: SubmissionStatus.ACCEPTED,
                runtime,
                gflops: averageGflops,
                benchmarkResults,
                passedTests,
                totalTests,
              });

              res.end();
              return;
            } else if (response.status === "error") {
              await db.submission.update({
                where: { id: submission.id },
                data: {
                  status: SubmissionStatus.ERROR,
                  errorMessage: response.error ?? "Unknown error",
                  errorDetails: response.details ?? "",
                },
              });

              sendSSE("error", {
                error: response.error ?? "Unknown error",
                details: response.details ?? "",
              });

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
            status: SubmissionStatus.ERROR,
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
  }
}
