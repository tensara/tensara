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
  BenchmarkResultResponse,
  CheckedResponse,
  SubmissionStatusType,
  SubmissionErrorType,
  TestResult,
  TestResultResponse,
  WrongAnswerResponse,
} from "~/types/submission";
import { proxyUpstreamSSE } from "./sseProxy";

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

  if (!session || "error" in session) {
    res.status(401).json({ error: session?.error ?? "Not authenticated" });
    return;
  }

  const { problemSlug, code, language, gpuType } = req.body as {
    problemSlug: string;
    code: string;
    language: string;
    gpuType: string;
  };

  const missing = Object.entries({ problemSlug, code, language, gpuType })
    .filter(([, v]) => v === undefined)
    .map(([k]) => k);
  if (missing.length) {
    res
      .status(400)
      .json({ error: `Missing required fields: ${missing.join(", ")}` });
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

  const remainingSubmissions = rateLimit.remainingSubmissions;

  const problem = await db.problem.findUnique({
    where: { slug: problemSlug },
    select: {
      id: true,
      definition: true,
    },
  });

  if (!problem) {
    res.status(404).json({ error: "Problem not found" });
    return;
  }

  res.status(200);
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

  // const sendSSE = (event: string, data: unknown) => {
  //   const payload = `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
  //   res.write(payload);

  //   try {
  //     if (
  //       "flush" in res &&
  //       typeof (res as { flush: () => void }).flush === "function"
  //     ) {
  //       (res as { flush: () => void }).flush();
  //     } else if (res.flushHeaders) {
  //       res.flushHeaders();
  //     }
  //   } catch (error) {
  //     console.warn("Failed to flush response:", error);
  //   }
  // };

  const heartbeat = setInterval(() => {
    try {
      res.write(`event: heartbeat\ndata: {"ts":${Date.now()}}\n\n`);
    } catch {}
  }, 30000);

  const controller = new AbortController();
  req.on("close", () => controller.abort());

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
  res.write(
    `event: ${SubmissionStatus.IN_QUEUE}\ndata: ${JSON.stringify({
      id: submission.id,
      remainingSubmissions,
    })}\n\n`
  );

  const payload = {
    solution_code: submission.code,
    problem: submission.problem.slug,
    problem_def: submission.problem.definition,
    gpu_type: submission.gpuType,
    dtype: "float32",
    language: submission.language,
  };

  await db.submission.update({
    where: { id: submission.id },
    data: { status: SubmissionStatus.CHECKING },
  });

  res.write(
    `event: ${SubmissionStatus.CHECKING}\ndata: {"status":"${SubmissionStatus.CHECKING}"}\n\n`
  );

  let passedTests = 0;
  let totalTests = 0;
  const seenTests = new Set<number>();

  const checkerResult = await proxyUpstreamSSE(
    res,
    `${env.MODAL_ENDPOINT}/checker-${submission.gpuType ?? "t4"}`,
    payload,
    async (evt: import("~/types/submission").SubmissionResponse) => {
      const s = evt?.status as string | undefined;
      if (!s) return "CONTINUE";

      if (s === SubmissionStatus.TEST_RESULT) {
        const r = evt as TestResultResponse;
        const id = r.result?.test_id;
        if (id !== undefined && !seenTests.has(id)) {
          seenTests.add(id);
          totalTests++;
          if (r.result?.status === "PASSED") passedTests++;
          await db.submission.update({
            where: { id: submission.id },
            data: { passedTests, totalTests },
          });
        }
        return "CONTINUE";
      }

      if (s === SubmissionStatus.CHECKED) {
        const r = evt as CheckedResponse;
        if (typeof r.total_tests === "number") totalTests = r.total_tests;
        if (typeof r.passed_tests === "number") passedTests = r.passed_tests;
        const checkerPassed = passedTests === totalTests && totalTests > 0;
        if (checkerPassed) {
          await db.submission.update({
            where: { id: submission.id },
            data: { passedTests, totalTests },
          });
          return "CONTINUE"; // proceed to benchmark
        }
        // If worker sends WRONG_ANSWER instead, we’ll catch it below.
        return "CONTINUE";
      }

      if (s === SubmissionStatus.WRONG_ANSWER) {
        const r = evt as WrongAnswerResponse;
        const failed = r.test_results?.find(
          (t: TestResult) => t.status === "FAILED"
        );
        await db.submission.update({
          where: { id: submission.id },
          data: {
            status: SubmissionStatus.WRONG_ANSWER,
            passedTests: r.passed_tests ?? passedTests,
            totalTests: r.total_tests ?? totalTests,
            errorMessage: failed
              ? `Failed on test ${failed.test_id} (${failed.name})`
              : "Wrong answer",
            errorDetails: JSON.stringify(r.debug_info ?? {}),
          },
        });
        return "STOP";
      }

      if (isSubmissionError(s)) {
        // evt is a SubmissionResponse union — narrow to ErrorResponse-like shape
        const err = evt as Partial<import("~/types/submission").ErrorResponse>;
        await db.submission.update({
          where: { id: submission.id },
          data: {
            status: s,
            errorMessage: err.message ?? "Unknown error",
            errorDetails: err.details ?? "",
            passedTests,
            totalTests,
          },
        });
        return "STOP";
      }

      return "CONTINUE";
    },
    controller.signal
  );

  if (checkerResult === "STOPPED") {
    clearInterval(heartbeat);
    try {
      res.end();
    } catch {}
    return;
  }

  // Optional UX nudge
  res.write(
    `event: ${SubmissionStatus.BENCHMARKING}\ndata: {"status":"${SubmissionStatus.BENCHMARKING}"}\n\n`
  );
  await db.submission.update({
    where: { id: submission.id },
    data: {
      status: SubmissionStatus.BENCHMARKING,
      passedTests,
      totalTests,
    },
  });

  // -------------------------------
  // PHASE 2: BENCHMARK (proxy SSE)
  // -------------------------------
  const benchResults: BenchmarkResultResponse["result"][] = [];

  await proxyUpstreamSSE(
    res,
    `${env.MODAL_ENDPOINT}/benchmark-${submission.gpuType ?? "t4"}`,
    payload,
    async (evt: import("~/types/submission").SubmissionResponse) => {
      const s = evt?.status as string | undefined;
      if (!s) return "CONTINUE";

      if (s === SubmissionStatus.BENCHMARK_RESULT) {
        const r = evt as import("~/types/submission").BenchmarkResultResponse;
        if (r.result) {
          benchResults.push(r.result);
          await db.submission.update({
            where: { id: submission.id },
            data: { benchmarkResults: benchResults },
          });
        }
        return "CONTINUE";
      }

      if (s === SubmissionStatus.BENCHMARKED) {
        const r = evt as import("~/types/submission").BenchmarkedResponse;
        // Worker may or may not emit ACCEPTED. Persist final numbers here.
        const updateData: Partial<Record<string, unknown>> & {
          status: SubmissionStatusType;
        } = {
          status: SubmissionStatus.ACCEPTED,
          benchmarkResults: benchResults,
        };
        if (typeof r.avg_runtime_ms === "number")
          (updateData as Record<string, unknown>).runtime = r.avg_runtime_ms;
        if (typeof r.avg_gflops === "number")
          (updateData as Record<string, unknown>).gflops = r.avg_gflops;

        await db.submission.update({
          where: { id: submission.id },
          data: updateData,
        });

        // If your worker does NOT emit ACCEPTED, you can also emit it here:
        res.write(
          `event: ${SubmissionStatus.ACCEPTED}\ndata: ${JSON.stringify({
            avg_runtime_ms: r.avg_runtime_ms,
            avg_gflops: r.avg_gflops,
            benchmark_results: benchResults,
            total_tests: benchResults.length,
          })}\n\n`
        );

        return "CONTINUE";
      }

      if (isSubmissionError(s)) {
        const err = evt as Partial<import("~/types/submission").ErrorResponse>;
        await db.submission.update({
          where: { id: submission.id },
          data: {
            status: s,
            errorMessage: err.message ?? "Unknown error",
            errorDetails: err.details ?? "",
          },
        });
        return "STOP";
      }

      return "CONTINUE";
    },
    controller.signal
  );

  clearInterval(heartbeat);
  try {
    res.end();
  } catch {}
}
