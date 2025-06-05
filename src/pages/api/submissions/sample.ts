import { type NextApiRequest, type NextApiResponse } from "next";
import { env } from "~/env";
import { DateTime } from "luxon";
import { combinedAuth } from "~/server/auth";
import { db } from "~/server/db";

import type { ServerResponse } from "http";
import {
  SampleStatus,
  type SampleStatusType,
  type SampleEvent,
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
  if (!session || "error" in session) {
    res.status(401).json({ error: session?.error ?? "Not authenticated" });
    return;
  }

  const user = await db.user.findUnique({
    where: { id: session.user.id },
    select: { sampleSubmissionCount: true, lastSampleSubmissionReset: true },
  });

  if (!user) {
    res.status(404).json({ error: "User not found" });
    return;
  }

  const today = DateTime.now().startOf("day");
  let sampleSubmissionCount = user.sampleSubmissionCount;
  let lastSampleSubmissionReset = DateTime.fromJSDate(
    user.lastSampleSubmissionReset
  ).startOf("day");

  if (lastSampleSubmissionReset < today) {
    // It's a new day, reset the limit
    await db.user.update({
      where: { id: session.user.id },
      data: {
        sampleSubmissionCount: 4,
        lastSampleSubmissionReset: today.toJSDate(),
      },
    });
    sampleSubmissionCount = 4;
  }

  if (sampleSubmissionCount <= 0) {
    res.status(429).json({ error: "Too Many Requests: Sample limit exceeded" });
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
      error: `Missing required fields: ${missingFields
        .map(([key]) => key)
        .join(", ")}`,
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

  const sendSSE = (event: string, data: unknown) => {
    const payload = `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
    res.write(payload);
    try {
      const flushableRes = res as ServerResponse & { flush?: () => void };
      if (typeof flushableRes.flush === "function") {
        flushableRes.flush();
      } else if (flushableRes.flushHeaders) {
        flushableRes.flushHeaders();
      }
    } catch (err) {
      console.warn("Flush error:", err);
    }
  };

  const heartbeat = setInterval(() => {
    sendSSE("heartbeat", { timestamp: Date.now() });
  }, 30000);

  try {
    sendSSE(SampleStatus.IN_QUEUE, { status: SampleStatus.IN_QUEUE });

    const sampleResponse = await fetch(
      `${env.MODAL_ENDPOINT}/sample-${gpuType}`,
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

    if (!sampleResponse.ok) {
      const errorText = await sampleResponse.text();
      sendSSE(SampleStatus.ERROR, {
        status: SampleStatus.ERROR,
        message: `Sample API returned ${sampleResponse.status}`,
        details: errorText,
      });
      res.end();
      return;
    }

    const reader = sampleResponse.body?.getReader();
    if (!reader) {
      sendSSE(SampleStatus.ERROR, {
        status: SampleStatus.ERROR,
        message: "No response body from sample runner",
      });
      res.end();
      return;
    }

    let partialMessage = "";
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
            const data = JSON.parse(message.slice(6).trim()) as SampleEvent;

            if (data.status === SampleStatus.COMPILING) {
              sendSSE(SampleStatus.COMPILING, {
                status: SampleStatus.COMPILING,
              });
              continue;
            }

            if (
              data.status === SampleStatus.ERROR ||
              data.status === SampleStatus.COMPILE_ERROR ||
              data.status === SampleStatus.RUNTIME_ERROR
            ) {
              sendSSE(data.status, {
                status: data.status,
                message: data.message,
                details: data.details,
              });
              res.end();
              return;
            }

            if (
              data.status === SampleStatus.PASSED ||
              data.status === SampleStatus.FAILED
            ) {
              // Increment sampleSubmissionCount
              await db.user.update({
                where: { id: session.user.id },
                data: {
                  sampleSubmissionCount: {
                    decrement: 1,
                  },
                },
              });
              const result = {
                status: data.status,
                input: data.input,
                output: data.output,
                debug_info: data.debug_info,
                stdout: data.stdout,
                stderr: data.stderr,
                expected_output: data.expected_output,
              };
              sendSSE("SAMPLE_RESULT", result);
              res.end();
              return;
            }
          } catch (e) {
            console.warn("Failed to parse SSE message", e);
          }
        }
      }
    } finally {
      reader.releaseLock();
    }
  } catch (err) {
    console.error("Sample runner error:", err);
    sendSSE(SampleStatus.ERROR, {
      status: SampleStatus.ERROR,
      message: err instanceof Error ? err.message : "Unknown error",
      details: err instanceof Error ? err.stack : undefined,
    });
  } finally {
    clearInterval(heartbeat);
    res.end();
  }
}
