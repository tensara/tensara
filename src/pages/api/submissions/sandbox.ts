import { type NextApiRequest, type NextApiResponse } from "next";
import { db } from "~/server/db";
import { env } from "~/env";
import { combinedAuth } from "~/server/auth";
import { checkRateLimit } from "~/hooks/useRateLimit";
import { SubmissionError, SubmissionStatus } from "~/types/submission";
import type { SubmissionErrorType } from "~/types/submission";
import { isSubmissionError } from "~/types/submission";

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

  const { code, gpuType } = req.body as { code: string; gpuType?: string };

  if (!code) {
    res.status(400).json({
      error: "Missing required field: code",
    });
    return;
  }

  // Default to T4 if no GPU type specified
  const selectedGpuType = gpuType ?? "T4";

  const rateLimit = await checkRateLimit(session.user.id);
  if (!rateLimit.allowed) {
    res.status(rateLimit.statusCode ?? 429).json({
      status: SubmissionError.RATE_LIMIT_EXCEEDED as SubmissionErrorType,
      error: rateLimit.error,
      details: rateLimit.error,
    });
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

  const submission = await db.sandboxSubmission.create({
    data: {
      code: code,
      language: "cuda",
      status: "IN_QUEUE",
      user: { connect: { id: session.user.id } },
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

    await db.sandboxSubmission.update({
      where: { id: submission.id },
      data: {
        status: "RUNNING",
      },
    });

    sendSSE(SubmissionStatus.IN_QUEUE, {
      status: SubmissionStatus.IN_QUEUE,
    });

    console.log("Starting sandbox process");

    // Route AMD GPUs to new dstack endpoint, others to Modal
    const isAMDGPU = selectedGpuType.startsWith("MI");
    const endpoint = isAMDGPU
      ? `http://localhost:${process.env.PORT || 3000}/api/amd/submit`
      : env.MODAL_ENDPOINT + "/sandbox-" + selectedGpuType;

    const sandboxResponse = await fetch(endpoint, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        solution_code: submission.code,
        code: submission.code,
        language: submission.language,
        problem: "sandbox",
        problem_def: "Sandbox execution",
        gpu_type: selectedGpuType,
        dtype: "float32",
        ...(isAMDGPU && {
          endpoint: "sandbox",
        }),
      }),
    });

    if (!sandboxResponse.ok) {
      const errorText = await sandboxResponse.text();
      sendSSE(SubmissionError.ERROR, {
        error: `Sandbox API returned ${sandboxResponse.status}`,
        message: `Sandbox API returned ${sandboxResponse.status}`,
        details: errorText,
      });

      res.end();
      return;
    }

    const reader = sandboxResponse.body?.getReader();
    if (!reader) {
      sendSSE(SubmissionError.ERROR, {
        error: "No response body from sandbox",
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
            const response_json = message.slice(6).trim();
            const parsed = JSON.parse(response_json) as {
              status: string;
            };
            if (!parsed) {
              continue;
            }
            const response_status = parsed.status;

            if (response_status === SubmissionStatus.COMPILING) {
              sendSSE(SubmissionStatus.COMPILING, {
                status: SubmissionStatus.COMPILING,
              });
            } else if (response_status === "SANDBOX_RUNNING") {
              sendSSE("SANDBOX_RUNNING", {
                status: "SANDBOX_RUNNING",
              });
            } else if (response_status === "SANDBOX_OUTPUT") {
              const response = JSON.parse(response_json) as {
                status: string;
                stream: "stdout" | "stderr";
                line: string;
                timestamp: number;
              };

              sendSSE("SANDBOX_OUTPUT", response);
            } else if (response_status === "SANDBOX_SUCCESS") {
              const response = JSON.parse(response_json) as {
                status: string;
                stdout: string;
                stderr: string;
                return_code: number;
              };
              await db.sandboxSubmission.update({
                where: { id: submission.id },
                data: { status: "COMPLETED" },
              });

              sendSSE("SANDBOX_SUCCESS", response);

              res.end();
              return;
            } else if (response_status === "SANDBOX_ERROR") {
              const response = JSON.parse(response_json) as {
                status: string;
                message: string;
                stdout?: string;
                stderr?: string;
                return_code?: number;
                details?: string;
              };

              await db.sandboxSubmission.update({
                where: { id: submission.id },
                data: { status: "FAILED" },
              });

              sendSSE("SANDBOX_ERROR", response);

              res.end();
              return;
            } else if (response_status === "SANDBOX_TIMEOUT") {
              const response = JSON.parse(response_json) as {
                status: string;
                message: string;
                details: string;
              };
              await db.sandboxSubmission.update({
                where: { id: submission.id },
                data: { status: "TIMEOUT" },
              });

              sendSSE("SANDBOX_TIMEOUT", response);

              res.end();
              return;
            } else if (isSubmissionError(response_status)) {
              const response = JSON.parse(response_json) as {
                status: SubmissionErrorType;
                error: string;
                details: string;
              };

              console.log("Sandbox error:", response);

              sendSSE(response_status, response);

              res.end();
              return;
            } else {
              sendSSE(response_status, {});
            }
          } catch (e) {
            console.error("Failed to parse sandbox SSE data:", e);
            continue;
          }
        }
      }
    } finally {
      reader.releaseLock();
    }
  } catch (error) {
    console.error("Error in sandbox handler:", error);

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
