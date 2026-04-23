import { type NextApiRequest, type NextApiResponse } from "next";
import { db } from "~/server/db";
import { env } from "~/env";
import { combinedAuth } from "~/server/auth";
import { checkRateLimit } from "~/hooks/useRateLimit";
import { type ProgrammingLanguage } from "~/types/misc";
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

  const { code, language } = req.body as {
    code: string;
    language?: ProgrammingLanguage;
  };
  const selectedLanguage = language ?? "cuda";

  if (!code) {
    res.status(400).json({
      error: "Missing required field: code",
    });
    return;
  }

  const supportedLanguages: ProgrammingLanguage[] = ["cuda", "mojo", "cute"];
  if (!supportedLanguages.includes(selectedLanguage)) {
    res.status(400).json({
      error: "Unsupported language for sandbox",
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
      language: selectedLanguage,
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

    const upstreamController = new AbortController();
    const sandboxResponse = await fetch(env.MODAL_ENDPOINT + "/sandbox-T4", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        code: submission.code,
        language: submission.language,
      }),
      signal: upstreamController.signal,
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

    const cancelUpstream = async (reason: string) => {
      upstreamController.abort();
      try {
        await reader.cancel(reason);
      } catch (cancelError) {
        console.warn("Failed to cancel sandbox stream:", cancelError);
      }
    };

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
              status?: string;
              [key: string]: unknown;
            };
            const response_status = parsed?.status;
            if (!response_status) {
              continue;
            }

            if (response_status === SubmissionStatus.COMPILING) {
              sendSSE(SubmissionStatus.COMPILING, {
                status: SubmissionStatus.COMPILING,
              });
              continue;
            }

            if (response_status === "SANDBOX_RUNNING") {
              sendSSE("SANDBOX_RUNNING", {
                status: "SANDBOX_RUNNING",
              });
              continue;
            }

            if (response_status === "SANDBOX_OUTPUT") {
              const response = parsed as {
                status: string;
                stream: "stdout" | "stderr";
                line: string;
                timestamp: number;
              };

              sendSSE("SANDBOX_OUTPUT", response);
              continue;
            }

            if (response_status === "SANDBOX_SUCCESS") {
              const response = parsed as {
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
            }

            if (response_status === "SANDBOX_OUTPUT_LIMIT") {
              const response = parsed as {
                status: string;
                message: string;
                stdout?: string;
                stderr?: string;
                details?: string;
              };

              await db.sandboxSubmission.update({
                where: { id: submission.id },
                data: {
                  status: "FAILED",
                  stdout_output: response.stdout,
                  error_message: response.message,
                  error_details: response.details,
                },
              });

              sendSSE("SANDBOX_OUTPUT_LIMIT", response);

              await cancelUpstream("sandbox-output-limit");

              res.end();
              return;
            }

            if (response_status === "SANDBOX_ERROR") {
              const response = parsed as {
                status: string;
                message: string;
                stdout?: string;
                stderr?: string;
                return_code?: number;
                details?: string;
              };

              await db.sandboxSubmission.update({
                where: { id: submission.id },
                data: {
                  status: "FAILED",
                  stdout_output: response.stdout,
                  error_message: response.message,
                  error_details: response.details,
                },
              });

              sendSSE("SANDBOX_ERROR", response);

              res.end();
              return;
            }

            if (response_status === "SANDBOX_TIMEOUT") {
              const response = parsed as {
                status: string;
                message: string;
                details: string;
              };
              await db.sandboxSubmission.update({
                where: { id: submission.id },
                data: {
                  status: "TIMEOUT",
                  error_message: response.message,
                  error_details: response.details,
                },
              });

              sendSSE("SANDBOX_TIMEOUT", response);

              await cancelUpstream("sandbox-timeout");

              res.end();
              return;
            }

            if (
              response_status === SubmissionStatus.PTX ||
              response_status === SubmissionStatus.SASS ||
              response_status === SubmissionStatus.WARNING
            ) {
              sendSSE(response_status, parsed);
              continue;
            }

            if (isSubmissionError(response_status)) {
              const response = parsed as {
                status: SubmissionErrorType;
                error?: string;
                message?: string;
                details: string;
              };

              const isTimeout =
                response_status === SubmissionError.TIME_LIMIT_EXCEEDED;

              await db.sandboxSubmission.update({
                where: { id: submission.id },
                data: {
                  status: isTimeout ? "TIMEOUT" : "FAILED",
                  error_message: response.error ?? response.message,
                  error_details: response.details,
                },
              });

              console.log("Sandbox error:", response);

              if (isTimeout) {
                await cancelUpstream("sandbox-timeout-error");
                sendSSE("SANDBOX_TIMEOUT", {
                  status: "SANDBOX_TIMEOUT",
                  message: response.message ?? "Time limit exceeded",
                  details: response.details,
                });
              } else {
                sendSSE(response_status, response);
              }

              res.end();
              return;
            }

            sendSSE(response_status, parsed);
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
