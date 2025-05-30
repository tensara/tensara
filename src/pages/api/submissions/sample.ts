import { type NextApiRequest, type NextApiResponse } from "next";
import { env } from "~/env";
import { combinedAuth } from "~/server/auth";
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
      if ("flush" in res && typeof (res as any).flush === "function") {
        (res as any).flush();
      } else if (res.flushHeaders) {
        res.flushHeaders();
      }
    } catch (err) {
      console.warn("Flush error:", err);
    }
  };

  const heartbeat = setInterval(() => {
    sendSSE("heartbeat", { timestamp: Date.now() });
  }, 30000);

  try {
    sendSSE("IN_QUEUE", { status: "IN_QUEUE" });

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
      sendSSE("ERROR", {
        error: `Sample API returned ${sampleResponse.status}`,
        message: `Sample API returned ${sampleResponse.status}`,
        details: errorText,
      });
      res.end();
      return;
    }

    const reader = sampleResponse.body?.getReader();
    if (!reader) {
      sendSSE("ERROR", { error: "No response body from sample runner" });
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
            const data = JSON.parse(message.slice(6).trim());

            if (data.status === "COMPILING") {
              sendSSE("COMPILING", { status: "COMPILING" });
              continue;
            }

            if (data.status === "ERROR" || data.status === "COMPILE_ERROR") {
              sendSSE("ERROR", {
                status: "ERROR",
                message: data.message,
                details: data.details,
              });
              res.end();
              return;
            }

            if (data.status === "PASSED" || data.status === "FAILED") {
              const result = {
                status: data.status,
                input: data.input,
                output: data.actual_output,
                debug_info: data.debug_info,
                stdout: data.stdout,
                stderr: data.stderr,
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
    sendSSE("ERROR", {
      error: err instanceof Error ? err.message : "Unknown error",
      message: err instanceof Error ? err.message : "Unknown error",
      details: err instanceof Error ? err.stack : undefined,
    });
  } finally {
    clearInterval(heartbeat);
    res.end();
  }
}
