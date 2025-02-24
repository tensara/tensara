import { type NextApiRequest, type NextApiResponse } from "next";
import { db } from "~/server/db";

const FINAL_STATUSES = ["ACCEPTED", "ERROR", "WRONG_ANSWER"] as const;
const MAX_RETRIES = 3;
const RETRY_DELAY = 1000; // 1 second

export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse
) {
  const { id } = req.query;

  if (req.method !== "GET") {
    res.status(405).end();
    return;
  }

  // Set SSE headers
  res.setHeader("Content-Type", "text/event-stream");
  res.setHeader("Cache-Control", "no-cache");
  res.setHeader("Connection", "keep-alive");

  try {
    let retryCount = 0;
    let initialSubmission = null;

    // Try to get the initial submission, with retries
    while (retryCount < MAX_RETRIES && !initialSubmission) {
      initialSubmission = await db.submission.findUnique({
        where: { id: id as string },
        select: {
          id: true,
          status: true,
          runtime: true,
          gflops: true,
          passedTests: true,
          totalTests: true,
          errorMessage: true,
          errorDetails: true,
          benchmarkResults: true,
        },
      });

      if (!initialSubmission) {
        retryCount++;
        if (retryCount < MAX_RETRIES) {
          await new Promise(resolve => setTimeout(resolve, RETRY_DELAY));
        }
      }
    }

    if (!initialSubmission) {
      res.status(404).end();
      return;
    }

    console.log("[status] Sending initial submission data:", initialSubmission);
    const initialMessage = `id: ${Date.now()}\ndata: ${JSON.stringify(initialSubmission)}\n\n`;
    console.log("[status] Writing message:", initialMessage);
    res.write(initialMessage);
    res.flushHeaders();

    const interval = setInterval(async () => {
      try {
        const submission = await db.submission.findUnique({
          where: { id: id as string },
          select: {
            id: true,
            status: true,
            runtime: true,
            gflops: true,
            passedTests: true,
            totalTests: true,
            errorMessage: true,
            errorDetails: true,
            benchmarkResults: true,
          },
        });

        if (!submission) {
          console.log("[status] No submission found, ending stream");
          clearInterval(interval);
          res.end();
          return;
        }
        const message = `id: ${Date.now()}\ndata: ${JSON.stringify(submission)}\n\n`;

        res.write(message);
        (res as any).flush?.();

        if (submission.status && FINAL_STATUSES.includes(submission.status as any)) {
          console.log("[status] Final status reached, ending stream:", submission.status);
          clearInterval(interval);
          res.end();
        }
      } catch (error) {
        console.error("[status] Error in SSE interval:", error);
        clearInterval(interval);
        res.end();
      }
    }, 1000);

    req.on("close", () => {
      console.log("[status] Client disconnected, cleaning up");
      clearInterval(interval);
    });
  } catch (error) {
    console.error("Error in SSE handler:", error);
    res.status(500).end();
  }
}
