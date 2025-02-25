/* eslint-disable @typescript-eslint/no-unsafe-call */
import { type NextApiRequest, type NextApiResponse } from "next";
import { db } from "~/server/db";

// Define allowed statuses
const FINAL_STATUSES = ["ACCEPTED", "ERROR", "WRONG_ANSWER"] as const;
type FinalStatus = (typeof FINAL_STATUSES)[number];

// Type guard
function isFinalStatus(status: string): status is FinalStatus {
  return FINAL_STATUSES.includes(status as FinalStatus);
}

export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse
) {
  const { id } = req.query;

  if (req.method !== "GET") {
    res.setHeader("Allow", ["GET"]);
    res.status(405).end(`Method ${req.method} Not Allowed`);
    return;
  }


  res.setHeader("Content-Type", "text/event-stream");
  res.setHeader("Cache-Control", "no-cache, no-transform");
  res.setHeader("Connection", "keep-alive");
  res.setHeader("X-Accel-Buffering", "no"); 


  const sendSSE = (data: unknown) => {
    res.write(`data: ${JSON.stringify(data)}\n\n`);
    // @ts-expect-error - flush exists on Node response objects
    if (res.flush) res.flush();
  };

  try {
    const initialSubmission = await db.submission.findUnique({
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
      res.status(404).end();
      return;
    }

    console.log("[status] initial submission", initialSubmission);
    sendSSE(initialSubmission);

    const interval = setInterval(() => {
      void (async () => {
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
            clearInterval(interval);
            res.end();
            return;
          }
          
          sendSSE(submission);

          // Check if we've reached a final status
          if (submission.status && isFinalStatus(submission.status)) {
            clearInterval(interval);
            res.end();
          }
        } catch (error) {
          console.error("Error in SSE interval:", error);
          clearInterval(interval);
          res.end();
        }
      })();
    }, 1000);

    // Clean up on client disconnect
    req.on("close", () => {
      clearInterval(interval);
    });
  } catch (error) {
    console.error("Error in SSE handler:", error);
    res.status(500).end();
  }
}
