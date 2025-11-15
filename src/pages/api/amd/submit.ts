/**
 * AMD GPU Task Submission API - Main Endpoint
 *
 * This endpoint handles submission of AMD GPU tasks to dstack.ai:
 * - Authenticates and validates the request
 * - Uses shared executePythonRunner to execute task
 * - Streams SSE events back to frontend
 * - Supports checker, benchmark, sample, and sandbox endpoints
 */

import { type NextApiRequest, type NextApiResponse } from "next";
import { combinedAuth } from "~/server/auth";
import {
  executePythonRunner,
  type TaskSubmissionPayload,
} from "~/server/amd/runner";

export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse
) {
  // Only allow POST requests
  if (req.method !== "POST") {
    res.setHeader("Allow", ["POST"]);
    res.status(405).end(`Method ${req.method} Not Allowed`);
    return;
  }

  // Authenticate user
  const session = await combinedAuth(req, res);
  if (!session || "error" in session) {
    res.status(401).json({
      error: session?.error ?? "Not authenticated",
    });
    return;
  }

  const payload = req.body as TaskSubmissionPayload;

  // Validate required fields
  const requiredFields = [
    "solution_code",
    "problem",
    "problem_def",
    "gpu_type",
    "dtype",
    "language",
  ];

  const missingFields = requiredFields.filter(
    (field) => !payload[field as keyof TaskSubmissionPayload]
  );

  if (missingFields.length > 0) {
    res.status(400).json({
      error: `Missing required fields: ${missingFields.join(", ")}`,
    });
    return;
  }

  // Verify it's an AMD GPU
  if (!payload.gpu_type.startsWith("MI")) {
    res.status(400).json({
      error:
        "This endpoint is only for AMD GPUs (MI210, MI250X, MI300A, MI300X)",
    });
    return;
  }

  // Set up SSE headers
  res.status(200);
  res.setHeader("Content-Type", "text/event-stream");
  res.setHeader("Cache-Control", "no-cache, no-transform");
  res.setHeader("Connection", "keep-alive");
  res.setHeader("X-Accel-Buffering", "no");
  res.setHeader("Transfer-Encoding", "chunked");
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "POST, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type");

  // Handle client disconnect
  let clientDisconnected = false;
  req.on("close", () => {
    clientDisconnected = true;
    console.log("[AMD Submit] Client disconnected");
  });

  try {
    // Send initial IN_QUEUE status
    res.write(
      `event: IN_QUEUE\ndata: ${JSON.stringify({
        status: "IN_QUEUE",
        message: "Task queued for AMD GPU execution",
      })}\n\n`
    );

    // Execute Python runner and stream results
    await executePythonRunner(payload, res);

    // End response if not already ended
    if (!res.writableEnded && !clientDisconnected) {
      res.end();
    }
  } catch (error) {
    console.error("[AMD Submit] Error:", error);

    // Send error event if client still connected
    if (!res.writableEnded && !clientDisconnected) {
      const errorMessage =
        error instanceof Error ? error.message : "Unknown error";

      res.write(
        `event: ERROR\ndata: ${JSON.stringify({
          status: "ERROR",
          error: errorMessage,
          message: "Task execution failed",
          details: error instanceof Error ? error.stack : undefined,
        })}\n\n`
      );

      res.end();
    }
  }
}
