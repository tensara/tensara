/**
 * AMD GPU Task Submission API - Main Endpoint
 *
 * This endpoint handles submission of AMD GPU tasks to dstack.ai:
 * - Authenticates and validates the request
 * - Spawns Python runner (amd_task_runner.py) to execute task
 * - Streams SSE events back to frontend
 * - Supports checker, benchmark, sample, and sandbox endpoints
 */

import { type NextApiRequest, type NextApiResponse } from "next";
import { spawn } from "child_process";
import path from "path";
import { combinedAuth } from "~/server/auth";

interface TaskSubmissionPayload {
  solution_code: string;
  problem: string;
  problem_def: string;
  gpu_type: string;
  dtype: string;
  language: string;
  endpoint?: string; // checker, benchmark, sample, sandbox
}

/**
 * Execute Python task runner and stream SSE output
 */
async function executePythonRunner(
  payload: TaskSubmissionPayload,
  res: NextApiResponse
): Promise<void> {
  return new Promise((resolve, reject) => {
    const enginePath = path.join(process.cwd(), "engine");
    const runnerScript = path.join(enginePath, "amd_task_runner.py");

    // Spawn Python process
    const pythonProcess = spawn(
      "python3",
      [
        runnerScript,
        JSON.stringify(payload), // Pass payload as command-line argument
      ],
      {
        cwd: enginePath,
        env: {
          ...process.env,
          PYTHONUNBUFFERED: "1", // Disable Python output buffering
        },
      }
    );

    let buffer = "";
    let hasOutput = false;

    // Handle stdout - Stream SSE events
    pythonProcess.stdout.on("data", (data: Buffer) => {
      hasOutput = true;
      buffer += data.toString();

      // Process complete lines
      const lines = buffer.split("\n");
      buffer = lines.pop() || "";

      for (const line of lines) {
        if (line.trim()) {
          res.write(line + "\n");
        }
      }

      // Flush output immediately
      try {
        if ("flush" in res && typeof (res as any).flush === "function") {
          (res as any).flush();
        }
      } catch (e) {
        // Ignore flush errors
      }
    });

    // Handle stderr - Log errors
    pythonProcess.stderr.on("data", (data: Buffer) => {
      console.error("[Python stderr]:", data.toString());
    });

    // Handle process completion
    pythonProcess.on("close", (code) => {
      // Flush any remaining buffer
      if (buffer.trim()) {
        res.write(buffer + "\n");
      }

      if (code === 0) {
        resolve();
      } else {
        // Send error event if no output was sent
        if (!hasOutput) {
          res.write(
            `event: ERROR\ndata: ${JSON.stringify({
              status: "ERROR",
              error: `Python process exited with code ${code}`,
              message: "Task execution failed",
            })}\n\n`
          );
        }
        reject(new Error(`Python process exited with code ${code}`));
      }
    });

    // Handle process errors
    pythonProcess.on("error", (error) => {
      console.error("[Python process error]:", error);

      res.write(
        `event: ERROR\ndata: ${JSON.stringify({
          status: "ERROR",
          error: error.message,
          message: "Failed to start Python process",
          details:
            "Ensure Python 3 is installed and engine/amd_task_runner.py exists",
        })}\n\n`
      );

      reject(error);
    });

    // Handle timeout (15 minutes max)
    const timeout = setTimeout(
      () => {
        pythonProcess.kill("SIGTERM");

        res.write(
          `event: ERROR\ndata: ${JSON.stringify({
            status: "ERROR",
            error: "Task execution timeout",
            message: "Task exceeded maximum execution time (15 minutes)",
          })}\n\n`
        );

        reject(new Error("Task execution timeout"));
      },
      15 * 60 * 1000
    );

    // Clear timeout on completion
    pythonProcess.on("close", () => {
      clearTimeout(timeout);
    });
  });
}

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
