/**
 * AMD GPU Task Runner - Shared Logic
 *
 * This module provides shared functionality for executing AMD GPU tasks via dstack.ai:
 * - Spawns Python runner (amd_task_runner.py) to execute tasks
 * - Streams SSE events back to the client (matching NVIDIA format)
 * - Updates database when checker/benchmark completes with real test counts
 * - Supports checker, benchmark, and full (checker+benchmark) endpoints
 *
 * Event Format (matches NVIDIA):
 *   IN_QUEUE -> PROVISIONING -> COMPILING -> CHECKING -> TEST_RESULT* -> CHECKED
 *                                         -> BENCHMARKING -> BENCHMARK_RESULT* -> BENCHMARKED
 *   Error events: COMPILE_ERROR, WRONG_ANSWER, RUNTIME_ERROR, ERROR
 */

import { type NextApiResponse } from "next";
import { spawn } from "child_process";
import path from "path";
import { db } from "~/server/db";
import { SubmissionStatus, SubmissionError } from "~/types/submission";

export interface TaskSubmissionPayload {
  solution_code: string;
  problem: string;
  problem_def: string;
  gpu_type: string;
  dtype: string;
  language: string;
  endpoint?: string; // checker, benchmark, full (default)
  submission_id?: string; // Database submission ID for persistence
}

// Track parsed results from events
interface ParsedResults {
  passedTests: number;
  totalTests: number;
  avgRuntimeMs: number;
  avgGflops: number | null;
  status: string;
}

/**
 * Execute Python task runner and stream SSE output
 * This function spawns the Python process, streams events to the client,
 * and updates the database when appropriate.
 *
 * @param payload - Task submission payload with code, problem, GPU type, etc.
 * @param res - Next.js API response object for streaming SSE events
 * @returns Promise that resolves when the task completes or rejects on error
 */
export async function executePythonRunner(
  payload: TaskSubmissionPayload,
  res: NextApiResponse
): Promise<void> {
  return new Promise((resolve, reject) => {
    const submissionId = payload.submission_id || "unknown";
    console.log(
      `[AMD Runner] Starting AMD task runner for submission ${submissionId}`
    );
    console.log(
      `[AMD Runner] GPU Type: ${payload.gpu_type}, Endpoint: ${payload.endpoint}, Language: ${payload.language}`
    );

    const enginePath = path.join(process.cwd(), "engine");
    const runnerScript = path.join(enginePath, "amd_task_runner.py");
    const venvPython = path.join(enginePath, ".venv", "bin", "python");

    console.log(`[AMD Runner] Spawning Python process with payload:`, {
      problem: payload.problem,
      gpu_type: payload.gpu_type,
      endpoint: payload.endpoint,
      submission_id: submissionId,
      code_length: payload.solution_code.length,
    });

    // Spawn Python process using virtual environment's Python
    const pythonProcess = spawn(
      venvPython,
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

    console.log(
      `[AMD Runner] Python process started with PID: ${pythonProcess.pid}`
    );

    let buffer = "";
    let hasOutput = false;

    // Track results from events for database update
    const parsedResults: ParsedResults = {
      passedTests: 0,
      totalTests: 0,
      avgRuntimeMs: 0,
      avgGflops: null,
      status: "PENDING",
    };

    // Handle stdout - Stream SSE events and update database
    pythonProcess.stdout.on("data", async (data: Buffer) => {
      hasOutput = true;
      const chunk = data.toString();
      console.log(`[AMD Runner] Python stdout: ${chunk}`);
      buffer += chunk;

      // Process complete lines
      const lines = buffer.split("\n");
      buffer = lines.pop() || "";

      for (const line of lines) {
        if (line.trim()) {
          // Filter and process only SSE events (lines starting with SSE_EVENT:)
          if (line.startsWith("SSE_EVENT:")) {
            // Strip the SSE_EVENT: prefix and write to response
            const sseContent = line.substring("SSE_EVENT:".length);
            res.write(sseContent + "\n");

            // Parse SSE events to track results and update database
            if (payload.submission_id) {
              try {
                const dataMatch = sseContent.match(/^data: (.+)$/);
                if (dataMatch && dataMatch[1]) {
                  const eventData = JSON.parse(dataMatch[1]);
                  const status = eventData.status;

                  // Track checker results
                  if (status === "CHECKED") {
                    parsedResults.passedTests =
                      eventData.passed_tests ?? eventData.total_tests ?? 0;
                    parsedResults.totalTests = eventData.total_tests ?? 0;
                    parsedResults.status = "CHECKED";
                    console.log(
                      `[AMD Runner] Checker completed: ${parsedResults.passedTests}/${parsedResults.totalTests} tests passed`
                    );
                  }

                  // Track wrong answer (partial pass)
                  if (status === "WRONG_ANSWER") {
                    parsedResults.passedTests = eventData.passed_tests ?? 0;
                    parsedResults.totalTests = eventData.total_tests ?? 0;
                    parsedResults.status = "WRONG_ANSWER";
                    console.log(
                      `[AMD Runner] Wrong answer: ${parsedResults.passedTests}/${parsedResults.totalTests} tests passed`
                    );

                    // Update database immediately for wrong answer
                    await db.submission.update({
                      where: { id: payload.submission_id },
                      data: {
                        status: SubmissionStatus.WRONG_ANSWER,
                        passedTests: parsedResults.passedTests,
                        totalTests: parsedResults.totalTests,
                      },
                    });
                  }

                  // Track benchmark results
                  if (status === "BENCHMARKED") {
                    parsedResults.avgRuntimeMs = eventData.avg_runtime_ms ?? 0;
                    parsedResults.avgGflops = eventData.avg_gflops ?? null;
                    parsedResults.status = "BENCHMARKED";
                    console.log(
                      `[AMD Runner] Benchmark completed: ${parsedResults.avgRuntimeMs}ms, ${parsedResults.avgGflops} GFLOPS`
                    );

                    // Update database with real results
                    await db.submission.update({
                      where: { id: payload.submission_id },
                      data: {
                        status: SubmissionStatus.ACCEPTED,
                        runtime: parsedResults.avgRuntimeMs,
                        gflops: parsedResults.avgGflops,
                        passedTests:
                          parsedResults.passedTests || parsedResults.totalTests,
                        totalTests:
                          parsedResults.totalTests ||
                          eventData.total_tests ||
                          1,
                      },
                    });
                    console.log(
                      `[AMD Runner] Successfully updated submission ${payload.submission_id}`
                    );
                  }

                  // Handle errors
                  if (
                    status === "ERROR" ||
                    status === "COMPILE_ERROR" ||
                    status === "RUNTIME_ERROR"
                  ) {
                    await db.submission.update({
                      where: { id: payload.submission_id },
                      data: {
                        status:
                          status === "COMPILE_ERROR"
                            ? SubmissionError.COMPILE_ERROR
                            : SubmissionError.RUNTIME_ERROR,
                        errorMessage:
                          eventData.error ||
                          eventData.message ||
                          "Execution failed",
                        errorDetails: eventData.details || "",
                      },
                    });
                    console.log(
                      `[AMD Runner] Updated submission ${payload.submission_id} to ${status}`
                    );
                  }
                }
              } catch (e) {
                // Ignore parsing errors, continue streaming
                console.error(
                  `[AMD Runner] Event parsing error for submission ${payload.submission_id}:`,
                  e
                );
              }
            }
          }
          // Non-SSE lines (Python logs) are not written to response, only logged
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
      const stderrOutput = data.toString();
      console.error(
        `[AMD Runner] Python stderr for submission ${submissionId}:`,
        stderrOutput
      );
    });

    // Handle process completion
    pythonProcess.on("close", (code) => {
      console.log(
        `[AMD Runner] Python process exited with code: ${code} for submission ${submissionId}`
      );

      // Flush any remaining buffer
      if (buffer.trim()) {
        res.write(buffer + "\n");
      }

      if (code === 0) {
        console.log(
          `[AMD Runner] Task completed successfully for submission ${submissionId}`
        );
        resolve();
      } else {
        console.error(
          `[AMD Runner] Task failed with exit code ${code} for submission ${submissionId}`
        );
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
      console.error(
        `[AMD Runner] Python process error for submission ${submissionId}:`,
        error.message
      );
      console.error(`[AMD Runner] Error stack:`, error.stack);

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
