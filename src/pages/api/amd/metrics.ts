/**
 * AMD DevCloud Metrics API
 *
 * Returns VM orchestrator metrics including:
 * - Cost tracking and credit usage
 * - VM pool status
 * - Task statistics (cold starts vs warm reuses)
 * - Active VMs
 */

import { type NextApiRequest, type NextApiResponse } from "next";
import { spawn } from "child_process";
import path from "path";

export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse
) {
  // Only allow GET requests
  if (req.method !== "GET") {
    res.setHeader("Allow", ["GET"]);
    return res.status(405).end(`Method ${req.method} Not Allowed`);
  }

  try {
    const enginePath = path.join(process.cwd(), "engine");
    const venvPython = path.join(enginePath, ".venv", "bin", "python");

    // Call Python script to get orchestrator metrics
    const pythonProcess = spawn(
      venvPython,
      [
        "-c",
        `
import sys
sys.path.insert(0, "${enginePath}")
from vm_orchestrator_client import get_orchestrator_metrics
import json

try:
    metrics = get_orchestrator_metrics()
    print(json.dumps(metrics))
except Exception as e:
    print(json.dumps({"error": str(e)}), file=sys.stderr)
    sys.exit(1)
        `.trim(),
      ],
      {
        cwd: enginePath,
        env: {
          ...process.env,
          PYTHONUNBUFFERED: "1",
        },
      }
    );

    let stdout = "";
    let stderr = "";

    pythonProcess.stdout.on("data", (data: Buffer) => {
      stdout += data.toString();
    });

    pythonProcess.stderr.on("data", (data: Buffer) => {
      stderr += data.toString();
    });

    pythonProcess.on("close", (code) => {
      if (code === 0 && stdout) {
        try {
          const metrics = JSON.parse(stdout);
          return res.status(200).json(metrics);
        } catch (e) {
          console.error("[AMD Metrics] Failed to parse metrics:", e);
          return res.status(500).json({
            error: "Failed to parse metrics",
            details: stdout,
          });
        }
      } else {
        console.error("[AMD Metrics] Python process failed:", stderr);
        return res.status(500).json({
          error: "Failed to get metrics",
          details: stderr || "Python process exited with non-zero code",
        });
      }
    });

    pythonProcess.on("error", (error) => {
      console.error("[AMD Metrics] Process error:", error);
      return res.status(500).json({
        error: "Failed to spawn Python process",
        details: error.message,
      });
    });
  } catch (error) {
    console.error("[AMD Metrics] Error:", error);
    return res.status(500).json({
      error: error instanceof Error ? error.message : "Unknown error",
    });
  }
}
