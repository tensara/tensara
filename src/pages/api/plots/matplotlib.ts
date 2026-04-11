import { spawn } from "child_process";
import { type NextApiRequest, type NextApiResponse } from "next";
import { z } from "zod";
import { combinedAuth } from "~/server/auth";

const plotPayloadSchema = z.object({
  problemSlug: z.string().min(1),
  title: z.string().min(1),
  subtitle: z.string().optional(),
  xLabel: z.string().min(1),
  yLabel: z.string().min(1),
  metricMode: z.enum(["runtime", "gflops", "temperature"]),
  categories: z.array(z.string()).min(1),
  series: z
    .array(
      z.object({
        id: z.string(),
        label: z.string(),
        color: z.string(),
        points: z.array(z.number().nullable()),
      })
    )
    .min(1),
});

const sanitizeFilename = (value: string) =>
  value.replace(/[^a-z0-9-_]+/gi, "-").toLowerCase();

const runMatplotlibRenderer = async (payload: unknown): Promise<Buffer> => {
  return await new Promise((resolve, reject) => {
    const child = spawn(
      "uv",
      [
        "run",
        "--project",
        "engine",
        "python",
        "engine/scripts/render_matplotlib_plot.py",
      ],
      {
        cwd: process.cwd(),
        stdio: ["pipe", "pipe", "pipe"],
      }
    );

    const stdoutChunks: Buffer[] = [];
    const stderrChunks: Buffer[] = [];

    child.stdout.on("data", (chunk: Buffer | string) => {
      stdoutChunks.push(
        typeof chunk === "string" ? Buffer.from(chunk) : Buffer.from(chunk)
      );
    });

    child.stderr.on("data", (chunk: Buffer | string) => {
      stderrChunks.push(
        typeof chunk === "string" ? Buffer.from(chunk) : Buffer.from(chunk)
      );
    });

    child.on("error", (error) => {
      reject(error);
    });

    child.on("close", (code) => {
      if (code === 0) {
        resolve(Buffer.concat(stdoutChunks));
        return;
      }

      reject(
        new Error(
          Buffer.concat(stderrChunks).toString("utf8").trim() ||
            `Matplotlib renderer exited with code ${code ?? "unknown"}`
        )
      );
    });

    child.stdin.end(JSON.stringify(payload));
  });
};

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
    res.status(401).json({ error: "Not authenticated" });
    return;
  }

  const parsed = plotPayloadSchema.safeParse(req.body);
  if (!parsed.success) {
    res.status(400).json({
      error: "Invalid plot payload",
      details: parsed.error.flatten(),
    });
    return;
  }

  try {
    const svg = await runMatplotlibRenderer(parsed.data);
    const filename = `${sanitizeFilename(parsed.data.problemSlug)}-analysis-plot.svg`;

    res.setHeader("Content-Type", "image/svg+xml; charset=utf-8");
    res.setHeader("Content-Disposition", `attachment; filename="${filename}"`);
    res.status(200).send(svg);
  } catch (error) {
    console.error("Matplotlib render failed", error);
    res.status(500).json({
      error: "Matplotlib render failed",
      details:
        error instanceof Error ? error.message : "Unknown renderer error",
    });
  }
}
