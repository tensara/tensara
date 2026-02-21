// /**
//  * /api/submissions/sample.ts
//  *
//  * API route that proxies sample submissions to the Modal GPU runner.
//  * - Authenticates the user and validates the request body.
//  * - Resets and decrements the user’s daily sample submission quota atomically.
//  * - Streams Modal’s raw SSE response directly back to the frontend (true proxy).
//  * - Aborts upstream request cleanly if the client disconnects.
//  *
//  * This is the main backend entrypoint for Tensara’s "Run Sample" feature.
//  */
import { type NextApiRequest, type NextApiResponse } from "next";
import { env } from "~/env";
import { DateTime } from "luxon";
import { combinedAuth } from "~/server/auth";
import { db } from "~/server/db";
import { proxyUpstreamSSE } from "./sseProxy";

export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse
) {
  res.socket?.setNoDelay?.(true);
  res.socket?.setKeepAlive?.(true);

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
    problemSlug?: string;
    code?: string;
    language?: string;
    gpuType?: string;
  };

  const missing = Object.entries({ problemSlug, code, language, gpuType })
    .filter(([, v]) => !v)
    .map(([k]) => k);
  if (missing.length) {
    res
      .status(400)
      .json({ error: `Missing required fields: ${missing.join(", ")}` });
    return;
  }

  const problem = await db.problem.findUnique({
    where: { slug: problemSlug! },
    select: { definition: true },
  });
  if (!problem) {
    res.status(404).json({ error: "Problem not found" });
    return;
  }

  // Quota: atomic reset + upfront decrement (Option B)
  const today = DateTime.now().startOf("day");
  const quotaOk = await db.$transaction(async (tx) => {
    const u = await tx.user.findUnique({
      where: { id: session.user.id },
      select: { sampleSubmissionCount: true, lastSampleSubmissionReset: true },
    });
    if (!u) return false;

    const lastReset = DateTime.fromJSDate(u.lastSampleSubmissionReset).startOf(
      "day"
    );
    if (lastReset < today) {
      await tx.user.update({
        where: { id: session.user.id },
        data: {
          sampleSubmissionCount: 200,
          lastSampleSubmissionReset: today.toJSDate(),
        },
      });
    }

    const dec = await tx.user.updateMany({
      where: { id: session.user.id, sampleSubmissionCount: { gt: 0 } },
      data: {
        sampleSubmissionCount: { decrement: 1 },
        totalSampleSubmissions: { increment: 1 },
      },
    });
    return dec.count === 1;
  });

  if (!quotaOk) {
    res.status(429).json({ error: "Too Many Requests: Sample limit exceeded" });
    return;
  }

  // SSE headers once
  res.status(200);
  res.setHeader("Content-Type", "text/event-stream");
  res.setHeader("Cache-Control", "no-cache, no-transform");
  res.setHeader("Connection", "keep-alive");
  res.setHeader("X-Accel-Buffering", "no");

  const controller = new AbortController();
  req.on("close", () => controller.abort());

  const payload = {
    solution_code: code,
    problem: problemSlug,
    problem_def: problem.definition,
    gpu_type: gpuType,
    language,
  };

  try {
    // No-op onEvent: we don't need server-side taps for Sample
    const result = await proxyUpstreamSSE(
      res,
      `${env.MODAL_ENDPOINT}/sample-${gpuType}`,
      payload,
      async () => "CONTINUE",
      controller.signal
    );

    // proxyUpstreamSSE already wrote any upstream error status/body; we just end.
    if (result === "STOPPED") return;
  } catch (e) {
    if (controller.signal.aborted) return; // client disconnected
    // Emit a minimal SSE error frame so clients handle it uniformly
    const message =
      e instanceof Error
        ? e.message
        : "Upstream error while streaming sample output";
    res.write(
      `event: ERROR\ndata: ${JSON.stringify({ status: "ERROR", message })}\n\n`
    );
  } finally {
    try {
      res.end();
    } catch {}
  }
}
