/**
 * /api/submissions/sample.ts
 *
 * API route that proxies sample submissions to the Modal GPU runner.
 * - Authenticates the user and validates the request body.
 * - Resets and decrements the user’s daily sample submission quota atomically.
 * - Streams Modal’s raw SSE response directly back to the frontend (true proxy).
 * - Aborts upstream request cleanly if the client disconnects.
 *
 * This is the main backend entrypoint for Tensara’s "Run Sample" feature.
 */

import { type NextApiRequest, type NextApiResponse } from "next";
import { env } from "~/env";
import { DateTime } from "luxon";
import { combinedAuth } from "~/server/auth";
import { db } from "~/server/db";

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

  // ----- Validate body -----
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

  // ----- Atomic daily reset + upfront decrement -----
  // Reset to 200 if day changed, then decrement by 1 if > 0. Count this attempt
  // regardless of success (your Option B).
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

    // Decrement only if > 0 (use updateMany for conditional)
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

  // ----- Prepare streaming proxy to Modal -----
  // We proxy raw SSE bytes from Modal to the client. No wrapping, no re-parsing.
  res.status(200);
  res.setHeader("Content-Type", "text/event-stream");
  res.setHeader("Cache-Control", "no-cache, no-transform");
  res.setHeader("Connection", "keep-alive");
  res.setHeader("X-Accel-Buffering", "no");
  // Note: do NOT set "Transfer-Encoding" manually; Node handles it.

  const abort = new AbortController();
  req.on("close", () => abort.abort());

  try {
    const upstream = await fetch(`${env.MODAL_ENDPOINT}/sample-${gpuType}`, {
      method: "POST",
      signal: abort.signal,
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        solution_code: code,
        problem: problemSlug,
        problem_def: problem.definition,
        gpu_type: gpuType,
        dtype: "float32",
        language,
      }),
    });

    if (!upstream.ok || !upstream.body) {
      // Bubble up the status and any body text for quick diagnosis
      const text = await upstream.text().catch(() => "");
      res.status(upstream.status);
      if (text) res.write(text);
      res.end();
      return;
    }

    // Pipe the SSE stream as-is
    const reader = upstream.body.getReader();

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      if (value?.byteLength) {
        // value: Uint8Array — write directly
        res.write(Buffer.from(value));
      }
    }
  } catch (e: unknown) {
    if (abort.signal.aborted) {
      // client disconnected — just end silently
      try {
        res.end();
      } catch {}
      return;
    }

    // Safely extract a message from unknown error
    let message = "Upstream error";
    if (e instanceof Error) {
      message = e.message;
    } else if (typeof e === "object" && e !== null && "message" in e) {
      const m = (e as Record<string, unknown>).message;
      if (typeof m === "string") message = m;
    }

    res.write(
      `event: ERROR\ndata: ${JSON.stringify({ status: "ERROR", message })}\n\n`
    );
  } finally {
    try {
      res.end();
    } catch {}
  }
}
