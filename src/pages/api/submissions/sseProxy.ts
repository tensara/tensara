/**
 * /server/sseProxy.ts
 *
 * Generic streaming proxy utility for Tensara API routes.
 * - Opens a POST request to an upstream (e.g., Modal GPU endpoint) and pipes its
 *   Server-Sent Event (SSE) stream directly to the client response.
 * - Optionally exposes each SSE frame to an `onEvent` callback for DB updates
 *   or early termination (e.g., on WRONG_ANSWER).
 * - Handles backpressure safely using `ReadableStreamDefaultReader`.
 * - Propagates upstream non-200 status codes and error text to the client.
 * - Aborts cleanly if the client disconnects via an AbortController.
 *
 * Used by both `/api/submissions/direct-submit` and `/api/submissions/sample`
 * for transparent, low-overhead streaming between Modal and the frontend.
 */

import type { NextApiResponse } from "next";
import type { SubmissionResponse } from "~/types/submission";

export type OnEvent = (
  data: SubmissionResponse
) => Promise<"CONTINUE" | "STOP">;

export async function proxyUpstreamSSE(
  res: NextApiResponse,
  url: string,
  payload: unknown,
  onEvent: OnEvent,
  abort: AbortSignal
): Promise<"DONE" | "STOPPED"> {
  const upstream = await fetch(url, {
    method: "POST",
    signal: abort,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (!upstream.ok || !upstream.body) {
    const text = await upstream.text().catch(() => "");
    res.status(upstream.status);
    if (text) res.write(text);
    return "STOPPED";
  }

  const reader = upstream.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      if (!value?.byteLength) continue;

      // 1) forward raw SSE bytes to client
      res.write(Buffer.from(value));

      // 2) also parse locally to trigger DB side-effects
      buffer += decoder.decode(value, { stream: true });
      const frames = buffer.split(/\r?\n\r?\n/);
      buffer = frames.pop() ?? "";

      for (const frame of frames) {
        const dataLine = frame
          .split(/\r?\n/)
          .find((l) => l.startsWith("data: "));
        if (!dataLine) continue;
        let json: SubmissionResponse | undefined;
        try {
          json = JSON.parse(dataLine.slice(6).trim()) as SubmissionResponse;
        } catch {
          continue;
        }
        const decision = await onEvent(json);
        if (decision === "STOP") return "STOPPED";
      }
    }
    return "DONE";
  } finally {
    try {
      reader.releaseLock();
    } catch {}
  }
}
