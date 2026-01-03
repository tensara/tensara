/**
 * /api/submissions/cancel.ts
 *
 * API endpoint to cancel an in-progress AMD submission.
 * - Verifies user ownership of the submission
 * - Kills the Python process and terminates the dstack task
 * - Deletes the submission from the database (cancelled submissions shouldn't show in UI)
 *
 * This is called when:
 * 1. User clicks the "Cancel" button during AMD provisioning
 * 2. User disconnects/reloads during AMD provisioning (auto-cleanup)
 */

import { type NextApiRequest, type NextApiResponse } from "next";
import { combinedAuth } from "~/server/auth";
import { db } from "~/server/db";
import { cancelAmdSubmission } from "~/server/amd/runner";

export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse
) {
  if (req.method !== "POST") {
    res.setHeader("Allow", ["POST"]);
    return res.status(405).json({ error: "Method not allowed" });
  }

  const session = await combinedAuth(req, res);
  if (!session || "error" in session) {
    return res.status(401).json({ error: "Not authenticated" });
  }

  const { submissionId } = req.body as { submissionId?: string };
  if (!submissionId) {
    return res.status(400).json({ error: "Missing submissionId" });
  }

  console.log(
    `[Cancel API] Cancellation requested for submission ${submissionId} by user ${session.user.id}`
  );

  // Verify ownership
  const submission = await db.submission.findUnique({
    where: { id: submissionId },
    select: { userId: true, status: true },
  });

  if (!submission) {
    // Already deleted or never existed - that's fine (idempotent)
    console.log(
      `[Cancel API] Submission ${submissionId} not found (already deleted or never existed)`
    );
    return res.status(200).json({ success: true, alreadyDeleted: true });
  }

  if (submission.userId !== session.user.id) {
    console.warn(
      `[Cancel API] User ${session.user.id} tried to cancel submission ${submissionId} owned by ${submission.userId}`
    );
    return res.status(403).json({ error: "Not authorized" });
  }

  // Cancel the AMD process if running
  const processKilled = cancelAmdSubmission(submissionId);

  // Delete the submission (cancelled submissions shouldn't show in UI)
  try {
    await db.submission.delete({ where: { id: submissionId } });
    console.log(
      `[Cancel API] Deleted submission ${submissionId}, processKilled: ${processKilled}`
    );
  } catch (e) {
    // Might fail if submission was already deleted by another process
    console.warn(
      `[Cancel API] Failed to delete submission ${submissionId}:`,
      e
    );
  }

  return res.status(200).json({ success: true, processKilled });
}
