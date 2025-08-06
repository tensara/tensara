// pages/api/snapshot/create.ts
import { combinedAuth } from "~/server/auth";
import { db } from "~/server/db";
import { type NextApiRequest, type NextApiResponse } from "next";

export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse
) {
  const session = await combinedAuth(req, res);
  if (!session) {
    res.status(401).json({ error: "Not authenticated" });
    return;
  }
  if (session && "error" in session) {
    res.status(401).json({ error: session.error });
    return;
  }

  if (req.method !== "POST") return res.status(405).end();
  type SnapshotRequestBody = {
    files: { name: string; content: string }[];
    main: string;
  };

  const { files, main } = req.body as SnapshotRequestBody;

  const snapshot = await db.snapshot.create({
    data: {
      files,
      main,
      userId: session.user.id ?? "anon",
    },
  });

  res.status(200).json({ id: snapshot.id });
}
