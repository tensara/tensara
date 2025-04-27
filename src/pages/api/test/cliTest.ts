// pages/api/protected-data.ts
import { type NextApiRequest, type NextApiResponse } from "next";
import { combinedAuth } from "~/server/auth";

async function handler(req: NextApiRequest, res: NextApiResponse) {
  const authResult = await combinedAuth(req, res);
  if (!authResult) {
    return res.status(401).json({ error: "Unauthorized" });
  }
  return res.status(200).json({
    success: true,
    userId: authResult.user.id,
    message: `Hello, ${authResult.user.name ?? "User"}! This is protected data.`,
    timestamp: new Date().toISOString(),
  });
}

export default handler;
