// pages/api/api-keys/index.ts
import { type NextApiRequest, type NextApiResponse } from "next";
import { getServerSession } from "next-auth/next";
import { authConfig } from "~/server/auth/config";
import crypto from "crypto";
import { db } from "~/server/db";

export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse
) {
  const session = await getServerSession(req, res, authConfig);

  if (!session?.user?.id) {
    return res.status(401).json({ error: "Unauthorized" });
  }

  if (req.method === "GET") {
    const apiKeys = await db.apiKey.findMany({
      where: { userId: session.user.id },
      select: {
        id: true,
        name: true,
        createdAt: true,
        expiresAt: true,
      },
    });

    return res.status(200).json(apiKeys);
  }

  if (req.method === "POST") {
    const { name, expiresIn } = req.body as {
      name: string;
      expiresIn?: number;
    };

    if (!name) {
      return res.status(400).json({ error: "Name is required" });
    }

    const apiKeyValue = `tsra_${crypto.randomBytes(32).toString("hex")}`;

    const hashedKey = crypto
      .createHash("sha256")
      .update(apiKeyValue)
      .digest("hex");

    const expiresAt = expiresIn
      ? new Date(Date.now() + expiresIn)
      : new Date(Date.now() + 30 * 24 * 60 * 60 * 1000); //if no expiration date is provided, it will default to 30 days

    const apiKey = await db.apiKey.create({
      data: {
        userId: session.user.id,
        name,
        key: hashedKey,
        expiresAt,
      },
    });

    return res.status(201).json({
      id: apiKey.id,
      name: apiKey.name,
      key: apiKeyValue,
      createdAt: apiKey.createdAt,
      expiresAt: apiKey.expiresAt,
    });
  }

  return res.status(405).json({ error: "Method not allowed" });
}
