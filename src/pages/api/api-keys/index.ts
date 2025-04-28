// pages/api/api-keys/index.ts
import { type NextApiRequest, type NextApiResponse } from "next";
import { getServerSession } from "next-auth/next";
import { authConfig } from "~/server/auth/config";
import crypto from "crypto";
import argon2 from "argon2";
import { db } from "~/server/db";

const generateApiKey = () => {
  const prefix = crypto.randomBytes(6).toString("hex");
  const keyBody = crypto.randomBytes(28).toString("hex");
  return {
    fullKey: `tsra_${prefix}_${keyBody}`,
    prefix,
    keyBody,
  };
};

const argon2Options = {
  type: argon2.argon2id,
  memoryCost: 4096,
  timeCost: 2,
  parallelism: 1,
};

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

    if (!name || name.length < 3 || typeof name !== "string") {
      return res
        .status(400)
        .json({ error: "Name must be at least 3 characters long" });
    }

    const { fullKey, prefix, keyBody } = generateApiKey();
    const hashedKey = await argon2.hash(keyBody, argon2Options);

    const expiresAt = expiresIn
      ? new Date(Date.now() + expiresIn)
      : new Date(Date.now() + 30 * 24 * 60 * 60 * 1000);

    const apiKey = await db.apiKey.create({
      data: {
        userId: session.user.id,
        name,
        keyPrefix: prefix,
        key: hashedKey,
        expiresAt,
      },
    });

    return res.status(201).json({
      id: apiKey.id,
      name: apiKey.name,
      key: fullKey,
      createdAt: apiKey.createdAt,
      expiresAt: apiKey.expiresAt,
    });
  }

  return res.status(405).json({ error: "Method not allowed" });
}
