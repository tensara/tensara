import { getServerSession, type Session } from "next-auth";
import { type GetServerSidePropsContext } from "next";
import { authConfig } from "./config";
import { db } from "../db";
import crypto from "crypto";

export const auth = (
  req?: GetServerSidePropsContext["req"],
  res?: GetServerSidePropsContext["res"]
) => {
  if (req && res) {
    return getServerSession(req, res, authConfig);
  }
  return null;
};

export const authAPIKey = async (
  authorization?: string
): Promise<Session | null> => {
  if (!authorization?.startsWith("Bearer ")) {
    return null;
  }

  try {
    const apiKey = authorization.substring(7); // Remove "Bearer " prefix
    const hashedApiKey = crypto
      .createHash("sha256")
      .update(apiKey)
      .digest("hex");

    const apiRecord = await db.apiKey.findUnique({
      where: { key: hashedApiKey },
      include: { user: true },
    });

    if (
      apiRecord?.user &&
      apiRecord.expiresAt &&
      apiRecord.expiresAt >= new Date()
    ) {
      const session = {
        user: apiRecord.user,
        expires: apiRecord.expiresAt.toISOString(),
      } as Session;
      return session;
    }
  } catch (error) {
    console.error("API key authentication error:", error);
  }

  return null;
};

export const combinedAuth = async (
  req?: GetServerSidePropsContext["req"],
  res?: GetServerSidePropsContext["res"]
) => {
  return (
    (await auth(req, res)) ?? (await authAPIKey(req?.headers.authorization))
  );
};

export { authConfig as config };
