import { getServerSession, type Session } from "next-auth";
import { type GetServerSidePropsContext } from "next";
import { authConfig } from "./config";
import { db } from "../db";
import argon2 from "argon2";

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
  if (!authorization) {
    return null;
  }

  if (!authorization.startsWith("Bearer ")) {
    return null;
  }

  try {
    const apiKey = authorization.substring(7); // Remove "Bearer " prefix
    const parts = apiKey.split("_");
    if (parts.length !== 3 || parts[0] !== "tsra") {
      return null;
    }

    const [_, prefix, body] = parts;
    if (!body) {
      return null;
    }

    const apiKeyRecord = await db.apiKey.findFirst({
      where: {
        keyPrefix: prefix,
        expiresAt: { gte: new Date() },
      },
      include: { user: true },
    });

    if (!apiKeyRecord) {
      return null;
    }

    try {
      const matches = await argon2.verify(apiKeyRecord.key, body);
      if (!matches) {
        return null;
      }
    } catch (err) {
      console.error("API key verification error:", err);
      return null;
    }

    const session = {
      user: apiKeyRecord.user,
      expires: apiKeyRecord.expiresAt?.toISOString(),
    } as Session;
    return session;
  } catch (error) {
    console.error("API key authentication error", error);
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
