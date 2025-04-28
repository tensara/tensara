import { getServerSession, type Session } from "next-auth";
import { type GetServerSidePropsContext } from "next";
import { authConfig } from "./config";
import { db } from "../db";
import argon2 from "argon2";

type AuthResult = Session | { error: string } | null;

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
): Promise<AuthResult> => {
  if (!authorization) {
    return { error: "No authorization header" };
  }

  if (!authorization.startsWith("Bearer ")) {
    return { error: "Invalid authorization header" };
  }

  try {
    const apiKey = authorization.substring(7); // Remove "Bearer " prefix
    const parts = apiKey.split("_");
    if (parts.length !== 3 || parts[0] !== "tsra") {
      return { error: "Invalid API key" };
    }

    const [, prefix, body] = parts;
    if (!body) {
      return { error: "Invalid API key" };
    }

    const apiKeyRecord = await db.apiKey.findFirst({
      where: {
        keyPrefix: prefix,
      },
      include: { user: true },
    });

    if (!apiKeyRecord) {
      return { error: "Invalid API key" };
    }

    try {
      const matches = await argon2.verify(apiKeyRecord.key, body);
      if (!matches) {
        return { error: "Invalid API key" };
      }
    } catch (err) {
      console.error("API key verification error:", err);
      return { error: "Invalid API key" };
    }

    if (apiKeyRecord.expiresAt && apiKeyRecord.expiresAt < new Date()) {
      return {
        error: "API key expired, generate a new one at https://tensara.org",
      };
    }

    const session = {
      user: apiKeyRecord.user,
      expires: apiKeyRecord.expiresAt?.toISOString(),
    } as Session;
    return session;
  } catch (error) {
    console.error("API key authentication error", error);
    return { error: "API key authentication error" };
  }
};

export const combinedAuth = async (
  req?: GetServerSidePropsContext["req"],
  res?: GetServerSidePropsContext["res"]
): Promise<AuthResult> => {
  return (
    (await auth(req, res)) ?? (await authAPIKey(req?.headers.authorization))
  );
};

export { authConfig as config };
