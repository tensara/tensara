/**
 * Generate API Key Script
 *
 * Creates an API key for a given user and stores it in the database.
 * The full key is printed to stdout — save it, as the raw key cannot be retrieved later.
 *
 * Usage:
 *   npx tsx src/scripts/generate-api-key.ts --userId <USER_ID> --name <KEY_NAME> [--expiresInDays <DAYS>]
 *   npx tsx src/scripts/generate-api-key.ts --email <EMAIL>   --name <KEY_NAME> [--expiresInDays <DAYS>]
 */

import "dotenv/config";
import { PrismaClient } from "@prisma/client";
import crypto from "crypto";
import argon2 from "argon2";

const prisma = new PrismaClient();

const argon2Options = {
  type: argon2.argon2id,
  memoryCost: 4096,
  timeCost: 2,
  parallelism: 1,
};

function generateApiKey() {
  const prefix = crypto.randomBytes(6).toString("hex");
  const keyBody = crypto.randomBytes(28).toString("hex");
  return {
    fullKey: `tsra_${prefix}_${keyBody}`,
    prefix,
    keyBody,
  };
}

function parseArgs() {
  const args = process.argv.slice(2);
  const parsed: Record<string, string> = {};

  for (let i = 0; i < args.length; i += 2) {
    const key = args[i];
    const value = args[i + 1];
    if (key && value && key.startsWith("--")) {
      parsed[key.slice(2)] = value;
    }
  }

  return parsed;
}

async function main() {
  const args = parseArgs();
  const { userId, email, name, expiresInDays } = args;

  if (!name) {
    console.error("Error: --name is required");
    console.error(
      "Usage: npx tsx src/scripts/generate-api-key.ts --userId <USER_ID> --name <KEY_NAME> [--expiresInDays <DAYS>]"
    );
    console.error(
      "       npx tsx src/scripts/generate-api-key.ts --email <EMAIL>   --name <KEY_NAME> [--expiresInDays <DAYS>]"
    );
    process.exit(1);
  }

  if (!userId && !email) {
    console.error("Error: either --userId or --email is required");
    process.exit(1);
  }

  // Resolve or create user
  let resolvedUserId: string;
  if (userId) {
    const user = await prisma.user.findUnique({ where: { id: userId } });
    if (!user) {
      console.error(`Error: no user found with id "${userId}"`);
      process.exit(1);
    }
    resolvedUserId = user.id;
    console.log(`Found user: ${user.name ?? user.email ?? user.id}`);
  } else {
    let user = await prisma.user.findUnique({ where: { email } });
    if (!user) {
      console.log(`No user found with email "${email}" — creating one...`);
      user = await prisma.user.create({
        data: { email },
      });
      console.log(`Created user: ${user.id} (${user.email})`);
    } else {
      console.log(`Found user: ${user.name ?? user.email ?? user.id}`);
    }
    resolvedUserId = user.id;
  }

  // Generate key
  const { fullKey, prefix, keyBody } = generateApiKey();
  const hashedKey = await argon2.hash(keyBody, argon2Options);

  const days = expiresInDays ? parseInt(expiresInDays, 10) : 30;
  const expiresAt = new Date(Date.now() + days * 24 * 60 * 60 * 1000);

  // Store in database
  const apiKey = await prisma.apiKey.create({
    data: {
      userId: resolvedUserId,
      name,
      keyPrefix: prefix,
      key: hashedKey,
      expiresAt,
    },
  });

  console.log("\n--- API Key Created ---");
  console.log(`  ID:         ${apiKey.id}`);
  console.log(`  Name:       ${apiKey.name}`);
  console.log(`  Expires:    ${apiKey.expiresAt.toISOString()}`);
  console.log(`  Full Key:   ${fullKey}`);
  console.log("\nSave this key now — it cannot be retrieved again.");
}

main()
  .catch((err) => {
    console.error("Fatal error:", err);
    process.exit(1);
  })
  .finally(() => prisma.$disconnect());
