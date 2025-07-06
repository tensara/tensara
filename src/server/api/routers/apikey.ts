import { z } from "zod";
import { createTRPCRouter, protectedProcedure } from "../trpc";
import { TRPCError } from "@trpc/server";
import crypto from "crypto";
import argon2 from "argon2";

const argon2Options = {
  type: argon2.argon2id,
  memoryCost: 4096,
  timeCost: 2,
  parallelism: 1,
};

const generateApiKey = () => {
  const prefix = crypto.randomBytes(6).toString("hex");
  const keyBody = crypto.randomBytes(28).toString("hex");
  return {
    fullKey: `tsra_${prefix}_${keyBody}`,
    prefix,
    keyBody,
  };
};

export const apiKeysRouter = createTRPCRouter({
  getAll: protectedProcedure.query(async ({ ctx }) => {
    try {
      const apiKeys = await ctx.db.apiKey.findMany({
        where: { userId: ctx.session.user.id },
        select: {
          id: true,
          name: true,
          createdAt: true,
          expiresAt: true,
        },
        orderBy: { createdAt: "desc" },
      });

      return apiKeys;
    } catch (error) {
      console.error("Error fetching API keys:", error);
      throw new TRPCError({
        code: "INTERNAL_SERVER_ERROR",
        message: "Failed to fetch API keys",
        cause: error,
      });
    }
  }),

  create: protectedProcedure
    .input(
      z.object({
        name: z.string().min(3, "Name must be at least 3 characters long"),
        expiresIn: z.number().optional(),
      })
    )
    .mutation(async ({ ctx, input }) => {
      try {
        const { fullKey, prefix, keyBody } = generateApiKey();
        const hashedKey = await argon2.hash(keyBody, argon2Options);

        const expiresAt = input.expiresIn
          ? new Date(Date.now() + input.expiresIn)
          : new Date(Date.now() + 30 * 24 * 60 * 60 * 1000);

        const apiKey = await ctx.db.apiKey.create({
          data: {
            userId: ctx.session.user.id,
            name: input.name,
            keyPrefix: prefix,
            key: hashedKey,
            expiresAt,
          },
        });

        return {
          id: apiKey.id,
          name: apiKey.name,
          key: fullKey,
          createdAt: apiKey.createdAt,
          expiresAt: apiKey.expiresAt,
        };
      } catch (error) {
        if (
          error instanceof Error &&
          error.message.includes("Unique constraint failed")
        ) {
          throw new TRPCError({
            code: "CONFLICT",
            message: "An API key with this name already exists",
          });
        }

        console.error("Error creating API key:", error);
        throw new TRPCError({
          code: "INTERNAL_SERVER_ERROR",
          message: "Failed to create API key",
          cause: error,
        });
      }
    }),

  delete: protectedProcedure
    .input(z.object({ id: z.string() }))
    .mutation(async ({ ctx, input }) => {
      try {
        const apiKey = await ctx.db.apiKey.findUnique({
          where: { id: input.id },
          select: { userId: true },
        });

        if (!apiKey) {
          throw new TRPCError({
            code: "NOT_FOUND",
            message: "API key not found",
          });
        }

        if (apiKey.userId !== ctx.session.user.id) {
          throw new TRPCError({
            code: "FORBIDDEN",
            message: "You can only delete your own API keys",
          });
        }

        await ctx.db.apiKey.delete({
          where: { id: input.id },
        });

        return { success: true };
      } catch (error) {
        if (error instanceof TRPCError) {
          throw error;
        }

        console.error("Error deleting API key:", error);
        throw new TRPCError({
          code: "INTERNAL_SERVER_ERROR",
          message: "Failed to delete API key",
          cause: error,
        });
      }
    }),
});
