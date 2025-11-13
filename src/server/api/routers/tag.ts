import { z } from "zod";
import { createTRPCRouter, publicProcedure, protectedProcedure } from "../trpc";
import { TRPCError } from "@trpc/server";
import type { PrismaClient, Prisma, TagCategory } from "@prisma/client";

// Define enum for validation
const TagCategoryEnum = z.enum([
  "GENERAL",
  "PROBLEM",
  "LANGUAGE",
  "OPTIMIZATION",
  "DIFFICULTY",
  "TOPIC",
]);

export const tagRouter = createTRPCRouter({
  getAll: publicProcedure
    .input(
      z.object({
        category: TagCategoryEnum.optional(),
        search: z.string().optional(),
      })
    )
    .query(async ({ ctx, input }) => {
      const db = ctx.db as PrismaClient;

      // Build where clause
      const where: Prisma.TagWhereInput = {};

      if (input.category) {
        where.category = input.category as TagCategory;
      }

      if (input.search) {
        where.name = {
          contains: input.search,
          mode: "insensitive",
        } as unknown as Prisma.StringFilter;
      }

      const tags = await db.tag.findMany({
        where,
        orderBy: { name: "asc" },
        include: {
          _count: {
            select: { posts: true },
          },
        },
      });

      return tags;
    }),

  getPopular: publicProcedure
    .input(
      z.object({
        limit: z.number().min(1).max(50).default(20),
        category: TagCategoryEnum.optional(),
      })
    )
    .query(async ({ ctx, input }) => {
      const db = ctx.db as PrismaClient;

      // Build where clause
      const where: Prisma.TagWhereInput = {};

      if (input.category) {
        where.category = input.category as TagCategory;
      }

      const tags = await db.tag.findMany({
        where,
        take: input.limit,
        orderBy: {
          posts: {
            _count: "desc",
          },
        },
        include: {
          _count: {
            select: { posts: true },
          },
        },
      });

      return tags;
    }),

  getBySlug: publicProcedure
    .input(
      z.object({
        slug: z.string(),
      })
    )
    .query(async ({ ctx, input }) => {
      const db = ctx.db as PrismaClient;

      const tag = await db.tag.findUnique({
        where: { slug: input.slug },
        include: {
          _count: {
            select: { posts: true },
          },
        },
      });

      if (!tag) {
        throw new TRPCError({
          code: "NOT_FOUND",
          message: "Tag not found",
        });
      }

      return tag;
    }),

  create: protectedProcedure
    .input(
      z.object({
        name: z.string().min(1).max(50),
        slug: z.string().min(1).max(50),
        description: z.string().max(500).optional(),
        category: TagCategoryEnum.default("GENERAL"),
      })
    )
    .mutation(async ({ ctx, input }) => {
      const db = ctx.db as PrismaClient;

      try {
        const tag = await db.tag.create({
          data: {
            name: input.name,
            slug: input.slug,
            description: input.description,
            category: input.category as TagCategory,
          },
        });

        return tag;
      } catch (error) {
        const err = error as
          | {
              code?: string;
              meta?: { target?: string[] } | undefined;
            }
          | undefined;
        // Handle unique constraint violation
        if (err?.code === "P2002") {
          const field = err.meta?.target?.[0] ?? "field";
          throw new TRPCError({
            code: "CONFLICT",
            message: `A tag with this ${field} already exists`,
          });
        }
        throw error;
      }
    }),

  update: protectedProcedure
    .input(
      z.object({
        id: z.string(),
        name: z.string().min(1).max(50).optional(),
        description: z.string().max(500).optional(),
        category: TagCategoryEnum.optional(),
      })
    )
    .mutation(async ({ ctx, input }) => {
      const db = ctx.db as PrismaClient;

      // Check if tag exists
      const existing = await db.tag.findUnique({
        where: { id: input.id },
        select: { id: true },
      });

      if (!existing) {
        throw new TRPCError({
          code: "NOT_FOUND",
          message: "Tag not found",
        });
      }

      // Build update data (excluding slug to preserve URLs)
      const data: Prisma.TagUpdateInput = {};

      if (input.name !== undefined) {
        data.name = input.name;
      }

      if (input.description !== undefined) {
        data.description = input.description;
      }

      if (input.category !== undefined) {
        data.category = input.category as TagCategory;
      }

      try {
        const tag = await db.tag.update({
          where: { id: input.id },
          data,
        });

        return tag;
      } catch (error) {
        const err = error as { code?: string } | undefined;
        // Handle unique constraint violation
        if (err?.code === "P2002") {
          throw new TRPCError({
            code: "CONFLICT",
            message: "A tag with this name already exists",
          });
        }
        throw error;
      }
    }),

  delete: protectedProcedure
    .input(
      z.object({
        id: z.string(),
      })
    )
    .mutation(async ({ ctx, input }) => {
      const db = ctx.db as PrismaClient;

      // Check if tag exists
      const existing = await db.tag.findUnique({
        where: { id: input.id },
        select: { id: true },
      });

      if (!existing) {
        throw new TRPCError({
          code: "NOT_FOUND",
          message: "Tag not found",
        });
      }

      // Delete the tag (will cascade delete BlogPostTag relations)
      await db.tag.delete({
        where: { id: input.id },
      });

      return { ok: true };
    }),
});
