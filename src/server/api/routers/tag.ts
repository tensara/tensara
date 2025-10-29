import { z } from "zod";
import { createTRPCRouter, publicProcedure, protectedProcedure } from "../trpc";
import { TRPCError } from "@trpc/server";

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
      const db = ctx.db as any;

      // Build where clause
      const where: any = {};

      if (input.category) {
        where.category = input.category;
      }

      if (input.search) {
        where.name = {
          contains: input.search,
          mode: "insensitive",
        };
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
      const db = ctx.db as any;

      // Build where clause
      const where: any = {};

      if (input.category) {
        where.category = input.category;
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
      const db = ctx.db as any;

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
      const db = ctx.db as any;

      try {
        const tag = await db.tag.create({
          data: {
            name: input.name,
            slug: input.slug,
            description: input.description,
            category: input.category,
          },
        });

        return tag;
      } catch (error: any) {
        // Handle unique constraint violation
        if (error.code === "P2002") {
          const field = error.meta?.target?.[0] || "field";
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
      const db = ctx.db as any;

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
      const data: any = {};

      if (input.name !== undefined) {
        data.name = input.name;
      }

      if (input.description !== undefined) {
        data.description = input.description;
      }

      if (input.category !== undefined) {
        data.category = input.category;
      }

      try {
        const tag = await db.tag.update({
          where: { id: input.id },
          data,
        });

        return tag;
      } catch (error: any) {
        // Handle unique constraint violation
        if (error.code === "P2002") {
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
      const db = ctx.db as any;

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
