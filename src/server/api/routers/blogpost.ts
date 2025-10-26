import { z } from "zod";
import { createTRPCRouter, protectedProcedure, publicProcedure } from "../trpc";
import { TRPCError } from "@trpc/server";

export const blogpostRouter = createTRPCRouter({
  getAll: publicProcedure.query(async ({ ctx }) => {
    const db = ctx.db as any;
    const posts: any[] = await db.blogPost.findMany({
      orderBy: { createdAt: "desc" },
      include: {
        author: { select: { id: true, name: true } },
        _count: { select: { comments: true, upvotes: true } },
      },
    });

    return posts.map((p) => ({
      id: p.id,
      title: p.title,
      // we intentionally keep content (could be large) but callers can choose to display summary
      content: p.content,
      author: p.author,
      createdAt: p.createdAt,
      updatedAt: p.updatedAt,
      commentCount: p._count?.comments ?? 0,
      upvoteCount: p._count?.upvotes ?? 0,
    }));
  }),

  getById: publicProcedure
    .input(z.object({ id: z.string() }))
    .query(async ({ ctx, input }) => {
      const db = ctx.db as any;
      const post = await db.blogPost.findUnique({
        where: { id: input.id },
        include: {
          author: { select: { id: true, name: true } },
          comments: {
            include: { author: { select: { id: true, name: true } } },
            orderBy: { createdAt: "asc" },
          },
          _count: { select: { comments: true, upvotes: true } },
        },
      });

      if (!post) {
        throw new TRPCError({
          code: "NOT_FOUND",
          message: "Blog post not found",
        });
      }

      return post;
    }),

  create: protectedProcedure
    .input(z.object({ title: z.string().min(1), content: z.string().min(1) }))
    .mutation(async ({ ctx, input }) => {
      const db = ctx.db as any;
      const post = await db.blogPost.create({
        data: {
          title: input.title,
          content: input.content,
          authorId: ctx.session.user.id,
        },
      });

      return post;
    }),

  update: protectedProcedure
    .input(
      z.object({
        id: z.string(),
        title: z.string().min(1).optional(),
        content: z.string().min(1).optional(),
      })
    )
    .mutation(async ({ ctx, input }) => {
      const db = ctx.db as any;

      const existing = await db.blogPost.findUnique({
        where: { id: input.id },
        select: { authorId: true },
      });
      if (!existing)
        throw new TRPCError({ code: "NOT_FOUND", message: "Post not found" });
      if (existing.authorId !== ctx.session.user.id)
        throw new TRPCError({
          code: "FORBIDDEN",
          message: "You can only edit your own posts",
        });

      const data: any = {};
      if (typeof input.title === "string") data.title = input.title;
      if (typeof input.content === "string") data.content = input.content;

      const updated = await db.blogPost.update({
        where: { id: input.id },
        data,
      });
      return updated;
    }),

  delete: protectedProcedure
    .input(z.object({ id: z.string() }))
    .mutation(async ({ ctx, input }) => {
      const db = ctx.db as any;
      const existing = await db.blogPost.findUnique({
        where: { id: input.id },
        select: { authorId: true },
      });

      if (!existing) {
        throw new TRPCError({
          code: "NOT_FOUND",
          message: "Blog post not found",
        });
      }

      if (existing.authorId !== ctx.session.user.id) {
        throw new TRPCError({
          code: "FORBIDDEN",
          message: "You can only delete your own posts",
        });
      }

      await db.blogPost.delete({ where: { id: input.id } });

      return { ok: true };
    }),

  getUserPosts: protectedProcedure.query(async ({ ctx }) => {
    const db = ctx.db as any;
    const posts = await db.blogPost.findMany({
      where: { authorId: ctx.session.user.id },
      orderBy: { createdAt: "desc" },
    });

    return posts as any[];
  }),
});
