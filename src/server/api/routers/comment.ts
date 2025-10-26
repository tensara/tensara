import { z } from "zod";
import { createTRPCRouter, protectedProcedure, publicProcedure } from "../trpc";
import { TRPCError } from "@trpc/server";

export const commentRouter = createTRPCRouter({
  create: protectedProcedure
    .input(z.object({ postId: z.string(), content: z.string().min(1) }))
    .mutation(async ({ ctx, input }) => {
      const db = ctx.db as any;

      // verify post exists
      const post = await db.blogPost.findUnique({
        where: { id: input.postId },
        select: { id: true },
      });
      if (!post) {
        throw new TRPCError({ code: "NOT_FOUND", message: "Post not found" });
      }

      const comment = await db.comment.create({
        data: {
          postId: input.postId,
          content: input.content,
          authorId: ctx.session.user.id,
        },
      });

      return comment;
    }),

  getByPost: publicProcedure
    .input(z.object({ postId: z.string() }))
    .query(async ({ ctx, input }) => {
      const db = ctx.db as any;

      const comments = await db.comment.findMany({
        where: { postId: input.postId },
        orderBy: { createdAt: "asc" },
        include: { author: { select: { id: true, name: true } } },
      });

      return comments;
    }),

  delete: protectedProcedure
    .input(z.object({ id: z.string() }))
    .mutation(async ({ ctx, input }) => {
      const db = ctx.db as any;

      const existing = await db.comment.findUnique({
        where: { id: input.id },
        select: { authorId: true },
      });
      if (!existing) {
        throw new TRPCError({
          code: "NOT_FOUND",
          message: "Comment not found",
        });
      }

      if (existing.authorId !== ctx.session.user.id) {
        throw new TRPCError({
          code: "FORBIDDEN",
          message: "You can only delete your own comments",
        });
      }

      await db.comment.delete({ where: { id: input.id } });
      return { ok: true };
    }),
});
