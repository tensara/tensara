import { z } from "zod";
import { createTRPCRouter, protectedProcedure, publicProcedure } from "../trpc";
import { TRPCError } from "@trpc/server";

export const postupvoteRouter = createTRPCRouter({
  toggle: protectedProcedure
    .input(z.object({ postId: z.string() }))
    .mutation(async ({ ctx, input }) => {
      const db = ctx.db as any;

      // ensure post exists
      const post = await db.blogPost.findUnique({
        where: { id: input.postId },
        select: { id: true },
      });
      if (!post)
        throw new TRPCError({ code: "NOT_FOUND", message: "Post not found" });

      const userId = ctx.session.user.id;

      // compound unique: postId + userId
      const existing = await db.postUpvote.findUnique({
        where: { postId_userId: { postId: input.postId, userId } },
      });

      if (existing) {
        await db.postUpvote.delete({ where: { id: existing.id } });
        return { added: false };
      }

      await db.postUpvote.create({ data: { postId: input.postId, userId } });
      return { added: true };
    }),

  count: publicProcedure
    .input(z.object({ postId: z.string() }))
    .query(async ({ ctx, input }) => {
      const db = ctx.db as any;
      const count = await db.postUpvote.count({
        where: { postId: input.postId },
      });
      return { count };
    }),
  // Check if current user has upvoted this post
  hasUpvoted: protectedProcedure
    .input(z.object({ postId: z.string() }))
    .query(async ({ ctx, input }) => {
      const db = ctx.db as any;
      const userId = ctx.session.user.id;
      const existing = await db.postUpvote.findUnique({
        where: { postId_userId: { postId: input.postId, userId } },
        select: { id: true },
      });
      return { hasUpvoted: !!existing };
    }),
});
