import { z } from "zod";
import { createTRPCRouter, protectedProcedure, publicProcedure } from "../trpc";
import { TRPCError } from "@trpc/server";

export const commentupvoteRouter = createTRPCRouter({
  toggle: protectedProcedure
    .input(z.object({ commentId: z.string() }))
    .mutation(async ({ ctx, input }) => {
      const db = ctx.db as any;

      const comment = await db.comment.findUnique({
        where: { id: input.commentId },
        select: { id: true },
      });
      if (!comment)
        throw new TRPCError({
          code: "NOT_FOUND",
          message: "Comment not found",
        });

      const userId = ctx.session.user.id;
      const existing = await db.commentUpvote.findUnique({
        where: { commentId_userId: { commentId: input.commentId, userId } },
      });

      if (existing) {
        await db.commentUpvote.delete({ where: { id: existing.id } });
        return { added: false };
      }

      await db.commentUpvote.create({
        data: { commentId: input.commentId, userId },
      });
      return { added: true };
    }),

  count: publicProcedure
    .input(z.object({ commentId: z.string() }))
    .query(async ({ ctx, input }) => {
      const db = ctx.db as any;
      const count = await db.commentUpvote.count({
        where: { commentId: input.commentId },
      });
      return { count };
    }),
  // whether current user has upvoted this comment
  hasUpvoted: protectedProcedure
    .input(z.object({ commentId: z.string() }))
    .query(async ({ ctx, input }) => {
      const db = ctx.db as any;
      const userId = ctx.session.user.id;
      const existing = await db.commentUpvote.findUnique({
        where: { commentId_userId: { commentId: input.commentId, userId } },
        select: { id: true },
      });
      return { hasUpvoted: !!existing };
    }),
});
