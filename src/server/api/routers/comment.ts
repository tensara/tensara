import { z } from "zod";
import { createTRPCRouter, protectedProcedure, publicProcedure } from "../trpc";
import { TRPCError } from "@trpc/server";

export const commentRouter = createTRPCRouter({
  create: protectedProcedure
    .input(
      z.object({
        postId: z.string(),
        content: z.string().min(1),
        parentCommentId: z.string().optional(),
      })
    )
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

      // if replying to a comment, ensure parent exists and belongs to same post
      if (input.parentCommentId) {
        const parent = await db.comment.findUnique({
          where: { id: input.parentCommentId },
          select: { id: true, postId: true },
        });
        if (!parent || parent.postId !== input.postId) {
          throw new TRPCError({
            code: "BAD_REQUEST",
            message: "Invalid parent comment",
          });
        }
      }

      const comment = await db.comment.create({
        data: {
          postId: input.postId,
          content: input.content,
          authorId: ctx.session.user.id,
          parentCommentId: input.parentCommentId ?? null,
        },
      });

      return comment;
    }),

  getByPost: publicProcedure
    .input(z.object({ postId: z.string() }))
    .query(async ({ ctx, input }) => {
      const db = ctx.db as any;

      // Fetch all comments for the post with their author. We'll build a
      // nested tree on the server so nested replies of arbitrary depth are
      // returned to the client under `children` arrays.
      const flatComments = await db.comment.findMany({
        where: { postId: input.postId },
        orderBy: { createdAt: "asc" },
        include: { author: { select: { id: true, name: true, image: true } } },
      });

      // Build map id -> comment (and initialize children arrays)
      const map: Record<string, any> = {};
      for (const c of flatComments) {
        map[c.id] = { ...c, children: [] };
      }

      // Attach children to their parents; collect roots
      const roots: any[] = [];
      for (const id in map) {
        const node = map[id];
        if (node.parentCommentId && map[node.parentCommentId]) {
          map[node.parentCommentId].children.push(node);
        } else {
          roots.push(node);
        }
      }

      return roots;
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
