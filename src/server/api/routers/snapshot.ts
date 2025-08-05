import { z } from "zod";
import { createTRPCRouter, protectedProcedure, publicProcedure } from "../trpc";
import { TRPCError } from "@trpc/server";

export const snapshotRouter = createTRPCRouter({
  create: protectedProcedure
    .input(
      z.object({
        files: z.array(z.object({ name: z.string(), content: z.string() })),
        main: z.string(),
        workspaceId: z.string().optional(),
      })
    )
    .mutation(async ({ ctx, input }) => {
      const snapshot = await ctx.db.snapshot.create({
        data: {
          files: input.files,
          main: input.main,
          workspaceId: input.workspaceId ?? null,
          userId: ctx.session.user.id,
        },
        select: {
          id: true,
        },
      });

      return snapshot;
    }),

  getById: publicProcedure
    .input(z.object({ id: z.string() }))
    .query(async ({ ctx, input }) => {
      const snapshot = await ctx.db.snapshot.findUnique({
        where: { id: input.id },
      });

      if (!snapshot) {
        throw new TRPCError({
          code: "NOT_FOUND",
          message: "Snapshot not found",
        });
      }

      return snapshot;
    }),
});
