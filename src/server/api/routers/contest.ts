import { z } from "zod";
import { createTRPCRouter, publicProcedure } from "~/server/api/trpc";

export const contestRouter = createTRPCRouter({
  getAll: publicProcedure
    .input(
      z.object({
        status: z.enum(["UPCOMING", "ACTIVE", "COMPLETED", "ARCHIVED", "ALL"]),
      })
    )
    .query(async ({ ctx, input }) => {
      if (input.status === "ALL") {
        const contests = await ctx.db.contest.findMany({
          select: {
            id: true,
            slug: true,
            title: true,
            description: true,
            status: true,
            startTime: true,
            endTime: true,
            participantCount: true,
            winners: true,
          },
        });
        return contests;
      }
      const contests = await ctx.db.contest.findMany({
        select: {
          id: true,
          slug: true,
          title: true,
          description: true,
          status: true,
          winners: true,
          startTime: true,
          endTime: true,
          participantCount: true,
        },
        where: {
          status: input.status,
        },
      });
      return contests;
    }),
  getBySlug: publicProcedure.input(z.string()).query(async ({ ctx, input }) => {
    const contest = await ctx.db.contest.findUnique({
      where: { slug: input },
    });
    return contest;
  }),

  getIfUserIsParticipant: publicProcedure
    .input(
      z.object({
        contestId: z.string(),
        userId: z.string(),
      })
    )
    .query(async ({ ctx, input }) => {
      const isParticipant = await ctx.db.contestParticipant.findFirst({
        where: {
          contestId: input.contestId,
          userId: input.userId,
        },
      });
      if (isParticipant) {
        return true;
      }
      return false;
    }),

  getContestProblems: publicProcedure
    .input(
      z.object({
        contestId: z.string(),
      })
    )
    .query(async ({ ctx, input }) => {
      const problems = await ctx.db.problem.findMany({
        where: {
          contestId: input.contestId,
        },
      });
      return problems;
    }),
});
