import { z } from "zod";
import { createTRPCRouter, protectedProcedure } from "~/server/api/trpc";

export const submissionsRouter = createTRPCRouter({
  getAllSubmissions: protectedProcedure.query(async ({ ctx }) => {
    const submissions = await ctx.db.submission.findMany({
      include: {
        problem: {
          select: {
            title: true,
            slug: true,
          },
        },
        user: {
          select: {
            name: true,
          },
        },
      },
      orderBy: {
        createdAt: "desc",
      },
    });

    return submissions;
  }),
  getSubmissionsByProblemId: protectedProcedure
    .input(z.object({ problemId: z.string() }))
    .query(async ({ ctx, input }) => {
      const submissions = await ctx.db.submission.findMany({
        where: {
          problemId: input.problemId,
        },
        include: {
          problem: {
            select: {
              title: true,
              slug: true,
            },
          },
          user: {
            select: {
              name: true,
            },
          },
        },
        orderBy: {
          createdAt: "desc",
        },
      });

      return submissions;
    }),
});
