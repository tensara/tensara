import { z } from "zod";
import {
  createTRPCRouter,
  protectedProcedure,
  publicProcedure,
} from "~/server/api/trpc";

export const submissionsRouter = createTRPCRouter({
  getAllSubmissions: publicProcedure.query(async ({ ctx }) => {
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
            username: true,
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
