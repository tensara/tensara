import { z } from "zod";
import {
  createTRPCRouter,
  protectedProcedure,
  publicProcedure,
} from "~/server/api/trpc";
import { TRPCError } from "@trpc/server";

export const submissionsRouter = createTRPCRouter({
  // all submissions (public or not) for the current user
  getAllUserSubmissions: protectedProcedure.query(async ({ ctx }) => {
    const submissions = await ctx.db.submission.findMany({
      where: {
        userId: ctx.session.user.id
      },
      include: {
        user: {
          select: {
            username: true,
          },
        },
        problem: {
          select: {
            title: true,
            slug: true,
          },
        },
      },
      orderBy: {
        createdAt: "desc",
      },
    });

    return submissions;
  }),

  // all submissions (public or not)
  getLeaderboardSubmissions: publicProcedure.query(async ({ ctx }) => {
    const submissions = await ctx.db.submission.findMany({
      where: {
        status: "ACCEPTED",
      },
      include: {
        user: {
          select: {
            username: true,
          },
        },
        problem: {
          select: {
            title: true,
            slug: true,
          },
        },
      },
      orderBy: {
        createdAt: "desc",
      },
    });

    return submissions;
  }),

  getSubmissionById: publicProcedure
    .input(z.object({ id: z.string() }))
    .query(async ({ ctx, input }) => {
      const submission = await ctx.db.submission.findUnique({
        where: { id: input.id },
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
      });

      if (!submission) {
        throw new TRPCError({
          code: "NOT_FOUND",
          message: "Submission not found",
        });
      }

      // If user is not logged in or not the owner and submission is not public
      if (
        (!ctx.session?.user || ctx.session.user.id !== submission.userId) &&
        !submission.isPublic
      ) {
        throw new TRPCError({
          code: "FORBIDDEN",
          message: "You don't have permission to view this submission",
        });
      }

      return submission;
    }),
});
