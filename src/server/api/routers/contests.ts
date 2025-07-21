import { z } from "zod";
import {
  createTRPCRouter,
  publicProcedure,
  protectedProcedure,
  adminProcedure,
} from "~/server/api/trpc";
import { ContestStatus, ProblemVisibility } from "@prisma/client";

export const contestsRouter = createTRPCRouter({
  create: adminProcedure
    .input(
      z.object({
        title: z.string(),
        description: z.string().optional(),
        startTime: z.date(),
        endTime: z.date(),
        private: z.boolean().optional(),
      })
    )
    .mutation(async ({ ctx, input }) => {
      return ctx.db.contest.create({
        data: {
          ...input,
          creatorId: ctx.session.user.id,
        },
      });
    }),

  getAll: publicProcedure.query(async ({ ctx }) => {
    const contests = await ctx.db.contest.findMany({
      include: {
        problems: {
          include: {
            problem: true,
          },
        },
      },
      orderBy: {
        startTime: "desc",
      },
    });

    if (ctx.session?.user.role === "ADMIN") {
      return contests;
    }

    return contests.map((contest) => ({
      ...contest,
      problems: contest.problems.filter(
        (p) => p.problem.visibility === ProblemVisibility.PUBLIC
      ),
    }));
  }),

  getAllAdmin: adminProcedure.query(async ({ ctx }) => {
    return ctx.db.contest.findMany({
      orderBy: {
        startTime: "desc",
      },
    });
  }),

  getById: publicProcedure
    .input(z.object({ id: z.string() }))
    .query(async ({ ctx, input }) => {
      const contest = await ctx.db.contest.findUnique({
        where: { id: input.id },
        include: {
          problems: {
            include: {
              problem: true,
            },
          },
          participants: true,
        },
      });

      if (!contest) {
        return null;
      }

      const isPrivate = contest.problems.some(
        (p) => p.problem.visibility === ProblemVisibility.PRIVATE
      );

      if (isPrivate && !ctx.session?.user) {
        return null;
      }

      return contest;
    }),

  register: protectedProcedure
    .input(z.object({ contestId: z.string() }))
    .mutation(async ({ ctx, input }) => {
      return ctx.db.contestUser.create({
        data: {
          contestId: input.contestId,
          userId: ctx.session.user.id,
        },
      });
    }),

  updateStatus: adminProcedure
    .input(
      z.object({
        id: z.string(),
        status: z.nativeEnum(ContestStatus),
      })
    )
    .mutation(async ({ ctx, input }) => {
      return ctx.db.contest.update({
        where: { id: input.id },
        data: {
          status: input.status,
        },
      });
    }),

  addProblemToContest: adminProcedure
    .input(
      z.object({
        contestId: z.string(),
        problemId: z.string(),
      })
    )
    .mutation(async ({ ctx, input }) => {
      return ctx.db.contestProblem.create({
        data: {
          contestId: input.contestId,
          problemId: input.problemId,
        },
      });
    }),

  removeProblemFromContest: adminProcedure
    .input(
      z.object({
        contestId: z.string(),
        problemId: z.string(),
      })
    )
    .mutation(async ({ ctx, input }) => {
      return ctx.db.contestProblem.delete({
        where: {
          contestId_problemId: {
            contestId: input.contestId,
            problemId: input.problemId,
          },
        },
      });
    }),
});
