import { z } from "zod";
import { createTRPCRouter, protectedProcedure, publicProcedure } from "../trpc";
import { TRPCError } from "@trpc/server";
// eslint-disable-next-line @typescript-eslint/no-unused-vars
import { env } from "~/env";

// // Simulated evaluation delay
// const EVAL_DELAY_MS = 1500;

// Add these types at the top of the file
type BenchmarkTestResult = {
  test_id: number;
  runtime_ms: number;
  gflops: number;
};

const SubmissionStatus = {
  CHECKING: "CHECKING",
  BENCHMARKING: "BENCHMARKING",
  ACCEPTED: "ACCEPTED",
  WRONG_ANSWER: "WRONG_ANSWER",
  ERROR: "ERROR",
} as const;

type SubmissionStatus =
  (typeof SubmissionStatus)[keyof typeof SubmissionStatus];

// type SubmissionResponse = SubmissionErrorResponse | SubmissionSuccessResponse;

// Add types for database submission
type SubmissionWithCustomFields = {
  id: string;
  status: string | null;
  passedTests: number | null;
  totalTests: number | null;
  runtime: number | null;
  gflops: number | null;
  benchmarkResults: BenchmarkTestResult[] | null;
  errorMessage?: string;
  errorDetails?: string;
};



export const problemsRouter = createTRPCRouter({
  getAll: publicProcedure.query(async ({ ctx }) => {
    try {
      const problems = await ctx.db.problem.findMany({
        select: {
          id: true,
          slug: true,
          title: true,
          difficulty: true,
          author: true,
          _count: {
            select: {
              submissions: true,
            },
          },
        },
      });

      console.log("Found problems:", problems);
      return problems.map((problem) => ({
        ...problem,
        submissionCount: problem._count.submissions,
      }));
    } catch (error) {
      console.error("Error fetching problems:", error);
      throw new TRPCError({
        code: "INTERNAL_SERVER_ERROR",
        message: "Failed to fetch problems",
        cause: error,
      });
    }
  }),

  // Get single problem with test cases
  getById: publicProcedure
    .input(z.object({ slug: z.string() }))
    .query(async ({ ctx, input }) => {
      const problem = await ctx.db.problem.findUniqueOrThrow({
        where: { slug: input.slug },
      });

      if (!problem) return null;

      return {
        ...problem,
      };
    }),

  createSubmission: protectedProcedure
    .input(
      z.object({
        problemSlug: z.string(),
        code: z.string(),
        language: z.enum(["cpp", "cuda", "python"]),
        gpuType: z.string(),
      })
    )
    .mutation(async ({ ctx, input }) => {
      const problem = await ctx.db.problem.findUniqueOrThrow({
        where: { slug: input.problemSlug },
      });

      const submission = await ctx.db.submission.create({
        data: {
          code: input.code,
          language: input.language,
          userId: ctx.session.user.id,
          problemId: problem.id,
          status: SubmissionStatus.CHECKING,
          gpuType: input.gpuType,
        },
      });

      return submission;
    }),

  submit: protectedProcedure
    .input(
      z.object({
        submissionId: z.string(),
      })
    )
    .mutation(async ({ ctx, input }) => {
      const submission = await ctx.db.submission.findUniqueOrThrow({
        where: { id: input.submissionId },
        include: {
          problem: {
            select: {
              slug: true,
              title: true,
            },
          },
        },
      });

      // Just return the submission ID - the client will connect to the streaming endpoint
      return {
        submissionId: submission.id,
        problemSlug: submission.problem.slug,
      };
    }),

  // Get submission history for a problem
  getSubmissions: protectedProcedure
    .input(
      z.object({
        problemSlug: z.string(),
        limit: z.number().min(1).max(100).default(50),
        cursor: z.string().nullish(),
      })
    )
    .query(async ({ ctx, input }) => {
      const submissions = (await ctx.db.submission.findMany({
        where: {
          userId: ctx.session.user.id,
          problem: { slug: input.problemSlug },
        },
        take: input.limit + 1,
        cursor: input.cursor ? { id: input.cursor } : undefined,
        orderBy: { createdAt: "desc" },
        include: {
          problem: {
            select: {
              title: true,
              slug: true,
            },
          },
        },
      })) as unknown as (SubmissionWithCustomFields & {
        problem: {
          title: string;
          slug: string;
        };
      })[];

      let nextCursor: typeof input.cursor = undefined;
      if (submissions.length > input.limit) {
        const nextItem = submissions.pop();
        nextCursor = nextItem!.id;
      }

      return {
        submissions,
        nextCursor,
      };
    }),

  // Get user's overall submission stats
  getUserStats: protectedProcedure.query(async ({ ctx }) => {
    const stats = await ctx.db.submission.groupBy({
      by: ["status"],
      where: { userId: ctx.session.user.id },
      _count: true,
    });

    return stats;
  }),

  // Get submission status
  getSubmissionStatus: publicProcedure
    .input(z.object({ submissionId: z.string() }))
    .query(async ({ ctx, input }) => {
      const submission = await ctx.db.submission.findUnique({
        where: { id: input.submissionId },
        include: {
          problem: {
            select: {
              title: true,
              slug: true,
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

      if (ctx.session?.user?.id != submission.userId && !submission.isPublic) {
        // eslint-disable-next-line @typescript-eslint/no-unused-vars
        const { code: _, ...submissionWithoutCode } = submission;
        return submissionWithoutCode;
      }

      return submission;
    }),

  // Toggle a submission's public status
  toggleSubmissionPublic: protectedProcedure
    .input(z.object({ submissionId: z.string(), isPublic: z.boolean() }))
    .mutation(async ({ ctx, input }) => {
      const submission = await ctx.db.submission.findUnique({
        where: { id: input.submissionId },
        select: { userId: true },
      });

      if (!submission) {
        throw new TRPCError({
          code: "NOT_FOUND",
          message: "Submission not found",
        });
      }

      if (submission.userId !== ctx.session.user.id) {
        throw new TRPCError({
          code: "FORBIDDEN",
          message: "You can only modify your own submissions",
        });
      }

      const updatedSubmission = await ctx.db.submission.update({
        where: { id: input.submissionId },
        data: { isPublic: input.isPublic },
        include: {
          problem: {
            select: {
              title: true,
              slug: true,
            },
          },
        },
      });

      return updatedSubmission;
    }),
});
