import { z } from "zod";
import { createTRPCRouter, protectedProcedure, publicProcedure } from "../trpc";
import { TRPCError } from "@trpc/server";

// Simulated evaluation delay
const EVAL_DELAY_MS = 1500;

export const problemsRouter = createTRPCRouter({
  // Get all problems (public)
  getAll: publicProcedure.query(async ({ ctx }) => {
    try {
      const problems = await ctx.db.problem.findMany({
        select: {
          id: true,
          slug: true,
          title: true,
          difficulty: true,
          author: true,
        },
      });

      console.log("Found problems:", problems);
      return problems;
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
      return ctx.db.problem.findUnique({
        where: { slug: input.slug },
        include: {
          testCases: {
            where: { isHidden: false },
            select: {
              id: true,
              input: true,
              expected: true,
            },
          },
        },
      });
    }),

  // Submit solution
  submit: protectedProcedure
    .input(
      z.object({
        problemSlug: z.string(),
        code: z.string(),
        language: z.enum(["python", "javascript", "typescript"]),
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
        },
      });

      // 2. Simulate code evaluation
      await new Promise((resolve) => setTimeout(resolve, EVAL_DELAY_MS));

      // 3. Generate dummy metrics
      const metrics = {
        status: "SUCCESS",
        runtime: Math.floor(Math.random() * 500) + 100, // 100-600ms
        memory: Math.floor(Math.random() * 50) + 20, // 20-70MB
        passedTests: Math.floor(Math.random() * 3) + 8, // 8-10 tests
        totalTests: 10,
      };

      // 4. Update submission with results
      const updatedSubmission = await ctx.db.submission.update({
        where: { id: submission.id },
        data: {
          status: metrics.status,
          runtime: metrics.runtime,
          memory: metrics.memory,
          passedTests: metrics.passedTests,
          totalTests: metrics.totalTests,
        },
      });

      return updatedSubmission;
    }),

  // Get submission history for a problem
  getSubmissions: protectedProcedure
    .input(
      z.object({
        problemSlug: z.string(),
        limit: z.number().min(1).max(100).default(10),
        cursor: z.string().nullish(),
      })
    )
    .query(async ({ ctx, input }) => {
      const submissions = await ctx.db.submission.findMany({
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
      });

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

  // Add this to your existing problemsRouter
  benchmarkGPU: protectedProcedure
    .input(
      z.object({
        code: z.string(),
        language: z.enum(["python", "javascript", "typescript"]),
        problemSlug: z.string(),
        gpuType: z.enum(["T4", "A100", "V100"]).default("T4"),
      })
    )
    .mutation(async ({ ctx, input }) => {
      // 1. Find the problem
      const problem = await ctx.db.problem.findUniqueOrThrow({
        where: { slug: input.problemSlug },
      });

      // 2. Create benchmark record
      const benchmark = await ctx.db.benchmarkResult.create({
        data: {
          gpuType: input.gpuType,
          timeMs: Math.random() * 1000, // Dummy: 0-1000ms
          memoryMB: Math.random() * 1024, // Dummy: 0-1GB
          logs: "GPU Kernel execution completed successfully",
          verified: true,
          submission: {
            create: {
              code: input.code,
              language: input.language,
              userId: ctx.session.user.id,
              problemId: problem.id,
              status: "COMPLETED",
            },
          },
        },
        include: {
          submission: true,
        },
      });

      return {
        metrics: {
          executionTime: benchmark.timeMs,
          memoryUsage: benchmark.memoryMB,
          gpuType: benchmark.gpuType,
        },
        submission: benchmark.submission,
        logs: benchmark.logs,
      };
    }),
});
