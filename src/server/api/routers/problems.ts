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
      const problem = await ctx.db.problem.findUniqueOrThrow({
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

      if (!problem) return null;

      return {
        ...problem,
        testCases: problem.testCases.map((tc) => ({
          ...tc,
          input: JSON.stringify(tc.input),
          expected: JSON.stringify(tc.expected),
        })),
      };
    }),

  // Submit solution
  submit: protectedProcedure
    .input(
      z.object({
        problemSlug: z.string(),
        code: z.string(),
        language: z.enum(["cpp", "cuda", "python", "javascript", "typescript"]),
      })
    )
    .mutation(async ({ ctx, input }) => {
      const problem = await (ctx.db.problem.findUniqueOrThrow({
        where: { slug: input.problemSlug },
        include: {
          testCases: true
        }
      }) as Promise<any>);

      // Create initial submission record
      const submission = await ctx.db.submission.create({
        data: {
          code: input.code,
          language: input.language,
          userId: ctx.session.user.id,
          problemId: problem.id,
          status: "PENDING",
        },
      });

      try {
        // First call Modal checker endpoint
        console.log("problem bindings", problem.bindings);
        console.log("problem reference", problem.reference);
        console.log("input code", input.code);

        // Update submission to indicate checker is running
        await ctx.db.submission.update({
          where: { id: submission.id },
          data: {
            status: "CHECKING",
          },
        });

        const checkerResponse = await fetch("https://labs-asterisk--tensara-checker.modal.run", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            solution_cu: input.code,
            cuda_bindings: problem.bindings,
            reference_py: problem.reference,
          }),
        });

        if (!checkerResponse.ok) {
          throw new Error(`Checker API returned ${checkerResponse.status}`);
        }

        const checkerResult = await checkerResponse.json();
        console.log("CHECKER RESULT");
        console.log(checkerResult);
        
        if (!checkerResult.passed) {
          const wrongAnswerSubmission = await ctx.db.submission.update({
            where: { id: submission.id },
            data: {
              status: "WRONG_ANSWER",
              passedTests: checkerResult.passed_tests || 0,
              totalTests: checkerResult.total_tests || 0,
            },
          });
          return wrongAnswerSubmission;
        }

        // Update submission to indicate benchmarking is starting
        await ctx.db.submission.update({
          where: { id: submission.id },
          data: {
            status: "BENCHMARKING",
            passedTests: checkerResult.total_tests,
            totalTests: checkerResult.total_tests,
          },
        });

        // If we get here, solution is correct - run benchmark
        const benchmarkResponse = await fetch("https://labs-asterisk--tensara-benchmark.modal.run", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            code: input.code,
          }),
        });

        if (!benchmarkResponse.ok) {
          throw new Error(`Benchmark API returned ${benchmarkResponse.status}`);
        }

        const benchmarkResult = await benchmarkResponse.json();
        console.log("BENCHMARK RESULT");
        console.log(benchmarkResult);

        // Update submission with successful results
        const updatedSubmission = await ctx.db.submission.update({
          where: { id: submission.id },
          data: {
            status: "ACCEPTED",
            runtime: Math.round(benchmarkResult.average_runtime_ms),
            memory: 0,
            passedTests: checkerResult.total_tests,
            totalTests: checkerResult.total_tests,
          },
        });

        return updatedSubmission;
      } catch (error) {
        // Update submission with error status
        const failedSubmission = await ctx.db.submission.update({
          where: { id: submission.id },
          data: {
            status: "ERROR",
            passedTests: 0,
            totalTests: 0,
          },
        });

        throw new TRPCError({
          code: "INTERNAL_SERVER_ERROR", 
          message: "Failed to evaluate submission",
          cause: error,
        });
      }
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

  // Get submission status
  getSubmissionStatus: protectedProcedure
    .input(z.object({ submissionId: z.string() }))
    .query(async ({ ctx, input }) => {
      const submission = await ctx.db.submission.findUnique({
        where: { id: input.submissionId },
      });

      if (!submission) {
        throw new TRPCError({
          code: "NOT_FOUND",
          message: "Submission not found",
        });
      }

      return submission;
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
