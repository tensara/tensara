import { z } from "zod";
import { createTRPCRouter, protectedProcedure, publicProcedure } from "../trpc";
import { TRPCError } from "@trpc/server";
import { Prisma } from "@prisma/client";

// Simulated evaluation delay
const EVAL_DELAY_MS = 1500;

// Add these types at the top of the file
type BenchmarkTestResult = {
  test_id: number;
  runtime_ms: number;
  gflops: number;
};

type BenchmarkSuccessResponse = {
  status: "success";
  test_results: BenchmarkTestResult[];
  average_gflops: number;
};

type BenchmarkErrorResponse = {
  error: string;
  details: string;
};

type BenchmarkResponse = BenchmarkSuccessResponse | BenchmarkErrorResponse;

const SubmissionStatus = {
  PENDING: "PENDING",
  CHECKING: "CHECKING",
  BENCHMARKING: "BENCHMARKING",
  ACCEPTED: "ACCEPTED",
  ERROR: "ERROR",
  WRONG_ANSWER: "WRONG_ANSWER",
  TIMEOUT: "TIMEOUT",
} as const;

type SubmissionStatus = typeof SubmissionStatus[keyof typeof SubmissionStatus];

// Add response types
type SubmissionErrorResponse = {
  status: SubmissionStatus;
  id: string;
  passedTests: number | null;
  totalTests: number | null;
  errorMessage: string;
  errorDetails?: string;
};

type SubmissionSuccessResponse = {
  status: SubmissionStatus;
  id: string;
  passedTests: number | null;
  totalTests: number | null;
  runtime: number | null;
  gflops: number | null;
  benchmarkResults: BenchmarkTestResult[];
};

type SubmissionResponse = SubmissionErrorResponse | SubmissionSuccessResponse;

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
      });

      if (!problem) return null;

      return {
        ...problem
      };
    }),

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
        where: { slug: input.problemSlug }
      }) as Promise<any>);

      const submission = await ctx.db.submission.create({
        data: {
          code: input.code,
          language: input.language,
          userId: ctx.session.user.id,
          problemId: problem.id,
          status: SubmissionStatus.PENDING,
        },
      });

      try {
        console.log("problem bindings", problem.tests);
        console.log("problem reference", problem.reference);
        console.log("input code", input.code);

        await ctx.db.submission.update({
          where: { id: submission.id },
          data: {
            status: SubmissionStatus.CHECKING,
          },
        });

        const checkerResponse = await fetch("https://labs-asterisk--tensara-checker.modal.run", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            solution_code: input.code,
            tests_code: problem.tests,
            reference_code: problem.reference,
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
              status: SubmissionStatus.WRONG_ANSWER,
              passedTests: checkerResult.passed_tests || 0,
              totalTests: checkerResult.total_tests || 0,
              errorMessage: checkerResult.error || "Solution produced incorrect results",
              errorDetails: checkerResult.details || JSON.stringify(checkerResult.test_results || []),
            } as any,
          }) as unknown as SubmissionWithCustomFields;
          
          const response: SubmissionErrorResponse = {
            status: wrongAnswerSubmission.status as SubmissionStatus,
            id: wrongAnswerSubmission.id,
            passedTests: wrongAnswerSubmission.passedTests,
            totalTests: wrongAnswerSubmission.totalTests,
            errorMessage: wrongAnswerSubmission.errorMessage || "Unknown error",
            errorDetails: wrongAnswerSubmission.errorDetails,
          };
          return response;
        }

        await ctx.db.submission.update({
          where: { id: submission.id },
          data: {
            status: SubmissionStatus.BENCHMARKING,
            passedTests: checkerResult.total_tests,
            totalTests: checkerResult.total_tests,
          },
        });

        const benchmarkResponse = await fetch("https://labs-asterisk--tensara-benchmark.modal.run", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            solution_code: input.code,
            tests_code: problem.tests,
          }),
        });

        if (!benchmarkResponse.ok) {
          throw new Error(`Benchmark API returned ${benchmarkResponse.status}`);
        }

        const benchmarkResult = await benchmarkResponse.json() as BenchmarkResponse;
        console.log("BENCHMARK RESULT");
        console.log(benchmarkResult);

        if ('error' in benchmarkResult) {
          // Handle error case
          const errorSubmission = await ctx.db.submission.update({
            where: { id: submission.id },
            data: {
              status: SubmissionStatus.ERROR,
              passedTests: checkerResult.total_tests,
              totalTests: checkerResult.total_tests,
              errorMessage: benchmarkResult.error,
              errorDetails: benchmarkResult.details,
            } as any,
          }) as unknown as SubmissionWithCustomFields;
          
          const response: SubmissionErrorResponse = {
            status: errorSubmission.status as SubmissionStatus,
            id: errorSubmission.id,
            passedTests: errorSubmission.passedTests,
            totalTests: errorSubmission.totalTests,
            errorMessage: errorSubmission.errorMessage || "Unknown error",
            errorDetails: errorSubmission.errorDetails,
          };
          return response;
        }

        // Update submission with successful results
        const updatedSubmission = await ctx.db.submission.update({
          where: { id: submission.id },
          data: {
            status: SubmissionStatus.ACCEPTED,
            runtime: benchmarkResult.test_results.reduce((acc: number, test: BenchmarkTestResult) => acc + test.runtime_ms, 0) / benchmarkResult.test_results.length,
            gflops: benchmarkResult.average_gflops,
            passedTests: checkerResult.total_tests,
            totalTests: checkerResult.total_tests,
            benchmarkResults: benchmarkResult.test_results,
          } as any,
        }) as unknown as SubmissionWithCustomFields;

        const response: SubmissionSuccessResponse = {
          status: updatedSubmission.status as SubmissionStatus,
          id: updatedSubmission.id,
          passedTests: updatedSubmission.passedTests,
          totalTests: updatedSubmission.totalTests,
          runtime: updatedSubmission.runtime,
          gflops: updatedSubmission.gflops,
          benchmarkResults: updatedSubmission.benchmarkResults || [],
        };
        return response;
      } catch (error) {
        // Update submission with error status
        const failedSubmission = await ctx.db.submission.update({
          where: { id: submission.id },
          data: {
            status: SubmissionStatus.ERROR,
            passedTests: 0,
            totalTests: 0,
            errorMessage: error instanceof Error ? error.message : "Unknown error occurred",
            errorDetails: error instanceof Error ? error.stack : undefined,
          } as any,
        }) as unknown as SubmissionWithCustomFields;

        throw new TRPCError({
          code: "INTERNAL_SERVER_ERROR", 
          message: failedSubmission.errorMessage || "Failed to evaluate submission",
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
      }) as unknown as (SubmissionWithCustomFields & {
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
});
