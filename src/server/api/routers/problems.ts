import { z } from "zod";
import { createTRPCRouter, protectedProcedure, publicProcedure } from "../trpc";
import { TRPCError } from "@trpc/server";

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

// Add this type definition near the other types at the top
const CheckerResultSchema = z.object({
  passed: z.boolean(),
  passed_tests: z.number().optional(),
  total_tests: z.number().optional(),
  error: z.string().optional(),
  details: z.string().optional(),
  test_results: z.array(z.unknown()).optional(),
});

// type CheckerResult = z.infer<typeof CheckerResultSchema>;

// Add this type for the submission update data
type SubmissionUpdateData = {
  status: SubmissionStatus;
  passedTests: number;
  totalTests: number;
  errorMessage: string;
  errorDetails: string;
};

type SuccessSubmissionUpdateData = {
  status: SubmissionStatus;
  runtime: number;
  gflops: number;
  passedTests: number;
  totalTests: number;
  benchmarkResults: BenchmarkTestResult[];
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
              submissions: true
            }
          }
        },
      });

      console.log("Found problems:", problems);
      return problems.map(problem => ({
        ...problem,
        submissionCount: problem._count.submissions
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
        language: z.enum(["cpp", "cuda", "python", "javascript", "typescript"]),
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
          problem: true,
        },
      });

      try {
        await ctx.db.submission.update({
          where: { id: submission.id },
          data: {
            status: SubmissionStatus.CHECKING,
          },
        });

        const checkerResponse = await fetch(
          "https://labs-asterisk--tensara-checker.modal.run",
          {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
            },
            body: JSON.stringify({
              solution_code: submission.code,
              tests_code: submission.problem.tests,
              reference_code: submission.problem.reference,
            }),
          }
        );

        if (!checkerResponse.ok) {
          throw new Error(`Checker API returned ${checkerResponse.status}`);
        }

        // Handle streaming response from checker
        const reader = checkerResponse.body?.getReader();
        if (!reader) throw new Error("No response body from checker");

        let passedTests = 0;
        let totalTests = 0;
        let finalResult = null;

        try {
          while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            const text = new TextDecoder().decode(value);
            const lines = text.trim().split('\n');
            
            for (const line of lines) {
              if (!line.trim()) continue;
              
              // Extract the JSON part after "data: " and parse it
              const match = /^data: (.+)$/.exec(line);
              if (!match?.[1]) continue;
              
              try {
                const response = JSON.parse(match[1].trim()) as {
                  status: string;
                  details?: string;
                  result?: {
                    status: string;
                  };
                  passed?: boolean;
                  error?: string;
                };

                if (response.status === "test_result" && response.result?.status === "PASSED") {
                  passedTests++;
                  totalTests++;
                  await ctx.db.submission.update({
                    where: { id: submission.id },
                    data: { passedTests, totalTests }
                  });
                } else if (response.status === "test_result") {
                  totalTests++;
                  await ctx.db.submission.update({
                    where: { id: submission.id },
                    data: { passedTests, totalTests }
                  });
                } else if (response.status === "complete") {
                  finalResult = response;
                } else if (response.status === "error") {
                  throw new Error(response.error ?? "Unknown error");
                }
              } catch (e) {
                console.error("Failed to parse SSE data:", e);
                continue;
              }
            }
          }
        } finally {
          reader.releaseLock();
        }

        if (!finalResult?.passed) {
          const updateData: SubmissionUpdateData = {
            status: SubmissionStatus.WRONG_ANSWER,
            passedTests,
            totalTests,
            errorMessage: finalResult?.error ?? "Solution produced incorrect results",
            errorDetails: finalResult?.details ?? "",
          };

          const wrongAnswerSubmission = await ctx.db.submission.update({
            where: { id: submission.id },
            data: updateData,
          });

          return {
            status: wrongAnswerSubmission.status as SubmissionStatus,
            id: wrongAnswerSubmission.id,
            passedTests: wrongAnswerSubmission.passedTests,
            totalTests: wrongAnswerSubmission.totalTests,
            errorMessage: wrongAnswerSubmission.errorMessage ?? "Unknown error",
            errorDetails: wrongAnswerSubmission.errorDetails,
          };
        }

        await ctx.db.submission.update({
          where: { id: submission.id },
          data: {
            status: SubmissionStatus.BENCHMARKING,
            passedTests,
            totalTests,
          },
        });

        const benchmarkResponse = await fetch(
          "https://labs-asterisk--tensara-benchmark.modal.run",
          {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
            },
            body: JSON.stringify({
              solution_code: submission.code,
              tests_code: submission.problem.tests,
            }),
          }
        );

        if (!benchmarkResponse.ok) {
          throw new Error(`Benchmark API returned ${benchmarkResponse.status}`);
        }

        // Handle streaming response from benchmark
        const benchmarkReader = benchmarkResponse.body?.getReader();
        if (!benchmarkReader) throw new Error("No response body from benchmark");

        let benchmarkResults: BenchmarkTestResult[] = [];
        let averageGflops = 0;
        let benchmarkError: string | null = null;
        let benchmarkErrorDetails: string | null = null;

        try {
          while (true) {
            const { done, value } = await benchmarkReader.read();
            if (done) break;

            const text = new TextDecoder().decode(value);
            const lines = text.trim().split('\n');
            
            for (const line of lines) {
              if (!line.trim()) continue;
              
              // Extract the JSON part after "data: " and parse it
              const match = /^data: (.+)$/.exec(line);
              if (!match?.[1]) continue;
              
              try {
                const response = JSON.parse(match[1].trim()) as {
                  status: string;
                  result?: {
                    test_id: number;
                    name: string;
                    runtime_ms: number;
                    gflops: number;
                  };
                  test_results?: Array<{
                    test_id: number;
                    name: string;
                    runtime_ms: number;
                    gflops: number;
                  }>;
                  average_gflops?: number;
                  error?: string;
                  details?: string;
                };

                if (response.status === "test_result" && response.result) {
                  benchmarkResults.push(response.result);
                  await ctx.db.submission.update({
                    where: { id: submission.id },
                    data: {
                      benchmarkResults: benchmarkResults
                    }
                  });
                } else if (response.status === "success" && response.test_results && response.average_gflops) {
                  averageGflops = response.average_gflops;
                  benchmarkResults = response.test_results;
                } else if (response.status === "error" && response.error) {
                  benchmarkError = response.error;
                  benchmarkErrorDetails = response.details ?? "";
                  break;
                }
              } catch (e) {
                console.error("Failed to parse benchmark SSE data:", e);
                continue;
              }
            }
          }
        } finally {
          benchmarkReader.releaseLock();
        }
        
        if (benchmarkError) {
          const errorSubmission = await ctx.db.submission.update({
            where: { id: submission.id },
            data: {
              status: SubmissionStatus.ERROR,
              passedTests,
              totalTests,
              errorMessage: benchmarkError,
              errorDetails: benchmarkErrorDetails ?? "",
            } satisfies SubmissionUpdateData,
          });

          return {
            status: errorSubmission.status as SubmissionStatus,
            id: errorSubmission.id,
            passedTests: errorSubmission.passedTests,
            totalTests: errorSubmission.totalTests,
            errorMessage: errorSubmission.errorMessage ?? "Unknown error",
            errorDetails: errorSubmission.errorDetails,
          };
        }

        // Update submission with successful results
        const updatedSubmission = await ctx.db.submission.update({
          where: { id: submission.id },
          data: {
            status: SubmissionStatus.ACCEPTED,
            runtime:
              benchmarkResults.reduce(
                (acc, test) => acc + test.runtime_ms,
                0
              ) / benchmarkResults.length,
            gflops: averageGflops,
            passedTests,
            totalTests,
            benchmarkResults: benchmarkResults,
          } satisfies SuccessSubmissionUpdateData,
        });

        return {
          status: updatedSubmission.status as SubmissionStatus,
          id: updatedSubmission.id,
          passedTests: updatedSubmission.passedTests,
          totalTests: updatedSubmission.totalTests,
          runtime: updatedSubmission.runtime,
          gflops: updatedSubmission.gflops,
          benchmarkResults: updatedSubmission.benchmarkResults ?? [],
        };

      } catch (error) {
        // Update submission with error status
        const failedSubmission = await ctx.db.submission.update({
          where: { id: submission.id },
          data: {
            status: SubmissionStatus.ERROR,
            passedTests: 0,
            totalTests: 0,
            errorMessage:
              error instanceof Error ? error.message : "Unknown error occurred",
            errorDetails: error instanceof Error ? error.stack ?? "" : "",
          } satisfies SubmissionUpdateData,
        });

        throw new TRPCError({
          code: "INTERNAL_SERVER_ERROR",
          message:
            failedSubmission.errorMessage ?? "Failed to evaluate submission",
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

      return submission;
    }),
});
