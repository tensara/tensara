import { z } from "zod";
import {
  createTRPCRouter,
  protectedProcedure as _protectedProcedure,
  publicProcedure,
} from "~/server/api/trpc";
import { TRPCError } from "@trpc/server";
import type { PrismaClient } from "@prisma/client";

// Define interfaces for submission and problem data
interface SubmissionData {
  id: string;
  gflops: number | null;
  runtime: number | null;
  createdAt: Date;
  problemId: string;
  problem: {
    id: string;
    difficulty: string;
  };
}

interface SubmissionBasic {
  id: string;
  userId: string;
  createdAt: Date;
}

/**
 * Enhanced Tensara Score Calculator
 *
 * This implements a more comprehensive scoring system that takes into account:
 * 1. Problem difficulty multiplier
 * 2. Efficiency bonus based on runtime percentile
 * 3. First-solve bonus for being among the early solvers
 * 4. Consistency component for regular activity
 * 5. Diminishing returns for repeated solutions to the same problem
 */

// Difficulty multipliers
const DIFFICULTY_MULTIPLIERS: Record<string, number> = {
  EASY: 1.0,
  MEDIUM: 1.5,
  HARD: 2.5,
  EXTREME: 4.0,
};

// Maximum percentage improvement bonus (if new solution is 100% better)
const MAX_IMPROVEMENT_BONUS = 0.5;

// Minimum improvement threshold to be considered significant (5%)
const MIN_IMPROVEMENT_THRESHOLD = 0.05;

// First N solvers get a bonus
const FIRST_SOLVE_COUNT = 5;
const FIRST_SOLVE_BONUS = 10;

// Consistency bonus parameters
const CONSISTENCY_WINDOW_DAYS = 30; // Look at last 30 days
const MAX_CONSISTENCY_BONUS = 0.2; // Maximum 20% bonus

// Define the type for problem submissions
const problemSubmissions = new Map<string, SubmissionBasic[]>();

// Define a normalization factor
const SCORE_NORMALIZATION_FACTOR = 1000;

// Calculate an improved score taking into account multiple factors
async function calculateEnhancedScore(
  ctx: { db: PrismaClient },
  userId: string
) {
  // Get all of the user's submissions with problem data
  const submissions = (await ctx.db.submission.findMany({
    where: {
      userId,
      status: "ACCEPTED",
    },
    select: {
      id: true,
      gflops: true,
      runtime: true,
      createdAt: true,
      problemId: true,
      problem: {
        select: {
          id: true,
          difficulty: true,
        },
      },
    },
    orderBy: {
      createdAt: "asc", // Sorted by time to determine first solutions
    },
  })) as SubmissionData[];

  if (submissions.length === 0) return 0;

  // Track best submission GFLOPS per problem
  const bestProblemSubmissions = new Map<
    string,
    { gflops: number; score: number }
  >();

  // Get all submissions for each problem to determine relative position
  await Promise.all(
    [...new Set(submissions.map((s: SubmissionData) => s.problemId))].map(
      async (problemId) => {
        const allProblemSubs = (await ctx.db.submission.findMany({
          where: {
            problemId,
            status: "ACCEPTED",
          },
          select: {
            id: true,
            userId: true,
            createdAt: true,
          },
          orderBy: {
            createdAt: "asc",
          },
        })) as SubmissionBasic[];

        problemSubmissions.set(problemId, allProblemSubs);
      }
    )
  );

  // Calculate submission dates for consistency bonus
  const thirtyDaysAgo = new Date();
  thirtyDaysAgo.setDate(thirtyDaysAgo.getDate() - CONSISTENCY_WINDOW_DAYS);

  const uniqueSubmissionDays = new Set(
    submissions
      .filter((sub: SubmissionData) => new Date(sub.createdAt) >= thirtyDaysAgo)
      .map((sub: SubmissionData) => sub.createdAt.toISOString().split("T")[0])
  );

  const consistencyFactor = Math.min(
    1,
    uniqueSubmissionDays.size / CONSISTENCY_WINDOW_DAYS
  );
  const consistencyBonus = 1 + MAX_CONSISTENCY_BONUS * consistencyFactor;

  // Calculate scores for each submission
  let totalScore = 0;

  for (const submission of submissions) {
    const { gflops, problemId, problem } = submission;
    if (!gflops) continue;

    // 1. Base score with difficulty multiplier
    const difficultyMultiplier =
      DIFFICULTY_MULTIPLIERS[problem.difficulty] ?? 1.0;
    let submissionScore = gflops * difficultyMultiplier;

    // 2. Check if this is an improvement over previous best for this problem
    const previousBest = bestProblemSubmissions.get(problemId) as
      | { gflops: number; score: number }
      | undefined;

    if (previousBest) {
      if (gflops > previousBest.gflops) {
        // Calculate improvement percentage (capped at 100%)
        const improvementPercentage = Math.min(
          1,
          (gflops - previousBest.gflops) / previousBest.gflops
        );

        // Only count as improvement if it meets the minimum threshold
        if (improvementPercentage >= MIN_IMPROVEMENT_THRESHOLD) {
          // Apply diminishing returns for repeated solutions unless significant improvement
          const improvementBonus =
            improvementPercentage * MAX_IMPROVEMENT_BONUS;
          submissionScore =
            gflops * difficultyMultiplier * (1 + improvementBonus);
          bestProblemSubmissions.set(problemId, {
            gflops,
            score: submissionScore,
          });
        } else {
          // Improvement less than threshold, don't update score
          continue;
        }
      } else {
        // Not an improvement, don't count this submission
        continue;
      }
    } else {
      // First submission for this problem
      bestProblemSubmissions.set(problemId, { gflops, score: submissionScore });

      // 3. Check if user was among first N solvers
      const allSubs = problemSubmissions.get(problemId) ?? [];
      if (Array.isArray(allSubs)) {
        const userPos = allSubs.findIndex(
          (s: SubmissionBasic) => s.userId === userId
        );

        if (userPos >= 0 && userPos < FIRST_SOLVE_COUNT) {
          // Apply bonus for being among first solvers (higher bonus for earlier solvers)
          const firstSolveMultiplier =
            1 +
            (FIRST_SOLVE_BONUS * (FIRST_SOLVE_COUNT - userPos)) /
              FIRST_SOLVE_COUNT;
          submissionScore *= firstSolveMultiplier;
        }
      }
    }

    totalScore += submissionScore;
  }

  // Apply consistency bonus to total score
  totalScore *= consistencyBonus;

  // Normalize the total score to a smaller range
  totalScore /= SCORE_NORMALIZATION_FACTOR; // Normalize the score

  return parseFloat(totalScore.toFixed(2)); // Return as a float to avoid trailing zeros
}

export const usersRouter = createTRPCRouter({
  getByUsername: publicProcedure
    .input(z.object({ username: z.string() }))
    .query(async ({ ctx, input }) => {
      const user = await ctx.db.user.findFirst({
        where: { username: input.username },
        select: {
          id: true,
          name: true,
          username: true,
          image: true,
          createdAt: true,
        },
      });

      if (!user) {
        throw new TRPCError({
          code: "NOT_FOUND",
          message: "User not found",
        });
      }

      // Get total submissions count
      const submissionsCount = await ctx.db.submission.count({
        where: { userId: user.id },
      });

      // Get number of solved problems (distinct problems with accepted submissions)
      const solvedProblems = await ctx.db.problem.count({
        where: {
          submissions: {
            some: {
              userId: user.id,
              status: "ACCEPTED",
            },
          },
        },
      });

      // Get user rank based on total score
      // First, get all users with their submission scores
      const allUserScores = await Promise.all(
        (
          await ctx.db.user.findMany({
            select: {
              id: true,
            },
          })
        ).map(async (u) => ({
          id: u.id,
          score: await calculateEnhancedScore(ctx, u.id),
        }))
      );

      // Sort by score in descending order
      allUserScores.sort((a, b) => b.score - a.score);

      // Find current user's rank
      const userRank = allUserScores.findIndex((u) => u.id === user.id) + 1;
      const userScore = allUserScores.find((u) => u.id === user.id)?.score ?? 0;

      // Get recent submissions
      const recentSubmissions = await ctx.db.submission.findMany({
        where: {
          userId: user.id,
          // Only include public submissions or if current user is the owner
          OR: [{ isPublic: true }, { userId: ctx.session?.user?.id }],
        },
        select: {
          id: true,
          createdAt: true,
          status: true,
          runtime: true,
          gflops: true,
          problem: {
            select: {
              id: true,
              title: true,
              slug: true,
            },
          },
        },
        orderBy: { createdAt: "desc" },
        take: 5,
      });

      // Get activity data for heatmap (last 6 months)
      const sixMonthsAgo = new Date();
      sixMonthsAgo.setMonth(sixMonthsAgo.getMonth() - 6);

      // Get all submission dates grouped by day
      const submissionDates = await ctx.db.submission.groupBy({
        by: ["createdAt"],
        where: {
          userId: user.id,
          createdAt: {
            gte: sixMonthsAgo,
          },
        },
        _count: {
          id: true,
        },
      });

      // Format the dates for the activity calendar
      const activityData = submissionDates.map((day) => {
        // Format date to YYYY-MM-DD
        const date = new Date(day.createdAt);
        const formattedDate = date.toISOString().split("T")[0];

        return {
          date: formattedDate,
          count: day._count.id,
        };
      });

      console.log({ userScore });

      return {
        username: user.username,
        name: user.name,
        image: user.image,
        joinedAt: user.createdAt.toISOString(),
        stats: {
          submissions: submissionsCount,
          solvedProblems,
          ranking: userRank,
          score: userScore,
        },
        recentSubmissions: recentSubmissions.map((sub) => ({
          id: sub.id,
          problemId: sub.problem.slug,
          problemName: sub.problem.title,
          date: sub.createdAt.toISOString().split("T")[0],
          status: (sub.status ?? "pending").toLowerCase(),
          runtime: sub.runtime ? `${(sub.runtime / 1000).toFixed(2)}s` : "N/A",
        })),
        activityData,
      };
    }),
});
