import { z } from "zod";
import {
  createTRPCRouter,
  publicProcedure,
} from "~/server/api/trpc";
import { TRPCError } from "@trpc/server";
import type { PrismaClient } from "@prisma/client";
import { LANGUAGE_PROFILE_DISPLAY_NAMES } from "~/constants/language";
import { PROBLEM_DIFFICULTY_MULTIPLIERS, START_RATING, ADJUSTMENT_FACTOR } from "~/constants/problem";

async function getUserRating(
  ctx: { db: PrismaClient },
  userId: string
): Promise<number> {
  // Get user's current rating or default to start rating
  const user = await ctx.db.user.findFirst({
    where: { id: userId },
    select: { rating: true },
  });
  
  if (!user) {
    throw new TRPCError({
      code: "NOT_FOUND",
      message: "User not found",
    });
  }
  
  // Start with base rating for new users
  let rating = START_RATING;
  let totalRatingChanges = 0;
  let problemsProcessed = 0;

  // Get user's best submissions for each problem-GPU combination
  const userBestSubmissions = await ctx.db.submission.groupBy({
    by: ["problemId", "gpuType"],
    _max: { gflops: true },
    where: { userId: userId },
  });

  // Process each submission to calculate rating change
  for (const submission of userBestSubmissions) {
    const gpuType = submission.gpuType;
    const gflops = submission._max.gflops;
    const problemId = submission.problemId;
    
    // Skip invalid submissions
    if (!gpuType || !gflops) continue;

    // Get problem details
    const problem = await ctx.db.problem.findUnique({
      where: { id: problemId },
      select: { slug: true, difficulty: true },
    });
    
    if (!problem?.difficulty) continue;
    const difficultyMultiplier = PROBLEM_DIFFICULTY_MULTIPLIERS[problem.difficulty as keyof typeof PROBLEM_DIFFICULTY_MULTIPLIERS];
    
    // Find all submissions for this problem-GPU combination
    const allSubmissionsForProblemGpu = await ctx.db.submission.findMany({
      where: {
        problemId: problemId,
        gpuType: gpuType,
      },
      select: {
        userId: true,
        gflops: true,
      },
      orderBy: {
        gflops: "desc",
      },
    });
    
    // Group submissions by user (keeping only best per user)
    const userBestGflops: Record<string, number> = {};
    for (const sub of allSubmissionsForProblemGpu) {
      if (sub?.gflops === null || sub?.userId === null) continue;
      if (!userBestGflops[sub.userId]) {
        userBestGflops[sub.userId] = sub.gflops;
      } else if (userBestGflops[sub.userId]! < sub.gflops) {
        userBestGflops[sub.userId] = sub.gflops;
      }
    }
    const uniqueGflopValues = Object.values(userBestGflops).sort((a, b) => b - a);
    
    const userRank = uniqueGflopValues.findIndex(value => value === gflops) + 1;
    const totalUniqueSubmissions = uniqueGflopValues.length;
    
    if (totalUniqueSubmissions <= 1) continue;
    
    const percentile = (userRank - 1) / Math.max(totalUniqueSubmissions - 1, 1);
    
    const performanceScore = calculatePerformanceScore(percentile, difficultyMultiplier);
    
    const expectedScore = 100 * difficultyMultiplier;
    
    const ratingChange = ADJUSTMENT_FACTOR * (performanceScore - expectedScore) / 100;
    
    totalRatingChanges += ratingChange;
    problemsProcessed++;
  }
  
  if (problemsProcessed > 0) {
    rating += totalRatingChanges;
  }
  
  return Math.round(rating);
}

function calculatePerformanceScore(percentile: number, difficultyMultiplier: number): number {
  // Top 1% gets ~190 * difficultyMultiplier
  // Top 10% gets ~150 * difficultyMultiplier
  // Top 25% gets ~120 * difficultyMultiplier
  // Top 50% gets ~100 * difficultyMultiplier
  // Bottom 25% gets ~70 * difficultyMultiplier
  // Bottom 10% gets ~50 * difficultyMultiplier
  return (200 - 150 * Math.pow(percentile, 0.7)) * difficultyMultiplier;
}


async function getUserRank(
  ctx: { db: PrismaClient },
  userId: string
) {
  const user = await ctx.db.user.findFirst({
    where: { id: userId },
    select: {
      rank: true,
    },
  });

  if (!user) {
    throw new TRPCError({
      code: "NOT_FOUND",
      message: "User not found",
    });
  }


  //first get the rating of the user
  const rating = await getUserRating(ctx, userId);

  //then get the rank of the user
  const rank = await ctx.db.user.count({
    where: { rating: { gt: rating } },
  }) + 1;


  //if the rank is NaN, throw an error
  if (isNaN(rank)) {
    throw new TRPCError({
      code: "INTERNAL_SERVER_ERROR",
      message: "Invalid rank value",
    });
  }

  //update the rank in the database
  await ctx.db.user.update({
    where: { id: userId },
    data: { rank: rank },
  });
  
  return rank; 
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
      
      //get the percentage of langauge used in solved problems
      const solvedProblemsWithLanguage = await ctx.db.submission.groupBy({
        by: ["language"],
        _count: {
          id: true,
        },
        where: {
          userId: user.id,
          status: "ACCEPTED",
        },
      });
      const totalSolvedProblems = solvedProblemsWithLanguage.reduce((acc, curr) => acc + curr._count.id, 0);
      const languagePercentage = solvedProblemsWithLanguage.map((language) => {
        return {
          language: LANGUAGE_PROFILE_DISPLAY_NAMES[language.language],
          percentage: Number(((language._count.id / totalSolvedProblems) * 100).toFixed(2)),
        };
      });
    
      // Find current user's rank
      const userRank = await getUserRank(ctx, user.id);
      const userRating = await getUserRating(ctx, user.id);

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
          gpuType: true,
          problem: {
            select: {
              id: true,
              title: true,
              slug: true,
            },
          },
          language: true,
        },
        orderBy: { createdAt: "desc" },
        take: 5,
      });


      // Get all submission dates grouped by day
      const submissionDates = await ctx.db.submission.groupBy({
        by: ["createdAt"],
        where: {
          userId: user.id,
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

      return {
        username: user.username,
        name: user.name,
        image: user.image,
        joinedAt: user.createdAt.toISOString(),
        stats: {
          submissions: submissionsCount,
          solvedProblems,
          ranking: userRank,
          rating: userRating,
        },
        recentSubmissions: recentSubmissions.map((sub) => ({
          id: sub.id,
          problemId: sub.problem.slug,
          problemName: sub.problem.title,
          date: sub.createdAt.toISOString().split("T")[0],
          status: (sub.status ?? "pending").toLowerCase(),
          runtime: sub.runtime ? `${(sub.runtime).toFixed(2)}ms` : "N/A",
          gflops: sub.gflops ? `${sub.gflops.toFixed(2)}` : "N/A",
          gpuType: sub.gpuType,
          language: sub.language,
        })),
        activityData,
        languagePercentage
      };
    }),
});
