import { z } from "zod";
import {
  createTRPCRouter,
  publicProcedure,
} from "~/server/api/trpc";
import { TRPCError } from "@trpc/server";
import type { PrismaClient } from "@prisma/client";
import { LANGUAGE_PROFILE_DISPLAY_NAMES } from "~/constants/language";
import { PROBLEM_DIFFICULTY_MULTIPLIERS, START_RATING } from "~/constants/problem";
import { ADJUSTMENT_FACTOR, BASELINE_UTILISATION, GPU_THEORETICAL_PERFORMANCE } from "~/constants/gpu";

async function getUserRating(
  ctx: { db: PrismaClient },
  userId: string
) {
  const user = await ctx.db.user.findFirst({
    where: {
      id: userId,
    },
    select: {
      rating: true,
    },
  });

  if (!user) {
    throw new TRPCError({
      code: "NOT_FOUND",
      message: "User not found",
    });
  }

  const best_submissions = await ctx.db.submission.groupBy({
    by: ["problemId", "gpuType"],
    _max: {
      gflops: true,
    },
    where: {
      userId: userId,
    },
  });
  let rating = START_RATING;
  for (const submission of best_submissions) {
    const gpu_type = submission.gpuType;
    const gflops = submission._max.gflops;
    const problem_id = submission.problemId;
    const problem = await ctx.db.problem.findUnique({
      where: {
        id: problem_id,
      },
      select: {
        slug: true,
        difficulty: true,
      },
    });
    if (!problem?.difficulty) {
      continue;
    }
    const difficulty_multiplier = PROBLEM_DIFFICULTY_MULTIPLIERS[problem.difficulty as keyof typeof PROBLEM_DIFFICULTY_MULTIPLIERS];
    if (!gpu_type || !gflops) {
      continue;
    }
    
    const theoretical_tflops = GPU_THEORETICAL_PERFORMANCE[gpu_type];
    if (!theoretical_tflops) {
      continue;
    }
    const utilisation = (gflops / (theoretical_tflops * 1000)) * 100;
    const scaled_utilisation = (utilisation / BASELINE_UTILISATION)^2 * 100;

    const performance_rating = difficulty_multiplier * scaled_utilisation;
    const expected_rating = difficulty_multiplier * 100;
    rating = ADJUSTMENT_FACTOR * (performance_rating - expected_rating) / 100 + rating;
    rating = Math.round(rating);
  }

  //put rating in the database
  await ctx.db.user.update({
    where: { id: userId },
    data: { rating: rating },
  });

  return rating;
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

  if (!user.rank) {
    //first get the rating of the user
    const rating = await getUserRating(ctx, userId);

    //then get the rank of the user
    const rank = await ctx.db.user.count({
      where: { rating: { gt: rating } },
    }) + 1;

    //update the rank in the database
    await ctx.db.user.update({
      where: { id: userId },
      data: { rank: rank },
    });

    return rank;
  } 

  const rank = Number(user.rank);
  if (isNaN(rank)) {
    throw new TRPCError({
      code: "INTERNAL_SERVER_ERROR",
      message: "Invalid rank value",
    });
  }

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
