import { z } from "zod";
import { createTRPCRouter, publicProcedure } from "~/server/api/trpc";
import { TRPCError } from "@trpc/server";
import type { PrismaClient } from "@prisma/client";
import { LANGUAGE_PROFILE_DISPLAY_NAMES } from "~/constants/language";
import {
  PROBLEM_DIFFICULTY_MULTIPLIERS,
  START_RATING,
  ADJUSTMENT_FACTOR,
  FIRST_SOLVE_BONUS,
} from "~/constants/problem";

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
  let rating = START_RATING ?? 1000;
  let totalRatingChanges = 0;
  let problemsProcessed = 0;

  // Get user's best submissions for each problem-GPU combination
  // Use legacySubmission table since that's where all existing data lives
  const userBestSubmissions = await ctx.db.legacySubmission.groupBy({
    by: ["problemId", "gpuType"],
    _max: { gflops: true },
    where: { userId: userId },
  });

  // Process each submission to calculate rating change
  for (const submission of userBestSubmissions) {
    const gpuType = submission.gpuType;
    const gflops = submission._max?.gflops;
    const problemId = submission.problemId;

    // Skip invalid submissions
    if (!gpuType || !gflops) continue;

    // Get problem details
    const problem = await ctx.db.problem.findUnique({
      where: { id: problemId },
      select: { slug: true, difficulty: true },
    });

    if (!problem?.difficulty) continue;
    const difficultyMultiplier =
      PROBLEM_DIFFICULTY_MULTIPLIERS[
        problem.difficulty as keyof typeof PROBLEM_DIFFICULTY_MULTIPLIERS
      ];

    // Find all submissions for this problem-GPU combination
    const allSubmissionsForProblemGpu = await ctx.db.legacySubmission.findMany({
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
    const uniqueGflopValues = Object.values(userBestGflops).sort(
      (a, b) => b - a
    );

    const userRank =
      uniqueGflopValues.findIndex((value) => value === gflops) + 1;
    const totalUniqueSubmissions = uniqueGflopValues.length;

    if (totalUniqueSubmissions <= 1) {
      totalRatingChanges += FIRST_SOLVE_BONUS;
      problemsProcessed++;
      continue;
    }

    const percentile = (userRank - 1) / Math.max(totalUniqueSubmissions - 1, 1);

    const performanceScore = calculatePerformanceScore(
      percentile,
      difficultyMultiplier
    );

    const expectedScore = 100 * difficultyMultiplier;

    const ratingChange =
      (ADJUSTMENT_FACTOR * (performanceScore - expectedScore)) / 100;

    totalRatingChanges += ratingChange;
    problemsProcessed++;
  }

  if (problemsProcessed > 0) {
    rating += totalRatingChanges;
  }

  const final_rating = Math.round(rating);
  //update the rating in the database
  await ctx.db.user.update({
    where: { id: userId },
    data: { rating: final_rating },
  });

  return final_rating;
}

function calculatePerformanceScore(
  percentile: number,
  difficultyMultiplier: number
): number {
  // Top 1% gets ~190 * difficultyMultiplier
  // Top 10% gets ~150 * difficultyMultiplier
  // Top 25% gets ~120 * difficultyMultiplier
  // Top 50% gets ~100 * difficultyMultiplier
  // Bottom 25% gets ~70 * difficultyMultiplier
  // Bottom 10% gets ~50 * difficultyMultiplier
  return (200 - 150 * Math.pow(percentile, 0.7)) * difficultyMultiplier;
}

async function getUserRank(ctx: { db: PrismaClient }, userId: string) {
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
  const rank =
    (await ctx.db.user.count({
      where: { rating: { gt: rating } },
    })) + 1;

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

// Normalized best submission type for frontend
type NormalizedBestSubmission = {
  id: string;
  gflops: number | null;
  runtimeMs: number | null;
  gpuType: string | null;
  isLegacy: boolean;
  problem: {
    title: string;
    slug: string;
  };
};

export const usersRouter = createTRPCRouter({
  getByUsername: publicProcedure
    .input(z.object({ username: z.string() }))
    .query(async ({ ctx, input }) => {
      const user = await ctx.db.user.findFirst({
        where: { username: { equals: input.username, mode: "insensitive" } },
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

      // Get total submissions count (from both tables)
      const [newSubmissionsCount, legacySubmissionsCount] = await Promise.all([
        ctx.db.submission.count({ where: { userId: user.id } }),
        ctx.db.legacySubmission.count({ where: { userId: user.id } }),
      ]);
      const submissionsCount = newSubmissionsCount + legacySubmissionsCount;

      // Get number of solved problems (distinct problems with accepted submissions from either table)
      const solvedProblems = await ctx.db.problem.count({
        where: {
          OR: [
            {
              submissions: {
                some: {
                  userId: user.id,
                  status: "ACCEPTED",
                },
              },
            },
            {
              legacySubmissions: {
                some: {
                  userId: user.id,
                  status: "ACCEPTED",
                },
              },
            },
          ],
        },
      });

      //get the percentage of langauge used in solved problems
      const solvedProblemsWithLanguage = await ctx.db.legacySubmission.groupBy({
        by: ["language"],
        _count: {
          id: true,
        },
        where: {
          userId: user.id,
          status: "ACCEPTED",
        },
      });
      const totalSolvedProblems = solvedProblemsWithLanguage.reduce(
        (acc: number, curr: { _count: { id: number } }) => acc + curr._count.id,
        0
      );
      const languagePercentage = solvedProblemsWithLanguage.map(
        (language: { language: string; _count: { id: number } }) => {
          return {
            language: LANGUAGE_PROFILE_DISPLAY_NAMES[language.language],
            percentage: Number(
              ((language._count.id / totalSolvedProblems) * 100).toFixed(2)
            ),
          };
        }
      );

      // Find current user's rank
      const userRank = await getUserRank(ctx, user.id);
      const userRating = await getUserRating(ctx, user.id);

      // Get recent submissions from legacy table (since new table is empty)
      const recentLegacySubmissions = await ctx.db.legacySubmission.findMany({
        where: {
          userId: user.id,
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
      const submissionDates = await ctx.db.legacySubmission.groupBy({
        by: ["createdAt"],
        where: {
          userId: user.id,
        },
        _count: {
          id: true,
        },
      });

      // Format the dates for the activity calendar
      const activityData = submissionDates.map(
        (day: { createdAt: Date; _count: { id: number } }) => {
          // Format date to YYYY-MM-DD
          const date = new Date(day.createdAt);
          const formattedDate = date.toISOString().split("T")[0];

          return {
            date: formattedDate,
            count: day._count.id,
          };
        }
      );

      const blogPosts = await ctx.db.blogPost.findMany({
        where: {
          authorId: user.id,
          status: "PUBLISHED",
        },
        select: {
          id: true,
          title: true,
          slug: true,
          publishedAt: true,
          createdAt: true,
          _count: {
            select: {
              upvotes: true,
            },
          },
        },
        orderBy: {
          publishedAt: "desc",
        },
        take: 5,
      });

      const [totalCommunityPosts, totalCommunityLikes] = await Promise.all([
        ctx.db.blogPost.count({
          where: {
            authorId: user.id,
            status: "PUBLISHED",
          },
        }),
        ctx.db.postUpvote.count({
          where: {
            post: {
              authorId: user.id,
            },
          },
        }),
      ]);

      return {
        id: user.id,
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
        recentSubmissions: recentLegacySubmissions.map(
          (sub: {
            id: string;
            createdAt: Date;
            status: string | null;
            runtime: number | null;
            gflops: number | null;
            gpuType: string | null;
            problem: { id: string; title: string; slug: string };
            language: string;
          }) => ({
            id: sub.id,
            problemId: sub.problem.slug,
            problemName: sub.problem.title,
            date: sub.createdAt.toISOString().split("T")[0],
            status: (sub.status ?? "pending").toLowerCase(),
            runtime: sub.runtime ? `${sub.runtime.toFixed(2)}ms` : "N/A",
            gflops: sub.gflops ? `${sub.gflops.toFixed(2)}` : "N/A",
            gpuType: sub.gpuType,
            language: sub.language,
          })
        ),
        blogPosts: blogPosts.map(
          (post: {
            id: string;
            title: string;
            slug: string | null;
            publishedAt: Date | null;
            createdAt: Date;
            _count: { upvotes: number };
          }) => ({
            id: post.id,
            title: post.title,
            slug: post.slug ?? "",
            publishedAt: (post.publishedAt ?? post.createdAt).toISOString(),
            votes: post._count.upvotes ?? 0,
          })
        ),
        communityStats: {
          totalPosts: totalCommunityPosts,
          totalLikes: totalCommunityLikes,
        },
        activityData,
        languagePercentage,
      };
    }),
  getTopRankedPlayers: publicProcedure
    .input(
      z.object({
        limit: z.number().min(1).max(1000).default(100),
        mode: z.enum(["legacy", "new"]).default("new"),
      })
    )
    .query(async ({ ctx, input }) => {
      const { limit, mode } = input;

      // Build where clause based on mode
      // Both modes query from legacySubmissions since new Submission table is empty
      const whereClause = {
        legacySubmissions: {
          some: {
            status: "ACCEPTED",
          },
        },
        rating: {
          not: null,
          gt: 0,
        },
      };

      // Get users ordered by rating
      const users = await ctx.db.user.findMany({
        where: whereClause,
        select: {
          id: true,
          name: true,
          username: true,
          image: true,
          rating: true,
          rank: true,
          _count: {
            select: {
              legacySubmissions: {
                where: {
                  status: "ACCEPTED",
                },
              },
              submissions: {
                where: {
                  status: "ACCEPTED",
                },
              },
            },
          },
        },
        orderBy: { rating: "desc" },
        take: limit,
      });

      // For each user, get their solved problems count and best submission
      const enhancedUsers = await Promise.all(
        users.map(
          async (user: {
            id: string;
            name: string | null;
            username: string | null;
            image: string | null;
            rating: number | null;
            rank: number | null;
            _count: { legacySubmissions: number; submissions: number };
          }) => {
            // Count solved problems - always from legacy submissions since new table is empty
            const solvedProblemsCount = await ctx.db.problem.count({
              where: {
                legacySubmissions: {
                  some: {
                    userId: user.id,
                    status: "ACCEPTED",
                  },
                },
              },
            });

            // Get best submission based on mode
            let bestSubmission: NormalizedBestSubmission | null = null;

            if (mode === "legacy") {
              // Legacy mode: best = highest GFLOPS, return runtimeMs: null
              const legacyBest = await ctx.db.legacySubmission.findFirst({
                where: {
                  userId: user.id,
                  gflops: { not: null },
                },
                orderBy: {
                  gflops: "desc",
                },
                select: {
                  id: true,
                  gflops: true,
                  gpuType: true,
                  problem: {
                    select: {
                      title: true,
                      slug: true,
                    },
                  },
                },
              });

              if (legacyBest) {
                bestSubmission = {
                  id: legacyBest.id,
                  gflops: legacyBest.gflops,
                  runtimeMs: null, // Don't show runtime in legacy mode
                  gpuType: legacyBest.gpuType,
                  isLegacy: true,
                  problem: legacyBest.problem,
                };
              }
            } else {
              // New mode: best = lowest runtime, return gflops: null
              // Query from legacySubmission since new table is empty
              const runtimeBest = await ctx.db.legacySubmission.findFirst({
                where: {
                  userId: user.id,
                  runtime: { not: null },
                },
                orderBy: {
                  runtime: "asc", // Lower runtime is better
                },
                select: {
                  id: true,
                  runtime: true,
                  gpuType: true,
                  problem: {
                    select: {
                      title: true,
                      slug: true,
                    },
                  },
                },
              });

              if (runtimeBest) {
                bestSubmission = {
                  id: runtimeBest.id,
                  gflops: null, // Don't show GFLOPS in runtime mode
                  runtimeMs: runtimeBest.runtime,
                  gpuType: runtimeBest.gpuType,
                  isLegacy: false, // Show as non-legacy (runtime-based)
                  problem: runtimeBest.problem,
                };
              }
            }

            // Only include users who have a best submission
            if (!bestSubmission) return null;

            return {
              id: user.id,
              username: user.username ?? "",
              name: user.name ?? "",
              image: user.image ?? "",
              rating: user.rating ?? 0,
              rank: user.rank ?? 9999,
              submissionsCount:
                user._count.legacySubmissions + user._count.submissions,
              solvedProblemsCount,
              bestSubmission,
            };
          }
        )
      );

      // Filter out null values and return only users with best submissions
      return enhancedUsers.filter(
        (
          user
        ): user is {
          id: string;
          username: string;
          name: string;
          image: string;
          rating: number;
          rank: number;
          submissionsCount: number;
          solvedProblemsCount: number;
          bestSubmission: NormalizedBestSubmission;
        } => user !== null
      );
    }),
});
