import { z } from "zod";
import {
  createTRPCRouter,
  protectedProcedure,
  publicProcedure,
} from "~/server/api/trpc";
import { TRPCError } from "@trpc/server";
import NodeCache from "node-cache";
import type { Problem, Submission, User } from "@prisma/client";
import { PrismaClient } from "@prisma/client";

// Create cache with 5 minute TTL
const leaderboardCache = new NodeCache({ stdTTL: 300 });
let requestCounter = 0;

// Type for our cached data
type ProblemLeaderboard = {
  slug: string;
  title: string;
  id: string;
  topSubmissions: Array<{
    id: string;
    gflops: number;
    gpuType: string | null;
    username: string | null;
    runtime: number | null;
  }>;
};

// Warm up cache on server start
async function warmupCache(ctx: { db: PrismaClient }) {
  console.log(`[CACHE WARMUP] Starting cache warmup for all GPU types`);
  const startTime = Date.now();

  try {
    // Precompute for 'all' and each GPU type
    const gpuTypes = ["all", "A100", "H100", "RTX4090"];
    await Promise.all(
      gpuTypes.map(async (gpuType) => {
        console.log(`[CACHE WARMUP] Processing GPU: ${gpuType}`);
        const data = await computeLeaderboardData(ctx, gpuType);
        leaderboardCache.set(`leaderboard-${gpuType}`, data);
      })
    );

    const duration = Date.now() - startTime;
    console.log(
      `[CACHE WARMUP COMPLETE] Leaderboard cache warmed up (took ${duration}ms)`
    );
  } catch (error) {
    console.error(`[CACHE WARMUP ERROR] Failed to warm up cache:`, error);
  }
}

// Export the function to be called elsewhere
export const initializeLeaderboardCache = async () => {
  if (process.env.NODE_ENV === "production") {
    const { db } = await import("~/server/db");
    void warmupCache({ db });
  }
};

export const submissionsRouter = createTRPCRouter({
  // all submissions (public or not) for the current user
  getAllUserSubmissions: protectedProcedure.query(async ({ ctx }) => {
    const submissions = await ctx.db.submission.findMany({
      where: {
        userId: ctx.session.user.id,
      },
      include: {
        user: {
          select: {
            username: true,
          },
        },
        problem: {
          select: {
            title: true,
            slug: true,
          },
        },
      },
      orderBy: {
        createdAt: "desc",
      },
    });

    return submissions;
  }),

  // all submissions (public or not)
  getLeaderboardSubmissions: publicProcedure.query(async ({ ctx }) => {
    const submissions = await ctx.db.submission.findMany({
      where: {
        status: "ACCEPTED",
      },
      include: {
        user: {
          select: {
            username: true,
          },
        },
        problem: {
          select: {
            title: true,
            slug: true,
          },
        },
      },
      orderBy: {
        createdAt: "desc",
      },
    });

    return submissions;
  }),

  getSubmissionById: publicProcedure
    .input(z.object({ id: z.string() }))
    .query(async ({ ctx, input }) => {
      const submission = await ctx.db.submission.findUnique({
        where: { id: input.id },
        include: {
          problem: {
            select: {
              title: true,
              slug: true,
            },
          },
          user: {
            select: {
              username: true,
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

      // If user is not logged in or not the owner and submission is not public
      if (
        (!ctx.session?.user || ctx.session.user.id !== submission.userId) &&
        !submission.isPublic
      ) {
        throw new TRPCError({
          code: "FORBIDDEN",
          message: "You don't have permission to view this submission",
        });
      }

      return submission;
    }),

  getBestSubmissionsByProblem: publicProcedure
    .input(
      z.object({
        gpuType: z.string().optional().default("all"),
      })
    )
    .query(async ({ ctx, input }) => {
      const cacheKey = `leaderboard-${input.gpuType}`;

      // Check cache first
      const cachedData = leaderboardCache.get<ProblemLeaderboard[]>(cacheKey);
      if (cachedData) {
        // Log each request with its counter value
        console.log(
          `[CACHE HIT] Leaderboard cache used for GPU: ${input.gpuType} (Request #${requestCounter})`
        );

        requestCounter++;
        if (requestCounter % 10 === 0) {
          console.log(
            `[CACHE REFRESH] Triggering background refresh after ${requestCounter} requests for GPU: ${input.gpuType}`
          );
          void refreshLeaderboardCache(ctx, input.gpuType, cacheKey);
        }
        return cachedData;
      }

      // Cache miss - compute data and store in cache
      console.log(
        `[CACHE MISS] Computing leaderboard data for GPU: ${input.gpuType}`
      );
      const data = await computeLeaderboardData(ctx, input.gpuType);
      console.log(
        `[CACHE STORE] Storing new leaderboard data for GPU: ${input.gpuType}`
      );
      leaderboardCache.set(cacheKey, data);
      return data;
    }),
});

async function computeLeaderboardData(
  ctx: { db: PrismaClient },
  gpuType: string
): Promise<ProblemLeaderboard[]> {
  // Get all problems with their top submissions in a single query
  const problems = await ctx.db.problem.findMany({
    select: {
      id: true,
      slug: true,
      title: true,
      submissions: {
        where: {
          status: "ACCEPTED",
          gflops: { not: null },
          ...(gpuType !== "all" ? { gpuType } : {}),
        },
        select: {
          id: true,
          gflops: true,
          gpuType: true,
          runtime: true,
          user: {
            select: {
              username: true,
            },
          },
        },
        orderBy: {
          gflops: "desc",
        },
        take: 100, // Get enough submissions to process top performers
      },
    },
    orderBy: {
      createdAt: "asc",
    },
  });

  // Process each problem's submissions server-side
  return problems.map(
    (problem: {
      id: string;
      slug: string;
      title: string;
      submissions: Array<{
        id: string;
        gflops: number | null;
        gpuType: string | null;
        runtime: number | null;
        user: { username: string | null };
      }>;
    }) => {
      const userBestMap = new Map<string, (typeof problem.submissions)[0]>();

      // Get best submission per user-GPU combination
      for (const submission of problem.submissions) {
        if (!submission.gflops) continue;
        const key = `${submission.user.username ?? "Anonymous"}-${
          submission.gpuType
        }`;
        const current = userBestMap.get(key);
        if (!current || submission.gflops > current.gflops!) {
          userBestMap.set(key, submission);
        }
      }

      // Get top 3 overall
      const topSubmissions = Array.from(userBestMap.values())
        .sort((a, b) => b.gflops! - a.gflops!)
        .slice(0, 3)
        .map((sub) => ({
          id: sub.id,
          gflops: sub.gflops!,
          gpuType: sub.gpuType,
          username: sub.user.username,
          runtime: sub.runtime,
        }));

      return {
        slug: problem.slug,
        title: problem.title,
        id: problem.id,
        topSubmissions,
      };
    }
  );
}

// Function to refresh cache in background
async function refreshLeaderboardCache(
  ctx: { db: PrismaClient },
  gpuType: string,
  cacheKey: string
) {
  try {
    console.log(`[CACHE REFRESH START] Beginning refresh for GPU: ${gpuType}`);
    const startTime = Date.now();

    const freshData = await computeLeaderboardData(ctx, gpuType);
    const duration = Date.now() - startTime;

    console.log(
      `[CACHE REFRESH COMPLETE] Refreshed cache for GPU: ${gpuType} (took ${duration}ms)`
    );
    leaderboardCache.set(cacheKey, freshData);
  } catch (error) {
    console.error(
      `[CACHE REFRESH ERROR] Failed to refresh cache for GPU: ${gpuType}`,
      error
    );
  }
}
