import { z } from "zod";
import { TRPCError } from "@trpc/server";
import type { PrismaClient } from "@prisma/client";
import {
  createTRPCRouter,
  publicProcedure,
  protectedProcedure,
} from "~/server/api/trpc";
import NodeCache from "node-cache";
import { gpuTypes } from "~/constants/gpu";
import { LeaderboardMode, type LeaderboardModeType } from "~/types/submission";

// Create cache with 5 minute TTL
const leaderboardCache = new NodeCache({ stdTTL: 300 });
let requestCounter = 0;

// Add a new map for problem-specific leaderboard cache
const problemLeaderboardCache = new NodeCache({ stdTTL: 300 });
let problemRequestCounter = 0;

// Type for our cached data
type ProblemLeaderboard = {
  slug: string;
  title: string;
  id: string;
  topSubmissions: Array<{
    id: string;
    gflops: number | null;
    gpuType: string | null;
    language: string | null;
    username: string | null;
    runtime: number | null;
    isLegacy: boolean;
  }>;
};

// Add a type for problem leaderboard data
type ProblemLeaderboardEntry = {
  id: string;
  username: string | null;
  gflops: number | null;
  runtime: number | null;
  createdAt: Date;
  gpuType: string | null;
  language: string | null;
  isPublic: boolean;
  isLegacy: boolean;
};

// Warm up cache on server start
async function warmupCache(ctx: { db: PrismaClient }) {
  console.log(`[CACHE WARMUP] Starting cache warmup for all GPU types`);
  const startTime = Date.now();

  try {
    // Precompute for each GPU type, for legacy and new modes
    const modes: LeaderboardModeType[] = [
      LeaderboardMode.LEGACY,
      LeaderboardMode.NEW,
    ];

    await Promise.all(
      gpuTypes.flatMap((gpuType) =>
        modes.map(async (mode) => {
          console.log(
            `[CACHE WARMUP] Processing GPU: ${gpuType}, Mode: ${mode}`
          );
          const data = await computeLeaderboardData(ctx, gpuType, mode);
          leaderboardCache.set(`leaderboard-${gpuType}-${mode}`, data);
        })
      )
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
              difficulty: true,
            },
          },
          user: {
            select: {
              username: true,
            },
          },
          testResults: {
            include: {
              runs: true,
            },
            orderBy: {
              testId: "asc",
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

      // Check if user has access to code
      const hasCodeAccess =
        submission.isPublic ||
        (ctx.session?.user && ctx.session.user.id === submission.userId);

      // Return submission without code if user doesn't have access
      if (!hasCodeAccess) {
        const { code: _, ...submissionWithoutCode } = submission;
        return submissionWithoutCode;
      }

      return submission;
    }),

  getBestSubmissionsByProblem: publicProcedure
    .input(
      z.object({
        gpuType: z.string().optional().default("all"),
        mode: z.enum(["legacy", "new"]).optional().default("legacy"),
      })
    )
    .query(async ({ ctx, input }) => {
      const mode = input.mode as LeaderboardModeType;
      const cacheKey = `leaderboard-${input.gpuType}-${mode}`;

      // Check cache first
      const cachedData = leaderboardCache.get<ProblemLeaderboard[]>(cacheKey);
      if (cachedData) {
        // Log each request with its counter value
        console.log(
          `[CACHE HIT] Leaderboard cache used for GPU: ${input.gpuType}, Mode: ${mode} (Request #${requestCounter})`
        );

        requestCounter++;
        if (requestCounter % 10 === 0) {
          console.log(
            `[CACHE REFRESH] Triggering background refresh after ${requestCounter} requests for GPU: ${input.gpuType}, Mode: ${mode}`
          );
          void refreshLeaderboardCache(ctx, input.gpuType, mode, cacheKey);
        }
        return cachedData;
      }

      // Cache miss - compute data and store in cache
      console.log(
        `[CACHE MISS] Computing leaderboard data for GPU: ${input.gpuType}, Mode: ${mode}`
      );
      const data = await computeLeaderboardData(ctx, input.gpuType, mode);
      console.log(
        `[CACHE STORE] Storing new leaderboard data for GPU: ${input.gpuType}, Mode: ${mode}`
      );
      leaderboardCache.set(cacheKey, data);
      return data;
    }),

  getProblemLeaderboard: publicProcedure
    .input(
      z.object({
        slug: z.string(),
        gpuType: z.string().optional().default("all"),
        mode: z.enum(["legacy", "new"]).optional().default("legacy"),
      })
    )
    .query(async ({ ctx, input }) => {
      const mode = input.mode as LeaderboardModeType;
      const cacheKey = `problem-leaderboard-${input.slug}-${input.gpuType}-${mode}`;

      // Check cache first
      const cachedData = problemLeaderboardCache.get(cacheKey);
      if (cachedData) {
        console.log(
          `[CACHE HIT] Problem leaderboard cache used for ${input.slug}, GPU: ${input.gpuType}, Mode: ${mode} (Request #${problemRequestCounter})`
        );

        problemRequestCounter++;
        if (problemRequestCounter % 10 === 0) {
          console.log(
            `[CACHE REFRESH] Triggering background refresh for problem ${input.slug}, GPU: ${input.gpuType}, Mode: ${mode}`
          );
          void refreshProblemLeaderboardCache(
            ctx,
            input.slug,
            input.gpuType,
            mode,
            cacheKey
          );
        }
        return cachedData;
      }

      // Cache miss - compute data
      console.log(
        `[CACHE MISS] Computing problem leaderboard data for ${input.slug}, GPU: ${input.gpuType}, Mode: ${mode}`
      );
      const data = await computeProblemLeaderboardData(
        ctx,
        input.slug,
        input.gpuType,
        mode
      );
      console.log(
        `[CACHE STORE] Storing new problem leaderboard data for ${input.slug}, GPU: ${input.gpuType}, Mode: ${mode}`
      );
      problemLeaderboardCache.set(cacheKey, data);
      return data;
    }),
  getForBlogPost: protectedProcedure
    .input(
      z.object({
        submissionId: z.string(),
      })
    )
    .query(async ({ ctx, input }) => {
      const submission = await ctx.db.submission.findUnique({
        where: { id: input.submissionId },
        include: {
          problem: {
            select: {
              id: true,
              title: true,
              slug: true,
              difficulty: true,
              description: true,
            },
          },
          user: {
            select: {
              id: true,
              name: true,
              username: true,
              image: true,
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

      // Check access - user must own it OR it must be public
      const hasAccess =
        submission.isPublic ||
        (ctx.session.user && ctx.session.user.id === submission.userId);

      if (!hasAccess) {
        // Return without code if no access
        const { code: _, ...submissionWithoutCode } = submission;
        return submissionWithoutCode;
      }

      return submission;
    }),

  getMultipleForComparison: protectedProcedure
    .input(
      z.object({
        submissionIds: z.array(z.string()).min(1).max(10),
      })
    )
    .query(async ({ ctx, input }) => {
      const submissions = await ctx.db.submission.findMany({
        where: {
          id: { in: input.submissionIds },
          OR: [{ isPublic: true }, { userId: ctx.session.user.id }],
        },
        include: {
          problem: {
            select: {
              id: true,
              title: true,
              slug: true,
              difficulty: true,
            },
          },
          user: {
            select: {
              id: true,
              name: true,
              username: true,
              image: true,
            },
          },
        },
        orderBy: { createdAt: "desc" },
      });

      return submissions;
    }),

  getUserRecentAcceptedSubmissions: protectedProcedure
    .input(
      z.object({
        limit: z.number().min(1).max(50).default(10),
      })
    )
    .query(async ({ ctx, input }) => {
      const submissions = await ctx.db.submission.findMany({
        where: {
          userId: ctx.session.user.id,
          status: "ACCEPTED",
        },
        include: {
          problem: {
            select: {
              id: true,
              title: true,
              slug: true,
              difficulty: true,
            },
          },
        },
        orderBy: { createdAt: "desc" },
        take: input.limit,
      });

      return submissions;
    }),
});

async function computeLeaderboardData(
  ctx: { db: PrismaClient },
  gpuType: string,
  mode: LeaderboardModeType = LeaderboardMode.LEGACY
): Promise<ProblemLeaderboard[]> {
  // Get all problems first
  const problems = await ctx.db.problem.findMany({
    select: {
      id: true,
      slug: true,
      title: true,
    },
    orderBy: {
      createdAt: "asc",
    },
  });

  // Build results for each problem
  const results: ProblemLeaderboard[] = [];

  for (const problem of problems) {
    type SubmissionData = {
      id: string;
      gflops: number | null;
      gpuType: string | null;
      runtime: number | null;
      language: string | null;
      username: string | null;
      isLegacy: boolean;
    };

    const allSubmissions: SubmissionData[] = [];

    // Query legacy submissions if mode is 'legacy'
    // Legacy mode: GFLOPS-based ranking, sorted by GFLOPS DESC (higher is better)
    if (mode === LeaderboardMode.LEGACY) {
      const legacySubmissions = await ctx.db.legacySubmission.findMany({
        where: {
          problemId: problem.id,
          status: "ACCEPTED",
          gflops: { not: null },
          ...(gpuType !== "all" ? { gpuType } : {}),
        },
        select: {
          id: true,
          gflops: true,
          gpuType: true,
          runtime: true,
          language: true,
          user: {
            select: {
              username: true,
            },
          },
        },
      });

      allSubmissions.push(
        ...legacySubmissions.map((s) => ({
          id: s.id,
          gflops: s.gflops,
          gpuType: s.gpuType,
          runtime: s.runtime,
          language: s.language,
          username: s.user.username,
          isLegacy: true,
        }))
      );
    }

    // Query new submissions if mode is 'new'
    // New mode: Runtime-based ranking, sorted by runtime ASC (lower is better)
    if (mode === LeaderboardMode.NEW) {
      // For now, query LegacySubmission since new Submission table is empty
      // This shows runtime-based rankings from legacy data
      const runtimeSubmissions = await ctx.db.legacySubmission.findMany({
        where: {
          problemId: problem.id,
          status: "ACCEPTED",
          runtime: { not: null },
          ...(gpuType !== "all" ? { gpuType } : {}),
        },
        select: {
          id: true,
          gflops: true,
          gpuType: true,
          runtime: true,
          language: true,
          user: {
            select: {
              username: true,
            },
          },
        },
      });

      allSubmissions.push(
        ...runtimeSubmissions.map((s) => ({
          id: s.id,
          gflops: s.gflops,
          gpuType: s.gpuType,
          runtime: s.runtime,
          language: s.language,
          username: s.user.username,
          isLegacy: false, // Shown as "new" (runtime-based) even though from legacy table
        }))
      );
    }

    // Get best submission per user-GPU combination
    const userBestMap = new Map<string, SubmissionData>();

    for (const submission of allSubmissions) {
      const key = `${submission.username ?? "Anonymous"}-${submission.gpuType}`;
      const current = userBestMap.get(key);

      if (!current) {
        userBestMap.set(key, submission);
      } else if (mode === LeaderboardMode.LEGACY) {
        // Legacy mode: higher GFLOPS is better
        if (
          submission.gflops !== null &&
          (current.gflops === null || submission.gflops > current.gflops)
        ) {
          userBestMap.set(key, submission);
        }
      } else {
        // New mode: lower runtime is better
        if (
          submission.runtime !== null &&
          (current.runtime === null || submission.runtime < current.runtime)
        ) {
          userBestMap.set(key, submission);
        }
      }
    }

    // Sort based on mode
    const topSubmissions = Array.from(userBestMap.values())
      .sort((a, b) => {
        if (mode === LeaderboardMode.LEGACY) {
          // Legacy: sort by GFLOPS DESC (higher is better)
          if (a.gflops !== null && b.gflops !== null) {
            return b.gflops - a.gflops;
          }
          if (a.gflops !== null) return -1;
          if (b.gflops !== null) return 1;
          return 0;
        } else {
          // New: sort by runtime ASC (lower is better)
          if (a.runtime !== null && b.runtime !== null) {
            return a.runtime - b.runtime;
          }
          if (a.runtime !== null) return -1;
          if (b.runtime !== null) return 1;
          return 0;
        }
      })
      .slice(0, 3)
      .map((sub) => ({
        id: sub.id,
        gflops: sub.gflops,
        gpuType: sub.gpuType,
        language: sub.language,
        username: sub.username,
        runtime: sub.runtime,
        isLegacy: sub.isLegacy,
      }));

    results.push({
      slug: problem.slug,
      title: problem.title,
      id: problem.id,
      topSubmissions,
    });
  }

  return results;
}

// Function to refresh cache in background
async function refreshLeaderboardCache(
  ctx: { db: PrismaClient },
  gpuType: string,
  mode: LeaderboardModeType,
  cacheKey: string
) {
  try {
    console.log(
      `[CACHE REFRESH START] Beginning refresh for GPU: ${gpuType}, Mode: ${mode}`
    );
    const startTime = Date.now();

    const freshData = await computeLeaderboardData(ctx, gpuType, mode);
    const duration = Date.now() - startTime;

    console.log(
      `[CACHE REFRESH COMPLETE] Refreshed cache for GPU: ${gpuType}, Mode: ${mode} (took ${duration}ms)`
    );
    leaderboardCache.set(cacheKey, freshData);
  } catch (error) {
    console.error(
      `[CACHE REFRESH ERROR] Failed to refresh cache for GPU: ${gpuType}, Mode: ${mode}`,
      error
    );
  }
}

// Function to compute problem leaderboard data
async function computeProblemLeaderboardData(
  ctx: { db: PrismaClient },
  slug: string,
  gpuType: string,
  mode: LeaderboardModeType = LeaderboardMode.LEGACY
): Promise<ProblemLeaderboardEntry[]> {
  // Get problem ID first
  const problem = await ctx.db.problem.findUnique({
    where: { slug },
    select: { id: true, title: true },
  });

  if (!problem) return [];

  type SubmissionData = {
    id: string;
    gflops: number | null;
    runtime: number | null;
    gpuType: string | null;
    language: string;
    createdAt: Date;
    isPublic: boolean;
    username: string | null;
    isLegacy: boolean;
  };

  const allSubmissions: SubmissionData[] = [];

  // Query legacy submissions if mode is 'legacy'
  // Legacy mode: GFLOPS-based ranking, sorted by GFLOPS DESC (higher is better)
  if (mode === LeaderboardMode.LEGACY) {
    const legacySubmissions = await ctx.db.legacySubmission.findMany({
      where: {
        problem: { slug },
        status: "ACCEPTED",
        gflops: { not: null },
        ...(gpuType !== "all" ? { gpuType } : {}),
      },
      select: {
        id: true,
        gflops: true,
        runtime: true,
        gpuType: true,
        language: true,
        createdAt: true,
        isPublic: true,
        user: {
          select: {
            username: true,
          },
        },
      },
    });

    allSubmissions.push(
      ...legacySubmissions.map((s) => ({
        id: s.id,
        gflops: s.gflops,
        runtime: s.runtime,
        gpuType: s.gpuType,
        language: s.language,
        createdAt: s.createdAt,
        isPublic: s.isPublic,
        username: s.user.username,
        isLegacy: true,
      }))
    );
  }

  // Query for runtime-based mode
  // New mode: Runtime-based ranking, sorted by runtime ASC (lower is better)
  if (mode === LeaderboardMode.NEW) {
    // For now, query LegacySubmission since new Submission table is empty
    const runtimeSubmissions = await ctx.db.legacySubmission.findMany({
      where: {
        problem: { slug },
        status: "ACCEPTED",
        runtime: { not: null },
        ...(gpuType !== "all" ? { gpuType } : {}),
      },
      select: {
        id: true,
        gflops: true,
        runtime: true,
        gpuType: true,
        language: true,
        createdAt: true,
        isPublic: true,
        user: {
          select: {
            username: true,
          },
        },
      },
    });

    allSubmissions.push(
      ...runtimeSubmissions.map((s) => ({
        id: s.id,
        gflops: s.gflops,
        runtime: s.runtime,
        gpuType: s.gpuType,
        language: s.language,
        createdAt: s.createdAt,
        isPublic: s.isPublic,
        username: s.user.username,
        isLegacy: false, // Shown as "new" (runtime-based) even though from legacy table
      }))
    );
  }

  // Calculate best submission per user-GPU combination
  const userGpuBestMap = new Map<string, SubmissionData>();

  for (const submission of allSubmissions) {
    const userGpuKey = `${submission.username ?? "Anonymous"}-${submission.gpuType}`;
    const currentBest = userGpuBestMap.get(userGpuKey);

    if (!currentBest) {
      userGpuBestMap.set(userGpuKey, submission);
    } else if (mode === LeaderboardMode.LEGACY) {
      // Legacy mode: higher GFLOPS is better
      if (
        submission.gflops !== null &&
        (currentBest.gflops === null || submission.gflops > currentBest.gflops)
      ) {
        userGpuBestMap.set(userGpuKey, submission);
      }
    } else {
      // New mode: lower runtime is better
      if (
        submission.runtime !== null &&
        (currentBest.runtime === null ||
          submission.runtime < currentBest.runtime)
      ) {
        userGpuBestMap.set(userGpuKey, submission);
      }
    }
  }

  // Sort based on mode
  return Array.from(userGpuBestMap.values())
    .sort((a, b) => {
      if (mode === LeaderboardMode.LEGACY) {
        // Legacy: sort by GFLOPS DESC (higher is better)
        if (a.gflops !== null && b.gflops !== null) {
          return b.gflops - a.gflops;
        }
        if (a.gflops !== null) return -1;
        if (b.gflops !== null) return 1;
        return 0;
      } else {
        // New: sort by runtime ASC (lower is better)
        if (a.runtime !== null && b.runtime !== null) {
          return a.runtime - b.runtime;
        }
        if (a.runtime !== null) return -1;
        if (b.runtime !== null) return 1;
        return 0;
      }
    })
    .map((sub) => ({
      id: sub.id,
      username: sub.username,
      gflops: sub.gflops,
      runtime: sub.runtime,
      createdAt: sub.createdAt,
      gpuType: sub.gpuType,
      language: sub.language,
      isPublic: sub.isPublic,
      isLegacy: sub.isLegacy,
    }));
}

// Function to refresh problem cache in background
async function refreshProblemLeaderboardCache(
  ctx: { db: PrismaClient },
  slug: string,
  gpuType: string,
  mode: LeaderboardModeType,
  cacheKey: string
) {
  try {
    console.log(
      `[CACHE REFRESH START] Beginning refresh for problem ${slug}, GPU: ${gpuType}, Mode: ${mode}`
    );
    const startTime = Date.now();

    const freshData = await computeProblemLeaderboardData(
      ctx,
      slug,
      gpuType,
      mode
    );

    const duration = Date.now() - startTime;
    console.log(
      `[CACHE REFRESH COMPLETE] Refreshed cache for problem ${slug}, GPU: ${gpuType}, Mode: ${mode} (took ${duration}ms)`
    );

    problemLeaderboardCache.set(cacheKey, freshData);
  } catch (error) {
    console.error(
      `[CACHE REFRESH ERROR] Failed to refresh cache for problem ${slug}, GPU: ${gpuType}, Mode: ${mode}`,
      error
    );
  }
}
