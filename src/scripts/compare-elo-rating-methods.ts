/**
 * Elo Rating Comparison Script: GFLOPS vs Runtime
 *
 * This script compares the current GFLOPS-based Elo rating system
 * against the proposed runtime-based system to understand impact.
 *
 * Usage:
 *   npx tsx src/scripts/compare-elo-rating-methods.ts
 */

import { PrismaClient } from "@prisma/client";

const prisma = new PrismaClient();

// Constants (matching production values from src/constants/problem.ts)
const PROBLEM_DIFFICULTY_MULTIPLIERS = {
  EASY: 1,
  MEDIUM: 1.5,
  HARD: 3,
} as const;

const FIRST_SOLVE_BONUS = 15;
const ADJUSTMENT_FACTOR = 32;
const START_RATING = 1000;

interface UserRatingComparison {
  userId: string;
  username: string;
  gflopsRating: number | null;
  runtimeRating: number | null;
  change: number | null;
  gflopsProblems: number;
  runtimeProblems: number;
}

interface RankingComparison {
  username: string;
  gflopsRank: number;
  runtimeRank: number;
  rankChange: number;
  gflopsRating: number;
  runtimeRating: number;
}

/**
 * Calculate performance score from percentile (shared between both methods)
 */
function calculatePerformanceScore(
  percentile: number,
  difficultyMultiplier: number
): number {
  return (200 - 150 * Math.pow(percentile, 0.7)) * difficultyMultiplier;
}

/**
 * Calculate GFLOPS-based rating (current system)
 */
async function calculateGflopsRating(userId: string): Promise<{
  rating: number;
  problemsProcessed: number;
}> {
  let rating = START_RATING;
  let totalRatingChanges = 0;
  let problemsProcessed = 0;

  // Get user's best submissions for each problem-GPU combination
  const userBestSubmissions = await prisma.submission.groupBy({
    by: ["problemId", "gpuType"],
    _max: { gflops: true },
    where: {
      userId: userId,
      status: "ACCEPTED",
      gflops: { not: null },
    },
  });

  for (const submission of userBestSubmissions) {
    const gpuType = submission.gpuType;
    const gflops = submission._max.gflops;
    const problemId = submission.problemId;

    if (!gpuType || !gflops) continue;

    // Get problem difficulty
    const problem = await prisma.problem.findUnique({
      where: { id: problemId },
      select: { difficulty: true },
    });

    if (!problem?.difficulty) continue;

    const difficultyMultiplier =
      PROBLEM_DIFFICULTY_MULTIPLIERS[
        problem.difficulty as keyof typeof PROBLEM_DIFFICULTY_MULTIPLIERS
      ];

    // Find all submissions for this problem-GPU combination
    const allSubmissions = await prisma.submission.findMany({
      where: {
        problemId: problemId,
        gpuType: gpuType,
        status: "ACCEPTED",
        gflops: { not: null },
      },
      select: {
        userId: true,
        gflops: true,
      },
    });

    // Group by user, keeping only best (highest) GFLOPS per user
    const userBestGflops: Record<string, number> = {};
    for (const sub of allSubmissions) {
      if (sub.gflops === null || sub.userId === null) continue;
      if (
        !userBestGflops[sub.userId] ||
        userBestGflops[sub.userId]! < sub.gflops
      ) {
        userBestGflops[sub.userId] = sub.gflops;
      }
    }

    // Sort descending (higher GFLOPS = better)
    const sortedGflops = Object.values(userBestGflops).sort((a, b) => b - a);

    const userRank = sortedGflops.findIndex((value) => value === gflops) + 1;
    const totalUsers = sortedGflops.length;

    if (totalUsers <= 1) {
      totalRatingChanges += FIRST_SOLVE_BONUS;
      problemsProcessed++;
      continue;
    }

    const percentile = (userRank - 1) / Math.max(totalUsers - 1, 1);
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

  return {
    rating: Math.round(rating),
    problemsProcessed,
  };
}

/**
 * Calculate runtime-based rating (proposed system)
 */
async function calculateRuntimeRating(userId: string): Promise<{
  rating: number;
  problemsProcessed: number;
}> {
  let rating = START_RATING;
  let totalRatingChanges = 0;
  let problemsProcessed = 0;

  // Get user's best (fastest) submissions for each problem-GPU combination
  const userBestSubmissions = await prisma.submission.groupBy({
    by: ["problemId", "gpuType"],
    _min: { runtime: true }, // Changed: MIN runtime instead of MAX gflops
    where: {
      userId: userId,
      status: "ACCEPTED",
      runtime: { not: null },
    },
  });

  for (const submission of userBestSubmissions) {
    const gpuType = submission.gpuType;
    const runtime = submission._min.runtime;
    const problemId = submission.problemId;

    if (!gpuType || !runtime) continue;

    // Get problem difficulty
    const problem = await prisma.problem.findUnique({
      where: { id: problemId },
      select: { difficulty: true },
    });

    if (!problem?.difficulty) continue;

    const difficultyMultiplier =
      PROBLEM_DIFFICULTY_MULTIPLIERS[
        problem.difficulty as keyof typeof PROBLEM_DIFFICULTY_MULTIPLIERS
      ];

    // Find all submissions for this problem-GPU combination
    const allSubmissions = await prisma.submission.findMany({
      where: {
        problemId: problemId,
        gpuType: gpuType,
        status: "ACCEPTED",
        runtime: { not: null },
      },
      select: {
        userId: true,
        runtime: true,
      },
    });

    // Group by user, keeping only best (lowest) runtime per user
    const userBestRuntimes: Record<string, number> = {};
    for (const sub of allSubmissions) {
      if (sub.runtime === null || sub.userId === null) continue;
      if (
        !userBestRuntimes[sub.userId] ||
        userBestRuntimes[sub.userId]! > sub.runtime
      ) {
        userBestRuntimes[sub.userId] = sub.runtime;
      }
    }

    // Sort ascending (lower runtime = better)
    const sortedRuntimes = Object.values(userBestRuntimes).sort(
      (a, b) => a - b
    );

    const userRank = sortedRuntimes.findIndex((value) => value === runtime) + 1;
    const totalUsers = sortedRuntimes.length;

    if (totalUsers <= 1) {
      totalRatingChanges += FIRST_SOLVE_BONUS;
      problemsProcessed++;
      continue;
    }

    const percentile = (userRank - 1) / Math.max(totalUsers - 1, 1);
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

  return {
    rating: Math.round(rating),
    problemsProcessed,
  };
}

/**
 * Analyze data coverage and identify gaps
 */
async function analyzeDataCoverage() {
  const submissionStats = await prisma.submission.aggregate({
    where: { status: "ACCEPTED" },
    _count: { id: true },
  });

  const withGflopsOnly = await prisma.submission.count({
    where: {
      status: "ACCEPTED",
      gflops: { not: null },
      runtime: null,
    },
  });

  const withRuntimeOnly = await prisma.submission.count({
    where: {
      status: "ACCEPTED",
      runtime: { not: null },
      gflops: null,
    },
  });

  const withBoth = await prisma.submission.count({
    where: {
      status: "ACCEPTED",
      gflops: { not: null },
      runtime: { not: null },
    },
  });

  const withNeither = await prisma.submission.count({
    where: {
      status: "ACCEPTED",
      gflops: null,
      runtime: null,
    },
  });

  return {
    total: submissionStats._count.id,
    withGflopsOnly,
    withRuntimeOnly,
    withBoth,
    withNeither,
  };
}

/**
 * Main comparison function
 */
async function compareRatingSystems() {
  console.log("=".repeat(60));
  console.log("Elo Rating Comparison: GFLOPS vs Runtime");
  console.log("=".repeat(60));
  console.log("");

  // 1. Analyze data coverage
  console.log("DATA COVERAGE");
  console.log("-".repeat(60));
  const coverage = await analyzeDataCoverage();
  console.log(`Total accepted submissions:        ${coverage.total}`);
  console.log(`  - With GFLOPS only:              ${coverage.withGflopsOnly}`);
  console.log(`  - With Runtime only:             ${coverage.withRuntimeOnly}`);
  console.log(`  - With both:                     ${coverage.withBoth}`);
  console.log(`  - With neither:                  ${coverage.withNeither}`);
  console.log("");

  // 2. Get all users with accepted submissions
  const users = await prisma.user.findMany({
    where: {
      submissions: {
        some: {
          status: "ACCEPTED",
        },
      },
    },
    select: {
      id: true,
      username: true,
    },
    orderBy: { username: "asc" },
  });

  console.log(`Analyzing ratings for ${users.length} users...`);
  console.log("");

  // 3. Calculate ratings for each user
  const comparisons: UserRatingComparison[] = [];
  let usersWithGflops = 0;
  let usersWithRuntime = 0;
  let usersWithBoth = 0;

  for (const user of users) {
    const gflopsResult = await calculateGflopsRating(user.id);
    const runtimeResult = await calculateRuntimeRating(user.id);

    const hasGflops = gflopsResult.problemsProcessed > 0;
    const hasRuntime = runtimeResult.problemsProcessed > 0;

    if (hasGflops) usersWithGflops++;
    if (hasRuntime) usersWithRuntime++;
    if (hasGflops && hasRuntime) usersWithBoth++;

    const gflopsRating = hasGflops ? gflopsResult.rating : null;
    const runtimeRating = hasRuntime ? runtimeResult.rating : null;
    const change =
      gflopsRating !== null && runtimeRating !== null
        ? runtimeRating - gflopsRating
        : null;

    comparisons.push({
      userId: user.id,
      username: user.username ?? "unknown",
      gflopsRating,
      runtimeRating,
      change,
      gflopsProblems: gflopsResult.problemsProcessed,
      runtimeProblems: runtimeResult.problemsProcessed,
    });
  }

  // 4. Print statistics
  console.log("RATING COMPARISON");
  console.log("-".repeat(60));
  console.log(`Users with GFLOPS data:            ${usersWithGflops}`);
  console.log(`Users with Runtime data:           ${usersWithRuntime}`);
  console.log(`Users with both:                   ${usersWithBoth}`);
  console.log(
    `Users excluded (no runtime data):  ${usersWithGflops - usersWithBoth}`
  );
  console.log("");

  // Filter to users with both ratings
  const validComparisons = comparisons.filter(
    (c) => c.gflopsRating !== null && c.runtimeRating !== null
  );

  if (validComparisons.length === 0) {
    console.log("No users with both GFLOPS and runtime data to compare.");
    return;
  }

  const changes = validComparisons.map((c) => c.change!);
  const increases = changes.filter((c) => c > 5).length;
  const decreases = changes.filter((c) => c < -5).length;
  const unchanged = changes.filter((c) => Math.abs(c) <= 5).length;

  const avgChange = changes.reduce((a, b) => a + b, 0) / changes.length;
  const medianChange = [...changes].sort((a, b) => a - b)[
    Math.floor(changes.length / 2)
  ]!;
  const maxIncrease = Math.max(...changes);
  const maxDecrease = Math.min(...changes);

  console.log("Statistics:");
  console.log(
    `  - Users with higher runtime rating:  ${increases} (${((increases / validComparisons.length) * 100).toFixed(1)}%)`
  );
  console.log(
    `  - Users with lower runtime rating:   ${decreases} (${((decreases / validComparisons.length) * 100).toFixed(1)}%)`
  );
  console.log(
    `  - Users unchanged (±5):              ${unchanged} (${((unchanged / validComparisons.length) * 100).toFixed(1)}%)`
  );
  console.log("");
  console.log(`  - Average change:                    ${avgChange.toFixed(1)}`);
  console.log(
    `  - Median change:                     ${medianChange.toFixed(1)}`
  );
  console.log(
    `  - Max increase:                      +${maxIncrease.toFixed(1)}`
  );
  console.log(
    `  - Max decrease:                      ${maxDecrease.toFixed(1)}`
  );
  console.log("");

  // 5. Show top gainers
  console.log("TOP 10 GAINERS (Runtime vs GFLOPS)");
  console.log("-".repeat(60));
  const topGainers = [...validComparisons]
    .sort((a, b) => b.change! - a.change!)
    .slice(0, 10);

  console.log(
    "Username".padEnd(20) +
      "GFLOPS".padEnd(10) +
      "Runtime".padEnd(10) +
      "Change"
  );
  console.log("-".repeat(60));
  for (const user of topGainers) {
    console.log(
      user.username.padEnd(20) +
        user.gflopsRating!.toString().padEnd(10) +
        user.runtimeRating!.toString().padEnd(10) +
        `+${user.change!.toFixed(1)}`
    );
  }
  console.log("");

  // 6. Show top losers
  console.log("TOP 10 LOSERS (Runtime vs GFLOPS)");
  console.log("-".repeat(60));
  const topLosers = [...validComparisons]
    .sort((a, b) => a.change! - b.change!)
    .slice(0, 10);

  console.log(
    "Username".padEnd(20) +
      "GFLOPS".padEnd(10) +
      "Runtime".padEnd(10) +
      "Change"
  );
  console.log("-".repeat(60));
  for (const user of topLosers) {
    console.log(
      user.username.padEnd(20) +
        user.gflopsRating!.toString().padEnd(10) +
        user.runtimeRating!.toString().padEnd(10) +
        user.change!.toFixed(1)
    );
  }
  console.log("");

  // 7. Ranking changes for top users
  console.log("RANKING CHANGES (Top 20 by GFLOPS rating)");
  console.log("-".repeat(60));

  // Sort by GFLOPS rating and assign ranks
  const gflopsSorted = [...validComparisons].sort(
    (a, b) => b.gflopsRating! - a.gflopsRating!
  );
  const runtimeSorted = [...validComparisons].sort(
    (a, b) => b.runtimeRating! - a.runtimeRating!
  );

  const gflopsRankMap = new Map(
    gflopsSorted.map((user, index) => [user.userId, index + 1])
  );
  const runtimeRankMap = new Map(
    runtimeSorted.map((user, index) => [user.userId, index + 1])
  );

  const rankingComparisons: RankingComparison[] = gflopsSorted
    .slice(0, 20)
    .map((user) => ({
      username: user.username,
      gflopsRank: gflopsRankMap.get(user.userId)!,
      runtimeRank: runtimeRankMap.get(user.userId)!,
      rankChange:
        gflopsRankMap.get(user.userId)! - runtimeRankMap.get(user.userId)!,
      gflopsRating: user.gflopsRating!,
      runtimeRating: user.runtimeRating!,
    }));

  console.log(
    "Username".padEnd(20) +
      "GFLOPS Rank".padEnd(13) +
      "Runtime Rank".padEnd(14) +
      "Change"
  );
  console.log("-".repeat(60));
  for (const comp of rankingComparisons) {
    const changeStr =
      comp.rankChange > 0
        ? `↓${comp.rankChange}`
        : comp.rankChange < 0
          ? `↑${Math.abs(comp.rankChange)}`
          : "—";
    console.log(
      comp.username.padEnd(20) +
        comp.gflopsRank.toString().padEnd(13) +
        comp.runtimeRank.toString().padEnd(14) +
        changeStr
    );
  }
  console.log("");

  // 8. Users with missing runtime data
  const usersWithMissingRuntime = comparisons.filter(
    (c) => c.gflopsRating !== null && c.runtimeRating === null
  );

  if (usersWithMissingRuntime.length > 0) {
    console.log("USERS WITH INCOMPLETE RUNTIME DATA");
    console.log("-".repeat(60));
    console.log(
      `${usersWithMissingRuntime.length} users have GFLOPS submissions but no runtime data:`
    );
    for (const user of usersWithMissingRuntime.slice(0, 10)) {
      console.log(
        `  - ${user.username}: ${user.gflopsProblems} problem(s) with GFLOPS, 0 with runtime`
      );
    }
    if (usersWithMissingRuntime.length > 10) {
      console.log(`  ... and ${usersWithMissingRuntime.length - 10} more`);
    }
    console.log("");
  }

  console.log("=".repeat(60));
  console.log("COMPARISON COMPLETE");
  console.log("=".repeat(60));
}

async function main() {
  try {
    await compareRatingSystems();
  } catch (error) {
    console.error("Comparison failed:", error);
    process.exit(1);
  } finally {
    await prisma.$disconnect();
  }
}

main();
