/**
 * Metric Variance Analysis Script: GFLOPS vs Runtime
 *
 * This script analyzes the coefficient of variation and percentile consistency
 * for GFLOPS vs Runtime metrics to determine which is more stable for ranking.
 *
 * Usage:
 *   bun tsx src/scripts/analyze-metric-variance.ts
 */

// @ts-expect-error - Prisma types may not be generated yet
import { PrismaClient } from "@prisma/client";

// @ts-expect-error - Prisma client instantiation
const prisma = new PrismaClient();

const GPU_TYPE = "H100"; // Standard GPU for analysis

interface ProblemStats {
  problemId: string;
  problemSlug: string;
  submissionCount: number;
  gflopsCV: number;
  runtimeCV: number;
  gflopsRange: { min: number; max: number };
  runtimeRange: { min: number; max: number };
}

interface UserPercentile {
  userId: string;
  username: string;
  problemId: string;
  problemSlug: string;
  gflopsPercentile: number;
  runtimePercentile: number;
  gflops: number;
  runtime: number;
  gflopsRank: number;
  runtimeRank: number;
  totalUsers: number;
}

/**
 * Calculate coefficient of variation (std dev / mean)
 */
function coefficientOfVariation(values: number[]): number {
  if (values.length === 0) return 0;

  const mean = values.reduce((a, b) => a + b, 0) / values.length;
  if (mean === 0) return 0;

  const variance =
    values.reduce((sum, v) => sum + Math.pow(v - mean, 2), 0) / values.length;
  const stdDev = Math.sqrt(variance);

  return stdDev / mean;
}

/**
 * Calculate standard deviation
 */
function stdDev(values: number[]): number {
  if (values.length === 0) return 0;

  const mean = values.reduce((a, b) => a + b, 0) / values.length;
  const variance =
    values.reduce((sum, v) => sum + Math.pow(v - mean, 2), 0) / values.length;

  return Math.sqrt(variance);
}

/**
 * Calculate percentile (0 = best, 1 = worst)
 */
function calculatePercentile(rank: number, total: number): number {
  if (total <= 1) return 0;
  return (rank - 1) / (total - 1);
}

/**
 * Analyze per-problem statistics
 */
async function analyzePerProblemStats(): Promise<ProblemStats[]> {
  // Get all problems with 10+ submissions on H100
  const problems = await prisma.problem.findMany({
    where: {
      submissions: {
        some: {
          status: "ACCEPTED",
          gpuType: GPU_TYPE,
          gflops: { not: null },
          runtime: { not: null },
        },
      },
    },
    select: {
      id: true,
      slug: true,
      title: true,
    },
  });

  const problemStats: ProblemStats[] = [];

  for (const problem of problems) {
    // Get all accepted submissions for this problem on H100
    const submissions = await prisma.submission.findMany({
      where: {
        problemId: problem.id,
        gpuType: GPU_TYPE,
        status: "ACCEPTED",
        gflops: { not: null },
        runtime: { not: null },
      },
      select: {
        userId: true,
        gflops: true,
        runtime: true,
      },
    });

    // Filter to 10+ submissions
    if (submissions.length < 10) continue;

    // Get best submission per user
    const userBestMap = new Map<string, { gflops: number; runtime: number }>();

    for (const sub of submissions) {
      if (!sub.userId || sub.gflops === null || sub.runtime === null) continue;

      const current = userBestMap.get(sub.userId);
      if (!current) {
        userBestMap.set(sub.userId, {
          gflops: sub.gflops,
          runtime: sub.runtime,
        });
      } else {
        // Keep best GFLOPS (highest) and best runtime (lowest)
        // Note: These might not be from the same submission, but that's okay
        // We're measuring the variability of the metrics themselves
        if (sub.gflops > current.gflops) {
          current.gflops = sub.gflops;
        }
        if (sub.runtime < current.runtime) {
          current.runtime = sub.runtime;
        }
      }
    }

    const userBests = Array.from(userBestMap.values());
    const gflopsValues = userBests.map((u) => u.gflops);
    const runtimeValues = userBests.map((u) => u.runtime);

    problemStats.push({
      problemId: problem.id,
      problemSlug: problem.slug,
      submissionCount: userBests.length,
      gflopsCV: coefficientOfVariation(gflopsValues),
      runtimeCV: coefficientOfVariation(runtimeValues),
      gflopsRange: {
        min: Math.min(...gflopsValues),
        max: Math.max(...gflopsValues),
      },
      runtimeRange: {
        min: Math.min(...runtimeValues),
        max: Math.max(...runtimeValues),
      },
    });
  }

  return problemStats;
}

/**
 * Analyze cross-problem percentile consistency for users with 3+ problems
 */
async function analyzeCrossUserConsistency(): Promise<{
  userPercentiles: Map<string, UserPercentile[]>;
  gflopsStdDevs: number[];
  runtimeStdDevs: number[];
}> {
  // Get users who solved 3+ problems on H100
  const users = await prisma.user.findMany({
    where: {
      submissions: {
        some: {
          status: "ACCEPTED",
          gpuType: GPU_TYPE,
          gflops: { not: null },
          runtime: { not: null },
        },
      },
    },
    select: {
      id: true,
      username: true,
    },
  });

  const userPercentiles = new Map<string, UserPercentile[]>();
  const gflopsStdDevs: number[] = [];
  const runtimeStdDevs: number[] = [];

  for (const user of users) {
    // Get user's best submissions per problem on H100
    const userSubmissions = await prisma.submission.groupBy({
      by: ["problemId"],
      _max: { gflops: true },
      _min: { runtime: true },
      where: {
        userId: user.id,
        gpuType: GPU_TYPE,
        status: "ACCEPTED",
        gflops: { not: null },
        runtime: { not: null },
      },
    });

    if (userSubmissions.length < 3) continue;

    const percentiles: UserPercentile[] = [];

    for (const submission of userSubmissions) {
      const problemId = submission.problemId;
      const userGflops = submission._max.gflops;
      const userRuntime = submission._min.runtime;

      if (!userGflops || !userRuntime) continue;

      // Get problem slug
      const problem = await prisma.problem.findUnique({
        where: { id: problemId },
        select: { slug: true },
      });

      if (!problem) continue;

      // Get all submissions for this problem to calculate percentiles
      const allSubmissions = await prisma.submission.findMany({
        where: {
          problemId: problemId,
          gpuType: GPU_TYPE,
          status: "ACCEPTED",
          gflops: { not: null },
          runtime: { not: null },
        },
        select: {
          userId: true,
          gflops: true,
          runtime: true,
        },
      });

      // Group by user, keeping best per user
      const userBestGflops = new Map<string, number>();
      const userBestRuntimes = new Map<string, number>();

      for (const sub of allSubmissions) {
        if (!sub.userId || sub.gflops === null || sub.runtime === null)
          continue;

        const currentGflops = userBestGflops.get(sub.userId);
        if (!currentGflops || sub.gflops > currentGflops) {
          userBestGflops.set(sub.userId, sub.gflops);
        }

        const currentRuntime = userBestRuntimes.get(sub.userId);
        if (!currentRuntime || sub.runtime < currentRuntime) {
          userBestRuntimes.set(sub.userId, sub.runtime);
        }
      }

      // Sort to get ranks
      const sortedGflops = Array.from(userBestGflops.values()).sort(
        (a, b) => b - a
      );
      const sortedRuntimes = Array.from(userBestRuntimes.values()).sort(
        (a, b) => a - b
      );

      const gflopsRank = sortedGflops.indexOf(userGflops) + 1;
      const runtimeRank = sortedRuntimes.indexOf(userRuntime) + 1;
      const totalUsers = sortedGflops.length;

      percentiles.push({
        userId: user.id,
        username: user.username ?? "unknown",
        problemId,
        problemSlug: problem.slug,
        gflopsPercentile: calculatePercentile(gflopsRank, totalUsers),
        runtimePercentile: calculatePercentile(runtimeRank, totalUsers),
        gflops: userGflops,
        runtime: userRuntime,
        gflopsRank,
        runtimeRank,
        totalUsers,
      });
    }

    if (percentiles.length >= 3) {
      userPercentiles.set(user.id, percentiles);

      // Calculate std dev of percentiles across problems
      const gflopsPercentileValues = percentiles.map((p) => p.gflopsPercentile);
      const runtimePercentileValues = percentiles.map(
        (p) => p.runtimePercentile
      );

      gflopsStdDevs.push(stdDev(gflopsPercentileValues));
      runtimeStdDevs.push(stdDev(runtimePercentileValues));
    }
  }

  return { userPercentiles, gflopsStdDevs, runtimeStdDevs };
}

/**
 * Deep dive analysis for specific users
 */
async function analyzeOutlierUser(
  username: string,
  userPercentiles: Map<string, UserPercentile[]>
): Promise<void> {
  // Find user
  const user = await prisma.user.findFirst({
    where: { username: { equals: username, mode: "insensitive" } },
    select: { id: true, username: true },
  });

  if (!user) {
    console.log(`User ${username} not found.`);
    return;
  }

  const percentiles = userPercentiles.get(user.id);
  if (!percentiles || percentiles.length === 0) {
    console.log(
      `No percentile data for ${username} (needs 3+ problems on H100).`
    );
    return;
  }

  console.log(`${user.username ?? username}:`);
  console.log(
    "Problem".padEnd(25) +
      "GFLOPS %ile".padEnd(13) +
      "Runtime %ile".padEnd(14) +
      "GFLOPS".padEnd(12) +
      "Runtime"
  );
  console.log("-".repeat(80));

  for (const p of percentiles) {
    const gflopsPercentileStr = `${(p.gflopsPercentile * 100).toFixed(1)}%`;
    const runtimePercentileStr = `${(p.runtimePercentile * 100).toFixed(1)}%`;
    const gflopsStr = p.gflops.toFixed(1);
    const runtimeStr = `${p.runtime.toFixed(2)}ms`;

    console.log(
      p.problemSlug.padEnd(25) +
        gflopsPercentileStr.padEnd(13) +
        runtimePercentileStr.padEnd(14) +
        gflopsStr.padEnd(12) +
        runtimeStr
    );
  }

  // Calculate avg percentiles
  const avgGflopsPercentile =
    percentiles.reduce((sum, p) => sum + p.gflopsPercentile, 0) /
    percentiles.length;
  const avgRuntimePercentile =
    percentiles.reduce((sum, p) => sum + p.runtimePercentile, 0) /
    percentiles.length;

  const gflopsStdDev = stdDev(percentiles.map((p) => p.gflopsPercentile));
  const runtimeStdDev = stdDev(percentiles.map((p) => p.runtimePercentile));

  console.log("");
  console.log(
    `Average GFLOPS percentile:  ${(avgGflopsPercentile * 100).toFixed(1)}% (std dev: ${(gflopsStdDev * 100).toFixed(1)}%)`
  );
  console.log(
    `Average Runtime percentile: ${(avgRuntimePercentile * 100).toFixed(1)}% (std dev: ${(runtimeStdDev * 100).toFixed(1)}%)`
  );
  console.log("");

  // Analysis
  if (avgRuntimePercentile < avgGflopsPercentile) {
    const diff = ((avgGflopsPercentile - avgRuntimePercentile) * 100).toFixed(
      1
    );
    console.log(
      `Analysis: Runtime percentiles are ${diff}% better on average than GFLOPS.`
    );
    console.log(
      `This explains the positive rating change when switching to runtime-based ranking.`
    );
  } else if (avgRuntimePercentile > avgGflopsPercentile) {
    const diff = ((avgRuntimePercentile - avgGflopsPercentile) * 100).toFixed(
      1
    );
    console.log(
      `Analysis: GFLOPS percentiles are ${diff}% better on average than Runtime.`
    );
    console.log(
      `This explains the negative rating change when switching to runtime-based ranking.`
    );
  } else {
    console.log(`Analysis: Both metrics show similar performance.`);
  }

  console.log("");
}

/**
 * Main analysis function
 */
async function main() {
  console.log("=".repeat(60));
  console.log("Metric Variance Analysis: GFLOPS vs Runtime");
  console.log("=".repeat(60));
  console.log(`GPU Type: ${GPU_TYPE}`);
  console.log(`Minimum submissions per problem: 10`);
  console.log(`Minimum problems per user: 3`);
  console.log("");

  // 1. Per-problem statistics
  console.log("Analyzing per-problem statistics...");
  const problemStats = await analyzePerProblemStats();

  if (problemStats.length === 0) {
    console.log("No problems found with 10+ submissions.");
    return;
  }

  const avgGflopsCV =
    problemStats.reduce((sum, p) => sum + p.gflopsCV, 0) / problemStats.length;
  const avgRuntimeCV =
    problemStats.reduce((sum, p) => sum + p.runtimeCV, 0) / problemStats.length;
  const ratio = avgGflopsCV / avgRuntimeCV;

  console.log("");
  console.log("SUMMARY");
  console.log("-".repeat(60));
  console.log(
    `Average GFLOPS CV across problems:    ${avgGflopsCV.toFixed(2)}`
  );
  console.log(
    `Average Runtime CV across problems:   ${avgRuntimeCV.toFixed(2)}`
  );
  console.log(
    `Ratio (GFLOPS/Runtime):              ${ratio.toFixed(2)}x more variable`
  );
  console.log("");

  // 2. Cross-user consistency
  console.log("Analyzing cross-problem percentile consistency...");
  const { userPercentiles, gflopsStdDevs, runtimeStdDevs } =
    await analyzeCrossUserConsistency();

  if (gflopsStdDevs.length === 0) {
    console.log("No users found with 3+ solved problems on H100.");
  } else {
    const avgGflopsStdDev =
      gflopsStdDevs.reduce((a, b) => a + b, 0) / gflopsStdDevs.length;
    const avgRuntimeStdDev =
      runtimeStdDevs.reduce((a, b) => a + b, 0) / runtimeStdDevs.length;
    const consistencyImprovement =
      ((avgGflopsStdDev - avgRuntimeStdDev) / avgGflopsStdDev) * 100;

    console.log("");
    console.log("User percentile consistency (3+ problems):");
    console.log(
      `  - GFLOPS percentile std dev:        ${avgGflopsStdDev.toFixed(2)}`
    );
    console.log(
      `  - Runtime percentile std dev:       ${avgRuntimeStdDev.toFixed(2)}`
    );
    console.log(
      `  Interpretation: Runtime rankings are ${consistencyImprovement.toFixed(1)}% more consistent`
    );
    console.log("");
  }

  // 3. Per-problem table
  console.log("PER-PROBLEM STATISTICS (10+ submissions on H100)");
  console.log("-".repeat(60));
  console.log(
    "Problem".padEnd(25) +
      "Subs".padEnd(7) +
      "GFLOPS CV".padEnd(12) +
      "Runtime CV".padEnd(12) +
      "Ratio"
  );
  console.log("-".repeat(60));

  // Sort by submission count descending
  const sortedProblems = [...problemStats].sort(
    (a, b) => b.submissionCount - a.submissionCount
  );

  for (const problem of sortedProblems.slice(0, 20)) {
    const problemRatio = problem.gflopsCV / problem.runtimeCV;
    console.log(
      problem.problemSlug.padEnd(25) +
        problem.submissionCount.toString().padEnd(7) +
        problem.gflopsCV.toFixed(2).padEnd(12) +
        problem.runtimeCV.toFixed(2).padEnd(12) +
        `${problemRatio.toFixed(2)}x`
    );
  }

  if (sortedProblems.length > 20) {
    console.log(`... and ${sortedProblems.length - 20} more problems`);
  }

  console.log("");

  // 4. Percentile distribution
  if (gflopsStdDevs.length > 0) {
    console.log("CROSS-PROBLEM CONSISTENCY (users with 3+ problems)");
    console.log("-".repeat(60));
    console.log(`Users analyzed: ${gflopsStdDevs.length}`);
    console.log("");

    const sortedGflops = [...gflopsStdDevs].sort((a, b) => a - b);
    const sortedRuntime = [...runtimeStdDevs].sort((a, b) => a - b);

    console.log("Percentile Std Dev Distribution:");
    console.log("                    | GFLOPS | Runtime");
    console.log("--------------------|--------|--------");
    console.log(
      `Min                 | ${sortedGflops[0]!.toFixed(2)}   | ${sortedRuntime[0]!.toFixed(2)}`
    );
    console.log(
      `25th percentile     | ${sortedGflops[Math.floor(sortedGflops.length * 0.25)]!.toFixed(2)}   | ${sortedRuntime[Math.floor(sortedRuntime.length * 0.25)]!.toFixed(2)}`
    );
    console.log(
      `Median              | ${sortedGflops[Math.floor(sortedGflops.length * 0.5)]!.toFixed(2)}   | ${sortedRuntime[Math.floor(sortedRuntime.length * 0.5)]!.toFixed(2)}`
    );
    console.log(
      `75th percentile     | ${sortedGflops[Math.floor(sortedGflops.length * 0.75)]!.toFixed(2)}   | ${sortedRuntime[Math.floor(sortedRuntime.length * 0.75)]!.toFixed(2)}`
    );
    console.log(
      `Max                 | ${sortedGflops[sortedGflops.length - 1]!.toFixed(2)}   | ${sortedRuntime[sortedRuntime.length - 1]!.toFixed(2)}`
    );
    console.log("");
  }

  // 5. Outlier analysis
  console.log("OUTLIER ANALYSIS: tugrul512bit (+1222 rating change)");
  console.log("-".repeat(60));
  await analyzeOutlierUser("tugrul512bit", userPercentiles);

  console.log("OUTLIER ANALYSIS: sagarreddypatil (-998 rating change)");
  console.log("-".repeat(60));
  await analyzeOutlierUser("sagarreddypatil", userPercentiles);

  // 6. Conclusion
  console.log("=".repeat(60));
  console.log("CONCLUSION");
  console.log("=".repeat(60));
  console.log(
    `GFLOPS has ${ratio.toFixed(2)}x higher coefficient of variation than runtime.`
  );

  if (gflopsStdDevs.length > 0) {
    const avgGflopsStdDev =
      gflopsStdDevs.reduce((a, b) => a + b, 0) / gflopsStdDevs.length;
    const avgRuntimeStdDev =
      runtimeStdDevs.reduce((a, b) => a + b, 0) / runtimeStdDevs.length;
    const consistencyImprovement =
      ((avgGflopsStdDev - avgRuntimeStdDev) / avgGflopsStdDev) * 100;
    console.log(
      `Runtime percentiles are ${consistencyImprovement.toFixed(1)}% more consistent across problems.`
    );
  }

  console.log("");
  console.log("Recommendation: Runtime-based ranking better reflects relative");
  console.log("performance skill rather than problem selection bias.");
  console.log("=".repeat(60));
}

main()
  .catch((error) => {
    console.error("Analysis failed:", error);
    process.exit(1);
  })
  .finally(async () => {
    await prisma.$disconnect();
  });
