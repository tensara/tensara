/**
 * Migration Script: Convert runtime values from arithmetic mean to geometric mean
 *
 * This script recalculates the `runtime` field for all ACCEPTED submissions
 * using geometric mean of per-test runtimes instead of arithmetic mean.
 *
 * Geometric mean is more appropriate for benchmark timings because:
 * 1. It reduces the impact of outliers
 * 2. It better represents multiplicative relationships in performance data
 * 3. It's more suitable when comparing ratios/speedups
 *
 * Usage:
 *   npx tsx src/scripts/migrate-runtime-to-geometric-mean.ts [--dry-run]
 *
 * Options:
 *   --dry-run    Show what would be changed without making actual updates
 */

// eslint-disable-next-line @typescript-eslint/no-require-imports
import { PrismaClient } from "@prisma/client";

const prisma = new PrismaClient();

interface BenchmarkResult {
  name: string;
  test_id: number;
  runtime_ms: number;
  gflops?: number;
  runs?: Array<{
    run_index: number;
    runtime_ms: number;
    gflops?: number;
    gpu_samples?: unknown[];
    gpu_metrics?: unknown;
  }>;
}

/**
 * Calculate geometric mean of an array of positive numbers
 */
function geometricMean(values: number[]): number {
  if (values.length === 0) return 0;

  // Filter out non-positive values (geometric mean requires positive numbers)
  const positiveValues = values.filter((v) => v > 0);
  if (positiveValues.length === 0) return 0;

  // Use log-sum-exp trick for numerical stability
  const logSum = positiveValues.reduce((sum, val) => sum + Math.log(val), 0);
  return Math.exp(logSum / positiveValues.length);
}

async function migrateRuntimeToGeometricMean(dryRun: boolean = false) {
  console.log("=".repeat(60));
  console.log("Runtime Migration: Arithmetic Mean -> Geometric Mean");
  console.log("=".repeat(60));
  console.log(`Mode: ${dryRun ? "DRY RUN (no changes will be made)" : "LIVE"}`);
  console.log("");

  // Fetch all ACCEPTED submissions with benchmarkResults
  const submissions = await prisma.submission.findMany({
    where: {
      status: "ACCEPTED",
      benchmarkResults: {
        not: null,
      },
      runtime: {
        not: null,
      },
    },
    select: {
      id: true,
      runtime: true,
      benchmarkResults: true,
      createdAt: true,
      problem: {
        select: {
          slug: true,
        },
      },
      user: {
        select: {
          username: true,
        },
      },
    },
    orderBy: {
      createdAt: "desc",
    },
  });

  console.log(
    `Found ${submissions.length} ACCEPTED submissions with benchmark data\n`
  );

  let updatedCount = 0;
  let skippedCount = 0;
  let errorCount = 0;

  const updates: Array<{
    id: string;
    oldRuntime: number;
    newRuntime: number;
    problem: string;
    user: string;
    testCount: number;
  }> = [];

  for (const submission of submissions) {
    try {
      const benchmarkResults = submission.benchmarkResults as
        | BenchmarkResult[]
        | null;

      if (
        !benchmarkResults ||
        !Array.isArray(benchmarkResults) ||
        benchmarkResults.length === 0
      ) {
        skippedCount++;
        continue;
      }

      // Extract per-test runtime values
      const testRuntimes = benchmarkResults
        .map((result) => result.runtime_ms)
        .filter(
          (runtime): runtime is number =>
            typeof runtime === "number" && runtime > 0
        );

      if (testRuntimes.length === 0) {
        skippedCount++;
        continue;
      }

      // Calculate geometric mean
      const oldRuntime = submission.runtime!;
      const newRuntime = geometricMean(testRuntimes);

      // Skip if the difference is negligible (less than 0.01%)
      const percentDiff =
        (Math.abs(newRuntime - oldRuntime) / oldRuntime) * 100;
      if (percentDiff < 0.01) {
        skippedCount++;
        continue;
      }

      updates.push({
        id: submission.id,
        oldRuntime,
        newRuntime,
        problem: submission.problem?.slug ?? "unknown",
        user: submission.user?.username ?? "unknown",
        testCount: testRuntimes.length,
      });

      if (!dryRun) {
        await prisma.submission.update({
          where: { id: submission.id },
          data: { runtime: newRuntime },
        });
      }

      updatedCount++;
    } catch (error) {
      console.error(`Error processing submission ${submission.id}:`, error);
      errorCount++;
    }
  }

  // Print summary
  console.log("-".repeat(60));
  console.log("MIGRATION SUMMARY");
  console.log("-".repeat(60));
  console.log(`Total submissions processed: ${submissions.length}`);
  console.log(`Updated: ${updatedCount}`);
  console.log(`Skipped (no change needed): ${skippedCount}`);
  console.log(`Errors: ${errorCount}`);
  console.log("");

  if (updates.length > 0) {
    console.log("-".repeat(60));
    console.log("CHANGES" + (dryRun ? " (would be applied)" : " (applied)"));
    console.log("-".repeat(60));

    // Show first 20 updates
    const displayUpdates = updates.slice(0, 20);
    for (const update of displayUpdates) {
      const changePercent = (
        ((update.newRuntime - update.oldRuntime) / update.oldRuntime) *
        100
      ).toFixed(2);
      const direction =
        update.newRuntime < update.oldRuntime ? "faster" : "slower";
      console.log(
        `  ${update.id.slice(0, 8)}... | ${update.problem.padEnd(25)} | ` +
          `${update.oldRuntime.toFixed(3)}ms -> ${update.newRuntime.toFixed(3)}ms ` +
          `(${Math.abs(parseFloat(changePercent)).toFixed(2)}% ${direction})`
      );
    }

    if (updates.length > 20) {
      console.log(`  ... and ${updates.length - 20} more updates`);
    }

    // Statistics on changes
    const improvements = updates.filter((u) => u.newRuntime < u.oldRuntime);
    const regressions = updates.filter((u) => u.newRuntime > u.oldRuntime);
    const avgChange =
      updates.reduce(
        (sum, u) => sum + ((u.newRuntime - u.oldRuntime) / u.oldRuntime) * 100,
        0
      ) / updates.length;

    console.log("");
    console.log("-".repeat(60));
    console.log("IMPACT ANALYSIS");
    console.log("-".repeat(60));
    console.log(
      `  Submissions with improved (lower) runtime: ${improvements.length}`
    );
    console.log(`  Submissions with higher runtime: ${regressions.length}`);
    console.log(`  Average change: ${avgChange.toFixed(2)}%`);
  }

  console.log("");
  console.log("=".repeat(60));
  console.log(
    dryRun ? "DRY RUN COMPLETE - No changes were made" : "MIGRATION COMPLETE"
  );
  console.log("=".repeat(60));
}

async function main() {
  const args = process.argv.slice(2);
  const dryRun = args.includes("--dry-run");

  try {
    await migrateRuntimeToGeometricMean(dryRun);
  } catch (error) {
    console.error("Migration failed:", error);
    process.exit(1);
  } finally {
    await prisma.$disconnect();
  }
}

main();
