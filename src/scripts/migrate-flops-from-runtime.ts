/**
 * Migration Script: Recalculate GFLOPS from runtime using analytical FLOPS
 *
 * This script recalculates the `gflops` field for all ACCEPTED submissions
 * using analytical FLOPS values from flops.json and average runtime.
 *
 * Previous approach: GFLOPS was calculated from individual run GFLOPS values
 * New approach: GFLOPS = (analytical_FLOPS / avg_runtime_in_seconds) / 1e9
 *
 * This ensures consistency with the analytical FLOPS values defined for each
 * problem and test case, rather than relying on per-run measurements.
 *
 * Usage:
 *   npx tsx src/scripts/migrate-flops-from-runtime.ts [--dry-run]
 *
 * Options:
 *   --dry-run    Show what would be changed without making actual updates
 */

// eslint-disable-next-line @typescript-eslint/no-require-imports
import { PrismaClient, type Prisma } from "@prisma/client";
import * as fs from "fs";
import * as path from "path";
import { fileURLToPath } from "url";

const prisma = new PrismaClient();

interface BenchmarkResult {
  name: string;
  test_id: number;
  runtime_ms: number;
  gflops?: number;
  avg_runtime_ms?: number;
  avg_gflops?: number;
  runs?: Array<{
    run_index: number;
    runtime_ms: number;
    gflops?: number;
    gpu_samples?: unknown[];
    gpu_metrics?: unknown;
  }>;
}

type FlopsData = Record<string, Record<string, number>>;

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

/**
 * Calculate GFLOPS from FLOPS and runtime
 * GFLOPS = (FLOPS / runtime_in_seconds) / 1e9
 *        = (FLOPS * 1000) / (runtime_ms * 1e9)
 *        = FLOPS / (runtime_ms * 1e6)
 */
function calculateGflops(flops: number, runtimeMs: number): number {
  if (runtimeMs <= 0) return 0;
  return flops / (runtimeMs * 1e6);
}

/**
 * Load FLOPS data from flops.json
 */
function loadFlopsData(): FlopsData {
  // Get directory of current file (works with tsx/ES modules)
  const currentFileUrl = import.meta.url;
  const currentFilePath = fileURLToPath(currentFileUrl);
  const currentDir = path.dirname(currentFilePath);
  const flopsPath = path.join(currentDir, "flops.json");
  const flopsContent = fs.readFileSync(flopsPath, "utf-8");
  return JSON.parse(flopsContent) as FlopsData;
}

async function migrateFlopsFromRuntime(dryRun = false) {
  console.log("=".repeat(60));
  console.log(
    "GFLOPS Migration: From Per-Run GFLOPS -> From Runtime + Analytical FLOPS"
  );
  console.log("=".repeat(60));
  console.log(`Mode: ${dryRun ? "DRY RUN (no changes will be made)" : "LIVE"}`);
  console.log("");

  // Load FLOPS data
  const flopsData = loadFlopsData();
  console.log(
    `Loaded FLOPS data for ${Object.keys(flopsData).length} problems\n`
  );

  // Fetch all ACCEPTED submissions with benchmarkResults
  const submissions = (await prisma.submission.findMany({
    where: {
      status: "ACCEPTED",
    },
    select: {
      id: true,
      gflops: true,
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
  })) as unknown as Array<{
    id: string;
    gflops: number | null;
    benchmarkResults: unknown;
    createdAt: Date;
    problem: { slug: string } | null;
    user: { username: string } | null;
  }>;

  console.log(
    `Found ${submissions.length} ACCEPTED submissions with benchmark data\n`
  );

  let updatedCount = 0;
  let skippedCount = 0;
  let errorCount = 0;

  const updates: Array<{
    id: string;
    oldGflops: number | null;
    newGflops: number | null;
    problem: string;
    user: string;
    testCasesUpdated: number;
    testCasesSkipped: number;
  }> = [];

  for (const submission of submissions) {
    try {
      // Filter out submissions without benchmarkResults
      if (!submission.benchmarkResults) {
        skippedCount++;
        continue;
      }

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

      const problemSlug = submission.problem?.slug;
      if (!problemSlug) {
        skippedCount++;
        continue;
      }

      // Get FLOPS data for this problem
      const problemFlops = flopsData[problemSlug];
      if (!problemFlops) {
        skippedCount++;
        continue;
      }

      // Update each test case's GFLOPS
      let testCasesUpdated = 0;
      let testCasesSkipped = 0;
      const updatedResults: BenchmarkResult[] = [];
      const testCaseGflops: number[] = [];

      for (const result of benchmarkResults) {
        const testCaseName = result.name;
        const runtimeMs = result.avg_runtime_ms ?? result.runtime_ms ?? 0;

        if (runtimeMs <= 0) {
          // Skip test cases with invalid runtime
          updatedResults.push(result);
          testCasesSkipped++;
          continue;
        }

        // Look up FLOPS for this test case
        const flops = problemFlops[testCaseName];

        if (flops === undefined || flops === -1) {
          // No FLOPS data available, keep existing gflops or skip
          updatedResults.push(result);
          testCasesSkipped++;
          continue;
        }

        // Calculate new GFLOPS from runtime
        const newGflops = calculateGflops(flops, runtimeMs);

        // Update the result
        const updatedResult: BenchmarkResult = {
          ...result,
          gflops: newGflops,
          // Also update avg_gflops if it exists
          avg_gflops: newGflops,
        };

        updatedResults.push(updatedResult);
        testCaseGflops.push(newGflops);
        testCasesUpdated++;
      }

      // Calculate overall submission GFLOPS as geometric mean of test case GFLOPS
      const oldGflops = submission.gflops;
      const newGflops =
        testCaseGflops.length > 0 ? geometricMean(testCaseGflops) : null;

      // Skip if no test cases were updated or if the change is negligible
      if (testCasesUpdated === 0) {
        skippedCount++;
        continue;
      }

      // Skip if the difference is negligible (less than 0.01%)
      if (oldGflops !== null && newGflops !== null) {
        const percentDiff = (Math.abs(newGflops - oldGflops) / oldGflops) * 100;
        if (percentDiff < 0.01) {
          skippedCount++;
          continue;
        }
      }

      updates.push({
        id: submission.id,
        oldGflops,
        newGflops,
        problem: problemSlug,
        user: submission.user?.username ?? "unknown",
        testCasesUpdated,
        testCasesSkipped,
      });

      if (!dryRun) {
        await prisma.submission.update({
          where: { id: submission.id },
          data: {
            gflops: newGflops,
            benchmarkResults:
              updatedResults as unknown as Prisma.InputJsonValue,
          },
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
      const oldStr = update.oldGflops?.toFixed(2) ?? "null";
      const newStr = update.newGflops?.toFixed(2) ?? "null";
      let changeStr = "";
      if (update.oldGflops !== null && update.newGflops !== null) {
        const changePercent =
          ((update.newGflops - update.oldGflops) / update.oldGflops) * 100;
        const direction =
          update.newGflops > update.oldGflops ? "higher" : "lower";
        changeStr = ` (${Math.abs(changePercent).toFixed(2)}% ${direction})`;
      }
      console.log(
        `  ${update.id.slice(0, 8)}... | ${update.problem.padEnd(25)} | ` +
          `${oldStr} -> ${newStr} GFLOPS${changeStr} | ` +
          `Tests: ${update.testCasesUpdated} updated, ${update.testCasesSkipped} skipped`
      );
    }

    if (updates.length > 20) {
      console.log(`  ... and ${updates.length - 20} more updates`);
    }

    // Statistics on changes
    const improvements = updates.filter(
      (u) =>
        u.oldGflops !== null &&
        u.newGflops !== null &&
        u.newGflops > u.oldGflops
    );
    const regressions = updates.filter(
      (u) =>
        u.oldGflops !== null &&
        u.newGflops !== null &&
        u.newGflops < u.oldGflops
    );
    const avgChange =
      updates
        .filter((u) => u.oldGflops !== null && u.newGflops !== null)
        .reduce(
          (sum, u) =>
            sum + ((u.newGflops! - u.oldGflops!) / u.oldGflops!) * 100,
          0
        ) /
      updates.filter((u) => u.oldGflops !== null && u.newGflops !== null)
        .length;

    const totalTestCasesUpdated = updates.reduce(
      (sum, u) => sum + u.testCasesUpdated,
      0
    );
    const totalTestCasesSkipped = updates.reduce(
      (sum, u) => sum + u.testCasesSkipped,
      0
    );

    console.log("");
    console.log("-".repeat(60));
    console.log("IMPACT ANALYSIS");
    console.log("-".repeat(60));
    console.log(`  Submissions with higher GFLOPS: ${improvements.length}`);
    console.log(`  Submissions with lower GFLOPS: ${regressions.length}`);
    if (avgChange !== undefined && !isNaN(avgChange)) {
      console.log(`  Average change: ${avgChange.toFixed(2)}%`);
    }
    console.log(`  Total test cases updated: ${totalTestCasesUpdated}`);
    console.log(`  Total test cases skipped: ${totalTestCasesSkipped}`);
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
    await migrateFlopsFromRuntime(dryRun);
  } catch (error) {
    console.error("Migration failed:", error);
    process.exit(1);
  } finally {
    await prisma.$disconnect();
  }
}

main();
