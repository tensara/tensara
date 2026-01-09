/**
 * Migration Script: Convert runtime field from arithmetic mean to geometric mean
 *
 * This script recalculates the `runtime` field in LegacySubmission table
 * from arithmetic mean to geometric mean using the per-test-case runtime_ms
 * values stored in the benchmarkResults JSON field.
 *
 * Usage:
 *   bunx tsx src/scripts/migrate-runtime-to-geometric-mean.ts --dry-run
 *   bunx tsx src/scripts/migrate-runtime-to-geometric-mean.ts
 *   bunx tsx src/scripts/migrate-runtime-to-geometric-mean.ts --batch-size=50
 *   bunx tsx src/scripts/migrate-runtime-to-geometric-mean.ts --checkpoint=checkpoint.csv
 *
 * Options:
 *   --dry-run         Preview changes without writing to database
 *   --batch-size=N    Process N submissions at a time (default: 100)
 *   --checkpoint=FILE Resume from a checkpoint file or create one for progress tracking
 */

import { PrismaClient } from "@prisma/client";
import * as fs from "fs";

const prisma = new PrismaClient();

// Parse command line arguments
const args = process.argv.slice(2);
const isDryRun = args.includes("--dry-run");
const batchSizeArg = args.find((arg) => arg.startsWith("--batch-size="));
const checkpointArg = args.find((arg) => arg.startsWith("--checkpoint="));

const BATCH_SIZE = batchSizeArg
  ? parseInt(batchSizeArg.split("=")[1] ?? "100")
  : 100;
const CHECKPOINT_FILE = checkpointArg?.split("=")[1] ?? null;

interface BenchmarkResult {
  name: string;
  test_id?: number;
  gflops?: number;
  runtime_ms: number;
}

/**
 * Calculate geometric mean of an array of positive numbers
 * Formula: exp(mean(log(values)))
 */
function geometricMean(values: number[]): number {
  if (values.length === 0) return 0;

  // Filter out non-positive values (can't take log of <= 0)
  const positiveValues = values.filter((v) => v > 0);
  if (positiveValues.length === 0) return 0;

  // Use log-sum-exp for numerical stability
  const logSum = positiveValues.reduce((sum, val) => sum + Math.log(val), 0);
  return Math.exp(logSum / positiveValues.length);
}

/**
 * Load processed submission IDs from checkpoint file
 */
function loadCheckpoint(file: string): Set<string> {
  const processed = new Set<string>();
  if (fs.existsSync(file)) {
    const content = fs.readFileSync(file, "utf-8");
    const lines = content.split("\n").filter((line) => line.trim());
    // Skip header
    for (let i = 1; i < lines.length; i++) {
      const id = lines[i]?.split(",")[0];
      if (id) processed.add(id);
    }
    console.log(
      `Loaded ${processed.size} processed submissions from checkpoint`
    );
  }
  return processed;
}

/**
 * Append a processed submission to checkpoint file
 */
function appendCheckpoint(
  file: string,
  id: string,
  oldRuntime: number | null,
  newRuntime: number
): void {
  if (!fs.existsSync(file)) {
    fs.writeFileSync(file, "id,old_runtime,new_runtime,timestamp\n");
  }
  const line = `${id},${oldRuntime ?? "null"},${newRuntime},${new Date().toISOString()}\n`;
  fs.appendFileSync(file, line);
}

async function main() {
  console.log("=".repeat(60));
  console.log("Runtime to Geometric Mean Migration Script");
  console.log("=".repeat(60));
  console.log(
    `Mode: ${isDryRun ? "DRY RUN (no changes will be made)" : "LIVE"}`
  );
  console.log(`Batch size: ${BATCH_SIZE}`);
  if (CHECKPOINT_FILE) {
    console.log(`Checkpoint file: ${CHECKPOINT_FILE}`);
  }
  console.log("=".repeat(60));

  // Load checkpoint if specified
  const processedIds = CHECKPOINT_FILE
    ? loadCheckpoint(CHECKPOINT_FILE)
    : new Set<string>();

  // Count total submissions with benchmarkResults
  // Using raw query to check for non-null JSON
  const totalCountResult = await prisma.$queryRaw<[{ count: bigint }]>`
    SELECT COUNT(*) as count 
    FROM "LegacySubmission" 
    WHERE "benchmarkResults" IS NOT NULL 
    AND status = 'ACCEPTED'
  `;
  const totalCount = Number(totalCountResult[0]?.count ?? 0);

  console.log(
    `\nTotal ACCEPTED submissions with benchmarkResults: ${totalCount}`
  );
  console.log(`Already processed (from checkpoint): ${processedIds.size}`);

  let processed = 0;
  let updated = 0;
  let skipped = 0;
  let errors = 0;
  let offset = 0;

  // Statistics for reporting
  let sumOldRuntime = 0;
  let sumNewRuntime = 0;
  let countWithChange = 0;

  // Cap for reasonable runtime values (1 hour in ms)
  const MAX_REASONABLE_RUNTIME_MS = 3600000;

  while (true) {
    // Fetch batch of submissions using raw query
    const submissions = await prisma.$queryRaw<
      Array<{
        id: string;
        runtime: number | null;
        benchmarkResults: unknown;
      }>
    >`
      SELECT id, runtime, "benchmarkResults"
      FROM "LegacySubmission"
      WHERE "benchmarkResults" IS NOT NULL
      AND status = 'ACCEPTED'
      ORDER BY id ASC
      LIMIT ${BATCH_SIZE}
      OFFSET ${offset}
    `;

    if (submissions.length === 0) break;
    offset += submissions.length;

    for (const submission of submissions) {
      // Skip if already processed (from checkpoint)
      if (processedIds.has(submission.id)) {
        skipped++;
        continue;
      }

      processed++;

      try {
        // Parse benchmarkResults JSON
        const results = submission.benchmarkResults as BenchmarkResult[] | null;
        if (!results || !Array.isArray(results)) {
          console.log(
            `  [SKIP] ${submission.id}: No valid benchmarkResults array`
          );
          skipped++;
          continue;
        }

        // Extract runtime_ms values, filtering out unreasonable values (> 1 hour)
        const runtimes = results
          .map((r) => r.runtime_ms)
          .filter(
            (r): r is number =>
              typeof r === "number" && r > 0 && r <= MAX_REASONABLE_RUNTIME_MS
          );

        if (runtimes.length === 0) {
          console.log(
            `  [SKIP] ${submission.id}: No valid runtime_ms values (may have unreasonable values)`
          );
          skipped++;
          continue;
        }

        // Calculate geometric mean
        const geoMean = geometricMean(runtimes);
        // Explicitly convert to number - raw queries may return Decimal as string
        const oldRuntime =
          submission.runtime !== null ? Number(submission.runtime) : null;

        // Track statistics (only for reasonable values)
        if (
          oldRuntime !== null &&
          !isNaN(oldRuntime) &&
          oldRuntime > 0 &&
          oldRuntime <= MAX_REASONABLE_RUNTIME_MS
        ) {
          sumOldRuntime += oldRuntime;
          sumNewRuntime += geoMean;
          countWithChange++;
        }

        // Log the change
        const change =
          oldRuntime !== null
            ? (((geoMean - oldRuntime) / oldRuntime) * 100).toFixed(2)
            : "N/A";
        console.log(
          `  [${isDryRun ? "PREVIEW" : "UPDATE"}] ${submission.id}: ` +
            `${oldRuntime?.toFixed(4) ?? "null"} ms -> ${geoMean.toFixed(4)} ms ` +
            `(${change}% change, ${runtimes.length} test cases)`
        );

        // Update database (unless dry run)
        if (!isDryRun) {
          await prisma.legacySubmission.update({
            where: { id: submission.id },
            data: { runtime: geoMean },
          });

          // Save to checkpoint
          if (CHECKPOINT_FILE) {
            appendCheckpoint(
              CHECKPOINT_FILE,
              submission.id,
              oldRuntime,
              geoMean
            );
          }
        }

        updated++;
      } catch (error) {
        console.error(`  [ERROR] ${submission.id}: ${error}`);
        errors++;
      }

      // Progress update every 100 submissions
      if (processed % 100 === 0) {
        console.log(
          `\nProgress: ${processed}/${totalCount - processedIds.size} processed, ${updated} updated, ${errors} errors`
        );
      }
    }
  }

  // Final summary
  console.log("\n" + "=".repeat(60));
  console.log("Migration Summary");
  console.log("=".repeat(60));
  console.log(`Total processed: ${processed}`);
  console.log(`Updated: ${updated}`);
  console.log(`Skipped: ${skipped}`);
  console.log(`Errors: ${errors}`);

  if (countWithChange > 0) {
    const avgOld = sumOldRuntime / countWithChange;
    const avgNew = sumNewRuntime / countWithChange;
    const avgChange = (((avgNew - avgOld) / avgOld) * 100).toFixed(2);
    console.log(
      `\nAverage runtime change: ${avgOld.toFixed(4)} ms -> ${avgNew.toFixed(4)} ms (${avgChange}%)`
    );
  }

  if (isDryRun) {
    console.log("\n[DRY RUN] No changes were made to the database.");
    console.log("Run without --dry-run to apply changes.");
  }

  console.log("=".repeat(60));
}

main()
  .catch((error) => {
    console.error("Migration failed:", error);
    process.exit(1);
  })
  .finally(() => {
    void prisma.$disconnect();
  });
