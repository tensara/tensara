/**
 * Migration Script: Elo Rating from GFLOPS to Runtime
 *
 * This script recalculates all user Elo ratings using runtime instead
 * of GFLOPS as the primary performance metric.
 *
 * Usage:
 *   bun tsx src/scripts/migrate-elo-to-runtime.ts --dry-run  # Preview
 *   bun tsx src/scripts/migrate-elo-to-runtime.ts            # Apply
 */

// @ts-expect-error - Prisma types may not be generated yet
import { PrismaClient } from "@prisma/client";
import * as fs from "fs";
import * as path from "path";

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
const BACKUP_DIR = "/Users/somesh/projects/stk/tensara/tensara-backups";

interface UserUpdate {
  userId: string;
  username: string | null;
  oldRating: number | null;
  newRating: number;
  oldRank: number | null;
  newRank?: number;
  change: number;
  problemsProcessed: number;
}

/**
 * Calculate performance score from percentile (shared formula)
 */
function calculatePerformanceScore(
  percentile: number,
  difficultyMultiplier: number
): number {
  return (200 - 150 * Math.pow(percentile, 0.7)) * difficultyMultiplier;
}

/**
 * Step 1: Backup current ratings to JSON file
 */
async function backupCurrentRatings(): Promise<string> {
  // Ensure backup directory exists
  if (!fs.existsSync(BACKUP_DIR)) {
    fs.mkdirSync(BACKUP_DIR, { recursive: true });
  }

  // Get all users with ratings
  const users = await prisma.user.findMany({
    where: {
      OR: [{ rating: { not: null } }, { rank: { not: null } }],
    },
    select: {
      id: true,
      username: true,
      rating: true,
      rank: true,
    },
    orderBy: { rating: "desc" },
  });

  const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
  const backupPath = path.join(
    BACKUP_DIR,
    `ratings-backup-gflops-${timestamp}.json`
  );

  fs.writeFileSync(
    backupPath,
    JSON.stringify(
      {
        timestamp: new Date().toISOString(),
        totalUsers: users.length,
        note: "Backup before migrating from GFLOPS-based to Runtime-based Elo ratings",
        users,
      },
      null,
      2
    )
  );

  return backupPath;
}

/**
 * Step 2: Calculate runtime-based rating for a user
 * (Uses the same logic as the updated getUserRating() in users.ts)
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
    _min: { runtime: true },
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
 * Step 3: Main migration function
 */
async function migrateEloRatings(dryRun: boolean) {
  console.log("=".repeat(60));
  console.log("Elo Rating Migration: GFLOPS -> Runtime");
  console.log("=".repeat(60));
  console.log(
    `Mode: ${dryRun ? "DRY RUN (preview only)" : "LIVE (will update database)"}`
  );
  console.log("");

  // Backup first (even for dry-run)
  console.log("Step 1: Backing up current ratings...");
  const backupPath = await backupCurrentRatings();
  console.log(`✓ Backup saved to: ${backupPath}`);
  console.log("");

  // Get all users with accepted submissions
  console.log("Step 2: Calculating new runtime-based ratings...");
  const users = await prisma.user.findMany({
    where: {
      submissions: {
        some: {
          status: "ACCEPTED",
          runtime: { not: null },
        },
      },
    },
    select: {
      id: true,
      username: true,
      rating: true,
      rank: true,
    },
  });

  console.log(`Processing ${users.length} users...`);

  const updates: UserUpdate[] = [];
  let processed = 0;

  for (const user of users) {
    const result = await calculateRuntimeRating(user.id);
    const change = result.rating - (user.rating ?? START_RATING);

    updates.push({
      userId: user.id,
      username: user.username,
      oldRating: user.rating,
      newRating: result.rating,
      oldRank: user.rank,
      change,
      problemsProcessed: result.problemsProcessed,
    });

    processed++;
    if (processed % 50 === 0) {
      console.log(`  Processed ${processed}/${users.length} users...`);
    }
  }

  console.log(`✓ Processed all ${users.length} users`);
  console.log("");

  // Sort by new rating descending and assign ranks
  updates.sort((a, b) => b.newRating - a.newRating);

  let currentRank = 1;
  for (let i = 0; i < updates.length; i++) {
    updates[i]!.newRank = currentRank;
    // Only increment rank if next user has different rating
    if (
      i < updates.length - 1 &&
      updates[i]!.newRating !== updates[i + 1]!.newRating
    ) {
      currentRank = i + 2;
    }
  }

  // Calculate statistics
  const changes = updates.map((u) => u.change);
  const avgChange = changes.reduce((a, b) => a + b, 0) / changes.length;
  const sortedChanges = [...changes].sort((a, b) => a - b);
  const medianChange = sortedChanges[Math.floor(changes.length / 2)]!;
  const maxIncrease = Math.max(...changes);
  const maxDecrease = Math.min(...changes);

  const increases = changes.filter((c) => c > 5).length;
  const decreases = changes.filter((c) => c < -5).length;
  const unchanged = changes.filter((c) => Math.abs(c) <= 5).length;

  // Print summary
  console.log("MIGRATION SUMMARY");
  console.log("-".repeat(60));
  console.log(`Users updated:                ${users.length}`);
  console.log(
    `Average rating change:        ${avgChange >= 0 ? "+" : ""}${avgChange.toFixed(1)}`
  );
  console.log(
    `Median rating change:         ${medianChange >= 0 ? "+" : ""}${medianChange.toFixed(1)}`
  );
  console.log(`Max increase:                 +${maxIncrease.toFixed(0)}`);
  console.log(`Max decrease:                 ${maxDecrease.toFixed(0)}`);
  console.log("");
  console.log(
    `Users with increased rating:  ${increases} (${((increases / users.length) * 100).toFixed(1)}%)`
  );
  console.log(
    `Users with decreased rating:  ${decreases} (${((decreases / users.length) * 100).toFixed(1)}%)`
  );
  console.log(
    `Users unchanged (±5):         ${unchanged} (${((unchanged / users.length) * 100).toFixed(1)}%)`
  );
  console.log("");

  // Show top changes
  console.log("TOP 10 RATING INCREASES");
  console.log("-".repeat(60));
  const topGainers = [...updates]
    .sort((a, b) => b.change - a.change)
    .slice(0, 10);
  console.log(
    "Username".padEnd(20) +
      "Old Rating".padEnd(12) +
      "New Rating".padEnd(12) +
      "Change"
  );
  console.log("-".repeat(60));
  for (const u of topGainers) {
    const username = (u.username ?? "unknown").padEnd(20);
    const oldRating = (u.oldRating?.toString() ?? "—").padEnd(12);
    const newRating = u.newRating.toString().padEnd(12);
    const change = `+${u.change.toFixed(0)}`;
    console.log(`${username}${oldRating}${newRating}${change}`);
  }
  console.log("");

  console.log("TOP 10 RATING DECREASES");
  console.log("-".repeat(60));
  const topLosers = [...updates]
    .sort((a, b) => a.change - b.change)
    .slice(0, 10);
  console.log(
    "Username".padEnd(20) +
      "Old Rating".padEnd(12) +
      "New Rating".padEnd(12) +
      "Change"
  );
  console.log("-".repeat(60));
  for (const u of topLosers) {
    const username = (u.username ?? "unknown").padEnd(20);
    const oldRating = (u.oldRating?.toString() ?? "—").padEnd(12);
    const newRating = u.newRating.toString().padEnd(12);
    const change = u.change.toFixed(0);
    console.log(`${username}${oldRating}${newRating}${change}`);
  }
  console.log("");

  // Show ranking changes for top users
  console.log("TOP 20 RANKING CHANGES");
  console.log("-".repeat(60));
  const top20 = updates.slice(0, 20);
  console.log(
    "Username".padEnd(20) +
      "Old Rank".padEnd(10) +
      "New Rank".padEnd(10) +
      "Change"
  );
  console.log("-".repeat(60));
  for (const u of top20) {
    const username = (u.username ?? "unknown").padEnd(20);
    const oldRank = (u.oldRank?.toString() ?? "—").padEnd(10);
    const newRank = (u.newRank?.toString() ?? "—").padEnd(10);
    let changeStr = "—";
    if (u.oldRank && u.newRank) {
      const rankChange = u.oldRank - u.newRank;
      if (rankChange > 0) {
        changeStr = `↑${rankChange}`;
      } else if (rankChange < 0) {
        changeStr = `↓${Math.abs(rankChange)}`;
      } else {
        changeStr = "—";
      }
    }
    console.log(`${username}${oldRank}${newRank}${changeStr}`);
  }
  console.log("");

  // Apply changes if not dry run
  if (!dryRun) {
    console.log("Step 3: Applying changes to database...");
    console.log("Running in transaction...");

    await prisma.$transaction(
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      async (tx: any) => {
        for (const update of updates) {
          await tx.user.update({
            where: { id: update.userId },
            data: {
              rating: update.newRating,
              rank: update.newRank,
            },
          });
        }
      }
    );

    console.log("✓ Database updated successfully");
    console.log("");
    console.log("=".repeat(60));
    console.log("MIGRATION COMPLETE");
    console.log("=".repeat(60));
    console.log("");
    console.log("All user ratings and ranks have been updated to use");
    console.log("runtime-based calculation instead of GFLOPS.");
    console.log("");
    console.log(`Backup available at: ${backupPath}`);
  } else {
    console.log("=".repeat(60));
    console.log("DRY RUN COMPLETE - No changes were made");
    console.log("=".repeat(60));
    console.log("");
    console.log("Review the changes above. If everything looks good,");
    console.log("run without --dry-run to apply the migration:");
    console.log("");
    console.log("  bun tsx src/scripts/migrate-elo-to-runtime.ts");
    console.log("");
  }
}

/**
 * Main entry point
 */
async function main() {
  const args = process.argv.slice(2);
  const dryRun = args.includes("--dry-run");

  try {
    await migrateEloRatings(dryRun);
  } catch (error) {
    console.error("Migration failed:", error);
    process.exit(1);
  } finally {
    await prisma.$disconnect();
  }
}

main();
