/**
 * Compute Tolerances Script
 *
 * This script computes optimal tolerance values (rtol/atol) for all problems
 * in the database using the tolerance API endpoint.
 *
 * Usage:
 *   bun tsx src/scripts/compute-tolerances.ts --percentile 10.0
 *   bun tsx src/scripts/compute-tolerances.ts --percentile 10.0 --update-db
 */

// @ts-expect-error - Prisma types may not be generated yet
import { PrismaClient } from "@prisma/client";
import * as fs from "fs";
import * as path from "path";

const prisma = new PrismaClient();

const DEFAULT_PERCENTILE = 10.0;
const OUTPUT_DIR = path.join(process.cwd(), "tolerance-results");

interface ToleranceResult {
  problemId: string;
  problemSlug: string;
  problemTitle: string;
  percentile: number;
  rtol_t: number[];
  atol_t: number[];
  test_case_stats: Array<{
    name: string;
    rtol: number;
    atol: number;
  }>;
  error?: string;
  computedAt: string;
}

interface ToleranceResponse {
  problem_name: string;
  percentile: number;
  rtol_t: number[];
  atol_t: number[];
  test_case_stats: Array<{
    name: string;
    rtol: number;
    atol: number;
  }>;
  error?: string;
}

/**
 * Extract class name from problem definition
 */
function extractClassName(problemDef: string, slug: string): string {
  // Try to find class definition
  const classMatch = problemDef.match(/class\s+(\w+)\s*\(/);
  if (classMatch) {
    return classMatch[1];
  }

  // Fallback: convert slug to class name (e.g., "conv-1d" -> "conv_1d")
  return slug.replace(/-/g, "_");
}

/**
 * Call the tolerance API endpoint
 */
async function computeTolerances(
  problemName: string,
  problemDef: string,
  percentile: number,
  modalEndpoint: string
): Promise<ToleranceResponse> {
  const response = await fetch(`${modalEndpoint}/compute-tolerances`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      problem_name: problemName,
      problem_def: problemDef,
      percentile: percentile,
    }),
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(
      `Tolerance API returned ${response.status}: ${errorText}`
    );
  }

  return (await response.json()) as ToleranceResponse;
}

/**
 * Compute tolerances for all problems
 */
async function computeAllTolerances(
  percentile: number,
  updateDb: boolean
): Promise<void> {
  const modalEndpoint = process.env.MODAL_ENDPOINT;
  if (!modalEndpoint) {
    throw new Error("MODAL_ENDPOINT environment variable is required");
  }

  console.log("=".repeat(60));
  console.log("COMPUTING TOLERANCES FOR ALL PROBLEMS");
  console.log("=".repeat(60));
  console.log(`Percentile: ${percentile}`);
  console.log(`Modal Endpoint: ${modalEndpoint}`);
  console.log(`Update DB: ${updateDb ? "Yes" : "No"}`);
  console.log("");

  // Fetch all problems with definitions
  const problems = await prisma.problem.findMany({
    where: {
      definition: {
        not: null,
      },
    },
    select: {
      id: true,
      slug: true,
      title: true,
      definition: true,
      baselineBenchmarks: true,
    },
    orderBy: {
      slug: "asc",
    },
  });

  console.log(`Found ${problems.length} problems with definitions`);
  console.log("");

  // Ensure output directory exists
  if (!fs.existsSync(OUTPUT_DIR)) {
    fs.mkdirSync(OUTPUT_DIR, { recursive: true });
  }

  const results: ToleranceResult[] = [];
  let successCount = 0;
  let errorCount = 0;

  for (let i = 0; i < problems.length; i++) {
    const problem = problems[i];
    if (!problem.definition) continue;

    process.stdout.write(
      `[${i + 1}/${problems.length}] Processing: ${problem.slug}... `
    );

    try {
      const className = extractClassName(problem.definition, problem.slug);

      const toleranceResult = await computeTolerances(
        className,
        problem.definition,
        percentile,
        modalEndpoint
      );

      if (toleranceResult.error) {
        console.log(`✗ Error: ${toleranceResult.error}`);
        errorCount++;

        results.push({
          problemId: problem.id,
          problemSlug: problem.slug,
          problemTitle: problem.title,
          percentile: percentile,
          rtol_t: [],
          atol_t: [],
          test_case_stats: [],
          error: toleranceResult.error,
          computedAt: new Date().toISOString(),
        });
      } else {
        const maxRtol =
          toleranceResult.rtol_t.length > 0
            ? Math.max(...toleranceResult.rtol_t)
            : 0;
        const maxAtol =
          toleranceResult.atol_t.length > 0
            ? Math.max(...toleranceResult.atol_t)
            : 0;
        console.log(
          `✓ (rtol: ${maxRtol.toExponential(2)}, atol: ${maxAtol.toExponential(2)})`
        );
        successCount++;

        const result: ToleranceResult = {
          problemId: problem.id,
          problemSlug: problem.slug,
          problemTitle: problem.title,
          percentile: percentile,
          rtol_t: toleranceResult.rtol_t,
          atol_t: toleranceResult.atol_t,
          test_case_stats: toleranceResult.test_case_stats,
          computedAt: new Date().toISOString(),
        };

        results.push(result);

        // Optionally update database
        if (updateDb) {
          // Store tolerances in baselineBenchmarks or create a new field
          // For now, we'll store in a JSON structure
          const tolerancesData = {
            rtol_t: toleranceResult.rtol_t,
            atol_t: toleranceResult.atol_t,
            percentile: percentile,
            computedAt: new Date().toISOString(),
          };

          await prisma.problem.update({
            where: { id: problem.id },
            data: {
              baselineBenchmarks: {
                ...((problem.baselineBenchmarks as Record<string, unknown>) ||
                  {}),
                tolerances: tolerancesData,
              },
            },
          });
        }
      }
    } catch (error) {
      console.log(`✗ Error: ${error instanceof Error ? error.message : String(error)}`);
      errorCount++;

      results.push({
        problemId: problem.id,
        problemSlug: problem.slug,
        problemTitle: problem.title,
        percentile: percentile,
        rtol_t: [],
        atol_t: [],
        test_case_stats: [],
        error: error instanceof Error ? error.message : String(error),
        computedAt: new Date().toISOString(),
      });
    }

    // Small delay to avoid overwhelming the API
    await new Promise((resolve) => setTimeout(resolve, 100));
  }

  // Save results to file
  const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
  const outputFile = path.join(
    OUTPUT_DIR,
    `tolerances-percentile${percentile}-${timestamp}.json`
  );

  const output = {
    timestamp: new Date().toISOString(),
    percentile: percentile,
    totalProblems: problems.length,
    successCount: successCount,
    errorCount: errorCount,
    results: results,
  };

  fs.writeFileSync(outputFile, JSON.stringify(output, null, 2));

  console.log("");
  console.log("=".repeat(60));
  console.log("COMPUTATION COMPLETE");
  console.log("=".repeat(60));
  console.log(`Total problems: ${problems.length}`);
  console.log(`Successful: ${successCount}`);
  console.log(`Errors: ${errorCount}`);
  console.log(`Results saved to: ${outputFile}`);
  if (updateDb) {
    console.log("Database updated with tolerance values");
  }
  console.log("");
}

/**
 * Main entry point
 */
async function main() {
  const args = process.argv.slice(2);

  // Parse percentile
  const percentileIndex = args.indexOf("--percentile");
  const percentile =
    percentileIndex !== -1 && args[percentileIndex + 1]
      ? parseFloat(args[percentileIndex + 1])
      : DEFAULT_PERCENTILE;

  if (isNaN(percentile) || percentile < 0 || percentile > 100) {
    console.error("Invalid percentile. Must be between 0 and 100.");
    process.exit(1);
  }

  // Check if we should update the database
  const updateDb = args.includes("--update-db");

  try {
    await computeAllTolerances(percentile, updateDb);
  } catch (error) {
    console.error("Script failed:", error);
    process.exit(1);
  } finally {
    await prisma.$disconnect();
  }
}

main();
