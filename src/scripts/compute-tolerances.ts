import { PrismaClient } from "@prisma/client";
import * as fs from "fs";
import * as path from "path";

const prisma = new PrismaClient();

const DEFAULT_PERCENTILE = 10.0;
const OUTPUT_DIR = path.join(process.cwd(), "tolerance-results");

type FilterOpts = {
  slugs?: string[];
  ids?: string[];
  slugRegex?: RegExp;
};

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
  filter?: FilterOpts
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
  if (filter?.slugs?.length) {
    console.log(`Slug filter: ${filter.slugs.join(", ")}`);
  }
  if (filter?.ids?.length) {
    console.log(`ID filter: ${filter.ids.join(", ")}`);
  }
  if (filter?.slugRegex) {
    console.log(`Slug regex filter: ${String(filter.slugRegex)}`);
  }
  console.log("");

  // Fetch all problems with definitions
  const where: Record<string, unknown> = {
    definition: {
      not: null,
    },
  };

  if (filter?.slugs?.length) {
    where.slug = { in: filter.slugs };
  }

  if (filter?.ids?.length) {
    where.id = { in: filter.ids };
  }

  if (filter?.slugRegex) {
    // Prisma doesn't support regex on all DBs consistently; do an in-memory filter below.
  }

  const problems = await prisma.problem.findMany({
    where,
    select: {
      id: true,
      slug: true,
      title: true,
      definition: true,
    },
    orderBy: {
      slug: "asc",
    },
  });

  const filteredProblems = filter?.slugRegex
    ? problems.filter((p) => filter.slugRegex!.test(p.slug))
    : problems;

  console.log(`Found ${filteredProblems.length} problems with definitions`);
  console.log("");

  // Ensure output directory exists
  if (!fs.existsSync(OUTPUT_DIR)) {
    fs.mkdirSync(OUTPUT_DIR, { recursive: true });
  }

  const results: ToleranceResult[] = [];
  let successCount = 0;
  let errorCount = 0;

  for (let i = 0; i < filteredProblems.length; i++) {
    const problem = filteredProblems[i];
    if (!problem.definition) continue;

    process.stdout.write(
      `[${i + 1}/${filteredProblems.length}] Processing: ${problem.slug}... `
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
    totalProblems: filteredProblems.length,
    successCount: successCount,
    errorCount: errorCount,
    results: results,
  };

  fs.writeFileSync(outputFile, JSON.stringify(output, null, 2));

  console.log("");
  console.log("=".repeat(60));
  console.log("COMPUTATION COMPLETE");
  console.log("=".repeat(60));
  console.log(`Total problems: ${filteredProblems.length}`);
  console.log(`Successful: ${successCount}`);
  console.log(`Errors: ${errorCount}`);
  console.log(`Results saved to: ${outputFile}`);
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

  const parseCommaSeparated = (raw: string | undefined): string[] => {
    if (!raw) return [];
    return raw
      .split(",")
      .map((s) => s.trim())
      .filter(Boolean);
  };

  // Optional filters
  const slugsIndex = args.indexOf("--slugs");
  const slugs =
    slugsIndex !== -1 && args[slugsIndex + 1]
      ? parseCommaSeparated(args[slugsIndex + 1])
      : [];

  const idsIndex = args.indexOf("--ids");
  const ids =
    idsIndex !== -1 && args[idsIndex + 1]
      ? parseCommaSeparated(args[idsIndex + 1])
      : [];

  const slugRegexIndex = args.indexOf("--slug-regex");
  const slugRegexRaw =
    slugRegexIndex !== -1 && args[slugRegexIndex + 1]
      ? args[slugRegexIndex + 1]
      : undefined;

  let slugRegex: RegExp | undefined = undefined;
  if (slugRegexRaw) {
    try {
      slugRegex = new RegExp(slugRegexRaw);
    } catch (e) {
      console.error(
        `Invalid --slug-regex value: ${slugRegexRaw}. Must be a valid JS RegExp pattern.`
      );
      process.exit(1);
    }
  }

  const filter: FilterOpts | undefined =
    slugs.length || ids.length || slugRegex
      ? {
          slugs: slugs.length ? slugs : undefined,
          ids: ids.length ? ids : undefined,
          slugRegex,
        }
      : undefined;

  if (slugs.length && slugRegex) {
    console.warn(
      "Warning: both --slugs and --slug-regex provided. Both will be applied (intersection)."
    );
  }
  if (ids.length && slugRegex) {
    console.warn(
      "Warning: both --ids and --slug-regex provided. Both will be applied (intersection)."
    );
  }

  try {
    await computeAllTolerances(percentile, filter);
  } catch (error) {
    console.error("Script failed:", error);
    process.exit(1);
  } finally {
    await prisma.$disconnect();
  }
}

main();
