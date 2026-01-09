# Runtime-Based Benchmarking Migration Guide

This document describes how to migrate Tensara from GFLOPS-based ranking to runtime-based ranking.

---

## Production Migration Steps (Quick Reference)

### Pre-Migration Checklist

- [ ] Create database backup
- [ ] Put site in maintenance mode or pause new submissions
- [ ] Ensure you have the updated Prisma schema deployed
- [ ] Test migration on local DB first

---

### Step 1: Database Backup (CRITICAL)

```bash
# Create backup directory
mkdir -p ~/tensara-backups

# Backup production database
pg_dump "postgresql://postgres.[project-ref]:[password]@aws-0-[region].pooler.supabase.com:5432/postgres" \
  --format=custom \
  --file=~/tensara-backups/tensara_prod_$(date +%Y%m%d_%H%M%S).dump

# Verify backup exists and has reasonable size (~30-40MB)
ls -lh ~/tensara-backups/
```

---

### Step 2: SQL Schema Migration

Connect to production database:

```bash
psql "postgresql://postgres.[project-ref]:[password]@aws-0-[region].pooler.supabase.com:5432/postgres"
```

Run these SQL commands **in order**:

```sql
-- ============================================
-- VERIFY CURRENT STATE
-- ============================================
SELECT COUNT(*) as submission_count FROM "Submission";
-- Should return ~39,437 rows

-- ============================================
-- STEP 1: Drop FK constraint from BlogPostSubmission
-- ============================================
ALTER TABLE "BlogPostSubmission" DROP CONSTRAINT "BlogPostSubmission_submissionId_fkey";

-- ============================================
-- STEP 2: Rename Submission table to LegacySubmission
-- ============================================
ALTER TABLE "Submission" RENAME TO "LegacySubmission";

-- ============================================
-- STEP 3: Rename indexes
-- ============================================
ALTER INDEX "Submission_pkey" RENAME TO "LegacySubmission_pkey";
ALTER INDEX "Submission_createdAt_idx" RENAME TO "LegacySubmission_createdAt_idx";
ALTER INDEX "Submission_problemId_idx" RENAME TO "LegacySubmission_problemId_idx";
ALTER INDEX "Submission_userId_idx" RENAME TO "LegacySubmission_userId_idx";

-- ============================================
-- STEP 4: Rename foreign key constraints
-- ============================================
ALTER TABLE "LegacySubmission" RENAME CONSTRAINT "Submission_problemId_fkey" TO "LegacySubmission_problemId_fkey";
ALTER TABLE "LegacySubmission" RENAME CONSTRAINT "Submission_userId_fkey" TO "LegacySubmission_userId_fkey";

-- ============================================
-- STEP 5: Update BlogPostSubmission for polymorphic references
-- ============================================
-- 5a: Add new legacySubmissionId column
ALTER TABLE "BlogPostSubmission" ADD COLUMN "legacySubmissionId" TEXT;

-- 5b: Copy existing submissionId values to legacySubmissionId
UPDATE "BlogPostSubmission" SET "legacySubmissionId" = "submissionId";

-- 5c: Make submissionId nullable
ALTER TABLE "BlogPostSubmission" ALTER COLUMN "submissionId" DROP NOT NULL;

-- 5d: Set submissionId to NULL (these all reference legacy submissions now)
UPDATE "BlogPostSubmission" SET "submissionId" = NULL;

-- 5e: Add FK constraint to LegacySubmission
ALTER TABLE "BlogPostSubmission"
ADD CONSTRAINT "BlogPostSubmission_legacySubmissionId_fkey"
FOREIGN KEY ("legacySubmissionId") REFERENCES "LegacySubmission"(id)
ON UPDATE CASCADE ON DELETE CASCADE;

-- 5f: Add unique constraint and index for legacySubmissionId
CREATE UNIQUE INDEX "BlogPostSubmission_postId_legacySubmissionId_key"
  ON "BlogPostSubmission"("postId", "legacySubmissionId");
CREATE INDEX "BlogPostSubmission_legacySubmissionId_idx"
  ON "BlogPostSubmission"("legacySubmissionId");

-- ============================================
-- VERIFY MIGRATION
-- ============================================
SELECT COUNT(*) as legacy_count FROM "LegacySubmission";
-- Should return ~39,437 rows
```

---

### Step 3: Run Prisma DB Push

```bash
cd /path/to/tensara-app

# Set DATABASE_URL to production
export DATABASE_URL="postgresql://postgres.[project-ref]:[password]@aws-0-[region].pooler.supabase.com:5432/postgres"

# Push the schema (creates new Submission, TestResult, BenchmarkRun tables)
bunx prisma db push

# Regenerate Prisma client
bunx prisma generate
```

Verify new tables exist:

```sql
SELECT COUNT(*) FROM "Submission";       -- Should be 0
SELECT COUNT(*) FROM "TestResult";       -- Should be 0
SELECT COUNT(*) FROM "BenchmarkRun";     -- Should be 0
SELECT COUNT(*) FROM "LegacySubmission"; -- Should be ~39,437
```

---

### Step 4: Geometric Mean Migration

This converts the `runtime` field from arithmetic mean to geometric mean:

```bash
cd /path/to/tensara-app

# Set DATABASE_URL to production
export DATABASE_URL="postgresql://..."

# ALWAYS run dry-run first
bunx tsx src/scripts/migrate-runtime-to-geometric-mean.ts --dry-run

# Expected output:
# - Total ACCEPTED submissions with benchmarkResults: 23000
# - Updated: 22998 (2 skipped - corrupt data with 1e20 runtime)
# - Average runtime change: ~60ms -> ~48ms (-20%)

# If dry-run looks good, run for real with checkpoint
bunx tsx src/scripts/migrate-runtime-to-geometric-mean.ts --checkpoint=geo-mean-migration.csv
```

---

### Step 5: Deploy Code Changes

Deploy the `winter-sols/migrate-to-runtime` branch which includes:

- Leaderboard mode system (legacy GFLOPS / runtime toggle)
- UI updates for new schema field names
- Migration script

---

### Step 6: Post-Migration Verification

```sql
-- 1. Check submission counts
SELECT
  (SELECT COUNT(*) FROM "LegacySubmission") as legacy_count,
  (SELECT COUNT(*) FROM "Submission") as new_count;
-- Expected: 39437, 0

-- 2. Check runtime values are reasonable
SELECT
  MIN(runtime) as min_runtime,
  AVG(runtime) as avg_runtime,
  MAX(runtime) as max_runtime
FROM "LegacySubmission"
WHERE status = 'ACCEPTED' AND runtime < 3600000;
-- avg_runtime should be ~48ms (down from ~60ms after geometric mean conversion)

-- 3. Check for remaining corrupt data
SELECT COUNT(*) FROM "LegacySubmission" WHERE runtime > 3600000;
-- Should be 2 (known corrupt submissions)

-- 4. Verify leaderboard works
-- Visit /leaderboard and test both "GFLOPS (Legacy)" and "Runtime" modes
```

---

### Rollback Plan (Emergency Only)

**Option A: Restore from backup**

```bash
pg_restore --clean --if-exists \
  -d "postgresql://..." \
  ~/tensara-backups/tensara_prod_YYYYMMDD_HHMMSS.dump
```

**Option B: Manual SQL rollback**

```sql
-- Drop new tables
DROP TABLE IF EXISTS "BenchmarkRun" CASCADE;
DROP TABLE IF EXISTS "TestResult" CASCADE;
DROP TABLE IF EXISTS "Submission" CASCADE;

-- Fix BlogPostSubmission
ALTER TABLE "BlogPostSubmission" DROP CONSTRAINT IF EXISTS "BlogPostSubmission_legacySubmissionId_fkey";
DROP INDEX IF EXISTS "BlogPostSubmission_postId_legacySubmissionId_key";
DROP INDEX IF EXISTS "BlogPostSubmission_legacySubmissionId_idx";
UPDATE "BlogPostSubmission" SET "submissionId" = "legacySubmissionId";
ALTER TABLE "BlogPostSubmission" DROP COLUMN "legacySubmissionId";
ALTER TABLE "BlogPostSubmission" ALTER COLUMN "submissionId" SET NOT NULL;

-- Rename LegacySubmission back to Submission
ALTER TABLE "LegacySubmission" RENAME TO "Submission";
ALTER INDEX "LegacySubmission_pkey" RENAME TO "Submission_pkey";
ALTER INDEX "LegacySubmission_createdAt_idx" RENAME TO "Submission_createdAt_idx";
ALTER INDEX "LegacySubmission_problemId_idx" RENAME TO "Submission_problemId_idx";
ALTER INDEX "LegacySubmission_userId_idx" RENAME TO "Submission_userId_idx";
ALTER TABLE "Submission" RENAME CONSTRAINT "LegacySubmission_problemId_fkey" TO "Submission_problemId_fkey";
ALTER TABLE "Submission" RENAME CONSTRAINT "LegacySubmission_userId_fkey" TO "Submission_userId_fkey";

-- Restore FK
ALTER TABLE "BlogPostSubmission"
ADD CONSTRAINT "BlogPostSubmission_submissionId_fkey"
FOREIGN KEY ("submissionId") REFERENCES "Submission"(id)
ON UPDATE CASCADE ON DELETE CASCADE;
```

---

### Known Data Issues

**2 corrupt submissions with `runtime = 1e20 ms`:**

| ID                          | Problem                     |
| --------------------------- | --------------------------- |
| `cmb5oohob006tge2o7ee4q225` | `cm7jl70y80006s89u4g7aw3qd` |
| `cmb5os0ae006zge2otcxsme4k` | `cm7jl70y80006s89u4g7aw3qd` |

These are skipped by the geometric mean migration script (filtered by `MAX_REASONABLE_RUNTIME_MS = 3600000`). Consider setting their runtime to NULL after migration:

```sql
UPDATE "LegacySubmission"
SET runtime = NULL
WHERE id IN ('cmb5oohob006tge2o7ee4q225', 'cmb5os0ae006zge2otcxsme4k');
```

---

## Technical Details

### Overview

The migration:

1. **Renames** the existing `Submission` table to `LegacySubmission` (preserving all historical data)
2. **Creates** a new `Submission` table with `avgRuntimeMs` as the primary metric
3. **Creates** new `TestResult` and `BenchmarkRun` tables for detailed per-run GPU metrics
4. **Updates** `BlogPostSubmission` to support polymorphic references to both old and new submissions
5. **Converts** the `runtime` field from arithmetic mean to geometric mean

### Why Manual SQL is Required

Prisma's `db push` cannot safely handle table renames. If you just run `db push` with the new schema, Prisma will:

1. DROP the existing `Submission` table (losing all data!)
2. CREATE a new `Submission` table with the new structure
3. CREATE `LegacySubmission` as an empty table

By manually renaming the table first via SQL, we preserve all existing data.

### Leaderboard Mode System

The application now supports two leaderboard modes:

| Mode     | Label             | Metric       | Sort                 | Badge                 |
| -------- | ----------------- | ------------ | -------------------- | --------------------- |
| `legacy` | "GFLOPS (Legacy)" | GFLOPS       | DESC (higher better) | Yellow "Legacy" badge |
| `new`    | "Runtime"         | Runtime (ms) | ASC (lower better)   | No badge              |

Both modes currently query the `LegacySubmission` table since the new `Submission` table is empty. Once submissions start using the new benchmarking system, the "Runtime" mode will query the new `Submission` table.

### Geometric Mean Calculation

The migration script converts runtime from arithmetic mean to geometric mean:

```typescript
function geometricMean(values: number[]): number {
  const positiveValues = values.filter((v) => v > 0);
  if (positiveValues.length === 0) return 0;
  const logSum = positiveValues.reduce((sum, val) => sum + Math.log(val), 0);
  return Math.exp(logSum / positiveValues.length);
}
```

This is more representative of "typical" performance across test cases with varying sizes, as it's less sensitive to outliers than arithmetic mean.

### Table Structure Reference

**LegacySubmission** (renamed from old Submission)

- Uses `gflops` as primary ranking metric
- Uses `runtime` for timing (now geometric mean after migration)
- Has `benchmarkResults` JSON blob with per-test-case data

**Submission** (new)

- Uses `avgRuntimeMs` as primary metric (lower is better)
- Uses `avgGflops` as secondary/informational metric
- Has relations to `TestResult` for per-test-case results

**TestResult** (new)

- One per test case per submission
- Aggregates metrics across all runs
- Has relations to `BenchmarkRun` for individual iterations

**BenchmarkRun** (new)

- One per benchmark iteration
- Contains raw `gpuSamples` JSON array
- Contains aggregated `gpuMetrics` stats

### benchmarkResults JSON Structure (LegacySubmission)

```json
[
  { "name": "n = 2^20", "test_id": 1, "gflops": 76.43, "runtime_ms": 0.1496 },
  { "name": "n = 2^22", "test_id": 2, "gflops": 209.49, "runtime_ms": 0.0208 },
  ...
]
```

---

## Files Changed

### Leaderboard Mode System

- `src/types/submission.ts` - LeaderboardMode enum (legacy/new)
- `src/server/api/routers/submissions.ts` - Dual mode leaderboard queries
- `src/server/api/routers/users.ts` - Dual mode user stats
- `src/pages/leaderboard/index.tsx` - Mode toggle UI
- `src/pages/leaderboard/[slug].tsx` - Mode toggle + Legacy badge

### UI Updates for New Schema

- `src/components/problem/MySubmissions.tsx` - Use avgGflops/avgRuntimeMs
- `src/components/problem/SubmissionResults.tsx` - Use avg_runtime_ms/avg_gflops
- `src/hooks/useSubmissionStream.ts` - Extract TestResultWithRuns correctly
- `src/pages/api/submissions/direct-submit.ts` - Fix Prisma JSON type casting
- `src/pages/submissions/[id].tsx` - Support both old and new schema
- `src/pages/submissions/index.tsx` - Use avgGflops/avgRuntimeMs

### Migration Tools

- `src/scripts/migrate-runtime-to-geometric-mean.ts` - Runtime conversion script
