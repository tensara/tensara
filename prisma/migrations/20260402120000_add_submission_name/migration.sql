-- Add optional user-facing label for a submission so exports/comparisons
-- can use stable human-readable names instead of only ids/timestamps.
ALTER TABLE "Submission"
ADD COLUMN "name" TEXT;
