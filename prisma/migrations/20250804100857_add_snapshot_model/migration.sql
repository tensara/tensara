/*
  Warnings:

  - You are about to drop the column `slug` on the `Snapshot` table. All the data in the column will be lost.

*/
-- DropIndex
DROP INDEX "Snapshot_slug_key";

-- AlterTable
ALTER TABLE "Snapshot" DROP COLUMN "slug";
