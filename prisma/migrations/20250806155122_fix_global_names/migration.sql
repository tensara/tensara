/*
  Warnings:

  - A unique constraint covering the columns `[slug,userId]` on the table `Workspace` will be added. If there are existing duplicate values, this will fail.

*/
-- DropIndex
DROP INDEX "Workspace_slug_key";

-- CreateIndex
CREATE UNIQUE INDEX "Workspace_slug_userId_key" ON "Workspace"("slug", "userId");
