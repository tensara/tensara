import { PrismaClient } from "@prisma/client";

/**
 * Data migration script for blog enhancements
 * This script updates existing blog posts to populate the new optional fields
 *
 * IMPORTANT: Run this AFTER:
 * 1. Pushing the schema changes: pnpm prisma db push
 * 2. Generating the Prisma client: pnpm prisma generate
 */
async function migrateExistingBlogPosts() {
  const prisma = new PrismaClient();

  try {
    console.log("Starting blog post data migration...");

    // Fetch all existing blog posts
    const posts = await prisma.blogPost.findMany();

    console.log(`Found ${posts.length} blog posts to process`);

    if (posts.length === 0) {
      console.log("✅ No posts found in database.");
      return;
    }

    let migratedCount = 0;

    for (const post of posts) {
      const updates: any = {};

      // Generate URL-friendly slug from title if missing
      if (!(post as any).slug) {
        const baseSlug = post.title
          .toLowerCase()
          .replace(/[^a-z0-9]+/g, "-")
          .replace(/^-+|-+$/g, "");

        // Ensure slug uniqueness by appending post ID
        updates.slug = `${baseSlug}-${post.id.slice(-6)}`;
      }

      // Generate excerpt from content if missing (first 200 characters)
      if (!(post as any).excerpt) {
        updates.excerpt =
          post.content.length > 200
            ? post.content.slice(0, 197) + "..."
            : post.content;
      }

      // Calculate estimated read time if missing (average 200 words per minute)
      if (!(post as any).readTimeMinutes) {
        const wordCount = post.content.split(/\s+/).length;
        updates.readTimeMinutes = Math.max(1, Math.ceil(wordCount / 200));
      }

      // Set publishedAt for already published posts
      // Since existing posts don't have status field yet, we'll set them as PUBLISHED
      // and use createdAt as publishedAt
      if (!(post as any).publishedAt) {
        updates.publishedAt = post.createdAt;
        updates.status = "PUBLISHED";
      }

      // Only update if there are changes
      if (Object.keys(updates).length > 0) {
        await prisma.blogPost.update({
          where: { id: post.id },
          data: updates,
        });

        migratedCount++;
        console.log(`✓ Migrated post: "${post.title}"`);
      }
    }

    console.log(
      `\n✅ Successfully migrated ${migratedCount} out of ${posts.length} blog posts`
    );

    if (migratedCount === 0) {
      console.log("All posts already have the new fields populated.");
    } else {
      // Summary of changes
      console.log("\nMigration Summary:");
      console.log("- Generated slugs for posts without them");
      console.log("- Created excerpts from content");
      console.log("- Calculated read times based on word count");
      console.log(
        "- Set publishedAt dates and PUBLISHED status for existing posts"
      );
    }
  } catch (error) {
    console.error("❌ Migration failed:", error);
    throw error;
  } finally {
    await prisma.$disconnect();
  }
}

// Run migration if this file is executed directly
if (import.meta.url === `file://${process.argv[1]}`) {
  migrateExistingBlogPosts()
    .then(() => {
      console.log("\n🎉 Migration completed successfully");
      process.exit(0);
    })
    .catch((error) => {
      console.error("\n💥 Migration failed:", error);
      process.exit(1);
    });
}

export { migrateExistingBlogPosts };
