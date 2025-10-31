import { z } from "zod";
import { createTRPCRouter, protectedProcedure, publicProcedure } from "../trpc";
import { TRPCError } from "@trpc/server";

// Define enums for validation
const PostTypeEnum = z.enum([
  "GENERAL",
  "SOLUTION",
  "COMPARISON",
  "TUTORIAL",
  "BENCHMARK_ANALYSIS",
]);

const PostStatusEnum = z.enum(["DRAFT", "PUBLISHED", "ARCHIVED"]);

const TagCategoryEnum = z.enum([
  "GENERAL",
  "PROBLEM",
  "LANGUAGE",
  "OPTIMIZATION",
  "DIFFICULTY",
  "TOPIC",
]);

// Helper function to generate a slug from a title
function slugify(text: string): string {
  return text
    .toLowerCase()
    .trim()
    .replace(/[^\w\s-]/g, "")
    .replace(/[\s_-]+/g, "-")
    .replace(/^-+|-+$/g, "");
}

// Helper function to generate a unique slug with random 7-digit number
async function generateUniqueSlug(
  db: any,
  title: string,
  maxRetries = 10
): Promise<string> {
  // Generate base slug from title
  const baseSlug = slugify(title);

  // Try up to maxRetries times
  for (let attempt = 0; attempt < maxRetries; attempt++) {
    // Generate random 7-digit number
    const randomNumber = Math.floor(1000000 + Math.random() * 9000000);
    const slug = `${baseSlug}-${randomNumber}`;

    // Check if slug exists
    const existing = await db.blogPost.findUnique({
      where: { slug },
      select: { id: true },
    });

    if (!existing) {
      return slug;
    }

    // If exists, loop will try again with new random number
  }

  // Fallback: use timestamp if all retries failed (extremely unlikely)
  const timestamp = Date.now();
  return `${baseSlug}-${timestamp}`;
}

// Helper function to calculate read time from content
function calculateReadTime(content: string): number {
  const wordsPerMinute = 200;
  const wordCount = content.trim().split(/\s+/).length;
  return Math.ceil(wordCount / wordsPerMinute);
}

// Helper function to generate title from submission
function generateTitleFromSubmission(submission: any, problem: any): string {
  const gflops = submission.gflops ? submission.gflops.toFixed(1) : "N/A";
  const language = submission.language.toUpperCase();
  return `My ${gflops} GFLOPS ${language} Solution to ${problem.title}`;
}

// Helper function to generate post content with submission markers
function generatePostContent(
  submissionId: string,
  approach?: string,
  challenges?: string,
  optimizations?: string
): string {
  let content = `# Solution Overview\n\n`;
  content += `{{submission:${submissionId}}}\n\n`;

  if (approach) {
    content += `## Approach\n\n${approach}\n\n`;
  }

  if (challenges) {
    content += `## Challenges\n\n${challenges}\n\n`;
  }

  if (optimizations) {
    content += `## Optimizations\n\n${optimizations}\n\n`;
  }

  return content;
}

// Helper function to generate or find auto tags
async function generateAutoTags(
  db: any,
  problem: any,
  language: string
): Promise<string[]> {
  const tagIds: string[] = [];

  // Create/find problem tag
  const problemTag = await db.tag.upsert({
    where: { slug: slugify(problem.title) },
    create: {
      name: problem.title,
      slug: slugify(problem.title),
      category: "PROBLEM",
      description: `Solutions for ${problem.title}`,
    },
    update: {},
  });
  tagIds.push(problemTag.id);

  // Create/find language tag
  const languageTag = await db.tag.upsert({
    where: { slug: language.toLowerCase() },
    create: {
      name: language.toUpperCase(),
      slug: language.toLowerCase(),
      category: "LANGUAGE",
      description: `${language.toUpperCase()} solutions`,
    },
    update: {},
  });
  tagIds.push(languageTag.id);

  return tagIds;
}

export const blogpostRouter = createTRPCRouter({
  create: protectedProcedure
    .input(
      z.object({
        title: z.string().min(1),
        content: z.string().min(1),
        postType: PostTypeEnum.default("GENERAL"),
        status: PostStatusEnum.default("DRAFT"),
        excerpt: z.string().optional(),
        submissionIds: z.array(z.string()).optional(),
        tagIds: z.array(z.string()).optional(),
      })
    )
    .mutation(async ({ ctx, input }) => {
      const db = ctx.db as any;

      // Generate unique slug
      const slug = await generateUniqueSlug(db, input.title);

      // Calculate read time
      const readTimeMinutes = calculateReadTime(input.content);

      // Prepare post data
      const postData: any = {
        title: input.title,
        content: input.content,
        slug,
        postType: input.postType,
        status: input.status,
        excerpt: input.excerpt,
        readTimeMinutes,
        authorId: ctx.session.user.id,
      };

      // Set publishedAt if status is PUBLISHED
      if (input.status === "PUBLISHED") {
        postData.publishedAt = new Date();
      }

      // Create the post
      const post = await db.blogPost.create({
        data: postData,
      });

      // Create BlogPostSubmission relations
      if (input.submissionIds && input.submissionIds.length > 0) {
        await db.blogPostSubmission.createMany({
          data: input.submissionIds.map(
            (submissionId: string, index: number) => ({
              postId: post.id,
              submissionId,
              order: index,
            })
          ),
        });
      }

      // Create BlogPostTag relations
      if (input.tagIds && input.tagIds.length > 0) {
        await db.blogPostTag.createMany({
          data: input.tagIds.map((tagId: string) => ({
            postId: post.id,
            tagId,
          })),
        });
      }

      return post;
    }),

  createFromSubmission: protectedProcedure
    .input(
      z.object({
        submissionId: z.string(),
        title: z.string().optional(),
        approach: z.string().optional(),
        challenges: z.string().optional(),
        optimizations: z.string().optional(),
        status: z.enum(["DRAFT", "PUBLISHED"]).default("DRAFT"),
      })
    )
    .mutation(async ({ ctx, input }) => {
      const db = ctx.db as any;

      // Fetch submission with problem details
      const submission = await db.submission.findUnique({
        where: { id: input.submissionId },
        include: {
          problem: true,
        },
      });

      if (!submission) {
        throw new TRPCError({
          code: "NOT_FOUND",
          message: "Submission not found",
        });
      }

      // Verify ownership
      if (submission.userId !== ctx.session.user.id) {
        throw new TRPCError({
          code: "FORBIDDEN",
          message: "You can only create posts from your own submissions",
        });
      }

      // Generate title if not provided
      const title =
        input.title ||
        generateTitleFromSubmission(submission, submission.problem);

      // Generate content template
      const content = generatePostContent(
        input.submissionId,
        input.approach,
        input.challenges,
        input.optimizations
      );

      // Generate unique slug
      const slug = await generateUniqueSlug(db, title);

      // Calculate read time
      const readTimeMinutes = calculateReadTime(content);

      // Auto-create tags for problem and language
      const tagIds = await generateAutoTags(
        db,
        submission.problem,
        submission.language
      );

      // Prepare post data
      const postData: any = {
        title,
        content,
        slug,
        postType: "SOLUTION",
        status: input.status,
        readTimeMinutes,
        authorId: ctx.session.user.id,
      };

      // Set publishedAt if status is PUBLISHED
      if (input.status === "PUBLISHED") {
        postData.publishedAt = new Date();
      }

      // Create the post
      const post = await db.blogPost.create({
        data: postData,
      });

      // Create BlogPostSubmission relation
      await db.blogPostSubmission.create({
        data: {
          postId: post.id,
          submissionId: input.submissionId,
          order: 0,
        },
      });

      // Create BlogPostTag relations
      await db.blogPostTag.createMany({
        data: tagIds.map((tagId: string) => ({
          postId: post.id,
          tagId,
        })),
      });

      return post;
    }),

  getById: publicProcedure
    .input(
      z.object({
        id: z.string().optional(),
        slug: z.string().optional(),
      })
    )
    .query(async ({ ctx, input }) => {
      const db = ctx.db as any;

      if (!input.id && !input.slug) {
        throw new TRPCError({
          code: "BAD_REQUEST",
          message: "Either id or slug must be provided",
        });
      }

      const whereClause = input.id ? { id: input.id } : { slug: input.slug };

      const post = await db.blogPost.findUnique({
        where: whereClause,
        include: {
          author: { select: { id: true, name: true, image: true } },
          comments: {
            include: {
              author: { select: { id: true, name: true, image: true } },
              upvotes: true,
            },
            orderBy: { createdAt: "asc" },
          },
          submissions: {
            include: {
              submission: {
                include: {
                  problem: true,
                },
              },
            },
            orderBy: { order: "asc" },
          },
          tags: {
            include: {
              tag: true,
            },
          },
          upvotes: true,
          _count: { select: { comments: true, upvotes: true } },
        },
      });

      if (!post) {
        throw new TRPCError({
          code: "NOT_FOUND",
          message: "Blog post not found",
        });
      }

      // Increment view count asynchronously (fire and forget)
      void db.blogPost
        .update({
          where: whereClause,
          data: { viewCount: { increment: 1 } },
        })
        .catch(() => {
          // Silently fail if view count update fails
        });

      return post;
    }),

  getAll: publicProcedure
    .input(
      z.object({
        status: PostStatusEnum.optional(),
        postType: PostTypeEnum.optional(),
        tagSlugs: z.array(z.string()).optional(),
        authorId: z.string().optional(),
        problemSlug: z.string().optional(),
        cursor: z.string().optional(),
        limit: z.number().min(1).max(100).default(10),
      })
    )
    .query(async ({ ctx, input }) => {
      const db = ctx.db as any;

      // Build where clause
      const where: any = {};

      // Default to PUBLISHED posts if no status specified
      // This ensures drafts don't appear in public listings
      if (input.status) {
        where.status = input.status;
      } else {
        where.status = "PUBLISHED";
      }

      if (input.postType) {
        where.postType = input.postType;
      }

      if (input.authorId) {
        where.authorId = input.authorId;
      }

      // Filter by tags
      if (input.tagSlugs && input.tagSlugs.length > 0) {
        where.tags = {
          some: {
            tag: {
              slug: { in: input.tagSlugs },
            },
          },
        };
      }

      // Filter by problem
      if (input.problemSlug) {
        where.submissions = {
          some: {
            submission: {
              problem: {
                slug: input.problemSlug,
              },
            },
          },
        };
      }

      // Add cursor to where clause
      if (input.cursor) {
        where.id = { lt: input.cursor };
      }

      const posts = await db.blogPost.findMany({
        where,
        take: input.limit,
        orderBy: { publishedAt: "desc" },
        include: {
          author: { select: { id: true, name: true, image: true } },
          submissions: {
            take: 1,
            include: {
              submission: {
                include: {
                  problem: true,
                },
              },
            },
            orderBy: { order: "asc" },
          },
          tags: {
            include: {
              tag: true,
            },
          },
          _count: { select: { comments: true, upvotes: true } },
        },
      });

      // Get next cursor
      const nextCursor =
        posts.length === input.limit ? posts[posts.length - 1]?.id : null;

      return {
        posts,
        nextCursor,
      };
    }),

  update: protectedProcedure
    .input(
      z.object({
        id: z.string(),
        title: z.string().min(1).optional(),
        content: z.string().min(1).optional(),
        status: PostStatusEnum.optional(),
        postType: PostTypeEnum.optional(),
        excerpt: z.string().optional(),
        submissionIds: z.array(z.string()).optional(),
        tagIds: z.array(z.string()).optional(),
      })
    )
    .mutation(async ({ ctx, input }) => {
      const db = ctx.db as any;

      // Check if post exists and user owns it
      const existing = await db.blogPost.findUnique({
        where: { id: input.id },
        select: { authorId: true, title: true, status: true },
      });

      if (!existing) {
        throw new TRPCError({ code: "NOT_FOUND", message: "Post not found" });
      }

      if (existing.authorId !== ctx.session.user.id) {
        throw new TRPCError({
          code: "FORBIDDEN",
          message: "You can only edit your own posts",
        });
      }

      // Build update data
      const data: any = {};

      if (input.title !== undefined) {
        data.title = input.title;
        // Re-generate slug if title changes
        if (input.title !== existing.title) {
          data.slug = await generateUniqueSlug(db, input.title);
        }
      }

      if (input.content !== undefined) {
        data.content = input.content;
        // Recalculate read time
        data.readTimeMinutes = calculateReadTime(input.content);
      }

      if (input.status !== undefined) {
        data.status = input.status;
        // Set publishedAt when transitioning to PUBLISHED
        if (input.status === "PUBLISHED" && existing.status !== "PUBLISHED") {
          data.publishedAt = new Date();
        }
      }

      if (input.postType !== undefined) {
        data.postType = input.postType;
      }

      if (input.excerpt !== undefined) {
        data.excerpt = input.excerpt;
      }

      // Update the post
      const updated = await db.blogPost.update({
        where: { id: input.id },
        data,
      });

      // Update submissions if provided
      if (input.submissionIds !== undefined) {
        // Delete existing submission relations
        await db.blogPostSubmission.deleteMany({
          where: { postId: input.id },
        });

        // Create new submission relations
        if (input.submissionIds.length > 0) {
          await db.blogPostSubmission.createMany({
            data: input.submissionIds.map(
              (submissionId: string, index: number) => ({
                postId: input.id,
                submissionId,
                order: index,
              })
            ),
          });
        }
      }

      // Update tags if provided
      if (input.tagIds !== undefined) {
        // Delete existing tag relations
        await db.blogPostTag.deleteMany({
          where: { postId: input.id },
        });

        // Create new tag relations
        if (input.tagIds.length > 0) {
          await db.blogPostTag.createMany({
            data: input.tagIds.map((tagId: string) => ({
              postId: input.id,
              tagId,
            })),
          });
        }
      }

      return updated;
    }),

  // Draft auto-save operations
  saveDraft: protectedProcedure
    .input(
      z.object({
        title: z.string(),
        content: z.string(),
        excerpt: z.string().optional(),
      })
    )
    .mutation(async ({ ctx, input }) => {
      const db = ctx.db as any;

      // Check if user has an existing DRAFT post (only one draft allowed per user for auto-save)
      const existingDraft = await db.blogPost.findFirst({
        where: {
          authorId: ctx.session.user.id,
          status: "DRAFT",
        },
        orderBy: { updatedAt: "desc" },
      });

      // Calculate read time
      const readTimeMinutes = calculateReadTime(input.content);

      if (existingDraft) {
        // Update existing draft
        const updated = await db.blogPost.update({
          where: { id: existingDraft.id },
          data: {
            title: input.title,
            content: input.content,
            excerpt: input.excerpt,
            readTimeMinutes,
            updatedAt: new Date(),
          },
        });
        return updated;
      } else {
        // Create new draft
        const slug = await generateUniqueSlug(db, input.title);
        const post = await db.blogPost.create({
          data: {
            title: input.title,
            content: input.content,
            excerpt: input.excerpt,
            slug,
            postType: "GENERAL",
            status: "DRAFT",
            readTimeMinutes,
            authorId: ctx.session.user.id,
          },
        });
        return post;
      }
    }),

  loadDraft: protectedProcedure.query(async ({ ctx }) => {
    const db = ctx.db as any;

    // Find the most recent draft post for the current user
    const draft = await db.blogPost.findFirst({
      where: {
        authorId: ctx.session.user.id,
        status: "DRAFT",
      },
      orderBy: { updatedAt: "desc" },
      include: {
        tags: {
          include: {
            tag: true,
          },
        },
        submissions: {
          include: {
            submission: {
              include: {
                problem: true,
              },
            },
          },
          orderBy: { order: "asc" },
        },
      },
    });

    return draft;
  }),

  deleteDraft: protectedProcedure
    .input(z.object({ id: z.string() }))
    .mutation(async ({ ctx, input }) => {
      const db = ctx.db as any;

      // Check if post exists and belongs to current user
      const existing = await db.blogPost.findUnique({
        where: { id: input.id },
        select: { authorId: true, status: true },
      });

      if (!existing) {
        throw new TRPCError({
          code: "NOT_FOUND",
          message: "Draft post not found",
        });
      }

      if (existing.authorId !== ctx.session.user.id) {
        throw new TRPCError({
          code: "FORBIDDEN",
          message: "You can only delete your own drafts",
        });
      }

      if (existing.status !== "DRAFT") {
        throw new TRPCError({
          code: "BAD_REQUEST",
          message: "Can only delete draft posts",
        });
      }

      await db.blogPost.delete({ where: { id: input.id } });

      return { ok: true };
    }),

  delete: protectedProcedure
    .input(z.object({ id: z.string() }))
    .mutation(async ({ ctx, input }) => {
      const db = ctx.db as any;
      const existing = await db.blogPost.findUnique({
        where: { id: input.id },
        select: { authorId: true },
      });

      if (!existing) {
        throw new TRPCError({
          code: "NOT_FOUND",
          message: "Blog post not found",
        });
      }

      if (existing.authorId !== ctx.session.user.id) {
        throw new TRPCError({
          code: "FORBIDDEN",
          message: "You can only delete your own posts",
        });
      }

      await db.blogPost.delete({ where: { id: input.id } });

      return { ok: true };
    }),

  getUserPosts: protectedProcedure.query(async ({ ctx }) => {
    const db = ctx.db as any;
    const posts = await db.blogPost.findMany({
      where: { authorId: ctx.session.user.id },
      orderBy: { createdAt: "desc" },
      include: {
        _count: { select: { comments: true, upvotes: true } },
        tags: {
          include: {
            tag: true,
          },
        },
      },
    });

    return posts as any[];
  }),
});
