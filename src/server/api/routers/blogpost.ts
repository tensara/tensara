import { z } from "zod";
import { createTRPCRouter, protectedProcedure, publicProcedure } from "../trpc";
import { TRPCError } from "@trpc/server";
import type {
  PrismaClient,
  Prisma,
  Submission,
  Problem,
  TagCategory,
} from "@prisma/client";

// Define enums for validation
const PostTypeEnum = z.enum([
  "GENERAL",
  "SOLUTION",
  "COMPARISON",
  "TUTORIAL",
  "BENCHMARK_ANALYSIS",
]);
const SortEnum = z.enum(["recent", "top"]);

const PostStatusEnum = z.enum(["DRAFT", "PUBLISHED", "ARCHIVED"]);
const StatusEnum = PostStatusEnum;

const _TagCategoryEnum = z.enum([
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
  db: PrismaClient,
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

// Convert free-form tag strings to Tag IDs (upsert if missing)
async function resolveTagIdsFromStrings(
  db: PrismaClient,
  rawTags?: string[],
  defaultCategory: z.infer<typeof _TagCategoryEnum> = "GENERAL"
): Promise<string[]> {
  if (!rawTags?.length) return [];

  // normalize -> slug -> dedupe while preserving order
  const slugs: string[] = [];
  for (const t of rawTags) {
    const s = slugify(String(t || "").trim());
    if (s && !slugs.includes(s)) slugs.push(s);
  }
  if (!slugs.length) return [];

  // fetch existing
  const existing = await db.tag.findMany({
    where: { slug: { in: slugs } },
    select: { id: true, slug: true },
  });

  const bySlug = new Map<string, string>();
  for (const t of existing) bySlug.set(t.slug, t.id);

  // create missing
  const toCreate = slugs.filter((s) => !bySlug.has(s));
  if (toCreate.length) {
    const created = await db.$transaction(
      toCreate.map((s: string) =>
        db.tag.create({
          data: {
            name: s.replace(/-/g, " ").toUpperCase(),
            slug: s,
            category: defaultCategory as unknown as TagCategory,
            description: undefined,
          },
          select: { id: true, slug: true },
        })
      )
    );
    for (const t of created) bySlug.set(t.slug, t.id);
  }

  // return ids in the same order as input slugs (deduped)
  const ids: string[] = [];
  for (const s of slugs) {
    const id = bySlug.get(s);
    if (id && !ids.includes(id)) ids.push(id);
  }
  return ids;
}

// Helper function to generate title from submission
function generateTitleFromSubmission(
  submission: Submission & { gflops?: number | null; language: string },
  problem: Problem
): string {
  const gflops = submission.gflops ? submission.gflops.toFixed(1) : "N/A";
  const language = submission.language.toUpperCase();
  return `My ${gflops} GFLOPS ${language} Solution to ${problem.title}`;
}

// Only allow slug regeneration while DRAFT; never after publish.
function shouldRegenerateSlugOnTitleChange(
  prevStatus: "DRAFT" | "PUBLISHED" | "ARCHIVED"
) {
  return prevStatus === "DRAFT";
}

// Cursor pagination helper
function toNextCursor<T extends { id: string }>(items: T[], limit: number) {
  return items.length === limit ? (items[items.length - 1]?.id ?? null) : null;
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

const ListInput = z.object({
  cursor: z.string().nullish(),
  limit: z.number().min(1).max(50).default(20),
  authorId: z.string().optional(),
  query: z.string().optional(),
  tagSlugs: z.array(z.string()).optional(),
  sort: SortEnum.default("recent"),
});

// Helper function to generate or find auto tags
async function generateAutoTags(
  db: PrismaClient,
  problem: Problem,
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
        tags: z.array(z.string()).optional(),
      })
    )
    .mutation(async ({ ctx, input }) => {
      const db = ctx.db as PrismaClient;

      // Generate unique slug
      const slug = await generateUniqueSlug(db, input.title);

      // Calculate read time
      const readTimeMinutes = calculateReadTime(input.content);

      // Prepare post data
      const postData: Prisma.BlogPostUncheckedCreateInput = {
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
      const resolved = await resolveTagIdsFromStrings(db, input.tags);
      const tagIds = [...new Set([...(input.tagIds ?? []), ...resolved])];
      if (tagIds.length) {
        await db.blogPostTag.createMany({
          data: tagIds.map((tagId) => ({ postId: post.id, tagId })),
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
      const db = ctx.db as PrismaClient;

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
        input.title ??
        generateTitleFromSubmission(
          submission as Submission & { problem: Problem },
          submission.problem
        );

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
      const postData: Prisma.BlogPostUncheckedCreateInput = {
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
      const db = ctx.db as PrismaClient;

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
      const db = ctx.db as PrismaClient;

      // Build where clause
      const where: Prisma.BlogPostWhereInput = {};

      if (input.status) {
        where.status = input.status;
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
        tags: z.array(z.string()).optional(),
      })
    )
    .mutation(async ({ ctx, input }) => {
      const db = ctx.db as PrismaClient;

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
      const data: Partial<Prisma.BlogPostUpdateInput> = {};

      if (input.title !== undefined) {
        data.title = input.title;

        // Re-generate slug only if the post is still a DRAFT
        if (
          input.title !== existing.title &&
          shouldRegenerateSlugOnTitleChange(existing.status)
        ) {
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
      if (input.tagIds !== undefined || input.tags !== undefined) {
        const resolved = await resolveTagIdsFromStrings(db, input.tags);
        const tagIds =
          input.tagIds !== undefined
            ? [...new Set([...(input.tagIds ?? []), ...resolved])]
            : resolved;

        await db.blogPostTag.deleteMany({ where: { postId: input.id } });
        if (tagIds.length) {
          await db.blogPostTag.createMany({
            data: tagIds.map((tagId) => ({ postId: input.id, tagId })),
          });
        }
      }

      return updated;
    }),

  delete: protectedProcedure
    .input(z.object({ id: z.string() }))
    .mutation(async ({ ctx, input }) => {
      const db = ctx.db as PrismaClient;
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
    const db = ctx.db as PrismaClient;
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

    return posts;
  }),

  listPublished: publicProcedure
    .input(ListInput)
    .query(async ({ ctx, input }) => {
      const db = ctx.db as PrismaClient;
      const { cursor, limit, authorId, query, tagSlugs, sort } = input;

      const where: Prisma.BlogPostWhereInput = {
        status: "PUBLISHED",
        ...(authorId ? { authorId } : {}),
      };

      if (query) {
        where.OR = [
          { title: { contains: query, mode: "insensitive" } },
          { content: { contains: query, mode: "insensitive" } },
          { excerpt: { contains: query, mode: "insensitive" } },
        ];
      }

      if (tagSlugs?.length) {
        where.tags = {
          some: { tag: { slug: { in: tagSlugs } } },
        };
      }
      const orderBy =
        sort === "top"
          ? [
              { upvotes: { _count: "desc" as const } },
              { publishedAt: "desc" as const },
              { id: "desc" as const },
            ]
          : [{ publishedAt: "desc" as const }, { id: "desc" as const }];

      const posts = await db.blogPost.findMany({
        where: cursor ? { ...where, id: { lt: cursor } } : where,
        take: limit,
        orderBy,
        include: {
          author: { select: { id: true, name: true, image: true } },
          _count: { select: { comments: true, upvotes: true } },
          tags: { include: { tag: true } },
        },
      });

      return { posts, nextCursor: toNextCursor(posts, limit) };
    }),

  listMine: protectedProcedure
    .input(
      z.object({
        cursor: z.string().nullish(),
        limit: z.number().min(1).max(50).default(20),
        status: StatusEnum.optional(),
        query: z.string().optional(),
        sort: SortEnum.default("recent"), // <-- NEW
      })
    )
    .query(async ({ ctx, input }) => {
      const db = ctx.db as PrismaClient;
      const { cursor, limit, status, query, sort } = input;

      const where: Prisma.BlogPostWhereInput = {
        authorId: ctx.session.user.id,
        ...(status ? { status } : {}),
      };
      if (query) {
        where.OR = [
          { title: { contains: query, mode: "insensitive" } },
          { content: { contains: query, mode: "insensitive" } },
          { excerpt: { contains: query, mode: "insensitive" } },
        ];
      }

      const orderBy =
        sort === "top"
          ? [
              { upvotes: { _count: "desc" as const } },
              { updatedAt: "desc" as const },
              { id: "desc" as const },
            ]
          : [{ updatedAt: "desc" as const }, { id: "desc" as const }];

      const posts = await db.blogPost.findMany({
        where: cursor ? { ...where, id: { lt: cursor } } : where,
        take: limit,
        orderBy,
        include: {
          author: { select: { id: true, name: true, image: true } },
          _count: { select: { comments: true, upvotes: true } },
        },
      });

      return { posts, nextCursor: toNextCursor(posts, limit) };
    }),

  createDraft: protectedProcedure
    .input(z.object({ title: z.string().default("Untitled draft") }))
    .mutation(async ({ ctx, input }) => {
      const db = ctx.db as PrismaClient;
      const post = await db.blogPost.create({
        data: {
          title: input.title.trim() || "Untitled draft",
          content: "",
          status: "DRAFT",
          authorId: ctx.session.user.id,
          // optional: excerpt: null,
        },
        select: { id: true, status: true, title: true },
      });
      return post;
    }),

  autosave: protectedProcedure
    .input(
      z.object({
        id: z.string(),
        title: z.string().optional(),
        content: z.string().optional(),
        excerpt: z.string().nullable().optional(),
        tagIds: z.array(z.string()).optional(),
        tags: z.array(z.string()).optional(),
      })
    )
    .mutation(async ({ ctx, input }) => {
      const db = ctx.db as PrismaClient;

      const post = await db.blogPost.findUnique({
        where: { id: input.id },
        select: { id: true, authorId: true, status: true, content: true },
      });
      if (!post) throw new TRPCError({ code: "NOT_FOUND" });
      if (post.authorId !== ctx.session.user.id)
        throw new TRPCError({ code: "FORBIDDEN" });

      const data: Partial<Prisma.BlogPostUpdateInput> = {};
      if (input.title !== undefined) data.title = input.title;
      if (input.content !== undefined) {
        data.content = input.content;
        data.readTimeMinutes = calculateReadTime(input.content);
      }
      if (input.excerpt !== undefined) data.excerpt = input.excerpt ?? null;

      const updated = await db.blogPost.update({
        where: { id: input.id },
        data,
        select: { id: true, title: true, updatedAt: true, status: true },
      });

      // --- Safe tag sync (idempotent, no unique-constraint races) ---
      if (input.tagIds !== undefined || input.tags !== undefined) {
        const resolvedFromStrings = await resolveTagIdsFromStrings(
          db,
          input.tags ?? []
        );
        const desiredTagIds = Array.from(
          new Set([...(input.tagIds ?? []), ...resolvedFromStrings])
        ).filter(Boolean);

        const tx: Array<
          ReturnType<typeof db.$transaction> extends Promise<infer _T>
            ? any
            : never
        > = [];

        // Remove any relations not in desired set (keeps already-linked ones)
        tx.push(
          db.blogPostTag.deleteMany({
            where: {
              postId: input.id,
              ...(desiredTagIds.length
                ? { tagId: { notIn: desiredTagIds } }
                : {}), // if empty desired -> delete all
            },
          })
        );

        // Create missing relations; skipDuplicates prevents unique constraint errors
        if (desiredTagIds.length) {
          tx.push(
            db.blogPostTag.createMany({
              data: desiredTagIds.map((tagId) => ({ postId: input.id, tagId })),
              skipDuplicates: true,
            })
          );
        }

        await db.$transaction(tx);
      }

      return updated;
    }),

  publish: protectedProcedure
    .input(z.object({ id: z.string(), desiredSlug: z.string().optional() }))
    .mutation(async ({ ctx, input }) => {
      const db = ctx.db as PrismaClient;
      const post = await db.blogPost.findUnique({
        where: { id: input.id },
        select: {
          id: true,
          authorId: true,
          title: true,
          content: true,
          slug: true,
          status: true,
          publishedAt: true,
        },
      });

      if (!post) throw new TRPCError({ code: "NOT_FOUND" });
      if (post.authorId !== ctx.session.user.id)
        throw new TRPCError({ code: "FORBIDDEN" });
      if (!post.title?.trim() || !post.content?.trim()) {
        throw new TRPCError({
          code: "BAD_REQUEST",
          message: "Title and content required to publish.",
        });
      }

      let slug = post.slug;
      if (!slug) {
        const base = input.desiredSlug?.trim().length
          ? input.desiredSlug
          : post.title;
        slug = await generateUniqueSlug(db, base);
      }

      const updated = await db.blogPost.update({
        where: { id: input.id },
        data: {
          status: "PUBLISHED",
          slug,
          publishedAt: post.publishedAt ?? new Date(),
        },
        select: { id: true, slug: true, status: true, publishedAt: true },
      });

      return updated;
    }),

  // ----- NEW: unpublish -> back to DRAFT, keeps slug (you may choose to clear) -----
  unpublish: protectedProcedure
    .input(z.object({ id: z.string() }))
    .mutation(async ({ ctx, input }) => {
      const db = ctx.db as PrismaClient;
      const post = await db.blogPost.findUnique({
        where: { id: input.id },
        select: { id: true, authorId: true },
      });
      if (!post) throw new TRPCError({ code: "NOT_FOUND" });
      if (post.authorId !== ctx.session.user.id)
        throw new TRPCError({ code: "FORBIDDEN" });

      return db.blogPost.update({
        where: { id: input.id },
        data: { status: "DRAFT" },
        select: { id: true, status: true },
      });
    }),

  // ----- NEW: archive -----
  archive: protectedProcedure
    .input(z.object({ id: z.string() }))
    .mutation(async ({ ctx, input }) => {
      const db = ctx.db as PrismaClient;
      const post = await db.blogPost.findUnique({
        where: { id: input.id },
        select: { id: true, authorId: true },
      });
      if (!post) throw new TRPCError({ code: "NOT_FOUND" });
      if (post.authorId !== ctx.session.user.id)
        throw new TRPCError({ code: "FORBIDDEN" });

      return db.blogPost.update({
        where: { id: input.id },
        data: { status: "ARCHIVED" },
        select: { id: true, status: true },
      });
    }),
});
