import { z } from "zod";
import { TRPCError } from "@trpc/server";
import {
  createTRPCRouter,
  protectedProcedure,
} from "~/server/api/trpc";

const MAX_GROUP_SIZE = 256;

const slugSchema = z
  .string()
  .min(2)
  .max(48)
  .regex(/^[a-z0-9]+(?:-[a-z0-9]+)*$/, "Slug must be lowercase alphanumeric with hyphens");

async function assertGroupMembership(
  db: Parameters<Parameters<typeof protectedProcedure.query>[0]>["ctx"]["db"],
  groupId: string,
  userId: string
) {
  const member = await db.groupMember.findUnique({
    where: { groupId_userId: { groupId, userId } },
  });
  if (!member) {
    throw new TRPCError({ code: "FORBIDDEN", message: "You are not a member of this group" });
  }
  return member;
}

async function assertGroupAdmin(
  db: Parameters<Parameters<typeof protectedProcedure.query>[0]>["ctx"]["db"],
  groupId: string,
  userId: string
) {
  const member = await assertGroupMembership(db, groupId, userId);
  if (member.role !== "OWNER" && member.role !== "ADMIN") {
    throw new TRPCError({ code: "FORBIDDEN", message: "Only owners and admins can perform this action" });
  }
  return member;
}

async function assertGroupOwner(
  db: Parameters<Parameters<typeof protectedProcedure.query>[0]>["ctx"]["db"],
  groupId: string,
  userId: string
) {
  const member = await assertGroupMembership(db, groupId, userId);
  if (member.role !== "OWNER") {
    throw new TRPCError({ code: "FORBIDDEN", message: "Only the group owner can perform this action" });
  }
  return member;
}

export const groupsRouter = createTRPCRouter({
  create: protectedProcedure
    .input(
      z.object({
        name: z.string().min(1).max(100),
        slug: slugSchema,
        description: z.string().max(500).optional(),
      })
    )
    .mutation(async ({ ctx, input }) => {
      const existing = await ctx.db.group.findUnique({ where: { slug: input.slug } });
      if (existing) {
        throw new TRPCError({ code: "CONFLICT", message: "A group with this slug already exists" });
      }

      const group = await ctx.db.group.create({
        data: {
          name: input.name,
          slug: input.slug,
          description: input.description,
          members: {
            create: {
              userId: ctx.session.user.id,
              role: "OWNER",
            },
          },
        },
        include: { _count: { select: { members: true, problems: true } } },
      });

      return group;
    }),

  getMyGroups: protectedProcedure.query(async ({ ctx }) => {
    const memberships = await ctx.db.groupMember.findMany({
      where: { userId: ctx.session.user.id },
      include: {
        group: {
          include: {
            _count: { select: { members: true, problems: true } },
          },
        },
      },
      orderBy: { joinedAt: "desc" },
    });

    return memberships.map((m) => ({
      ...m.group,
      role: m.role,
      joinedAt: m.joinedAt,
    }));
  }),

  getBySlug: protectedProcedure
    .input(z.object({ slug: z.string() }))
    .query(async ({ ctx, input }) => {
      const group = await ctx.db.group.findUnique({
        where: { slug: input.slug },
        include: { _count: { select: { members: true, problems: true } } },
      });

      if (!group) {
        throw new TRPCError({ code: "NOT_FOUND", message: "Group not found" });
      }

      const membership = await ctx.db.groupMember.findUnique({
        where: { groupId_userId: { groupId: group.id, userId: ctx.session.user.id } },
      });

      if (!membership) {
        throw new TRPCError({ code: "FORBIDDEN", message: "You are not a member of this group" });
      }

      return { ...group, currentUserRole: membership.role };
    }),

  getMembers: protectedProcedure
    .input(z.object({ groupSlug: z.string() }))
    .query(async ({ ctx, input }) => {
      const group = await ctx.db.group.findUnique({ where: { slug: input.groupSlug } });
      if (!group) {
        throw new TRPCError({ code: "NOT_FOUND", message: "Group not found" });
      }

      await assertGroupMembership(ctx.db, group.id, ctx.session.user.id);

      const members = await ctx.db.groupMember.findMany({
        where: { groupId: group.id },
        include: {
          user: {
            select: { id: true, username: true, name: true, image: true },
          },
        },
        orderBy: [{ role: "asc" }, { joinedAt: "asc" }],
      });

      return members;
    }),

  addMember: protectedProcedure
    .input(z.object({ groupSlug: z.string(), username: z.string() }))
    .mutation(async ({ ctx, input }) => {
      const group = await ctx.db.group.findUnique({
        where: { slug: input.groupSlug },
        include: { _count: { select: { members: true } } },
      });
      if (!group) {
        throw new TRPCError({ code: "NOT_FOUND", message: "Group not found" });
      }

      await assertGroupAdmin(ctx.db, group.id, ctx.session.user.id);

      if (group._count.members >= MAX_GROUP_SIZE) {
        throw new TRPCError({
          code: "PRECONDITION_FAILED",
          message: `Group has reached the maximum size of ${MAX_GROUP_SIZE} members`,
        });
      }

      const targetUser = await ctx.db.user.findFirst({ where: { username: input.username } });
      if (!targetUser) {
        throw new TRPCError({ code: "NOT_FOUND", message: "User not found" });
      }

      const existingMember = await ctx.db.groupMember.findUnique({
        where: { groupId_userId: { groupId: group.id, userId: targetUser.id } },
      });
      if (existingMember) {
        throw new TRPCError({ code: "CONFLICT", message: "User is already a member of this group" });
      }

      const member = await ctx.db.groupMember.create({
        data: { groupId: group.id, userId: targetUser.id, role: "MEMBER" },
        include: { user: { select: { id: true, username: true, name: true, image: true } } },
      });

      return member;
    }),

  removeMember: protectedProcedure
    .input(z.object({ groupSlug: z.string(), userId: z.string() }))
    .mutation(async ({ ctx, input }) => {
      const group = await ctx.db.group.findUnique({ where: { slug: input.groupSlug } });
      if (!group) {
        throw new TRPCError({ code: "NOT_FOUND", message: "Group not found" });
      }

      await assertGroupAdmin(ctx.db, group.id, ctx.session.user.id);

      const targetMember = await ctx.db.groupMember.findUnique({
        where: { groupId_userId: { groupId: group.id, userId: input.userId } },
      });
      if (!targetMember) {
        throw new TRPCError({ code: "NOT_FOUND", message: "Member not found" });
      }
      if (targetMember.role === "OWNER") {
        throw new TRPCError({ code: "FORBIDDEN", message: "Cannot remove the group owner" });
      }

      await ctx.db.groupMember.delete({
        where: { groupId_userId: { groupId: group.id, userId: input.userId } },
      });

      return { success: true };
    }),

  updateMemberRole: protectedProcedure
    .input(
      z.object({
        groupSlug: z.string(),
        userId: z.string(),
        role: z.enum(["ADMIN", "MEMBER"]),
      })
    )
    .mutation(async ({ ctx, input }) => {
      const group = await ctx.db.group.findUnique({ where: { slug: input.groupSlug } });
      if (!group) {
        throw new TRPCError({ code: "NOT_FOUND", message: "Group not found" });
      }

      await assertGroupOwner(ctx.db, group.id, ctx.session.user.id);

      const targetMember = await ctx.db.groupMember.findUnique({
        where: { groupId_userId: { groupId: group.id, userId: input.userId } },
      });
      if (!targetMember) {
        throw new TRPCError({ code: "NOT_FOUND", message: "Member not found" });
      }
      if (targetMember.role === "OWNER") {
        throw new TRPCError({ code: "FORBIDDEN", message: "Cannot change the owner's role" });
      }

      const updated = await ctx.db.groupMember.update({
        where: { groupId_userId: { groupId: group.id, userId: input.userId } },
        data: { role: input.role },
      });

      return updated;
    }),

  leaveGroup: protectedProcedure
    .input(z.object({ groupSlug: z.string() }))
    .mutation(async ({ ctx, input }) => {
      const group = await ctx.db.group.findUnique({ where: { slug: input.groupSlug } });
      if (!group) {
        throw new TRPCError({ code: "NOT_FOUND", message: "Group not found" });
      }

      const member = await assertGroupMembership(ctx.db, group.id, ctx.session.user.id);

      if (member.role === "OWNER") {
        throw new TRPCError({
          code: "PRECONDITION_FAILED",
          message: "The owner cannot leave. Transfer ownership or delete the group.",
        });
      }

      await ctx.db.groupMember.delete({
        where: { groupId_userId: { groupId: group.id, userId: ctx.session.user.id } },
      });

      return { success: true };
    }),

  deleteGroup: protectedProcedure
    .input(z.object({ groupSlug: z.string() }))
    .mutation(async ({ ctx, input }) => {
      const group = await ctx.db.group.findUnique({ where: { slug: input.groupSlug } });
      if (!group) {
        throw new TRPCError({ code: "NOT_FOUND", message: "Group not found" });
      }

      await assertGroupOwner(ctx.db, group.id, ctx.session.user.id);

      await ctx.db.group.delete({ where: { id: group.id } });

      return { success: true };
    }),

  addProblem: protectedProcedure
    .input(z.object({ groupSlug: z.string(), problemSlug: z.string() }))
    .mutation(async ({ ctx, input }) => {
      const group = await ctx.db.group.findUnique({ where: { slug: input.groupSlug } });
      if (!group) {
        throw new TRPCError({ code: "NOT_FOUND", message: "Group not found" });
      }

      await assertGroupAdmin(ctx.db, group.id, ctx.session.user.id);

      const problem = await ctx.db.problem.findUnique({ where: { slug: input.problemSlug } });
      if (!problem) {
        throw new TRPCError({ code: "NOT_FOUND", message: "Problem not found" });
      }

      const existing = await ctx.db.groupProblem.findUnique({
        where: { groupId_problemId: { groupId: group.id, problemId: problem.id } },
      });
      if (existing) {
        throw new TRPCError({ code: "CONFLICT", message: "Problem is already in this group" });
      }

      const groupProblem = await ctx.db.groupProblem.create({
        data: {
          groupId: group.id,
          problemId: problem.id,
          addedById: ctx.session.user.id,
        },
        include: {
          problem: {
            select: { id: true, title: true, slug: true, difficulty: true, tags: true },
          },
        },
      });

      return groupProblem;
    }),

  removeProblem: protectedProcedure
    .input(z.object({ groupSlug: z.string(), problemSlug: z.string() }))
    .mutation(async ({ ctx, input }) => {
      const group = await ctx.db.group.findUnique({ where: { slug: input.groupSlug } });
      if (!group) {
        throw new TRPCError({ code: "NOT_FOUND", message: "Group not found" });
      }

      await assertGroupAdmin(ctx.db, group.id, ctx.session.user.id);

      const problem = await ctx.db.problem.findUnique({ where: { slug: input.problemSlug } });
      if (!problem) {
        throw new TRPCError({ code: "NOT_FOUND", message: "Problem not found" });
      }

      await ctx.db.groupProblem.delete({
        where: { groupId_problemId: { groupId: group.id, problemId: problem.id } },
      });

      return { success: true };
    }),

  getProblems: protectedProcedure
    .input(z.object({ groupSlug: z.string() }))
    .query(async ({ ctx, input }) => {
      const group = await ctx.db.group.findUnique({ where: { slug: input.groupSlug } });
      if (!group) {
        throw new TRPCError({ code: "NOT_FOUND", message: "Group not found" });
      }

      await assertGroupMembership(ctx.db, group.id, ctx.session.user.id);

      const memberUserIds = (
        await ctx.db.groupMember.findMany({
          where: { groupId: group.id },
          select: { userId: true },
        })
      ).map((m) => m.userId);

      const totalMembers = memberUserIds.length;

      const groupProblems = await ctx.db.groupProblem.findMany({
        where: { groupId: group.id },
        include: {
          problem: {
            select: {
              id: true,
              title: true,
              slug: true,
              difficulty: true,
              tags: true,
            },
          },
        },
        orderBy: { addedAt: "asc" },
      });

      const problemIds = groupProblems.map((gp) => gp.problem.id);

      // Get accepted submissions from group members for these problems
      const acceptedSubmissions = await ctx.db.submission.findMany({
        where: {
          problemId: { in: problemIds },
          userId: { in: memberUserIds },
          status: "ACCEPTED",
        },
        select: { problemId: true, userId: true },
        distinct: ["problemId", "userId"],
      });

      // Build a map of problemId -> set of userIds who solved it
      const solvedMap = new Map<string, Set<string>>();
      for (const sub of acceptedSubmissions) {
        if (!solvedMap.has(sub.problemId)) {
          solvedMap.set(sub.problemId, new Set());
        }
        solvedMap.get(sub.problemId)!.add(sub.userId);
      }

      return groupProblems.map((gp) => {
        const solvers = solvedMap.get(gp.problem.id);
        return {
          ...gp.problem,
          addedAt: gp.addedAt,
          solvedCount: solvers?.size ?? 0,
          totalMembers,
          solvedByCurrentUser: solvers?.has(ctx.session.user.id) ?? false,
        };
      });
    }),

  getProblemLeaderboard: protectedProcedure
    .input(
      z.object({
        groupSlug: z.string(),
        problemSlug: z.string(),
        gpuType: z.string().optional().default("all"),
      })
    )
    .query(async ({ ctx, input }) => {
      const group = await ctx.db.group.findUnique({ where: { slug: input.groupSlug } });
      if (!group) {
        throw new TRPCError({ code: "NOT_FOUND", message: "Group not found" });
      }

      await assertGroupMembership(ctx.db, group.id, ctx.session.user.id);

      const problem = await ctx.db.problem.findUnique({ where: { slug: input.problemSlug } });
      if (!problem) {
        throw new TRPCError({ code: "NOT_FOUND", message: "Problem not found" });
      }

      const memberUserIds = (
        await ctx.db.groupMember.findMany({
          where: { groupId: group.id },
          select: { userId: true },
        })
      ).map((m) => m.userId);

      const submissions = await ctx.db.submission.findMany({
        where: {
          problemId: problem.id,
          userId: { in: memberUserIds },
          status: "ACCEPTED",
          runtime: { not: null },
          ...(input.gpuType !== "all" ? { gpuType: input.gpuType } : {}),
        },
        select: {
          id: true,
          runtime: true,
          gflops: true,
          gpuType: true,
          language: true,
          createdAt: true,
          isPublic: true,
          user: {
            select: { id: true, username: true, image: true },
          },
        },
        orderBy: { runtime: "asc" },
      });

      // Best submission per user (lowest runtime)
      const userBestMap = new Map<string, (typeof submissions)[0]>();
      for (const sub of submissions) {
        const current = userBestMap.get(sub.user.id);
        if (!current || (sub.runtime ?? Infinity) < (current.runtime ?? Infinity)) {
          userBestMap.set(sub.user.id, sub);
        }
      }

      return Array.from(userBestMap.values())
        .sort((a, b) => (a.runtime ?? Infinity) - (b.runtime ?? Infinity))
        .map((sub, idx) => ({
          rank: idx + 1,
          submissionId: sub.id,
          username: sub.user.username,
          image: sub.user.image,
          runtime: sub.runtime,
          gflops: sub.gflops,
          gpuType: sub.gpuType,
          language: sub.language,
          createdAt: sub.createdAt,
          isPublic: sub.isPublic,
        }));
    }),
});
