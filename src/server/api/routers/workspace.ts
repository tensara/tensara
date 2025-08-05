import { z } from "zod";
import { createTRPCRouter, protectedProcedure } from "../trpc";
import { TRPCError } from "@trpc/server";
import slugify from "slugify";

export const workspaceRouter = createTRPCRouter({
  list: protectedProcedure.query(async ({ ctx }) => {
    return ctx.db.workspace.findMany({
      where: { userId: ctx.session.user.id },
      orderBy: { createdAt: "desc" },
    });
  }),
  getAll: protectedProcedure.query(async ({ ctx }) => {
    try {
      const workspaces = await ctx.db.workspace.findMany({
        where: { userId: ctx.session.user.id },
        orderBy: { updatedAt: "desc" },
        select: {
          id: true,
          name: true,
          slug: true,
          createdAt: true,
          updatedAt: true,
        },
      });

      return workspaces;
    } catch (error) {
      console.error("Error fetching workspaces:", error);
      throw new TRPCError({
        code: "INTERNAL_SERVER_ERROR",
        message: "Failed to fetch workspaces",
        cause: error,
      });
    }
  }),

  create: protectedProcedure
    .input(z.object({ name: z.string().min(3) }))
    .mutation(async ({ ctx, input }) => {
      try {
        const slug = slugify(input.name, { lower: true, strict: true });

        const existing = await ctx.db.workspace.findUnique({
          where: { slug, userId: ctx.session.user.id },
        });
        if (existing) {
          throw new TRPCError({
            code: "CONFLICT",
            message: "A workspace with this name already exists.",
          });
        }

        const workspace = await ctx.db.workspace.create({
          data: {
            name: input.name,
            slug,
            userId: ctx.session.user.id,
            files: [
              {
                name: "main.cu",
                content: `#include <stdio.h>\n
__global__ void hello() {
  printf("Hello, world!\\n");
}
int main() {
  hello<<<1, 1>>>();
  cudaDeviceSynchronize();
  return 0;
}`,
              },
            ],
            main: "main.cu",
          },
        });

        return {
          id: workspace.id,
          name: workspace.name,
          slug: workspace.slug,
          createdAt: workspace.createdAt,
          updatedAt: workspace.updatedAt,
        };
      } catch (error) {
        if (error instanceof TRPCError) throw error;
        console.error("Error creating workspace:", error);

        throw new TRPCError({
          code: "INTERNAL_SERVER_ERROR",
          message: "Failed to create workspace",
          cause: error,
        });
      }
    }),

  getBySlug: protectedProcedure
    .input(
      z.object({
        username: z.string(),
        slug: z.string(),
      })
    )
    .query(async ({ ctx, input }) => {
      const workspace = await ctx.db.workspace.findFirst({
        where: {
          slug: input.slug,
          user: {
            username: {
              equals: input.username,
              mode: "insensitive",
            },
          },
        },
      });

      if (!workspace) {
        throw new TRPCError({
          code: "NOT_FOUND",
          message: "Workspace not found",
        });
      }

      // Check that the current user is the owner
      if (workspace.userId !== ctx.session.user.id) {
        throw new TRPCError({
          code: "FORBIDDEN",
          message: "Not your workspace",
        });
      }

      return workspace;
    }),

  update: protectedProcedure
    .input(
      z.object({
        id: z.string(),
        files: z.array(z.object({ name: z.string(), content: z.string() })),
        main: z.string(),
      })
    )
    .mutation(async ({ ctx, input }) => {
      const workspace = await ctx.db.workspace.findUnique({
        where: { id: input.id },
      });

      if (!workspace) {
        throw new TRPCError({
          code: "NOT_FOUND",
          message: "Workspace not found",
        });
      }

      if (workspace.userId !== ctx.session.user.id) {
        throw new TRPCError({
          code: "FORBIDDEN",
          message: "Not your workspace",
        });
      }

      await ctx.db.workspace.update({
        where: { id: input.id },
        data: {
          files: input.files,
          main: input.main,
        },
      });

      return { success: true };
    }),

  delete: protectedProcedure
    .input(z.object({ id: z.string() }))
    .mutation(async ({ ctx, input }) => {
      const workspace = await ctx.db.workspace.findUnique({
        where: { id: input.id },
      });

      if (!workspace) {
        throw new TRPCError({
          code: "NOT_FOUND",
          message: "Workspace not found",
        });
      }

      if (workspace.userId !== ctx.session.user.id) {
        throw new TRPCError({
          code: "FORBIDDEN",
          message: "You can only delete your own workspaces",
        });
      }

      await ctx.db.workspace.delete({ where: { id: input.id } });

      return { success: true };
    }),
  rename: protectedProcedure
    .input(z.object({ id: z.string(), name: z.string().min(1) }))
    .mutation(async ({ ctx, input }) => {
      const newSlug = slugify(input.name, { lower: true, strict: true });

      const existing = await ctx.db.workspace.findFirst({
        where: {
          slug: newSlug,
          userId: ctx.session.user.id,
          NOT: { id: input.id },
        },
      });

      if (existing) {
        throw new TRPCError({
          code: "CONFLICT",
          message: "A workspace with this name already exists.",
        });
      }

      const updated = await ctx.db.workspace.update({
        where: {
          id: input.id,
          userId: ctx.session.user.id,
        },
        data: {
          name: input.name,
          slug: newSlug,
          updatedAt: new Date(),
        },
        select: {
          id: true,
          name: true,
          slug: true,
          updatedAt: true,
        },
      });

      return updated;
    }),

  createSnapshot: protectedProcedure
    .input(z.object({ id: z.string() }))
    .mutation(async ({ ctx, input }) => {
      const workspace = await ctx.db.workspace.findUnique({
        where: { id: input.id },
      });

      if (!workspace) {
        throw new TRPCError({
          code: "NOT_FOUND",
          message: "Workspace not found",
        });
      }

      if (workspace.userId !== ctx.session.user.id) {
        throw new TRPCError({
          code: "FORBIDDEN",
          message: "Not your workspace",
        });
      }
      const snapshot = await ctx.db.snapshot.create({
        data: {
          files: workspace.files ?? [],
          main: workspace.main,
          workspaceId: workspace.id,
          userId: ctx.session.user.id,
        },
      });

      return snapshot;
    }),
});
