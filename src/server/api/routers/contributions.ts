import { createTRPCRouter, protectedProcedure } from "~/server/api/trpc";
import { z } from "zod";
import { TRPCError } from "@trpc/server";

// Schema for contribution data
const contributionSchema = z.object({
  id: z.string(),
  title: z.string(),
  description: z.string(),
  difficulty: z.enum(["easy", "medium", "hard"]),
  referenceCode: z.string(),
  testCases: z.string(),
  prUrl: z.string().optional(),
  createdAt: z.date(),
  updatedAt: z.date(),
});

export const contributionsRouter = createTRPCRouter({
  create: protectedProcedure
    .input(
      z.object({
        title: z.string().min(5),
        description: z.string().min(20),
        difficulty: z.enum(["easy", "medium", "hard"]),
        referenceCode: z.string().min(10),
        testCases: z.string().min(10),
      })
    )
    .mutation(async ({ input, ctx }) => {
      if (!ctx.session.user) {
        throw new TRPCError({
          code: "UNAUTHORIZED",
          message: "You must be logged in to submit a problem",
        });
      }

      try {
        // Send to tensara-bot to create PR
        const response = await fetch("http://localhost:3001/create-pr", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            ...input,
            userId: ctx.session.user.id,
          }),
        });

        if (!response.ok) {
          const error = await response.json();
          throw new Error(error.error || "Failed to create PR");
        }

        const result = await response.json();

        return {
          id: `contribution-${Date.now()}`,
          title: input.title,
          description: input.description,
          difficulty: input.difficulty,
          referenceCode: input.referenceCode,
          testCases: input.testCases,
          prUrl: result.prUrl,
          createdAt: new Date(),
          updatedAt: new Date(),
        };
      } catch (error) {
        throw new TRPCError({
          code: "INTERNAL_SERVER_ERROR",
          message:
            error instanceof Error
              ? error.message
              : "Failed to create contribution",
        });
      }
    }),

  get: protectedProcedure
    .input(z.object({ id: z.string() }))
    .query(async ({ input, ctx }) => {
      // Fetch single contribution - placeholder
      return {
        id: input.id,
        title: "Sample Problem",
        description: "Sample description",
        difficulty: "medium",
        referenceCode: "// Sample code",
        testCases: "// Sample tests",
        prUrl: "https://github.com/tensara/tensara-problems/pull/123",
        createdAt: new Date(),
        updatedAt: new Date(),
      };
    }),

  update: protectedProcedure
    .input(
      z.object({
        id: z.string(),
        title: z.string().min(5).optional(),
        description: z.string().min(20).optional(),
        difficulty: z.enum(["easy", "medium", "hard"]).optional(),
        referenceCode: z.string().min(10).optional(),
        testCases: z.string().min(10).optional(),
      })
    )
    .mutation(async ({ input, ctx }) => {
      if (!ctx.session.user) {
        throw new TRPCError({
          code: "UNAUTHORIZED",
          message: "You must be logged in to modify a contribution",
        });
      }

      // Placeholder implementation
      return {
        success: true,
        message: "Contribution updated successfully",
      };
    }),

  list: protectedProcedure
    .output(
      z.array(
        contributionSchema.pick({
          id: true,
          title: true,
          difficulty: true,
          prUrl: true,
          createdAt: true,
        })
      )
    )
    .query(async ({ ctx }) => {
      // Fetch user's contributions from DB - placeholder
      return [
        {
          id: "sample-id",
          title: "Sample Problem",
          difficulty: "medium",
          prUrl: "https://github.com/tensara/tensara-problems/pull/123",
          createdAt: new Date(),
        },
      ];
    }),
});
