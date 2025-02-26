import { z } from "zod";
import { createTRPCRouter, protectedProcedure } from "../trpc";

export const userRouter = createTRPCRouter({
  updateGithubUsername: protectedProcedure
    .input(z.object({ accountId: z.string() }))
    .mutation(async ({ ctx, input }) => {
      const response = await fetch(
        `https://api.github.com/user/${input.accountId}`
      );
      const data = await response.json();

      await ctx.db.user.update({
        where: { id: ctx.session.user.id },
        data: { username: data.login },
      });

      return data.login;
    }),
});
