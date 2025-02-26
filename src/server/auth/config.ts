import { PrismaAdapter } from "@auth/prisma-adapter";
import type { NextAuthOptions, DefaultSession } from "next-auth";
import GithubProvider from "next-auth/providers/github";

import { db } from "~/server/db";
import { env } from "~/env";

/**
 * Module augmentation for `next-auth` types. Allows us to add custom properties to the `session`
 * object and keep type safety.
 *
 * @see https://next-auth.js.org/getting-started/typescript#module-augmentation
 */
declare module "next-auth" {
  interface Session extends DefaultSession {
    user: {
      id: string;
      username?: string;
    } & DefaultSession["user"];
  }

  interface User {
    username?: string;
  }
}

/**
 * Options for NextAuth.js used to configure adapters, providers, callbacks, etc.
 *
 * @see https://next-auth.js.org/configuration/options
 */
export const authConfig: NextAuthOptions = {
  adapter: PrismaAdapter(db),
  providers: [
    GithubProvider({
      clientId: env.AUTH_GITHUB_ID,
      clientSecret: env.AUTH_GITHUB_SECRET,
    }),
    /**
     * ...add more providers here.
     *
     * Most other providers require a bit more work than the Discord provider. For example, the
     * GitHub provider requires you to add the `refresh_token_expires_in` field to the Account
     * model. Refer to the NextAuth.js docs for the provider you want to use. Example:
     *
     * @see https://next-auth.js.org/providers/github
     */
  ],
  session: {
    strategy: "database",
  },
  secret: process.env.AUTH_SECRET,
  callbacks: {
    signIn: async ({ user, account }) => {
      if (account?.provider === "github") {
        const response = await fetch(
          `https://api.github.com/user/${account.providerAccountId}`
        );
        const data = await response.json();

        await db.user.update({
          where: { id: user.id },
          data: { username: data.login },
        });
      }
      return true;
    },
    session: ({ session, user }) => ({
      ...session,
      user: {
        ...session.user,
        id: user.id,
        username: user.username,
      },
    }),
  },
} satisfies NextAuthOptions;
