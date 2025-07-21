import { PrismaAdapter } from "@auth/prisma-adapter";
import type { DefaultSession, NextAuthOptions } from "next-auth";
import GithubProvider from "next-auth/providers/github";
import type { OAuthConfig, OAuthUserConfig } from "next-auth/providers/oauth";
import { type UserRole, type User } from "@prisma/client";

import { db } from "~/server/db";
import { env } from "~/env.mjs";

/**
 * Module augmentation for `next-auth` types. Allows us to add custom properties to the `session`
 * object and keep type safety.
 *
 * @see https://next-auth.js.org/getting-started/typescript#module-augmentation
 */
declare module "next-auth" {
  interface Session {
    user: {
      id: string;
      role: UserRole;
      username?: string | null;
      isAdmin?: boolean;
    } & DefaultSession["user"];
  }
  interface User {
    role: UserRole;
    username?: string | null;
    isAdmin?: boolean;
  }
}

declare module "next-auth/adapters" {
  interface AdapterUser extends User {
    role: UserRole;
  }
}

// Unused type with leading underscore to indicate it's not directly used
// but kept for documentation/reference purposes
type _GitHubUser = {
  id: number;
  login: string;
  avatar_url: string;
  name: string | null;
  email: string | null;
};

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
    async signIn({ user, account, profile }) {
      if (account?.provider === "github" && profile) {
        const githubProfile = profile as { login: string };
        user.username = githubProfile.login;
      }
      return true;
    },
    session({ session, user }) {
      if (session.user) {
        session.user.id = user.id;
        session.user.role = user.role;
        session.user.username = user.username;
        session.user.isAdmin = env.ADMIN_GITHUB_USERNAMES.includes(
          user.username ?? ""
        );
      }
      return session;
    },
  },
  events: {
    async createUser({ user }) {
      if (user.username) {
        await db.user.update({
          where: { id: user.id },
          data: { username: user.username },
        });
      }
    },
  },
} satisfies NextAuthOptions;

interface DiscordProfile {
  id: string;
  username: string;
  discriminator: string;
  avatar: string | null;
  email: string | null;
}

export const discordAuth = (
  config: OAuthUserConfig<DiscordProfile>
): OAuthConfig<DiscordProfile> => ({
  id: "discord",
  name: "Discord",
  type: "oauth",
  authorization:
    "https://discord.com/api/oauth2/authorize?scope=identify+email",
  token: "https://discord.com/api/oauth2/token",
  userinfo: "https://discord.com/api/users/@me",
  profile(profile) {
    return {
      id: profile.id,
      name: profile.username,
      email: profile.email,
      image: profile.avatar
        ? `https://cdn.discordapp.com/avatars/${profile.id}/${profile.avatar}.png`
        : null,
      role: "USER",
    };
  },
  ...config,
});
