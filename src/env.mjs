import { createEnv } from "@t3-oss/env-nextjs";
import { z } from "zod";

export const env = createEnv({
  /**
   * Specify your server-side environment variables schema here. This way you can ensure the app isn't
   * built with invalid env vars.
   */
  server: {
    DATABASE_URL: z.string().url(),
    AUTH_GITHUB_ID: z.string(),
    AUTH_GITHUB_SECRET: z.string(),
    AUTH_SECRET: z.string().optional(),
    MODAL_CHECKER_SLUG: z.string(),
    MODAL_BENCHMARK_SLUG: z.string(),
  },

  /**
   * Specify your client-side environment variables schema here.
   * For them to be exposed to the client, prefix them with `NEXT_PUBLIC_`.
   */
  client: {
    NEXT_PUBLIC_GA_ID: z.string().min(1),
  },

  /**
   * You can't destruct `process.env` as a regular object in the Next.js edge runtimes (e.g.
   * middlewares) or client-side so we need to destruct manually.
   */
  runtimeEnv: {
    DATABASE_URL: process.env.DATABASE_URL,
    AUTH_GITHUB_ID: process.env.AUTH_GITHUB_ID,
    AUTH_GITHUB_SECRET: process.env.AUTH_GITHUB_SECRET,
    AUTH_SECRET: process.env.AUTH_SECRET,
    NEXT_PUBLIC_GA_ID: process.env.NEXT_PUBLIC_GA_ID,
    MODAL_CHECKER_SLUG: process.env.MODAL_CHECKER_SLUG,
    MODAL_BENCHMARK_SLUG: process.env.MODAL_BENCHMARK_SLUG,
  },
  skipValidation: !!process.env.SKIP_ENV_VALIDATION,
}); 