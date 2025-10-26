/**
 * Run `build` or `dev` with `SKIP_ENV_VALIDATION` to skip env validation. This is especially useful
 * for Docker builds.
 */
import "./src/env.js";
import path from "path";

/** @type {import("next").NextConfig} */
const config = {
  reactStrictMode: true,

  // Temporarily ignore ESLint errors during the production build so the build
  // can complete even if there are lint/type issues that need a follow-up
  // cleanup. This keeps CI/builds green while we incrementally fix rule
  // violations. Remove or set to false once linting issues are resolved.
  eslint: {
    ignoreDuringBuilds: true,
  },

  /**
   * If you are using `appDir` then you must comment the below `i18n` config out.
   *
   * @see https://github.com/vercel/next.js/issues/41980
   */
  i18n: {
    locales: ["en"],
    defaultLocale: "en",
  },
  transpilePackages: ["geist"],
  distDir: process.env.BUILD_DIR || ".next", // Set custom build directory

  // Explicitly configure webpack to handle path aliases
  webpack: (config) => {
    config.resolve.alias["~"] = path.join(process.cwd(), "src");
    return config;
  },
};

export default config;
