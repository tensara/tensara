/**
 * Run `build` or `dev` with `SKIP_ENV_VALIDATION` to skip env validation. This is especially useful
 * for Docker builds.
 */
import "./src/env.js";
import path from 'path';

const isProd = process.env.NODE_ENV === "production";

/** @type {import("next").NextConfig} */
const config = {
  reactStrictMode: true,

  /**
   * If you are using `appDir` then you must comment the below `i18n` config out.
   *
   * @see https://github.com/vercel/next.js/issues/41980
   */
  i18n: {
    locales: ["en"],
    defaultLocale: "en",
  },
  transpilePackages: [
    "geist",
    "@octokit/app",
    "@octokit/auth-app",
    "@octokit/auth-oauth-app",
    "@octokit/auth-oauth-device",
    "@octokit/auth-oauth-user",
    "@octokit/auth-token",
    "@octokit/auth-unauthenticated",
    "@octokit/core",
    "@octokit/endpoint",
    "@octokit/graphql",
    "@octokit/oauth-app",
    "@octokit/oauth-authorization-url",
    "@octokit/oauth-methods",
    "@octokit/openapi-types",
    "@octokit/plugin-paginate-rest",
    "@octokit/plugin-request-log",
    "@octokit/plugin-rest-endpoint-methods",
    "@octokit/request",
    "@octokit/request-error",
    "@octokit/rest",
    "@octokit/types",
    "@octokit/webhooks",
    "@octokit/webhooks-methods",
    "@octokit/webhooks-types",
  ],
  distDir: process.env.BUILD_DIR || ".next", // Set custom build directory
  ...(isProd && {
    experimental: {
      esmExternals: "loose",
    },
  }),
};

export default config;
