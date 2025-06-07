import { App } from "@octokit/app";
import type { Octokit } from "@octokit/rest"; // Use import type
import type { RestEndpointMethodTypes } from "@octokit/plugin-rest-endpoint-methods"; // Add this import

import { env } from "~/env";

const GITHUB_APP_ID = env.GITHUB_APP_ID;
const GITHUB_APP_PRIVATE_KEY = env.GITHUB_APP_PRIVATE_KEY;
const GITHUB_APP_INSTALLATION_ID = env.GITHUB_APP_INSTALLATION_ID;

const app = new App({
  appId: GITHUB_APP_ID,
  privateKey: GITHUB_APP_PRIVATE_KEY,
});

let octokitInstance: Octokit | null = null;

export async function getOctokitInstance(): Promise<Octokit> {
  if (octokitInstance) {
    return octokitInstance;
  }

  const installationId = parseInt(GITHUB_APP_INSTALLATION_ID!);
  octokitInstance = (await app.getInstallationOctokit(
    installationId
  )) as Octokit; // Explicit cast

  return octokitInstance;
}

export async function createBranch(
  owner: string,
  repo: string,
  baseBranch: string,
  newBranch: string
): Promise<void> {
  const octokit = await getOctokitInstance();
  const { data: baseBranchRef } = await octokit.git.getRef({
    owner,
    repo,
    ref: `heads/${baseBranch}`,
  });

  await octokit.git.createRef({
    owner,
    repo,
    ref: `refs/heads/${newBranch}`,
    sha: baseBranchRef.object.sha,
  });
}

export async function createOrUpdateFile(
  owner: string,
  repo: string,
  branch: string,
  path: string,
  content: string,
  message: string
): Promise<void> {
  const octokit = await getOctokitInstance();
  let sha: string | undefined;

  try {
    const { data: file } = await octokit.repos.getContent({
      owner,
      repo,
      path,
      ref: branch,
    });
    if (Array.isArray(file)) {
      throw new Error("Path is a directory, not a file.");
    }
    sha = file.sha;
  } catch (error: unknown) {
    // Change to unknown
    if (
      typeof error === "object" &&
      error !== null &&
      "status" in error &&
      (error as { status: number }).status !== 404
    ) {
      // Type guard for status
      if (error instanceof Error) {
        throw error;
      }
      throw new Error("An unknown error occurred during GitHub API call.");
    }
    // File does not exist, so sha remains undefined for creation
  }

  await octokit.repos.createOrUpdateFileContents({
    owner,
    repo,
    path,
    message,
    content: Buffer.from(content).toString("base64"),
    branch,
    sha,
  });
}

export async function createPullRequest(
  owner: string,
  repo: string,
  head: string,
  base: string,
  title: string,
  body: string
): Promise<string> {
  const octokit = await getOctokitInstance();
  const { data: pullRequest } = await octokit.pulls.create({
    owner,
    repo,
    head,
    base,
    title,
    body,
  });
  return pullRequest.html_url;
}

export async function getFileContent(
  owner: string,
  repo: string,
  path: string,
  ref: string
): Promise<string | null> {
  const octokit = await getOctokitInstance();
  try {
    const { data: file } = await octokit.repos.getContent({
      owner,
      repo,
      path,
      ref,
    });
    if (Array.isArray(file)) {
      throw new Error("Path is a directory, not a file.");
    }
    if ("content" in file && file.content) {
      return Buffer.from(file.content, "base64").toString("utf8");
    }
    return null;
  } catch (error: unknown) {
    // Change to unknown
    if (
      typeof error === "object" &&
      error !== null &&
      "status" in error &&
      (error as { status: number }).status === 404
    ) {
      // Type guard for status
      return null;
    }
    throw error;
  }
}

export async function listPullRequests(
  owner: string,
  repo: string,
  author?: string,
  state?: "open" | "closed" | "all"
): Promise<RestEndpointMethodTypes["pulls"]["list"]["response"]["data"]> {
  // Stronger type
  const octokit = await getOctokitInstance();
  const params: {
    owner: string;
    repo: string;
    state?: "open" | "closed" | "all";
    creator?: string;
  } = {
    owner,
    repo,
    state: state ?? "all", // Use nullish coalescing
  };
  if (author) {
    params.creator = author;
  }
  const { data: pullRequests } = await octokit.pulls.list(params);
  return pullRequests;
}
