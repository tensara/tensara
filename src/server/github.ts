import { App } from "@octokit/app";
import { Octokit } from "@octokit/rest";
import type { RestEndpointMethodTypes } from "@octokit/plugin-rest-endpoint-methods";

import { env } from "~/env";

const GITHUB_APP_ID = env.GITHUB_APP_ID;
const GITHUB_APP_PRIVATE_KEY = env.GITHUB_APP_PRIVATE_KEY;
const GITHUB_APP_INSTALLATION_ID = env.GITHUB_APP_INSTALLATION_ID;

let octokitInstance: Octokit | null = null;

const app = new App({
  appId: GITHUB_APP_ID,
  privateKey: GITHUB_APP_PRIVATE_KEY,
  Octokit: Octokit, // Use the Octokit class from "@octokit/rest"
});

export async function getOctokitInstance(): Promise<Octokit> {
  if (octokitInstance) {
    return octokitInstance;
  }

  if (!GITHUB_APP_INSTALLATION_ID) {
    throw new Error(
      "GITHUB_APP_INSTALLATION_ID is not defined in environment variables."
    );
  }
  const installationIdString = GITHUB_APP_INSTALLATION_ID;
  const installationId = parseInt(installationIdString);
  if (isNaN(installationId)) {
    throw new Error(
      `Invalid GITHUB_APP_INSTALLATION_ID: "${installationIdString}" is not a valid integer.`
    );
  }

  // getInstallationOctokit returns an Octokit instance directly
  octokitInstance = await app.getInstallationOctokit(installationId);

  return octokitInstance;
}

export async function createBranch(
  owner: string,
  repo: string,
  baseBranch: string,
  newBranch: string
): Promise<void> {
  const octokit = await getOctokitInstance();
  let baseBranchRefData;
  try {
    const response = await octokit.git.getRef({
      owner,
      repo,
      ref: `heads/${baseBranch}`,
    });
    baseBranchRefData = response.data;
  } catch (error: any) {
    if (error.status === 404) {
      const errorMessage = `Failed to get reference for base branch '${baseBranch}' in repository '${owner}/${repo}'. The ref 'heads/${baseBranch}' was not found.
Please verify:
1. The repository '${owner}/${repo}' exists and the GitHub App has access.
2. The branch '${baseBranch}' (e.g., 'main', 'master') exists in '${owner}/${repo}'.
3. The GitHub App has 'Read' access to 'Contents' permission for this repository.
Original error: ${error.message}`;
      console.error(errorMessage, error);
      // It's often better to throw a new error with a more specific message,
      // or rethrow the original if it's already descriptive enough from Octokit.
      // For this example, we'll throw a new, more detailed error.
      throw new Error(errorMessage);
    }
    // Log and re-throw other unexpected errors
    console.error(
      `An unexpected error occurred while trying to get ref 'heads/${baseBranch}' from '${owner}/${repo}':`,
      error
    );
    throw error;
  }

  try {
    await octokit.git.createRef({
      owner,
      repo,
      ref: `refs/heads/${newBranch}`,
      sha: baseBranchRefData.object.sha,
    });
  } catch (error: any) {
    // Log error during branch creation
    console.error(
      `Failed to create new branch '${newBranch}' in '${owner}/${repo}' from base SHA '${baseBranchRefData.object.sha}':`,
      error
    );
    throw error;
  }
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
    if (
      typeof error === "object" &&
      error !== null &&
      "status" in error &&
      typeof (error as { status?: unknown }).status === "number"
    ) {
      const statusError = error as { status: number; message?: string };
      if (statusError.status !== 404) {
        // Re-throw if it's not a "file not found" error
        const errorMessage =
          statusError.message || "Failed to check file existence";
        throw new Error(
          `GitHub API error (status ${statusError.status}): ${errorMessage}`
        );
      }
      // If status is 404, it means the file doesn't exist.
      // sha remains undefined, which is correct for creating a new file.
    } else if (error instanceof Error) {
      // Re-throw other standard errors (e.g., network issues before a status code is assigned)
      throw error;
    } else {
      // For truly unknown errors
      throw new Error(
        "An unknown error occurred while checking for file existence."
      );
    }
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
    if (
      typeof error === "object" &&
      error !== null &&
      "status" in error &&
      typeof (error as { status?: unknown }).status === "number"
    ) {
      const statusError = error as { status: number; message?: string };
      if (statusError.status === 404) {
        return null; // File not found, return null as per function's contract
      }
      // For other HTTP errors, re-throw a more informative error
      const errorMessage = statusError.message || "Failed to get file content";
      throw new Error(
        `GitHub API error (status ${statusError.status}): ${errorMessage}`
      );
    } else if (error instanceof Error) {
      // Re-throw other standard errors
      throw error;
    } else {
      // For truly unknown errors
      throw new Error("An unknown error occurred while fetching file content.");
    }
  }
}

export async function listPullRequests(
  owner: string,
  repo: string,
  author?: string,
  state?: "open" | "closed" | "all"
): Promise<RestEndpointMethodTypes["pulls"]["list"]["response"]["data"]> {
  const octokit = await getOctokitInstance();
  const { data: pullRequests } = await octokit.pulls.list({
    owner,
    repo,
    state: state ?? "all",
  });

  if (author) {
    return pullRequests.filter((pr) => pr.user?.login === author);
  }

  return pullRequests;
}
