import { App } from "@octokit/app";
import { Octokit } from "@octokit/rest";
import type { RestEndpointMethodTypes } from "@octokit/plugin-rest-endpoint-methods";

import { env } from "~/env";

const BOT_GITHUB_APP_ID = env.BOT_GITHUB_APP_ID;
const BOT_GITHUB_APP_PRIVATE_KEY = env.BOT_GITHUB_APP_PRIVATE_KEY;
const BOT_GITHUB_APP_INSTALLATION_ID = env.BOT_GITHUB_APP_INSTALLATION_ID;

let octokitInstance: Octokit | null = null;
let appInstance: App | null = null;

async function getApp(): Promise<App> {
  if (appInstance) {
    return appInstance;
  }
  appInstance = new App({
    appId: BOT_GITHUB_APP_ID,
    privateKey: BOT_GITHUB_APP_PRIVATE_KEY,
    Octokit: Octokit,
  });
  return appInstance;
}

export async function getOctokitInstance(): Promise<Octokit> {
  if (octokitInstance) {
    return octokitInstance;
  }

  if (!BOT_GITHUB_APP_INSTALLATION_ID) {
    throw new Error(
      "BOT_GITHUB_APP_INSTALLATION_ID is not defined in environment variables."
    );
  }
  const installationIdString = BOT_GITHUB_APP_INSTALLATION_ID;
  const installationId = parseInt(installationIdString);
  if (isNaN(installationId)) {
    throw new Error(
      `Invalid BOT_GITHUB_APP_INSTALLATION_ID: "${installationIdString}" is not a valid integer.`
    );
  }

  const app = await getApp();
  octokitInstance = (await app.getInstallationOctokit(
    installationId
  )) as Octokit;

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
  } catch (error: unknown) {
    if (
      typeof error === "object" &&
      error !== null &&
      "status" in error &&
      (error as { status: unknown }).status === 404
    ) {
      const originalErrorMessage =
        typeof error === "object" && error !== null && "message" in error
          ? String((error as { message: unknown }).message)
          : "Unknown error";
      const errorMessage = `Failed to get reference for base branch '${baseBranch}' in repository '${owner}/${repo}'. The ref 'heads/${baseBranch}' was not found.
Please verify:
1. The repository '${owner}/${repo}' exists and the GitHub App has access.
2. The branch '${baseBranch}' (e.g., 'main', 'master') exists in '${owner}/${repo}'.
3. The GitHub App has 'Read' access to 'Contents' permission for this repository.
Original error: ${originalErrorMessage}`;
      console.error(errorMessage, error);
      throw new Error(errorMessage);
    }
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
  } catch (error: unknown) {
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
    sha = (file as { sha: string }).sha;
  } catch (error: unknown) {
    if (
      typeof error === "object" &&
      error !== null &&
      "status" in error &&
      typeof (error as { status?: unknown }).status === "number"
    ) {
      const statusError = error as { status: number; message?: string };
      if (statusError.status !== 404) {
        const errorMessage =
          statusError.message ?? "Failed to check file existence";
        throw new Error(
          `GitHub API error (status ${statusError.status}): ${errorMessage}`
        );
      }
    } else if (error instanceof Error) {
      throw error;
    } else {
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

interface FileForCommit {
  path: string;
  content: string;
}

export async function createFilesAndCommit(
  owner: string,
  repo: string,
  branch: string,
  baseBranch: string,
  files: FileForCommit[],
  commitMessage: string
): Promise<string> {
  const octokit = await getOctokitInstance();

  const { data: baseBranchRef } = await octokit.git.getRef({
    owner,
    repo,
    ref: `heads/${baseBranch}`,
  });
  const latestBaseCommitSha = baseBranchRef.object.sha;

  const { data: baseCommit } = await octokit.git.getCommit({
    owner,
    repo,
    commit_sha: latestBaseCommitSha,
  });
  const baseCommitTreeSha = baseCommit.tree.sha;

  const blobShas: Record<string, string> = {};
  for (const file of files) {
    const { data: blob } = await octokit.git.createBlob({
      owner,
      repo,
      content: Buffer.from(file.content).toString("base64"),
      encoding: "base64",
    });
    blobShas[file.path] = blob.sha;
  }

  const treeObjects = files.map((file) => ({
    path: file.path,
    mode: "100644" as const,
    type: "blob" as const,
    sha: blobShas[file.path],
  }));

  const { data: newTree } = await octokit.git.createTree({
    owner,
    repo,
    base_tree: baseCommitTreeSha,
    tree: treeObjects,
  });

  const { data: newCommit } = await octokit.git.createCommit({
    owner,
    repo,
    message: commitMessage,
    tree: newTree.sha,
    parents: [latestBaseCommitSha],
  });
  const newCommitSha = newCommit.sha;

  await octokit.git.updateRef({
    owner,
    repo,
    ref: `heads/${branch}`,
    sha: newCommitSha,
  });

  return newCommitSha;
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
        return null;
      }
      const errorMessage = statusError.message ?? "Failed to get file content";
      throw new Error(
        `GitHub API error (status ${statusError.status}): ${errorMessage}`
      );
    } else if (error instanceof Error) {
      throw error;
    } else {
      throw new Error("An unknown error occurred while fetching file content.");
    }
  }
}

export async function listPullRequests(
  owner: string,
  repo: string,
  state?: "open" | "closed" | "all"
): Promise<RestEndpointMethodTypes["pulls"]["list"]["response"]["data"]> {
  const octokit = await getOctokitInstance();
  const { data: pullRequests } = await octokit.pulls.list({
    owner,
    repo,
    state: state ?? "all",
  });

  return pullRequests;
}
