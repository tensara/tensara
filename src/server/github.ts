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
  problemMdContent: string, // Added: Content of problem.md
  githubUsername: string // Added: Contributor's GitHub username
): Promise<string> {
  const octokit = await getOctokitInstance();

  // Extract problem description from problem.md content
  let problemDescription = "";
  const frontmatterEndMarker = "\n---"; // YAML frontmatter ends with --- on a new line
  const secondMarkerIndex = problemMdContent.indexOf(
    frontmatterEndMarker,
    problemMdContent.indexOf(frontmatterEndMarker) + frontmatterEndMarker.length
  );
  if (secondMarkerIndex !== -1) {
    problemDescription = problemMdContent
      .substring(secondMarkerIndex + frontmatterEndMarker.length)
      .trim();
  } else {
    // Fallback if frontmatter isn't found as expected, though problem.md should always have it
    // Or, if the first '---' is the only one, take everything after it.
    const firstMarkerIndex = problemMdContent.indexOf("---");
    if (
      firstMarkerIndex !== -1 &&
      problemMdContent.substring(firstMarkerIndex + 3).indexOf("---") === -1
    ) {
      // This case handles if there's only one '---' block, implying the rest is description.
      // However, the typical structure is two '---' blocks.
      // A more robust parser might be needed for complex cases, but this handles simple frontmatter.
      // For now, if only one '---' is found, or parsing is ambiguous, we might leave description empty or use full content.
      // Given the example, the second '---' is key.
      // If the second '---' is not found, it implies malformed frontmatter or no description after.
      // For safety, let's log a warning or handle it gracefully.
      console.warn(
        "Could not accurately parse problem description from problem.md content. PR body might be incomplete."
      );
      // Defaulting to a placeholder or the full content if parsing fails might be an option.
      // For now, we'll proceed with potentially empty description if parsing fails.
    }
  }

  // Construct the new PR body
  const tensaraProfileLink = `https://tensara.org/${githubUsername}`;
  const prBody = `This PR introduces a new problem: "${title}".

Contributed by: @${githubUsername} ([Tensara Profile Link](${tensaraProfileLink}))

**Problem Description:**
${problemDescription}`;

  const { data: pullRequest } = await octokit.pulls.create({
    owner,
    repo,
    head,
    base,
    title,
    body: prBody, // Use the newly constructed body
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
  branch: string, // The new problem branch (e.g., new-problem-slug-timestamp)
  baseBranch: string, // The branch to base the commit on (e.g., "main")
  files: FileForCommit[],
  commitMessage: string
): Promise<string> {
  // Returns the SHA of the new commit
  const octokit = await getOctokitInstance();

  // 1. Get the SHA of the latest commit on the baseBranch
  const { data: baseBranchRef } = await octokit.git.getRef({
    owner,
    repo,
    ref: `heads/${baseBranch}`,
  });
  const latestBaseCommitSha = baseBranchRef.object.sha;

  // 2. Get the tree SHA of this base commit
  const { data: baseCommit } = await octokit.git.getCommit({
    owner,
    repo,
    commit_sha: latestBaseCommitSha,
  });
  const baseCommitTreeSha = baseCommit.tree.sha;

  // 3. Create blobs for each file
  const blobShas: { [path: string]: string } = {};
  for (const file of files) {
    const { data: blob } = await octokit.git.createBlob({
      owner,
      repo,
      content: Buffer.from(file.content).toString("base64"),
      encoding: "base64",
    });
    blobShas[file.path] = blob.sha;
  }

  // 4. Create a new tree
  const treeObjects = files.map((file) => ({
    path: file.path,
    mode: "100644" as const, // '100644' for file (blob)
    type: "blob" as const,
    sha: blobShas[file.path],
  }));

  const { data: newTree } = await octokit.git.createTree({
    owner,
    repo,
    base_tree: baseCommitTreeSha,
    tree: treeObjects,
  });

  // 5. Create a single commit
  const { data: newCommit } = await octokit.git.createCommit({
    owner,
    repo,
    message: commitMessage,
    tree: newTree.sha,
    parents: [latestBaseCommitSha],
  });
  const newCommitSha = newCommit.sha;

  // 6. Update the head of the branch (the new problem branch)
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
