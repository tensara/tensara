import { createTRPCRouter, protectedProcedure } from "~/server/api/trpc";
import { z } from "zod";
import { TRPCError } from "@trpc/server";
import matter from "gray-matter";
import type { Octokit } from "@octokit/rest";
import {
  createBranch,
  createFilesAndCommit, // Added
  createOrUpdateFile, // Re-added for updateProblemPR
  createPullRequest,
  getFileContent,
  getOctokitInstance,
  listPullRequests,
} from "../../github";

const ProblemMetadataSchema = z.object({
  title: z.string(),
  difficulty: z.union([
    z.literal("EASY"),
    z.literal("MEDIUM"),
    z.literal("HARD"),
    z.literal("EXPERT"),
  ]),
  tags: z.array(z.string()),
  tensaraAppUserId: z.string().optional(),
});

const GITHUB_REPO_OWNER = "tensara";
const GITHUB_REPO_NAME = "problems";
const GITHUB_BASE_BRANCH = "main";

export const contributionsRouter = createTRPCRouter({
  submitNewProblem: protectedProcedure
    .input(
      z.object({
        contributorGithubUsername: z.string(),
        tensaraAppUserId: z.string(),
        problemDetails: z.object({
          title: z.string(),
          slug: z.string(),
          description: z.string(),
          difficulty: z.enum(["EASY", "MEDIUM", "HARD", "EXPERT"]),
          tags: z.array(z.string()),
          parameters: z.array(
            z.object({
              name: z.string(),
              type: z.string(),
              pointer: z.boolean().optional(),
              const: z.boolean().optional(),
            })
          ),
          referenceSolutionCode: z.string(),
          testCases: z.array(
            z.object({ input: z.string(), output: z.string() })
          ),
          flopsFormula: z.string(),
        }),
      })
    )
    .mutation(async ({ input, ctx }) => {
      if (!ctx.session.user) {
        throw new TRPCError({
          code: "UNAUTHORIZED",
          message: "You must be logged in to submit a problem",
        });
      }

      // Fetch the contributor's user record to get the correct username
      const user = await ctx.db.user.findUniqueOrThrow({
        where: { id: ctx.session.user.id },
        select: { username: true, name: true },
      });
      // Use database username if available, otherwise fallback to the input (which is required)
      const githubUsername = user.username ?? input.contributorGithubUsername;

      const newBranchName = `new-problem-${input.problemDetails.slug}-${Date.now()}`;
      const problemDirPath = `problems/${input.problemDetails.slug}/`;

      try {
        await createBranch(
          GITHUB_REPO_OWNER,
          GITHUB_REPO_NAME,
          GITHUB_BASE_BRANCH,
          newBranchName
        );

        // Construct problem.md content
        // The 'authorName' will be used in the YAML frontmatter.
        // It prioritizes the authenticated user's name (user.name from DB), falling back to the provided contributor GitHub username.
        const authorName = user.name ?? input.contributorGithubUsername;

        // YAML frontmatter for problem.md
        // Note: The 'parameters' field mentioned in the requirements is not present in the current
        // 'input.problemDetails' schema. It has been omitted from this frontmatter.
        const yamlFrontmatter = `---
title: "${input.problemDetails.title}"
slug: "${input.problemDetails.slug}"
difficulty: "${input.problemDetails.difficulty}"
author: "${authorName}"
parameters:
${input.problemDetails.parameters
  .map((p) => {
    let paramYaml = `  - name: ${p.name}\n    type: ${p.type}`;
    if (p.pointer) {
      paramYaml += `\n    pointer: true`;
    }
    if (p.const) {
      paramYaml += `\n    const: true`;
    }
    return paramYaml;
  })
  .join("\n")}
---`;

        const problemMdContent = `${yamlFrontmatter}\n\n${input.problemDetails.description}`;

        const defPyContent = `import torch
from tensara.problem import Problem

class MyProblem(Problem):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_function_signature(self):
        return (${input.problemDetails.parameters
          .map((p) => `"${p.name}"`)
          .join(", ")},)

    def reference_solution(self, ${input.problemDetails.parameters
      .map((p) => p.name)
      .join(", ")}):
${input.problemDetails.referenceSolutionCode
  .split("\n")
  .map((line) => `        ${line}`)
  .join("\n")}

    def verify_result(self, my_result, torch_result):
        return torch.allclose(my_result, torch_result)

    def generate_sample(self):
        # This should be implemented by the user to generate a sample input
        pass

    def generate_test_cases(self):
        return [
${input.problemDetails.testCases
  .map(
    (tc) =>
      `            {"input": """${tc.input}""", "output": """${tc.output}"""},`
  )
  .join("\n")}
        ]

    def get_flops(self, ${input.problemDetails.parameters
      .map((p) => p.name)
      .join(", ")}):
        return ${input.problemDetails.flopsFormula}

    def get_extra_params(self):
        return {}
`;

        const filesToCommit = [
          {
            path: `${problemDirPath}problem.md`,
            content: problemMdContent,
          },
          {
            path: `${problemDirPath}def.py`,
            content: defPyContent,
          },
        ];

        const commitMessage = `feat(new problem): ${input.problemDetails.title} (by @${githubUsername})`;

        await createFilesAndCommit(
          GITHUB_REPO_OWNER,
          GITHUB_REPO_NAME,
          newBranchName,
          GITHUB_BASE_BRANCH,
          filesToCommit,
          commitMessage
        );

        const prTitle = `Contribution: ${input.problemDetails.title}`;
        const prBody = `Problem Description:\n\n${input.problemDetails.description}`;

        const prUrl = await createPullRequest(
          GITHUB_REPO_OWNER,
          GITHUB_REPO_NAME,
          newBranchName,
          GITHUB_BASE_BRANCH,
          prTitle,
          prBody
        );

        return { prUrl };
      } catch (error) {
        throw new TRPCError({
          code: "INTERNAL_SERVER_ERROR",
          message:
            error instanceof Error
              ? `Failed to submit new problem: ${error.message}`
              : "Failed to submit new problem",
        });
      }
    }),

  updateProblemPR: protectedProcedure
    .input(
      z.object({
        prUrl: z.string().url(),
        problemDetails: z.object({
          title: z.string(),
          slug: z.string(),
          description: z.string(),
          difficulty: z.enum(["EASY", "MEDIUM", "HARD", "EXPERT"]),
          tags: z.array(z.string()),
          parameters: z.array(
            z.object({
              name: z.string(),
              type: z.string(),
              pointer: z.boolean().optional(),
              const: z.boolean().optional(),
            })
          ),
          referenceSolutionCode: z.string(),
          testCases: z.array(
            z.object({ input: z.string(), output: z.string() })
          ),
          flopsFormula: z.string(),
        }),
      })
    )
    .mutation(async ({ input, ctx }) => {
      if (!ctx.session.user) {
        throw new TRPCError({
          code: "UNAUTHORIZED",
          message: "You must be logged in to update a problem PR",
        });
      }

      try {
        const user = await ctx.db.user.findUniqueOrThrow({
          where: { id: ctx.session.user.id },
          select: { username: true, name: true },
        });

        const urlParts = input.prUrl.split("/");
        if (urlParts.length < 7) {
          throw new TRPCError({
            code: "BAD_REQUEST",
            message: "Invalid PR URL format",
          });
        }
        const owner = urlParts[3]!;
        const repo = urlParts[4]!;
        const pull_number = parseInt(urlParts[6]!);

        const octokit: Octokit = await getOctokitInstance();
        const { data: pullRequest } = await octokit.pulls.get({
          owner,
          repo,
          pull_number,
        });
        const headBranch: string = pullRequest.head.ref;
        const problemDirPath = `problems/${input.problemDetails.slug}/`;

        const authorName = user.name ?? user.username;

        const yamlFrontmatter = `---
title: "${input.problemDetails.title}"
slug: "${input.problemDetails.slug}"
difficulty: "${input.problemDetails.difficulty}"
author: "${authorName}"
parameters:
${input.problemDetails.parameters
  .map((p) => {
    let paramYaml = `  - name: ${p.name}\n    type: ${p.type}`;
    if (p.pointer) {
      paramYaml += `\n    pointer: true`;
    }
    if (p.const) {
      paramYaml += `\n    const: true`;
    }
    return paramYaml;
  })
  .join("\n")}
---`;

        const problemMdContent = `${yamlFrontmatter}\n\n${input.problemDetails.description}`;

        const defPyContent = `import torch
from tensara.problem import Problem

class MyProblem(Problem):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_function_signature(self):
        return (${input.problemDetails.parameters
          .map((p) => `"${p.name}"`)
          .join(", ")},)

    def reference_solution(self, ${input.problemDetails.parameters
      .map((p) => p.name)
      .join(", ")}):
${input.problemDetails.referenceSolutionCode
  .split("\n")
  .map((line) => `        ${line}`)
  .join("\n")}

    def verify_result(self, my_result, torch_result):
        return torch.allclose(my_result, torch_result)

    def generate_sample(self):
        # This should be implemented by the user to generate a sample input
        pass

    def generate_test_cases(self):
        return [
${input.problemDetails.testCases
  .map(
    (tc) =>
      `            {"input": """${tc.input}""", "output": """${tc.output}"""},`
  )
  .join("\n")}
        ]

    def get_flops(self, ${input.problemDetails.parameters
      .map((p) => p.name)
      .join(", ")}):
        return ${input.problemDetails.flopsFormula}

    def get_extra_params(self):
        return {}
`;
        // Update problem.md
        await createOrUpdateFile(
          owner,
          repo,
          headBranch,
          `${problemDirPath}problem.md`,
          problemMdContent,
          `fix: Update problem.md for ${input.problemDetails.slug}`
        );

        // Update def.py
        await createOrUpdateFile(
          owner,
          repo,
          headBranch,
          `${problemDirPath}def.py`,
          defPyContent,
          `fix: Update def.py for ${input.problemDetails.slug}`
        );

        return { success: true, message: "Problem PR updated successfully" };
      } catch (error) {
        const message =
          error instanceof Error
            ? `Failed to update problem PR: ${error.message}`
            : "Failed to update problem PR";
        throw new TRPCError({
          code: "INTERNAL_SERVER_ERROR",
          message,
        });
      }
    }),

  getMyContributions: protectedProcedure.query(async ({ ctx }) => {
    if (!ctx.session.user) {
      throw new TRPCError({
        code: "UNAUTHORIZED",
        message: "You must be logged in to view your contributions",
      });
    }

    const currentUserGithubUsername = ctx.session.user.username;

    if (!currentUserGithubUsername) {
      throw new TRPCError({
        code: "UNAUTHORIZED",
        message:
          "GitHub username not found in session. Please re-authenticate.",
      });
    }

    try {
      console.log(
        `[getMyContributions Debug] Session GitHub Username: '${currentUserGithubUsername}'`
      );

      // Fetch all pull requests from the repository
      const allPullRequests = await listPullRequests(
        GITHUB_REPO_OWNER,
        GITHUB_REPO_NAME,
        // No longer passing currentUserGithubUsername here, listPullRequests was changed
        "all" // Fetch all states: open, closed, merged
      );

      console.log(
        `[getMyContributions Debug] Total PRs fetched from ${GITHUB_REPO_OWNER}/${GITHUB_REPO_NAME}: ${allPullRequests.length}`
      );

      const userContributionsPRs = allPullRequests.filter((pr) => {
        if (!pr.body) {
          console.log(
            `[getMyContributions Debug] PR Title: '${pr.title}' has no body, skipping.`
          );
          return false;
        }
        // Regex to find "Contributed by: @username"
        // It captures the username. Handles potential variations like trailing spaces or different markdown link styles.
        const contributorMatch = /Contributed by: @([a-zA-Z0-9_-]+)/i.exec(
          pr.body
        );
        if (contributorMatch?.[1]) {
          const mentionedUsername = contributorMatch[1];
          console.log(
            `[getMyContributions Debug] PR Title: '${pr.title}', Found mentioned user in body: '${mentionedUsername}' (Comparing with: '${currentUserGithubUsername}')`
          );
          return (
            mentionedUsername.toLowerCase() ===
            currentUserGithubUsername.toLowerCase()
          );
        }
        console.log(
          `[getMyContributions Debug] PR Title: '${pr.title}', No 'Contributed by: @username' match in body.`
        );
        return false;
      });

      console.log(
        `[getMyContributions Debug] PRs found for '${currentUserGithubUsername}' after body parsing: ${userContributionsPRs.length}`
      );

      if (userContributionsPRs.length === 0 && currentUserGithubUsername) {
        console.log(
          `[getMyContributions Debug] No PRs found for user '${currentUserGithubUsername}' after body parsing. Listing PRs by tensarabot (or any other) for review of body format:`
        );
        allPullRequests.slice(0, 20).forEach((p) => {
          // Log first 20 to avoid excessive logging
          console.log(
            `[getMyContributions Debug] Sample PR: Title: '${p.title}', Author: ${p.user?.login}, Body: "${p.body?.substring(0, 150).replace(/\n/g, "\\n")}..."`
          );
        });
      }

      const contributions = await Promise.all(
        userContributionsPRs.map(async (pr) => {
          console.log(
            `[getMyContributions Debug] Processing PR for contribution: Title: '${pr.title}', URL: '${pr.html_url}', PR Author: '${pr.user?.login}'`
          );
          let problemTitle: string | null = null;
          let problemSlug: string | null = null;
          let problemDifficulty: string | null = null;

          // Try to parse problem slug from PR title, e.g., "feat: New Problem: XYZ (xyz-slug)"
          // or "Contribution: Problem Title (problem-slug)"
          const titleSlugMatch = /\(([^)]+)\)$/.exec(pr.title);
          const potentialSlug = titleSlugMatch?.[1] ?? null;
          console.log(
            `[getMyContributions Debug] PR Title: '${pr.title}', Extracted potentialSlug: '${potentialSlug}'`
          );

          if (potentialSlug) {
            // Verify if this slug exists in the database
            const problem = await ctx.db.problem.findUnique({
              where: { slug: potentialSlug },
              select: { title: true, slug: true, difficulty: true },
            });
            if (problem) {
              console.log(
                `[getMyContributions Debug] Found problem in DB for slug '${potentialSlug}': Title: '${problem.title}'`
              );
              problemTitle = problem.title;
              problemSlug = problem.slug;
              problemDifficulty = problem.difficulty;
            } else {
              console.log(
                `[getMyContributions Debug] No problem found in DB for slug '${potentialSlug}' (PR Title: '${pr.title}')`
              );
            }
          } else {
            console.log(
              `[getMyContributions Debug] No slug extracted for PR Title: '${pr.title}'`
            );
          }

          return {
            prUrl: pr.html_url,
            prTitle: pr.title,
            prStatus: pr.state, // 'open', 'closed' (merged PRs are 'closed' with merged_at set)
            prCreatedAt: new Date(pr.created_at),
            prUpdatedAt: new Date(pr.updated_at),
            prMergedAt: pr.merged_at ? new Date(pr.merged_at) : null,
            problemTitle,
            problemSlug,
            problemDifficulty,
            // Add other PR details as needed: pr.number, pr.body, pr.labels etc.
          };
        })
      );
      console.log(
        `[getMyContributions Debug] Final contributions array for '${currentUserGithubUsername}':`,
        JSON.stringify(contributions, null, 2)
      );
      return contributions;
    } catch (error) {
      const message =
        error instanceof Error
          ? `Failed to fetch contributions: ${error.message}`
          : "Failed to fetch contributions";
      console.error(
        `[getMyContributions Debug] Error for user ${currentUserGithubUsername}:`,
        message
      );
      throw new TRPCError({
        code: "INTERNAL_SERVER_ERROR",
        message,
      });
    }
  }),

  getProblemDetails: protectedProcedure
    .input(
      z
        .object({
          slug: z.string().optional(),
          prUrl: z.string().url().optional(),
        })
        .refine((data) => data.slug ?? data.prUrl, {
          message: "Either slug or prUrl must be provided",
        })
    )
    .query(async ({ input, ctx }) => {
      if (!ctx.session.user) {
        throw new TRPCError({
          code: "UNAUTHORIZED",
          message: "You must be logged in to view problem details",
        });
      }

      try {
        const problemContent: {
          problemMd: string | null;
          defPy: string | null;
        } = {
          problemMd: null,
          defPy: null,
        };
        let problemSlug: string;
        let branchToFetch: string;
        let owner = GITHUB_REPO_OWNER;
        let repo = GITHUB_REPO_NAME;

        if (input.slug) {
          problemSlug = input.slug;
          branchToFetch = GITHUB_BASE_BRANCH;
        } else if (input.prUrl) {
          const urlParts = input.prUrl.split("/");
          if (urlParts.length < 7) {
            throw new TRPCError({
              code: "BAD_REQUEST",
              message: "Invalid PR URL format",
            });
          }
          owner = urlParts[3]!;
          repo = urlParts[4]!;
          const pull_number = parseInt(urlParts[6]!);

          const octokit = await getOctokitInstance();
          const { data: pullRequest } = await octokit.pulls.get({
            owner,
            repo,
            pull_number,
          });
          branchToFetch = pullRequest.head.ref;

          // Attempt to derive slug from PR title or branch name if not explicitly provided
          const slugMatch = /\(([^)]+)\)$/.exec(pullRequest.title);
          if (slugMatch?.[1]) {
            problemSlug = slugMatch[1];
          } else {
            // Fallback: try to get slug from branch name if it follows the new-problem-{slug}-... pattern
            const branchSlugMatch = /new-problem-([^-]+)-/.exec(branchToFetch);
            if (branchSlugMatch?.[1]) {
              problemSlug = branchSlugMatch[1];
            } else {
              throw new TRPCError({
                code: "BAD_REQUEST",
                message:
                  "Could not determine problem slug from PR URL or title/branch name.",
              });
            }
          }
        } else {
          // This case should ideally be caught by the .refine() in the input schema
          throw new TRPCError({
            code: "BAD_REQUEST",
            message: "Either slug or prUrl must be provided.",
          });
        }

        const problemDirPath = `problems/${problemSlug}/`;

        problemContent.problemMd = await getFileContent(
          owner,
          repo,
          `${problemDirPath}problem.md`,
          branchToFetch
        );
        problemContent.defPy = await getFileContent(
          owner,
          repo,
          `${problemDirPath}def.py`,
          branchToFetch
        );

        if (!problemContent.problemMd || !problemContent.defPy) {
          throw new TRPCError({
            code: "NOT_FOUND",
            message: "Problem markdown file not found.",
          });
        }

        const { data: metadata, content: description } = matter(
          problemContent.problemMd
        );
        const parsedMetadata = ProblemMetadataSchema.parse(metadata);

        // Functions to extract code snippets from def.py
        const getPythonFunctionBody = (
          code: string,
          funcName: string
        ): string => {
          const funcRegex = new RegExp(
            `def ${funcName}\\(self, .*\\):\\n((?:\\s{8}.*\\n|\\n)*)`
          );
          const match = code.match(funcRegex);
          return match?.[1]?.trim() ?? "";
        };

        const getFlopsFormula = (code: string): string => {
          const flopsRegex = /return (.*)/;
          const match = flopsRegex.exec(
            getPythonFunctionBody(code, "get_flops")
          );
          return match?.[1]?.trim() ?? "";
        };

        const referenceSolutionCode = getPythonFunctionBody(
          problemContent.defPy,
          "reference_solution"
        );
        const generateTestCasesCode = getPythonFunctionBody(
          problemContent.defPy,
          "generate_test_cases"
        );
        const flopsFormula = getFlopsFormula(problemContent.defPy);

        return {
          title: parsedMetadata.title,
          slug: problemSlug,
          description,
          difficulty: parsedMetadata.difficulty,
          tags: parsedMetadata.tags,
          referenceSolutionCode,
          generateTestCasesCode,
          flopsFormula,
          parameters: [], // Placeholder, as this is not stored in a recoverable format
          tensaraAppUserId: parsedMetadata.tensaraAppUserId,
        };
      } catch (error) {
        const message =
          error instanceof Error
            ? `Failed to fetch problem details: ${error.message}`
            : "An unknown error occurred";
        throw new TRPCError({
          code: "INTERNAL_SERVER_ERROR",
          message,
          cause: error,
        });
      }
    }),
});
