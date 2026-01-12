import type { NextApiRequest, NextApiResponse } from "next";

import { db } from "~/server/db";

type LandingActivityResponse = {
  repoStars: number | null;
  prs: Array<{ title: string; url: string; number: number; updatedAt: string }>;
  problems: Array<{
    title: string;
    slug: string;
    difficulty: string;
    createdAt: string;
  }>;
  blogPosts: Array<{
    title: string;
    slug: string;
    publishedAt: string;
    authorUsername: string | null;
  }>;
};

function logGithubCall(params: {
  requestId: string;
  endpoint: "repo" | "pulls";
  repo: string;
  status: number;
  remaining: string | null;
  reset: string | null;
}) {
  console.log("[github]", params);
}

async function fetchRepoStars(requestId: string): Promise<number | null> {
  const repo = process.env.GITHUB_REPO ?? "tensara/tensara";
  const token = process.env.GITHUB_TOKEN;

  const res = await fetch(`https://api.github.com/repos/${repo}`, {
    headers: {
      Accept: "application/vnd.github+json",
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
  });

  logGithubCall({
    requestId,
    endpoint: "repo",
    repo,
    status: res.status,
    remaining: res.headers.get("x-ratelimit-remaining"),
    reset: res.headers.get("x-ratelimit-reset"),
  });

  if (!res.ok) return null;
  const json = (await res.json()) as { stargazers_count?: number };
  return typeof json.stargazers_count === "number"
    ? json.stargazers_count
    : null;
}

async function fetchLatestMergedPullRequests(
  requestId: string,
): Promise<
  LandingActivityResponse["prs"]
> {
  const repo = process.env.GITHUB_REPO ?? "tensara/tensara";
  const token = process.env.GITHUB_TOKEN;

  const res = await fetch(
    `https://api.github.com/repos/${repo}/pulls?state=closed&per_page=20&sort=updated&direction=desc`,
    {
      headers: {
        Accept: "application/vnd.github+json",
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
      },
    }
  );

  logGithubCall({
    requestId,
    endpoint: "pulls",
    repo,
    status: res.status,
    remaining: res.headers.get("x-ratelimit-remaining"),
    reset: res.headers.get("x-ratelimit-reset"),
  });

  if (!res.ok) return [];
  const json = (await res.json()) as Array<{
    title: string;
    html_url: string;
    number: number;
    updated_at: string;
    merged_at: string | null;
  }>;

  return json
    .filter((pr) => pr.merged_at)
    .slice(0, 3)
    .map((pr) => ({
      title: pr.title,
      url: pr.html_url,
      number: pr.number,
      updatedAt: pr.merged_at ?? pr.updated_at,
    }));
}

export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse<LandingActivityResponse>
) {
  const requestId = `${Date.now()}-${Math.random().toString(16).slice(2)}`;

  if (req.method !== "GET") {
    res.setHeader("Allow", "GET");
    return res.status(405).json({
      repoStars: null,
      prs: [],
      problems: [],
      blogPosts: [],
    });
  }

  try {
    const [repoStars, prs, problems, blogPosts] = await Promise.all([
      fetchRepoStars(requestId).catch(() => null),
      fetchLatestMergedPullRequests(requestId).catch(() => []),
      db.problem
        .findMany({
          orderBy: { createdAt: "desc" },
          take: 3,
          select: {
            title: true,
            slug: true,
            difficulty: true,
            createdAt: true,
          },
        })
        .catch(() => []),
      db.blogPost
        .findMany({
          where: {
            status: "PUBLISHED",
            slug: { not: null },
          },
          orderBy: [{ publishedAt: "desc" }, { createdAt: "desc" }],
          take: 3,
          select: {
            title: true,
            slug: true,
            publishedAt: true,
            author: { select: { username: true, name: true } },
          },
        })
        .catch(() => []),
    ]);

    res.setHeader("Cache-Control", "s-maxage=300, stale-while-revalidate=600");
    return res.status(200).json({
      repoStars,
      prs,
      problems: problems.map((p) => ({
        title: p.title,
        slug: p.slug,
        difficulty: p.difficulty,
        createdAt: p.createdAt.toISOString(),
      })),
      blogPosts: blogPosts
        .filter((p) => p.slug && p.publishedAt)
        .map((p) => ({
          title: p.title,
          slug: p.slug!,
          publishedAt: p.publishedAt!.toISOString(),
          authorUsername: p.author.username ?? p.author.name ?? null,
        })),
    });
  } catch (err) {
    console.error(err);
    return res.status(200).json({
      repoStars: null,
      prs: [],
      problems: [],
      blogPosts: [],
    });
  }
}
