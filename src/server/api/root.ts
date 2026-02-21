import { createCallerFactory, createTRPCRouter } from "~/server/api/trpc";
import { problemsRouter } from "./routers/problems";
import { submissionsRouter } from "~/server/api/routers/submissions";
import { usersRouter } from "~/server/api/routers/users";
import { apiKeysRouter } from "~/server/api/routers/apikey";
import { workspaceRouter } from "./routers/workspace";
import { snapshotRouter } from "./routers/snapshot";
import { blogpostRouter } from "./routers/blogpost";
import { commentRouter } from "./routers/comment";
import { postupvoteRouter } from "./routers/postupvote";
import { commentupvoteRouter } from "./routers/commentupvote";
import { tagRouter } from "./routers/tag";
import { groupsRouter } from "./routers/groups";

/**
 * This is the primary router for your server.
 *
 * All routers added in /api/routers should be manually added here.
 */
export const appRouter = createTRPCRouter({
  problems: problemsRouter,
  submissions: submissionsRouter,
  users: usersRouter,
  apiKeys: apiKeysRouter,
  workspace: workspaceRouter,
  snapshot: snapshotRouter,
  blogpost: blogpostRouter,
  comments: commentRouter,
  postUpvote: postupvoteRouter,
  commentUpvote: commentupvoteRouter,
  tag: tagRouter,
  groups: groupsRouter,
});

// export type definition of API
export type AppRouter = typeof appRouter;

/**
 * Create a server-side caller for the tRPC API.
 * @example
 * const trpc = createCaller(createContext);
 * const res = await trpc.post.all();
 *       ^? Post[]
 */
export const createCaller = createCallerFactory(appRouter);
