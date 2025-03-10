import { initializeLeaderboardCache } from "./server/api/routers/submissions";

// Call after server is started
if (process.env.NODE_ENV === "production") {
  initializeLeaderboardCache().catch(console.error);
}
