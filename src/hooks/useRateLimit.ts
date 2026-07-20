import { db } from "~/server/db";

const DAILY_SUBMISSION_LIMIT = 100;
const NEW_ACCOUNT_WINDOW_MS = 48 * 60 * 60 * 1000;

export async function checkRateLimit(
  userId: string,
  options?: { newAccountDailyLimit?: number }
) {
  if (!userId) {
    return {
      allowed: false,
      error: "Not authenticated",
      statusCode: 401,
    };
  }
  const user = await db.user.findUnique({
    where: { id: userId },
    select: {
      id: true,
      createdAt: true,
      lastLimitReset: true,
      currentLimit: true,
    },
  });

  if (!user) {
    return {
      allowed: false,
      error: "User not found",
      statusCode: 404,
    };
  }

  const now = new Date();
  const dailyLimit =
    options?.newAccountDailyLimit &&
    now.getTime() - user.createdAt.getTime() < NEW_ACCOUNT_WINDOW_MS
      ? options.newAccountDailyLimit
      : DAILY_SUBMISSION_LIMIT;

  if (
    !user.lastLimitReset ||
    user.currentLimit === null ||
    user.currentLimit === undefined ||
    isMoreThanOneDay(user.lastLimitReset, now)
  ) {
    await db.user.update({
      where: { id: userId },
      data: {
        lastLimitReset: now,
        currentLimit: dailyLimit - 1,
      },
    });

    return {
      allowed: true,
      remainingSubmissions: dailyLimit - 1,
    };
  }

  const usedToday = DAILY_SUBMISSION_LIMIT - user.currentLimit;
  const currentLimit = Math.max(dailyLimit - usedToday, 0);

  if (currentLimit <= 0) {
    const nextReset = new Date(user.lastLimitReset);
    nextReset.setDate(nextReset.getDate() + 1);
    const timeUntilReset = nextReset.getTime() - now.getTime();
    const hoursUntilReset = Math.floor(timeUntilReset / (1000 * 60 * 60));
    const minutesUntilReset = Math.floor(
      (timeUntilReset % (1000 * 60 * 60)) / (1000 * 60)
    );

    return {
      allowed: false,
      error: `Rate limit exceeded. Please try again in ${hoursUntilReset}h ${minutesUntilReset}m.`,
      statusCode: 429,
    };
  }

  await db.user.update({
    where: { id: userId },
    data: {
      currentLimit: currentLimit - 1,
    },
  });

  return {
    allowed: true,
    remainingSubmissions: currentLimit - 1,
  };
}

function isMoreThanOneDay(date1: Date, date2: Date): boolean {
  const diffInMs = date2.getTime() - date1.getTime();
  const msInDay = 1000 * 60 * 60 * 24;
  return diffInMs >= msInDay;
}
