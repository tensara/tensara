import { db } from "~/server/db";

const DAILY_SUBMISSION_LIMIT = 100;

export async function checkRateLimit(userId: string) {
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
        currentLimit: DAILY_SUBMISSION_LIMIT - 1,
      },
    });

    return {
      allowed: true,
      remainingSubmissions: DAILY_SUBMISSION_LIMIT - 1,
    };
  }

  if (user.currentLimit <= 0) {
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
      currentLimit: user.currentLimit - 1,
    },
  });

  return {
    allowed: true,
    remainingSubmissions: user.currentLimit - 1,
  };
}

function isMoreThanOneDay(date1: Date, date2: Date): boolean {
  const diffInMs = date2.getTime() - date1.getTime();
  const msInDay = 1000 * 60 * 60 * 24;
  return diffInMs >= msInDay;
}
