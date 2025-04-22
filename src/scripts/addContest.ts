// scripts/seedUpcomingContest.ts
import { PrismaClient, type Problem } from "@prisma/client";

const prisma = new PrismaClient();
async function seedUpcomingContest() {
  try {
    const nowDate = new Date();
    const weekInMs = 7 * 24 * 60 * 60 * 1000;

    const upcomingContest = await prisma.contest.create({
      data: {
        title: "Beat the LLM",
        slug: "beat-the-llm",
        description: `
Compete against an advanced Language Model in a series of coding challenges. Test your skills, speed, and problem-solving abilities in this unique competition.

## Rules
1. Individual participation only
2. No external help allowed
3. Solutions must be submitted within the time limit
4. Code must be original and written during the contest

## Registration
Register now to secure your spot for the contest!`,
        status: "UPCOMING",
        startTime: new Date(nowDate.getTime() + weekInMs),
        endTime: new Date(nowDate.getTime() + 2 * weekInMs),
        registrationStartTime: nowDate,
        registrationEndTime: new Date(
          nowDate.getTime() + 6 * 24 * 60 * 60 * 1000
        ),
      },
    });

    const problems = await prisma.problem.findMany({ take: 3 });

    if (problems.length > 0) {
      await prisma.contest.update({
        where: { id: upcomingContest.id },
        data: {
          problemIds: problems.map((p) => p.id),
        },
      });
    }

    console.log("Upcoming contest created successfully!", upcomingContest);
  } catch (error) {
    console.error("Error creating upcoming contest:", error);
  } finally {
    await prisma.$disconnect();
  }
}

void seedUpcomingContest();
