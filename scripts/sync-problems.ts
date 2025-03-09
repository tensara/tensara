import { PrismaClient } from '@prisma/client';
import { readFileSync, readdirSync, existsSync } from 'fs';
import path from 'path';
import matter from 'gray-matter';

const prisma = new PrismaClient();

// Path utility functions
const getProblemsDir = () => path.join(process.cwd(), 'problems');
const getProblemPath = (slug: string) => path.join(getProblemsDir(), slug, 'problem.md');
const getStarterPath = (slug: string) => path.join(getProblemsDir(), slug, 'starter.cu');
const getTestsPath = (slug: string) => path.join(getProblemsDir(), slug, 'tests.hpp');
const getReferencePath = (slug: string) => path.join(getProblemsDir(), slug, 'reference.cu');

// Helper to safely read file contents
const safeReadFile = (path: string): string | null => {
  try {
    return existsSync(path) ? readFileSync(path, 'utf8') : null;
  } catch (error) {
    console.warn(`Warning: Could not read file at ${path}`);
    return null;
  }
};

async function main() {
  const problemsDir = getProblemsDir();
  const problemSlugs = readdirSync(problemsDir).filter(slug => slug !== '.DS_Store');

  for (const slug of problemSlugs) {
    const problemPath = getProblemPath(slug);

    const fileContents = readFileSync(problemPath, 'utf8');
    const { data: frontmatter, content } = matter(fileContents);

    const requiredFields = ['slug', 'title', 'difficulty', 'author', 'parameters'];
    const missingFields = requiredFields.filter(field => !frontmatter[field]);
    if (missingFields.length > 0) {
      throw new Error(`Problem ${slug} is missing required frontmatter: ${missingFields.join(', ')}`);
    }

    const tests = safeReadFile(getTestsPath(slug));
    const reference = safeReadFile(getReferencePath(slug));

    // Upsert problem in database
    const problem = await prisma.problem.upsert({
      where: { slug },
      update: {
        title: frontmatter.title,
        description: content,
        difficulty: frontmatter.difficulty,
        author: frontmatter.author,
        tests: tests,
        reference: reference,
        parameters: frontmatter.parameters,
      },
      create: {
        slug,
        title: frontmatter.title,
        description: content,
        difficulty: frontmatter.difficulty,
        author: frontmatter.author,
        tests: tests,
        reference: reference,
        parameters: frontmatter.parameters,
      }
    });

    console.log(`Synced problem: ${slug}`);
    console.log(`  - Title: ${frontmatter.title ? '✓' : '✗'}`);
    console.log(`  - Difficulty: ${frontmatter.difficulty ? '✓' : '✗'}`);
    console.log(`  - Parameters: ${frontmatter.parameters ? '✓' : '✗'}`);
    console.log(`  - Tests: ${tests ? '✓' : '✗'}`);
    console.log(`  - Reference: ${reference ? '✓' : '✗'}`);
  }
}

main()
  .catch(e => {
    console.error('❌ Sync failed:', e);
    process.exit(1);
  })
  .finally(async () => {
    await prisma.$disconnect();
  });
