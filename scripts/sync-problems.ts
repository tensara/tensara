import { PrismaClient } from '@prisma/client';
import { readFileSync, readdirSync, existsSync } from 'fs';
import path from 'path';
import matter from 'gray-matter';

const prisma = new PrismaClient();

// Path utility functions
const getProblemsDir = () => path.join(process.cwd(), 'problems');
const getProblemPath = (slug: string) => path.join(getProblemsDir(), slug, 'problem.md');
const getTestsDir = (slug: string) => path.join(getProblemsDir(), slug, 'tests');
const getStarterPath = (slug: string) => path.join(getProblemsDir(), slug, 'starter.cu');
const getBindingsPath = (slug: string) => path.join(getProblemsDir(), slug, 'bindings.cpp');
const getReferencePath = (slug: string) => path.join(getProblemsDir(), slug, 'reference.py');

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
  const problemSlugs = readdirSync(problemsDir);

  for (const slug of problemSlugs) {
    const problemPath = getProblemPath(slug);
    const testsDir = getTestsDir(slug);

    // Parse markdown content
    const fileContents = readFileSync(problemPath, 'utf8');
    const { data: frontmatter, content } = matter(fileContents);

    // Validate required fields
    const requiredFields = ['slug', 'title', 'difficulty', 'author'];
    const missingFields = requiredFields.filter(field => !frontmatter[field]);
    if (missingFields.length > 0) {
      throw new Error(`Problem ${slug} is missing required frontmatter: ${missingFields.join(', ')}`);
    }

    // Read additional files
    const starterCode = safeReadFile(getStarterPath(slug));
    const bindings = safeReadFile(getBindingsPath(slug));
    const reference = safeReadFile(getReferencePath(slug));

    // Upsert problem in database
    const problem = await prisma.problem.upsert({
      where: { slug },
      update: {
        title: frontmatter.title,
        description: content,
        difficulty: frontmatter.difficulty,
        author: frontmatter.author,
        starterCode: starterCode,
        bindings: bindings,
        reference: reference,
      },
      create: {
        slug,
        title: frontmatter.title,
        description: content,
        difficulty: frontmatter.difficulty,
        author: frontmatter.author,
        starterCode: starterCode,
        bindings: bindings,
        reference: reference,
      }
    });

    // Sync test cases if directory exists
    if (existsSync(testsDir)) {
      const testFiles = readdirSync(testsDir).filter(f =>
        f.endsWith('.json') &&
        !['setup.json', 'config.json'].includes(f.toLowerCase())
      );

      await prisma.testCase.deleteMany({ where: { problemId: problem.id } });

      for (const testFile of testFiles) {
        const testPath = path.join(testsDir, testFile);
        const testContent = JSON.parse(readFileSync(testPath, 'utf8'));
        const isHidden = testFile.toLowerCase().includes('hidden');

        await prisma.testCase.create({
          data: {
            input: testContent.input,
            expected: testContent.expected,
            isHidden,
            problem: { connect: { id: problem.id } }
          }
        });
      }
      console.log(`Synced problem: ${slug} with ${testFiles.length} test cases`);
      console.log(`  - Starter code: ${starterCode ? '✓' : '✗'}`);
      console.log(`  - Bindings: ${bindings ? '✓' : '✗'}`);
      console.log(`  - Reference: ${reference ? '✓' : '✗'}`);
    } else {
      console.log(`Synced problem: ${slug} (no test cases found)`);
    }
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
