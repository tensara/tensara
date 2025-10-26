import { Box, Container, Heading, Text, VStack } from "@chakra-ui/react";
import { Layout } from "~/components/layout";
import Link from "next/link";
import fs from "fs";
import path from "path";
import type { GetStaticProps } from "next";
import matter from "gray-matter";
import { format } from "date-fns";

interface BlogPost {
  slug: string;
  title: string;
  date: string;
}

interface BlogIndexProps {
  posts: BlogPost[];
}

export default function BlogIndex({ posts }: BlogIndexProps) {
  return (
    <Layout
      title="Blog"
      ogTitle="Blog | Tensara"
      ogDescription="Blog posts about how we built Tensara, GPU programming, and more."
    >
      <Container maxW="4xl" py={8}>
        <Heading
          as="h1"
          size="2xl"
          mb={8}
          fontFamily="Space Grotesk, sans-serif"
        >
          Blog
        </Heading>

        <VStack spacing={6} align="stretch">
          {posts.map((post) => (
            <Box
              key={post.slug}
              as={Link}
              href={`/blog/${post.slug}`}
              p={6}
              borderRadius="xl"
              bg="brand.secondary"
              _hover={{ bg: "whiteAlpha.100" }}
              borderWidth="1px"
              borderColor="whiteAlpha.100"
              transition="all 0.2s"
              textDecoration="none !important" // Prevent underline on hover
            >
              <Heading size="lg" mb={2}>
                {post.title}
              </Heading>
              <Text color="gray.400" fontSize="sm">
                {format(new Date(post.date), "MMMM d, yyyy")}
              </Text>
            </Box>
          ))}
        </VStack>
      </Container>
    </Layout>
  );
}

export const getStaticProps: GetStaticProps<BlogIndexProps> = async () => {
  const contentDirectory = path.join(process.cwd(), "public/content");
  let filenames: string[] = [];

  try {
    filenames = fs.readdirSync(contentDirectory);
  } catch (error) {
    console.log(error);
    filenames = [];
  }

  const posts = filenames
    .filter((filename) => filename.endsWith(".md"))
    .map((filename) => {
      const filePath = path.join(contentDirectory, filename);
      const fileContents = fs.readFileSync(filePath, "utf8");
      const { data } = matter(fileContents);

      return {
        slug: filename.replace(".md", ""),
        title: data.title as string,
        date: data.date as string,
      };
    })
    .sort((a, b) => {
      const dateA = new Date(String(a.date));
      const dateB = new Date(String(b.date));
      return dateB.getTime() - dateA.getTime();
    });

  return {
    props: {
      posts,
    },
  };
};
