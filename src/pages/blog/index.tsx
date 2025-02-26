import { Box, Container, Heading, Text, VStack, Link } from "@chakra-ui/react";
import { Layout } from "~/components/layout";
import fs from "fs";
import path from "path";
import matter from "gray-matter";
import { format } from "date-fns";
import { GetStaticProps } from "next";

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
    <Layout title="Blog">
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
  const filenames = fs.readdirSync(contentDirectory);
  
  const posts = filenames
    .filter((filename) => filename.endsWith(".md"))
    .map((filename) => {
      const filePath = path.join(contentDirectory, filename);
      const fileContents = fs.readFileSync(filePath, "utf8");
      const { data, content } = matter(fileContents);
      
      
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
