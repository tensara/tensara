import { Box, Container, Heading, Text, Code } from "@chakra-ui/react";
import fs from "fs";
import path from "path";
import { Layout } from "~/components/layout";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import rehypeHighlight from "rehype-highlight";
import type { GetStaticProps } from "next";
import matter from "gray-matter";
import { format } from "date-fns";

interface BlogPostProps {
  title: string;
  date: string;
  content: string;
}

export default function BenchmarkingSolutionsPage({
  title,
  date,
  content,
}: BlogPostProps) {
  const formattedDate = format(new Date(date), "MMMM d, yyyy");

  return (
    <Layout title={title}>
      <Container maxW="4xl" py={8}>
        <Box mb={10}>
          <Heading
            as="h1"
            size="2xl"
            mb={4}
            fontFamily="Space Grotesk, sans-serif"
          >
            {title}
          </Heading>
          <Text fontSize="lg" color="gray.400">
            {formattedDate}
          </Text>
        </Box>

        <Box className="markdown" color="gray.100">
          <ReactMarkdown
            remarkPlugins={[remarkGfm, remarkMath]}
            rehypePlugins={[rehypeKatex, rehypeHighlight]}
            components={{
              h1: (props) => (
                <Heading as="h2" size="xl" mt={8} mb={4} {...props} />
              ),
              h2: (props) => (
                <Heading as="h3" size="lg" mt={6} mb={3} {...props} />
              ),
              h3: (props) => (
                <Heading as="h4" size="md" mt={4} mb={2} {...props} />
              ),
              ul: (props) => <Box as="ul" pl={8} mb={4} {...props} />,
              ol: (props) => <Box as="ol" pl={8} mb={4} {...props} />,
              li: (props) => <Box as="li" pl={2} mb={2} {...props} />,
              code: (props) => (
                <Code
                  px={2}
                  py={1}
                  bg="gray.800"
                  color="gray.100"
                  borderRadius="md"
                  {...props}
                />
              ),
              pre: (props) => (
                <Box
                  as="pre"
                  p={6}
                  bg="brand.secondary"
                  borderRadius="xl"
                  overflowX="auto"
                  mb={6}
                  borderWidth="1px"
                  borderColor="whiteAlpha.100"
                  {...props}
                />
              ),
            }}
          >
            {content}
          </ReactMarkdown>
        </Box>
      </Container>
    </Layout>
  );
}

export const getStaticProps: GetStaticProps<BlogPostProps> = async () => {
  const filePath = path.join(process.cwd(), "public/content/benchmarking.md");
  const fileContents = fs.readFileSync(filePath, "utf8");

  const { data, content } = matter(fileContents);

  return {
    props: {
      title: data.title as string,
      date: data.date as string,
      content,
    },
  };
};
