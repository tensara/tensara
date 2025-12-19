import fs from "fs/promises";
import path from "path";
import type { GetStaticProps } from "next";
import { Box, Container } from "@chakra-ui/react";
import { Layout } from "~/components/layout";
import { MarkdownRenderer } from "~/components/blog";
import { markdownContentStyles } from "~/constants/blog";

type PolicyPageProps = {
  content: string;
};

export const getStaticProps: GetStaticProps<PolicyPageProps> = async () => {
  const filePath = path.join(process.cwd(), "TERMS_OF_SERVICE.md");
  const content = await fs.readFile(filePath, "utf8");

  return {
    props: { content },
  };
};

export default function TermsOfService({ content }: PolicyPageProps) {
  const displayContent = content.trim().length
    ? content
    : "_Terms of service will be available soon._";

  return (
    <Layout title="Terms of Service" ogTitle="Terms of Service">
      <Container maxW="5xl" py={{ base: 8, md: 12 }}>
        <Box
          className="markdown-content"
          sx={markdownContentStyles}
          w="90%"
          mx="auto"
        >
          <MarkdownRenderer content={displayContent} />
        </Box>
      </Container>
    </Layout>
  );
}
