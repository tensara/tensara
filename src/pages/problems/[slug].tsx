import { useRouter } from "next/router";
import { api } from "~/utils/api";
import {
  Box,
  Heading,
  Text,
  HStack,
  Spinner,
  Code,
  VStack,
} from "@chakra-ui/react";
import { useState } from "react";
import { Layout } from "~/components/layout";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import rehypeHighlight from "rehype-highlight";
import Editor from "@monaco-editor/react";

export default function ProblemPage() {
  const router = useRouter();
  const { slug } = router.query;
  const [code, setCode] = useState(`#include <iostream>

int main() {
    int a, b;

    cout << "Enter numbers: ";
    cin >> a >> b;

    cout << a << " + " << b << " = " << a + b << endl;
    
    return 0
}`);

  const { data: problem, isLoading } = api.problems.getById.useQuery(
    { slug: slug as string },
    { enabled: !!slug }
  );

  if (isLoading) {
    return (
      <Layout title="Loading...">
        <Box display="flex" justifyContent="center" alignItems="center">
          <Spinner size="xl" />
        </Box>
      </Layout>
    );
  }

  if (!problem) {
    return (
      <Layout title="Not Found">
        <Box p={8}>
          <Text>Problem not found</Text>
        </Box>
      </Layout>
    );
  }

  return (
    <Layout title={problem.title}>
      <HStack align="start" spacing={8} h="100%" maxH="calc(100vh - 120px)">
        {/* Problem Description */}
        <Box w="50%" h="100%" overflowY="auto" p={6}>
          <Heading size="lg" mb={2}>
            {problem.title}
          </Heading>
          <Text color="gray.400" mb={6}>
            Difficulty: {problem.difficulty}
          </Text>

          <Box className="markdown" color="gray.100">
            <ReactMarkdown
              remarkPlugins={[remarkGfm, remarkMath]}
              rehypePlugins={[rehypeKatex, rehypeHighlight]}
              components={{
                h1: (props) => (
                  <Heading as="h1" size="xl" mt={6} mb={4} {...props} />
                ),
                h2: (props) => (
                  <Heading as="h2" size="lg" mt={5} mb={3} {...props} />
                ),
                h3: (props) => (
                  <Heading as="h3" size="md" mt={4} mb={2} {...props} />
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
                    p={4}
                    bg="gray.800"
                    borderRadius="md"
                    overflowX="auto"
                    mb={4}
                    {...props}
                  />
                ),
              }}
            >
              {problem.description}
            </ReactMarkdown>
          </Box>

          <Heading size="md" mb={4}>
            Example Test Cases
          </Heading>
          <VStack spacing={4} align="stretch">
            {problem.testCases.map((testCase) => (
              <Box key={testCase.id} p={4} bg="gray.800" borderRadius="md">
                <Text color="gray.300" mb={2}>
                  <Text as="span" fontWeight="bold" color="gray.200">
                    Input:{" "}
                  </Text>
                  <Code bg="gray.700" px={2}>
                    {testCase.input}
                  </Code>
                </Text>
                <Text color="gray.300">
                  <Text as="span" fontWeight="bold" color="gray.200">
                    Expected:{" "}
                  </Text>
                  <Code bg="gray.700" px={2}>
                    {testCase.expected}
                  </Code>
                </Text>
              </Box>
            ))}
          </VStack>
        </Box>

        {/* Code Editor */}
        <Box w="50%" h="100%" bg="gray.800" borderRadius="xl" overflow="hidden">
          <Editor
            height="100%"
            defaultLanguage="cpp"
            theme="vs-dark"
            value={code}
            onChange={(value) => setCode(value ?? "")}
            options={{
              minimap: { enabled: false },
              fontSize: 14,
              lineNumbers: "on",
              scrollBeyondLastLine: false,
              automaticLayout: true,
              padding: { top: 16, bottom: 16 },
              fontFamily: "JetBrains Mono, monospace",
            }}
          />
        </Box>
      </HStack>
    </Layout>
  );
}
