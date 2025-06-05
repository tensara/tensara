import { useState, useEffect } from "react";
import { type NextPage } from "next";
import { useSession } from "next-auth/react";
import { useRouter } from "next/router";
import { api } from "~/utils/api";
import { Layout } from "~/components/layout";
import {
  Box,
  Heading,
  Text,
  FormControl,
  FormLabel,
  Input,
  Textarea,
  Select,
  Button,
  useToast,
  useColorModeValue,
  VStack,
  HStack,
} from "@chakra-ui/react";
import PageHeader from "~/components/contribute/PageHeader";
import CodeEditor from "~/components/problem/CodeEditor";
import type { Difficulty } from "~/constants/problem";

const ModifyContributionPage: NextPage = () => {
  const router = useRouter();
  const { id } = router.query;
  const { data: session } = useSession();
  const getContribution = api.contributions.getProblemDetails.useQuery(
    { prUrl: id as string },
    { enabled: !!id } // Only enable query if id (prUrl) is available
  );
  const updateContribution = api.contributions.updateProblemPR.useMutation();
  const toast = useToast();

  const [title, setTitle] = useState("");
  const [description, setDescription] = useState("");
  const [difficulty, setDifficulty] = useState<Difficulty>("medium");
  const [referenceCode, setReferenceCode] = useState("");
  const [testCases, setTestCases] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);

  const cardBg = useColorModeValue("white", "gray.800");
  const cardBorder = useColorModeValue("gray.200", "gray.700");

  // Load contribution data when component mounts
  useEffect(() => {
    if (getContribution.data) {
      const problemDetails = getContribution.data;
      if (problemDetails) {
        setTitle(problemDetails.title);
        setDescription(problemDetails.description);
        setDifficulty(problemDetails.difficulty.toLowerCase() as Difficulty); // Convert to lowercase for local state
        setReferenceCode(problemDetails.referenceCode ?? ""); // Handle null
        setTestCases(problemDetails.testCases ?? ""); // Handle null
      }
    }
  }, [getContribution.data]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsSubmitting(true);

    if (!session?.user) {
      toast({
        title: "Not authenticated",
        description: "You must be logged in to modify a problem",
        status: "error",
        duration: 5000,
        isClosable: true,
      });
      setIsSubmitting(false);
      return;
    }

    const currentProblemDetails = getContribution.data; // Make it accessible here
    if (!currentProblemDetails) {
      // Handle case where data is not loaded yet
      toast({
        title: "Error",
        description: "Problem details not loaded yet. Please try again.",
        status: "error",
        duration: 5000,
        isClosable: true,
      });
      setIsSubmitting(false);
      return;
    }

    try {
      await updateContribution.mutateAsync({
        prUrl: id as string,
        problemDetails: {
          title,
          slug: currentProblemDetails?.slug ?? "", // Use existing slug or generate a placeholder
          description,
          difficulty: difficulty.toUpperCase() as
            | "EASY"
            | "MEDIUM"
            | "HARD"
            | "EXPERT", // Convert to uppercase for backend
          tags: currentProblemDetails?.tags ?? [], // Added this line
          referenceCode,
          testCases,
          parameters: currentProblemDetails?.parameters ?? [], // Preserve existing parameters
        },
      });

      toast({
        title: "Problem updated",
        description: "Your problem has been updated",
        status: "success",
        duration: 5000,
        isClosable: true,
      });

      void router.push("/contributions/view");
    } catch (err) {
      toast({
        title: "Error",
        description: "Failed to update problem. Please try again.",
        status: "error",
        duration: 5000,
        isClosable: true,
      });
      console.error(err);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <Layout>
      <Box maxW="6xl" mx="auto" px={4} py={8}>
        <PageHeader
          title="Modify Contribution"
          description="Edit your submitted problem"
        />

        <Box
          as="form"
          onSubmit={handleSubmit}
          bg={cardBg}
          borderWidth="1px"
          borderColor={cardBorder}
          borderRadius="xl"
          p={6}
          boxShadow="md"
        >
          <VStack spacing={6} align="stretch">
            <FormControl id="title" isRequired>
              <FormLabel fontWeight="medium">Problem Title</FormLabel>
              <Input
                type="text"
                value={title}
                onChange={(e) => setTitle(e.target.value)}
                size="lg"
              />
            </FormControl>

            <FormControl id="description" isRequired>
              <FormLabel fontWeight="medium">
                Problem Description (Markdown)
              </FormLabel>
              <Textarea
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                rows={8}
                size="lg"
              />
            </FormControl>

            <FormControl id="difficulty" isRequired>
              <FormLabel fontWeight="medium">Difficulty</FormLabel>
              <Select
                value={difficulty}
                onChange={(e) => setDifficulty(e.target.value as Difficulty)}
                size="lg"
              >
                <option value="easy">Easy</option>
                <option value="medium">Medium</option>
                <option value="hard">Hard</option>
              </Select>
            </FormControl>

            <FormControl id="referenceCode" isRequired>
              <FormLabel fontWeight="medium">
                Reference Implementation
              </FormLabel>
              <Box borderWidth="1px" borderRadius="md" overflow="hidden">
                <div style={{ height: "300px" }}>
                  <CodeEditor
                    code={referenceCode}
                    setCode={setReferenceCode}
                    selectedLanguage="cuda"
                  />
                </div>
              </Box>
            </FormControl>

            <FormControl id="testCases" isRequired>
              <FormLabel fontWeight="medium">Test Cases</FormLabel>
              <Box borderWidth="1px" borderRadius="md" overflow="hidden">
                <div style={{ height: "200px" }}>
                  <CodeEditor
                    code={testCases}
                    setCode={setTestCases}
                    selectedLanguage="cuda"
                  />
                </div>
              </Box>
            </FormControl>

            <HStack justify="flex-end" mt={6}>
              <Button
                type="submit"
                colorScheme="blue"
                size="lg"
                isLoading={isSubmitting}
                loadingText="Updating..."
              >
                Update Problem
              </Button>
            </HStack>
          </VStack>
        </Box>
      </Box>
    </Layout>
  );
};

export default ModifyContributionPage;
