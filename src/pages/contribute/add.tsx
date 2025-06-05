import { useState } from "react";
import { type NextPage } from "next";
import { useSession } from "next-auth/react";
import { useRouter } from "next/router";
import { api } from "~/utils/api";
import { Layout } from "~/components/layout";
import {
  Flex,
  Box,
  Heading,
  Text,
  FormControl,
  FormLabel,
  Input,
  Select,
  Button,
  useToast,
  useColorModeValue,
  Grid,
  GridItem,
} from "@chakra-ui/react";
import PageHeader from "~/components/contribute/PageHeader";
import CodeEditor from "~/components/problem/CodeEditor";
import MarkdownEditor from "~/components/contribute/MarkdownEditor";
import type { Difficulty } from "~/constants/problem";

const AddContributionPage: NextPage = () => {
  const { data: session } = useSession();
  const router = useRouter();
  const createContribution = api.contributions.submitNewProblem.useMutation();
  const toast = useToast();

  const [title, setTitle] = useState("");
  const [description, setDescription] = useState("");
  const [difficulty, setDifficulty] = useState<Difficulty>("medium");
  const [referenceCode, setReferenceCode] = useState("");
  const [testCases, setTestCases] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);

  const cardBg = useColorModeValue("white", "gray.800");
  const cardBorder = useColorModeValue("gray.200", "gray.700");

  const borderColorDefault = useColorModeValue("gray.300", "gray.600");
  const bgColorDefault = useColorModeValue("gray.100", "gray.700");
  const textColorDefault = useColorModeValue("gray.800", "gray.200");
  const hoverBorderColorDefault = useColorModeValue("gray.400", "gray.500");

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsSubmitting(true);

    if (!session?.user) {
      toast({
        title: "Not authenticated",
        description: "You must be logged in to submit a problem",
        status: "error",
        duration: 5000,
        isClosable: true,
      });
      setIsSubmitting(false);
      return;
    }

    try {
      const contribution = await createContribution.mutateAsync({
        contributorGithubUsername: session.user.name ?? "unknown", // Assuming session.user.name is GitHub username
        tensaraAppUserId: session.user.id,
        problemDetails: {
          title,
          slug: title
            .toLowerCase()
            .replace(/[^a-z0-9]+/g, "-")
            .replace(/^-*|-*$/g, ""), // Simple slug generation
          description,
          difficulty: difficulty.toUpperCase() as
            | "EASY"
            | "MEDIUM"
            | "HARD"
            | "EXPERT",
          tags: [], // Added this line
          referenceCode,
          testCases,
          parameters: [], // Assuming no parameters for now, or add a form for them
        },
      });

      toast({
        title: "Problem submitted",
        description: "Your problem has been submitted for review",
        status: "success",
        duration: 5000,
        isClosable: true,
      });

      if (contribution.prUrl) {
        void router.push(`/contributions/view?pr=${contribution.prUrl}`);
      } else {
        void router.push("/contributions/view");
      }
    } catch (err) {
      toast({
        title: "Error",
        description: "Failed to submit problem. Please try again.",
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
      <Box maxW="8xl" mx="auto" px={4} py={12}>
        <PageHeader
          title="Add New Problem"
          description="Submit a new problem to Tensara for review"
        />

        <Box
          as="form"
          onSubmit={handleSubmit}
          bg={cardBg}
          borderWidth="1px"
          borderColor={cardBorder}
          borderRadius="2xl"
          p={8}
          boxShadow="xl"
        >
          <Grid templateColumns={{ base: "1fr", md: "2fr 3fr" }} gap={10}>
            <GridItem>
              <FormControl id="title" isRequired mb={8}>
                <FormLabel fontWeight="semibold" fontSize="lg">
                  Problem Title
                </FormLabel>
                <Input
                  type="text"
                  value={title}
                  onChange={(e) => setTitle(e.target.value)}
                  size="lg"
                  focusBorderColor="blue.400"
                />
              </FormControl>

              <FormControl id="difficulty" isRequired mb={8}>
                <FormLabel fontWeight="semibold" fontSize="lg">
                  Difficulty
                </FormLabel>
                <Flex gap={4}>
                  {(["easy", "medium", "hard"] as Difficulty[]).map((level) => {
                    const isSelected = difficulty === level;
                    return (
                      <Box
                        key={level}
                        flex={1}
                        p={4}
                        borderWidth="2px"
                        borderRadius="lg"
                        borderColor={
                          isSelected ? `${level}.500` : borderColorDefault
                        }
                        bg={isSelected ? `${level}.500` : bgColorDefault}
                        color={isSelected ? "white" : textColorDefault}
                        cursor="pointer"
                        transition="all 0.2s"
                        _hover={{
                          borderColor: isSelected
                            ? `${level}.600`
                            : hoverBorderColorDefault,
                          transform: "translateY(-2px)",
                          boxShadow: "md",
                        }}
                        onClick={() => setDifficulty(level)}
                      >
                        <Text
                          textAlign="center"
                          fontWeight="bold"
                          fontSize="md"
                        >
                          {level.charAt(0).toUpperCase() + level.slice(1)}
                        </Text>
                      </Box>
                    );
                  })}
                </Flex>
              </FormControl>

              <FormControl id="description" isRequired mb={8}>
                <FormLabel fontWeight="semibold" fontSize="lg">
                  Problem Description
                </FormLabel>
                <MarkdownEditor value={description} onChange={setDescription} />
              </FormControl>
            </GridItem>

            <GridItem>
              <FormControl id="referenceCode" isRequired mb={8}>
                <FormLabel fontWeight="semibold" fontSize="lg">
                  Reference Implementation
                </FormLabel>
                <Box
                  borderWidth="1px"
                  borderRadius="lg"
                  boxShadow="sm"
                  overflow="hidden"
                >
                  <div style={{ height: "300px" }}>
                    <CodeEditor
                      code={referenceCode}
                      setCode={setReferenceCode}
                      selectedLanguage="cuda"
                    />
                  </div>
                </Box>
              </FormControl>

              <FormControl id="testCases" isRequired mb={8}>
                <FormLabel fontWeight="semibold" fontSize="lg">
                  Test Cases
                </FormLabel>
                <Box
                  borderWidth="1px"
                  borderRadius="lg"
                  boxShadow="sm"
                  overflow="hidden"
                >
                  <div style={{ height: "200px" }}>
                    <CodeEditor
                      code={testCases}
                      setCode={setTestCases}
                      selectedLanguage="cuda"
                    />
                  </div>
                </Box>
              </FormControl>

              <Flex justify="flex-end" mt={8}>
                <Button
                  type="submit"
                  colorScheme="blue"
                  size="lg"
                  isLoading={isSubmitting}
                  loadingText="Submitting..."
                  _hover={{
                    transform: "translateY(-2px)",
                    boxShadow: "lg",
                  }}
                >
                  Submit Problem
                </Button>
              </Flex>
            </GridItem>
          </Grid>
        </Box>
      </Box>
    </Layout>
  );
};

export default AddContributionPage;
