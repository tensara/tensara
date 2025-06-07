import { useState, useEffect } from "react";
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
  Table,
  Thead,
  Tbody,
  Tr,
  Th,
  Td,
  IconButton,
  TabList,
  Tab,
  Tabs,
  TabPanels,
  TabPanel,
  HStack,
} from "@chakra-ui/react";
import { AddIcon, DeleteIcon } from "@chakra-ui/icons";
import PageHeader from "~/components/contribute/PageHeader";
import CodeEditor from "~/components/problem/CodeEditor";
import MarkdownEditor from "~/components/contribute/MarkdownEditor";
import type { Difficulty } from "~/constants/problem";
import { tags as existingTags } from "~/constants/problem";
import CreatableSelect from "react-select/creatable";

type FunctionParameter = {
  name: string;
  type: string;
  pointer: boolean;
  const: boolean;
};

const AddContributionPage: NextPage = () => {
  const { data: session } = useSession();
  const router = useRouter();
  const createContribution = api.contributions.submitNewProblem.useMutation();
  const toast = useToast();

  const [title, setTitle] = useState("");
  const [slug, setSlug] = useState("");
  const [description, setDescription] = useState("");
  const [editorMode, setEditorMode] = useState<"wysiwyg" | "markdown">(
    "wysiwyg"
  );
  const [difficulty, setDifficulty] = useState<Difficulty>("medium");
  const [tags, setTags] = useState<string[]>([]);
  const [referenceSolutionCode, setReferenceSolutionCode] = useState("");
  const [testCases, setTestCases] = useState<
    { input: string; output: string }[]
  >([{ input: "", output: "" }]);
  const [functionParameters, setFunctionParameters] = useState<
    FunctionParameter[]
  >([]);
  const [isSubmitting, setIsSubmitting] = useState(false);

  const cardBg = useColorModeValue("white", "gray.800");
  const cardBorder = useColorModeValue("gray.200", "gray.700");

  const borderColorDefault = useColorModeValue("gray.300", "gray.600");
  const bgColorDefault = useColorModeValue("gray.100", "gray.700");
  const textColorDefault = useColorModeValue("gray.800", "gray.200");
  const hoverBorderColorDefault = useColorModeValue("gray.400", "gray.500");

  // Generate slug from title
  useEffect(() => {
    if (title) {
      const generatedSlug = title
        .toLowerCase()
        .replace(/[^a-z0-9]+/g, "-")
        .replace(/^-*|-*$/g, "");
      setSlug(generatedSlug);
    }
  }, [title]);

  // Add new parameter row
  const addParameter = () => {
    setFunctionParameters([
      ...functionParameters,
      { name: "", type: "", pointer: false, const: false },
    ]);
  };

  // Update parameter field
  const updateParameter = (
    index: number,
    field: keyof FunctionParameter,
    value: string | boolean
  ) => {
    setFunctionParameters((prevParams) => {
      const newParams = [...prevParams];
      newParams[index] = {
        ...newParams[index],
        [field]: field === "name" || field === "type" ? String(value) : value,
      } as FunctionParameter; // Explicitly cast the result
      return newParams;
    });
  };

  // Remove parameter row
  const removeParameter = (index: number) => {
    setFunctionParameters((prevParams) => {
      const newParams = [...prevParams];
      newParams.splice(index, 1);
      return newParams;
    });
  };

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
        contributorGithubUsername: session.user.name ?? "unknown",
        tensaraAppUserId: session.user.id,
        problemDetails: {
          title,
          slug,
          description,
          difficulty: difficulty.toUpperCase() as
            | "EASY"
            | "MEDIUM"
            | "HARD"
            | "EXPERT",
          tags: tags,
          referenceCode: referenceSolutionCode,
          testCases: JSON.stringify(testCases), // Convert test cases array to JSON string
          parameters: functionParameters,
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
                  Title
                </FormLabel>
                <Input
                  type="text"
                  value={title}
                  onChange={(e) => setTitle(e.target.value)}
                  size="lg"
                  focusBorderColor="blue.400"
                  placeholder="eg. Square Matrix Multiplication"
                />
              </FormControl>

              <FormControl id="slug" isRequired mb={8}>
                <FormLabel fontWeight="semibold" fontSize="lg">
                  Slug
                </FormLabel>
                <Input
                  type="text"
                  value={slug}
                  onChange={(e) => setSlug(e.target.value)}
                  size="lg"
                  focusBorderColor="blue.400"
                  placeholder="eg. square-matmul, shows up in the URL: tensara.org/problems/square-matmul"
                />
              </FormControl>

              <FormControl id="tags" mb={8}>
                <FormLabel fontWeight="semibold" fontSize="lg">
                  Tags
                </FormLabel>
                <CreatableSelect
                  isMulti
                  options={existingTags.map((tag) => ({
                    value: tag,
                    label: tag,
                  }))}
                  value={tags.map((tag) => ({ value: tag, label: tag }))}
                  onChange={(newValue) =>
                    setTags(newValue ? newValue.map((v) => v.value) : [])
                  }
                  placeholder="Select or create tags"
                  styles={{
                    control: (provided, state) => ({
                      ...provided,
                      backgroundColor: "#2B3647",
                      border: "none",
                      "&:hover": {
                        backgroundColor: "#2E3849",
                      },
                      "&:focus": {
                        outline: "none !important",
                      },
                      boxShadow: state.isFocused ? "0 0 0 1px #63B3ED" : "none", // blue.400
                      borderRadius: "10px", // Apply rounding to the control
                      padding: "0.25rem 0.25rem", // Adjust padding to match other inputs
                      outline: 0,
                      cursor: "pointer",
                    }),
                    input: (provided) => ({
                      ...provided,
                      color: textColorDefault, // Ensure input text color is visible
                      outline: 0,
                      "&:hover": {
                        outline: "none !important",
                      },
                      "&:focus": {
                        outline: "none !important",
                      },
                    }),
                    placeholder: (provided) => ({
                      ...provided,
                      color: "#A0AEC0", // Set placeholder color to match other inputs
                    }),
                    singleValue: (provided) => ({
                      ...provided,
                      color: textColorDefault,
                    }),
                    multiValue: (provided) => ({
                      ...provided,
                      backgroundColor: "#1E293B", // A darker blue for selected tags
                      borderRadius: "5px",
                    }),
                    multiValueLabel: (provided) => ({
                      ...provided,
                      color: "white",
                    }),
                    multiValueRemove: (provided) => ({
                      ...provided,
                      // color: "whiteAlpha.700",
                      // "&:hover": {
                      //   backgroundColor: "blue.800",
                      //   color: "white",
                      // },
                    }),
                    menu: (provided) => ({
                      ...provided,
                      backgroundColor: "#1A202C", // Explicitly set a dark background color
                      borderColor: cardBorder,
                      boxShadow: "lg",
                      borderRadius: "10px", // Apply rounding to the menu
                      overflow: "hidden", // Ensure content respects border-radius
                    }),
                    option: (provided, state) => ({
                      ...provided,
                      backgroundColor: state.isFocused ? "gray.700" : "#1A202C", // Explicitly set background for options
                      color: textColorDefault,
                      "&:active": {
                        backgroundColor: "#0F1723",
                      },
                      padding: "0.75rem 1rem", // Adjust padding for options
                    }),
                  }}
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
                {/* <Flex
                  justifyContent="space-between"
                  alignItems="center"
                  mb={4}
                  direction={{ base: "column", sm: "row" }}
                  gap={{ base: 4, sm: 0 }}
                > */}
                <Tabs
                  index={editorMode === "wysiwyg" ? 0 : 1}
                  onChange={(index) =>
                    setEditorMode(index === 0 ? "wysiwyg" : "markdown")
                  }
                  variant="soft-rounded"
                  colorScheme="blue"
                >
                  <Flex alignItems="center" justifyContent="space-between">
                    <FormLabel fontWeight="semibold" fontSize="lg">
                      Description
                    </FormLabel>

                    <TabList
                      bg="whiteAlpha.50"
                      p={1}
                      borderRadius="lg"
                      h="40px"
                      gap={1}
                    >
                      <Tab
                        _selected={{ color: "white", bg: "whiteAlpha.100" }}
                        _hover={{
                          bg: "whiteAlpha.100",
                          transition: "all 0.3s ease-in-out",
                        }}
                        color="white"
                        borderRadius="lg"
                        px={4}
                        h="32px"
                        onClick={() => setEditorMode("wysiwyg")}
                      >
                        Visual
                      </Tab>
                      <Tab
                        _selected={{ color: "white", bg: "whiteAlpha.100" }}
                        _hover={{
                          bg: "whiteAlpha.100",
                          transition: "all 0.3s ease-in-out",
                        }}
                        color="white"
                        borderRadius="lg"
                        px={4}
                        h="32px"
                        onClick={() => setEditorMode("markdown")}
                      >
                        Markdown
                      </Tab>
                    </TabList>
                  </Flex>
                  <TabPanels mt={4}>
                    <TabPanel p={0}>
                      <MarkdownEditor
                        value={description}
                        onChange={setDescription}
                        mode={editorMode}
                        onModeChange={setEditorMode}
                      />
                    </TabPanel>
                    <TabPanel p={0}>
                      <MarkdownEditor
                        value={description}
                        onChange={setDescription}
                        mode={editorMode}
                        onModeChange={setEditorMode}
                      />
                    </TabPanel>
                  </TabPanels>
                </Tabs>
                {/* </Flex> */}
              </FormControl>
            </GridItem>

            <GridItem>
              <FormControl id="referenceSolution" isRequired mb={8}>
                <FormLabel fontWeight="semibold" fontSize="lg">
                  Reference Solution (Cuda C++, Triton, Mojo)
                </FormLabel>
                <Box
                  borderWidth="1px"
                  borderRadius="lg"
                  boxShadow="sm"
                  overflow="hidden"
                >
                  <div style={{ height: "200px" }}>
                    <CodeEditor
                      code={referenceSolutionCode}
                      setCode={setReferenceSolutionCode}
                      selectedLanguage="cuda"
                    />
                  </div>
                </Box>
              </FormControl>

              <FormControl id="testCases" isRequired mb={8}>
                <FormLabel fontWeight="semibold" fontSize="lg">
                  Test Cases
                </FormLabel>
                <Flex direction="column" gap={4}>
                  {testCases.map((testCase, index) => (
                    <Box
                      key={index}
                      p={4}
                      borderWidth="1px"
                      borderRadius="lg"
                      borderColor={cardBorder}
                      bg={cardBg}
                      boxShadow="sm"
                    >
                      <Flex justify="space-between" align="center" mb={2}>
                        <Text fontWeight="bold">Test Case {index + 1}</Text>
                        <IconButton
                          aria-label="Remove test case"
                          icon={<DeleteIcon />}
                          onClick={() =>
                            setTestCases((prev) =>
                              prev.filter((_, i) => i !== index)
                            )
                          }
                          size="sm"
                          variant="ghost"
                        />
                      </Flex>
                      <FormControl mb={2}>
                        <FormLabel fontSize="sm">Input</FormLabel>
                        <Input
                          value={testCase.input}
                          onChange={(e) =>
                            setTestCases((prev) =>
                              prev.map((tc, i) =>
                                i === index
                                  ? { ...tc, input: e.target.value }
                                  : tc
                              )
                            )
                          }
                          placeholder="Enter input"
                          size="sm"
                        />
                      </FormControl>
                      <FormControl>
                        <FormLabel fontSize="sm">Expected Output</FormLabel>
                        <Input
                          value={testCase.output}
                          onChange={(e) =>
                            setTestCases((prev) =>
                              prev.map((tc, i) =>
                                i === index
                                  ? { ...tc, output: e.target.value }
                                  : tc
                              )
                            )
                          }
                          placeholder="Enter expected output"
                          size="sm"
                        />
                      </FormControl>
                    </Box>
                  ))}
                  <Button
                    leftIcon={<AddIcon />}
                    onClick={() =>
                      setTestCases((prev) => [
                        ...prev,
                        { input: "", output: "" },
                      ])
                    }
                    mt={2}
                  >
                    Add Test Case
                  </Button>
                </Flex>
              </FormControl>
            </GridItem>
          </Grid>

          <Box mt={10}>
            <Heading size="md" mb={4}>
              Function Parameters
            </Heading>
            <Table variant="simple" mb={4}>
              <Thead>
                <Tr>
                  <Th>Name</Th>
                  <Th>Type</Th>
                  <Th>Pointer</Th>
                  <Th>Const</Th>
                  <Th>Actions</Th>
                </Tr>
              </Thead>
              <Tbody>
                {functionParameters.map((param, index) => (
                  <Tr key={index}>
                    <Td>
                      <Input
                        value={param.name}
                        onChange={(e) =>
                          updateParameter(index, "name", e.target.value)
                        }
                        placeholder="e.g., input_matrix"
                      />
                    </Td>
                    <Td>
                      <Input
                        value={param.type}
                        onChange={(e) =>
                          updateParameter(index, "type", e.target.value)
                        }
                        placeholder="e.g., float"
                      />
                    </Td>
                    <Td>
                      <Select
                        value={param.pointer ? "true" : "false"}
                        onChange={(e) =>
                          updateParameter(
                            index,
                            "pointer",
                            e.target.value === "true"
                          )
                        }
                        size="sm"
                        variant="filled"
                        focusBorderColor="blue.400"
                      >
                        <option value="true">Yes</option>
                        <option value="false">No</option>
                      </Select>
                    </Td>
                    <Td>
                      <Select
                        value={param.const ? "true" : "false"}
                        onChange={(e) =>
                          updateParameter(
                            index,
                            "const",
                            e.target.value === "true"
                          )
                        }
                        size="sm"
                        variant="filled"
                        focusBorderColor="blue.400"
                      >
                        <option value="true">Yes</option>
                        <option value="false">No</option>
                      </Select>
                    </Td>
                    <Td>
                      <IconButton
                        aria-label="Delete parameter"
                        icon={<DeleteIcon />}
                        onClick={() => removeParameter(index)}
                        size="sm"
                        variant="ghost"
                        colorScheme="red"
                      />
                    </Td>
                  </Tr>
                ))}
              </Tbody>
            </Table>
            <Button
              leftIcon={<AddIcon />}
              onClick={addParameter}
              colorScheme="blue"
              size="md"
            >
              Add Parameter
            </Button>
          </Box>

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
        </Box>
      </Box>
    </Layout>
  );
};

export default AddContributionPage;
