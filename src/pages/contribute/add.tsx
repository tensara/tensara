import { useEffect, useState } from "react";
import { type NextPage } from "next";
import { useSession } from "next-auth/react";
import { useRouter } from "next/router";
import { api } from "~/utils/api";
import TurndownService from "turndown";
import showdown from "showdown";
import { Layout } from "~/components/layout";
import { useContributionFormPersistence } from "~/hooks/useContributionFormPersistence";
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
  IconButton,
  TabList,
  Tab,
  Tabs,
  TabPanels,
  TabPanel,
  HStack,
  Textarea,
  Accordion,
  AccordionItem,
  AccordionButton,
  AccordionPanel,
  AccordionIcon,
  Modal,
  ModalOverlay,
  ModalContent,
  ModalHeader,
  ModalFooter,
  ModalBody,
  ModalCloseButton,
} from "@chakra-ui/react";
import { AddIcon, DeleteIcon } from "@chakra-ui/icons";
import { MdRestartAlt } from "react-icons/md";
import PageHeader from "~/components/contribute/PageHeader";
import CodeEditor from "~/components/problem/CodeEditor";
import MarkdownEditor from "~/components/contribute/MarkdownEditor";
import type { Difficulty } from "~/constants/problem";
import { tags as existingTags } from "~/constants/problem";
import CreatableSelect from "react-select/creatable";

const AddContributionPage: NextPage = () => {
  const { data: session } = useSession();
  const router = useRouter();
  const createContribution = api.contributions.submitNewProblem.useMutation();
  const toast = useToast();

  const {
    title,
    setTitle,
    slug,
    setSlug,
    description,
    setDescription,
    difficulty,
    setDifficulty,
    tags,
    setTags,
    referenceSolutionCode,
    setReferenceSolutionCode,
    testCases,
    addTestCase,
    updateTestCase,
    removeTestCase,
    handleReset,
  } = useContributionFormPersistence();

  const turndownService = new TurndownService();
  const showdownConverter = new showdown.Converter();

  const [editorMode, setEditorMode] = useState<"wysiwyg" | "markdown">(
    "markdown"
  );
  const [accordionIndex, setAccordionIndex] = useState<number | number[]>(0);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isClient, setIsClient] = useState(false);
  const [isResetModalOpen, setIsResetModalOpen] = useState(false);

  const cardBg = useColorModeValue("white", "gray.800");
  const cardBorder = useColorModeValue("gray.200", "gray.700");

  const borderColorDefault = useColorModeValue("gray.300", "gray.600");
  const bgColorDefault = useColorModeValue("gray.100", "gray.700");
  const textColorDefault = useColorModeValue("gray.800", "gray.200");
  const hoverBorderColorDefault = useColorModeValue("gray.400", "gray.500");

  useEffect(() => {
    setIsClient(true);
  }, []);

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
        <Flex justifyContent="space-between" alignItems="center" mb={4}>
          <PageHeader
            title="Add New Problem"
            description="Submit a new problem to Tensara for review"
          />
          <Flex justify="flex-end" gap={4}>
            <Button
              type="button"
              colorScheme="gray"
              variant="ghost"
              size="lg"
              onClick={() => setIsResetModalOpen(true)}
              leftIcon={<MdRestartAlt />}
            >
              Reset Form
            </Button>
            <Button
              type="submit"
              colorScheme="green"
              size="lg"
              isLoading={isSubmitting}
              loadingText="Submitting..."
              form="contribution-form"
            >
              Submit Problem
            </Button>
          </Flex>
        </Flex>

        <Box
          as="form"
          id="contribution-form"
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
                  onChange={(index) => {
                    setEditorMode(index === 0 ? "wysiwyg" : "markdown");
                  }}
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
                      >
                        Markdown
                      </Tab>
                    </TabList>
                  </Flex>
                </Tabs>
                <MarkdownEditor
                  value={description}
                  onChange={setDescription}
                  mode={editorMode}
                  onModeChange={setEditorMode}
                />
                {/* </Flex> */}
              </FormControl>
            </GridItem>

            <GridItem>
              <FormControl id="referenceSolution" isRequired mb={8}>
                <FormLabel fontWeight="semibold" fontSize="lg">
                  Reference Solution (PyTorch)
                </FormLabel>
                <Box
                  borderWidth="1px"
                  borderRadius="lg"
                  boxShadow="sm"
                  overflow="hidden"
                  p={4}
                  bg={cardBg}
                  borderColor={cardBorder}
                >
                  <div style={{ height: "400px" }}>
                    <CodeEditor
                      code={referenceSolutionCode}
                      setCode={setReferenceSolutionCode}
                      selectedLanguage="python"
                    />
                  </div>
                </Box>
              </FormControl>

              <FormControl id="testCases" isRequired mb={8}>
                <FormLabel fontWeight="semibold" fontSize="lg">
                  Test Cases
                </FormLabel>
                {isClient && (
                  <Accordion
                    allowToggle
                    index={accordionIndex}
                    onChange={(newIndex) => setAccordionIndex(newIndex)}
                  >
                    {testCases.map((testCase, index) => (
                      <AccordionItem
                        key={index}
                        mt={index > 0 ? 4 : 0}
                        borderWidth="1px"
                        borderRadius="lg"
                        borderColor={cardBorder}
                        bg={cardBg}
                        boxShadow="sm"
                      >
                        <h2>
                          <AccordionButton _expanded={{ bg: cardBg }}>
                            <Box as="span" flex="1" textAlign="left">
                              <Text fontWeight="bold">
                                Test Case {index + 1}
                              </Text>
                            </Box>
                            <IconButton
                              aria-label="Remove test case"
                              icon={<DeleteIcon />}
                              onClick={(e) => {
                                e.stopPropagation();
                                removeTestCase(index);
                                setAccordionIndex(
                                  testCases.length - 1 > 0
                                    ? testCases.length - 2
                                    : 0
                                );
                              }}
                              size="sm"
                              variant="ghost"
                            />
                            <AccordionIcon />
                          </AccordionButton>
                        </h2>
                        <AccordionPanel pb={4}>
                          <Flex direction="row" gap={4}>
                            <FormControl flex={1}>
                              <FormLabel fontSize="sm">Input</FormLabel>
                              <Textarea
                                value={testCase.input}
                                onChange={(e) =>
                                  updateTestCase(index, {
                                    input: e.target.value,
                                  })
                                }
                                placeholder="Input"
                                size="sm"
                                minHeight="120px"
                                resize="vertical"
                                borderRadius="lg"
                                bg={useColorModeValue("gray.100", "gray.700")}
                                focusBorderColor="blue.400"
                              />
                            </FormControl>
                            <FormControl flex={1}>
                              <FormLabel fontSize="sm">
                                Expected Output
                              </FormLabel>
                              <Textarea
                                value={testCase.output}
                                onChange={(e) =>
                                  updateTestCase(index, {
                                    output: e.target.value,
                                  })
                                }
                                placeholder="Expected Output"
                                size="sm"
                                minHeight="120px"
                                resize="vertical"
                                borderRadius="lg"
                                bg={useColorModeValue("gray.100", "gray.700")}
                                focusBorderColor="blue.400"
                              />
                            </FormControl>
                          </Flex>
                        </AccordionPanel>
                      </AccordionItem>
                    ))}
                  </Accordion>
                )}
                <Button
                  leftIcon={<AddIcon />}
                  onClick={() => {
                    addTestCase();
                    setAccordionIndex(testCases.length);
                  }}
                  mt={4}
                >
                  Add Test Case
                </Button>
              </FormControl>
            </GridItem>
          </Grid>
        </Box>
      </Box>

      <Modal
        isOpen={isResetModalOpen}
        onClose={() => setIsResetModalOpen(false)}
        isCentered
      >
        <ModalOverlay />
        <ModalContent bg={cardBg}>
          <ModalHeader>Confirm Reset</ModalHeader>
          <ModalCloseButton />
          <ModalBody>
            <Text>
              Are you sure you want to reset the form? All unsaved progress will
              be lost.
            </Text>
          </ModalBody>
          <ModalFooter>
            <Button
              variant="ghost"
              mr={3}
              onClick={() => setIsResetModalOpen(false)}
            >
              Cancel
            </Button>
            <Button
              colorScheme="red"
              onClick={() => {
                handleReset();
                setIsResetModalOpen(false);
              }}
            >
              Confirm Reset
            </Button>
          </ModalFooter>
        </ModalContent>
      </Modal>
    </Layout>
  );
};

export default AddContributionPage;
