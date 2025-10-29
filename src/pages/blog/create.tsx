import { useState, useEffect } from "react";
import {
  Box,
  Container,
  Heading,
  VStack,
  Text,
  Button,
  FormControl,
  FormLabel,
  Input,
  Textarea,
  Flex,
  HStack,
  Icon,
  Checkbox,
  useToast,
  AlertDialog,
  AlertDialogBody,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogContent,
  AlertDialogOverlay,
  useDisclosure,
  Tabs,
  TabList,
  TabPanels,
  Tab,
  TabPanel,
  IconButton,
} from "@chakra-ui/react";
import { Layout } from "~/components/layout";
import { useRouter } from "next/router";
import { useSession, signIn } from "next-auth/react";
import { FiArrowLeft, FiSave, FiTrash2, FiEye, FiEdit } from "react-icons/fi";
import { api } from "~/utils/api";
import ReactMarkdown from "react-markdown";
import { useRef } from "react";

interface BlogDraft {
  title: string;
  content: string;
  description?: string;
  isPublished: boolean;
  tags?: string[];
  lastSaved: string;
}

export default function CreateBlogPost() {
  const router = useRouter();
  const { data: session, status } = useSession();
  const toast = useToast();
  const cancelRef = useRef<HTMLButtonElement>(null);
  const { isOpen, onOpen, onClose } = useDisclosure();

  // Form state
  const [title, setTitle] = useState("");
  const [content, setContent] = useState("");
  const [description, setDescription] = useState("");
  const [isPublished, setIsPublished] = useState(true);
  const [lastSaved, setLastSaved] = useState<Date | null>(null);
  const [showPreview, setShowPreview] = useState(false);

  // Redirect if not authenticated
  useEffect(() => {
    if (status === "unauthenticated") {
      void signIn();
    }
  }, [status]);

  // Load draft from localStorage on mount
  useEffect(() => {
    if (session?.user?.id) {
      const draftKey = `blog-draft-${session.user.id}`;
      const savedDraft = localStorage.getItem(draftKey);
      if (savedDraft) {
        try {
          const draft: BlogDraft = JSON.parse(savedDraft);
          setTitle(draft.title || "");
          setContent(draft.content || "");
          setDescription(draft.description || "");
          setIsPublished(draft.isPublished ?? true);
          setLastSaved(new Date(draft.lastSaved));
          toast({
            title: "Draft restored",
            description: "Your previous draft has been restored.",
            status: "info",
            duration: 3000,
            isClosable: true,
          });
        } catch (e) {
          console.error("Failed to parse draft:", e);
        }
      }
    }
  }, [session?.user?.id, toast]);

  // Auto-save to localStorage
  useEffect(() => {
    if (!session?.user?.id) return;

    const timer = setTimeout(() => {
      if (title || content) {
        const draftKey = `blog-draft-${session.user.id}`;
        const draft: BlogDraft = {
          title,
          content,
          description,
          isPublished,
          lastSaved: new Date().toISOString(),
        };
        localStorage.setItem(draftKey, JSON.stringify(draft));
        setLastSaved(new Date());
      }
    }, 2000); // 2 second debounce

    return () => clearTimeout(timer);
  }, [title, content, description, isPublished, session?.user?.id]);

  const utils = api.useContext();
  const createPost = api.blogpost.create.useMutation({
    onSuccess: async (post) => {
      // Clear localStorage draft
      if (session?.user?.id) {
        const draftKey = `blog-draft-${session.user.id}`;
        localStorage.removeItem(draftKey);
      }
      await utils.blogpost.getAll.invalidate();
      toast({
        title: "Post published!",
        description: "Your blog post has been created successfully.",
        status: "success",
        duration: 3000,
        isClosable: true,
      });
      void router.push(`/blog/${post.slug ?? post.id}`);
    },
    onError: (error) => {
      toast({
        title: "Error creating post",
        description: error.message,
        status: "error",
        duration: 5000,
        isClosable: true,
      });
    },
  });

  const handlePublish = () => {
    if (!title.trim()) {
      toast({
        title: "Title required",
        description: "Please enter a title for your post.",
        status: "warning",
        duration: 3000,
        isClosable: true,
      });
      return;
    }

    if (!content.trim()) {
      toast({
        title: "Content required",
        description: "Please enter content for your post.",
        status: "warning",
        duration: 3000,
        isClosable: true,
      });
      return;
    }

    createPost.mutate({
      title,
      content,
      excerpt: description || undefined,
      status: isPublished ? "PUBLISHED" : "DRAFT",
    });
  };

  const handleSaveDraft = () => {
    if (session?.user?.id) {
      const draftKey = `blog-draft-${session.user.id}`;
      const draft: BlogDraft = {
        title,
        content,
        description,
        isPublished,
        lastSaved: new Date().toISOString(),
      };
      localStorage.setItem(draftKey, JSON.stringify(draft));
      setLastSaved(new Date());
      toast({
        title: "Draft saved",
        description: "Your draft has been saved to local storage.",
        status: "success",
        duration: 2000,
        isClosable: true,
      });
    }
  };

  const handleClearDraft = () => {
    if (session?.user?.id) {
      const draftKey = `blog-draft-${session.user.id}`;
      localStorage.removeItem(draftKey);
    }
    setTitle("");
    setContent("");
    setDescription("");
    setIsPublished(true);
    setLastSaved(null);
    onClose();
    toast({
      title: "Draft cleared",
      description: "All fields have been cleared and the draft deleted.",
      status: "info",
      duration: 2000,
      isClosable: true,
    });
  };

  const formatTime = (date: Date) => {
    return date.toLocaleTimeString("en-US", {
      hour: "2-digit",
      minute: "2-digit",
    });
  };

  // Don't render the form until we know the session status
  if (status === "loading") {
    return (
      <Layout title="Create Blog Post">
        <Box bg="gray.900" minH="100vh">
          <Container maxW="5xl" py={12}>
            <Flex justify="center" align="center" minH="60vh">
              <Text color="gray.500" fontSize="lg">
                Loading...
              </Text>
            </Flex>
          </Container>
        </Box>
      </Layout>
    );
  }

  if (!session) {
    return null;
  }

  return (
    <Layout title="Create Blog Post">
      <Box bg="gray.900" minH="100vh">
        <Container maxW="5xl" py={12}>
          {/* Header */}
          <Flex mb={8} align="center" justify="space-between">
            <HStack spacing={4}>
              <IconButton
                aria-label="Go back"
                icon={<Icon as={FiArrowLeft} />}
                onClick={() => router.back()}
                variant="ghost"
                colorScheme="gray"
                size="lg"
              />
              <Box>
                <Heading
                  as="h1"
                  fontSize={{ base: "3xl", md: "4xl" }}
                  fontWeight="800"
                  bgGradient="linear(to-r, green.400, green.600)"
                  bgClip="text"
                >
                  Create Blog Post
                </Heading>
                {lastSaved && (
                  <Text color="gray.500" fontSize="sm" mt={1}>
                    Draft saved at {formatTime(lastSaved)}
                  </Text>
                )}
              </Box>
            </HStack>

            <HStack spacing={3}>
              <Button
                variant="outline"
                colorScheme="red"
                leftIcon={<Icon as={FiTrash2} />}
                onClick={onOpen}
                size="md"
              >
                Clear
              </Button>
              <Button
                variant="outline"
                colorScheme="green"
                leftIcon={<Icon as={FiSave} />}
                onClick={handleSaveDraft}
                size="md"
              >
                Save Draft
              </Button>
            </HStack>
          </Flex>

          {/* Form */}
          <Box
            bg="whiteAlpha.50"
            backdropFilter="blur(10px)"
            borderRadius="2xl"
            borderWidth="1px"
            borderColor="whiteAlpha.100"
            p={8}
          >
            <VStack spacing={6} align="stretch">
              {/* Title */}
              <FormControl isRequired>
                <FormLabel color="gray.300" fontWeight="600" fontSize="lg">
                  Title
                </FormLabel>
                <Input
                  value={title}
                  onChange={(e) => setTitle(e.target.value)}
                  placeholder="Enter an engaging title..."
                  size="lg"
                  bg="whiteAlpha.50"
                  borderColor="whiteAlpha.200"
                  _hover={{ borderColor: "green.500" }}
                  _focus={{
                    borderColor: "green.500",
                    boxShadow: "0 0 0 1px var(--chakra-colors-green-500)",
                  }}
                />
              </FormControl>

              {/* Description */}
              <FormControl>
                <FormLabel color="gray.300" fontWeight="600" fontSize="lg">
                  Description{" "}
                  <Text as="span" color="gray.500" fontWeight="normal">
                    (optional)
                  </Text>
                </FormLabel>
                <Input
                  value={description}
                  onChange={(e) => setDescription(e.target.value.slice(0, 200))}
                  placeholder="Brief excerpt for post preview..."
                  size="lg"
                  bg="whiteAlpha.50"
                  borderColor="whiteAlpha.200"
                  _hover={{ borderColor: "green.500" }}
                  _focus={{
                    borderColor: "green.500",
                    boxShadow: "0 0 0 1px var(--chakra-colors-green-500)",
                  }}
                />
                <Text color="gray.500" fontSize="sm" mt={2}>
                  {description.length}/200 characters
                </Text>
              </FormControl>

              {/* Content with Preview Toggle */}
              <FormControl isRequired>
                <Flex justify="space-between" align="center" mb={2}>
                  <FormLabel
                    color="gray.300"
                    fontWeight="600"
                    fontSize="lg"
                    mb={0}
                  >
                    Content
                  </FormLabel>
                  <HStack>
                    <Button
                      size="sm"
                      variant={!showPreview ? "solid" : "ghost"}
                      colorScheme="green"
                      leftIcon={<Icon as={FiEdit} />}
                      onClick={() => setShowPreview(false)}
                    >
                      Edit
                    </Button>
                    <Button
                      size="sm"
                      variant={showPreview ? "solid" : "ghost"}
                      colorScheme="green"
                      leftIcon={<Icon as={FiEye} />}
                      onClick={() => setShowPreview(true)}
                    >
                      Preview
                    </Button>
                  </HStack>
                </Flex>

                {!showPreview ? (
                  <>
                    <Textarea
                      value={content}
                      onChange={(e) => setContent(e.target.value)}
                      placeholder="# Your Story&#10;&#10;Write your post in markdown...&#10;&#10;**Bold text**, *italic*, [links](url), and more!"
                      minH="400px"
                      fontFamily="'JetBrains Mono', monospace"
                      fontSize="sm"
                      bg="whiteAlpha.50"
                      borderColor="whiteAlpha.200"
                      _hover={{ borderColor: "green.500" }}
                      _focus={{
                        borderColor: "green.500",
                        boxShadow: "0 0 0 1px var(--chakra-colors-green-500)",
                      }}
                    />
                    <Text color="gray.500" fontSize="xs" mt={2}>
                      💡 Tip: Use markdown syntax for formatting
                    </Text>
                  </>
                ) : (
                  <Box
                    minH="400px"
                    p={6}
                    bg="whiteAlpha.50"
                    borderRadius="lg"
                    borderWidth="1px"
                    borderColor="whiteAlpha.200"
                    overflowY="auto"
                    css={{
                      "& h1": {
                        fontSize: "2xl",
                        fontWeight: "bold",
                        marginBottom: "0.5em",
                      },
                      "& h2": {
                        fontSize: "xl",
                        fontWeight: "bold",
                        marginBottom: "0.5em",
                        marginTop: "1em",
                      },
                      "& h3": {
                        fontSize: "lg",
                        fontWeight: "bold",
                        marginBottom: "0.5em",
                        marginTop: "1em",
                      },
                      "& p": { marginBottom: "1em", lineHeight: "1.7" },
                      "& ul, & ol": {
                        marginLeft: "1.5em",
                        marginBottom: "1em",
                      },
                      "& li": { marginBottom: "0.5em" },
                      "& code": {
                        background: "rgba(0,0,0,0.3)",
                        padding: "0.2em 0.4em",
                        borderRadius: "0.25em",
                        fontFamily: "'JetBrains Mono', monospace",
                      },
                      "& pre": {
                        background: "rgba(0,0,0,0.3)",
                        padding: "1em",
                        borderRadius: "0.5em",
                        marginBottom: "1em",
                        overflowX: "auto",
                      },
                      "& a": {
                        color: "var(--chakra-colors-green-400)",
                        textDecoration: "underline",
                      },
                      "& strong": { fontWeight: "bold" },
                      "& em": { fontStyle: "italic" },
                    }}
                  >
                    {content ? (
                      <ReactMarkdown>{content}</ReactMarkdown>
                    ) : (
                      <Text color="gray.500" fontStyle="italic">
                        No content to preview yet. Start writing to see the
                        preview.
                      </Text>
                    )}
                  </Box>
                )}
              </FormControl>

              {/* Is Published Checkbox */}
              <FormControl>
                <Checkbox
                  isChecked={isPublished}
                  onChange={(e) => setIsPublished(e.target.checked)}
                  colorScheme="green"
                  size="lg"
                >
                  <Text color="gray.300" fontWeight="600">
                    Publish immediately
                  </Text>
                </Checkbox>
                <Text color="gray.500" fontSize="sm" mt={1} ml={8}>
                  Uncheck to save as draft
                </Text>
              </FormControl>

              {/* Action Buttons */}
              <Flex justify="flex-end" pt={4}>
                <Button
                  size="lg"
                  bgGradient="linear(to-r, green.600, green.800)"
                  color="white"
                  _hover={{
                    bgGradient: "linear(to-r, green.700, green.900)",
                    transform: "translateY(-2px)",
                    shadow: "xl",
                  }}
                  transition="all 0.2s"
                  onClick={handlePublish}
                  isLoading={createPost.isPending}
                  loadingText={isPublished ? "Publishing..." : "Saving..."}
                  px={12}
                >
                  {isPublished ? "Publish Post" : "Save as Draft"}
                </Button>
              </Flex>
            </VStack>
          </Box>
        </Container>

        {/* Clear Confirmation Dialog */}
        <AlertDialog
          isOpen={isOpen}
          leastDestructiveRef={cancelRef}
          onClose={onClose}
        >
          <AlertDialogOverlay backdropFilter="blur(10px)">
            <AlertDialogContent bg="gray.800" borderColor="whiteAlpha.200">
              <AlertDialogHeader fontSize="lg" fontWeight="bold">
                Clear Draft
              </AlertDialogHeader>

              <AlertDialogBody>
                Are you sure? This will clear all fields and delete the saved
                draft. This action cannot be undone.
              </AlertDialogBody>

              <AlertDialogFooter>
                <Button ref={cancelRef} onClick={onClose}>
                  Cancel
                </Button>
                <Button colorScheme="red" onClick={handleClearDraft} ml={3}>
                  Clear
                </Button>
              </AlertDialogFooter>
            </AlertDialogContent>
          </AlertDialogOverlay>
        </AlertDialog>
      </Box>
    </Layout>
  );
}
