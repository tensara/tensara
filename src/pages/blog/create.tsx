import { useState, useEffect, useRef } from "react";
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
  IconButton,
  Divider,
} from "@chakra-ui/react";
import { Layout } from "~/components/layout";
import { useRouter } from "next/router";
import { useSession, signIn } from "next-auth/react";
import { FiArrowLeft, FiSave, FiTrash2, FiEye, FiEdit } from "react-icons/fi";
import { api } from "~/utils/api";
import ReactMarkdown from "react-markdown";

export default function CreateBlogPost() {
  const router = useRouter();
  const { data: session, status } = useSession();
  const toast = useToast();
  const cancelRef = useRef<HTMLButtonElement>(null);
  const draftRestoredRef = useRef(false);
  const { isOpen, onOpen, onClose } = useDisclosure();

  // Form state
  const [title, setTitle] = useState("");
  const [content, setContent] = useState("");
  const [description, setDescription] = useState("");
  const [lastSaved, setLastSaved] = useState<Date | null>(null);
  const [showPreview, setShowPreview] = useState(false);
  const [draftId, setDraftId] = useState<string | null>(null);

  // Redirect if not authenticated
  useEffect(() => {
    if (status === "unauthenticated") {
      void signIn();
    }
  }, [status]);

  // Load draft from database on mount
  const { data: loadedDraft } = api.blogpost.loadDraft.useQuery(undefined, {
    enabled: !!session?.user?.id,
  });

  useEffect(() => {
    if (loadedDraft && !draftRestoredRef.current) {
      // Populate form fields
      setTitle(loadedDraft.title || "");
      setContent(loadedDraft.content || "");
      setDescription(loadedDraft.excerpt || "");
      setDraftId(loadedDraft.id);
      setLastSaved(new Date(loadedDraft.updatedAt));

      // Show toast only once
      toast({
        title: "Draft restored",
        description: "Your previous draft has been restored.",
        status: "info",
        duration: 3000,
        isClosable: true,
      });

      // Mark as restored
      draftRestoredRef.current = true;
    }
  }, [loadedDraft, toast]);

  const utils = api.useContext();

  // Auto-save draft mutation
  const saveDraftMutation = api.blogpost.saveDraft.useMutation({
    onSuccess: (draft) => {
      setDraftId(draft.id);
      setLastSaved(new Date());
    },
  });

  // Delete draft mutation
  const deleteDraftMutation = api.blogpost.deleteDraft.useMutation({
    onSuccess: () => {
      setDraftId(null);
      void utils.blogpost.loadDraft.invalidate();
    },
  });

  // Auto-save to database with debounce
  useEffect(() => {
    if (!session?.user?.id) return;

    const timer = setTimeout(() => {
      if (title || content) {
        saveDraftMutation.mutate({
          title: title || "Untitled",
          content,
          excerpt: description,
        });
      }
    }, 2000);

    return () => clearTimeout(timer);
  }, [title, content, description, session?.user?.id]);

  const createPost = api.blogpost.create.useMutation({
    onSuccess: async (post) => {
      // If publishing (not just saving draft), delete the draft
      if (post.status === "PUBLISHED" && draftId) {
        try {
          await deleteDraftMutation.mutateAsync({ id: draftId });
        } catch (error) {
          console.error("Failed to delete draft:", error);
          // Continue anyway - the post was successfully published
        }
      }

      // Clear local state
      setDraftId(null);
      setTitle("");
      setContent("");
      setDescription("");

      await utils.blogpost.getAll.invalidate();
      const isPublished = post.status === "PUBLISHED";
      toast({
        title: isPublished ? "Post published!" : "Draft saved!",
        description: isPublished
          ? "Your blog post has been created successfully."
          : "Your draft has been created successfully.",
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

  const handlePublish = (status: "PUBLISHED" | "DRAFT" = "PUBLISHED") => {
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
      status,
    });
  };

  const handleClearDraft = async () => {
    if (draftId) {
      await deleteDraftMutation.mutateAsync({ id: draftId });
      draftRestoredRef.current = false;
    }
    setTitle("");
    setContent("");
    setDescription("");
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

  if (!session) return null;

  return (
    <Layout title="Create Blog Post">
      <Box bg="gray.900" minH="100vh">
        {/* Wider shell + centered reading column */}
        <Container maxW="5xl" py={10}>
          {/* Header */}
          <Flex mb={6} align="center" justify="space-between">
            <HStack spacing={3}>
              <IconButton
                aria-label="Go back"
                icon={<Icon as={FiArrowLeft} />}
                onClick={() => router.back()}
                variant="ghost"
                colorScheme="gray"
                size="md"
              />
              <Box>
                <Heading
                  as="h1"
                  fontSize={{ base: "2xl", md: "3xl" }}
                  fontWeight="800"
                  color="white"
                  letterSpacing="-0.01em"
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

            <HStack spacing={2}>
              <Button
                variant="outline"
                colorScheme="red"
                leftIcon={<Icon as={FiTrash2} />}
                onClick={onOpen}
                size="sm"
              >
                Clear
              </Button>

              <Button
                size="sm"
                bg="#1DA94C"
                color="white"
                _hover={{ bg: "#178a3d" }}
                _active={{ bg: "#156f32" }}
                onClick={() => handlePublish("DRAFT")}
                isLoading={createPost.isPending}
                loadingText="Saving…"
              >
                Save as Draft
              </Button>

              <Button
                size="sm"
                variant="outline"
                borderColor="#1DA94C"
                color="#1DA94C"
                _hover={{ bg: "rgba(29, 169, 76, 0.1)" }}
                _active={{ bg: "rgba(29, 169, 76, 0.2)" }}
                onClick={() => handlePublish("PUBLISHED")}
                isLoading={createPost.isPending}
                loadingText="Publishing…"
              >
                Publish Post
              </Button>
            </HStack>
          </Flex>

          <Divider mb={8} borderColor="whiteAlpha.300" />

          {/* Form wrapper: minimal chrome, no gradients */}
          <Box maxW="900px" mx="auto">
            <VStack spacing={8} align="stretch">
              {/* Title */}
              <FormControl isRequired>
                <FormLabel
                  color="gray.300"
                  fontWeight="600"
                  fontSize="sm"
                  mb={2}
                >
                  Title
                </FormLabel>
                <Input
                  value={title}
                  onChange={(e) => setTitle(e.target.value)}
                  placeholder="Enter an engaging title…"
                  size="md"
                  bg="transparent"
                  borderColor="whiteAlpha.300"
                  color="gray.100"
                  _hover={{ borderColor: "whiteAlpha.400" }}
                  _focus={{
                    borderColor: "green.400",
                    boxShadow: "0 0 0 1px var(--chakra-colors-green-400)",
                  }}
                />
              </FormControl>

              {/* Description */}
              <FormControl>
                <FormLabel
                  color="gray.300"
                  fontWeight="600"
                  fontSize="sm"
                  mb={2}
                >
                  Description{" "}
                  <Text as="span" color="gray.500" fontWeight="normal">
                    (optional)
                  </Text>
                </FormLabel>
                <Input
                  value={description}
                  onChange={(e) => setDescription(e.target.value.slice(0, 200))}
                  placeholder="Brief excerpt for post preview…"
                  size="md"
                  bg="transparent"
                  borderColor="whiteAlpha.300"
                  color="gray.100"
                  _hover={{ borderColor: "whiteAlpha.400" }}
                  _focus={{
                    borderColor: "green.400",
                    boxShadow: "0 0 0 1px var(--chakra-colors-green-400)",
                  }}
                />
                <Text color="gray.500" fontSize="xs" mt={2}>
                  {description.length}/200 characters
                </Text>
              </FormControl>

              {/* Content with Preview Toggle */}
              <FormControl isRequired>
                <Flex justify="space-between" align="center" mb={2}>
                  <FormLabel
                    color="gray.300"
                    fontWeight="600"
                    fontSize="sm"
                    mb={0}
                  >
                    Content
                  </FormLabel>
                  <HStack spacing={1}>
                    <Button
                      size="xs"
                      variant={!showPreview ? "solid" : "ghost"}
                      {...(!showPreview
                        ? {
                            bg: "#1DA94C",
                            color: "white",
                            _hover: { bg: "#178a3d" },
                            _active: { bg: "#156f32" },
                          }
                        : {
                            color: "#1DA94C",
                            _hover: { bg: "rgba(29, 169, 76, 0.1)" },
                            _active: { bg: "rgba(29, 169, 76, 0.2)" },
                          })}
                      leftIcon={<Icon as={FiEdit} />}
                      onClick={() => setShowPreview(false)}
                    >
                      Edit
                    </Button>
                    <Button
                      size="xs"
                      variant={showPreview ? "solid" : "ghost"}
                      {...(showPreview
                        ? {
                            bg: "#1DA94C",
                            color: "white",
                            _hover: { bg: "#178a3d" },
                            _active: { bg: "#156f32" },
                          }
                        : {
                            color: "#1DA94C",
                            _hover: { bg: "rgba(29, 169, 76, 0.1)" },
                            _active: { bg: "rgba(29, 169, 76, 0.2)" },
                          })}
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
                      placeholder="# Your Story\n\nWrite your post in markdown…\n\n**Bold**, *italic*, [links](url), and more!"
                      minH="420px"
                      fontFamily="'JetBrains Mono', ui-monospace, SFMono-Regular"
                      fontSize="sm"
                      bg="transparent"
                      borderColor="whiteAlpha.300"
                      color="gray.100"
                      _hover={{ borderColor: "whiteAlpha.400" }}
                      _focus={{
                        borderColor: "green.400",
                        boxShadow: "0 0 0 1px var(--chakra-colors-green-400)",
                      }}
                    />
                    <Text color="gray.500" fontSize="xs" mt={2}>
                      Tip: Use markdown for faster formatting
                    </Text>
                  </>
                ) : (
                  <Box
                    minH="420px"
                    p={0}
                    color="gray.100"
                    className="markdown-preview"
                    sx={{
                      "& h1": {
                        fontSize: "2xl",
                        fontWeight: "800",
                        mt: 6,
                        mb: 3,
                        color: "white",
                      },
                      "& h2": {
                        fontSize: "xl",
                        fontWeight: "700",
                        mt: 5,
                        mb: 2,
                        color: "white",
                      },
                      "& h3": {
                        fontSize: "lg",
                        fontWeight: "700",
                        mt: 4,
                        mb: 2,
                        color: "gray.100",
                      },
                      "& p": { mb: 3, lineHeight: 1.85, color: "gray.200" },
                      "& ul, & ol": { pl: 6, mb: 3, color: "gray.200" },
                      "& li": { mb: 1.5 },
                      "& code": {
                        bg: "whiteAlpha.200",
                        px: 1.5,
                        py: 0.5,
                        borderRadius: "md",
                        fontFamily:
                          "'JetBrains Mono', ui-monospace, SFMono-Regular",
                      },
                      "& pre": {
                        bg: "gray.800",
                        p: 4,
                        borderRadius: "lg",
                        overflowX: "auto",
                        mb: 4,
                        borderWidth: "1px",
                        borderColor: "whiteAlpha.200",
                      },
                      "& a": {
                        color: "green.300",
                        textDecoration: "underline",
                        _hover: { color: "green.200" },
                      },
                      "& hr": { borderColor: "whiteAlpha.300", my: 6 },
                      "& img": { borderRadius: "md", my: 4 },
                    }}
                  >
                    {content ? (
                      <ReactMarkdown>{content}</ReactMarkdown>
                    ) : (
                      <Box
                        borderWidth="1px"
                        borderColor="whiteAlpha.200"
                        borderRadius="md"
                        p={6}
                      >
                        <Text color="gray.500" fontStyle="italic">
                          No content to preview yet. Start writing to see the
                          preview.
                        </Text>
                      </Box>
                    )}
                  </Box>
                )}
              </FormControl>
            </VStack>
          </Box>
        </Container>

        {/* Clear Confirmation Dialog */}
        <AlertDialog
          isOpen={isOpen}
          leastDestructiveRef={cancelRef}
          onClose={onClose}
        >
          <AlertDialogOverlay backdropFilter="blur(2px)">
            <AlertDialogContent
              bg="gray.900"
              border="1px solid"
              borderColor="whiteAlpha.300"
            >
              <AlertDialogHeader fontSize="lg" fontWeight="bold" color="white">
                Clear Draft
              </AlertDialogHeader>

              <AlertDialogBody color="gray.200">
                Are you sure? This will clear all fields and delete the saved
                draft. This action cannot be undone.
              </AlertDialogBody>

              <AlertDialogFooter>
                <Button ref={cancelRef} onClick={onClose} variant="ghost">
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
