import { useRouter } from "next/router";
import type { RouterOutputs } from "~/utils/api";
import { useEffect, useState } from "react";
import { Layout } from "~/components/layout";
import {
  Box,
  Container,
  Heading,
  Text,
  Button,
  VStack,
  Flex,
  Avatar,
  HStack,
  Modal,
  ModalOverlay,
  ModalContent,
  ModalHeader,
  ModalBody,
  ModalFooter,
  ModalCloseButton,
  FormControl,
  FormLabel,
  Input,
  Textarea,
  useDisclosure,
  Icon,
  IconButton,
} from "@chakra-ui/react";
import { api } from "~/utils/api";
import { useSession, signIn } from "next-auth/react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  FiEdit3,
  FiTrash2,
  FiCalendar,
  FiArrowLeft,
  FiThumbsUp,
  FiMessageSquare,
  FiSend,
} from "react-icons/fi";
import Link from "next/link";

// Typed helper aliases for tRPC outputs
type Post = RouterOutputs["blogpost"]["getById"];
type CommentType = RouterOutputs["comments"]["getByPost"][number];

export default function TestBlogPost() {
  const router = useRouter();
  const { id } = router.query;
  const { data: session } = useSession();

  const { data: post, isLoading } = api.blogpost.getById.useQuery(
    typeof id === "string" ? { id } : { id: "" },
    { enabled: typeof id === "string" }
  );

  const { isOpen, onOpen, onClose } = useDisclosure();
  const [title, setTitle] = useState("");
  const [content, setContent] = useState("");

  useEffect(() => {
    if (post) {
      setTitle(post.title ?? "");
      setContent(post.content ?? "");
    }
  }, [post]);

  const utils = api.useContext();
  const update = api.blogpost.update.useMutation({
    onSuccess: () => {
      void utils.blogpost.getById.invalidate();
      void utils.blogpost.getAll.invalidate();
      onClose();
    },
  });

  const del = api.blogpost.delete.useMutation({
    onSuccess: () => {
      void utils.blogpost.getAll.invalidate();
      void router.push("/blog");
    },
  });

  // Post-related hooks (must be declared before any early returns)
  const isAuthor = session?.user?.id === post?.author?.id;

  // Post upvote count and toggle
  const { data: postUpvoteCount } = api.postUpvote.count.useQuery(
    { postId: post?.id ?? "" },
    { enabled: !!post }
  );
  // whether current user has upvoted (protected)
  const { data: postHasUpvoted } = api.postUpvote.hasUpvoted.useQuery(
    { postId: post?.id ?? "" },
    { enabled: !!post && !!session }
  );
  const postUpvoteToggle = api.postUpvote.toggle.useMutation({
    onSuccess: async () => {
      if (post?.id)
        await utils.postUpvote.count.invalidate({ postId: post.id });
      if (post?.id && session)
        await utils.postUpvote.hasUpvoted.invalidate({ postId: post.id });
      await utils.blogpost.getById.invalidate();
    },
  });

  // Comments
  const { data: comments } = api.comments.getByPost.useQuery(
    { postId: post?.id ?? "" },
    { enabled: !!post }
  );
  const createComment = api.comments.create.useMutation({
    onSuccess: async () => {
      if (post?.id)
        await utils.comments.getByPost.invalidate({ postId: post.id });
      setNewComment("");
    },
  });
  const deleteComment = api.comments.delete.useMutation({
    onSuccess: async () => {
      if (post?.id)
        await utils.comments.getByPost.invalidate({ postId: post.id });
    },
  });

  // Local state for new comment
  const [newComment, setNewComment] = useState("");

  // Comment item component (defined here but used later)
  function CommentItem({ comment }: { comment: CommentType }) {
    const [replyOpen, setReplyOpen] = useState(false);
    const [replyText, setReplyText] = useState("");
    const cuCount = api.commentUpvote.count.useQuery(
      { commentId: comment.id },
      { enabled: !!comment?.id }
    );
    const { data: cuHasUpvoted } = api.commentUpvote.hasUpvoted.useQuery(
      { commentId: comment.id },
      { enabled: !!comment?.id && !!session }
    );
    const cuToggle = api.commentUpvote.toggle.useMutation({
      onSuccess: async () => {
        if (comment?.id) {
          await utils.commentUpvote.count.invalidate({ commentId: comment.id });
          if (session)
            await utils.commentUpvote.hasUpvoted.invalidate({
              commentId: comment.id,
            });
        }
        if (post?.id)
          await utils.comments.getByPost.invalidate({ postId: post.id });
      },
    });

    const isCommentAuthor = session?.user?.id === comment.author?.id;

    return (
      <Box
        key={comment.id}
        p={4}
        borderRadius="lg"
        bg="whiteAlpha.50"
        borderWidth="1px"
        borderColor="whiteAlpha.100"
      >
        <Flex align="start" justify="space-between">
          <HStack align="start">
            <Avatar
              size="sm"
              src={comment.author?.image ?? undefined}
              name={comment.author?.name ?? undefined}
              bg="green.600"
            />
            <Box>
              <Text color="white" fontWeight="600" fontSize="sm">
                {comment.author?.name}
              </Text>
              <Text color="gray.400" fontSize="xs">
                {new Date(comment.createdAt).toLocaleString()}
              </Text>
            </Box>
          </HStack>

          <HStack spacing={2}>
            <Button
              variant={cuHasUpvoted?.hasUpvoted ? "solid" : "ghost"}
              leftIcon={<Icon as={FiThumbsUp} />}
              onClick={() => cuToggle.mutate({ commentId: comment.id })}
              colorScheme="green"
            >
              {cuCount.data?.count ?? 0}
            </Button>
            <Button
              variant="ghost"
              size="sm"
              leftIcon={<Icon as={FiMessageSquare} />}
              onClick={() => setReplyOpen((s) => !s)}
            >
              Reply
            </Button>
            {isCommentAuthor && (
              <IconButton
                aria-label="Delete comment"
                icon={<Icon as={FiTrash2} />}
                size="sm"
                variant="ghost"
                colorScheme="red"
                onClick={() => deleteComment.mutate({ id: comment.id })}
              />
            )}
          </HStack>
        </Flex>
        <Box mt={3} color="gray.300">
          <Text whiteSpace="pre-wrap">{comment.content}</Text>
        </Box>

        {replyOpen && (
          <Box mt={3}>
            <Textarea
              value={replyText}
              onChange={(e) => setReplyText(e.target.value)}
              rows={2}
              placeholder="Write a reply..."
              bg="whiteAlpha.50"
              borderColor="whiteAlpha.100"
              color="gray.100"
            />
            <Flex justify="flex-end" mt={2} gap={2}>
              <Button
                size="sm"
                variant="ghost"
                onClick={() => {
                  setReplyOpen(false);
                  setReplyText("");
                }}
              >
                Cancel
              </Button>
              <Button
                size="sm"
                colorScheme="green"
                onClick={() => {
                  if (!replyText.trim() || !post?.id) return;
                  createComment.mutate({
                    postId: post.id,
                    content: replyText,
                    parentCommentId: comment.id,
                  });
                  setReplyText("");
                  setReplyOpen(false);
                }}
              >
                Reply
              </Button>
            </Flex>
          </Box>
        )}

        {/* Render immediate children replies (compact) - children are included by the query */}
        {comment.children?.map((child: CommentType) => (
          <Box
            key={child.id}
            mt={3}
            ml={8}
            borderLeftWidth="1px"
            borderColor="whiteAlpha.50"
            pl={4}
          >
            <CommentItem comment={child as CommentType} />
          </Box>
        ))}
      </Box>
    );
  }

  if (isLoading)
    return (
      <Layout title="Loading...">
        <Box bg="gray.900" minH="100vh" py={10}>
          <Container maxW="4xl">
            <Text color="gray.500" textAlign="center" fontSize="lg">
              Loading post...
            </Text>
          </Container>
        </Box>
      </Layout>
    );

  if (!post)
    return (
      <Layout title="Not found">
        <Box bg="gray.900" minH="100vh" py={10}>
          <Container maxW="4xl">
            <VStack spacing={4}>
              <Text color="gray.500" fontSize="xl">
                Post not found
              </Text>
              <Button
                as={Link}
                href="/blog"
                leftIcon={<Icon as={FiArrowLeft} />}
                variant="ghost"
                colorScheme="green"
              >
                Back to posts
              </Button>
            </VStack>
          </Container>
        </Box>
      </Layout>
    );

  return (
    <Layout title={post.title}>
      <Box bg="gray.900" minH="100vh">
        <Container maxW="4xl" py={12}>
          {/* Back Button */}
          <Button
            as={Link}
            href="/blog"
            leftIcon={<Icon as={FiArrowLeft} />}
            variant="ghost"
            colorScheme="green"
            mb={8}
            _hover={{ bg: "whiteAlpha.100" }}
          >
            Back to posts
          </Button>

          <VStack align="stretch" spacing={8}>
            {/* Article Header */}
            <Box>
              <Heading
                fontSize={{ base: "3xl", md: "5xl" }}
                fontWeight="800"
                mb={6}
                bgGradient="linear(to-r, green.400, green.600, green.800)"
                bgClip="text"
                lineHeight="shorter"
              >
                {post.title}
              </Heading>

              <Flex
                align="center"
                justify="space-between"
                flexWrap="wrap"
                gap={4}
                pb={6}
                borderBottomWidth="1px"
                borderColor="whiteAlpha.200"
              >
                <HStack spacing={4}>
                  <Avatar
                    size="md"
                    src={post.author?.image ?? undefined}
                    name={post.author?.name ?? undefined}
                    bg="green.600"
                  />
                  <Box>
                    <Text color="white" fontWeight="600" fontSize="lg">
                      {post.author?.name}
                    </Text>
                    <HStack spacing={2} color="gray.500" fontSize="sm">
                      <Icon as={FiCalendar} />
                      <Text>
                        {new Date(post.createdAt).toLocaleDateString("en-US", {
                          month: "long",
                          day: "numeric",
                          year: "numeric",
                        })}
                      </Text>
                    </HStack>
                  </Box>
                </HStack>

                <HStack>
                  <Button
                    variant={postHasUpvoted?.hasUpvoted ? "solid" : "ghost"}
                    leftIcon={<Icon as={FiThumbsUp} />}
                    onClick={() => postUpvoteToggle.mutate({ postId: post.id })}
                    colorScheme="green"
                    size="lg"
                  >
                    {postUpvoteCount?.count ?? 0}
                  </Button>

                  {isAuthor && (
                    <HStack>
                      <IconButton
                        aria-label="Edit post"
                        icon={<Icon as={FiEdit3} />}
                        onClick={onOpen}
                        colorScheme="green"
                        variant="ghost"
                        size="lg"
                      />
                      <IconButton
                        aria-label="Delete post"
                        icon={<Icon as={FiTrash2} />}
                        onClick={() => del.mutate({ id: post.id })}
                        isLoading={
                          (del as any).isPending ?? (del as any).isLoading
                        }
                        colorScheme="red"
                        variant="ghost"
                        size="lg"
                      />
                    </HStack>
                  )}
                </HStack>
              </Flex>
            </Box>

            {/* Article Content */}

            {/* Article Content */}
            <Box
              p={8}
              borderRadius="2xl"
              bg="whiteAlpha.50"
              backdropFilter="blur(10px)"
              borderWidth="1px"
              borderColor="whiteAlpha.100"
              className="markdown-content"
              sx={{
                "& h1": {
                  fontSize: "3xl",
                  fontWeight: "800",
                  mb: 4,
                  mt: 8,
                  color: "white",
                  borderBottomWidth: "2px",
                  borderColor: "green.700",
                  pb: 2,
                },
                "& h2": {
                  fontSize: "2xl",
                  fontWeight: "700",
                  mb: 3,
                  mt: 6,
                  color: "white",
                },
                "& h3": {
                  fontSize: "xl",
                  fontWeight: "600",
                  mb: 2,
                  mt: 4,
                  color: "gray.200",
                },
                "& p": {
                  fontSize: "lg",
                  lineHeight: "tall",
                  mb: 4,
                  color: "gray.300",
                },
                "& a": {
                  color: "green.400",
                  textDecoration: "underline",
                  _hover: { color: "green.300" },
                },
                "& ul, & ol": {
                  pl: 6,
                  mb: 4,
                  color: "gray.300",
                },
                "& li": {
                  mb: 2,
                  fontSize: "lg",
                },
                "& blockquote": {
                  borderLeftWidth: "4px",
                  borderColor: "green.600",
                  pl: 4,
                  py: 2,
                  fontStyle: "italic",
                  color: "gray.400",
                  bg: "whiteAlpha.50",
                  borderRadius: "md",
                  mb: 4,
                },
                "& code": {
                  bg: "whiteAlpha.200",
                  px: 2,
                  py: 1,
                  borderRadius: "md",
                  fontSize: "sm",
                  fontFamily: "'JetBrains Mono', monospace",
                  color: "green.300",
                },
                "& pre": {
                  bg: "gray.800",
                  p: 4,
                  borderRadius: "lg",
                  overflow: "auto",
                  mb: 4,
                  borderWidth: "1px",
                  borderColor: "whiteAlpha.200",
                },
                "& pre code": {
                  bg: "transparent",
                  p: 0,
                  color: "gray.100",
                },
                "& img": {
                  borderRadius: "lg",
                  my: 6,
                },
                "& hr": {
                  borderColor: "whiteAlpha.200",
                  my: 8,
                },
                "& table": {
                  width: "100%",
                  mb: 4,
                  borderCollapse: "collapse",
                },
                "& th, & td": {
                  borderWidth: "1px",
                  borderColor: "whiteAlpha.200",
                  p: 3,
                  textAlign: "left",
                },
                "& th": {
                  bg: "whiteAlpha.100",
                  fontWeight: "600",
                  color: "white",
                },
                "& td": {
                  color: "gray.300",
                },
              }}
            >
              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                {post.content ?? ""}
              </ReactMarkdown>
            </Box>
          </VStack>
        </Container>

        {/* Comments Section (moved below article) */}
        <Container maxW="4xl" py={6}>
          <Box>
            <Heading size="lg" color="white" mb={4} fontWeight="700">
              Comments
            </Heading>

            {/* New Comment Form */}
            <Box
              mb={6}
              p={4}
              borderRadius="lg"
              bg="whiteAlpha.50"
              borderWidth="1px"
              borderColor="whiteAlpha.100"
            >
              {session ? (
                <Flex gap={4} align="start">
                  <Avatar
                    size="sm"
                    src={session.user?.image ?? undefined}
                    name={session.user?.name ?? undefined}
                    bg="green.600"
                  />
                  <Box flex={1}>
                    <Textarea
                      value={newComment}
                      onChange={(e) => setNewComment(e.target.value)}
                      placeholder="Write a thoughtful comment..."
                      rows={3}
                      bg="transparent"
                      borderColor="whiteAlpha.200"
                      color="gray.100"
                    />
                    <Flex justify="flex-end" mt={2}>
                      <Button
                        bgGradient="linear(to-r, green.700, green.900)"
                        color="white"
                        size="sm"
                        leftIcon={<Icon as={FiSend} />}
                        onClick={() =>
                          createComment.mutate({
                            postId: post.id,
                            content: newComment,
                          })
                        }
                        isLoading={(createComment as any).isLoading}
                      >
                        Post Comment
                      </Button>
                    </Flex>
                  </Box>
                </Flex>
              ) : (
                <Button
                  onClick={() => signIn()}
                  colorScheme="green"
                  variant="outline"
                >
                  Sign in to comment
                </Button>
              )}
            </Box>

            {/* Comments List */}
            <VStack spacing={4} align="stretch">
              {comments?.map((c: CommentType) => (
                <CommentItem key={c.id} comment={c} />
              ))}
              {comments?.length === 0 && (
                <Text color="gray.500">
                  No comments yet. Be the first to comment!
                </Text>
              )}
            </VStack>
          </Box>
        </Container>

        {/* Edit Modal */}
        <Modal isOpen={isOpen} onClose={onClose} size="2xl">
          <ModalOverlay backdropFilter="blur(10px)" />
          <ModalContent
            bg="gray.800"
            borderColor="whiteAlpha.200"
            borderWidth="1px"
          >
            <ModalHeader
              borderBottomWidth="1px"
              borderColor="whiteAlpha.200"
              fontSize="2xl"
              fontWeight="700"
            >
              Edit Post
            </ModalHeader>
            <ModalCloseButton />
            <ModalBody py={6}>
              <VStack spacing={4}>
                <FormControl>
                  <FormLabel color="gray.300" fontWeight="600">
                    Title
                  </FormLabel>
                  <Input
                    value={title}
                    onChange={(e) => setTitle(e.target.value)}
                    size="lg"
                    bg="whiteAlpha.50"
                    borderColor="whiteAlpha.200"
                    _hover={{ borderColor: "green.600" }}
                    _focus={{
                      borderColor: "green.600",
                      boxShadow: "0 0 0 1px var(--chakra-colors-green-600)",
                    }}
                  />
                </FormControl>
                <FormControl>
                  <FormLabel color="gray.300" fontWeight="600">
                    Content (Markdown)
                  </FormLabel>
                  <Textarea
                    value={content}
                    onChange={(e) => setContent(e.target.value)}
                    rows={12}
                    fontFamily="'JetBrains Mono', monospace"
                    fontSize="sm"
                    bg="whiteAlpha.50"
                    borderColor="whiteAlpha.200"
                    _hover={{ borderColor: "green.600" }}
                    _focus={{
                      borderColor: "green.600",
                      boxShadow: "0 0 0 1px var(--chakra-colors-green-600)",
                    }}
                  />
                </FormControl>
              </VStack>
            </ModalBody>

            <ModalFooter borderTopWidth="1px" borderColor="whiteAlpha.200">
              <Button variant="ghost" mr={3} onClick={onClose}>
                Cancel
              </Button>
              <Button
                bgGradient="linear(to-r, green.700, green.900)"
                color="white"
                _hover={{
                  bgGradient: "linear(to-r, green.800, green.900)",
                }}
                onClick={() => update.mutate({ id: post.id, title, content })}
                isLoading={
                  (update as any).isPending ?? (update as any).isLoading
                }
                loadingText="Saving..."
                size="lg"
              >
                Save Changes
              </Button>
            </ModalFooter>
          </ModalContent>
        </Modal>
      </Box>
    </Layout>
  );
}
