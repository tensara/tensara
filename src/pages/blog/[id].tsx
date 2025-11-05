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
  Divider,
  Tooltip,
} from "@chakra-ui/react";
import { api } from "~/utils/api";
import { useSession, signIn } from "next-auth/react";
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
import { format } from "date-fns";
import { MarkdownRenderer } from "~/components/blog";

// Typed helper aliases for tRPC outputs
type CommentType = RouterOutputs["comments"]["getByPost"][number];

export default function TestBlogPost() {
  const router = useRouter();
  const { id } = router.query;
  const { data: session } = useSession();

  const { data: post, isLoading } = api.blogpost.getById.useQuery(
    typeof id === "string" ? { slug: id } : { slug: "" },
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

  // small helper to safely check mutation running state without `any` casts
  function mutationIsRunning(m: unknown): boolean {
    if (!m || typeof m !== "object") return false;
    const mm = m as { isPending?: boolean; isLoading?: boolean };
    return !!mm.isPending || !!mm.isLoading;
  }

  const del = api.blogpost.delete.useMutation({
    onSuccess: () => {
      void utils.blogpost.getAll.invalidate();
      void router.push("/blog");
    },
  });

  // Post-related hooks
  const isAuthor = session?.user?.id === post?.author?.id;

  // Post upvote
  const { data: postUpvoteCount } = api.postUpvote.count.useQuery(
    { postId: post?.id ?? "" },
    { enabled: !!post }
  );
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

  const [newComment, setNewComment] = useState("");

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
      <Box key={comment.id} py={4}>
        <Flex align="start" justify="space-between" gap={4}>
          <HStack align="start" spacing={3}>
            <Avatar
              size="sm"
              src={comment.author?.image ?? undefined}
              name={comment.author?.name ?? undefined}
              bg="green.600"
            />
            <Box>
              <HStack spacing={2} wrap="wrap">
                <Text color="white" fontWeight="600" fontSize="sm">
                  {comment.author?.name}
                </Text>
                <Text color="gray.500" fontSize="xs">
                  {format(new Date(comment.createdAt), "MMM d, yyyy • h:mm a")}
                </Text>
              </HStack>
              <Box mt={2} color="gray.200">
                <Text whiteSpace="pre-wrap" fontSize="sm" lineHeight="taller">
                  {comment.content}
                </Text>
              </Box>

              <HStack mt={3} spacing={2}>
                <Button
                  size="xs"
                  variant={cuHasUpvoted?.hasUpvoted ? "solid" : "ghost"}
                  leftIcon={<Icon as={FiThumbsUp} />}
                  onClick={() => cuToggle.mutate({ commentId: comment.id })}
                  colorScheme="green"
                >
                  {cuCount.data?.count ?? 0}
                </Button>
                <Button
                  size="xs"
                  variant="ghost"
                  leftIcon={<Icon as={FiMessageSquare} />}
                  onClick={() => setReplyOpen((s) => !s)}
                >
                  Reply
                </Button>
                {isCommentAuthor && (
                  <IconButton
                    aria-label="Delete comment"
                    icon={<Icon as={FiTrash2} />}
                    size="xs"
                    variant="ghost"
                    colorScheme="red"
                    onClick={() => deleteComment.mutate({ id: comment.id })}
                  />
                )}
              </HStack>

              {replyOpen && (
                <Box mt={3}>
                  <Textarea
                    value={replyText}
                    onChange={(e) => setReplyText(e.target.value)}
                    rows={2}
                    placeholder="Write a reply…"
                    bg="transparent"
                    borderColor="whiteAlpha.200"
                    color="gray.100"
                  />
                  <HStack justify="flex-end" mt={2} spacing={2}>
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
                  </HStack>
                </Box>
              )}
            </Box>
          </HStack>
        </Flex>

        {/* children */}
        {comment.children?.length ? (
          <VStack align="stretch" mt={3} ml={8} spacing={0}>
            {comment.children.map((child: CommentType) => (
              <Box
                key={child.id}
                pl={3}
                borderLeft="1px solid"
                borderColor="whiteAlpha.200"
              >
                <CommentItem comment={child} />
              </Box>
            ))}
          </VStack>
        ) : null}

        <Divider mt={4} borderColor="whiteAlpha.200" />
      </Box>
    );
  }

  if (isLoading)
    return (
      <Layout title="Loading...">
        <Box bg="gray.900" minH="100vh" py={16}>
          <Container maxW="5xl">
            <Text color="gray.500" textAlign="center" fontSize="lg">
              Loading post…
            </Text>
          </Container>
        </Box>
      </Layout>
    );

  if (!post)
    return (
      <Layout title="Not found">
        <Box bg="gray.900" minH="100vh" py={16}>
          <Container maxW="5xl">
            <VStack spacing={4}>
              <Text color="gray.400" fontSize="xl">
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
        {/* Wider container */}
        <Container maxW="5xl" py={10}>
          <Button
            as={Link}
            href="/blog"
            leftIcon={<Icon as={FiArrowLeft} />}
            variant="ghost"
            colorScheme="green"
            mb={6}
            _hover={{ bg: "whiteAlpha.100" }}
          >
            Back to posts
          </Button>

          <VStack align="stretch" spacing={0}>
            {/* Header */}
            <Box>
              <Heading
                fontSize={{ base: "3xl", md: "4xl" }}
                fontWeight="800"
                letterSpacing="-0.01em"
                color="white"
              >
                {post.title}
              </Heading>

              <HStack spacing={4} mt={3} color="gray.400" fontSize="sm">
                <HStack spacing={2}>
                  <Avatar
                    size="sm"
                    src={post.author?.image ?? undefined}
                    name={post.author?.name ?? undefined}
                    bg="green.600"
                  />
                  <Text color="gray.200" fontWeight="600">
                    {post.author?.name}
                  </Text>
                </HStack>
                <HStack spacing={1}>
                  <Icon as={FiCalendar} />
                  <Text>
                    {new Date(post.createdAt).toLocaleDateString("en-US", {
                      month: "long",
                      day: "numeric",
                      year: "numeric",
                    })}
                  </Text>
                </HStack>

                <HStack ml="auto" spacing={2}>
                  <Tooltip
                    label={postHasUpvoted?.hasUpvoted ? "Unlike" : "Like"}
                  >
                    <Button
                      size="sm"
                      variant={postHasUpvoted?.hasUpvoted ? "solid" : "ghost"}
                      leftIcon={<Icon as={FiThumbsUp} />}
                      onClick={() =>
                        postUpvoteToggle.mutate({ postId: post.id })
                      }
                      colorScheme="green"
                    >
                      {postUpvoteCount?.count ?? 0}
                    </Button>
                  </Tooltip>

                  {isAuthor && (
                    <HStack spacing={1}>
                      <IconButton
                        aria-label="Edit post"
                        icon={<Icon as={FiEdit3} />}
                        onClick={onOpen}
                        colorScheme="green"
                        variant="ghost"
                        size="sm"
                      />
                      <IconButton
                        aria-label="Delete post"
                        icon={<Icon as={FiTrash2} />}
                        onClick={() => del.mutate({ id: post.id })}
                        isLoading={mutationIsRunning(del)}
                        colorScheme="red"
                        variant="ghost"
                        size="sm"
                      />
                    </HStack>
                  )}
                </HStack>
              </HStack>
            </Box>

            <Divider my={6} borderColor="whiteAlpha.300" />

            <Box
              maxW="900px"
              mx="auto"
              className="markdown-content"
              sx={{
                "& h1": {
                  fontSize: "2xl",
                  fontWeight: "800",
                  mt: 10,
                  mb: 4,
                  color: "white",
                },
                "& h2": {
                  fontSize: "xl",
                  fontWeight: "700",
                  mt: 8,
                  mb: 3,
                  color: "white",
                },
                "& h3": {
                  fontSize: "lg",
                  fontWeight: "700",
                  mt: 6,
                  mb: 2,
                  color: "gray.100",
                },
                "& p": {
                  fontSize: "lg",
                  lineHeight: 1.85,
                  mb: 4,
                  color: "gray.200",
                },
                "& a": {
                  color: "green.300",
                  textDecoration: "underline",
                  _hover: { color: "green.200" },
                },
                "& ul, & ol": {
                  pl: 6,
                  mb: 4,
                  color: "gray.200",
                },
                "& li": {
                  mb: 2,
                  fontSize: "lg",
                },
                "& blockquote": {
                  borderLeftWidth: "3px",
                  borderColor: "whiteAlpha.400",
                  pl: 4,
                  py: 1,
                  fontStyle: "italic",
                  color: "gray.300",
                  my: 4,
                },
                "& code": {
                  bg: "whiteAlpha.200",
                  px: 1.5,
                  py: 0.5,
                  borderRadius: "md",
                  fontSize: "0.9em",
                  fontFamily: "'JetBrains Mono', ui-monospace, SFMono-Regular",
                  color: "green.200",
                },
                "& pre": {
                  bg: "gray.800",
                  p: 4,
                  borderRadius: "lg",
                  overflow: "auto",
                  mb: 5,
                  borderWidth: "1px",
                  borderColor: "whiteAlpha.200",
                },
                "& pre code": {
                  bg: "transparent",
                  p: 0,
                  color: "gray.100",
                },
                "& img": {
                  borderRadius: "md",
                  my: 6,
                },
                "& hr": {
                  borderColor: "whiteAlpha.300",
                  my: 10,
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
                  color: "gray.200",
                },
              }}
            >
              <MarkdownRenderer content={post.content ?? ""} />
            </Box>
          </VStack>
        </Container>

        {/* Comments */}
        <Container maxW="5xl" py={8}>
          <Box maxW="900px" mx="auto">
            <Heading
              size="md"
              color="white"
              mb={4}
              fontWeight="800"
              letterSpacing="-0.01em"
            >
              Comments
            </Heading>

            {/* New comment */}
            <Box mb={6}>
              {session ? (
                <Flex gap={3} align="start">
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
                      placeholder="Write a thoughtful comment…"
                      rows={3}
                      bg="transparent"
                      borderColor="whiteAlpha.300"
                      color="gray.100"
                    />
                    <HStack justify="flex-end" mt={2}>
                      <Button
                        bgColor="green.600"
                        color="white"
                        size="sm"
                        leftIcon={<Icon as={FiSend} />}
                        onClick={() =>
                          createComment.mutate({
                            postId: post.id,
                            content: newComment,
                          })
                        }
                        isLoading={mutationIsRunning(createComment)}
                      >
                        Post
                      </Button>
                    </HStack>
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

            {/* List */}
            <VStack spacing={0} align="stretch">
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
      </Box>

      {/* (Optional) Edit modal kept intact; style is already minimal */}
      <Modal isOpen={isOpen} onClose={onClose} size="lg">
        <ModalOverlay />
        <ModalContent
          bg="gray.900"
          border="1px solid"
          borderColor="whiteAlpha.200"
        >
          <ModalHeader color="white">Edit post</ModalHeader>
          <ModalCloseButton />
          <ModalBody>
            <VStack spacing={4} align="stretch">
              <FormControl>
                <FormLabel color="gray.200">Title</FormLabel>
                <Input
                  value={title}
                  onChange={(e) => setTitle(e.target.value)}
                  bg="transparent"
                  borderColor="whiteAlpha.300"
                  color="gray.100"
                />
              </FormControl>
              <FormControl>
                <FormLabel color="gray.200">Content</FormLabel>
                <Textarea
                  value={content}
                  onChange={(e) => setContent(e.target.value)}
                  rows={10}
                  bg="transparent"
                  borderColor="whiteAlpha.300"
                  color="gray.100"
                />
              </FormControl>
            </VStack>
          </ModalBody>
          <ModalFooter>
            <Button variant="ghost" mr={3} onClick={onClose}>
              Cancel
            </Button>
            <Button
              colorScheme="green"
              onClick={() => update.mutate({ id: post.id, title, content })}
              isLoading={mutationIsRunning(update)}
            >
              Save
            </Button>
          </ModalFooter>
        </ModalContent>
      </Modal>
    </Layout>
  );
}
