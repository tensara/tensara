import { useRouter } from "next/router";
import type { RouterOutputs } from "~/utils/api";
import { useState } from "react";
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
  Textarea,
  Icon,
  IconButton,
  Divider,
  Collapse,
  useToast,
} from "@chakra-ui/react";
import { api } from "~/utils/api";
import { useSession, signIn } from "next-auth/react";
import {
  FiEdit3,
  FiTrash2,
  FiArrowLeft,
  FiThumbsUp,
  FiChevronDown,
  FiChevronUp,
  FiShare2,
} from "react-icons/fi";
import Link from "next/link";
import { format } from "date-fns";
import { MarkdownRenderer } from "~/components/blog";
import { markdownContentStyles } from "~/constants/blog";

// Typed helper aliases for tRPC outputs
type CommentType = RouterOutputs["comments"]["getByPost"][number];

export default function BlogPost() {
  const router = useRouter();
  const { id } = router.query;
  const { data: session } = useSession();
  const toast = useToast();

  const { data: post, isLoading } = api.blogpost.getById.useQuery(
    typeof id === "string" ? { slug: id } : { slug: "" },
    { enabled: typeof id === "string" }
  );

  const utils = api.useContext();

  // delete post
  const del = api.blogpost.delete.useMutation({
    onSuccess: async () => {
      await utils.blogpost.getAll.invalidate();
      void router.push("/blog");
    },
  });

  // Post-related helpers
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
      if (post?.id) {
        await utils.postUpvote.count.invalidate({ postId: post.id });
        if (session)
          await utils.postUpvote.hasUpvoted.invalidate({ postId: post.id });
        await utils.blogpost.getById.invalidate();
      }
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

  // small helper to safely check mutation running state
  function mutationIsRunning(m: unknown): boolean {
    if (!m || typeof m !== "object") return false;
    const mm = m as { isPending?: boolean; isLoading?: boolean };
    return !!mm.isPending || !!mm.isLoading;
  }

  const [newComment, setNewComment] = useState("");
  const [commentsOpen, setCommentsOpen] = useState(false);

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
      <Box key={comment.id} py={3}>
        <VStack align="stretch" spacing={2}>
          <HStack spacing={2} align="baseline">
            <Text color="gray.400" fontSize="xs" fontWeight="500">
              {comment.author?.name}
            </Text>
            <Text as="span" color="gray.600" fontSize="xs">
              ·
            </Text>
            <Text color="gray.600" fontSize="xs">
              {format(new Date(comment.createdAt), "MMM d")}
            </Text>
          </HStack>

          <Text
            whiteSpace="pre-wrap"
            fontSize="sm"
            color="gray.300"
            lineHeight="1.6"
          >
            {comment.content}
          </Text>

          <HStack spacing={3} mt={1}>
            <Button
              size="xs"
              variant="ghost"
              leftIcon={<Icon as={FiThumbsUp} />}
              onClick={() => cuToggle.mutate({ commentId: comment.id })}
              color={cuHasUpvoted?.hasUpvoted ? "green.400" : "gray.500"}
              _hover={{
                bg: "transparent",
                color: cuHasUpvoted?.hasUpvoted ? "green.300" : "gray.400",
              }}
              fontWeight="normal"
              h="auto"
              py={1}
              px={2}
            >
              {cuCount.data?.count ?? 0}
            </Button>
            <Button
              size="xs"
              variant="ghost"
              onClick={() => setReplyOpen((s) => !s)}
              color="gray.500"
              _hover={{ color: "gray.400", bg: "transparent" }}
              fontWeight="normal"
              h="auto"
              py={1}
              px={2}
            >
              Reply
            </Button>
            {isCommentAuthor && (
              <IconButton
                aria-label="Delete comment"
                icon={<Icon as={FiTrash2} />}
                size="xs"
                variant="ghost"
                color="gray.500"
                _hover={{ color: "red.400", bg: "transparent" }}
                onClick={() => deleteComment.mutate({ id: comment.id })}
                isLoading={mutationIsRunning(deleteComment)}
                h="auto"
                minW="auto"
                w="auto"
              />
            )}
          </HStack>

          {replyOpen && (
            <Box
              mt={2}
              pl={3}
              borderLeftWidth="1px"
              borderColor="whiteAlpha.100"
            >
              <Textarea
                value={replyText}
                onChange={(e) => setReplyText(e.target.value)}
                rows={2}
                placeholder="Write a reply…"
                bg="transparent"
                borderColor="whiteAlpha.200"
                color="gray.100"
                _focus={{ borderColor: "whiteAlpha.300" }}
                fontSize="sm"
              />
              <HStack justify="flex-end" mt={2} spacing={2}>
                <Button
                  size="xs"
                  variant="ghost"
                  onClick={() => {
                    setReplyOpen(false);
                    setReplyText("");
                  }}
                  color="gray.500"
                  _hover={{ color: "gray.300", bg: "transparent" }}
                >
                  Cancel
                </Button>
                <Button
                  size="xs"
                  variant="ghost"
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

          {/* children */}
          {comment.children?.length ? (
            <VStack align="stretch" mt={2} ml={4} spacing={0}>
              {comment.children.map((child: CommentType) => (
                <Box
                  key={child.id}
                  pl={3}
                  borderLeftWidth="1px"
                  borderColor="whiteAlpha.100"
                >
                  <CommentItem comment={child} />
                </Box>
              ))}
            </VStack>
          ) : null}
        </VStack>
      </Box>
    );
  }

  if (isLoading)
    return (
      <Layout title="Loading...">
        <Box minH="100vh" py={16}>
          <Container maxW="7xl" mx="auto">
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
        <Box minH="100vh" py={16}>
          <Container maxW="7xl" mx="auto">
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
                Back
              </Button>
            </VStack>
          </Container>
        </Box>
      </Layout>
    );

  return (
    <Layout title={post.title}>
      <Box minH="100vh">
        {/* Wider container */}
        <Container maxW="7xl" mx="auto" py={10}>
          <Button
            as={Link}
            href="/blog"
            leftIcon={<Icon as={FiArrowLeft} />}
            variant="ghost"
            colorScheme="green"
            mb={6}
            size="sm"
            _hover={{ bg: "whiteAlpha.100" }}
          >
            Back
          </Button>

          <VStack align="stretch" spacing={0}>
            {/* Header */}
            <Box>
              <Heading
                fontSize={{ base: "3xl", md: "4xl" }}
                fontWeight="800"
                letterSpacing="-0.01em"
                color="white"
                fontFamily="Space Grotesk"
                textAlign="center"
              >
                {post.title}
              </Heading>

              <Flex
                justify="space-between"
                align="center"
                mt={3}
                position="relative"
              >
                <Box flex={1} />
                <HStack
                  spacing={2}
                  color="gray.500"
                  fontSize="sm"
                  position="absolute"
                  left="50%"
                  transform="translateX(-50%)"
                >
                  <HStack spacing={2}>
                    <Avatar
                      size="xs"
                      src={post.author?.image ?? undefined}
                      name={post.author?.name ?? undefined}
                      bg="green.600"
                    />
                    <Text>{post.author?.name}</Text>
                  </HStack>
                  <Text>·</Text>
                  <Text>
                    {new Date(post.createdAt).toLocaleDateString("en-US", {
                      month: "short",
                      day: "numeric",
                      year: "numeric",
                    })}
                  </Text>

                  {isAuthor && (
                    <>
                      <Text>·</Text>
                      <HStack spacing={1}>
                        <IconButton
                          aria-label="Edit post"
                          icon={<Icon as={FiEdit3} />}
                          onClick={() => router.push(`/blog/edit/${post.id}`)}
                          variant="ghost"
                          color="gray.500"
                          size="xs"
                          h="auto"
                          bg="none"
                          _hover={{ bg: "none" }}
                        />
                        <IconButton
                          aria-label="Delete post"
                          icon={<Icon as={FiTrash2} />}
                          onClick={() => del.mutate({ id: post.id })}
                          isLoading={mutationIsRunning(del)}
                          variant="ghost"
                          color="gray.500"
                          size="xs"
                          h="auto"
                          bg="none"
                          _hover={{ bg: "none" }}
                        />
                      </HStack>
                    </>
                  )}
                </HStack>
                <Flex flex={1} justify="flex-end" gap={4}>
                  <HStack
                    as="button"
                    spacing={1}
                    color="gray.500"
                    _hover={{ color: "white" }}
                    transition="color 0.5s ease"
                    onClick={async () => {
                      try {
                        await navigator.clipboard.writeText(
                          window.location.href
                        );
                        toast({
                          title: "Link copied",
                          status: "success",
                          duration: 2000,
                          isClosable: true,
                        });
                      } catch (err) {
                        console.error(err);
                      }
                    }}
                    cursor="pointer"
                    fontSize="sm"
                  >
                    <Icon as={FiShare2} />
                  </HStack>

                  <HStack
                    as="button"
                    spacing={1}
                    color={postHasUpvoted?.hasUpvoted ? "white" : "gray.500"}
                    _hover={{ color: "white" }}
                    transition="color 0.5s ease"
                    onClick={() => postUpvoteToggle.mutate({ postId: post.id })}
                    cursor="pointer"
                    fontSize="sm"
                  >
                    <Icon as={FiThumbsUp} />
                    <Text>{postUpvoteCount?.count ?? 0}</Text>
                  </HStack>
                </Flex>
              </Flex>
            </Box>

            <Divider my={6} borderColor="whiteAlpha.300" />

            <Box
              w="90%"
              mx="auto"
              className="markdown-content"
              sx={markdownContentStyles}
            >
              <MarkdownRenderer content={post.content ?? ""} />
            </Box>
            <Divider my={6} borderColor="whiteAlpha.300" />
          </VStack>
        </Container>

        {/* Collapsible Comments Section */}
        <Container maxW="7xl" mx="auto" py={8} mb={10}>
          <Box w="80%" mx="auto">
            <Flex align="center" justify="space-between" mb={4}>
              <Heading
                fontSize="2xl"
                fontWeight="700"
                color="white"
                fontFamily="Space Grotesk"
              >
                Comments
              </Heading>
              <Button
                variant="ghost"
                onClick={() => setCommentsOpen(!commentsOpen)}
                color="gray.400"
                _hover={{ color: "gray.300", bg: "transparent" }}
                rightIcon={
                  <Icon as={commentsOpen ? FiChevronUp : FiChevronDown} />
                }
                fontWeight="normal"
                fontSize="sm"
                px={0}
                h="auto"
                py={1}
              >
                {comments?.length ?? 0}{" "}
                {comments?.length === 1 ? "comment" : "comments"}
              </Button>
            </Flex>

            <Collapse in={commentsOpen} animateOpacity>
              <VStack align="stretch" spacing={6}>
                {/* New comment */}
                {session ? (
                  <Box>
                    <Textarea
                      value={newComment}
                      onChange={(e) => setNewComment(e.target.value)}
                      placeholder="Write a comment…"
                      rows={3}
                      bg="whiteAlpha.50"
                      border="1px solid"
                      borderColor="transparent"
                      color="white"
                      _hover={{ borderColor: "gray.600" }}
                      _focus={{ borderColor: "blue.500", boxShadow: "none" }}
                      fontSize="sm"
                    />
                    <HStack justify="flex-end" mt={3} spacing={2}>
                      <Button
                        size="xs"
                        variant="ghost"
                        onClick={() => setNewComment("")}
                        color="gray.500"
                        _hover={{ color: "gray.300", bg: "transparent" }}
                      >
                        Clear
                      </Button>
                      <Button
                        size="xs"
                        variant="ghost"
                        colorScheme="green"
                        onClick={() =>
                          createComment.mutate({
                            postId: post.id,
                            content: newComment,
                          })
                        }
                        isLoading={mutationIsRunning(createComment)}
                        isDisabled={!newComment.trim()}
                      >
                        Post
                      </Button>
                    </HStack>
                  </Box>
                ) : (
                  <Button
                    onClick={() => signIn()}
                    variant="ghost"
                    color="gray.500"
                    _hover={{ color: "gray.300", bg: "transparent" }}
                    size="sm"
                    fontWeight="normal"
                  >
                    Sign in to comment
                  </Button>
                )}

                {/* Comments list */}
                <VStack spacing={0} align="stretch" mt={2}>
                  {comments?.map((c: CommentType) => (
                    <CommentItem key={c.id} comment={c} />
                  ))}
                  {comments?.length === 0 && (
                    <Text color="gray.500" fontSize="sm" py={4}>
                      No comments yet.
                    </Text>
                  )}
                </VStack>
              </VStack>
            </Collapse>
          </Box>
        </Container>
      </Box>
    </Layout>
  );
}
