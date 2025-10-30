import { useState } from "react";
import {
  Box,
  Container,
  Heading,
  VStack,
  Text,
  Button,
  Modal,
  ModalOverlay,
  ModalContent,
  ModalHeader,
  ModalBody,
  ModalFooter,
  ModalCloseButton,
  useDisclosure,
  FormControl,
  FormLabel,
  Input,
  Textarea,
  Flex,
  Avatar,
  HStack,
  Divider,
  Popover,
  PopoverTrigger,
  PopoverContent,
  PopoverBody,
  LinkBox,
  LinkOverlay,
  Portal,
} from "@chakra-ui/react";
import { Layout } from "~/components/layout";
import Link from "next/link";
import { useRouter } from "next/router";
import { api } from "~/utils/api";
import { useSession, signIn } from "next-auth/react";

function timeAgo(d: string | Date) {
  const s = Math.max(
    1,
    Math.floor((Date.now() - new Date(d).getTime()) / 1000)
  );
  const steps = [
    { n: 31536000, u: "y" },
    { n: 2592000, u: "mo" },
    { n: 86400, u: "d" },
    { n: 3600, u: "h" },
    { n: 60, u: "m" },
  ];
  for (const { n, u } of steps) if (s >= n) return `${Math.floor(s / n)}${u}`;
  return `${s}s`;
}

export default function TestBlogIndex() {
  const router = useRouter();
  const { data: session } = useSession();
  const { data, isLoading } = api.blogpost.getAll.useQuery({});
  const posts = data?.posts;
  const { isOpen, onOpen, onClose } = useDisclosure();
  const [title, setTitle] = useState("");
  const [content, setContent] = useState("");
  const [showMyPosts, setShowMyPosts] = useState(false);

  const utils = api.useContext();
  const create = api.blogpost.create.useMutation({
    onSuccess: async () => {
      await utils.blogpost.getAll.invalidate();
      onClose();
      setTitle("");
      setContent("");
    },
  });

  const filteredPosts =
    showMyPosts && session
      ? posts?.filter((p: any) => p.author.id === session.user.id)
      : posts;

  const getPreview = (markdown: string, maxLength = 220) => {
    const text = markdown
      .replace(/#{1,6}\s/g, "")
      .replace(/[*_~`]/g, "")
      .replace(/\[([^\]]+)\]\([^)]+\)/g, "$1")
      .trim();
    return text.slice(0, maxLength) + (text.length > maxLength ? "…" : "");
  };

  function PostRow({ post }: { post: any }) {
    const upvote = api.postUpvote.count.useQuery(
      { postId: post.id },
      { enabled: !!post.id }
    );

    return (
      <Popover
        trigger="hover"
        placement="auto-start"
        openDelay={200}
        isLazy
        modifiers={[
          { name: "offset", options: { offset: [0, 8] } },
          {
            name: "preventOverflow",
            options: { padding: 8, boundary: "viewport" },
          },
        ]}
      >
        <PopoverTrigger>
          <LinkBox
            as="article"
            py={2.5}
            px={2}
            borderBottom="1px solid"
            borderColor="gray.800"
            _hover={{ bg: "gray.850" }}
            transition="background 0.15s ease"
          >
            <Flex gap={3} align="center" minW={0}>
              {/* votes */}
              <Box
                w="44px"
                textAlign="center"
                flexShrink={0}
                rounded="md"
                border="1px solid"
                borderColor="gray.800"
                bg="gray.900"
                py={1}
              >
                <Text fontSize="sm" fontWeight="600" color="gray.300">
                  {upvote.data?.count ?? 0}
                </Text>
              </Box>

              {/* title + author */}
              <Flex flex="1" minW={0} align="center" gap={2}>
                <LinkOverlay
                  as={Link}
                  href={`/blog/${post.slug ?? post.id}`}
                  _hover={{ textDecoration: "none", color: "blue.300" }}
                >
                  <Text
                    color="white"
                    fontWeight="600"
                    noOfLines={1}
                    letterSpacing="-0.01em"
                  >
                    {post.title}
                  </Text>
                </LinkOverlay>
                <HStack spacing={2} color="gray.500" flexShrink={0}>
                  <Avatar
                    size="xs"
                    src={post.author?.image ?? undefined}
                    name={post.author?.name ?? undefined}
                  />
                  <Text fontSize="xs" noOfLines={1}>
                    {post.author?.name}
                  </Text>
                </HStack>
              </Flex>

              {/* time on the right */}
              <Text
                fontSize="xs"
                color="gray.500"
                flexShrink={0}
                w="48px"
                textAlign="right"
              >
                {timeAgo(post.createdAt)}
              </Text>
            </Flex>
          </LinkBox>
        </PopoverTrigger>

        {/* compact preview that never escapes viewport */}
        <Portal>
          <PopoverContent
            bg="gray.900"
            borderColor="gray.700"
            maxW="420px"
            boxShadow="xl"
            _focus={{ boxShadow: "xl" }}
            zIndex={1600}
          >
            <PopoverBody p={4}>
              <Text color="white" fontWeight="600" mb={2}>
                {post.title}
              </Text>
              <Text color="gray.300" fontSize="sm" lineHeight="1.6">
                {getPreview(post.content || "", 380)}
              </Text>
            </PopoverBody>
          </PopoverContent>
        </Portal>
      </Popover>
    );
  }

  return (
    <Layout title="Blog">
      <Box bg="gray.900" minH="100vh">
        {/* narrower, readable column similar to LW */}
        <Container maxW={{ base: "100%", md: "760px", lg: "860px" }} py={8}>
          <Flex
            mb={6}
            align="center"
            justify="space-between"
            gap={4}
            wrap="wrap"
          >
            <Box>
              <Heading
                as="h1"
                fontSize="2xl"
                fontWeight="700"
                color="white"
                mb={1}
              >
                Blog
              </Heading>
              <Text color="gray.400" fontSize="sm">
                Thoughts, ideas, and stories
              </Text>
            </Box>
            <HStack spacing={2}>
              {session ? (
                <>
                  <Button
                    size="sm"
                    variant={showMyPosts ? "solid" : "outline"}
                    onClick={() => setShowMyPosts(!showMyPosts)}
                    colorScheme={showMyPosts ? "blue" : "blue"}
                  >
                    {showMyPosts ? "All Posts" : "Your Posts"}
                  </Button>
                  <Button
                    size="sm"
                    colorScheme="blue"
                    onClick={() => router.push("/blog/create")}
                  >
                    New Post
                  </Button>
                </>
              ) : (
                <Button size="sm" colorScheme="blue" onClick={() => signIn()}>
                  Sign in to post
                </Button>
              )}
            </HStack>
          </Flex>

          <Divider mb={4} borderColor="gray.800" />

          {isLoading ? (
            <Flex justify="center" py={16}>
              <Text color="gray.500">Loading posts…</Text>
            </Flex>
          ) : (
            <VStack spacing={0} align="stretch">
              {filteredPosts?.map((post: any) => (
                <PostRow key={post.id} post={post} />
              ))}
            </VStack>
          )}
        </Container>

        {/* create modal (unchanged) */}
        <Modal isOpen={isOpen} onClose={onClose} size="2xl">
          <ModalOverlay />
          <ModalContent bg="gray.800" borderColor="gray.700" borderWidth="1px">
            <ModalHeader fontSize="xl" fontWeight="600" color="white">
              Create New Post
            </ModalHeader>
            <ModalCloseButton color="gray.400" />
            <ModalBody py={6}>
              <VStack spacing={4}>
                <FormControl>
                  <FormLabel fontWeight="600" fontSize="sm" color="gray.300">
                    Title
                  </FormLabel>
                  <Input
                    value={title}
                    onChange={(e) => setTitle(e.target.value)}
                    placeholder="Enter title…"
                    bg="gray.900"
                    borderColor="gray.600"
                    color="white"
                    _placeholder={{ color: "gray.500" }}
                  />
                </FormControl>
                <FormControl>
                  <FormLabel fontWeight="600" fontSize="sm" color="gray.300">
                    Content (Markdown Supported)
                  </FormLabel>
                  <Textarea
                    value={content}
                    onChange={(e) => setContent(e.target.value)}
                    placeholder="Write your post in markdown…"
                    rows={12}
                    fontFamily="monospace"
                    fontSize="sm"
                    bg="gray.900"
                    borderColor="gray.600"
                    color="white"
                    _placeholder={{ color: "gray.500" }}
                  />
                  <Text color="gray.500" fontSize="xs" mt={2}>
                    Markdown formatting is supported
                  </Text>
                </FormControl>
              </VStack>
            </ModalBody>

            <ModalFooter>
              <Button variant="ghost" mr={3} onClick={onClose} color="gray.400">
                Cancel
              </Button>
              <Button
                colorScheme="blue"
                onClick={() => create.mutate({ title, content })}
                isLoading={
                  (create as any).isPending ?? (create as any).isLoading
                }
              >
                Create Post
              </Button>
            </ModalFooter>
          </ModalContent>
        </Modal>
      </Box>
    </Layout>
  );
}
