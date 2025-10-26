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
  Badge,
  Icon,
} from "@chakra-ui/react";
import { Layout } from "~/components/layout";
import Link from "next/link";
import { api } from "~/utils/api";
import { useSession, signIn } from "next-auth/react";
import { FiEdit3, FiCalendar, FiThumbsUp } from "react-icons/fi";

export default function TestBlogIndex() {
  const { data: session } = useSession();
  const { data: posts, isLoading } = api.blogpost.getAll.useQuery();
  const { isOpen, onOpen, onClose } = useDisclosure();
  const [title, setTitle] = useState("");
  const [content, setContent] = useState("");

  const utils = api.useContext();
  const create = api.blogpost.create.useMutation({
    onSuccess: async () => {
      await utils.blogpost.getAll.invalidate();
      onClose();
      setTitle("");
      setContent("");
    },
  });

  // Extract first paragraph from markdown for preview
  const getPreview = (markdown: string) => {
    const text = markdown
      .replace(/#{1,6}\s/g, "") // Remove headers
      .replace(/[*_~`]/g, "") // Remove formatting
      .replace(/\[([^\]]+)\]\([^)]+\)/g, "$1") // Remove links
      .trim();
    return text.slice(0, 200) + (text.length > 200 ? "..." : "");
  };

  function PostCard({ post }: { post: any }) {
    const upvote = api.postUpvote.count.useQuery(
      { postId: post.id },
      { enabled: !!post.id }
    );

    return (
      <Box
        as={Link}
        href={`/testblog/${post.id}`}
        position="relative"
        overflow="hidden"
        borderRadius="2xl"
        bg="whiteAlpha.50"
        backdropFilter="blur(10px)"
        borderWidth="1px"
        borderColor="whiteAlpha.100"
        transition="all 0.3s cubic-bezier(0.4, 0, 0.2, 1)"
        _hover={{
          borderColor: "green.500",
          transform: "translateY(-4px)",
          shadow: "2xl",
          bg: "whiteAlpha.100",
        }}
        textDecoration="none !important"
        _before={{
          content: '""',
          position: "absolute",
          top: 0,
          left: 0,
          right: 0,
          height: "3px",
          bgGradient: "linear(to-r, green.400, green.600, green.800)",
          opacity: 0,
          transition: "opacity 0.3s",
        }}
        _after={{
          content: '""',
          position: "absolute",
          inset: 0,
          bgGradient: "linear(to-br, green.600, green.800)",
          opacity: 0,
          transition: "opacity 0.3s",
          zIndex: -1,
        }}
        sx={{
          "&:hover::before": { opacity: 1 },
          "&:hover::after": { opacity: 0.05 },
        }}
      >
        <Box p={8}>
          <Flex align="start" justify="space-between" mb={4}>
            <Box flex={1}>
              <Heading
                size="xl"
                mb={3}
                color="white"
                fontWeight="700"
                letterSpacing="tight"
              >
                {post.title}
              </Heading>
              <Text
                color="gray.400"
                fontSize="md"
                lineHeight="tall"
                noOfLines={2}
              >
                {getPreview(post.content || "")}
              </Text>
            </Box>
          </Flex>

          <Flex align="center" justify="space-between" mt={6}>
            <HStack spacing={3}>
              <Avatar size="sm" name={post.author?.name ?? ""} bg="green.600" />
              <Box>
                <Text color="gray.300" fontSize="sm" fontWeight="600">
                  {post.author?.name}
                </Text>
                <HStack spacing={2} color="gray.500" fontSize="xs">
                  <Icon as={FiCalendar} />
                  <Text>
                    {new Date(post.createdAt).toLocaleDateString("en-US", {
                      month: "short",
                      day: "numeric",
                      year: "numeric",
                    })}
                  </Text>
                </HStack>
              </Box>
            </HStack>

            <HStack spacing={4} align="center">
              <HStack color="gray.300" spacing={2}>
                <Icon as={FiThumbsUp} />
                <Text>{upvote.data?.count ?? 0}</Text>
              </HStack>
              <Badge
                colorScheme="green"
                px={3}
                py={1}
                borderRadius="full"
                fontSize="xs"
                textTransform="uppercase"
                letterSpacing="wider"
              >
                Read more →
              </Badge>
            </HStack>
          </Flex>
        </Box>
      </Box>
    );
  }

  return (
    <Layout title="Blog">
      <Box bg="gray.900" minH="100vh">
        <Container maxW="5xl" py={12}>
          {/* Header */}
          <Flex mb={12} align="center" justify="space-between">
            <Box>
              <Heading
                as="h1"
                fontSize={{ base: "4xl", md: "5xl" }}
                fontWeight="800"
                bgGradient="linear(to-r, green.400, green.600, green.800)"
                bgClip="text"
                mb={2}
              >
                Blog
              </Heading>
              <Text color="gray.400" fontSize="lg">
                Thoughts, ideas, and stories
              </Text>
            </Box>
            <HStack>
              {session ? (
                <Button
                  colorScheme="green"
                  size="lg"
                  leftIcon={<Icon as={FiEdit3} />}
                  onClick={onOpen}
                  bgGradient="linear(to-r, green.600, green.800)"
                  _hover={{
                    bgGradient: "linear(to-r, green.700, green.900)",
                    transform: "translateY(-2px)",
                    shadow: "xl",
                  }}
                  transition="all 0.2s"
                >
                  New Post
                </Button>
              ) : (
                <Button
                  colorScheme="green"
                  size="lg"
                  onClick={() => signIn()}
                  variant="outline"
                  borderColor="green.500"
                  color="green.400"
                  _hover={{
                    bg: "green.500",
                    color: "white",
                    transform: "translateY(-2px)",
                  }}
                  transition="all 0.2s"
                >
                  Sign in to post
                </Button>
              )}
            </HStack>
          </Flex>

          {/* Posts Grid */}
          <VStack spacing={6} align="stretch">
            {isLoading && (
              <Flex justify="center" py={20}>
                <Text color="gray.500" fontSize="lg">
                  Loading posts...
                </Text>
              </Flex>
            )}
            {posts?.map((post: any) => <PostCard key={post.id} post={post} />)}
          </VStack>
        </Container>

        {/* Create Post Modal */}
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
              Create New Post
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
                    placeholder="Enter an engaging title..."
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
                    Content (Markdown Supported)
                  </FormLabel>
                  <Textarea
                    value={content}
                    onChange={(e) => setContent(e.target.value)}
                    placeholder="# Your Story&#10;&#10;Write your post in markdown...&#10;&#10;**Bold text**, *italic*, [links](url), and more!"
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
                  <Text color="gray.500" fontSize="xs" mt={2}>
                    💡 Tip: Use markdown syntax for formatting
                  </Text>
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
                onClick={() => create.mutate({ title, content })}
                isLoading={
                  (create as any).isPending ?? (create as any).isLoading
                }
                loadingText="Creating..."
                size="lg"
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
