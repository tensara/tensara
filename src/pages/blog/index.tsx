import {
  Box,
  Container,
  Heading,
  Text,
  Button,
  HStack,
  VStack,
  Flex,
  Input,
  InputGroup,
  InputLeftElement,
  Badge,
  Icon,
  IconButton,
  useToast,
  ButtonGroup,
  Menu,
  MenuButton,
  MenuList,
  MenuItem,
} from "@chakra-ui/react";
import Link from "next/link";
import { Layout } from "~/components/layout";
import { api } from "~/utils/api";
import { useSession, signIn } from "next-auth/react";
import { useRouter } from "next/router";
import { useState, useEffect } from "react";
import { saveBlogActiveTab, loadBlogActiveTab } from "~/utils/localStorage";
import { FiEdit, FiTrash, FiFilePlus } from "react-icons/fi";
import { FaSortAmountDown, FaSearch } from "react-icons/fa";

function useDebouncedValue<T>(value: T, delay = 300) {
  const [v, setV] = useState(value);
  useEffect(() => {
    const t = setTimeout(() => setV(value), delay);
    return () => clearTimeout(t);
  }, [value, delay]);
  return v;
}
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

function TagChips({
  tags,
}: {
  tags?: { tag: { id: string; name: string; slug: string } }[];
}) {
  if (!tags?.length) return null;
  return (
    <HStack spacing={1.5} mt={1} flexWrap="wrap">
      {tags.map((t) => (
        <Badge
          key={t.tag.id}
          px={2}
          py={0.5}
          rounded="md"
          colorScheme="green"
          variant="subtle"
          bg="green.900"
          color="green.200"
          border="1px solid"
          borderColor="green.700"
        >
          {t.tag.name}
        </Badge>
      ))}
    </HStack>
  );
}

function VotePill({ count }: { count: number | null | undefined }) {
  return (
    <Box
      minW="52px"
      textAlign="center"
      px={2.5}
      py={1.5}
      rounded="lg"
      border="1px solid"
      borderColor="gray.700"
      bg="gray.800"
    >
      <Text fontSize="sm" fontWeight="700" color="white">
        {typeof count === "number" ? count : 0}
      </Text>
      <Text fontSize="10px" color="gray.400" mt={-0.5}>
        votes
      </Text>
    </Box>
  );
}

export default function BlogIndex() {
  const router = useRouter();
  const { data: session } = useSession();

  const [query, setQuery] = useState("");
  const debouncedQuery = useDebouncedValue(query, 350);
  const [sort, setSort] = useState<"recent" | "top">("recent");
  const [activeTab, setActiveTab] = useState<"all" | "myPosts" | "myDrafts">(
    () => {
      // Load saved tab from localStorage on mount
      return loadBlogActiveTab() ?? "all";
    }
  );

  // Save tab to localStorage whenever it changes
  useEffect(() => {
    saveBlogActiveTab(activeTab);
  }, [activeTab]);
  const toast = useToast();
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const autosave = api.blogpost.autosave.useMutation();
  const createDraftMut = api.blogpost.createDraft.useMutation();

  const fetchTemplate = async (name: string) => {
    if (name === "blank") return "# ";
    const tries = [`/templates/${name}.md`, `/template/${name}.md`];
    for (const url of tries) {
      const res = await fetch(url);
      if (res.ok) return await res.text();
    }
    throw new Error(`Template "${name}" not found`);
  };

  const createFromTemplate = async (
    tpl: "blank" | "worklog" | "tutorial-submission" | "daily-log"
  ) => {
    if (!session) {
      await signIn();
      return;
    }
    try {
      // 1) create empty draft
      const draft = await createDraftMut.mutateAsync({
        title: "Untitled draft",
      });

      // 2) load template content
      const md = await fetchTemplate(tpl);

      // 3) prefill via autosave (no refetch → no rollback while typing)
      await autosave.mutateAsync({
        id: draft.id,
        title: tpl === "blank" ? "Untitled draft" : `Draft: ${tpl}`,
        content: md,
      });

      // 4) jump to editor
      await router.push(`/blog/edit/${draft.id}`);
    } catch (e: unknown) {
      if (e instanceof Error) {
        toast({
          title: "Could not start from template",
          description: e.message,
          status: "error",
        });
      } else {
        toast({
          title: "Could not start from template",
          description: "Unknown error",
          status: "error",
        });
      }
    }
  };

  const deletePost = api.blogpost.delete.useMutation({
    onMutate: async ({ id }) => {
      setDeletingId(id); // mark which row is deleting
    },
    onSettled: async () => {
      setDeletingId(null); // clear regardless of success/fail
    },
    onSuccess: async () => {
      await Promise.all([
        myDrafts.refetch(),
        minePublished.refetch(),
        pub.refetch(),
      ]);
      toast({ title: "Draft deleted", status: "info" });
    },
    onError: (err) => {
      toast({
        title: "Delete failed",
        description: err.message,
        status: "error",
      });
    },
  });

  // use debouncedQuery in queries (not query)
  const pub = api.blogpost.listPublished.useInfiniteQuery(
    { limit: 20, query: debouncedQuery, sort },
    { getNextPageParam: (last) => last.nextCursor, staleTime: 10_000 }
  );

  const minePublished = api.blogpost.listMine.useInfiniteQuery(
    { limit: 20, status: "PUBLISHED", query: debouncedQuery },
    {
      enabled: !!session,
      getNextPageParam: (last) => last.nextCursor,
      staleTime: 10_000,
    }
  );

  const myDrafts = api.blogpost.listMine.useInfiniteQuery(
    { limit: 20, status: "DRAFT", query: debouncedQuery },
    {
      enabled: !!session,
      getNextPageParam: (last) => last.nextCursor,
      staleTime: 10_000,
    }
  );

  const allPublished = pub.data?.pages.flatMap((p) => p.posts) ?? [];
  const minePub = minePublished.data?.pages.flatMap((p) => p.posts) ?? [];
  const drafts = myDrafts.data?.pages.flatMap((p) => p.posts) ?? [];

  const getGhostBtnStyles = (tabName?: "all" | "myPosts" | "myDrafts") => {
    const isTabButton = tabName !== undefined;
    const isActive = isTabButton && tabName === activeTab;

    return {
      bg: "gray.800",
      color: "gray.100",
      borderColor: "gray.700",
      _hover: {
        bg: isActive ? "green.700" : "gray.700",
        borderColor: isActive ? "green.600" : "gray.600",
      },
      _active: { bg: "green.700", borderColor: "green.600" },
      transition: "all 0.5s ease",
      rounded: "lg",
      cursor: "pointer",
    };
  };

  return (
    <Layout title="Blog">
      <Box>
        <Container maxW="7xl" mx="auto" py={8}>
          <Flex direction="row" align="center" mb={6} gap={4}>
            <Box w="full">
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

            <HStack spacing={4} w="full" align="stretch">
              <InputGroup flex="1">
                <InputLeftElement pointerEvents="none">
                  <FaSearch color="#d4d4d8" />
                </InputLeftElement>
                <Input
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  placeholder="Search title, text, tags…"
                  bg="whiteAlpha.50"
                  _hover={{ borderColor: "gray.600" }}
                  _focus={{ borderColor: "blue.500", boxShadow: "none" }}
                  color="white"
                />
              </InputGroup>
              {session ? (
                <Menu isLazy>
                  <MenuButton
                    as={Button}
                    leftIcon={<Icon as={FiFilePlus} />}
                    bg="green.700"
                    border="1px solid"
                    borderColor="whiteAlpha.100"
                    _hover={{ bg: "green.600", borderColor: "whiteAlpha.200" }}
                    _active={{ bg: "green.600", borderColor: "whiteAlpha.200" }}
                    _expanded={{
                      bg: "green.600",
                      borderColor: "whiteAlpha.200",
                    }}
                    _focus={{
                      bg: "green.600",
                      borderColor: "whiteAlpha.200",
                      boxShadow: "none",
                    }}
                    color="white"
                    borderRadius="md"
                    h="40px"
                    flexShrink={0}
                    sx={{
                      "&[aria-expanded='true']": {
                        bg: "green.600",
                        borderColor: "whiteAlpha.200",
                      },
                    }}
                  >
                    New
                  </MenuButton>
                  <MenuList
                    bg="brand.secondary"
                    borderColor="gray.800"
                    p={0}
                    borderRadius="md"
                    minW="100px"
                  >
                    <MenuItem
                      onClick={() => createFromTemplate("blank")}
                      bg="brand.secondary"
                      _hover={{ bg: "gray.700" }}
                      color="white"
                      borderRadius="md"
                    >
                      Blank Draft
                    </MenuItem>
                    <MenuItem
                      onClick={() => createFromTemplate("worklog")}
                      bg="brand.secondary"
                      _hover={{ bg: "gray.700" }}
                      color="white"
                      borderRadius="md"
                    >
                      Optimization Worklog
                    </MenuItem>
                    <MenuItem
                      onClick={() => createFromTemplate("tutorial-submission")}
                      bg="brand.secondary"
                      _hover={{ bg: "gray.700" }}
                      color="white"
                      borderRadius="md"
                    >
                      Tutorial Submission
                    </MenuItem>
                    <MenuItem
                      onClick={() => createFromTemplate("daily-log")}
                      bg="brand.secondary"
                      _hover={{ bg: "gray.700" }}
                      color="white"
                      borderRadius="md"
                    >
                      Daily Log
                    </MenuItem>
                  </MenuList>
                </Menu>
              ) : (
                <Button size="sm" colorScheme="blue" onClick={() => signIn()}>
                  Sign in
                </Button>
              )}
            </HStack>
          </Flex>

          <Flex
            align="center"
            justify="space-between"
            mb={4}
            gap={4}
            flexWrap="wrap"
          >
            <ButtonGroup size="sm" spacing={2}>
              <Button
                onClick={() => setActiveTab("all")}
                {...getGhostBtnStyles("all")}
                aria-pressed={activeTab === "all"}
                {...(activeTab === "all"
                  ? {
                      bg: "green.700",
                      borderColor: "green.600",
                      color: "white",
                    }
                  : {})}
              >
                All
              </Button>
              <Button
                onClick={() => session && setActiveTab("myPosts")}
                isDisabled={!session}
                {...getGhostBtnStyles("myPosts")}
                aria-pressed={activeTab === "myPosts"}
                {...(activeTab === "myPosts"
                  ? {
                      bg: "green.700",
                      borderColor: "green.600",
                      color: "white",
                    }
                  : {})}
                _disabled={{
                  opacity: 0.4,
                  cursor: "not-allowed",
                }}
              >
                My Posts
                {session && (
                  <Badge
                    ml={1.5}
                    px={1.5}
                    py={0.5}
                    bg={activeTab === "myPosts" ? "green.600" : "gray.700"}
                    color="white"
                    fontSize="10px"
                    fontWeight="700"
                    borderRadius="md"
                    lineHeight="1"
                    minW="20px"
                    height="18px"
                    display="inline-flex"
                    alignItems="center"
                    justifyContent="center"
                    border="none"
                    transition="all 0.5s ease"
                  >
                    {minePub.length}
                  </Badge>
                )}
              </Button>
              <Button
                onClick={() => session && setActiveTab("myDrafts")}
                isDisabled={!session}
                {...getGhostBtnStyles("myDrafts")}
                aria-pressed={activeTab === "myDrafts"}
                {...(activeTab === "myDrafts"
                  ? {
                      bg: "green.700",
                      borderColor: "green.600",
                      color: "white",
                    }
                  : {})}
                _disabled={{
                  opacity: 0.4,
                  cursor: "not-allowed",
                }}
              >
                My Drafts
                {session && (
                  <Badge
                    ml={1.5}
                    px={1.5}
                    py={0.5}
                    bg={activeTab === "myDrafts" ? "green.600" : "gray.700"}
                    color="white"
                    fontSize="10px"
                    fontWeight="700"
                    borderRadius="md"
                    lineHeight="1"
                    minW="20px"
                    height="18px"
                    display="inline-flex"
                    alignItems="center"
                    justifyContent="center"
                    border="none"
                    transition="all 0.5s ease"
                  >
                    {drafts.length}
                  </Badge>
                )}
              </Button>
            </ButtonGroup>

            {activeTab === "all" && (
              <Button
                size="sm"
                leftIcon={<Icon as={FaSortAmountDown} />}
                onClick={() => setSort(sort === "recent" ? "top" : "recent")}
                {...getGhostBtnStyles()}
                fontSize="sm"
                fontWeight="medium"
              >
                {sort === "recent" ? "Most recent" : "Most voted"}
              </Button>
            )}
          </Flex>

          <Box>
            {activeTab === "all" && (
              <VStack align="stretch" spacing={0}>
                {allPublished.length === 0 ? (
                  <Box py={8} textAlign="center">
                    <Text color="gray.400" fontSize="md">
                      {debouncedQuery
                        ? "No posts found matching your search."
                        : "No published posts yet."}
                    </Text>
                  </Box>
                ) : (
                  allPublished.map((post) => (
                    <Flex
                      key={post.id}
                      py={3}
                      borderBottom="1px solid"
                      borderColor="gray.800"
                      align="center"
                      gap={3}
                      flexWrap="wrap"
                    >
                      <Box flex="1" minW={0}>
                        <Link href={`/blog/${post.slug ?? post.id}`}>
                          <Text color="white" fontWeight="600" noOfLines={1}>
                            {post.title}
                          </Text>
                        </Link>
                        <Text color="gray.500" fontSize="sm" noOfLines={1}>
                          by {post.author?.name ?? "Unknown"} •{" "}
                          {timeAgo(post.publishedAt ?? post.createdAt)}
                        </Text>
                      </Box>

                      <TagChips tags={post.tags} />

                      <VotePill count={post._count?.upvotes} />
                    </Flex>
                  ))
                )}
              </VStack>
            )}

            {activeTab === "myPosts" && (
              <VStack align="stretch" spacing={0}>
                {minePub.length === 0 ? (
                  <Box py={8} textAlign="center">
                    <Text color="gray.400" fontSize="md">
                      {debouncedQuery
                        ? "No published posts found matching your search."
                        : "You haven't published any posts yet."}
                    </Text>
                  </Box>
                ) : (
                  minePub.map((post) => (
                    <Flex
                      key={post.id}
                      py={3}
                      borderBottom="1px solid"
                      borderColor="gray.800"
                      align="center"
                      gap={3}
                      flexWrap="wrap"
                    >
                      <Box flex="1" minW={0}>
                        <Link href={`/blog/${post.slug ?? post.id}`}>
                          <Text color="white" fontWeight="600" noOfLines={1}>
                            {post.title}
                          </Text>
                        </Link>
                        <Text color="gray.500" fontSize="sm">
                          {timeAgo(post.publishedAt ?? post.updatedAt)}
                        </Text>
                      </Box>
                      <HStack>
                        <Link
                          href={`/blog/edit/${post.id}`}
                          passHref
                          legacyBehavior
                        >
                          <IconButton
                            as="a"
                            size="sm"
                            icon={<Icon as={FiEdit} />}
                            aria-label="Edit post"
                            {...getGhostBtnStyles()}
                          />
                        </Link>
                      </HStack>
                    </Flex>
                  ))
                )}
              </VStack>
            )}

            {activeTab === "myDrafts" && (
              <VStack align="stretch" spacing={0}>
                {drafts.length === 0 ? (
                  <Box py={8} textAlign="center">
                    <Text color="gray.400" fontSize="md">
                      {debouncedQuery
                        ? "No drafts found matching your search."
                        : "You don't have any drafts yet. Create one to get started!"}
                    </Text>
                  </Box>
                ) : (
                  drafts.map((post) => (
                    <Flex
                      key={post.id}
                      py={3}
                      borderBottom="1px solid"
                      borderColor="gray.800"
                      align="center"
                      gap={3}
                      flexWrap="wrap"
                    >
                      <Box flex="1" minW={0}>
                        <Link href={`/blog/edit/${post.id}`}>
                          <Text color="white" fontWeight="600" noOfLines={1}>
                            {post.title || "Untitled draft"}
                          </Text>
                        </Link>
                        <Text color="gray.500" fontSize="sm">
                          saved {timeAgo(post.updatedAt)}
                        </Text>
                      </Box>

                      <HStack spacing={2}>
                        <IconButton
                          size="sm"
                          onClick={() => {
                            if (deletingId) return;
                            if (confirm("Delete this draft permanently?")) {
                              deletePost.mutate({ id: post.id });
                            }
                          }}
                          isLoading={
                            deletingId === post.id && deletePost.isPending
                          }
                          isDisabled={!!deletingId && deletingId !== post.id}
                          icon={<Icon as={FiTrash} />}
                          aria-label="Delete draft"
                          {...getGhostBtnStyles()}
                        />
                      </HStack>
                    </Flex>
                  ))
                )}
              </VStack>
            )}
          </Box>

          {activeTab === "all" && pub.hasNextPage && (
            <Flex justify="center" py={4}>
              <Button
                onClick={() => pub.fetchNextPage()}
                isLoading={pub.isFetchingNextPage}
                aria-label="Load more posts"
                size="sm"
                {...getGhostBtnStyles()}
                rounded="full"
                px={4}
              >
                <Box
                  as="svg"
                  width="18"
                  height="18"
                  viewBox="0 0 24 24"
                  fill="none"
                  xmlns="http://www.w3.org/2000/svg"
                  mr={2}
                >
                  <path
                    d="M6 9l6 6 6-6"
                    stroke="currentColor"
                    strokeWidth="2"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                </Box>
                Load more
              </Button>
            </Flex>
          )}
        </Container>
      </Box>
    </Layout>
  );
}
