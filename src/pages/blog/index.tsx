import {
  Box,
  Container,
  Heading,
  Text,
  Button,
  Tabs,
  TabList,
  Tab,
  TabPanels,
  TabPanel,
  HStack,
  VStack,
  Flex,
  Input,
  Badge,
  Icon,
  useToast,
  ButtonGroup,
  VisuallyHidden,
} from "@chakra-ui/react";
import Link from "next/link";
import { Layout } from "~/components/layout";
import { api } from "~/utils/api";
import { useSession, signIn } from "next-auth/react";
import { useRouter } from "next/router";
import { useState, useEffect } from "react";
import { FiEdit, FiTrash } from "react-icons/fi";

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

  // inside BlogIndex()
  const [query, setQuery] = useState("");
  const debouncedQuery = useDebouncedValue(query, 350);
  const [sort, setSort] = useState<"recent" | "top">("recent");
  const toast = useToast();
  const [deletingId, setDeletingId] = useState<string | null>(null);

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

  const utils = api.useContext();
  const createDraft = api.blogpost.createDraft.useMutation({
    onSuccess: async (post) => {
      await utils.blogpost.listMine.invalidate();
      router.push(`/blog/edit/${post.id}`);
    },
  });

  const allPublished = pub.data?.pages.flatMap((p) => p.posts) ?? [];
  const minePub = minePublished.data?.pages.flatMap((p) => p.posts) ?? [];
  const drafts = myDrafts.data?.pages.flatMap((p) => p.posts) ?? [];

  // common button styles (dark theme friendly)
  const solidBtn = {
    bg: "green.600",
    color: "white",
    border: "1px solid",
    borderColor: "green.500",
    _hover: { bg: "green.500", borderColor: "green.400" },
    _active: { bg: "green.700", borderColor: "green.600" },
    rounded: "lg",
  } as const;

  const ghostBtn = {
    bg: "gray.800",
    color: "gray.100",
    border: "1px solid",
    borderColor: "gray.700",
    _hover: { bg: "gray.700", borderColor: "gray.600" },
    _active: { bg: "gray.600", borderColor: "gray.500" },
    rounded: "lg",
  } as const;

  return (
    <Layout title="Blog">
      <Box minH="100vh">
        <Container maxW="860px" py={8}>
          <Flex
            align="center"
            justify="space-between"
            mb={6}
            gap={4}
            flexWrap="wrap"
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

            <HStack
              spacing={2}
              flexWrap="wrap"
              w={{ base: "100%", md: "auto" }}
            >
              <Input
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Search title, text, tags…"
                size="sm"
                bg="gray.800"
                borderColor="gray.700"
                color="gray.100"
                w={{ base: "100%", md: "260px" }}
                _placeholder={{ color: "gray.500" }}
                rounded="lg"
              />
              <ButtonGroup size="sm" isAttached variant="outline">
                <Button
                  onClick={() => setSort("recent")}
                  {...ghostBtn}
                  borderRightRadius={0}
                  aria-pressed={sort === "recent"}
                  {...(sort === "recent"
                    ? {
                        bg: "green.700",
                        borderColor: "green.600",
                        color: "white",
                      }
                    : {})}
                >
                  Most recent
                </Button>
                <Button
                  onClick={() => setSort("top")}
                  {...ghostBtn}
                  borderLeftRadius={0}
                  aria-pressed={sort === "top"}
                  {...(sort === "top"
                    ? {
                        bg: "green.700",
                        borderColor: "green.600",
                        color: "white",
                      }
                    : {})}
                >
                  Most voted
                </Button>
              </ButtonGroup>
              {session ? (
                <Button size="sm" leftIcon={<Icon as={FiEdit} />} {...solidBtn}>
                  <Link
                    href="#"
                    onClick={() =>
                      createDraft.mutate({ title: "Untitled draft" })
                    }
                  >
                    New Draft
                  </Link>
                </Button>
              ) : (
                <Button size="sm" {...solidBtn} onClick={() => signIn()}>
                  Sign in
                </Button>
              )}
            </HStack>
          </Flex>

          <Tabs colorScheme="green" variant="enclosed" isFitted>
            <TabList borderColor="gray.800">
              <Tab
                _selected={{
                  bg: "gray.800",
                  color: "white",
                  borderColor: "green.600",
                }}
              >
                All
              </Tab>
              <Tab
                isDisabled={!session}
                _selected={{
                  bg: "gray.800",
                  color: "white",
                  borderColor: "green.600",
                }}
              >
                My Posts{" "}
                {session ? (
                  <Badge ml={2} colorScheme="green">
                    {minePub.length}
                  </Badge>
                ) : null}
              </Tab>
              <Tab
                isDisabled={!session}
                _selected={{
                  bg: "gray.800",
                  color: "white",
                  borderColor: "green.600",
                }}
              >
                My Drafts{" "}
                {session ? (
                  <Badge ml={2} colorScheme="green">
                    {drafts.length}
                  </Badge>
                ) : null}
              </Tab>
            </TabList>
            <TabPanels>
              <TabPanel px={0}>
                <VStack align="stretch" spacing={0}>
                  {allPublished.map((post) => (
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
                  ))}
                </VStack>
              </TabPanel>

              <TabPanel px={0}>
                <VStack align="stretch" spacing={0}>
                  {minePub.map((post) => (
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
                        <Button
                          as={Link}
                          href={`/blog/edit/${post.id}`}
                          size="xs"
                          {...ghostBtn}
                          leftIcon={<Icon as={FiEdit} />}
                          aria-label="Edit post"
                        >
                          <VisuallyHidden>Edit</VisuallyHidden>
                        </Button>
                      </HStack>
                    </Flex>
                  ))}
                </VStack>
              </TabPanel>

              <TabPanel px={0}>
                <VStack align="stretch" spacing={0}>
                  {drafts.map((post) => (
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
                        <Button
                          size="xs"
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
                          leftIcon={<Icon as={FiTrash} />}
                          {...ghostBtn}
                          aria-label="Delete draft"
                        >
                          <VisuallyHidden>Delete</VisuallyHidden>
                        </Button>
                      </HStack>
                    </Flex>
                  ))}
                </VStack>
              </TabPanel>
            </TabPanels>
          </Tabs>
        </Container>
      </Box>

      {pub.hasNextPage && (
        <Flex justify="center" py={4} bg="black">
          <Button
            onClick={() => pub.fetchNextPage()}
            isLoading={pub.isFetchingNextPage}
            aria-label="Load more posts"
            size="sm"
            {...ghostBtn}
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
    </Layout>
  );
}
