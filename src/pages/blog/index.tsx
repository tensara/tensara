// src/pages/blog/index.tsx
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
} from "@chakra-ui/react";
import Link from "next/link";
import { Layout } from "~/components/layout";
import { api } from "~/utils/api";
import { useSession, signIn } from "next-auth/react";
import { useRouter } from "next/router";
import { useState, useEffect } from "react";

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

function StatusBadge({
  status,
}: {
  status: "DRAFT" | "PUBLISHED" | "ARCHIVED";
}) {
  const color =
    status === "PUBLISHED" ? "green" : status === "DRAFT" ? "yellow" : "gray";
  return (
    <Badge colorScheme={color} variant="subtle" rounded="md" fontWeight="600">
      {status.toLowerCase()}
    </Badge>
  );
}

export default function BlogIndex() {
  const router = useRouter();
  const { data: session } = useSession();

  // inside BlogIndex()
  const [query, setQuery] = useState("");
  const debouncedQuery = useDebouncedValue(query, 350);

  // use debouncedQuery in queries (not query)
  const pub = api.blogpost.listPublished.useInfiniteQuery(
    { limit: 20, query: debouncedQuery },
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

  return (
    <Layout title="Blog">
      <Box bg="gray.900" minH="100vh">
        <Container maxW="860px" py={8}>
          <Flex align="center" justify="space-between" mb={6} gap={4}>
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
              <Input
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Search title, text, tags…"
                size="sm"
                bg="gray.800"
                borderColor="gray.700"
                color="gray.100"
                width="260px"
              />
              {session ? (
                <Button
                  size="sm"
                  colorScheme="blue"
                  onClick={() =>
                    createDraft.mutate({ title: "Untitled draft" })
                  }
                >
                  New Draft
                </Button>
              ) : (
                <Button size="sm" colorScheme="blue" onClick={() => signIn()}>
                  Sign in
                </Button>
              )}
            </HStack>
          </Flex>

          <Tabs colorScheme="blue" variant="enclosed">
            <TabList>
              <Tab>All</Tab>
              <Tab isDisabled={!session}>
                My Posts{" "}
                {session ? <Badge ml={2}>{minePub.length}</Badge> : null}
              </Tab>
              <Tab isDisabled={!session}>
                My Drafts{" "}
                {session ? <Badge ml={2}>{drafts.length}</Badge> : null}
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
                      align="baseline"
                      gap={3}
                    >
                      <Box flex="1">
                        <Link href={`/blog/${post.slug ?? post.id}`}>
                          <Text color="white" fontWeight="600" noOfLines={1}>
                            {post.title}
                          </Text>
                        </Link>
                        <Text color="gray.500" fontSize="sm">
                          by {post.author?.name ?? "Unknown"} •{" "}
                          {timeAgo(post.publishedAt ?? post.createdAt)}
                        </Text>
                      </Box>
                      <StatusBadge status="PUBLISHED" />
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
                        <StatusBadge status="PUBLISHED" />
                        <Button
                          as={Link}
                          href={`/blog/edit/${post.id}`}
                          size="xs"
                          variant="outline"
                        >
                          Edit
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
                      <StatusBadge status="DRAFT" />
                    </Flex>
                  ))}
                </VStack>
              </TabPanel>
            </TabPanels>
          </Tabs>
        </Container>
      </Box>

      {pub.hasNextPage && (
        <Flex justify="center" py={4}>
          <Button
            onClick={() => pub.fetchNextPage()}
            isLoading={pub.isFetchingNextPage}
            size="sm"
            variant="outline"
          >
            Load more
          </Button>
        </Flex>
      )}
    </Layout>
  );
}
