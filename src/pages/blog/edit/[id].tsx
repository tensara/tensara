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
  useToast,
  Divider,
  Kbd,
  Badge,
} from "@chakra-ui/react";
import { Layout } from "~/components/layout";
import { useSession, signIn } from "next-auth/react";
import { useRouter } from "next/router";
import { useEffect, useMemo, useRef, useState } from "react";
import { api } from "~/utils/api";
import { FiArrowLeft, FiSave, FiEye, FiEdit, FiUpload } from "react-icons/fi";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

function useDebounced(fn: () => void, deps: unknown[], delay = 1500) {
  const timeout = useRef<ReturnType<typeof setTimeout>>();
  useEffect(() => {
    clearTimeout(timeout.current);
    timeout.current = setTimeout(fn, delay);
    return () => clearTimeout(timeout.current);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);
}

export default function EditPost() {
  const router = useRouter();
  const { id } = router.query as { id?: string };
  const { data: session, status } = useSession();
  const toast = useToast();
  const utils = api.useContext();

  // Fetch existing (draft or published)
  const postQ = api.blogpost.getById.useQuery(
    { id: id ?? "" },
    { enabled: !!id }
  );
  const autosave = api.blogpost.autosave.useMutation();
  const publish = api.blogpost.publish.useMutation({
    onSuccess: async (p) => {
      await Promise.all([
        utils.blogpost.listMine.invalidate(),
        utils.blogpost.listPublished.invalidate(),
        utils.blogpost.getById.invalidate({ id: p.id }),
      ]);
      toast({ title: "Published!", status: "success" });
      router.push(`/blog/${p.slug ?? p.id}`);
    },
    onError: (err) =>
      toast({
        title: "Publish failed",
        description: err.message,
        status: "error",
      }),
  });

  const [title, setTitle] = useState("");
  const [content, setContent] = useState("");
  const [excerpt, setExcerpt] = useState("");
  const [tags, setTags] = useState<string[]>([]);
  const [showPreview, setShowPreview] = useState(false);
  const [dirty, setDirty] = useState(false);

  // hydrate form whenever post changes
  useEffect(() => {
    const p = postQ.data;
    if (!p) return;
    setTitle(p.title ?? "");
    setContent(p.content ?? "");
    setExcerpt(p.excerpt ?? "");
    // Accept either string[] or { tag: { slug } }[]
    const nextTags =
      Array.isArray(p.tags) && typeof p.tags[0] !== "string"
        ? (p.tags as any[])
            .map((t) => t?.tag?.slug || t?.tag?.name || "")
            .filter(Boolean)
        : ((p.tags as any[]) ?? []);
    setTags(nextTags);
    setDirty(false);
  }, [postQ.data]);

  // mark dirty on changes
  useEffect(() => {
    const p = postQ.data;
    if (!p) return;
    const equalTags =
      JSON.stringify(tags) ===
      JSON.stringify(
        Array.isArray(p.tags) && typeof p.tags[0] !== "string"
          ? (p.tags as any[])
              .map((t) => t?.tag?.slug || t?.tag?.name || "")
              .filter(Boolean)
          : ((p.tags as any[]) ?? [])
      );
    setDirty(
      title !== (p.title ?? "") ||
        content !== (p.content ?? "") ||
        excerpt !== (p.excerpt ?? "") ||
        !equalTags
    );
  }, [title, content, excerpt, tags, postQ.data]);

  // Debounced server autosave
  useDebounced(
    () => {
      if (!dirty || !id) return;
      autosave.mutate(
        // If your server expects tagIds instead, resolve them there and keep this 'tags' string[].
        { id, title, content, excerpt },
        { onSuccess: () => postQ.refetch() }
      );
    },
    [dirty, title, content, excerpt, id],
    1200
  );

  // beforeunload guard
  useEffect(() => {
    const handler = (e: BeforeUnloadEvent) => {
      if (!dirty) return;
      e.preventDefault();
      e.returnValue = "";
    };
    window.addEventListener("beforeunload", handler);
    return () => window.removeEventListener("beforeunload", handler);
  }, [dirty]);

  // route change guard
  useEffect(() => {
    const onRoute = (url: string) => {
      if (dirty && url !== router.asPath) {
        if (!confirm("You have unsaved changes. Leave anyway?")) {
          router.events.emit("routeChangeError");
          throw "routeChange aborted";
        }
      }
    };
    router.events.on("routeChangeStart", onRoute);
    return () => router.events.off("routeChangeStart", onRoute);
  }, [dirty, router]);

  // cmd/ctrl+s save
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "s") {
        e.preventDefault();
        if (id) autosave.mutate({ id, title, content, excerpt });
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [id, title, content, excerpt]);

  // status badge
  const statusBadge = useMemo(() => {
    const s = postQ.data?.status as
      | "PUBLISHED"
      | "DRAFT"
      | "ARCHIVED"
      | undefined;
    if (!s) return null;
    const color =
      s === "PUBLISHED" ? "green" : s === "DRAFT" ? "yellow" : "gray";
    return (
      <Badge colorScheme={color} variant="subtle" rounded="md" fontWeight="600">
        {s.toLowerCase()}
      </Badge>
    );
  }, [postQ.data?.status]);

  if (status === "loading" || postQ.isLoading) {
    return (
      <Layout title="Edit Post">
        <Container maxW="5xl" py={12}>
          <Text color="gray.500">Loading…</Text>
        </Container>
      </Layout>
    );
  }

  if (status === "unauthenticated") {
    void signIn();
    return null;
  }

  if (postQ.isError) {
    return (
      <Layout title="Edit Post">
        <Container maxW="5xl" py={12}>
          <Text color="red.300">Post not found.</Text>
          <Button mt={4} onClick={() => router.push("/blog")}>
            Back to blog
          </Button>
        </Container>
      </Layout>
    );
  }

  const onPublish = () => {
    if (!id) return;
    if (autosave.isPending) {
      toast({
        title: "Saving…",
        description: "Please wait a moment.",
        status: "info",
      });
      return;
    }
    if (!title.trim() || !content.trim()) {
      toast({ title: "Title and content required", status: "warning" });
      return;
    }
    publish.mutate({ id });
  };

  const onSaveNow = () => {
    if (!id) return;
    autosave.mutate(
      { id, title, content, excerpt },
      { onSuccess: () => toast({ title: "Saved", status: "success" }) }
    );
  };

  return (
    <Layout title={dirty ? "● Editing…" : "Edit Post"}>
      <Box bg="gray.900" minH="100vh">
        <Container maxW="900px" py={8}>
          <Flex align="center" justify="space-between" mb={4}>
            <HStack>
              <Button
                variant="ghost"
                leftIcon={<Icon as={FiArrowLeft} />}
                onClick={() => router.push("/blog")}
              >
                Back
              </Button>
              {statusBadge}
            </HStack>
            <HStack spacing={2}>
              <Button
                variant="outline"
                leftIcon={<Icon as={FiSave} />}
                onClick={onSaveNow}
                isLoading={autosave.isPending}
              >
                Save <Kbd ml={2}>⌘/Ctrl</Kbd>+<Kbd>S</Kbd>
              </Button>
              <Button
                colorScheme="green"
                leftIcon={<Icon as={FiUpload} />}
                onClick={onPublish}
                isDisabled={
                  autosave.isPending || !title.trim() || !content.trim()
                }
                isLoading={publish.isPending}
              >
                Publish
              </Button>
            </HStack>
          </Flex>

          <Divider mb={6} borderColor="whiteAlpha.300" />

          <VStack align="stretch" spacing={6}>
            <FormControl isRequired>
              <FormLabel color="gray.300" fontWeight="600" fontSize="sm">
                Title
              </FormLabel>
              <Input
                value={title}
                onChange={(e) => setTitle(e.target.value)}
                placeholder="Catchy, clear title"
                bg="transparent"
                borderColor="whiteAlpha.300"
                color="gray.100"
              />
            </FormControl>

            <FormControl>
              <FormLabel color="gray.300" fontWeight="600" fontSize="sm">
                Excerpt (optional)
              </FormLabel>
              <Input
                value={excerpt}
                onChange={(e) => setExcerpt(e.target.value.slice(0, 200))}
                placeholder="One-liner used in previews"
                bg="transparent"
                borderColor="whiteAlpha.300"
                color="gray.100"
              />
              <Text color="gray.500" fontSize="xs">
                {excerpt.length}/200
              </Text>
            </FormControl>

            <FormControl>
              <FormLabel color="gray.300" fontWeight="600" fontSize="sm">
                Tags (comma separated)
              </FormLabel>
              <Input
                value={tags.join(", ")}
                onChange={(e) =>
                  setTags(
                    e.currentTarget.value
                      .split(",")
                      .map((t) => t.trim())
                      .filter(Boolean)
                  )
                }
                placeholder="zk, rust, distributed-systems"
                bg="transparent"
                borderColor="whiteAlpha.300"
                color="gray.100"
              />
            </FormControl>

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
                    colorScheme="green"
                    leftIcon={<Icon as={FiEdit} />}
                    onClick={() => setShowPreview(false)}
                  >
                    Edit
                  </Button>
                  <Button
                    size="xs"
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
                <Textarea
                  value={content}
                  onChange={(e) => setContent(e.target.value)}
                  placeholder="# Your Story\n\nWrite your post in markdown…"
                  minH="420px"
                  fontFamily="'JetBrains Mono', ui-monospace, SFMono-Regular"
                  fontSize="sm"
                  bg="transparent"
                  borderColor="whiteAlpha.300"
                  color="gray.100"
                />
              ) : (
                <Box
                  minH="420px"
                  border="1px solid"
                  borderColor="whiteAlpha.200"
                  rounded="md"
                  p={4}
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
                    "& a": { color: "green.300", textDecoration: "underline" },
                  }}
                >
                  {content ? (
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>
                      {content}
                    </ReactMarkdown>
                  ) : (
                    <Text color="gray.500" fontStyle="italic">
                      No content yet.
                    </Text>
                  )}
                </Box>
              )}
            </FormControl>

            <Flex justify="space-between" pt={2}>
              <HStack color="gray.500" fontSize="sm">
                <Text>Autosaves run after idle.</Text>
                {dirty ? (
                  <Badge colorScheme="yellow">unsaved</Badge>
                ) : (
                  <Badge colorScheme="green">saved</Badge>
                )}
              </HStack>
            </Flex>
          </VStack>
        </Container>
      </Box>
    </Layout>
  );
}
