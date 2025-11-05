// src/pages/blog/edit/[id].tsx
import {
  Box,
  Container,
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
  type BadgeProps,
} from "@chakra-ui/react";
import { Layout } from "~/components/layout";
import { useSession, signIn } from "next-auth/react";
import { useRouter } from "next/router";
import { useEffect, useMemo, useRef, useState } from "react";
import { api } from "~/utils/api";
import { FiArrowLeft, FiSave, FiEye, FiEdit, FiUpload } from "react-icons/fi";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

// --- utils ---
function useDebounced(fn: () => void, deps: unknown[], delay = 1200) {
  const timeout = useRef<ReturnType<typeof setTimeout>>();
  useEffect(() => {
    clearTimeout(timeout.current);
    timeout.current = setTimeout(fn, delay);
    return () => clearTimeout(timeout.current);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);
}

function parseTags(input: string): string[] {
  // accept commas OR spaces as separators; dedupe; lowercase slugs
  return Array.from(
    new Set(
      input
        .split(/[,\s]+/g)
        .map((s) => s.trim())
        .filter(Boolean)
        .map((s) => s.toLowerCase())
    )
  );
}

function tagsToInput(tags: string[]): string {
  // render as comma+space separated for readability
  return tags.join(", ");
}

export default function EditPost() {
  const router = useRouter();
  const { id } = router.query as { id?: string };
  const { status } = useSession();
  const toast = useToast();
  const utils = api.useContext();

  // Fetch existing (draft or published)
  const postQ = api.blogpost.getById.useQuery(
    { id: id ?? "" },
    { enabled: !!id }
  );

  // Mutations
  const autosave = api.blogpost.autosave.useMutation();
  const publish = api.blogpost.publish.useMutation({
    onSuccess: async (p) => {
      await Promise.all([
        utils.blogpost.listMine.invalidate(),
        utils.blogpost.listPublished.invalidate(),
        utils.blogpost.getById.invalidate({ id: p.id }),
      ]);
      toast({ title: "Published!", status: "success" });
      void router.push(`/blog/${p.slug ?? p.id}`);
    },
    onError: (err) =>
      toast({
        title: "Publish failed",
        description: err.message,
        status: "error",
      }),
  });

  // --- form state (single source of truth) ---
  const hasHydrated = useRef(false);
  const lastServerUpdatedAt = useRef<number | null>(null);

  const [title, setTitle] = useState("");
  const [content, setContent] = useState("");

  // tags: store as array + input string (accept comma OR space)
  const [tags, setTags] = useState<string[]>([]);
  const [tagsInput, setTagsInput] = useState("");

  const [showPreview, setShowPreview] = useState(false);
  const [dirty, setDirty] = useState(false);
  const [saving, setSaving] = useState<"idle" | "saving" | "saved">("idle");

  // hydrate ONCE when data arrives; never force-reset while typing
  useEffect(() => {
    if (!postQ.data || hasHydrated.current) return;
    const p = postQ.data;

    setTitle(p.title ?? "");
    setContent(p.content ?? "");

    // normalise tags safely (avoid using `any`)
    const tagToString = (entry: unknown): string => {
      if (!entry || typeof entry !== "object") return "";
      const obj = entry as Record<string, unknown>;
      const tagField = obj.tag;
      if (tagField && typeof tagField === "object") {
        const tagObj = tagField as Record<string, unknown>;
        const slug = tagObj.slug;
        const name = tagObj.name;
        if (typeof slug === "string" && slug) return slug;
        if (typeof name === "string" && name) return name;
      }
      const slug = obj.slug;
      const name = obj.name;
      if (typeof slug === "string" && slug) return slug;
      if (typeof name === "string" && name) return name;
      return "";
    };

    const initialTags: string[] =
      Array.isArray(p.tags) && p.tags.length
        ? (p.tags as unknown[])
            .map(tagToString)
            .filter(Boolean)
            .map((s) => s.toLowerCase())
        : [];

    setTags(initialTags);
    setTagsInput(tagsToInput(initialTags));
    setDirty(false);
    hasHydrated.current = true;

    // remember server timestamp if available
    if (p.updatedAt)
      lastServerUpdatedAt.current = new Date(p.updatedAt).getTime();
  }, [postQ.data]);

  // mark dirty on any change (simple shallow compare to hydrated snapshot)
  useEffect(() => {
    if (!hasHydrated.current || !postQ.data) return;
    const p = postQ.data;
    const tagToString = (entry: unknown): string => {
      if (!entry || typeof entry !== "object") return "";
      const obj = entry as Record<string, unknown>;
      const tagField = obj.tag;
      if (tagField && typeof tagField === "object") {
        const tagObj = tagField as Record<string, unknown>;
        const slug = tagObj.slug;
        const name = tagObj.name;
        if (typeof slug === "string" && slug) return slug;
        if (typeof name === "string" && name) return name;
      }
      const slug = obj.slug;
      const name = obj.name;
      if (typeof slug === "string" && slug) return slug;
      if (typeof name === "string" && name) return name;
      return "";
    };

    const serverTags =
      Array.isArray(p.tags) && p.tags.length
        ? (p.tags as unknown[])
            .map(tagToString)
            .filter(Boolean)
            .map((s) => s.toLowerCase())
        : [];

    const equal =
      title === (p.title ?? "") &&
      content === (p.content ?? "") &&
      JSON.stringify(tags) === JSON.stringify(serverTags);

    setDirty(!equal);
  }, [title, content, tags, postQ.data]);

  // Debounced autosave: NEVER refetch here (prevents rollbacks)
  useDebounced(
    () => {
      if (!dirty || !id) return;
      setSaving("saving");
      // prefix with void to satisfy no-floating-promises
      void autosave.mutate(
        { id, title, content, tags },
        {
          onSuccess: (res) => {
            // mark as saved; do not refetch
            setSaving("saved");
            setTimeout(() => setSaving("idle"), 1200);
            // optionally track server updatedAt if mutation returns it
            try {
              const result = res as { updatedAt?: string | Date };
              if (result?.updatedAt) {
                // support Date from server (Prisma) or ISO string
                lastServerUpdatedAt.current = new Date(
                  result.updatedAt
                ).getTime();
              }
            } catch {}
          },
          onError: (err) => {
            setSaving("idle");
            toast({
              title: "Autosave failed",
              description: err.message,
              status: "error",
            });
          },
        }
      );
    },
    // include autosave and toast to satisfy exhaustive-deps
    [dirty, title, content, tags, id, autosave, toast],
    800
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
          // throwing cancels the navigation in Next.js
          throw new Error("routeChange aborted");
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
        if (id) {
          setSaving("saving");
          void autosave.mutate(
            { id, title, content, tags },
            {
              onSuccess: () => {
                setSaving("saved");
                setTimeout(() => setSaving("idle"), 1200);
                toast({ title: "Saved", status: "success" });
              },
              onError: (err) => {
                setSaving("idle");
                toast({
                  title: "Save failed",
                  description: err.message,
                  status: "error",
                });
              },
            }
          );
        }
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [id, title, content, tags, autosave, toast]);

  // status badge
  const statusBadge = useMemo(() => {
    const s = postQ.data?.status as
      | "PUBLISHED"
      | "DRAFT"
      | "ARCHIVED"
      | undefined;
    if (!s) return null;
    const color: BadgeProps["colorScheme"] =
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
    setSaving("saving");
    void autosave.mutate(
      { id, title, content, tags },
      {
        onSuccess: () => {
          setSaving("saved");
          setTimeout(() => setSaving("idle"), 1200);
          toast({ title: "Saved", status: "success" });
        },
        onError: (err) => {
          setSaving("idle");
          toast({
            title: "Save failed",
            description: err.message,
            status: "error",
          });
        },
      }
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
                isLoading={autosave.isPending || saving === "saving"}
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
            {/* Title */}
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

            {/* Tags */}
            <FormControl>
              <FormLabel color="gray.300" fontWeight="600" fontSize="sm">
                Tags (comma or space separated)
              </FormLabel>
              <Input
                value={tagsInput}
                onChange={(e) => {
                  const val = e.currentTarget.value;
                  setTagsInput(val);
                  setTags(parseTags(val));
                }}
                placeholder="matrix, graphs, cryptography"
                bg="transparent"
                borderColor="whiteAlpha.300"
                color="gray.100"
              />
              {tags.length ? (
                <HStack spacing={1.5} mt={2} wrap="wrap">
                  {tags.map((t) => (
                    <Badge
                      key={t}
                      px={2}
                      py={0.5}
                      rounded="md"
                      colorScheme="purple"
                      variant="subtle"
                    >
                      {t}
                    </Badge>
                  ))}
                </HStack>
              ) : null}
            </FormControl>

            {/* Content + Preview */}
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

            {/* Save state */}
            <Flex justify="space-between" pt={2}>
              <HStack color="gray.500" fontSize="sm">
                <Text>
                  {saving === "saving"
                    ? "Saving…"
                    : saving === "saved"
                      ? "All changes saved."
                      : "Autosaves after you pause typing."}
                </Text>
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
