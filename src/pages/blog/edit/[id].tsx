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
  Badge,
  type BadgeProps,
} from "@chakra-ui/react";
import { Layout } from "~/components/layout";
import { useSession, signIn } from "next-auth/react";
import { useRouter } from "next/router";
import { useEffect, useMemo, useRef, useState, useCallback } from "react";
import { api } from "~/utils/api";
import { FiArrowLeft, FiEye, FiEdit, FiImage, FiUpload } from "react-icons/fi";
import { MarkdownRenderer } from "~/components/blog";
import { markdownContentStyles } from "~/constants/blog";
import { env } from "~/env";
// --- utils ---
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
  const contentRef = useRef<HTMLTextAreaElement>(null);
  const imageInputRef = useRef<HTMLInputElement>(null);

  const postQ = api.blogpost.getById.useQuery(
    { id: id ?? "" },
    { enabled: !!id, refetchOnMount: true, refetchOnWindowFocus: false }
  );

  const updatePost = api.blogpost.update.useMutation();
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

  const hasHydrated = useRef(false);

  const [title, setTitle] = useState("");
  const [content, setContent] = useState("");
  const [tags, setTags] = useState<string[]>([]);
  const [tagsInput, setTagsInput] = useState("");
  const [showPreview, setShowPreview] = useState(false);
  const [isUploadingImage, setIsUploadingImage] = useState(false);
  const MAX_IMAGE_BYTES = 5 * 1024 * 1024; // 5MB
  const MAX_IMAGES_PER_POST = 20;
  const ALLOWED_IMAGE_MIME_TYPES = useMemo(
    () =>
      new Set([
        "image/jpeg",
        "image/png",
        "image/webp",
        "image/gif",
        "image/avif",
      ]),
    []
  );

  const countImagesInMarkdown = useCallback((markdown: string): number => {
    // Counts basic inline markdown images: ![alt](url "title")
    const regex = /!\[[^\]]*]\([^)\n]+\)/g;
    return markdown.match(regex)?.length ?? 0;
  }, []);

  const insertMarkdownAtCursor = useCallback((markdown: string) => {
    const el = contentRef.current;
    if (!el) {
      setContent((prev) => prev + markdown);
      return;
    }
    const start = el.selectionStart ?? 0;
    const end = el.selectionEnd ?? 0;
    setContent((prev) => prev.slice(0, start) + markdown + prev.slice(end));
    requestAnimationFrame(() => {
      el.focus();
      const pos = start + markdown.length;
      el.setSelectionRange(pos, pos);
    });
  }, []);

  const uploadImageToCloudinary = useCallback(async (file: File) => {
    const cloudName = env.NEXT_PUBLIC_CLOUDINARY_CLOUD_NAME;
    const uploadPreset = env.NEXT_PUBLIC_CLOUDINARY_UPLOAD_PRESET;
    if (!cloudName || !uploadPreset) {
      throw new Error(
        "Missing NEXT_PUBLIC_CLOUDINARY_CLOUD_NAME or NEXT_PUBLIC_CLOUDINARY_UPLOAD_PRESET (add them to .env.local)."
      );
    }

    const body = new FormData();
    body.append("file", file);
    body.append("upload_preset", uploadPreset);

    const res = await fetch(
      `https://api.cloudinary.com/v1_1/${cloudName}/image/upload`,
      { method: "POST", body }
    );

    const data: unknown = await res.json().catch(() => null);
    if (!res.ok) {
      const message =
        data &&
        typeof data === "object" &&
        "error" in data &&
        (data as Record<string, unknown>).error &&
        typeof (data as { error?: { message?: unknown } }).error?.message ===
          "string"
          ? (data as { error: { message: string } }).error.message
          : `Image upload failed (${res.status})`;
      throw new Error(message);
    }

    const record = data as Record<string, unknown> | null;
    const url =
      record && typeof record.secure_url === "string"
        ? record.secure_url
        : record && typeof record.url === "string"
          ? record.url
          : null;
    if (!url) {
      throw new Error("Image upload failed (unexpected response).");
    }
    return url;
  }, []);

  const uploadAndInsertImage = useCallback(
    async (file: File) => {
      if (!file.type.startsWith("image/")) {
        toast({
          title: "Not an image",
          description: "Please upload an image file.",
          status: "warning",
        });
        return;
      }
      if (!ALLOWED_IMAGE_MIME_TYPES.has(file.type)) {
        toast({
          title: "Unsupported image type",
          description: `Type ${file.type} is not allowed.`,
          status: "warning",
        });
        return;
      }
      if (file.size > MAX_IMAGE_BYTES) {
        toast({
          title: "Image too large",
          description: `Max size is ${Math.round(MAX_IMAGE_BYTES / 1024 / 1024)}MB.`,
          status: "warning",
        });
        return;
      }
      const currentImageCount = countImagesInMarkdown(content);
      if (currentImageCount >= MAX_IMAGES_PER_POST) {
        toast({
          title: "Too many images",
          description: `Max ${MAX_IMAGES_PER_POST} images per post.`,
          status: "warning",
        });
        return;
      }

      if (isUploadingImage) return;

      setIsUploadingImage(true);
      try {
        const url = await uploadImageToCloudinary(file);
        insertMarkdownAtCursor(`\n\n![](${url})\n\n`);
        toast({ title: "Image added", status: "success", duration: 2000 });
      } catch (err) {
        const message =
          err instanceof Error ? err.message : "Failed to upload image";
        toast({
          title: "Upload failed",
          description: message,
          status: "error",
        });
      } finally {
        setIsUploadingImage(false);
      }
    },
    [
      ALLOWED_IMAGE_MIME_TYPES,
      MAX_IMAGE_BYTES,
      MAX_IMAGES_PER_POST,
      content,
      countImagesInMarkdown,
      insertMarkdownAtCursor,
      isUploadingImage,
      toast,
      uploadImageToCloudinary,
    ]
  );

  const onSaveNow = useCallback(async () => {
    if (!id || !hasHydrated.current) return;
    if (!title.trim() || !content.trim()) {
      toast({
        title: "Title and content required",
        status: "warning",
      });
      return;
    }

    try {
      await updatePost.mutateAsync({
        id,
        title,
        content,
        tags,
      });
      void utils.blogpost.getById.invalidate({ id });
      toast({
        title: "Saved",
        status: "success",
        duration: 2000,
      });
    } catch (err) {
      const message = err instanceof Error ? err.message : "Failed to save";
      toast({
        title: "Save failed",
        description: message,
        status: "error",
      });
    }
  }, [id, title, content, tags, updatePost, toast, utils]);

  useEffect(() => {
    hasHydrated.current = false;
  }, [id]);

  useEffect(() => {
    if (!postQ.data || hasHydrated.current || !id) return;
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

    const initialTags: string[] =
      Array.isArray(p.tags) && p.tags.length
        ? (p.tags as unknown[])
            .map(tagToString)
            .filter(Boolean)
            .map((s) => s.toLowerCase())
        : [];

    const initialTitle = p.title ?? "";
    const initialContent = p.content ?? "";

    setTitle(initialTitle);
    setContent(initialContent);
    setTags(initialTags);
    setTagsInput(tagsToInput(initialTags));

    hasHydrated.current = true;
  }, [postQ.data, id]);

  // cmd/ctrl+s keyboard shortcut for manual save
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "s") {
        e.preventDefault();
        void onSaveNow();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onSaveNow]);

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

  // Button styles matching index.tsx
  const getGhostBtnStyles = () =>
    ({
      bg: "gray.800",
      color: "gray.100",
      borderColor: "gray.700",
      px: 4,
      _hover: {
        bg: "gray.700",
        borderColor: "gray.600",
      },
      _active: { bg: "green.700", borderColor: "green.600" },
      transition: "all 0.5s ease",
      rounded: "lg",
    }) as const;

  const solidBtn = {
    bg: "green.600",
    color: "white",
    px: 4,
    borderColor: "green.500",
    _hover: { bg: "green.500", borderColor: "green.400" },
    _active: { bg: "green.700", borderColor: "green.600" },
    rounded: "lg",
  } as const;

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

  const onPublish = async () => {
    if (!id) return;
    if (updatePost.isPending || publish.isPending) {
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

    try {
      // Save the content first before publishing
      await updatePost.mutateAsync({
        id,
        title,
        content,
        tags,
      });
      // Then publish
      await publish.mutateAsync({ id });
    } catch (err) {
      const message = err instanceof Error ? err.message : "Failed to publish";
      toast({
        title: "Publish failed",
        description: message,
        status: "error",
      });
    }
  };

  return (
    <Layout title="Edit Post">
      <Box minH="100vh">
        <Container maxW="7xl" mx="auto" py={8}>
          <Flex align="center" justify="space-between" mb={4}>
            <HStack>
              <Button
                variant="ghost"
                size="sm"
                leftIcon={<Icon as={FiArrowLeft} />}
                onClick={() => router.push("/blog")}
                _active={{ bg: "gray.700", color: "gray.100" }}
              >
                Back
              </Button>
              {statusBadge}
            </HStack>
            <HStack spacing={1.5} align="center">
              <Button
                size="sm"
                onClick={onSaveNow}
                isLoading={updatePost.isPending}
                {...getGhostBtnStyles()}
              >
                Save
              </Button>
              <Button
                size="sm"
                leftIcon={<Icon as={FiUpload} />}
                onClick={onPublish}
                isDisabled={
                  updatePost.isPending || !title.trim() || !content.trim()
                }
                isLoading={publish.isPending}
                {...solidBtn}
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
                bg="whiteAlpha.50"
                border="1px solid"
                borderColor="transparent"
                color="white"
                _hover={{ borderColor: "gray.600" }}
                _focus={{ borderColor: "blue.500", boxShadow: "none" }}
              />
            </FormControl>

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
                bg="whiteAlpha.50"
                border="1px solid"
                borderColor="transparent"
                color="white"
                _hover={{ borderColor: "gray.600" }}
                _focus={{ borderColor: "blue.500", boxShadow: "none" }}
              />
              {tags.length ? (
                <HStack spacing={1.5} mt={2} wrap="wrap">
                  {tags.map((t) => (
                    <Badge
                      key={t}
                      px={2}
                      py={0.5}
                      rounded="md"
                      colorScheme="green"
                      variant="subtle"
                    >
                      {t}
                    </Badge>
                  ))}
                </HStack>
              ) : null}
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
                <HStack spacing={1.5}>
                  {!showPreview ? (
                    <>
                      <Button
                        size="sm"
                        leftIcon={<Icon as={FiImage} />}
                        onClick={() => imageInputRef.current?.click()}
                        isLoading={isUploadingImage}
                        {...getGhostBtnStyles()}
                        px={3}
                      >
                        Image
                      </Button>
                      <input
                        type="file"
                        accept="image/*"
                        ref={imageInputRef}
                        style={{ display: "none" }}
                        onChange={(e) => {
                          const file = e.target.files?.[0];
                          e.target.value = "";
                          if (!file) return;
                          void uploadAndInsertImage(file);
                        }}
                      />
                    </>
                  ) : null}
                  <Button
                    size="sm"
                    leftIcon={<Icon as={FiEdit} />}
                    onClick={() => setShowPreview(false)}
                    {...(!showPreview ? solidBtn : getGhostBtnStyles())}
                    px={3}
                  >
                    Edit
                  </Button>
                  <Button
                    size="sm"
                    leftIcon={<Icon as={FiEye} />}
                    onClick={() => setShowPreview(true)}
                    {...(showPreview ? solidBtn : getGhostBtnStyles())}
                    px={3}
                  >
                    Preview
                  </Button>
                </HStack>
              </Flex>

              {!showPreview ? (
                <Textarea
                  ref={contentRef}
                  value={content}
                  onChange={(e) => setContent(e.target.value)}
                  onPaste={(e) => {
                    const file = Array.from(e.clipboardData.files).find((f) =>
                      f.type.startsWith("image/")
                    );
                    if (!file) return;
                    e.preventDefault();
                    void uploadAndInsertImage(file);
                  }}
                  onDragOver={(e) => {
                    e.preventDefault();
                  }}
                  onDrop={(e) => {
                    e.preventDefault();
                    const file = Array.from(e.dataTransfer.files).find((f) =>
                      f.type.startsWith("image/")
                    );
                    if (!file) return;
                    void uploadAndInsertImage(file);
                  }}
                  placeholder="# Your Story\n\nWrite your post in markdown…"
                  minH="420px"
                  fontSize="md"
                  lineHeight="1.6"
                  bg="whiteAlpha.50"
                  border="1px solid"
                  borderColor="transparent"
                  color="white"
                  _hover={{ borderColor: "gray.600" }}
                  _focus={{ borderColor: "blue.500", boxShadow: "none" }}
                />
              ) : (
                <Box
                  minH="420px"
                  border="1px solid"
                  borderColor="whiteAlpha.200"
                  rounded="md"
                  p={4}
                  sx={markdownContentStyles}
                >
                  {content ? (
                    <MarkdownRenderer content={content} />
                  ) : (
                    <Text color="gray.500" fontStyle="italic">
                      No content yet.
                    </Text>
                  )}
                </Box>
              )}
            </FormControl>
          </VStack>
        </Container>
      </Box>
    </Layout>
  );
}
