// src/pages/blog/create.tsx
import { useEffect } from "react";
import {
  Box,
  Container,
  Heading,
  Text,
  Button,
  VStack,
} from "@chakra-ui/react";
import { Layout } from "~/components/layout";
import { useSession, signIn } from "next-auth/react";
import { useRouter } from "next/router";
import { api } from "~/utils/api";

export default function CreateDraftRedirect() {
  const { data: session, status } = useSession();
  const router = useRouter();
  const utils = api.useContext();

  const createDraft = api.blogpost.createDraft.useMutation({
    onSuccess: async (post) => {
      await utils.blogpost.listMine.invalidate();
      router.replace(`/blog/edit/${post.id}`);
    },
  });

  useEffect(() => {
    if (status === "unauthenticated") {
      void signIn();
      return;
    }
    if (status === "authenticated" && !createDraft.isPending) {
      createDraft.mutate({ title: "Untitled draft" });
    }
  }, [status]); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <Layout title="New Draft">
      <Box bg="gray.900" minH="100vh">
        <Container maxW="600px" py={16}>
          <VStack spacing={4} align="stretch">
            <Heading as="h1" size="lg" color="white">
              Creating your draft…
            </Heading>
            <Text color="gray.400">
              We’ll take you to the editor in a moment. If nothing happens, you
              can try again.
            </Text>
            {status === "authenticated" && (
              <Button
                onClick={() => createDraft.mutate({ title: "Untitled draft" })}
                isLoading={createDraft.isPending}
                colorScheme="blue"
                alignSelf="start"
              >
                Create draft now
              </Button>
            )}
          </VStack>
        </Container>
      </Box>
    </Layout>
  );
}
