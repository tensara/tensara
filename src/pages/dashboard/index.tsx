import { Box, Text, VStack } from "@chakra-ui/react";
import { Layout } from "~/components/layout";
import { useSession } from "next-auth/react";
import { useRouter } from "next/router";
import { useEffect } from "react";

export default function DashboardPage() {
  const { data: session, status } = useSession();
  const router = useRouter();

  useEffect(() => {
    if (status === "unauthenticated") {
      void router.push("/");
    }
  }, [status, router]);

  if (status === "loading") {
    return null;
  }

  return (
    <Layout title="Dashboard">
      <VStack align="stretch" spacing={6}>
        <Text fontSize="2xl" fontWeight="bold">
          Welcome back, {session?.user?.name}!
        </Text>

        <Box bg="whiteAlpha.100" p={6} borderRadius="xl">
          <Text>Your dashboard content will go here...</Text>
        </Box>
      </VStack>
    </Layout>
  );
}
