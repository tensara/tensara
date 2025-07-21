import {
  Box,
  Flex,
  Heading,
  Spinner,
  Text,
  Alert,
  AlertIcon,
  VStack,
  Link,
  Badge,
  Button,
} from "@chakra-ui/react";
import { Layout } from "~/components/layout";
import { api } from "~/utils/api";
import NextLink from "next/link";
import { type Contest, ContestStatus } from "@prisma/client";
import { useSession } from "next-auth/react";

const getStatusBadgeColor = (status: ContestStatus) => {
  switch (status) {
    case "UPCOMING":
      return "blue";
    case "ACTIVE":
      return "green";
    case "FINISHED":
      return "gray";
  }
};

export default function ContestsPage() {
  const { data: session } = useSession();
  const isAdmin = session?.user?.role === "ADMIN";
  const {
    data: contests,
    isLoading,
    error,
  } = isAdmin
    ? api.contests.getAllAdmin.useQuery()
    : api.contests.getAll.useQuery();

  return (
    <Layout
      title="Contests | Tensara"
      ogTitle="Contests | Tensara"
      ogDescription="Join programming contests on Tensara."
    >
      <Box maxW="7xl" mx="auto" px={4} py={8}>
        <Flex
          justifyContent="space-between"
          alignItems="center"
          mb={6}
          direction={{ base: "column", sm: "row" }}
          gap={{ base: 4, sm: 0 }}
        >
          <Heading size="lg">Contests</Heading>
          {session?.user.role === "ADMIN" && (
            <Button as={NextLink} href="/contests/create" colorScheme="blue">
              Create Contest
            </Button>
          )}
        </Flex>

        {isLoading && (
          <Flex justify="center" align="center" h="400px">
            <Spinner size="xl" />
          </Flex>
        )}

        {error && (
          <Alert status="error">
            <AlertIcon />
            There was an error fetching the contests.
          </Alert>
        )}

        {contests && (
          <VStack spacing={4} align="stretch">
            {contests.map((contest: Contest) => (
              <Link
                as={NextLink}
                href={`/contests/${contest.id}`}
                key={contest.id}
                _hover={{ textDecoration: "none" }}
              >
                <Box
                  p={5}
                  shadow="md"
                  borderWidth="1px"
                  borderRadius="lg"
                  _hover={{ shadow: "lg" }}
                >
                  <Flex justify="space-between" align="center">
                    <Heading fontSize="xl">{contest.title}</Heading>
                    <Badge
                      colorScheme={getStatusBadgeColor(contest.status)}
                      p={2}
                      borderRadius="md"
                    >
                      {contest.status}
                    </Badge>
                  </Flex>
                  <Text mt={4}>
                    Starts: {new Date(contest.startTime).toLocaleString()}
                  </Text>
                </Box>
              </Link>
            ))}
          </VStack>
        )}
      </Box>
    </Layout>
  );
}
