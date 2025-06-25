import { type NextPage } from "next";
import { useSession } from "next-auth/react";
import { Layout } from "~/components/layout";
import {
  Box,
  Table,
  Thead,
  Tbody,
  Tr,
  Th,
  Td,
  Badge,
  Button,
  useColorModeValue,
  Text,
  Spinner,
  Alert,
  AlertIcon,
  AlertTitle,
  AlertDescription,
  Link as ChakraLink,
  VStack,
  Icon,
} from "@chakra-ui/react";
import PageHeader from "~/components/contribute/PageHeader";
import { api } from "~/utils/api";
import NextLink from "next/link";
import { FiExternalLink } from "react-icons/fi";
import { useRouter } from "next/router";
import { useEffect } from "react";

const ViewContributionsPage: NextPage = () => {
  const { data: session, status: sessionStatus } = useSession();
  const router = useRouter();

  const cardBg = useColorModeValue("white", "gray.800");
  const cardBorder = useColorModeValue("gray.200", "gray.700");

  const {
    data: contributions,
    isLoading,
    isError,
    error,
  } = api.contributions.getMyContributions.useQuery(undefined, {
    enabled: sessionStatus === "authenticated", // Only fetch if authenticated
  });

  useEffect(() => {
    if (sessionStatus === "unauthenticated") {
      void router.push("/api/auth/signin"); // Redirect to sign-in page
    }
  }, [sessionStatus, router]);

  const getStatusColorScheme = (
    prStatus: string | undefined,
    prMergedAt: Date | null | undefined
  ) => {
    if (prMergedAt) return "purple"; // Merged
    if (prStatus === "open") return "green"; // Open
    if (prStatus === "closed") return "red"; // Closed (and not merged)
    return "gray"; // Default
  };

  const getDisplayStatus = (
    prStatus: string | undefined,
    prMergedAt: Date | null | undefined
  ) => {
    if (prMergedAt) return "Merged";
    if (prStatus === "open") return "Open";
    if (prStatus === "closed") return "Closed";
    return "Unknown";
  };

  if (sessionStatus === "loading" || isLoading) {
    return (
      <Layout>
        <Box
          display="flex"
          justifyContent="center"
          alignItems="center"
          h="50vh"
        >
          <Spinner size="xl" />
        </Box>
      </Layout>
    );
  }

  if (sessionStatus === "unauthenticated") {
    return (
      <Layout>
        <Box textAlign="center" py={10}>
          <Text fontSize="xl">Please sign in to view your contributions.</Text>
        </Box>
      </Layout>
    );
  }

  if (isError) {
    return (
      <Layout>
        <Alert status="error" mt={8} mx="auto" maxW="lg">
          <AlertIcon />
          <AlertTitle>Error fetching contributions!</AlertTitle>
          <AlertDescription>{error?.message}</AlertDescription>
        </Alert>
      </Layout>
    );
  }

  return (
    <Layout>
      <Box maxW="6xl" mx="auto" px={4} py={8}>
        <PageHeader
          title="Your Contributions"
          description="View your submitted problems and their status on GitHub."
        />

        <Box
          bg={cardBg}
          borderWidth="1px"
          borderColor={cardBorder}
          borderRadius="xl"
          p={6}
          boxShadow="md"
          overflowX="auto"
        >
          {!contributions || contributions.length === 0 ? (
            <Box textAlign="center" py={10}>
              <Text fontSize="xl" mb={4}>
                You haven't made any contributions yet.
              </Text>
              <NextLink href="/contribute/add" passHref>
                <Button as="a" colorScheme="blue">
                  Submit Your First Problem
                </Button>
              </NextLink>
            </Box>
          ) : (
            <Table variant="simple" size="md">
              <Thead>
                <Tr>
                  <Th>Contribution Title</Th>
                  <Th>Problem Link</Th>
                  <Th>Status</Th>
                  <Th>Created</Th>
                  <Th>Last Updated</Th>
                  <Th>GitHub PR</Th>
                </Tr>
              </Thead>
              <Tbody>
                {contributions.map((contrib) => (
                  <Tr key={contrib.prUrl}>
                    <Td fontWeight="medium">{contrib.prTitle}</Td>
                    <Td>
                      {contrib.problemSlug && contrib.problemTitle ? (
                        <NextLink
                          href={`/problems/${contrib.problemSlug}`}
                          passHref
                        >
                          <ChakraLink color="blue.500" isExternal={false}>
                            {contrib.problemTitle}
                          </ChakraLink>
                        </NextLink>
                      ) : (
                        <Text as="em" color="gray.500">
                          N/A
                        </Text>
                      )}
                    </Td>
                    <Td>
                      <Badge
                        colorScheme={getStatusColorScheme(
                          contrib.prStatus,
                          contrib.prMergedAt
                        )}
                        px={2}
                        py={1}
                        borderRadius="full"
                        textTransform="capitalize"
                      >
                        {getDisplayStatus(contrib.prStatus, contrib.prMergedAt)}
                      </Badge>
                    </Td>
                    <Td>
                      {new Date(contrib.prCreatedAt).toLocaleDateString()}
                    </Td>
                    <Td>
                      {new Date(contrib.prUpdatedAt).toLocaleDateString()}
                    </Td>
                    <Td>
                      <ChakraLink href={contrib.prUrl} isExternal>
                        <Button
                          size="sm"
                          variant="outline"
                          rightIcon={<Icon as={FiExternalLink} />}
                        >
                          View PR
                        </Button>
                      </ChakraLink>
                    </Td>
                  </Tr>
                ))}
              </Tbody>
            </Table>
          )}
        </Box>
      </Box>
    </Layout>
  );
};

export default ViewContributionsPage;
