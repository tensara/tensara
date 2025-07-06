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
  Icon,
  ButtonGroup,
  Flex,
  Heading,
} from "@chakra-ui/react";
import { api } from "~/utils/api";
import NextLink from "next/link";
import { FiExternalLink } from "react-icons/fi";
import { useRouter } from "next/router";
import { useEffect, useState, useMemo } from "react";
import { formatContributionDate } from "~/utils/date";

type FilterStatus = "all" | "open" | "closed";

const ViewContributionsPage: NextPage = () => {
  const { data: session, status: sessionStatus } = useSession();
  const router = useRouter();
  const [filter, setFilter] = useState<FilterStatus>("all");

  const cardBg = useColorModeValue("white", "gray.800");
  const cardBorder = useColorModeValue("gray.200", "gray.700");
  const mutedColor = useColorModeValue("gray.600", "gray.400");

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

  const filteredContributions = useMemo(() => {
    if (!contributions) return [];
    if (filter === "all") return contributions;
    return contributions.filter((contrib) => {
      const status = getDisplayStatus(contrib.prStatus, contrib.prMergedAt);
      if (filter === "open") return status === "Open";
      if (filter === "closed")
        return status === "Closed" || status === "Merged";
      return false;
    });
  }, [contributions, filter]);

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
        <Flex justifyContent="space-between" alignItems="center" mb={6}>
          <Box>
            <Heading as="h1" size="lg" fontWeight="bold">
              Your Contributions
            </Heading>
            <Text mt={1} color={mutedColor} fontSize="lg">
              View your submitted problems and their status on GitHub.
            </Text>
          </Box>
          <ButtonGroup isAttached variant="outline" size="sm">
            <Button
              onClick={() => setFilter("all")}
              isActive={filter === "all"}
            >
              All
            </Button>
            <Button
              onClick={() => setFilter("open")}
              isActive={filter === "open"}
            >
              Open
            </Button>
            <Button
              onClick={() => setFilter("closed")}
              isActive={filter === "closed"}
            >
              Closed
            </Button>
          </ButtonGroup>
        </Flex>

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
                You haven&apos;t made any contributions yet.
              </Text>
              <NextLink href="/contribute/add" passHref>
                <Button as="a" colorScheme="blue">
                  Submit Your First Problem
                </Button>
              </NextLink>
            </Box>
          ) : (
            <>
              {filteredContributions.length > 0 ? (
                <Table variant="simple" size="md">
                  <Thead>
                    <Tr>
                      <Th>Contribution Title</Th>
                      <Th>Status</Th>
                      <Th>Created</Th>
                      <Th>Last Updated</Th>
                      <Th>GitHub PR</Th>
                    </Tr>
                  </Thead>
                  <Tbody>
                    {filteredContributions.map((contrib) => (
                      <Tr key={contrib.prUrl}>
                        <Td fontWeight="medium">
                          {contrib.prTitle.replace("Contribution: ", "")}
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
                            {getDisplayStatus(
                              contrib.prStatus,
                              contrib.prMergedAt
                            )}
                          </Badge>
                        </Td>
                        <Td>
                          {formatContributionDate(
                            new Date(contrib.prCreatedAt)
                          )}
                        </Td>
                        <Td>
                          {formatContributionDate(
                            new Date(contrib.prUpdatedAt)
                          )}
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
              ) : (
                <Box textAlign="center" py={10}>
                  <Text fontSize="xl">
                    No contributions match the filter &quot;{filter}&quot;.
                  </Text>
                </Box>
              )}
            </>
          )}
        </Box>
      </Box>
    </Layout>
  );
};

export default ViewContributionsPage;
