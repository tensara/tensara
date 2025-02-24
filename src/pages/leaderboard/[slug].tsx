import { type NextPage } from "next";
import {
  Box,
  Table,
  Thead,
  Tbody,
  Tr,
  Th,
  Td,
  Text,
  Badge,
  Link as ChakraLink,
  Tooltip,
  Spinner,
  Flex,
  Heading,
} from "@chakra-ui/react";
import { api } from "~/utils/api";
import { Layout } from "~/components/layout";
import Link from "next/link";
import { useRouter } from "next/router";
import { formatDistanceToNow } from "date-fns";

interface LeaderboardEntry {
  id: string;
  gflops: number | null;
  runtime: number | null;
  createdAt: Date;
  status: string | null;
  user: {
    name: string | null;
  };
  problem: {
    title: string;
    slug: string;
  };
}

const LeaderboardPage: NextPage = () => {
  const router = useRouter();
  const { slug } = router.query;

  const { data: submissions, isLoading } =
    api.submissions.getAllSubmissions.useQuery();

  // Filter submissions for the current problem
  const problemSubmissions = submissions?.filter(
    (submission) => submission.problem.slug === slug
  );

  // Process submissions to get the best submission per user
  const getBestSubmissions = (submissions: LeaderboardEntry[] | undefined) => {
    if (!submissions) return [];

    const userBestMap = new Map<string, LeaderboardEntry>();

    submissions.forEach((submission) => {
      if (submission.status !== "ACCEPTED" || !submission.gflops) return;

      const userId = submission.user.name ?? "Anonymous";
      const currentBest = userBestMap.get(userId);

      if (!currentBest || submission.gflops > currentBest.gflops!) {
        userBestMap.set(userId, submission);
      }
    });

    return Array.from(userBestMap.values()).sort(
      (a, b) => (b.gflops ?? 0) - (a.gflops ?? 0)
    );
  };

  const leaderboardEntries = getBestSubmissions(problemSubmissions);

  if (isLoading) {
    return (
      <Layout title="Leaderboard">
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

  if (!submissions?.[0]?.problem) {
    return (
      <Layout title="Leaderboard">
        <Box maxW="7xl" mx="auto" px={4} py={8}>
          <Text>Problem not found</Text>
        </Box>
      </Layout>
    );
  }

  return (
    <Layout title={`Leaderboard - ${submissions[0]?.problem.title}`}>
      <Box maxW="7xl" mx="auto" px={4} py={8}>
        <Flex direction="column" gap={6}>
          <Heading size="lg">Leaderboard</Heading>
          <Heading size="md">{submissions[0]?.problem.title}</Heading>

          <Box overflowX="auto">
            <Table variant="unstyled">
              <Thead>
                <Tr>
                  <Th borderBottom="1px solid" borderColor="whiteAlpha.200">
                    Rank
                  </Th>
                  <Th borderBottom="1px solid" borderColor="whiteAlpha.200">
                    User
                  </Th>
                  <Th borderBottom="1px solid" borderColor="whiteAlpha.200">
                    Performance
                  </Th>
                  <Th borderBottom="1px solid" borderColor="whiteAlpha.200">
                    Submitted
                  </Th>
                </Tr>
              </Thead>
              <Tbody>
                {leaderboardEntries.map((entry, index) => (
                  <Tr
                    key={entry.id}
                    _hover={{ bg: "whiteAlpha.50" }}
                    transition="background-color 0.2s"
                  >
                    <Td borderBottom="1px solid" borderColor="whiteAlpha.100">
                      <Badge
                        colorScheme={index < 3 ? "yellow" : "gray"}
                        fontSize="md"
                        px={3}
                        py={1}
                        borderRadius="full"
                      >
                        #{index + 1}
                      </Badge>
                    </Td>
                    <Td borderBottom="1px solid" borderColor="whiteAlpha.100">
                      <Text fontWeight={index < 3 ? "bold" : "normal"}>
                        {entry.user.name ?? "Anonymous"}
                      </Text>
                    </Td>
                    <Td borderBottom="1px solid" borderColor="whiteAlpha.100">
                      <ChakraLink as={Link} href={`/submissions/${entry.id}`}>
                        <Tooltip
                          label={`Runtime: ${entry.runtime?.toFixed(2)} ms`}
                        >
                          <Text fontWeight="medium">
                            {entry.gflops?.toFixed(2)} GFLOPS
                          </Text>
                        </Tooltip>
                      </ChakraLink>
                    </Td>
                    <Td borderBottom="1px solid" borderColor="whiteAlpha.100">
                      <Tooltip
                        label={new Date(entry.createdAt).toLocaleString()}
                      >
                        <Text>
                          {formatDistanceToNow(new Date(entry.createdAt))} ago
                        </Text>
                      </Tooltip>
                    </Td>
                  </Tr>
                ))}
              </Tbody>
            </Table>
          </Box>
        </Flex>
      </Box>
    </Layout>
  );
};

export default LeaderboardPage;
