import { useRouter } from "next/router";
import { type inferRouterOutputs } from "@trpc/server";
import { type AppRouter } from "~/server/api/root";
import { api } from "~/utils/api";
import {
  Box,
  Button,
  Heading,
  Text,
  VStack,
  Spinner,
  Alert,
  AlertIcon,
  HStack,
  Tag,
  Flex,
  Badge,
  List,
  ListItem,
} from "@chakra-ui/react";
import { Select, Switch } from "@chakra-ui/react";
import NextLink from "next/link";
import { useSession } from "next-auth/react";
import { Layout } from "~/components/layout";
import { ContestStatus } from "@prisma/client";

type RouterOutput = inferRouterOutputs<AppRouter>;
type ContestWithProblems = RouterOutput["contests"]["getById"];
type ContestProblem = NonNullable<ContestWithProblems>["problems"][number];

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

const ContestPage = () => {
  const router = useRouter();
  const { id } = router.query;

  const contestId = typeof id === "string" ? id : "";
  const { data: session } = useSession();

  const {
    data: contest,
    isLoading,
    error,
  } = api.contests.getById.useQuery(
    { id: contestId },
    { enabled: !!contestId }
  );

  const registerMutation = api.contests.register.useMutation({
    onSuccess: () => {
      // Optionally refetch contest data to show updated registration status
      // utils.contests.getById.invalidate({ id: contestId });
    },
  });

  const handleRegister = () => {
    registerMutation.mutate({ contestId });
  };

  if (isLoading) {
    return (
      <Layout title="Loading Contest | Tensara">
        <Flex justify="center" align="center" h="400px">
          <Spinner size="xl" />
        </Flex>
      </Layout>
    );
  }

  if (error) {
    return (
      <Layout title="Error | Tensara">
        <Box display="flex" justifyContent="center" py={8}>
          <Alert status="error">
            <AlertIcon />
            Error loading contest: {error.message}
          </Alert>
        </Box>
      </Layout>
    );
  }

  if (!contest) {
    return (
      <Layout title="Not Found | Tensara">
        <Box display="flex" justifyContent="center" py={8}>
          <Alert status="warning">
            <AlertIcon />
            Contest not found.
          </Alert>
        </Box>
      </Layout>
    );
  }

  return (
    <Layout
      title={`${contest.title} | Tensara`}
      ogTitle={`${contest.title} | Tensara`}
    >
      <Box maxW="7xl" mx="auto" px={4} py={8}>
        <Flex
          justifyContent="space-between"
          alignItems="center"
          mb={6}
          direction={{ base: "column", sm: "row" }}
          gap={{ base: 4, sm: 0 }}
        >
          <Heading size="lg">{contest.title}</Heading>
          <Badge
            colorScheme={getStatusBadgeColor(contest.status)}
            p={2}
            borderRadius="md"
          >
            {contest.status}
          </Badge>
        </Flex>

        <Box p={5} shadow="md" borderWidth="1px" borderRadius="lg">
          <VStack spacing={8} align="stretch">
            <Box>
              <Text mt={2} color="gray.500">
                {contest.description ?? "No description available."}
              </Text>
            </Box>

            <Box>
              <Text>
                <strong>Starts:</strong>{" "}
                {new Date(contest.startTime).toLocaleString()}
              </Text>
              <Text>
                <strong>Ends:</strong>{" "}
                {new Date(contest.endTime).toLocaleString()}
              </Text>
            </Box>

            <Box>
              <Button
                colorScheme="teal"
                onClick={handleRegister}
                isLoading={registerMutation.isPending}
              >
                Register for Contest
              </Button>
            </Box>

            <Box>
              <Heading as="h2" size="lg" mb={4}>
                Problems
              </Heading>
              <VStack spacing={4} align="stretch">
                {contest.problems.map((contestProblem: ContestProblem) => (
                  <NextLink
                    href={`/problems/${contestProblem.problem.slug}`}
                    passHref
                    key={contestProblem.problem.id}
                  >
                    <Box
                      as="a"
                      p={5}
                      shadow="md"
                      borderWidth="1px"
                      borderRadius="lg"
                      _hover={{ shadow: "lg", textDecoration: "none" }}
                    >
                      <Heading fontSize="xl">
                        {contestProblem.problem.title}
                      </Heading>
                    </Box>
                  </NextLink>
                ))}
              </VStack>
            </Box>
          </VStack>
        </Box>
        {session?.user.role === "ADMIN" && (
          <Box mt={10} p={5} shadow="md" borderWidth="1px" borderRadius="lg">
            <AdminProblemManager contestId={contestId} />
          </Box>
        )}
      </Box>
    </Layout>
  );
};

const AdminProblemManager = ({ contestId }: { contestId: string }) => {
  const utils = api.useUtils();
  const { data: problems } = api.problems.getAll.useQuery();
  const { data: contest } = api.contests.getById.useQuery({ id: contestId });

  const addProblemMutation = api.contests.addProblemToContest.useMutation({
    onSuccess: () => {
      utils.contests.getById.invalidate({ id: contestId });
    },
  });

  const removeProblemMutation =
    api.contests.removeProblemFromContest.useMutation({
      onSuccess: () => {
        utils.contests.getById.invalidate({ id: contestId });
      },
    });

  const updateVisibilityMutation =
    api.problems.updateProblemVisibility.useMutation({
      onSuccess: () => {
        utils.contests.getById.invalidate({ id: contestId });
      },
    });

  const handleAddProblem = (problemId: string) => {
    addProblemMutation.mutate({ contestId, problemId });
  };

  const handleRemoveProblem = (problemId: string) => {
    removeProblemMutation.mutate({ contestId, problemId });
  };

  const handleVisibilityChange = (
    problemId: string,
    visibility: "PUBLIC" | "PRIVATE"
  ) => {
    updateVisibilityMutation.mutate({ problemId, visibility });
  };

  return (
    <Box mt={10} p={5} borderWidth="1px" borderRadius="lg">
      <Heading as="h3" size="lg" mb={4}>
        Manage Problems
      </Heading>
      <VStack spacing={4} align="stretch">
        <Box>
          <Heading as="h4" size="md" mb={2}>
            Add Problem
          </Heading>
          <Select
            placeholder="Select a problem to add"
            onChange={(e) => handleAddProblem(e.target.value)}
          >
            {problems?.map((problem) => (
              <option key={problem.id} value={problem.id}>
                {problem.title}
              </option>
            ))}
          </Select>
        </Box>
        <Box>
          <Heading as="h4" size="md" mb={2}>
            Current Problems
          </Heading>
          <List spacing={3}>
            {contest?.problems.map((contestProblem) => (
              <ListItem key={contestProblem.problem.id}>
                <HStack justifyContent="space-between">
                  <Text>{contestProblem.problem.title}</Text>
                  <HStack>
                    <Switch
                      isChecked={
                        (contestProblem.problem as any).visibility === "PUBLIC"
                      }
                      onChange={() =>
                        handleVisibilityChange(
                          contestProblem.problem.id,
                          (contestProblem.problem as any).visibility ===
                            "PUBLIC"
                            ? "PRIVATE"
                            : "PUBLIC"
                        )
                      }
                    />
                    <Button
                      size="sm"
                      colorScheme="red"
                      onClick={() =>
                        handleRemoveProblem(contestProblem.problem.id)
                      }
                    >
                      Remove
                    </Button>
                  </HStack>
                </HStack>
              </ListItem>
            ))}
          </List>
        </Box>
      </VStack>
    </Box>
  );
};

export default ContestPage;
