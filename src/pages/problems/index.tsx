import {
  Box,
  Table,
  Thead,
  Tbody,
  Tr,
  Th,
  Td,
  Text,
  Input,
  Select,
  HStack,
  VStack,
  Badge,
  InputGroup,
  InputLeftElement,
  Spinner,
} from "@chakra-ui/react";
import { Layout } from "~/components/layout";
import { api } from "~/utils/api";
import { useState } from "react";
import { SearchIcon } from "@chakra-ui/icons";
import { TriangleDownIcon, TriangleUpIcon } from "@chakra-ui/icons";
import { createServerSideHelpers } from "@trpc/react-query/server";
import { appRouter } from "~/server/api/root";
import { createInnerTRPCContext } from "~/server/api/trpc";
import superjson from "superjson";
import type { GetServerSideProps } from "next";

type SortField = "title" | "difficulty" | "submissionCount";
type SortDirection = "asc" | "desc";

export const getDifficultyColor = (difficulty: string) => {
  switch (difficulty.toLowerCase()) {
    case "easy":
      return "green";
    case "medium":
      return "yellow";
    case "hard":
      return "red";
    default:
      return "gray";
  }
};

const getDifficultyValue = (difficulty: string) => {
  switch (difficulty.toLowerCase()) {
    case "easy":
      return 1;
    case "medium":
      return 2;
    case "hard":
      return 3;
    default:
      return 0;
  }
};

export const getServerSideProps: GetServerSideProps = async () => {
  const helpers = createServerSideHelpers({
    router: appRouter,
    ctx: createInnerTRPCContext({ session: null }),
    transformer: superjson,
  });

  await helpers.problems.getAll.prefetch();

  return {
    props: {
      trpcState: helpers.dehydrate(),
    },
  };
};

export default function ProblemsPage() {
  const { data: problems = [], isLoading } = api.problems.getAll.useQuery(
    undefined,
    {
      refetchOnMount: false,
      refetchOnWindowFocus: false,
    }
  );
  const [searchQuery, setSearchQuery] = useState("");
  const [difficultyFilter, setDifficultyFilter] = useState("all");
  const [sortField, setSortField] = useState<SortField>("title");
  const [sortDirection, setSortDirection] = useState<SortDirection>("asc");

  const handleSort = (field: SortField) => {
    if (sortField === field) {
      setSortDirection(sortDirection === "asc" ? "desc" : "asc");
    } else {
      setSortField(field);
      setSortDirection("asc");
    }
  };

  const filteredAndSortedProblems = problems
    ?.filter((problem) => {
      const matchesSearch = problem.title
        .toLowerCase()
        .includes(searchQuery.toLowerCase());
      const matchesDifficulty =
        difficultyFilter === "all" ||
        problem.difficulty.toLowerCase() === difficultyFilter.toLowerCase();
      return matchesSearch && matchesDifficulty;
    })
    .sort((a, b) => {
      const multiplier = sortDirection === "asc" ? 1 : -1;

      switch (sortField) {
        case "title":
          return multiplier * a.title.localeCompare(b.title);
        case "difficulty":
          return (
            multiplier *
            (getDifficultyValue(a.difficulty) -
              getDifficultyValue(b.difficulty))
          );
        case "submissionCount":
          return multiplier * (a.submissionCount - b.submissionCount);
        default:
          return 0;
      }
    });

  if (isLoading) {
    return (
      <Layout title="Problems">
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

  const SortIcon = ({ field }: { field: SortField }) => {
    if (sortField !== field) return null;
    return sortDirection === "asc" ? (
      <TriangleUpIcon ml={1} w={3} h={3} />
    ) : (
      <TriangleDownIcon ml={1} w={3} h={3} />
    );
  };

  return (
    <Layout title="Problems">
      <VStack spacing={6} align="stretch" w="full">
        <HStack spacing={4} w="full">
          <InputGroup maxW="400px">
            <InputLeftElement pointerEvents="none">
              <SearchIcon color="gray.300" />
            </InputLeftElement>
            <Input
              placeholder="Search problems..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              bg="whiteAlpha.50"
              border="1px solid"
              borderColor="gray.700"
              _hover={{ borderColor: "gray.600" }}
              _focus={{ borderColor: "blue.500", boxShadow: "none" }}
              color="white"
            />
          </InputGroup>
          <Select
            value={difficultyFilter}
            onChange={(e) => setDifficultyFilter(e.target.value)}
            maxW="200px"
            bg="whiteAlpha.50"
            border="1px solid"
            borderColor="gray.700"
            _hover={{ borderColor: "gray.600" }}
            _focus={{ borderColor: "blue.500", boxShadow: "none" }}
            color="white"
          >
            <option value="all">All Difficulties</option>
            <option value="easy">Easy</option>
            <option value="medium">Medium</option>
            <option value="hard">Hard</option>
          </Select>
        </HStack>

        <Text color="gray.400" fontSize="sm">
          Showing {filteredAndSortedProblems?.length} of {problems?.length}{" "}
          problems
        </Text>

        <Box
          overflowX="auto"
          borderRadius="xl"
          bg="whiteAlpha.50"
          border="1px solid"
          borderColor="gray.700"
        >
          <Table variant="simple">
            <Thead bg="gray.800">
              <Tr>
                <Th
                  color="gray.300"
                  fontSize="md"
                  py={4}
                  borderBottom="1px solid"
                  borderColor="gray.700"
                  cursor="pointer"
                  onClick={() => handleSort("title")}
                  _hover={{ color: "white" }}
                >
                  Title
                  <SortIcon field="title" />
                </Th>
                <Th
                  color="gray.300"
                  fontSize="md"
                  width="150px"
                  borderBottom="1px solid"
                  borderColor="gray.700"
                  cursor="pointer"
                  onClick={() => handleSort("difficulty")}
                  _hover={{ color: "white" }}
                >
                  Difficulty
                  <SortIcon field="difficulty" />
                </Th>
                <Th
                  color="gray.300"
                  fontSize="md"
                  width="200px"
                  display={{ base: "none", md: "table-cell" }}
                  borderBottom="1px solid"
                  borderColor="gray.700"
                  cursor="pointer"
                  onClick={() => handleSort("submissionCount")}
                  _hover={{ color: "white" }}
                >
                  Submissions
                  <SortIcon field="submissionCount" />
                </Th>
              </Tr>
            </Thead>
            <Tbody>
              {filteredAndSortedProblems?.map((problem) => (
                <Tr
                  key={problem.id}
                  _hover={{ bg: "gray.700", transform: "translateY(-1px)" }}
                  transition="all 0.2s"
                  cursor="pointer"
                  onClick={() =>
                    (window.location.href = `/problems/${problem.slug}`)
                  }
                  borderBottom="1px solid"
                  borderColor="gray.800"
                >
                  <Td color="white" borderBottom="none">
                    {problem.title}
                  </Td>
                  <Td borderBottom="none">
                    <Badge
                      colorScheme={getDifficultyColor(problem.difficulty)}
                      px={2}
                      py={1}
                      borderRadius="full"
                    >
                      {problem.difficulty}
                    </Badge>
                  </Td>
                  <Td borderBottom="none">
                    <Text color="gray.400" fontSize="sm">
                      {problem.submissionCount}
                    </Text>
                  </Td>
                </Tr>
              ))}
            </Tbody>
          </Table>
        </Box>
      </VStack>
    </Layout>
  );
}
