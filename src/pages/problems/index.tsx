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
} from "@chakra-ui/react";
import { Layout } from "~/components/layout";
import { api } from "~/utils/api";
import { useState } from "react";
import { SearchIcon } from "@chakra-ui/icons";

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

export default function ProblemsPage() {
  const { data: problems, isLoading } = api.problems.getAll.useQuery();
  const [searchQuery, setSearchQuery] = useState("");
  const [difficultyFilter, setDifficultyFilter] = useState("all");

  const filteredProblems = problems?.filter((problem) => {
    const matchesSearch = problem.title
      .toLowerCase()
      .includes(searchQuery.toLowerCase());
    const matchesDifficulty =
      difficultyFilter === "all" || problem.difficulty === difficultyFilter;
    return matchesSearch && matchesDifficulty;
  });

  if (isLoading) {
    return (
      <Layout title="Problems">
        <Text color="white">Loading problems...</Text>
      </Layout>
    );
  }

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
            <option value="Easy">Easy</option>
            <option value="Medium">Medium</option>
            <option value="Hard">Hard</option>
          </Select>
        </HStack>

        <Text color="gray.400" fontSize="sm">
          Showing {filteredProblems?.length} of {problems?.length} problems
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
                >
                  Title
                </Th>
                <Th
                  color="gray.300"
                  fontSize="md"
                  width="150px"
                  borderBottom="1px solid"
                  borderColor="gray.700"
                >
                  Difficulty
                </Th>
                <Th
                  color="gray.300"
                  fontSize="md"
                  width="200px"
                  display={{ base: "none", md: "table-cell" }}
                  borderBottom="1px solid"
                  borderColor="gray.700"
                >
                  Author
                </Th>
              </Tr>
            </Thead>
            <Tbody>
              {filteredProblems?.map((problem) => (
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
                  <Td color="white" fontWeight="medium" borderBottom="none">
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
                  <Td
                    color="gray.400"
                    display={{ base: "none", md: "table-cell" }}
                    borderBottom="none"
                  >
                    {problem.author}
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
