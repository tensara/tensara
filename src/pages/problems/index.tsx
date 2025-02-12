import { Box, Table, Thead, Tbody, Tr, Th, Td, Text } from "@chakra-ui/react";
import { Layout } from "~/components/layout";
import { api } from "~/utils/api";

export default function ProblemsPage() {
  const { data: problems, isLoading } = api.problems.getAll.useQuery();

  if (isLoading) {
    return (
      <Layout title="Problems">
        <Text color="white">Loading problems...</Text>
      </Layout>
    );
  }

  return (
    <Layout title="Problems">
      <Box overflowX="auto">
        <Table variant="simple">
          <Thead>
            <Tr>
              <Th color="gray.300" fontSize="md">
                Title
              </Th>
              <Th color="gray.300" fontSize="md">
                Difficulty
              </Th>
              <Th color="gray.300" fontSize="md">
                Author
              </Th>
            </Tr>
          </Thead>
          <Tbody>
            {problems?.map((problem) => (
              <Tr
                key={problem.id}
                _hover={{ bg: "whiteAlpha.100", cursor: "pointer" }}
                onClick={() =>
                  (window.location.href = `/problems/${problem.slug}`)
                }
              >
                <Td color="white">{problem.title}</Td>
                <Td color="white">{problem.difficulty}</Td>
                <Td color="white">{problem.author}</Td>
              </Tr>
            ))}
          </Tbody>
        </Table>
      </Box>
    </Layout>
  );
}
