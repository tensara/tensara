import { type NextPage } from "next";
import { useSession } from "next-auth/react";
import { Layout } from "~/components/layout";
import {
  Box,
  Heading,
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
} from "@chakra-ui/react";
import PageHeader from "~/components/contribute/PageHeader";

const ViewContributionsPage: NextPage = () => {
  const { data: session } = useSession();
  const cardBg = useColorModeValue("white", "gray.800");
  const cardBorder = useColorModeValue("gray.200", "gray.700");

  // Mock data - will be replaced with real API call
  const contributions = [
    {
      id: "1",
      title: "Vector Addition",
      status: "Pending",
      date: "2025-05-15",
      prUrl: "#",
    },
    {
      id: "2",
      title: "Matrix Multiplication",
      status: "Approved",
      date: "2025-05-10",
      prUrl: "#",
    },
    {
      id: "3",
      title: "Convolution 2D",
      status: "Rejected",
      date: "2025-05-05",
      prUrl: "#",
    },
  ];

  const getStatusColor = (status: string) => {
    switch (status) {
      case "Approved":
        return "green";
      case "Rejected":
        return "red";
      default:
        return "yellow";
    }
  };

  return (
    <Layout>
      <Box maxW="6xl" mx="auto" px={4} py={8}>
        <PageHeader
          title="Your Contributions"
          description="View and manage your submitted problems"
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
          {contributions.length === 0 ? (
            <Box textAlign="center" py={10}>
              <Text fontSize="xl" mb={4}>
                You haven&apos;t submitted any problems yet
              </Text>
              <Button colorScheme="blue">Submit Your First Problem</Button>
            </Box>
          ) : (
            <Table variant="simple">
              <Thead>
                <Tr>
                  <Th>Problem</Th>
                  <Th>Status</Th>
                  <Th>Date Submitted</Th>
                  <Th>Actions</Th>
                </Tr>
              </Thead>
              <Tbody>
                {contributions.map((contribution) => (
                  <Tr key={contribution.id}>
                    <Td fontWeight="medium">{contribution.title}</Td>
                    <Td>
                      <Badge
                        colorScheme={getStatusColor(contribution.status)}
                        px={2}
                        py={1}
                        borderRadius="full"
                      >
                        {contribution.status}
                      </Badge>
                    </Td>
                    <Td>{contribution.date}</Td>
                    <Td>
                      <Button size="sm" mr={2}>
                        View
                      </Button>
                      <Button size="sm" colorScheme="blue">
                        Edit
                      </Button>
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
