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
  Select,
  HStack,
  SimpleGrid,
  Card,
  CardHeader,
  CardBody,
} from "@chakra-ui/react";
import { useState } from "react";
import { api } from "~/utils/api";
import { Layout } from "~/components/layout";
import Link from "next/link";
import { useRouter } from "next/router";
import { createServerSideHelpers } from "@trpc/react-query/server";
import { appRouter } from "~/server/api/root";
import { createInnerTRPCContext } from "~/server/api/trpc";
import superjson from "superjson";
import type { GetServerSideProps } from "next";
import { GPU_DISPLAY_NAMES } from "~/constants/gpu";

export const getServerSideProps: GetServerSideProps = async (_context) => {
  const helpers = createServerSideHelpers({
    router: appRouter,
    ctx: createInnerTRPCContext({ session: null }),
    transformer: superjson,
  });

  // Prefetch ALL GPU types during server-side render
  const gpuTypes = ["all", "A100", "H100", "RTX4090"];
  await Promise.all(
    gpuTypes.map((gpuType) =>
      helpers.submissions.getBestSubmissionsByProblem.prefetch({ gpuType })
    )
  );

  return {
    props: {
      trpcState: helpers.dehydrate(),
    },
  };
};

const LeaderboardIndexPage: NextPage = () => {
  const router = useRouter();
  const [selectedGpu, setSelectedGpu] = useState<string>("all");

  const { data: leaderboardData, isLoading } =
    api.submissions.getBestSubmissionsByProblem.useQuery(
      { gpuType: selectedGpu },
      {
        placeholderData: (prev) => prev,
        refetchOnMount: false,
        refetchOnWindowFocus: false,
        staleTime: 300000,
      }
    );

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

  return (
    <Layout title={`Leaderboards: ${GPU_DISPLAY_NAMES[selectedGpu]}`}>
      <Box maxW="7xl" mx="auto" px={4} py={8}>
        <Flex direction="column" gap={6}>
          <HStack justify="space-between" align="center">
            <Heading size="lg">Leaderboards</Heading>
            <Select
              value={selectedGpu}
              onChange={(e) => setSelectedGpu(e.target.value)}
              w="200px"
              bg="whiteAlpha.50"
              color="white"
              borderColor="whiteAlpha.200"
              _hover={{ borderColor: "whiteAlpha.400" }}
              _focus={{ borderColor: "blue.500" }}
            >
              {Object.entries(GPU_DISPLAY_NAMES).map(([key, value]) => (
                <option key={key} value={key}>
                  {value}
                </option>
              ))}
            </Select>
          </HStack>

          <SimpleGrid columns={{ base: 1, md: 2, lg: 3 }} spacing={6}>
            {leaderboardData?.map((problem) => {
              const topSubmissions = problem.topSubmissions;

              return (
                <Card
                  key={problem.slug}
                  bg="gray.800"
                  borderColor="whiteAlpha.200"
                  borderWidth={1}
                  _hover={{ borderColor: "whiteAlpha.400" }}
                >
                  <CardHeader>
                    <ChakraLink
                      as={Link}
                      href={`/leaderboard/${problem.slug}${
                        selectedGpu !== "all" ? `?gpu=${selectedGpu}` : ""
                      }`}
                    >
                      <Heading
                        size="md"
                        color="white"
                        _hover={{ color: "blue.400" }}
                      >
                        {problem.title}
                      </Heading>
                    </ChakraLink>
                  </CardHeader>
                  <CardBody>
                    {topSubmissions.length === 0 ? (
                      <Text color="whiteAlpha.700">
                        No submissions yet
                        {selectedGpu !== "all"
                          ? ` for ${GPU_DISPLAY_NAMES[selectedGpu]}`
                          : ""}
                      </Text>
                    ) : (
                      <Table variant="unstyled" size="sm">
                        <Thead>
                          <Tr>
                            <Th pl={2} color="whiteAlpha.600">
                              Rank
                            </Th>
                            <Th color="whiteAlpha.600">User</Th>
                            <Th color="whiteAlpha.600">GPU</Th>
                            <Th isNumeric color="whiteAlpha.600">
                              GFLOPS
                            </Th>
                          </Tr>
                        </Thead>
                        <Tbody>
                          {topSubmissions.map((submission, index) => (
                            <Tr
                              key={submission.id}
                              onClick={() =>
                                router.push(`/submissions/${submission.id}`)
                              }
                              cursor="pointer"
                              _hover={{
                                bg: "gray.600",
                                transform: "scale(1.02)",
                                transition:
                                  "background-color 0.2s, transform 0.2s",
                              }}
                              px={4}
                              rounded="full"
                            >
                              <Td pl={2}>
                                <Badge
                                  colorScheme={
                                    index === 0
                                      ? "yellow"
                                      : index === 1
                                        ? "gray"
                                        : "blue"
                                  }
                                  variant="solid"
                                  fontSize="sm"
                                  px={2}
                                >
                                  #{index + 1}
                                </Badge>
                              </Td>
                              <Td color="white">
                                {submission.username ?? "Anonymous"}
                              </Td>
                              <Td>
                                <Badge
                                  colorScheme="purple"
                                  variant="subtle"
                                  fontSize="xs"
                                >
                                  {submission.gpuType}
                                </Badge>
                              </Td>
                              <Td isNumeric>
                                <Tooltip
                                  label={`Runtime: ${submission.runtime?.toFixed(
                                    2
                                  )} ms`}
                                >
                                  <Text color="green.300" fontWeight="semibold">
                                    {submission.gflops?.toFixed(2)}
                                  </Text>
                                </Tooltip>
                              </Td>
                            </Tr>
                          ))}
                        </Tbody>
                      </Table>
                    )}
                  </CardBody>
                </Card>
              );
            })}
          </SimpleGrid>
        </Flex>
      </Box>
    </Layout>
  );
};

export default LeaderboardIndexPage;
