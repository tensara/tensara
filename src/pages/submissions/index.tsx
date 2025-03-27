import { useRouter } from "next/router";
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
  Select,
  HStack,
  Badge,
  Link as ChakraLink,
  Input,
  IconButton,
  Tooltip,
  Spinner,
} from "@chakra-ui/react";
import { useState } from "react";
import { api } from "~/utils/api";
import { Layout } from "~/components/layout";
import Link from "next/link";
import { ChevronUpIcon, ChevronDownIcon } from "@chakra-ui/icons";
import { formatDistanceToNow } from "date-fns";
import type { Submission } from "@prisma/client";
import { createServerSideHelpers } from "@trpc/react-query/server";
import { appRouter } from "~/server/api/root";
import { createInnerTRPCContext } from "~/server/api/trpc";
import superjson from "superjson";
import type { GetServerSideProps } from "next";
import { auth } from "~/server/auth";
import { GPU_DISPLAY_NAMES } from "~/constants/gpu";
import { LANGUAGE_DISPLAY_NAMES } from "~/constants/language";

type SortField = "createdAt" | "status" | "problem" | "performance" | "gpuType" | "language";
type SortOrder = "asc" | "desc";

interface SubmissionWithProblem extends Submission {
  problem: {
    title: string;
    slug: string;
  };
  user: {
    username: string | null;
  };
}

export const getServerSideProps: GetServerSideProps = async (context) => {
  const session = await auth(context.req, context.res);

  // Redirect to login if not authenticated
  if (!session) {
    return {
      redirect: {
        destination: "/api/auth/signin",
        permanent: false,
      },
    };
  }

  const helpers = createServerSideHelpers({
    router: appRouter,
    ctx: createInnerTRPCContext({ session }),
    transformer: superjson,
  });

  try {
    await helpers.submissions.getAllUserSubmissions.prefetch();

    return {
      props: {
        trpcState: helpers.dehydrate(),
      },
    };
  } catch (e) {
    console.error(e);
    return { notFound: true };
  }
};

const SubmissionsPage: NextPage = () => {
  const [statusFilter, setStatusFilter] = useState<string>("all");
  const [gpuFilter, setGpuFilter] = useState<string>("all");
  const [languageFilter, setLanguageFilter] = useState<string>("all");
  const [searchQuery, setSearchQuery] = useState("");
  const [sortField, setSortField] = useState<SortField>("createdAt");
  const [sortOrder, setSortOrder] = useState<SortOrder>("desc");
  const router = useRouter();

  const { data: submissions, isLoading } =
    api.submissions.getAllUserSubmissions.useQuery();

  const getStatusColor = (status: string | null) => {
    switch (status) {
      case "ACCEPTED":
        return "green";
      case "WRONG_ANSWER":
      case "ERROR":
        return "red";
      case "CHECKING":
      case "BENCHMARKING":
        return "blue";
      default:
        return "gray";
    }
  };

  const formatStatus = (status: string | null) => {
    switch (status) {
      case "ACCEPTED":
        return "Accepted";
      case "WRONG_ANSWER":
        return "Wrong Answer";
      case "ERROR":
        return "Error";
      case "CHECKING":
        return "Checking";
      case "BENCHMARKING":
        return "Benchmarking";
      default:
        return status ?? "Unknown";
    }
  };

  const handleSort = (field: SortField) => {
    if (sortField === field) {
      setSortOrder(sortOrder === "asc" ? "desc" : "asc");
    } else {
      setSortField(field);
      setSortOrder("desc");
    }
  };

  const filteredAndSortedSubmissions = submissions
    ?.filter((submission: SubmissionWithProblem) => {
      const matchesStatus =
        statusFilter === "all" || submission.status === statusFilter;
      const matchesGpu =
        gpuFilter === "all" || submission.gpuType === gpuFilter;
      const matchesLanguage =
        languageFilter === "all" || submission.language === languageFilter;
      const matchesSearch =
        searchQuery === "" ||
        submission.problem.title
          .toLowerCase()
          .includes(searchQuery.toLowerCase());
      return matchesStatus && matchesGpu && matchesLanguage && matchesSearch;
    })
    .sort((a: SubmissionWithProblem, b: SubmissionWithProblem) => {
      const order = sortOrder === "asc" ? 1 : -1;
      switch (sortField) {
        case "createdAt":
          return (
            (new Date(a.createdAt).getTime() -
              new Date(b.createdAt).getTime()) *
            order
          );
        case "status":
          return (a.status ?? "").localeCompare(b.status ?? "") * order;
        case "problem":
          return a.problem.title.localeCompare(b.problem.title) * order;
        case "performance":
          const aPerf = a.gflops ?? 0;
          const bPerf = b.gflops ?? 0;
          return (aPerf - bPerf) * order;
        case "gpuType":
          return (a.gpuType ?? "").localeCompare(b.gpuType ?? "") * order;
        case "language":
          return a.language.localeCompare(b.language) * order;
        default:
          return 0;
      }
    });

  if (isLoading) {
    return (
      <Layout title="Submissions">
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
    <Layout title="My Submissions">
      <Box maxW="7xl" mx="auto" px={4} py={8}>
        <Text fontSize="2xl" fontWeight="bold" mb={6}>
          My Submissions
        </Text>

        {/* Filters */}
        <HStack spacing={4} mb={6}>
          <Select
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value)}
            w="250px"
            bg="whiteAlpha.50"
            borderColor="transparent"
            _hover={{ borderColor: "gray.600" }}
            _focus={{ borderColor: "gray.500" }}
          >
            <option value="all">All Statuses</option>
            <option value="ACCEPTED">Accepted</option>
            <option value="WRONG_ANSWER">Wrong Answer</option>
            <option value="ERROR">Error</option>
            <option value="CHECKING">Checking</option>
            <option value="BENCHMARKING">Benchmarking</option>
          </Select>

          <Select
            value={gpuFilter}
            onChange={(e) => setGpuFilter(e.target.value)}
            w="250px"
            bg="whiteAlpha.50"
            borderColor="transparent"
            _hover={{ borderColor: "gray.600" }}
            _focus={{ borderColor: "gray.500" }}
          >
            {
              Object.entries(GPU_DISPLAY_NAMES)
                .map(([key, value]) => (
                  <option key={key} value={key}>{value}</option>
                ))
            }
          </Select>
          <Select
            value={languageFilter}
            onChange={(e) => setLanguageFilter(e.target.value)}
            w="275px"
            bg="whiteAlpha.50"
            borderColor="transparent"
            _hover={{ borderColor: "gray.600" }}
            _focus={{ borderColor: "gray.500" }}
          >
            {
              Object.entries(LANGUAGE_DISPLAY_NAMES)
                .map(([key, value]) => (
                  <option key={key} value={key}>{value}</option>
                ))
            }
          </Select>

          <Input
            placeholder="Search by problem name..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            bg="whiteAlpha.50"
            _focus={{ borderColor: "gray.500" }}
          />
        </HStack>

        {/* Submissions Table */}
        <Box overflowX="auto">
          <Table variant="unstyled">
            <Thead>
              <Tr>
                <Th borderBottom="1px solid" borderColor="whiteAlpha.200">
                  <HStack
                    spacing={2}
                    cursor="pointer"
                    onClick={() => handleSort("problem")}
                  >
                    <Text>Problem</Text>
                    {sortField === "problem" && (
                      <IconButton
                        aria-label={`Sort ${
                          sortOrder === "asc" ? "descending" : "ascending"
                        }`}
                        icon={
                          sortOrder === "asc" ? (
                            <ChevronUpIcon />
                          ) : (
                            <ChevronDownIcon />
                          )
                        }
                        size="xs"
                        variant="ghost"
                      />
                    )}
                  </HStack>
                </Th>
                <Th borderBottom="1px solid" borderColor="whiteAlpha.200">
                  <HStack
                    spacing={2}
                    cursor="pointer"
                    onClick={() => handleSort("status")}
                  >
                    <Text>Status</Text>
                    {sortField === "status" && (
                      <IconButton
                        aria-label={`Sort ${
                          sortOrder === "asc" ? "descending" : "ascending"
                        }`}
                        icon={
                          sortOrder === "asc" ? (
                            <ChevronUpIcon />
                          ) : (
                            <ChevronDownIcon />
                          )
                        }
                        size="xs"
                        variant="ghost"
                      />
                    )}
                  </HStack>
                </Th>
                <Th borderBottom="1px solid" borderColor="whiteAlpha.200">
                  <HStack
                    spacing={2}
                    cursor="pointer"
                    onClick={() => handleSort("performance")}
                  >
                    <Text>Performance</Text>
                    {sortField === "performance" && (
                      <IconButton
                        aria-label={`Sort ${
                          sortOrder === "asc" ? "descending" : "ascending"
                        }`}
                        icon={
                          sortOrder === "asc" ? (
                            <ChevronUpIcon />
                          ) : (
                            <ChevronDownIcon />
                          )
                        }
                        size="xs"
                        variant="ghost"
                      />
                    )}
                  </HStack>
                </Th>
                <Th borderBottom="1px solid" borderColor="whiteAlpha.200">
                  <HStack
                    spacing={2}
                    cursor="pointer"
                    onClick={() => handleSort("gpuType")}
                  >
                    <Text>GPU</Text>
                    {sortField === "gpuType" && (
                      <IconButton
                        aria-label={`Sort ${
                          sortOrder === "asc" ? "descending" : "ascending"
                        }`}
                        icon={
                          sortOrder === "asc" ? (
                            <ChevronUpIcon />
                          ) : (
                            <ChevronDownIcon />
                          )
                        }
                        size="xs"
                        variant="ghost"
                      />
                    )}
                  </HStack>
                </Th>
                <Th borderBottom="1px solid" borderColor="whiteAlpha.200">
                  <HStack
                    spacing={2}
                    cursor="pointer"
                    onClick={() => handleSort("language")}
                  >
                    <Text>Language</Text>
                    {sortField === "language" && (
                      <IconButton
                        aria-label={`Sort ${
                          sortOrder === "asc" ? "descending" : "ascending"
                        }`}
                        icon={
                          sortOrder === "asc" ? (
                            <ChevronUpIcon />
                          ) : (
                            <ChevronDownIcon />
                          )
                        }
                        size="xs"
                        variant="ghost"
                      />
                    )}
                  </HStack>
                </Th>
                <Th borderBottom="1px solid" borderColor="whiteAlpha.200">
                  <HStack
                    spacing={2}
                    cursor="pointer"
                    onClick={() => handleSort("createdAt")}
                  >
                    <Text>Submitted</Text>
                    {sortField === "createdAt" && (
                      <IconButton
                        aria-label={`Sort ${
                          sortOrder === "asc" ? "descending" : "ascending"
                        }`}
                        icon={
                          sortOrder === "asc" ? (
                            <ChevronUpIcon />
                          ) : (
                            <ChevronDownIcon />
                          )
                        }
                        size="xs"
                        variant="ghost"
                      />
                    )}
                  </HStack>
                </Th>
              </Tr>
            </Thead>
            <Tbody>
              {filteredAndSortedSubmissions?.map((submission) => (
                <Tr
                  key={submission.id}
                  _hover={{ bg: "whiteAlpha.50" }}
                  transition="background-color 0.2s"
                >
                  <Td borderBottom="1px solid" borderColor="whiteAlpha.100">
                    <ChakraLink
                      as={Link}
                      href={`/problems/${submission.problem.slug}`}
                      style={{  display: 'block', cursor: 'pointer' }}
                      color="blue.400"
                      _hover={{ color: "blue.300" }}
                    >
                      {submission.problem.title}
                    </ChakraLink>
                  </Td>
                  <Td borderBottom="1px solid" borderColor="whiteAlpha.100">
                    <Link
                      href={`/submissions/${submission.id}`}
                      style={{ textDecoration: 'none', color: 'inherit', display: 'block', cursor: 'pointer' }}
                    >
                      <Badge colorScheme={getStatusColor(submission.status)}>
                        {formatStatus(submission.status)}
                      </Badge>
                    </Link>
                  </Td>
                  <Td borderBottom="1px solid" borderColor="whiteAlpha.100">
                    <Link
                      href={`/submissions/${submission.id}`}
                      style={{ textDecoration: 'none', color: 'inherit', display: 'block', cursor: 'pointer' }}
                    >
                      {submission.status === "ACCEPTED" ? (
                        <Tooltip
                          label={`Runtime: ${submission.runtime?.toFixed(2)} ms`}
                        >
                          <Text fontWeight="medium">
                            {submission.gflops?.toFixed(2)} GFLOPS
                          </Text>
                        </Tooltip>
                      ) : (
                        <Text color="whiteAlpha.700">-</Text>
                      )}
                    </Link>
                  </Td>
                  <Td borderBottom="1px solid" borderColor="whiteAlpha.100">
                    <Link
                      href={`/submissions/${submission.id}`}
                      style={{ textDecoration: 'none', color: 'inherit', display: 'block', cursor: 'pointer' }}
                    >
                      {GPU_DISPLAY_NAMES[submission.gpuType ?? "T4"]}
                    </Link>
                  </Td>
                  <Td borderBottom="1px solid" borderColor="whiteAlpha.100">
                    <Link
                      href={`/submissions/${submission.id}`}
                      style={{ textDecoration: 'none', color: 'inherit', display: 'block', cursor: 'pointer' }}
                    >
                      {LANGUAGE_DISPLAY_NAMES[submission.language]}
                    </Link>
                  </Td>
                  <Td borderBottom="1px solid" borderColor="whiteAlpha.100">
                    <Link
                      href={`/submissions/${submission.id}`}
                      style={{ textDecoration: 'none', color: 'inherit', display: 'block', cursor: 'pointer' }}
                    >
                      <Tooltip
                        label={new Date(submission.createdAt).toLocaleString()}
                      >
                        <Text>
                          {formatDistanceToNow(new Date(submission.createdAt))}{" "}
                          ago
                        </Text>
                      </Tooltip>
                    </Link>
                  </Td>
                </Tr>
              ))}
            </Tbody>
          </Table>
        </Box>
      </Box>
    </Layout>
  );
};

export default SubmissionsPage;
