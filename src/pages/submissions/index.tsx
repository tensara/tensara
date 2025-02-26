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

type SortField = "createdAt" | "status" | "problem" | "performance";
type SortOrder = "asc" | "desc";

interface SubmissionWithProblem extends Submission {
  problem: {
    title: string;
    slug: string;
  };
  user: {
    name: string | null;
  };
}

const GPU_DISPLAY_NAMES: Record<string, string> = {
  "T4": "NVIDIA T4",
  "H100": "NVIDIA H100",
  "A100-80GB": "NVIDIA A100-80GB",
  "A10G": "NVIDIA A10G",
  "L4": "NVIDIA L4"
};


const SubmissionsPage: NextPage = () => {
  const [statusFilter, setStatusFilter] = useState<string>("all");
  const [gpuFilter, setGpuFilter] = useState<string>("all");
  const [searchQuery, setSearchQuery] = useState("");
  const [sortField, setSortField] = useState<SortField>("createdAt");
  const [sortOrder, setSortOrder] = useState<SortOrder>("desc");
  const router = useRouter();

  const { data: submissions, isLoading } =
    api.submissions.getAllSubmissions.useQuery();

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
      const matchesSearch =
        searchQuery === "" ||
        submission.problem.title
          .toLowerCase()
          .includes(searchQuery.toLowerCase());
      return matchesStatus && matchesGpu && matchesSearch;
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
    <Layout title="All Submissions">
      <Box maxW="7xl" mx="auto" px={4} py={8}>
        <Text fontSize="2xl" fontWeight="bold" mb={6}>
          All Submissions
        </Text>

        {/* Filters */}
        <HStack spacing={4} mb={6}>
          <Select
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value)}
            w="200px"
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
            w="200px"
            bg="whiteAlpha.50"
            borderColor="transparent"
            _hover={{ borderColor: "gray.600" }}
            _focus={{ borderColor: "gray.500" }}
          >
            <option value="all">All GPUs</option>
            <option value="T4">NVIDIA T4</option>
            <option value="H100">NVIDIA H100</option>
            <option value="A10G">NVIDIA A10G</option>
            <option value="A100-80GB">NVIDIA A100-80GB</option>
          </Select>

          <Input
            placeholder="Search by problem name..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            w="300px"
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
                  User
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
                    onClick={() => handleSort("createdAt")}
                  >
                    <Text>GPU</Text>
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
                  _hover={{ bg: "whiteAlpha.50", cursor: "pointer" }}
                  transition="background-color 0.2s"
                  onClick={() => router.push(`/submissions/${submission.id}`)}
                >
                  <Td borderBottom="1px solid" borderColor="whiteAlpha.100">
                    <ChakraLink
                      as={Link}
                      href={`/problems/${submission.problem.slug}`}
                      color="blue.400"
                      _hover={{ color: "blue.300" }}
                    >
                      {submission.problem.title}
                    </ChakraLink>
                  </Td>
                  <Td borderBottom="1px solid" borderColor="whiteAlpha.100">
                    <HStack spacing={2}>
                      <Text>{submission.user.name ?? "Anonymous"}</Text>
                    </HStack>
                  </Td>
                  <Td borderBottom="1px solid" borderColor="whiteAlpha.100">
                    <Badge colorScheme={getStatusColor(submission.status)}>
                      {formatStatus(submission.status)}
                    </Badge>
                  </Td>
                  <Td borderBottom="1px solid" borderColor="whiteAlpha.100">
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
                  </Td>
                  <Td borderBottom="1px solid" borderColor="whiteAlpha.100">
                    {GPU_DISPLAY_NAMES[submission.gpuType ?? "T4"]}
                  </Td>
                  <Td borderBottom="1px solid" borderColor="whiteAlpha.100">
                    <Tooltip
                      label={new Date(submission.createdAt).toLocaleString()}
                    >
                      <Text>
                        {formatDistanceToNow(new Date(submission.createdAt))}{" "}
                        ago
                      </Text>
                    </Tooltip>
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
