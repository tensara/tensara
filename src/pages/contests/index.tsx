import React, { useState } from "react";
import {
  Box,
  Text,
  Flex,
  Heading,
  Button,
  Badge,
  Tabs,
  TabList,
  Tab,
  SimpleGrid,
  Card,
  CardHeader,
  CardBody,
  Divider,
  Link as ChakraLink,
  Icon,
  Progress,
  useMediaQuery,
} from "@chakra-ui/react";
import { Layout } from "~/components/layout";
import {
  FaCalendarAlt,
  FaClock,
  FaUsers,
  FaExternalLinkAlt,
} from "react-icons/fa";
import Link from "next/link";
import { motion } from "framer-motion";
import { useRouter } from "next/router";
import { api } from "~/utils/api";

// Helper functions
const formatTimeRemaining = (endTime: Date) => {
  const now = new Date();
  const diff = endTime.getTime() - now.getTime();

  if (diff <= 0) return "Ended";

  const days = Math.floor(diff / (1000 * 60 * 60 * 24));
  const hours = Math.floor((diff % (1000 * 60 * 60 * 24)) / (1000 * 60 * 60));

  if (days > 0) {
    return `${days}d ${hours}h remaining`;
  } else {
    const minutes = Math.floor((diff % (1000 * 60 * 60)) / (1000 * 60));
    return `${hours}h ${minutes}m remaining`;
  }
};

const getContestProgress = (startTime: Date, endTime: Date) => {
  const now = new Date();
  const total = endTime.getTime() - startTime.getTime();
  const elapsed = now.getTime() - startTime.getTime();

  if (elapsed <= 0) return 0;
  if (elapsed >= total) return 100;

  return Math.floor((elapsed / total) * 100);
};

const getStatusColor = (status: string) => {
  switch (status) {
    case "ACTIVE":
      return "green";
    case "UPCOMING":
      return "blue";
    case "COMPLETED":
      return "gray";
    case "ARCHIVED":
      return "gray";
    default:
      return "gray";
  }
};

const getStatusName = (status: string) => {
  switch (status) {
    case "active":
      return "Active";
    case "upcoming":
      return "Upcoming";
    case "completed":
      return "Completed";
    default:
      return status;
  }
};

export default function ContestsPage() {
  const [statusFilter, setStatusFilter] = useState<
    "UPCOMING" | "ACTIVE" | "COMPLETED" | "ARCHIVED" | "ALL"
  >("ALL");
  const router = useRouter();
  const [isMobile] = useMediaQuery("(max-width: 768px)");

  const {
    data: contests = [],
    isLoading,
    error,
  } = api.contest.getAll.useQuery({
    status: statusFilter,
  });

  return (
    <Layout
      title="Contests | Tensara"
      ogTitle="Contests | Tensara"
      ogDescription="Join programming contests on Tensara."
    >
      <Box maxW="7xl" mx="auto" px={4} py={8}>
        <Flex
          justifyContent="space-between"
          alignItems="center"
          mb={6}
          direction={{ base: "column", sm: "row" }}
          gap={{ base: 4, sm: 0 }}
        >
          <Heading size="lg">Contests</Heading>

          <Flex align="center" gap={4}>
            <Tabs variant="soft-rounded" colorScheme="blue">
              <TabList bg="whiteAlpha.100" p={1} borderRadius="full">
                <Tab
                  _selected={{ color: "white", bg: "whiteAlpha.100" }}
                  onClick={() => setStatusFilter("ALL")}
                >
                  All
                </Tab>
                <Tab
                  _selected={{ color: "white", bg: "whiteAlpha.100" }}
                  onClick={() => setStatusFilter("ACTIVE")}
                >
                  Active
                </Tab>
                <Tab
                  _selected={{ color: "white", bg: "whiteAlpha.100" }}
                  onClick={() => setStatusFilter("UPCOMING")}
                >
                  Upcoming
                </Tab>
                <Tab
                  _selected={{ color: "white", bg: "whiteAlpha.100" }}
                  onClick={() => setStatusFilter("COMPLETED")}
                >
                  Completed
                </Tab>
              </TabList>
            </Tabs>
          </Flex>
        </Flex>

        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -10 }}
          transition={{ duration: 0.3 }}
        >
          {contests.length > 0 ? (
            <SimpleGrid columns={{ base: 1, lg: 2 }} spacing={6}>
              {contests.map((contest) => (
                <Card
                  key={contest.id}
                  bg="gray.800"
                  borderColor="whiteAlpha.200"
                  borderWidth={1}
                  borderRadius="md"
                  transition="transform 0.2s, box-shadow 0.2s"
                  _hover={{
                    transform: "translateY(-2px)",
                    boxShadow: "lg",
                  }}
                  position="relative"
                  overflow="hidden"
                >
                  {contest.status === "ACTIVE" && (
                    <Progress
                      value={getContestProgress(
                        contest.startTime,
                        contest.endTime
                      )}
                      size="xs"
                      colorScheme="blue"
                      position="absolute"
                      top={0}
                      left={0}
                      right={0}
                      borderTopLeftRadius="md"
                      borderTopRightRadius="md"
                    />
                  )}

                  <CardHeader pb={2}>
                    <Flex
                      justifyContent="space-between"
                      alignItems="center"
                      mb={1}
                    >
                      <Badge
                        colorScheme={getStatusColor(contest.status)}
                        variant="subtle"
                        px={2}
                        py={1}
                        borderRadius="full"
                        fontSize="xs"
                      >
                        {getStatusName(contest.status)}
                      </Badge>
                      <Flex align="center" gap={2}>
                        <Icon as={FaUsers} boxSize={3} color="gray.400" />
                        <Text fontSize="sm" color="gray.400">
                          {contest.participantCount} participants
                        </Text>
                      </Flex>
                    </Flex>

                    <Flex gap={3} alignItems="center" mt={4}>
                      <ChakraLink as={Link} href={`/contests/${contest.slug}`}>
                        <Heading
                          size="md"
                          color="white"
                          _hover={{ color: "blue.400" }}
                          noOfLines={1}
                        >
                          {contest.title}
                        </Heading>
                      </ChakraLink>
                      <ChakraLink href={`/contests/${contest.slug}`}>
                        <Icon
                          as={FaExternalLinkAlt}
                          color="gray.400"
                          boxSize={3}
                          _hover={{ color: "blue.400" }}
                        />
                      </ChakraLink>
                    </Flex>

                    <Text color="gray.400" fontSize="sm" mt={1} noOfLines={2}>
                      {contest.description}
                    </Text>
                  </CardHeader>

                  <CardBody pt={2}>
                    <Flex direction="column" gap={4}>
                      <Flex justify="space-between" wrap="wrap" gap={2}>
                        <Flex align="center" gap={2}>
                          <Icon
                            as={FaCalendarAlt}
                            boxSize={3}
                            color="gray.400"
                          />
                          <Text fontSize="sm" color="gray.300">
                            {contest.startTime.toLocaleDateString()} -{" "}
                            {contest.endTime.toLocaleDateString()}
                          </Text>
                        </Flex>

                        <Flex align="center" gap={2}>
                          <Icon as={FaClock} boxSize={3} color="gray.400" />
                          <Text fontSize="sm" color="gray.300">
                            {contest.status === "UPCOMING"
                              ? `Starts in ${Math.ceil(
                                  (contest.startTime.getTime() -
                                    new Date().getTime()) /
                                    (1000 * 60 * 60 * 24)
                                )} days`
                              : formatTimeRemaining(contest.endTime)}
                          </Text>
                        </Flex>
                      </Flex>

                      <Flex justify="center" align="center">
                        <Button
                          bg="brand.navbar"
                          _hover={{ bg: "green.500" }}
                          size="sm"
                          onClick={() =>
                            router.push(`/contests/${contest.slug}`)
                          }
                        >
                          {contest.status === "COMPLETED"
                            ? "View Results"
                            : "Participate"}
                        </Button>
                      </Flex>

                      {contest.status === "COMPLETED" && contest.winners && (
                        <>
                          <Divider borderColor="whiteAlpha.200" />
                        </>
                      )}
                    </Flex>
                  </CardBody>
                </Card>
              ))}
            </SimpleGrid>
          ) : (
            <Box
              p={6}
              bg="gray.800"
              borderRadius="md"
              borderColor="whiteAlpha.200"
              borderWidth={1}
              textAlign="center"
            >
              <Text>No contests found matching the selected filter.</Text>
            </Box>
          )}
        </motion.div>
      </Box>
    </Layout>
  );
}
