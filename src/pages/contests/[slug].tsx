import { useState, useEffect, useCallback } from "react";
import {
  Box,
  Text,
  VStack,
  HStack,
  Heading,
  Badge,
  Button,
  Flex,
  Spinner,
  useColorModeValue,
  Icon,
  SimpleGrid,
  Container,
  Fade,
  SlideFade,
  Image,
} from "@chakra-ui/react";
import { motion } from "framer-motion";
import { useRouter } from "next/router";
import { api } from "~/utils/api";
import { useSession } from "next-auth/react";
import { Layout } from "~/components/layout";
import { FaCalendarAlt, FaUsers, FaTrophy, FaClock } from "react-icons/fa";
import { type Problem, type Contest } from "@prisma/client";
import { createServerSideHelpers } from "@trpc/react-query/server";
import { appRouter } from "~/server/api/root";
import { createInnerTRPCContext } from "~/server/api/trpc";
import superjson from "superjson";
import { type GetServerSidePropsContext, type GetServerSideProps } from "next";
import contestImage from "./contests.png";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeHighlight from "rehype-highlight";
import "highlight.js/styles/github-dark.css";
import type { Components } from "react-markdown";
const MotionBox = motion(Box);

// Helper function to format dates
const formatDate = (date: Date) => {
  return date.toLocaleString("en-US", {
    weekday: "short",
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
    timeZoneName: "short",
  });
};

// Helper to determine contest status
const getStatusInfo = (contest: Contest) => {
  const now = new Date();
  const start = new Date(contest.startTime);
  const end = new Date(contest.endTime);
  const regStart = new Date(contest.registrationStartTime);
  const regEnd = new Date(contest.registrationEndTime);

  if (now < regStart) {
    return {
      status: "COMING SOON",
      color: "gray.500",
      badgeColor: "gray",
      message: "Registration opens soon",
      bgGradient: "linear(to-r, gray.500, gray.600)",
    };
  } else if (now >= regStart && now < regEnd) {
    return {
      status: "REGISTRATION OPEN",
      color: "green.500",
      badgeColor: "green",
      message: "Register now to participate",
      bgGradient: "linear(to-r, green.500, green.600)",
    };
  } else if (now >= regEnd && now < start) {
    return {
      status: "REGISTRATION CLOSED",
      color: "orange.500",
      badgeColor: "orange",
      message: "Contest starts soon",
      bgGradient: "linear(to-r, orange.500, orange.600)",
    };
  } else if (now >= start && now < end) {
    return {
      status: "LIVE",
      color: "green.500",
      badgeColor: "green",
      message: "Contest is currently running",
      bgGradient: "linear(to-r, green.500, green.600)",
    };
  } else {
    return {
      status: "ENDED",
      color: "red.500",
      badgeColor: "red",
      message: "Contest has concluded",
      bgGradient: "linear(to-r, red.500, red.600)",
    };
  }
};

const CountdownTimer = ({ targetDate }: { targetDate: Date }) => {
  const [timeLeft, setTimeLeft] = useState({
    days: 0,
    hours: 0,
    minutes: 0,
    seconds: 0,
  });

  useEffect(() => {
    const interval = setInterval(() => {
      const now = new Date();
      const difference = new Date(targetDate).getTime() - now.getTime();

      if (difference <= 0) {
        clearInterval(interval);
        return;
      }

      const days = Math.floor(difference / (1000 * 60 * 60 * 24));
      const hours = Math.floor((difference / (1000 * 60 * 60)) % 24);
      const minutes = Math.floor((difference / (1000 * 60)) % 60);
      const seconds = Math.floor((difference / 1000) % 60);

      setTimeLeft({ days, hours, minutes, seconds });
    }, 1000);

    return () => clearInterval(interval);
  }, [targetDate]);

  const timeUnits = [
    { value: timeLeft.days, label: "DAYS" },
    { value: timeLeft.hours, label: "HOURS" },
    { value: timeLeft.minutes, label: "MINS" },
    { value: timeLeft.seconds, label: "SECS" },
  ];

  const bg = useColorModeValue("whiteAlpha.100", "blackAlpha.200");

  return (
    <HStack spacing={4} my={4}>
      {timeUnits.map((unit, index) => (
        <VStack
          key={index}
          p={4}
          bg={bg}
          borderRadius="xl"
          minW="70px"
          transition="all 0.2s"
          _hover={{ transform: "scale(1.05)", shadow: "lg" }}
          boxShadow="md"
        >
          <Text fontSize="2xl" fontWeight="bold" color="white">
            {unit.value}
          </Text>
          <Text fontSize="xs" color="whiteAlpha.800">
            {unit.label}
          </Text>
        </VStack>
      ))}
    </HStack>
  );
};

export const getServerSideProps = async (
  context: GetServerSidePropsContext
) => {
  const slug = context.params?.slug;

  const helpers = createServerSideHelpers({
    router: appRouter,
    ctx: createInnerTRPCContext({ session: null }),
    transformer: superjson,
  });

  try {
    await helpers.contest.getBySlug.prefetch(slug as string);
    return {
      props: {
        slug,
        trpcState: helpers.dehydrate(),
      },
    };
  } catch (e) {
    console.error(e);
    return {
      notFound: true,
    };
  }
};

export default function ContestPage({ slug }: { slug: string }) {
  const { data: session } = useSession();
  const router = useRouter();
  const { data: contest, isLoading } = api.contest.getBySlug.useQuery(slug);
  const [isParticipant, setIsParticipant] = useState(false);
  const [problems, setProblems] = useState<Problem[]>([]);
  const [loadingProblems, setLoadingProblems] = useState(false);
  const problemBgColor = useColorModeValue("blackAlpha.50", "whiteAlpha.50");
  const borderColor = useColorModeValue("gray.200", "gray.700");

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1,
      },
    },
  };

  const itemVariants = {
    hidden: { y: 20, opacity: 0 },
    visible: {
      y: 0,
      opacity: 1,
      transition: {
        duration: 0.5,
      },
    },
  };

  const userParticipantStatus = api.contest.getIfUserIsParticipant.useQuery(
    {
      contestId: contest?.id ?? "",
      userId: session?.user?.id ?? "",
    },
    { enabled: !!contest && !!session }
  );

  const joinContest = async () => {
    setIsParticipant(true);
  };

  const loadProblems = useCallback(async () => {
    if (!contest) return;
    const problems = api.contest.getContestProblems.useQuery({
      contestId: contest.id,
    });
    setProblems(problems.data ?? []);
  }, [contest]);

  useEffect(() => {
    if (contest && isParticipant) {
      const now = new Date();
      const start = new Date(contest.startTime);
      const end = new Date(contest.endTime);

      if (now >= start && now < end) {
        void loadProblems();
      }
    }
  }, [contest, isParticipant, loadProblems]);

  if (isLoading) {
    return (
      <Layout title="Loading Contest | Tensara">
        <Flex justify="center" align="center" h="50vh">
          <Spinner size="xl" />
        </Flex>
      </Layout>
    );
  }

  if (!contest) {
    return (
      <Layout title="Contest Not Found | Tensara">
        <Box p={8} textAlign="center">
          <Heading mb={4}>Contest Not Found</Heading>
          <Text>
            We couldn&apos;t find the contest you&apos;re looking for.
          </Text>
          <Button mt={6} onClick={() => router.push("/contests")}>
            Back to Contests
          </Button>
        </Box>
      </Layout>
    );
  }

  const statusInfo = getStatusInfo(contest);
  const now = new Date();
  const contestStarted = new Date(contest.startTime) <= now;
  const contestEnded = new Date(contest.endTime) <= now;
  const registrationOpen =
    new Date(contest.registrationStartTime) <= now &&
    new Date(contest.registrationEndTime) > now;

  return (
    <Layout
      title={`${contest.title} | Tensara`}
      ogTitle={contest.title}
      ogImgSubtitle={`Contest | Tensara`}
    >
      <Container maxW="container.xl">
        <MotionBox
          initial="hidden"
          animate="visible"
          variants={containerVariants}
          mb={10}
        >
          {/* Header Section */}
          <MotionBox variants={itemVariants} mb={8}>
            <SlideFade in={true} offsetY={20}>
              <Flex
                direction={{ base: "column", md: "row" }}
                align={{ base: "center", md: "flex-start" }}
                mb={6}
                gap={8}
              >
                <Box
                  position="relative"
                  overflow="hidden"
                  borderRadius="2xl"
                  maxW={{ base: "100%", md: "400px" }}
                  boxShadow="xl"
                >
                  <Image
                    src={contestImage.src}
                    alt={contest.title}
                    width={600}
                    height={400}
                    objectFit="cover"
                  />
                  <Badge
                    position="absolute"
                    top={4}
                    right={4}
                    colorScheme={statusInfo.badgeColor}
                    fontSize="sm"
                    py={1}
                    px={3}
                    borderRadius="full"
                    bgGradient={statusInfo.bgGradient}
                    color="white"
                    boxShadow="md"
                  >
                    {statusInfo.status}
                  </Badge>
                </Box>

                <VStack align="stretch" flex={1} spacing={6}>
                  <VStack align="stretch" spacing={4}>
                    <Heading size="xl" fontFamily="Space Grotesk, sans-serif">
                      {contest.title}
                    </Heading>
                    <Text
                      color={statusInfo.color}
                      fontWeight="medium"
                      fontSize="lg"
                    >
                      {statusInfo.message}
                    </Text>
                  </VStack>

                  <SimpleGrid columns={{ base: 1, md: 2 }} spacing={6}>
                    <VStack
                      align="start"
                      spacing={2}
                      p={4}
                      bg="whiteAlpha.50"
                      borderRadius="xl"
                    >
                      <HStack>
                        <Icon as={FaCalendarAlt} color="blue.400" />
                        <Text fontWeight="medium">Start Time</Text>
                      </HStack>
                      <Text>{formatDate(contest.startTime)}</Text>
                    </VStack>

                    <VStack
                      align="start"
                      spacing={2}
                      p={4}
                      bg="whiteAlpha.50"
                      borderRadius="xl"
                    >
                      <HStack>
                        <Icon as={FaClock} color="red.400" />
                        <Text fontWeight="medium">End Time</Text>
                      </HStack>
                      <Text>{formatDate(contest.endTime)}</Text>
                    </VStack>

                    <VStack
                      align="start"
                      spacing={2}
                      p={4}
                      bg="whiteAlpha.50"
                      borderRadius="xl"
                    >
                      <HStack>
                        <Icon as={FaUsers} color="purple.400" />
                        <Text fontWeight="medium">Participants</Text>
                      </HStack>
                      <Text>{contest.participantCount || 0} registered</Text>
                    </VStack>

                    {contest.winners && (
                      <VStack
                        align="start"
                        spacing={2}
                        p={4}
                        bg="whiteAlpha.50"
                        borderRadius="xl"
                      >
                        <HStack>
                          <Icon as={FaTrophy} color="yellow.400" />
                          <Text fontWeight="medium">Winners</Text>
                        </HStack>
                        <Text>Announced</Text>
                      </VStack>
                    )}
                  </SimpleGrid>

                  {!contestEnded && (
                    <Box
                      p={6}
                      bg="whiteAlpha.50"
                      borderRadius="xl"
                      textAlign="center"
                    >
                      {!contestStarted ? (
                        <>
                          <Text fontWeight="bold" fontSize="lg" mb={4}>
                            Contest starts in:
                          </Text>
                          <CountdownTimer
                            targetDate={new Date(contest.startTime)}
                          />
                        </>
                      ) : (
                        <>
                          <Text fontWeight="bold" fontSize="lg" mb={4}>
                            Contest ends in:
                          </Text>
                          <CountdownTimer
                            targetDate={new Date(contest.endTime)}
                          />
                        </>
                      )}
                    </Box>
                  )}

                  {session && (
                    <Flex
                      mt={4}
                      gap={4}
                      direction={{ base: "column", sm: "row" }}
                    >
                      {!isParticipant && registrationOpen && (
                        <Button
                          bgGradient="linear(to-r, brand.primary, brand.navbar)"
                          leftIcon={<FaUsers />}
                          onClick={joinContest}
                          size="lg"
                          transition="all 0.3s"
                          _hover={{
                            transform: "translateY(-2px)",
                            shadow: "lg",
                            bgGradient:
                              "linear(to-r, brand.navbar, brand.primary)",
                          }}
                          color="white"
                        >
                          Register for Contest
                        </Button>
                      )}
                      {isParticipant && contestStarted && !contestEnded && (
                        <Button
                          colorScheme="green"
                          size="lg"
                          onClick={() =>
                            router.push(`/contests/${slug}/problems`)
                          }
                          transition="all 0.3s"
                          _hover={{
                            transform: "translateY(-2px)",
                            shadow: "lg",
                          }}
                        >
                          Go to Problems
                        </Button>
                      )}
                      {!session && (
                        <Button
                          variant="outline"
                          onClick={() => router.push("/auth/signin")}
                          color="white"
                          _hover={{ bg: "whiteAlpha.100" }}
                        >
                          Sign in to participate
                        </Button>
                      )}
                    </Flex>
                  )}
                </VStack>
              </Flex>
            </SlideFade>
          </MotionBox>

          {/* Description Section */}
          <MotionBox variants={itemVariants} mb={8}>
            <Fade in={true} delay={0.2}>
              <Box
                p={8}
                borderRadius="2xl"
                bg="whiteAlpha.50"
                boxShadow="lg"
                borderWidth="1px"
                borderColor="whiteAlpha.200"
              >
                <Heading
                  size="md"
                  mb={6}
                  fontFamily="Space Grotesk, sans-serif"
                >
                  About this Contest
                </Heading>
                <ReactMarkdown
                  remarkPlugins={[remarkGfm]}
                  rehypePlugins={[rehypeHighlight]}
                  className="markdown-content"
                  components={{
                    p: ({ children }) => (
                      <Text mb={4} color="whiteAlpha.900">
                        {children}
                      </Text>
                    ),
                    h1: ({ children }) => (
                      <Heading size="xl" mb={4} color="white">
                        {children}
                      </Heading>
                    ),
                    h2: ({ children }) => (
                      <Heading size="lg" mb={3} color="white">
                        {children}
                      </Heading>
                    ),
                    h3: ({ children }) => (
                      <Heading size="md" mb={2} color="white">
                        {children}
                      </Heading>
                    ),
                    ul: ({ children }) => (
                      <Box as="ul" pl={6} mb={4} color="whiteAlpha.900">
                        {children}
                      </Box>
                    ),
                    ol: ({ children }) => (
                      <Box as="ol" pl={6} mb={4} color="whiteAlpha.900">
                        {children}
                      </Box>
                    ),
                    li: ({ children }) => (
                      <Box as="li" mb={1} color="whiteAlpha.900">
                        {children}
                      </Box>
                    ),
                    code: ({ className, children }) => {
                      return (
                        <Box
                          as="pre"
                          p={4}
                          bg="gray.800"
                          color="white"
                          borderRadius="md"
                          overflowX="auto"
                          mb={4}
                        >
                          <code className={className}>{children} </code>
                        </Box>
                      );
                    },
                  }}
                >
                  {contest.description ??
                    "No description available for this contest."}
                </ReactMarkdown>
              </Box>
            </Fade>
          </MotionBox>

          {/* Problems Section - Only visible for participants during active contest */}
          {isParticipant && contestStarted && !contestEnded && (
            <MotionBox variants={itemVariants}>
              <SlideFade in={true} offsetY={20} delay={0.3}>
                <Box
                  p={8}
                  borderRadius="2xl"
                  bg="whiteAlpha.50"
                  boxShadow="lg"
                  borderWidth="1px"
                  borderColor="whiteAlpha.200"
                >
                  <Flex justify="space-between" align="center" mb={6}>
                    <Heading size="md" fontFamily="Space Grotesk, sans-serif">
                      Contest Problems
                    </Heading>
                    <Button
                      size="sm"
                      onClick={() => router.push(`/contests/${slug}/problems`)}
                      colorScheme="blue"
                    >
                      View All
                    </Button>
                  </Flex>

                  {loadingProblems ? (
                    <Flex justify="center" py={8}>
                      <Spinner size="xl" color="brand.primary" />
                    </Flex>
                  ) : problems.length > 0 ? (
                    <SimpleGrid columns={{ base: 1, md: 2, lg: 3 }} spacing={6}>
                      {problems.map((problem, index) => (
                        <Box
                          key={index}
                          p={6}
                          borderRadius="xl"
                          bg="whiteAlpha.100"
                          _hover={{
                            transform: "translateY(-2px)",
                            shadow: "lg",
                            transition: "all 0.2s ease-in-out",
                          }}
                          cursor="pointer"
                          onClick={() =>
                            router.push(
                              `/contests/${slug}/problems/${problem.id}`
                            )
                          }
                          borderWidth="1px"
                          borderColor="whiteAlpha.200"
                        >
                          <Heading size="sm" mb={3} color="white">
                            {problem.title}
                          </Heading>
                          <Badge
                            colorScheme={
                              problem.difficulty === "EASY"
                                ? "green"
                                : problem.difficulty === "MEDIUM"
                                  ? "yellow"
                                  : "red"
                            }
                            fontSize="xs"
                            px={2}
                            py={1}
                            borderRadius="full"
                          >
                            {problem.difficulty}
                          </Badge>
                        </Box>
                      ))}
                    </SimpleGrid>
                  ) : (
                    <Text
                      textAlign="center"
                      py={8}
                      color="whiteAlpha.800"
                      fontSize="lg"
                    >
                      Problems will be available when the contest starts.
                    </Text>
                  )}
                </Box>
              </SlideFade>
            </MotionBox>
          )}
        </MotionBox>
      </Container>
    </Layout>
  );
}
