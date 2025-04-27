import {
  Box,
  Button,
  Container,
  Heading,
  Text,
  VStack,
  Icon,
  SimpleGrid,
  Link,
  Divider,
  Badge,
  Flex,
  Card,
  Grid,
  Stat,
  StatLabel,
  StatHelpText,
  StatNumber,
} from "@chakra-ui/react";
import { motion, AnimatePresence } from "framer-motion"; // Import AnimatePresence
import React, { useState, useEffect, useCallback } from "react"; // Import useCallback
import { Layout } from "~/components/layout";
import {
  FiCpu,
  FiAward,
  FiUsers,
  FiTerminal,
  FiZap,
  FiArrowRight,
  FiExternalLink,
  FiGitPullRequest,
  FiCode,
  FiBookOpen,
} from "react-icons/fi";
import { FaDiscord, FaGithub, FaTwitter, FaEnvelope } from "react-icons/fa";
import { type IconType } from "react-icons";
import AnimatedCudaEditor from "~/components/CudaEditor";
import LandingBenchmarkDisplay from "~/components/landing/LandingBenchmarkDisplay"; // Import the new component

// Create motion components
const MotionVStack = motion(VStack);
const MotionBox = motion(Box);
const MotionSimpleGrid = motion(SimpleGrid);
const MotionFlex = motion(Flex);
const MotionCard = motion(Card);
const MotionGrid = motion(Grid);

const FeatureCard = ({
  icon,
  title,
  description,
}: {
  icon: IconType;
  title: string;
  description: string;
}) => {
  return (
    <Box
      bg="brand.card"
      borderRadius="xl"
      p={6}
      borderWidth="1px"
      borderColor="rgba(46, 204, 113, 0.3)"
      transition="all 0.3s"
      _hover={{
        transform: "translateY(-5px)",
        boxShadow: "0 10px 30px rgba(46, 204, 113, 0.2)",
        borderColor: "rgba(46, 204, 113, 0.5)",
      }}
    >
      <Flex direction="column" align="flex-start">
        <Flex
          bg="rgba(14, 129, 68, 0.2)"
          p={3}
          borderRadius="md"
          mb={4}
          color="#2ecc71"
        >
          <Icon as={icon} boxSize={6} />
        </Flex>
        <Heading
          size="md"
          fontFamily="Space Grotesk, sans-serif"
          mb={2}
          color="white"
        >
          {title}
        </Heading>
        <Text color="whiteAlpha.800">{description}</Text>
      </Flex>
    </Box>
  );
};

// Update card component
const UpdateCard = ({
  title,
  type,
  description,
  date,
}: {
  title: string;
  type: string;
  description: string;
  date: string;
}) => {
  const getBadgeColor = (type: string) => {
    switch (type) {
      case "FEATURE":
        return "green";
      case "IMPROVEMENT":
        return "blue";
      case "RELEASE":
        return "purple";
      case "IN PROGRESS":
        return "yellow";
      default:
        return "gray";
    }
  };

  return (
    <Box
      p={4}
      borderBottomWidth="1px"
      borderColor="whiteAlpha.100"
      _hover={{
        bg: "rgba(14, 129, 68, 0.05)",
      }}
    >
      <Flex justify="space-between" align="center" mb={2}>
        <Heading size="sm" color="white">
          {title}
        </Heading>
        <Badge
          colorScheme={getBadgeColor(type)}
          fontSize="xs"
          borderRadius="md"
          px={2}
          py={1}
        >
          {type}
        </Badge>
      </Flex>
      <Text color="whiteAlpha.800" fontSize="sm" mb={1}>
        {description}
      </Text>
      <Text color="whiteAlpha.500" fontSize="xs">
        {date}
      </Text>
    </Box>
  );
};

export default function HomePage() {
  const [isMobile, setIsMobile] = useState(false);

  // State for animation sequence
  const [isTypingCode, setIsTypingCode] = useState(true); // Start with code typing
  const [isFadingCode, setIsFadingCode] = useState(false); // Code is not fading initially
  const [isShowingBenchmarks, setIsShowingBenchmarks] = useState(false); // Benchmarks hidden initially

  // Dummy data for the benchmark display
  const dummyBenchmarkData = [
    { id: "1", name: "n = 2^20", runtime_ms: 0.05, gflops: 19.79 },
    { id: "2", name: "n = 2^22", runtime_ms: 0.2, gflops: 21.08 },
    { id: "3", name: "n = 2^23", runtime_ms: 0.39, gflops: 21.29 },
    { id: "4", name: "n = 2^25", runtime_ms: 1.56, gflops: 21.49 },
    { id: "5", name: "n = 2^26", runtime_ms: 3.12, gflops: 21.53 },
    { id: "6", name: "n = 2^29", runtime_ms: 24.94, gflops: 21.53 },
    { id: "7", name: "n = 2^30", runtime_ms: 50.1, gflops: 21.43 },
  ];

  const handleTypingComplete = useCallback(() => {
    console.log("HomePage: handleTypingComplete called");
    const fadeOutTimer = setTimeout(() => {
      console.log("HomePage: Setting isFadingCode=true, isTypingCode=false");
      setIsFadingCode(true);
      setIsTypingCode(false);

      const fadeInTimer = setTimeout(() => {
        console.log("HomePage: Setting isShowingBenchmarks=true");
        setIsShowingBenchmarks(true);
      }, 500);

      return () => {
        console.log("HomePage: Clearing fadeInTimer");
        clearTimeout(fadeInTimer);
      };
    }, 1000);

    return () => {
      console.log("HomePage: Clearing fadeOutTimer");
      clearTimeout(fadeOutTimer);
    };
  }, []);

  useEffect(() => {
    const checkMobile = () => {
      setIsMobile(window.innerWidth <= 768);
    };

    checkMobile();
    window.addEventListener("resize", checkMobile);

    return () => window.removeEventListener("resize", checkMobile);
  }, []);

  console.log("HomePage Rendering:", {
    isTypingCode,
    isFadingCode,
    isShowingBenchmarks,
  }); // Log state on render

  return (
    <Layout title="Home" ogImage="/tensara_ogimage.png">
      <Box color="white" minH="100vh" p={10}>
        {/* Hero Section */}
        <MotionBox
          pt={10}
          pb={20}
          position="relative"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.8 }}
          // overflow="hidden" // Removed to allow glow to extend
          _before={{
            content: '""',
            position: "absolute",
            top: 0,
            left: 0,
            right: 0,
            height: "100%",
            zIndex: 0,
          }}
        >
          <Container maxW="8xl" position="relative" zIndex={1}>
            {/* Removed incorrect dedicated glow Box */}
            <Flex
              direction={{ base: "column", lg: "row" }}
              align="center"
              justify="space-between"
            >
              <MotionVStack
                align="flex-start"
                maxW={{ base: "full", lg: "40%" }}
                spacing={8}
                initial={{ opacity: 0, y: 30 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.8, delay: 0.3 }}
              >
                {/* Heading */}
                <Heading
                  fontSize={{ base: "4xl", md: "5xl", lg: "6xl" }}
                  fontWeight="bold"
                  lineHeight="1.1"
                  color="white"
                  position="relative"
                  sx={{
                    "&::after": {
                      content: '""',
                      position: "absolute",
                      top: "-10px",
                      left: "-10px",
                      right: "-10px",
                      bottom: "-10px",
                      zIndex: -1,
                      borderRadius: "xl",
                      filter: "blur(8px)",
                    },
                  }}
                >
                  <MotionBox
                    initial={{ y: 20, opacity: 0 }}
                    animate={{ y: 0, opacity: 1 }}
                    transition={{
                      duration: 0.7,
                      delay: 0.2,
                      type: "spring",
                      stiffness: 100,
                    }}
                    bgGradient="linear(to-r, brand.primary, brand.navbar)"
                    bgClip="text"
                    display="inline-block"
                    fontFamily="Space Grotesk, sans-serif"
                  >
                    Optimize
                  </MotionBox>
                  <MotionBox
                    initial={{ y: 20, opacity: 0 }}
                    animate={{ y: 0, opacity: 1 }}
                    transition={{
                      duration: 0.7,
                      delay: 0.4,
                      type: "spring",
                      stiffness: 100,
                    }}
                    bgGradient="linear(to-r, brand.primary, brand.navbar, brand.secondary)"
                    bgClip="text"
                    display="inline-block"
                    mt={2}
                    w="100%"
                    fontFamily="Space Grotesk, sans-serif"
                  >
                    Benchmark
                  </MotionBox>
                  <MotionBox
                    initial={{ y: 20, opacity: 0 }}
                    animate={{ y: 0, opacity: 1 }}
                    transition={{
                      duration: 0.7,
                      delay: 0.6,
                      type: "spring",
                      stiffness: 100,
                    }}
                    bgGradient="linear(to-r, brand.primary, brand.navbar)"
                    bgClip="text"
                    display="inline-block"
                    mt={2}
                    fontFamily="Space Grotesk, sans-serif"
                  >
                    Repeat
                  </MotionBox>
                </Heading>

                <MotionBox
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.5, delay: 0.8 }}
                >
                  <Text fontSize="xl" color="whiteAlpha.800">
                    A platform for GPU programming challenges. Write efficient
                    GPU kernels and compare your solutions with other
                    developers.
                  </Text>
                </MotionBox>

                <MotionFlex
                  gap={4}
                  direction={{ base: "column", sm: "row" }}
                  width={{ base: "full", sm: "auto" }}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.5, delay: 1 }}
                >
                  <Button
                    as={motion.button}
                    size="lg"
                    bg="#0e8144"
                    color="white"
                    _hover={{
                      bg: "#0a6434",
                      transform: "translateY(-2px)",
                      boxShadow: "0 5px 20px rgba(46, 204, 113, 0.4)",
                    }}
                    height="60px"
                    px={8}
                    leftIcon={<Icon as={FiCode} />}
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.98 }}
                  >
                    Start Solving
                  </Button>

                  <Button
                    as={motion.button}
                    size="lg"
                    variant="outline"
                    borderColor="whiteAlpha.300"
                    color="white"
                    _hover={{
                      borderColor: "whiteAlpha.500",
                      bg: "whiteAlpha.50",
                    }}
                    height="60px"
                    px={8}
                    leftIcon={<Icon as={FaGithub} />}
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.98 }}
                  >
                    View on GitHub
                  </Button>
                </MotionFlex>

                <MotionFlex
                  align="center"
                  gap={3}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.5, delay: 1.2 }}
                >
                  <Text color="whiteAlpha.600">
                    Need help getting started? Join our community on{" "}
                    <Link
                      href="https://discord.gg/YzBTfMxVQK"
                      isExternal
                      color="#2ecc71"
                      textDecoration="underline"
                    >
                      Discord
                    </Link>
                  </Text>
                </MotionFlex>
              </MotionVStack>
              {/* Right side container for animation - Adjusted for centering */}
              {!isMobile && (
                <Flex
                  // Removed flex={1}
                  w={{ base: "full", lg: "60%" }}
                  maxW="container.lg"
                  mx="auto"
                  justify="center"
                  align="center"
                  position="relative"
                  minH="420px"
                >
                  <AnimatePresence initial={false}>
                    <Box
                      position="absolute"
                      w="full"
                      display="flex"
                      justifyContent="center"
                    >
                      {(isTypingCode || isFadingCode) && (
                        <AnimatedCudaEditor
                          key="cuda-editor"
                          onTypingComplete={handleTypingComplete}
                          isFadingOut={isFadingCode}
                        />
                      )}
                    </Box>
                    <Box
                      position="absolute"
                      w="full"
                      display="flex"
                      justifyContent="center"
                    >
                      {isShowingBenchmarks && (
                        <LandingBenchmarkDisplay
                          key="benchmarks"
                          isVisible={isShowingBenchmarks}
                          dummyData={dummyBenchmarkData}
                        />
                      )}
                    </Box>
                  </AnimatePresence>
                </Flex>
              )}
            </Flex>
          </Container>
        </MotionBox>

        {/* Features Section */}
        <Container maxW="8xl" py={16}>
          <VStack align="flex-start" spacing={16}>
            <VStack align="flex-start" spacing={4} maxW="2xl">
              <Heading fontSize={{ base: "2xl", md: "3xl" }} color="white">
                Why Tensara?
              </Heading>
              <Text color="whiteAlpha.800" fontSize="lg">
                Tensara provides a unique platform for honing your GPU
                programming skills through competitive challenges and detailed
                benchmarking.
              </Text>
            </VStack>

            <SimpleGrid columns={{ base: 1, md: 3 }} spacing={8} w="full">
              <FeatureCard
                icon={FiCpu}
                title="Real Hardware Benchmarking"
                description="Submissions are run on standardized GPU hardware for fair and accurate performance comparisons."
              />
              <FeatureCard
                icon={FiAward}
                title="Competitive Leaderboards"
                description="See how your solutions stack up against others on detailed leaderboards for each problem."
              />
              <FeatureCard
                icon={FiUsers}
                title="Community & Collaboration"
                description="Discuss strategies, share insights, and learn from fellow GPU programming enthusiasts."
              />
            </SimpleGrid>
          </VStack>
        </Container>

        {/* Updates Section */}
        <Box py={16}>
          <Container maxW="8xl">
            <VStack align="flex-start" spacing={8}>
              <Heading fontSize={{ base: "2xl", md: "3xl" }} color="white">
                Latest Updates
              </Heading>

              <SimpleGrid columns={{ base: 1, md: 3 }} spacing={8} w="full">
                <Box
                  bg="brand.card"
                  borderRadius="xl"
                  overflow="hidden"
                  borderWidth="1px"
                  borderColor="whiteAlpha.100"
                >
                  <Flex
                    p={4}
                    borderBottomWidth="1px"
                    borderColor="brand.primary"
                    align="center"
                  >
                    <Icon as={FiGitPullRequest} color="brand.primary" mr={2} />
                    <Heading size="sm" color="white">
                      Core Platform
                    </Heading>
                  </Flex>
                  <Box>
                    <UpdateCard
                      title="Rating System"
                      type="FEATURE"
                      description="New rating system for user rankings."
                      date="2 weeks ago"
                    />
                    <UpdateCard
                      title="Triton Kernel Support"
                      type="FEATURE"
                      description="Added support for Triton-based kernel submissions."
                      date="3 weeks ago"
                    />
                    <UpdateCard
                      title="Error Handling"
                      type="IMPROVEMENT"
                      description="Improved error handling and rate limiting."
                      date="3 weeks ago"
                    />
                  </Box>
                </Box>

                <Box
                  bg="brand.card"
                  borderRadius="xl"
                  overflow="hidden"
                  borderWidth="1px"
                  borderColor="whiteAlpha.100"
                >
                  <Flex
                    p={4}
                    borderBottomWidth="1px"
                    borderColor="brand.primary"
                    align="center"
                  >
                    <Icon as={FiTerminal} color="brand.primary" mr={2} />
                    <Heading size="sm" color="white">
                      CLI Tool
                    </Heading>
                  </Flex>
                  <Box>
                    <UpdateCard
                      title="CLI Submissions"
                      type="IN PROGRESS"
                      description="Working on allowing direct submissions via CLI."
                      date="1 week ago"
                    />
                    <UpdateCard
                      title="CLI v0.1 Release"
                      type="RELEASE"
                      description="Initial release of the Tensara CLI."
                      date="1 month ago"
                    />
                    <UpdateCard
                      title="Local Benchmarking"
                      type="IMPROVEMENT"
                      description="Improved local benchmarking accuracy."
                      date="1 month ago"
                    />
                  </Box>
                </Box>

                <Box
                  bg="brand.card"
                  borderRadius="xl"
                  overflow="hidden"
                  borderWidth="1px"
                  borderColor="whiteAlpha.100"
                >
                  <Flex
                    p={4}
                    borderBottomWidth="1px"
                    borderColor="brand.primary"
                    align="center"
                  >
                    <Icon as={FiBookOpen} color="brand.primary" mr={2} />
                    <Heading size="sm" color="white">
                      Problems
                    </Heading>
                  </Flex>
                  <Box>
                    <UpdateCard
                      title="Convolution Problems"
                      type="FEATURE"
                      description="New set of convolution challenges available."
                      date="1 week ago"
                    />
                    <UpdateCard
                      title="3D/4D Tensor Matmul Problems"
                      type="FEATURE"
                      description="Added new matrix multiplication problems."
                      date="2 weeks ago"
                    />
                    <UpdateCard
                      title="Problem Difficulty Tags"
                      type="IMPROVEMENT"
                      description="Added difficulty tags to problems."
                      date="2 weeks ago"
                    />
                  </Box>
                </Box>
              </SimpleGrid>
            </VStack>
          </Container>
        </Box>

        {/* Call to Action Section */}
        <Container maxW="8xl" py={20}>
          <Box
            bgGradient="linear(to-br, brand.primary, brand.secondary)"
            borderRadius="3xl"
            p={{ base: 8, md: 16 }}
            textAlign="center"
            position="relative"
            overflow="hidden"
            _before={{
              content: '""',
              position: "absolute",
              top: "-50%",
              left: "-20%",
              width: "100%",
              height: "100%",
              bg: "radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 60%)",
              transform: "rotate(30deg)",
            }}
          >
            <Box position="relative" zIndex={1}>
              <Heading
                fontSize={{ base: "2xl", md: "4xl" }}
                color="white"
                mb={{ base: 3, md: 4 }}
                fontFamily="Space Grotesk, sans-serif"
                px={{ base: 2, md: 0 }}
              >
                Ready to Optimize?
              </Heading>
              <Text
                fontSize={{ base: "lg", md: "xl" }}
                color="whiteAlpha.900"
                mb={{ base: 6, md: 8 }}
                maxW="xl"
                mx="auto"
                px={{ base: 4, md: 0 }}
              >
                Dive into our GPU programming challenges, submit your kernels,
                and climb the leaderboards.
              </Text>
              <Link
                href="/problems"
                _hover={{ textDecoration: "none" }}
                display="inline-block"
                w={{ base: "full", md: "auto" }}
              >
                <Button
                  size="lg"
                  bg="rgba(255, 255, 255, 0.1)"
                  color="white"
                  _hover={{
                    bg: "rgba(255, 255, 255, 0.2)",
                    transform: "translateY(-2px)",
                  }}
                  height={{ base: "50px", md: "60px" }}
                  px={{ base: 6, md: 10 }}
                  rightIcon={<Icon as={FiArrowRight} />}
                  as={motion.button}
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.98 }}
                  backdropFilter="blur(8px)"
                  border="1px solid rgba(255, 255, 255, 0.2)"
                  w={{ base: "full", md: "auto" }}
                >
                  Start Solving Now
                </Button>
              </Link>
            </Box>
          </Box>
        </Container>

        {/* Footer Links */}
        <Box bg="transparent" pt={16} borderRadius="2xl">
          <Container maxW="8xl">
            {/* <Divider mb={10} borderColor="whiteAlpha.200" /> */}
            <SimpleGrid
              columns={{ base: 1, sm: 2, md: 4 }}
              spacing={8}
              color="whiteAlpha.700"
            >
              {/* About */}
              <VStack align="flex-start" spacing={4}>
                <Heading size="sm" color="white" mb={2}>
                  Tensara
                </Heading>
                <Text fontSize="sm">
                  GPU Programming Challenges & Benchmarking Platform
                </Text>
                {/* Social Icons */}
                <Flex gap={4} mt={2}>
                  <Link href="https://github.com/tensara" isExternal>
                    <Icon
                      as={FaGithub}
                      boxSize={5}
                      _hover={{ color: "white" }}
                    />
                  </Link>
                  <Link href="#" isExternal>
                    <Icon
                      as={FaTwitter}
                      boxSize={5}
                      _hover={{ color: "white" }}
                    />
                  </Link>
                  <Link href="https://discord.gg/YzBTfMxVQK" isExternal>
                    <Icon
                      as={FaDiscord}
                      boxSize={5}
                      _hover={{ color: "white" }}
                    />
                  </Link>
                  <Link href="mailto:support@tensara.ai" isExternal>
                    <Icon
                      as={FaEnvelope}
                      boxSize={5}
                      _hover={{ color: "white" }}
                    />
                  </Link>
                </Flex>
              </VStack>

              {/* Navigation */}
              <VStack align="flex-start" spacing={4}>
                <Heading size="sm" color="white" mb={2}>
                  Navigate
                </Heading>
                <Link href="/problems" _hover={{ color: "white" }}>
                  Problems
                </Link>
                <Link href="/leaderboard" _hover={{ color: "white" }}>
                  Leaderboards
                </Link>
                <Link href="/contests" _hover={{ color: "white" }}>
                  Contests
                </Link>
                <Link href="/blog" _hover={{ color: "white" }}>
                  Blog
                </Link>
              </VStack>

              {/* Resources */}
              <VStack align="flex-start" spacing={4}>
                <Heading size="sm" color="white" mb={2}>
                  Resources
                </Heading>
                <Link
                  href="https://discord.gg/YzBTfMxVQK"
                  isExternal
                  _hover={{ color: "white" }}
                >
                  Learn
                </Link>
                <Link
                  href="https://github.com/orgs/tensara/projects/1"
                  isExternal
                  _hover={{ color: "white" }}
                >
                  Roadmap
                </Link>
                <Link
                  href="https://github.com/tensara/tensara-cli"
                  isExternal
                  _hover={{ color: "white" }}
                >
                  CLI Tool
                </Link>
              </VStack>

              {/* Legal */}
              <VStack align="flex-start" spacing={4}>
                <Heading size="sm" color="white" mb={2}>
                  Legal
                </Heading>
                <Link href="/terms" _hover={{ color: "white" }}>
                  Terms of Service
                </Link>
                <Link href="/privacy" _hover={{ color: "white" }}>
                  Privacy Policy
                </Link>
              </VStack>
            </SimpleGrid>
            <Divider my={10} borderColor="whiteAlpha.200" />
            <Text textAlign="center" fontSize="sm">
              &copy; {new Date().getFullYear()} Tensara. All rights reserved.
            </Text>
          </Container>
        </Box>
      </Box>
    </Layout>
  );
}
