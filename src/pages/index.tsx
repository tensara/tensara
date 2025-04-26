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
import { motion } from "framer-motion";
import React, { useState, useEffect } from "react";
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
        <Badge colorScheme={getBadgeColor(type)} fontSize="xs" borderRadius="md" px={2} py={1}>
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

  useEffect(() => {
    const checkMobile = () => {
      setIsMobile(window.innerWidth <= 768);
    };

    checkMobile();
    window.addEventListener("resize", checkMobile);

    return () => window.removeEventListener("resize", checkMobile);
  }, []);

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
          overflow="hidden"
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
                    Want to code from your own IDE? Check out the{" "}
                    <Link
                      href="https://github.com/tensara/tensara-cli"
                      isExternal
                      color="#2ecc71"
                      textDecoration="underline"
                    >
                      CLI tool
                    </Link>
                  </Text>
                </MotionFlex>
              </MotionVStack>
              <AnimatedCudaEditor />
            </Flex>
          </Container>
        </MotionBox>

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
                      description="Allows users to submit to the leaderboard from the CLI."
                      date="4 days ago"
                    />
                    <UpdateCard
                      title="Authentication"
                      type="FEATURE"
                      description="GitHub authentication via CLI."
                      date="1 week ago"
                    />
                    <UpdateCard
                      title="v0.2.0 (Beta) Release"
                      type="RELEASE"
                      description="Released the beta version for the CLI tool."
                      date="2 weeks ago"
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
                      title="3D/4D Tensor matmul problems"
                      type="UPDATE"
                      description="Higher dimensional tensor matmul problems."
                      date="4 days ago"
                    />
                    <UpdateCard
                      title="Sigmoid, Tanh, and more activation"
                      type="UPDATE"
                      description="Added sigmoid, tanh, and more activation functions."
                      date="1 week ago"
                    />
                    <UpdateCard
                      title="Python problem definitions"
                      type="FEATURE"
                      description="Adding problems is easier, check our GitHub repo."
                      date="2 weeks ago"
                    />
                  </Box>
                </Box>
              </SimpleGrid>
            </VStack>
          </Container>
        </Box>

        {/* Feature Section */}
        <Container maxW="8xl" py={16}>
          <VStack align="flex-start" spacing={16}>
            <VStack align="flex-start" spacing={4} maxW="2xl">
              <Heading fontSize={{ base: "2xl", md: "3xl" }} color="white">
                A platform built for GPU programmers
              </Heading>
              <Text color="whiteAlpha.800" fontSize="lg">
                Focus on writing efficient kernels while we handle the
                infrastructure, benchmarking, and comparisons.
              </Text>
            </VStack>

            <SimpleGrid columns={{ base: 1, md: 3 }} spacing={8} w="full">
              <FeatureCard
                icon={FiCpu}
                title="Real GPU Challenges"
                description="Battle-tested problems that push your kernel optimization skills to the limit."
              />
              <FeatureCard
                icon={FiAward}
                title="Live Rankings"
                description="Compete for the highest GFLOPS and lowest kernel execution times on the leaderboard."
              />
              <FeatureCard
                icon={FiUsers}
                title="Global Competition"
                description="Join the ranks of GPU programmers worldwide and prove your parallel programming prowess."
              />
            </SimpleGrid>
          </VStack>
        </Container>

        {/* CTA Section */}
        <Container maxW="8xl" py={20}>
          <Box
            bg="rgba(14, 129, 68, 0.15)"
            borderRadius="2xl"
            p={{ base: 8, md: 12 }}
            position="relative"
            overflow="hidden"
            borderWidth="1px"
            borderColor="rgba(46, 204, 113, 0.3)"
            boxShadow="0 0 30px rgba(14, 129, 68, 0.15)"
          >
            {/* Background pattern */}
            <Box
              position="absolute"
              top="0"
              left="0"
              right="0"
              bottom="0"
              opacity="0.1"
              backgroundImage="url('/grid-pattern.svg')"
              backgroundSize="cover"
              zIndex={0}
            />

            <Flex
              direction={{ base: "column", md: "row" }}
              align="center"
              justify="space-between"
              gap={8}
              position="relative"
              zIndex={1}
            >
              <VStack
                align={{ base: "center", md: "flex-start" }}
                spacing={4}
                maxW={{ base: "full", md: "60%" }}
              >
                <Icon as={FiZap} boxSize={8} color="#2ecc71" />
                <Heading fontSize={{ base: "2xl", md: "3xl" }} color="white">
                  Ready to push your GPU skills further?
                </Heading>
                <Text fontSize="lg" color="whiteAlpha.900">
                  Join our growing community of GPU programmers and tackle
                  real-world performance challenges.
                </Text>
              </VStack>

              <Button
                size="lg"
                height="60px"
                px={8}
                fontSize="lg"
                bg="#0e8144"
                color="white"
                _hover={{
                  transform: "translateY(-2px)",
                  boxShadow: "0 0 20px rgba(46, 204, 113, 0.4)",
                  bg: "#0a6434",
                }}
                transition="all 0.3s"
                rightIcon={<Icon as={FiArrowRight} />}
              >
                Explore Problems
              </Button>
            </Flex>
          </Box>
        </Container>

        {/* Footer */}
        <Box bg="transparent" py={16} borderRadius="2xl">
          <Container maxW="8xl">
            <Divider mb={10} borderColor="whiteAlpha.200" />

            <SimpleGrid
              columns={{ base: 1, md: 3 }}
              spacing={8}
              justifyItems={"space-between"}
            >
              {/* Company Info */}
              <VStack align="flex-start" spacing={4}>
                <Heading
                  size="md"
                  fontFamily="Space Grotesk, sans-serif"
                  color="white"
                >
                  Tensara
                </Heading>
                <Text color="whiteAlpha.800">
                  Write efficient GPU code and compete with other developers.
                </Text>
                <Flex gap={4} mt={2}>
                  <Link
                    href="https://github.com/tensara"
                    isExternal
                    color="whiteAlpha.800"
                    _hover={{ color: "#2ecc71" }}
                  >
                    <Icon as={FaGithub} boxSize={5} />
                  </Link>
                  <Link
                    href="https://twitter.com/tensarahq"
                    isExternal
                    color="whiteAlpha.800"
                    _hover={{ color: "#2ecc71" }}
                  >
                    <Icon as={FaTwitter} boxSize={5} />
                  </Link>
                  <Link
                    href="mailto:hello@tensara.org"
                    color="whiteAlpha.800"
                    _hover={{ color: "#2ecc71" }}
                  >
                    <Icon as={FaEnvelope} boxSize={5} />
                  </Link>
                  <Link
                    href="https://discord.gg/YzBTfMxVQK"
                    isExternal
                    color="whiteAlpha.800"
                    _hover={{ color: "#2ecc71" }}
                  >
                    <Icon as={FaDiscord} boxSize={5} />
                  </Link>
                </Flex>
              </VStack>

              {/* Quick Links */}
              <VStack align="flex-start" spacing={4}>
                <Heading
                  size="md"
                  fontFamily="Space Grotesk, sans-serif"
                  color="white"
                >
                  Quick Links
                </Heading>
                <Link
                  href="/problems"
                  color="whiteAlpha.800"
                  _hover={{ color: "#2ecc71" }}
                >
                  Problems
                </Link>
                <Link
                  href="/submissions"
                  color="whiteAlpha.800"
                  _hover={{ color: "#2ecc71" }}
                >
                  Submissions
                </Link>
                <Link
                  href="/leaderboard"
                  color="whiteAlpha.800"
                  _hover={{ color: "#2ecc71" }}
                >
                  Leaderboard
                </Link>
                <Link
                  href="/blog"
                  color="whiteAlpha.800"
                  _hover={{ color: "#2ecc71" }}
                >
                  Blog
                </Link>
              </VStack>

              {/* Resources */}
              <VStack align="flex-start" spacing={4}>
                <Heading
                  size="md"
                  fontFamily="Space Grotesk, sans-serif"
                  color="white"
                >
                  Resources
                </Heading>
                <Link
                  href="/docs"
                  color="whiteAlpha.800"
                  _hover={{ color: "#2ecc71" }}
                >
                  Documentation
                </Link>
                <Link
                  href="/docs/api"
                  color="whiteAlpha.800"
                  _hover={{ color: "#2ecc71" }}
                >
                  API Reference
                </Link>
                <Link
                  href="/docs/tutorials"
                  color="whiteAlpha.800"
                  _hover={{ color: "#2ecc71" }}
                >
                  Tutorials
                </Link>
                <Link
                  href="https://github.com/tensara/tensara/issues"
                  isExternal
                  color="whiteAlpha.800"
                  _hover={{ color: "#2ecc71" }}
                >
                  <Flex align="center" gap={1}>
                    Report Issues
                    <Icon as={FiExternalLink} boxSize={3} />
                  </Flex>
                </Link>
              </VStack>
            </SimpleGrid>

            <Text color="whiteAlpha.600" fontSize="sm" mt={10}>
              © {new Date().getFullYear()} Tensara. All rights reserved.
            </Text>
          </Container>
        </Box>
      </Box>
    </Layout>
  );
}
