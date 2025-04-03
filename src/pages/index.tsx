import {
  Box,
  Button,
  Container,
  Heading,
  Text,
  VStack,
  Icon,
  Image,
  SimpleGrid,
  HStack,
  Link,
  Divider,
  Badge,
  Flex,
  useBreakpointValue,
  Card,
  CardBody,
  Stack,
  Grid,
  GridItem,
} from "@chakra-ui/react";
import { motion } from "framer-motion";
import React, { useState, useEffect } from "react";
import { Layout } from "~/components/layout";
import {
  FiCpu,
  FiAward,
  FiUsers,
  FiCode,
  FiGitPullRequest,
  FiPackage,
  FiTerminal,
  FiZap,
} from "react-icons/fi";
import {
  FaDiscord,
  FaGithub,
  FaTwitter,
  FaEnvelope,
  FaCode,
} from "react-icons/fa";
import { type IconType } from "react-icons";

// Create motion components
const MotionVStack = motion(VStack);
const MotionBox = motion(Box);
const MotionSimpleGrid = motion(SimpleGrid);
const MotionFlex = motion(Flex);
const MotionCard = motion(Card);
const MotionGrid = motion(Grid);

// Define update types with different colors
const updateTypes = {
  FEATURE: { color: "green.400", label: "Feature" },
  IMPROVEMENT: { color: "blue.400", label: "Improvement" },
  FIX: { color: "orange.400", label: "Fix" },
  RELEASE: { color: "purple.400", label: "Release" },
};

export default function HomePage() {
  // const { data: session } = useSession();
  const [videoLoaded, setVideoLoaded] = useState(false);
  const [isMobile, setIsMobile] = useState(false);

  useEffect(() => {
    const checkMobile = () => {
      setIsMobile(window.innerWidth <= 768);
    };

    checkMobile();
    window.addEventListener("resize", checkMobile);

    return () => window.removeEventListener("resize", checkMobile);
  }, []);

  const updateBadgeSize = useBreakpointValue({ base: "sm", md: "md" });

  // Add this function to handle smooth scrolling
  const scrollToSection = (id: string) => (e: React.MouseEvent) => {
    e.preventDefault();
    const element = document.getElementById(id);
    if (element) {
      element.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  };

  return (
    <Layout title="Home" ogImage="/tensara_ogimage.png">
      {/* Hero Section with Background */}
      <Box position="relative" overflow="hidden" mb={10}>
        {/* Background Graphic */}
        <Box
          position="absolute"
          top="0"
          left="0"
          right="0"
          bottom="0"
          bgGradient="linear(to-b, rgba(15, 23, 42, 0.9), rgba(15, 23, 42, 1))"
          zIndex={-1}
        />

        {/* Animated Particles */}
        <Box
          position="absolute"
          top="0"
          left="0"
          right="0"
          bottom="0"
          opacity="0.2"
          backgroundImage="url('/grid-pattern.svg')"
          backgroundSize="cover"
          zIndex={-1}
        />

        <Container
          maxW="8xl"
          px={{ base: 4, md: 8 }}
          pt={{ base: 16, md: 20 }}
          pb={{ base: 16, md: 24 }}
        >
          <MotionVStack
            spacing={{ base: 8, md: 10 }}
            textAlign="center"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
          >
            <Heading
              as="h1"
              fontSize={{ base: "3.5rem", md: "5rem", lg: "5rem" }}
              fontWeight="semibold"
              letterSpacing="tight"
              fontFamily="Space Grotesk, sans-serif"
              lineHeight={{ base: "1.2", md: "1.1" }}
              color="white"
              position="relative"
              _before={{
                content: '""',
                position: "absolute",
                left: "50%",
                transform: "translateX(-50%)",
                bottom: "-10px",
                height: "4px",
                width: "80px",
                bg: "#0e8144",
                borderRadius: "full",
              }}
            >
              Optimize, Benchmark, Repeat
            </Heading>

            <Text
              fontSize={{ base: "xl", md: "2xl" }}
              color="whiteAlpha.900"
              maxW="3xl"
              lineHeight="tall"
              px={{ base: 4, md: 0 }}
            >
              A platform for GPU programming challenges. Write efficient GPU
              kernels and compare your solutions with other developers.
            </Text>

            <Flex
              direction={{ base: "column", md: "row" }}
              gap={6}
              w="100%"
              justifyContent="center"
            >
              <Button
                as="a"
                href="/problems"
                size={{ base: "lg", md: "lg" }}
                height={{ base: "14", md: "16" }}
                px={{ base: "8", md: "10" }}
                fontSize={{ base: "md", md: "lg" }}
                bg="#0e8144"
                color="white"
                leftIcon={<FiCode size={24} />}
                _hover={{
                  transform: "translateY(-2px)",
                  boxShadow: "0 0 20px rgba(46, 204, 113, 0.4)",
                  bg: "#0a6434",
                }}
                transition="all 0.3s"
                w={{ base: "full", md: "auto" }}
              >
                Start solving
              </Button>

              <Button
                onClick={() =>
                  window.open("https://github.com/tensara/tensara", "_blank")
                }
                variant="outline"
                size={{ base: "lg", md: "lg" }}
                height={{ base: "14", md: "16" }}
                px={{ base: "8", md: "10" }}
                fontSize={{ base: "md", md: "lg" }}
                borderColor="whiteAlpha.400"
                borderWidth="2px"
                color="white"
                leftIcon={<FaGithub size={24} />}
                _hover={{
                  transform: "translateY(-2px)",
                  boxShadow: "0 0 20px rgba(255, 255, 255, 0.2)",
                  borderColor: "white",
                  bg: "whiteAlpha.100",
                }}
                transition="all 0.3s"
                w={{ base: "full", md: "auto" }}
                cursor="pointer"
              >
                Contribute!
              </Button>
            </Flex>

            {/* Product Screenshot */}
            <MotionBox
              w="full"
              maxW="7xl"
              mt={{ base: 8, md: 12 }}
              position="relative"
              initial={{ opacity: 0, y: 40 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8, delay: 0.2 }}
            >
              {/* Static Image for Mobile / Video Loading */}
              <Box
                position="relative"
                borderRadius="2xl"
                overflow="hidden"
                boxShadow="0 0 40px rgba(0, 0, 0, 0.3)"
              >
                <Image
                  src="/demo-poster.jpg"
                  alt="Demo preview"
                  w="full"
                  objectFit="cover"
                  display={isMobile || !videoLoaded ? "block" : "none"}
                />
                {/* Video only for desktop */}
                {!isMobile && (
                  <Box
                    as="video"
                    src="/demo.mp4"
                    poster="/demo-poster.jpg"
                    preload="auto"
                    autoPlay
                    loop
                    muted
                    playsInline
                    w="full"
                    objectFit="cover"
                    display={videoLoaded ? "block" : "none"}
                    onLoadedData={() => setVideoLoaded(true)}
                  />
                )}

                {/* Overlay gradient */}
                <Box
                  position="absolute"
                  top="0"
                  left="0"
                  right="0"
                  bottom="0"
                  bgGradient="linear(to-t, rgba(15, 23, 42, 0.7) 0%, rgba(15, 23, 42, 0) 40%)"
                  pointerEvents="none"
                />
              </Box>
            </MotionBox>
          </MotionVStack>
        </Container>
      </Box>

      <Container maxW="8xl" px={{ base: 4, md: 8 }}>
        {/* Features Grid - Main Highlights */}
        <Box py={{ base: 10, md: 16 }} position="relative">
          <MotionGrid
            templateColumns={{ base: "1fr", md: "repeat(3, 1fr)" }}
            gap={{ base: 8, md: 10 }}
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true, margin: "-100px" }}
            variants={{
              visible: {
                transition: {
                  staggerChildren: 0.2,
                },
              },
            }}
          >
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
          </MotionGrid>
        </Box>

        {/* Recent Updates Section */}
        <Box id="updates" py={{ base: 14, md: 20 }} scrollMarginTop="100px">
          <MotionVStack
            spacing={10}
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            transition={{ duration: 0.6 }}
            viewport={{ once: true, margin: "-100px" }}
          >
            <Box textAlign="center" mb={4}>
              <Heading
                fontSize={{ base: "2xl", md: "4xl" }}
                fontFamily="Space Grotesk, sans-serif"
                color="white"
                position="relative"
                _after={{
                  content: '""',
                  position: "absolute",
                  width: "60px",
                  height: "4px",
                  bottom: "-10px",
                  left: "50%",
                  transform: "translateX(-50%)",
                  bg: "#0e8144",
                  borderRadius: "full",
                }}
              >
                Recent Platform Updates
              </Heading>
              <Text
                mt={6}
                fontSize="lg"
                color="whiteAlpha.800"
                maxW="3xl"
                mx="auto"
              >
                Check out the latest improvements across all Tensara projects
              </Text>
            </Box>

            <SimpleGrid
              columns={{ base: 1, md: 3 }}
              spacing={{ base: 6, md: 8 }}
              width="100%"
            >
              <UpdateCard
                icon={FiGitPullRequest}
                title="Core Platform"
                repo="tensara/tensara"
                updates={[
                  { type: "FEATURE", text: "Triton submission support" },
                  { type: "FEATURE", text: "Rating & leaderboard system" },
                  {
                    type: "IMPROVEMENT",
                    text: "Rate limiting & error handling",
                  },
                  { type: "FEATURE", text: "Test result streaming with SSE" },
                ]}
              />

              <UpdateCard
                icon={FiCpu}
                title="Problem Repository"
                repo="tensara/problems"
                updates={[
                  { type: "FEATURE", text: "PyTorch-based test cases" },
                  {
                    type: "FEATURE",
                    text: "3D/4D Tensor matmul problems ",
                  },
                  { type: "IMPROVEMENT", text: "Python problem definitions" },
                  {
                    type: "FEATURE",
                    text: "Sigmoid, Tanh, and more activation functions",
                  },
                ]}
              />

              <UpdateCard
                icon={FiTerminal}
                title="CLI Tool"
                repo="tensara/tensara-cli"
                updates={[
                  { type: "FEATURE", text: "CLI interface for platform" },
                  { type: "FEATURE", text: "GitHub authentication system" },
                  { type: "IMPROVEMENT", text: "Pretty printing for results" },
                  { type: "RELEASE", text: "Version 0.2.0 released" },
                ]}
              />
            </SimpleGrid>
          </MotionVStack>
        </Box>

        {/* Joining CTA */}
        <MotionBox
          mt={{ base: 10, md: 16 }}
          mb={{ base: 20, md: 28 }}
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          viewport={{ once: true }}
        >
          <Box
            bg="rgba(14, 129, 68, 0.15)"
            borderRadius="2xl"
            p={{ base: 8, md: 12 }}
            textAlign="center"
            position="relative"
            overflow="hidden"
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

            <VStack spacing={6} position="relative" zIndex={1}>
              <Icon as={FiZap} boxSize={10} color="#2ecc71" />
              <Heading fontSize={{ base: "2xl", md: "3xl" }} color="white">
                Ready to push your GPU skills further?
              </Heading>
              <Text fontSize="lg" color="whiteAlpha.900" maxW="2xl">
                Join our growing community of GPU programmers and tackle
                real-world performance challenges.
              </Text>
              <Button
                as="a"
                href="/problems"
                size="lg"
                height="16"
                px="10"
                fontSize="lg"
                bg="#0e8144"
                color="white"
                _hover={{
                  transform: "translateY(-2px)",
                  boxShadow: "0 0 20px rgba(46, 204, 113, 0.4)",
                  bg: "#0a6434",
                }}
                transition="all 0.3s"
              >
                Explore Problems
              </Button>
            </VStack>
          </Box>
        </MotionBox>

        {/* Footer */}
        <MotionBox
          as="footer"
          py={16}
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
          transition={{ duration: 0.8 }}
        >
          <Divider mb={10} />
          <SimpleGrid columns={{ base: 1, md: 3 }} spacing={8}>
            {/* Company Info */}
            <VStack align="start" spacing={4}>
              <Heading size="md" fontFamily="Space Grotesk, sans-serif">
                Tensara
              </Heading>
              <Text color="whiteAlpha.800">
                Write efficient GPU code and compete with other developers.
              </Text>
            </VStack>

            {/* Quick Links */}
            <VStack align="start" spacing={4}>
              <Heading size="md" fontFamily="Space Grotesk, sans-serif">
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
                href="/blog"
                color="whiteAlpha.800"
                _hover={{ color: "#2ecc71" }}
              >
                Blog
              </Link>
            </VStack>

            {/* Social Links */}
            <VStack align="start" spacing={4}>
              <Heading size="md" fontFamily="Space Grotesk, sans-serif">
                Connect
              </Heading>
              <HStack spacing={4}>
                <Link
                  href="https://github.com/tensara"
                  isExternal
                  color="whiteAlpha.800"
                  _hover={{ color: "#2ecc71" }}
                >
                  <Icon as={FaGithub} boxSize={6} />
                </Link>
                <Link
                  href="https://twitter.com/tensarahq"
                  isExternal
                  color="whiteAlpha.800"
                  _hover={{ color: "#2ecc71" }}
                >
                  <Icon as={FaTwitter} boxSize={6} />
                </Link>
                <Link
                  href="mailto:hello@tensara.org"
                  color="whiteAlpha.800"
                  _hover={{ color: "#2ecc71" }}
                >
                  <Icon as={FaEnvelope} boxSize={6} />
                </Link>
                <Link
                  href="https://discord.gg/YzBTfMxVQK"
                  isExternal
                  color="whiteAlpha.800"
                  _hover={{ color: "#2ecc71" }}
                >
                  <Icon as={FaDiscord} boxSize={6} />
                </Link>
              </HStack>
              <Text color="whiteAlpha.600" fontSize="sm" mt={4}>
                © {new Date().getFullYear()} Tensara. All rights reserved.
              </Text>
            </VStack>
          </SimpleGrid>
        </MotionBox>
      </Container>
    </Layout>
  );
}

function FeatureCard({
  icon,
  title,
  description,
}: {
  icon: IconType;
  title: string;
  description: string;
}) {
  return (
    <MotionVStack
      bg="rgba(20, 40, 60, 0.6)"
      p={8}
      borderRadius="xl"
      spacing={5}
      align="start"
      backdropFilter="blur(10px)"
      borderWidth="1px"
      borderColor="rgba(46, 204, 113, 0.2)"
      boxShadow="0 4px 20px rgba(0, 0, 0, 0.3)"
      position="relative"
      overflow="hidden"
      _before={{
        content: '""',
        position: "absolute",
        top: 0,
        left: 0,
        width: "6px",
        height: "60%",
        bg: "#0e8144",
        borderTopLeftRadius: "xl",
      }}
      _hover={{
        transform: "translateY(-8px)",
        boxShadow: "0 12px 30px rgba(0, 0, 0, 0.4)",
        borderColor: "rgba(46, 204, 113, 0.4)",
      }}
      style={{ transition: "all 0.3s ease" }}
      variants={{
        hidden: { opacity: 0, y: 20 },
        visible: { opacity: 1, y: 0, transition: { duration: 0.6 } },
      }}
    >
      <Flex w="full" align="center" justify="space-between">
        <Box
          p={2}
          bg="rgba(14, 129, 68, 0.2)"
          borderRadius="lg"
          boxShadow="0 0 10px rgba(46, 204, 113, 0.1)"
        >
          <Icon as={icon} boxSize={7} color="#2ecc71" />
        </Box>
      </Flex>
      <Heading size="md" mt={2}>
        {title}
      </Heading>
      <Text color="whiteAlpha.900">{description}</Text>
    </MotionVStack>
  );
}

interface UpdateItem {
  type: keyof typeof updateTypes;
  text: string;
}

function UpdateCard({
  icon,
  title,
  repo,
  updates,
}: {
  icon: IconType;
  title: string;
  repo: string;
  updates: UpdateItem[];
}) {
  return (
    <MotionCard
      bg="rgba(20, 40, 60, 0.7)"
      borderRadius="xl"
      overflow="hidden"
      boxShadow="0 4px 20px rgba(0, 0, 0, 0.3)"
      borderWidth="1px"
      borderColor="rgba(45, 85, 125, 0.5)"
      position="relative"
      variants={{
        hidden: { opacity: 0, y: 20 },
        visible: { opacity: 1, y: 0, transition: { duration: 0.6 } },
      }}
      initial="hidden"
      whileInView="visible"
      viewport={{ once: true }}
      whileHover={{
        y: -8,
        boxShadow: "0 12px 30px rgba(0, 0, 0, 0.4)",
        borderColor: "rgba(46, 204, 113, 0.3)",
      }}
      transition={{ duration: 0.3 }}
    >
      <CardBody>
        <VStack align="start" spacing={4}>
          <Flex w="100%" align="center" justify="space-between">
            <HStack spacing={3}>
              <Box
                p={2}
                bg="rgba(14, 129, 68, 0.2)"
                borderRadius="lg"
                boxShadow="0 0 10px rgba(46, 204, 113, 0.1)"
                width="36px"
                height="36px"
                display="flex"
                alignItems="center"
                justifyContent="center"
              >
                <Icon as={icon} boxSize={5} color="#2ecc71" />
              </Box>
              <Heading
                size="md"
                fontFamily="Space Grotesk, sans-serif"
                color="white"
              >
                {title}
              </Heading>
            </HStack>
            <Link
              href={`https://github.com/${repo}`}
              isExternal
              p={2}
              borderRadius="full"
              transition="all 0.2s"
            >
              <Icon color="white" as={FaGithub} boxSize={5} />
            </Link>
          </Flex>

          <Divider opacity={0.3} />

          <Stack spacing={3} width="100%">
            {updates.map((update, index) => (
              <Flex key={index} align="center" width="100%">
                <Badge
                  colorScheme={updateTypes[update.type].color.split(".")[0]}
                  px={2}
                  py={1}
                  borderRadius="md"
                  minW="90px"
                  textAlign="center"
                  fontSize="xs"
                  mr={3}
                  fontWeight="semibold"
                  letterSpacing="0.4px"
                  textTransform="uppercase"
                >
                  {updateTypes[update.type].label}
                </Badge>
                <Text fontSize="sm" color="whiteAlpha.900" noOfLines={1}>
                  {update.text}
                </Text>
              </Flex>
            ))}
          </Stack>
        </VStack>
      </CardBody>
    </MotionCard>
  );
}
