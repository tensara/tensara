import { type NextPage } from "next";
import {
  Box,
  Container,
  Heading,
  Text,
  SimpleGrid,
  Card,
  CardBody,
  CardHeader,
  Flex,
  Icon,
  Link as ChakraLink,
  VStack,
  Button,
} from "@chakra-ui/react";
import Link from "next/link";
import { useRouter } from "next/router";
import { Layout } from "~/components/layout";
import { FiAward, FiCpu } from "react-icons/fi";
import { motion } from "framer-motion";
import { gpuPuzzles, type Puzzle } from "./sample";

// Create motion components
const MotionBox = motion(Box);
const MotionSimpleGrid = motion(SimpleGrid);
const MotionCard = motion(Card);

const PuzzleCard = ({ puzzle }: { puzzle: Puzzle }) => {
  const router = useRouter();

  return (
    <MotionCard
      bg="rgba(20, 40, 60, 0.7)"
      borderRadius="xl"
      overflow="hidden"
      boxShadow="0 4px 20px rgba(0, 0, 0, 0.3)"
      borderWidth="1px"
      borderColor="rgba(45, 85, 125, 0.5)"
      position="relative"
      whileHover={{
        y: -8,
        boxShadow: "0 12px 30px rgba(0, 0, 0, 0.4)",
        borderColor: "rgba(46, 204, 113, 0.3)",
      }}
      transition={{ duration: 0.3 }}
      height="100%"
      cursor="pointer"
      onClick={() => router.push(`/learn/${puzzle.id}`)}
    >
      <CardHeader>
        <Flex justifyContent="space-between" alignItems="flex-start">
          <Heading
            size="md"
            fontFamily="Space Grotesk, sans-serif"
            color="white"
          >
            {puzzle.title}
          </Heading>
        </Flex>
      </CardHeader>
      <CardBody>
        <VStack spacing={4} align="start">
          <Text color="whiteAlpha.900" minH={12}>
            {puzzle.description}
          </Text>
          <Flex
            w="100%"
            justifyContent="space-between"
            alignItems="center"
            mt={2}
          >
            <Text color="whiteAlpha.700" fontSize="sm">
              by {puzzle.author}
            </Text>
          </Flex>
        </VStack>
      </CardBody>
    </MotionCard>
  );
};

const LearnIndexPage: NextPage = () => {
  return (
    <Layout title="Learn GPU Programming">
      {/* Hero Section */}
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

        {/* Animated Pattern */}
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
          maxW="7xl"
          px={{ base: 4, md: 8 }}
          pt={{ base: 12, md: 16 }}
          pb={{ base: 12, md: 16 }}
        >
          <MotionBox
            textAlign="center"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
          >
            <Flex
              direction="column"
              align="center"
              justify="center"
              maxW="3xl"
              mx="auto"
            >
              <Box
                p={2}
                bg="rgba(14, 129, 68, 0.2)"
                borderRadius="lg"
                boxShadow="0 0 10px rgba(46, 204, 113, 0.1)"
                mb={4}
              >
                <Icon as={FiCpu} boxSize={8} color="#2ecc71" />
              </Box>

              <Heading
                as="h1"
                fontSize={{ base: "3xl", md: "4xl" }}
                fontWeight="semibold"
                letterSpacing="tight"
                fontFamily="Space Grotesk, sans-serif"
                lineHeight={{ base: "1.2", md: "1.1" }}
                color="white"
                mb={4}
              >
                GPU Puzzles
              </Heading>

              <Text
                fontSize={{ base: "lg", md: "xl" }}
                color="whiteAlpha.900"
                lineHeight="tall"
                mb={4}
                textAlign="center"
              >
                Inspired by{" "}
                <ChakraLink
                  href="http://rush-nlp.com"
                  isExternal
                  color="green.400"
                >
                  Sasha Rush
                </ChakraLink>{" "}
                -
                <ChakraLink
                  href="https://twitter.com/srush_nlp"
                  isExternal
                  color="green.400"
                  ml={1}
                >
                  @srush_nlp
                </ChakraLink>
              </Text>

              <Text
                fontSize={{ base: "md", md: "lg" }}
                color="whiteAlpha.900"
                lineHeight="tall"
                textAlign="center"
              >
                Learn GPU programming by solving hands-on puzzles.
                <br />
                This interactive learning path will take you from basic CUDA
                kernels to implementing the algorithms that power modern deep
                learning.
              </Text>
            </Flex>
          </MotionBox>
        </Container>
      </Box>

      <Container maxW="7xl" px={{ base: 4, md: 8 }} pb={16}>
        {/* Filter tabs could go here */}

        {/* Puzzle Grid */}
        <MotionSimpleGrid
          columns={{ base: 1, md: 2, lg: 3 }}
          spacing={8}
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.5, delay: 0.2 }}
        >
          {Object.values(gpuPuzzles).map((puzzle) => (
            <PuzzleCard key={puzzle.id} puzzle={puzzle} />
          ))}
        </MotionSimpleGrid>

        {/* CTA Section */}
        <MotionBox
          mt={{ base: 12, md: 16 }}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.4 }}
        >
          <Box
            bg="rgba(14, 129, 68, 0.15)"
            borderRadius="2xl"
            p={{ base: 6, md: 8 }}
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

            <VStack spacing={5} position="relative" zIndex={1}>
              <Icon as={FiAward} boxSize={8} color="#2ecc71" />
              <Heading fontSize={{ base: "xl", md: "2xl" }} color="white">
                Ready to apply what you&apos;ve learned?
              </Heading>
              <Text fontSize="lg" color="whiteAlpha.900" maxW="2xl">
                After completing these puzzles, challenge yourself with our
                competitive GPU programming problems and see how your
                optimizations stack up on the leaderboard.
              </Text>
              <Button
                as={Link}
                href="/problems"
                size="lg"
                height="14"
                px="8"
                fontSize="md"
                bg="#0e8144"
                color="white"
                _hover={{
                  transform: "translateY(-2px)",
                  boxShadow: "0 0 20px rgba(46, 204, 113, 0.4)",
                  bg: "#0a6434",
                }}
                transition="all 0.3s"
              >
                Explore Tensara Problems
              </Button>
            </VStack>
          </Box>
        </MotionBox>
      </Container>
    </Layout>
  );
};

export default LearnIndexPage;
