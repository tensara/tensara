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
} from "@chakra-ui/react";
import { motion } from "framer-motion";
import { Layout } from "~/components/layout";
import { FiCpu, FiAward, FiUsers, FiCode, FiGithub, FiTwitter, FiMail } from "react-icons/fi";
import { useSession, signIn } from "next-auth/react";
import { type IconType } from "react-icons";

// Create motion components
const MotionVStack = motion(VStack);
const MotionBox = motion(Box);
const MotionSimpleGrid = motion(SimpleGrid);
const MotionHStack = motion(HStack);

export default function HomePage() {
  const { data: session } = useSession();

  const handleStartSolving = () => {
    if (!session) {
      signIn("github").catch(console.error);
    } else {
      window.location.href = "/problems";
    }
  };

  return (
    <Layout title="Home">
      <Container maxW="8xl">
        {/* Hero Section */}
        <MotionVStack
          spacing={8}
          py={16}
          textAlign="center"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
        >
          <Heading
            as="h1"
            fontSize="5rem"
            fontWeight="semibold"
            letterSpacing="tight"
            fontFamily="Space Grotesk, sans-serif"
          >
            Optimize, Benchmark, Repeat
          </Heading>

          <Text
            fontSize="2xl"
            color="whiteAlpha.900"
            maxW="3xl"
            lineHeight="tall"
          >
            A platform for GPU programming challenges. Write efficient CUDA code and
            compare your solutions with other developers.
          </Text>

          <Button
            onClick={handleStartSolving}
            size="lg"
            height="16"
            px="8"
            fontSize="lg"
            bg="#0e8144"
            color="white"
            leftIcon={<FiCode size={24} />}
            _hover={{
              transform: "translateY(-2px)",
              boxShadow: "xl",
              bg: "#0a6434",
            }}
            transition="all 0.2s"
          >
            Start solving
          </Button>

          {/* Product Screenshot */}
          <MotionBox
            w="full"
            maxW="7xl"
            mt={8}
            position="relative"
            initial={{ opacity: 0, y: 40 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.2 }}
          >
            <Box
              as="video"
              src="/demo.mp4"
              autoPlay
              loop
              muted
              playsInline
              w="full"
              boxShadow="2xl"
              borderRadius="xl"
              objectFit="cover"
            />
          </MotionBox>
        </MotionVStack>

        {/* Features Grid */}
        <MotionSimpleGrid
          columns={{ base: 1, md: 3 }}
          spacing={10}
          py={20}
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
            description="Battle-tested problems that push your CUDA optimization skills to the limit."
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
        </MotionSimpleGrid>

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
              <Link href="/problems" color="whiteAlpha.800" _hover={{ color: "#2ecc71" }}>
                Problems
              </Link>
              <Link href="/submissions" color="whiteAlpha.800" _hover={{ color: "#2ecc71" }}>
                Submissions
              </Link>
              <Link href="/blog" color="whiteAlpha.800" _hover={{ color: "#2ecc71" }}>
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
                  <Icon as={FiGithub} boxSize={6} />
                </Link>
                <Link 
                  href="https://twitter.com/someshkar" 
                  isExternal
                  color="whiteAlpha.800"
                  _hover={{ color: "#2ecc71" }}
                >
                  <Icon as={FiTwitter} boxSize={6} />
                </Link>
                <Link 
                  href="mailto:hello@tensara.org"
                  color="whiteAlpha.800"
                  _hover={{ color: "#2ecc71" }}
                >
                  <Icon as={FiMail} boxSize={6} />
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
      bg="rgba(14, 129, 68, 0.1)"
      p={8}
      borderRadius="xl"
      spacing={4}
      align="start"
      style={{ transition: "all 0.2s" }}
      _hover={{ transform: "translateY(-4px)", bg: "rgba(14, 129, 68, 0.15)" }}
      variants={{
        hidden: { opacity: 0, y: 20 },
        visible: { opacity: 1, y: 0, transition: { duration: 0.6 } },
      }}
    >
      <Icon as={icon} boxSize={8} color="#2ecc71" />
      <Heading size="md">{title}</Heading>
      <Text color="whiteAlpha.900">{description}</Text>
    </MotionVStack>
  );
}
