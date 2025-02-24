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
} from "@chakra-ui/react";
import { Layout } from "~/components/layout";
import { FiCpu, FiAward, FiUsers, FiCode } from "react-icons/fi";
import { useSession, signIn } from "next-auth/react";
import { type IconType } from "react-icons";

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
        <VStack spacing={8} py={16} textAlign="center">
          <Heading
            as="h1"
            fontSize="7rem"
            fontWeight="semibold"
            letterSpacing="tight"
          >
            Clash of the Kernels
          </Heading>

          <Text
            fontSize="2xl"
            color="whiteAlpha.900"
            maxW="3xl"
            lineHeight="tall"
          >
            The ultimate battleground for GPU programmers. Write, optimize, and
            benchmark your CUDA kernels against the community.
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
          <Box w="full" maxW="7xl" mt={8} position="relative">
            <Image
              src="/tensara_landing_ss_v2.jpg"
              alt="Tensara Platform Interface"
              w="full"
              loading="eager"
              boxShadow="2xl"
              borderRadius="xl"
            />
          </Box>
        </VStack>

        {/* Features Grid */}
        <SimpleGrid columns={{ base: 1, md: 3 }} spacing={10} py={20}>
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
        </SimpleGrid>
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
    <VStack
      bg="rgba(14, 129, 68, 0.1)"
      p={8}
      borderRadius="xl"
      spacing={4}
      align="start"
      transition="all 0.2s"
      _hover={{ transform: "translateY(-4px)", bg: "rgba(14, 129, 68, 0.15)" }}
    >
      <Icon as={icon} boxSize={8} color="#2ecc71" />
      <Heading size="md">{title}</Heading>
      <Text color="whiteAlpha.900">{description}</Text>
    </VStack>
  );
}
