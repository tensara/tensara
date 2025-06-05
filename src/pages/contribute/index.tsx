import { type NextPage } from "next";
import Head from "next/head";
import { Layout } from "~/components/layout";
import {
  Box,
  Heading,
  Text,
  Grid,
  List,
  ListItem,
  ListIcon,
  Link as ChakraLink,
  useColorModeValue,
} from "@chakra-ui/react";
import { FaPlus, FaEye, FaEdit, FaCheck } from "react-icons/fa";
import PageHeader from "~/components/contribute/PageHeader";

const ContributeHomePage: NextPage = () => {
  const cardBg = useColorModeValue("white", "gray.800");
  const cardBorder = useColorModeValue("gray.200", "gray.700");
  const buttonBg = useColorModeValue("blue.500", "blue.200");
  const buttonHoverBg = useColorModeValue("blue.600", "blue.300");
  const iconBg = useColorModeValue("whiteAlpha.800", "whiteAlpha.200");

  return (
    <Layout>
      <Head>
        <title>Contribute | Tensara</title>
        <meta name="description" content="Contribute problems to Tensara" />
      </Head>

      <Box maxW="6xl" mx="auto" px={4} py={8}>
        <PageHeader
          title="Contribute to Tensara"
          description="Help expand our problem library by submitting new computational challenges"
        />

        <Grid templateColumns={{ base: "1fr", md: "1fr 1fr" }} gap={8} my={10}>
          <Box
            bg={cardBg}
            borderWidth="1px"
            borderColor={cardBorder}
            borderRadius="xl"
            p={6}
            boxShadow="md"
          >
            <Heading as="h2" size="lg" mb={4} color="blue.500">
              What is a Problem?
            </Heading>
            <Text mb={4}>
              A problem in Tensara is a computational challenge that requires an
              optimized solution, typically targeting GPU acceleration. Each
              problem consists of:
            </Text>
            <List spacing={2}>
              <ListItem>
                <ListIcon as={FaCheck} color="green.500" />A clear problem
                description
              </ListItem>
              <ListItem>
                <ListIcon as={FaCheck} color="green.500" />
                Input/output specifications
              </ListItem>
              <ListItem>
                <ListIcon as={FaCheck} color="green.500" />A reference
                implementation
              </ListItem>
              <ListItem>
                <ListIcon as={FaCheck} color="green.500" />
                Test cases for validation
              </ListItem>
            </List>
          </Box>

          <Box
            bg={cardBg}
            borderWidth="1px"
            borderColor={cardBorder}
            borderRadius="xl"
            p={6}
            boxShadow="md"
          >
            <Heading as="h2" size="lg" mb={4} color="blue.500">
              How to Contribute
            </Heading>
            <List spacing={3}>
              <ListItem display="flex" alignItems="flex-start">
                <ListIcon as={FaPlus} color="blue.500" mt={1} />
                <Box>
                  <ChakraLink
                    href="/contribute/add"
                    color="blue.500"
                    fontWeight="medium"
                  >
                    Submit a new problem
                  </ChakraLink>{" "}
                  using our contribution form
                </Box>
              </ListItem>
              <ListItem display="flex" alignItems="flex-start">
                <ListIcon as={FaCheck} color="green.500" mt={1} />
                <Box>
                  Our team will review your submission through a GitHub pull
                  request
                </Box>
              </ListItem>
              <ListItem display="flex" alignItems="flex-start">
                <ListIcon as={FaCheck} color="green.500" mt={1} />
                <Box>
                  Once approved, your problem will be added to Tensara&apos;s
                  problem library
                </Box>
              </ListItem>
              <ListItem display="flex" alignItems="flex-start">
                <ListIcon as={FaEye} color="blue.500" mt={1} />
                <Box>
                  Track your contributions in the{" "}
                  <ChakraLink
                    href="/contributions/view"
                    color="blue.500"
                    fontWeight="medium"
                  >
                    contributions dashboard
                  </ChakraLink>
                </Box>
              </ListItem>
            </List>
          </Box>
        </Grid>

        <Grid
          templateColumns={{ base: "1fr", md: "repeat(3, 1fr)" }}
          gap={6}
          mt={8}
        >
          <ChakraLink
            href="/contribute/add"
            _hover={{ textDecoration: "none" }}
          >
            <Box
              w="100%"
              bg="blue.600"
              _hover={{ bg: "blue.700" }}
              color="white"
              borderRadius="lg"
              p={6}
              display="flex"
              flexDirection="column"
              alignItems="center"
              textAlign="center"
            >
              <Box bg={iconBg} borderRadius="full" p={3} mb={4}>
                <FaPlus size={24} color={buttonBg} />
              </Box>
              <Heading as="h3" size="md" mb={2}>
                Add New Problem
              </Heading>
              <Text fontSize="sm" opacity={0.9}>
                Submit a new problem to Tensara
              </Text>
            </Box>
          </ChakraLink>

          <ChakraLink
            href="/contribute/view"
            _hover={{ textDecoration: "none" }}
          >
            <Box
              w="100%"
              bg="green.500"
              _hover={{ bg: "green.600" }}
              color="white"
              borderRadius="lg"
              p={6}
              display="flex"
              flexDirection="column"
              alignItems="center"
              textAlign="center"
            >
              <Box bg={iconBg} borderRadius="full" p={3} mb={4}>
                <FaEye size={24} color="green.500" />
              </Box>
              <Heading as="h3" size="md" mb={2}>
                View Contributions
              </Heading>
              <Text fontSize="sm" opacity={0.9}>
                See your submitted problems
              </Text>
            </Box>
          </ChakraLink>

          <ChakraLink
            href="/contribute/modify"
            _hover={{ textDecoration: "none" }}
          >
            <Box
              w="100%"
              bg="purple.500"
              _hover={{ bg: "purple.600" }}
              color="white"
              borderRadius="lg"
              p={6}
              display="flex"
              flexDirection="column"
              alignItems="center"
              textAlign="center"
            >
              <Box bg={iconBg} borderRadius="full" p={3} mb={4}>
                <FaEdit size={24} color="purple.500" />
              </Box>
              <Heading as="h3" size="md" mb={2}>
                Modify Contributions
              </Heading>
              <Text fontSize="sm" opacity={0.9}>
                Edit your existing problems
              </Text>
            </Box>
          </ChakraLink>
        </Grid>
      </Box>
    </Layout>
  );
};

export default ContributeHomePage;
