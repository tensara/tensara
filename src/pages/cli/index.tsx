import {
  Box,
  Flex,
  Heading,
  Tabs,
  TabList,
  Tab,
  TabPanels,
  TabPanel,
  Text,
  VStack,
  HStack,
  Code,
  Button,
  useDisclosure,
  Modal,
  ModalOverlay,
  ModalContent,
  ModalHeader,
  ModalBody,
  ModalFooter,
  ModalCloseButton,
  FormControl,
  FormLabel,
  Input,
  Select,
  Table,
  Thead,
  Tbody,
  Tr,
  Th,
  Td,
  Badge,
  useToast,
  IconButton,
  Tooltip,
  useClipboard,
  Divider,
  Spacer,
  Grid,
  GridItem,
  useColorModeValue,
  Icon,
  SimpleGrid,
} from "@chakra-ui/react";
import { useState } from "react";
import { Layout } from "~/components/layout";
import { motion } from "framer-motion";
import {
  FiCopy,
  FiPlus,
  FiTrash2,
  FiTerminal,
  FiKey,
  FiInfo,
  FiCheck,
  FiDownload,
  FiBook,
  FiArrowRight,
  FiCode,
  FiPlay,
} from "react-icons/fi";
import { api } from "~/utils/api";
import { type ApiKey } from "~/types/misc";

// Create motion components
const MotionBox = motion(Box);
const MotionHeading = motion(Heading);
const MotionText = motion(Text);
const MotionFlex = motion(Flex);
const MotionVStack = motion(VStack);
const MotionButton = motion(Button);
const MotionBadge = motion(Badge);

export default function CLI() {
  const [tabIndex, setTabIndex] = useState(0);
  const [newKeyData, setNewKeyData] = useState({
    name: "",
    expiresIn: 30 * 24 * 60 * 60 * 1000,
  });
  const [newKey, setNewKey] = useState<ApiKey | null>(null);
  const { isOpen, onOpen, onClose } = useDisclosure();
  const toast = useToast();

  // tRPC queries and mutations
  const { data: apiKeys = [], refetch: refetchApiKeys } =
    api.apiKeys.getAll.useQuery();
  const createApiKeyMutation = api.apiKeys.create.useMutation({
    onSuccess: (data) => {
      setNewKey(data);
      void refetchApiKeys();
      toast({
        title: "API key created successfully",
        status: "success",
        duration: 3000,
        isClosable: true,
        position: "top-right",
        variant: "subtle",
      });
    },
    onError: (error) => {
      toast({
        title: "Failed to create API key",
        description: error.message,
        status: "error",
        duration: 3000,
        isClosable: true,
        position: "top-right",
        variant: "subtle",
      });
    },
  });
  const deleteApiKeyMutation = api.apiKeys.delete.useMutation({
    onSuccess: () => {
      void refetchApiKeys();
      toast({
        title: "API key deleted successfully",
        status: "success",
        duration: 3000,
        isClosable: true,
        position: "top-right",
        variant: "subtle",
      });
    },
    onError: (error) => {
      toast({
        title: "Failed to delete API key",
        description: error.message,
        status: "error",
        duration: 3000,
        isClosable: true,
        position: "top-right",
        variant: "subtle",
      });
    },
  });

  // CLI installation steps
  const installSteps = [
    { title: "Install via NPM", command: "npm install -g tensara-cli" },
    { title: "Install via Yarn", command: "yarn global add tensara-cli" },
  ];

  // CLI usage examples
  const usageExamples = [
    {
      title: "Authentication",
      command: "tensara login --key <your-api-key>",
      description: "Authenticate with your Tensara API key",
    },
    {
      title: "Checker Command",
      command: "tensara checker -p <problem-id> -s <solution-file>",
      description: "Command to check the solution against the problem",
    },
    {
      title: "Help",
      command: "tensara --help",
      description: "Show available commands and options",
    },
  ];

  const handleInputChange = (
    e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>
  ) => {
    const { name, value } = e.target;
    if (name === "expiresIn") {
      // Convert days to milliseconds
      const days = parseInt(value);
      const milliseconds = days * 24 * 60 * 60 * 1000;
      setNewKeyData({ ...newKeyData, [name]: milliseconds });
    } else {
      setNewKeyData({ ...newKeyData, [name]: value });
    }
  };

  const formatDate = (date: Date) => {
    return date.toLocaleDateString("en-US", {
      year: "numeric",
      month: "short",
      day: "numeric",
    });
  };

  const handleCreateApiKey = () => {
    if (newKeyData.name.length < 3) {
      toast({
        title: "Name must be at least 3 characters long",
        status: "error",
        duration: 3000,
        isClosable: true,
        position: "top-right",
        variant: "subtle",
      });
      return;
    }
    createApiKeyMutation.mutate(newKeyData);
  };

  const handleDeleteApiKey = (id: string) => {
    deleteApiKeyMutation.mutate({ id });
  };

  // Terminal Box Component with animations
  const TerminalBox = ({ command }: { command: string }) => {
    const { hasCopied, onCopy } = useClipboard(command);

    return (
      <MotionBox
        bg="gray.900"
        color="whiteAlpha.900"
        borderRadius="md"
        overflow="hidden"
        whileHover={{ scale: 1.01 }}
        transition={{ duration: 0.2 }}
      >
        {/* Terminal Content */}
        <Flex p={3} position="relative" fontFamily="mono" fontSize="sm">
          <Text color="green.400" mr={2}>
            $
          </Text>
          <Text flex="1">{command}</Text>
          <Tooltip
            label={hasCopied ? "Copied!" : "Copy command"}
            placement="top"
            hasArrow
          >
            <IconButton
              aria-label="Copy command"
              icon={hasCopied ? <FiCheck /> : <FiCopy />}
              size="xs"
              variant="ghost"
              color={hasCopied ? "green.400" : "whiteAlpha.700"}
              _hover={{ color: "whiteAlpha.900" }}
              position="absolute"
              right={2}
              top={2}
              onClick={onCopy}
            />
          </Tooltip>
        </Flex>
      </MotionBox>
    );
  };

  // Animation variants
  const fadeIn = {
    initial: { opacity: 0 },
    animate: { opacity: 1 },
    exit: { opacity: 0 },
    transition: { duration: 0.3 },
  };

  const slideUp = {
    initial: { opacity: 0, y: 20 },
    animate: { opacity: 1, y: 0 },
    exit: { opacity: 0, y: -20 },
    transition: { duration: 0.4 },
  };

  const staggerChildren = {
    animate: {
      transition: {
        staggerChildren: 0.1,
      },
    },
  };

  return (
    <Layout
      title="CLI & API Keys | Tensara"
      ogTitle="CLI & API Keys | Tensara"
      ogDescription="Tensara CLI and API key management tools."
    >
      <Box maxW="7xl" mx="auto" px={4} py={8}>
        <Tabs
          index={tabIndex}
          onChange={setTabIndex}
          colorScheme="brand"
          variant="enclosed"
        >
          <TabList borderBottomColor="gray.700">
            <Tab
              fontWeight="medium"
              _selected={{
                fontWeight: "bold",
                color: "brand.primary",
                borderBottomColor: "brand.primary",
              }}
            >
              <HStack spacing={2}>
                <FiTerminal />
                <Text>CLI</Text>
              </HStack>
            </Tab>
            <Tab
              fontWeight="medium"
              _selected={{
                fontWeight: "bold",
                color: "brand.primary",
                borderBottomColor: "brand.primary",
              }}
            >
              <HStack spacing={2}>
                <FiKey />
                <Text>API Keys</Text>
              </HStack>
            </Tab>
          </TabList>
          <TabPanels>
            {/* CLI Tab Content */}
            {/* CLI Tab Content */}
            <TabPanel>
              <MotionBox
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ duration: 0.4 }}
              >
                <Box maxW="900px" mx="auto">
                  <MotionHeading
                    size="lg"
                    mb={6}
                    bgGradient="linear(to-r, brand.primary, green.400)"
                    bgClip="text"
                    initial={{ opacity: 0, y: -10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.5 }}
                    fontWeight="bold"
                    letterSpacing="tight"
                  >
                    Tensara Command Line Interface
                  </MotionHeading>

                  {/* Introduction Card */}
                  <MotionBox
                    bg="gray.800"
                    borderRadius="xl"
                    p={6}
                    mb={8}
                    boxShadow="lg"
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.5, delay: 0.1 }}
                  >
                    <Flex gap={4} align="center" mb={4}>
                      <Icon as={FiTerminal} boxSize={6} color="brand.primary" />
                      <Text fontSize="lg" fontWeight="medium" color="white">
                        Write GPU code from your IDE and compete at Tensara
                      </Text>
                    </Flex>
                    <Text color="gray.300" lineHeight="tall">
                      The Tensara CLI provides a seamless way to run GPU code
                      for problems on Tensara.
                    </Text>
                  </MotionBox>

                  {/* Installation Section */}
                  <MotionFlex
                    direction="column"
                    mb={10}
                    initial="initial"
                    animate="animate"
                    variants={staggerChildren}
                  >
                    <Flex align="center" mb={4}>
                      <Icon as={FiDownload} color="green.400" mr={2} />
                      <Heading size="md">Installation</Heading>
                    </Flex>

                    <SimpleGrid columns={{ base: 1, md: 2 }} spacing={4}>
                      {installSteps.map((step, index) => (
                        <MotionBox
                          key={index}
                          variants={slideUp}
                          bg="gray.900"
                          borderRadius="lg"
                          overflow="hidden"
                          borderWidth="1px"
                          borderColor="gray.700"
                          transition={{ duration: 0.3 }}
                          _hover={{
                            borderColor: "brand.primary",
                            transform: "translateY(-2px)",
                            boxShadow: "0 4px 12px rgba(0,0,0,0.2)",
                          }}
                        >
                          <Box p={4} bg="gray.800">
                            <Text fontWeight="medium" color="white">
                              {step.title}
                            </Text>
                          </Box>
                          <TerminalBox command={step.command} />
                        </MotionBox>
                      ))}
                    </SimpleGrid>
                  </MotionFlex>

                  {/* Getting Started Section */}
                  <MotionBox
                    mb={10}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.5, delay: 0.3 }}
                  >
                    <Flex align="center" mb={4}>
                      <Icon as={FiPlay} color="green.400" mr={2} />
                      <Heading size="md">Getting Started</Heading>
                    </Flex>

                    <Box
                      bg="gray.800"
                      borderRadius="lg"
                      p={5}
                      mb={5}
                      borderLeft="4px solid"
                      borderColor="brand.primary"
                    >
                      <Text mb={4} color="gray.300" fontSize="sm">
                        After installation, authenticate the CLI with your API
                        key. You can create one in the
                        <Button
                          variant="link"
                          colorScheme="brand"
                          size="sm"
                          onClick={() => setTabIndex(1)}
                          mx={2}
                          fontWeight="bold"
                          _hover={{
                            textDecoration: "none",
                            color: "brand.600",
                          }}
                        >
                          API Keys
                        </Button>
                        tab.
                      </Text>
                      <TerminalBox command="tensara auth --key tsra_yourapikey" />
                    </Box>
                  </MotionBox>

                  {/* Usage Examples Section */}
                  <MotionFlex
                    direction="column"
                    mb={10}
                    initial="initial"
                    animate="animate"
                    variants={staggerChildren}
                  >
                    <Flex align="center" mb={5}>
                      <Icon as={FiCode} color="green.400" mr={2} />
                      <Heading size="md">Command Examples</Heading>
                    </Flex>

                    <SimpleGrid columns={{ base: 1, md: 1, lg: 1 }} spacing={4}>
                      {usageExamples.map((example, index) => (
                        <MotionBox
                          key={index}
                          variants={slideUp}
                          bg="gray.800"
                          borderRadius="lg"
                          p={5}
                          transition={{ duration: 0.3 }}
                          _hover={{
                            bg: "gray.750",
                            transform: "translateY(-2px)",
                            boxShadow: "0 4px 12px rgba(0,0,0,0.15)",
                          }}
                        >
                          <Flex justify="space-between" align="center" mb={2}>
                            <Text fontWeight="bold" color="white">
                              {example.title}
                            </Text>
                            <Badge colorScheme="green" variant="subtle">
                              Command
                            </Badge>
                          </Flex>
                          <TerminalBox command={example.command} />
                          <Text fontSize="sm" color="gray.400" mt={2}>
                            {example.description}
                          </Text>
                        </MotionBox>
                      ))}
                    </SimpleGrid>
                  </MotionFlex>

                  {/* Documentation Section */}
                  <MotionBox
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ duration: 0.5, delay: 0.6 }}
                    bg="gray.800"
                    borderRadius="xl"
                    p={6}
                    borderWidth="1px"
                    borderColor="gray.700"
                  >
                    <Flex align="center" mb={4}>
                      <Icon
                        as={FiBook}
                        color="brand.primary"
                        boxSize={5}
                        mr={3}
                      />
                      <Heading size="md">Documentation</Heading>
                    </Flex>
                    <Text color="gray.300" mb={4}>
                      For detailed documentation and advanced usage examples,
                      visit our comprehensive guides.
                    </Text>
                    <Button
                      as="a"
                      href="/docs/cli"
                      rightIcon={<FiArrowRight />}
                      colorScheme="brand"
                      size="md"
                      borderRadius="md"
                      fontWeight="medium"
                      _hover={{ transform: "translateY(-1px)" }}
                      transition="all 0.2s"
                    >
                      View CLI Documentation
                    </Button>
                  </MotionBox>
                </Box>
              </MotionBox>
            </TabPanel>

            {/* API Keys Tab Content */}
            <TabPanel>
              <MotionBox
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5 }}
              >
                <Flex justify="space-between" align="center" mb={6}>
                  <MotionHeading
                    size="lg"
                    bgGradient="linear(to-r, brand.primary, green.400)"
                    bgClip="text"
                    initial={{ opacity: 0, y: -10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.5, delay: 0.1 }}
                  >
                    API Keys
                  </MotionHeading>
                  <MotionButton
                    leftIcon={<FiPlus />}
                    colorScheme="brand"
                    onClick={onOpen}
                    isLoading={createApiKeyMutation.isPending}
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    transition={{ duration: 0.2 }}
                  >
                    Create New Key
                  </MotionButton>
                </Flex>
                <MotionText
                  mb={6}
                  color="gray.400"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ duration: 0.4, delay: 0.1 }}
                >
                  API keys allow secure access to Tensara services from your
                  applications or the CLI. Keep your keys secure and never share
                  them in public repositories or client-side code.
                </MotionText>

                {apiKeys.length > 0 ? (
                  <MotionBox
                    borderWidth="1px"
                    borderColor="gray.700"
                    borderRadius="lg"
                    overflow="hidden"
                    boxShadow="xl"
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.5, delay: 0.2 }}
                  >
                    <Table variant="simple">
                      <Thead bg="gray.800">
                        <Tr>
                          <Th borderColor="gray.700" color="gray.300">
                            Name
                          </Th>
                          <Th borderColor="gray.700" color="gray.300">
                            Created
                          </Th>
                          <Th borderColor="gray.700" color="gray.300">
                            Expires
                          </Th>
                          <Th
                            borderColor="gray.700"
                            color="gray.300"
                            width="100px"
                            textAlign="center"
                          >
                            Actions
                          </Th>
                        </Tr>
                      </Thead>
                      <Tbody>
                        {apiKeys.map((key, index) => {
                          const isExpired =
                            new Date(key.expiresAt) < new Date();

                          return (
                            <MotionBox
                              as={Tr}
                              key={key.id}
                              initial={{ opacity: 0 }}
                              animate={{
                                opacity: 1,
                                transition: { delay: 0.1 * (index + 1) },
                              }}
                              _hover={{ bg: "gray.800" }}
                              transition={{ duration: 0.2 }}
                            >
                              <Td borderColor="gray.700">
                                <HStack>
                                  <FiKey color="#10B981" />
                                  <Text fontWeight="medium">{key.name}</Text>
                                </HStack>
                              </Td>
                              <Td borderColor="gray.700" color="gray.400">
                                {formatDate(key.createdAt)}
                              </Td>
                              <Td borderColor="gray.700">
                                <Flex align="center">
                                  <Text color="gray.400">
                                    {formatDate(key.expiresAt)}
                                  </Text>
                                  {isExpired && (
                                    <MotionBadge
                                      ml={2}
                                      colorScheme="red"
                                      variant="subtle"
                                      initial={{ scale: 0.8 }}
                                      animate={{ scale: 1 }}
                                    >
                                      Expired
                                    </MotionBadge>
                                  )}
                                </Flex>
                              </Td>
                              <Td borderColor="gray.700" textAlign="center">
                                <Tooltip
                                  label="Delete key"
                                  placement="top"
                                  hasArrow
                                >
                                  <IconButton
                                    aria-label="Delete key"
                                    icon={<FiTrash2 />}
                                    size="sm"
                                    colorScheme="red"
                                    variant="ghost"
                                    onClick={() => handleDeleteApiKey(key.id)}
                                    isLoading={deleteApiKeyMutation.isPending}
                                  />
                                </Tooltip>
                              </Td>
                            </MotionBox>
                          );
                        })}
                      </Tbody>
                    </Table>
                  </MotionBox>
                ) : (
                  <MotionBox
                    textAlign="center"
                    py={10}
                    borderWidth="1px"
                    borderColor="gray.700"
                    borderRadius="lg"
                    borderStyle="dashed"
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.5, delay: 0.2 }}
                  >
                    <Text mb={4} color="gray.500">
                      No API keys created yet
                    </Text>
                    <Button
                      leftIcon={<FiPlus />}
                      colorScheme="brand"
                      onClick={onOpen}
                      size="sm"
                    >
                      Create your first API key
                    </Button>
                  </MotionBox>
                )}

                <MotionBox
                  mt={8}
                  p={6}
                  bg="gray.800"
                  borderRadius="md"
                  border="1px"
                  borderColor="gray.700"
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.5, delay: 0.3 }}
                >
                  <HStack mb={4}>
                    <FiInfo color="#10B981" />
                    <Heading size="md">About API Keys</Heading>
                  </HStack>
                  <VStack spacing={4} align="stretch" color="gray.300">
                    <Text>
                      <Text as="span" fontWeight="bold" color="white">
                        Format:
                      </Text>{" "}
                      API keys follow the format{" "}
                      <Code colorScheme="gray">tsra_prefix_keyBody</Code> where
                      only the prefix is stored in our database for
                      identification.
                    </Text>
                    <Text>
                      <Text as="span" fontWeight="bold" color="white">
                        Security:
                      </Text>{" "}
                      Keys are hashed using Argon2id before storage. The full
                      key is only shown once at creation time.
                    </Text>
                    <Text>
                      <Text as="span" fontWeight="bold" color="white">
                        Expiration:
                      </Text>{" "}
                      Keys have a default 30-day expiration which can be
                      customized.
                    </Text>
                    <Text>
                      <Text as="span" fontWeight="bold" color="white">
                        Permissions:
                      </Text>{" "}
                      Currently all keys have full access to your account.
                      Scoped permissions are coming soon.
                    </Text>
                  </VStack>
                </MotionBox>
              </MotionBox>

              {/* Create API Key Modal */}
              <Modal
                isOpen={isOpen}
                onClose={() => {
                  onClose();
                  setNewKey(null);
                }}
                size="md"
              >
                <ModalOverlay backdropFilter="blur(4px)" />
                <ModalContent
                  bg="gray.800"
                  borderColor="gray.700"
                  borderWidth="1px"
                  shadow="2xl"
                >
                  <ModalHeader borderBottomWidth="1px" borderColor="gray.700">
                    {newKey ? "API Key Created" : "Create New API Key"}
                  </ModalHeader>
                  <ModalCloseButton />
                  <ModalBody py={6}>
                    {newKey ? (
                      <MotionVStack
                        spacing={6}
                        align="stretch"
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        transition={{ delay: 0.2 }}
                      >
                        <Box
                          p={4}
                          bg="green.900"
                          opacity={0.3}
                          borderRadius="md"
                          borderWidth="1px"
                          borderColor="green.700"
                        >
                          <HStack mb={2}>
                            <FiCheck color="#4ADE80" />
                            <Text fontWeight="bold" color="green.400">
                              API Key Created Successfully
                            </Text>
                          </HStack>
                          <Text fontSize="sm" mb={4} color="gray.300">
                            This is the only time your full API key will be
                            shown. Copy it now!
                          </Text>
                          <Box
                            bg="gray.900"
                            p={3}
                            borderRadius="md"
                            color="whiteAlpha.900"
                            fontFamily="mono"
                            fontSize="sm"
                            position="relative"
                            mb={2}
                          >
                            <Text wordBreak="break-all">{newKey.key}</Text>
                            <IconButton
                              aria-label="Copy API key"
                              icon={<FiCopy />}
                              size="sm"
                              position="absolute"
                              top={2}
                              right={2}
                              onClick={() => {
                                void navigator.clipboard.writeText(newKey.key);
                                toast({
                                  title: "API key copied to clipboard",
                                  status: "success",
                                  duration: 2000,
                                  isClosable: true,
                                  position: "top-right",
                                  variant: "subtle",
                                });
                              }}
                              variant="ghost"
                            />
                          </Box>
                          <Text fontSize="xs" color="gray.500">
                            API Key ID: {newKey.id}
                          </Text>
                        </Box>

                        <Box>
                          <Text fontWeight="bold" mb={2} color="white">
                            API Key Details:
                          </Text>
                          <Text fontSize="sm" color="gray.300">
                            <Text as="span" fontWeight="semibold">
                              Name:
                            </Text>{" "}
                            {newKey.name}
                          </Text>
                          <Text fontSize="sm" color="gray.300">
                            <Text as="span" fontWeight="semibold">
                              Expires:
                            </Text>{" "}
                            {formatDate(newKey.expiresAt)}
                          </Text>
                        </Box>
                      </MotionVStack>
                    ) : (
                      <VStack spacing={4} align="stretch">
                        <FormControl>
                          <FormLabel>Key Name</FormLabel>
                          <Input
                            name="name"
                            value={newKeyData.name}
                            onChange={handleInputChange}
                            placeholder="e.g. Development Key"
                          />
                        </FormControl>
                        <FormControl>
                          <FormLabel>Expiration</FormLabel>
                          <Select
                            name="expiresIn"
                            value={newKeyData.expiresIn / (24 * 60 * 60 * 1000)}
                            onChange={handleInputChange}
                          >
                            <option value={7}>7 days</option>
                            <option value={30}>30 days</option>
                            <option value={90}>90 days</option>
                            <option value={365}>1 year</option>
                            <option value={3650}>Never</option>
                          </Select>
                        </FormControl>
                      </VStack>
                    )}
                  </ModalBody>
                  <ModalFooter borderTopWidth="1px" borderColor="gray.700">
                    {newKey ? (
                      <Button
                        colorScheme="brand"
                        onClick={() => {
                          onClose();
                          setNewKey(null);
                        }}
                      >
                        Done
                      </Button>
                    ) : (
                      <Button
                        colorScheme="brand"
                        onClick={handleCreateApiKey}
                        isLoading={createApiKeyMutation.isPending}
                      >
                        Create Key
                      </Button>
                    )}
                  </ModalFooter>
                </ModalContent>
              </Modal>
            </TabPanel>
          </TabPanels>
        </Tabs>
      </Box>
    </Layout>
  );
}
