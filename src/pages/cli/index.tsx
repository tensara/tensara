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
  Menu,
  MenuButton,
  MenuList,
  MenuItem,
  UnorderedList,
  ListItem,
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
  FiCoffee,
  FiPackage,
  FiBox,
  FiLock,
  FiHelpCircle,
  FiSend,
  FiCheckCircle,
  FiZap,
  FiGitBranch,
  FiLayers,
  FiCloud,
  FiFolder,
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

interface FeatureProps {
  icon: React.ElementType;
  title: string;
  description: string;
}

interface SectionHeaderProps {
  icon: React.ElementType;
  title: string;
  description?: string;
}

interface CircleProps {
  size: string;
  bg: string;
  mr?: number;
}

export default function CLI() {
  const [tabIndex, setTabIndex] = useState(0);
  const [newKeyData, setNewKeyData] = useState({
    name: "",
    expiresIn: 30 * 24 * 60 * 60 * 1000,
  });
  const [newKey, setNewKey] = useState<ApiKey | null>(null);
  const [deletingKeyId, setDeletingKeyId] = useState<string | null>(null);
  const [hasCopiedKey, setHasCopiedKey] = useState(false);
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
        variant: "solid",
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
        variant: "solid",
      });
    },
  });
  const deleteApiKeyMutation = api.apiKeys.delete.useMutation({
    onSuccess: () => {
      void refetchApiKeys();
    },
    onError: (error) => {
      toast({
        title: "Failed to delete API key",
        description: error.message,
        status: "error",
        duration: 3000,
        isClosable: true,
        position: "top-right",
        variant: "solid",
      });
    },
  });

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
        variant: "solid",
      });
      return;
    }
    createApiKeyMutation.mutate(newKeyData);
  };

  const handleDeleteApiKey = (id: string) => {
    setDeletingKeyId(id);
    deleteApiKeyMutation.mutate(
      { id },
      {
        onSuccess: () => {
          setDeletingKeyId(null);
          void refetchApiKeys();
        },
        onError: (error) => {
          setDeletingKeyId(null);
          toast({
            title: "Failed to delete API key",
            description: error.message,
            status: "error",
            duration: 3000,
            isClosable: true,
            position: "top-right",
            variant: "solid",
          });
        },
      }
    );
  };

  // CLI installation steps
  const installSteps = [
    {
      title: "Mac/Linux",
      commands: {
        bash: "curl -sSL https://get.tensara.org/install.sh | bash",
        zsh: "curl -sSL https://get.tensara.org/install.sh | zsh",
      },
      icon: FiPackage,
    },
    {
      title: "Windows",
      commands: {
        powershell: "iwr -useb https://get.tensara.org/install.ps1 | iex",
      },
      icon: FiBox,
    },
  ];

  // CLI usage examples
  const usageExamples = [
    {
      title: "Initialization",
      command: "tensara init <directory> -p <problem_name> -l <language>",
      description:
        "Create a template solution file and a problem file in the specified directory",
      icon: FiFolder,
    },
    {
      title: "Checker Command",
      command: "tensara checker -g <gpu> -p <problem> -s <solution_file>",
      description: "Validate your solution against the problem specification",
      icon: FiCheckCircle,
    },
    {
      title: "Benchmark Command",
      command: "tensara benchmark -g <gpu> -p <problem> -s <solution_file>",
      description: "Benchmark your solution on Tensara problems",
      icon: FiZap,
    },
    {
      title: "Submit Solution",
      command: "tensara submit -g <gpu> -p <problem> -s <solution_file>",
      description: "Submit your solution for official evaluation",
      icon: FiSend,
    },
  ];

  // Animation variants
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        when: "beforeChildren",
        staggerChildren: 0.1,
        duration: 0.5,
      },
    },
  };

  const itemVariants = {
    hidden: { y: 20, opacity: 0 },
    visible: {
      y: 0,
      opacity: 1,
      transition: { duration: 0.4 },
    },
  };

  const TerminalBox = ({ command }: { command: string }) => {
    const { hasCopied, onCopy } = useClipboard(command);
    const [isHovered, setIsHovered] = useState(false);

    return (
      <MotionBox
        bg="gray.900"
        color="whiteAlpha.900"
        borderRadius="lg"
        overflow="hidden"
        borderWidth="1px"
        borderColor={"gray.700"}
        boxShadow={"0 4px 6px rgba(0,0,0,0.1)"}
        transition={{ duration: 0.2 }}
        onMouseEnter={() => setIsHovered(true)}
        onMouseLeave={() => setIsHovered(false)}
      >
        {/* Terminal Header */}
        <Flex
          bg="gray.800"
          p={2}
          borderBottomWidth="1px"
          borderColor="gray.700"
          align="center"
        >
          <Flex mr={2}>
            <Circle size="10px" bg="red.500" mr={1} />
            <Circle size="10px" bg="yellow.500" mr={1} />
            <Circle size="10px" bg="green.500" />
          </Flex>
          <Text fontSize="xs" color="gray.400" ml="auto">
            tensara-cli
          </Text>
        </Flex>

        {/* Terminal Content */}
        <Flex p={2} position="relative" fontFamily="mono" fontSize="sm">
          <Text color="green.400" mr={2} fontWeight="bold">
            $
          </Text>
          <Text flex="1" fontSize="small">
            {command}
          </Text>
          <Tooltip
            label={hasCopied ? "Copied!" : "Copy command"}
            placement="top"
            hasArrow
          >
            <IconButton
              aria-label="Copy command"
              icon={hasCopied ? <FiCheck /> : <FiCopy />}
              size="small"
              variant="ghost"
              color={hasCopied ? "green.400" : "whiteAlpha.700"}
              _hover={{ color: "brand.500" }}
              onClick={onCopy}
            />
          </Tooltip>
        </Flex>
      </MotionBox>
    );
  };

  // Feature Component
  const Feature = ({ icon, title, description }: FeatureProps) => {
    return (
      <HStack align="start" spacing={4}>
        <Flex
          bg="rgba(66, 153, 225, 0.1)"
          w="40px"
          h="40px"
          borderRadius="md"
          color="brand.400"
          align="center"
          justify="center"
          flexShrink={0}
        >
          <Icon as={icon} boxSize={5} />
        </Flex>
        <Box>
          <Text
            fontWeight="bold"
            color="white"
            mb={1}
            fontFamily="Space Grotesk, sans-serif"
          >
            {title}
          </Text>
          <Text color="gray.400" fontSize="sm">
            {description}
          </Text>
        </Box>
      </HStack>
    );
  };

  // Section Header Component
  const SectionHeader = ({ icon, title, description }: SectionHeaderProps) => {
    return (
      <Box mb={6}>
        <Flex align="center" mb={2}>
          <Icon as={icon} color="brand.400" mr={2} />
          <Heading
            size="md"
            color="white"
            fontFamily="Space Grotesk, sans-serif"
          >
            {title}
          </Heading>
        </Flex>
        {description && (
          <Text color="gray.400" fontSize="md">
            {description}
          </Text>
        )}
      </Box>
    );
  };

  const Circle = ({ size, bg, mr }: CircleProps) => {
    return <Box borderRadius="full" w={size} h={size} bg={bg} mr={mr} />;
  };

  return (
    <Layout
      title="CLI & API Keys"
      ogTitle="CLI & API Keys"
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
            <TabPanel>
              <MotionBox
                initial="hidden"
                animate="visible"
                variants={containerVariants}
                overflow="hidden"
                w="full"
              >
                {/* Hero Section */}
                <MotionBox mb={12} textAlign="center" mt={8}>
                  <Heading
                    as="h1"
                    size="2xl"
                    bgGradient="linear(to-r, #2ecc71, #27ae60)"
                    bgClip="text"
                    letterSpacing="tight"
                    fontWeight="extrabold"
                    mb={4}
                    fontFamily="Space Grotesk, sans-serif"
                  >
                    Tensara CLI
                  </Heading>
                  <Text fontSize="xl" color="gray.300" maxW="2xl" mx="auto">
                    Write & test GPU code from your IDE
                  </Text>
                </MotionBox>

                {/* Main Features Card */}
                <MotionBox
                  variants={itemVariants}
                  bg="gray.800"
                  borderRadius="2xl"
                  p={8}
                  mb={12}
                  boxShadow="xl"
                  borderWidth="1px"
                  borderColor="gray.700"
                  position="relative"
                  overflow="hidden"
                >
                  <Box
                    position="absolute"
                    top="-10%"
                    right="-5%"
                    width="300px"
                    height="300px"
                    bg="radial-gradient(circle, rgba(0,128,255,0.1) 0%, rgba(0,0,0,0) 70%)"
                    zIndex="0"
                  />

                  <Flex direction="column" position="relative" zIndex="1">
                    <Flex align="center" mb={6}>
                      <Icon
                        as={FiTerminal}
                        boxSize={10}
                        color="brand.400"
                        mr={4}
                      />
                      <Box>
                        <Text color="gray.300" lineHeight="tall" fontSize="lg">
                          The Tensara CLI provides a way to develop, test, and
                          submit GPU-accelerated solutions directly from your
                          local environment.
                        </Text>
                      </Box>
                    </Flex>

                    <SimpleGrid columns={{ base: 1, md: 2 }} spacing={6}>
                      <Feature
                        icon={FiZap}
                        title="Lightning-fast Execution"
                        description="Run your code on our powerful GPU infrastructure without complex setup"
                      />
                      <Feature
                        icon={FiCloud}
                        title="Seamless Cloud Integration"
                        description="Submit solutions directly to Tensara problems with one command"
                      />
                      <Feature
                        icon={FiGitBranch}
                        title="Version Control Friendly"
                        description="Works perfectly with Git and your existing development workflow"
                      />
                      <Feature
                        icon={FiLayers}
                        title="Cross-platform"
                        description="Available for Windows, macOS, and Linux systems"
                      />
                    </SimpleGrid>
                  </Flex>
                </MotionBox>

                {/* Installation Section */}
                <MotionBox variants={itemVariants} mb={16}>
                  <SectionHeader
                    icon={FiDownload}
                    title="Installation"
                    description="Get started with Tensara CLI in seconds"
                  />

                  {installSteps.map((step, index) => (
                    <Box
                      key={index}
                      mb={6}
                      p={6}
                      bg="gray.800"
                      borderRadius="lg"
                      borderWidth="1px"
                      borderColor="gray.700"
                    >
                      <Flex align="center" mb={4}>
                        <Icon
                          as={step.icon}
                          color="brand.400"
                          boxSize={5}
                          mr={2}
                        />
                        <Text fontWeight="bold" color="white">
                          {step.title}
                        </Text>
                      </Flex>
                      {Object.entries(step.commands).map(([shell, command]) => (
                        <Box key={shell} mb={shell === "bash" ? 4 : 0}>
                          {Object.keys(step.commands).length > 1 && (
                            <Text color="gray.400" mb={2} fontSize="sm">
                              Using {shell}:
                            </Text>
                          )}
                          <TerminalBox command={command as string} />
                        </Box>
                      ))}
                    </Box>
                  ))}
                </MotionBox>

                {/* Authentication Section */}
                <MotionBox variants={itemVariants} mb={16}>
                  <SectionHeader
                    icon={FiKey}
                    title="Authentication"
                    description="Securely connect to your Tensara account"
                  />

                  <Box
                    bg="gray.800"
                    borderRadius="xl"
                    p={6}
                    borderWidth="1px"
                    borderColor="gray.700"
                    mb={6}
                  >
                    <Text mb={4} color="gray.300">
                      After installation, authenticate the CLI with your API
                      key. You can create or manage your keys in the
                      <Menu>
                        <MenuButton
                          as={Button}
                          variant="link"
                          colorScheme="brand"
                          mx={2}
                          fontWeight="bold"
                          _hover={{
                            textDecoration: "none",
                            color: "brand.400",
                          }}
                        >
                          API Keys
                        </MenuButton>
                        <MenuList>
                          <MenuItem onClick={() => setTabIndex(1)}>
                            View API Keys
                          </MenuItem>
                          <MenuItem onClick={onOpen}>Create New Key</MenuItem>
                        </MenuList>
                      </Menu>
                      tab.
                    </Text>
                    <TerminalBox command="tensara auth -t <your_api_token>" />
                  </Box>
                </MotionBox>

                {/* Commands Section */}
                <MotionBox variants={itemVariants} mb={16}>
                  <SectionHeader
                    icon={FiCode}
                    title="Command Examples"
                    description="Essential commands to get you started"
                  />

                  <SimpleGrid columns={{ base: 1, md: 2 }} spacing={6}>
                    {usageExamples.map((example, index) => (
                      <MotionBox
                        key={index}
                        variants={itemVariants}
                        bg="gray.800"
                        borderRadius="xl"
                        p={6}
                        borderWidth="1px"
                        borderColor="gray.700"
                        _hover={{
                          borderColor: "brand.500",
                          boxShadow: "0 8px 16px rgba(0,0,0,0.2)",
                          transform: "translateY(-4px)",
                        }}
                        transition={{ duration: 0.3 }}
                      >
                        <Flex justify="space-between" align="center" mb={3}>
                          <HStack>
                            <Icon as={example.icon} color="brand.500" />
                            <Text fontWeight="bold" color="white">
                              {example.title}
                            </Text>
                          </HStack>
                        </Flex>
                        <TerminalBox command={example.command} />
                        <Text fontSize="sm" color="gray.400" mt={3}>
                          {example.description}
                        </Text>
                      </MotionBox>
                    ))}
                  </SimpleGrid>
                </MotionBox>
              </MotionBox>
            </TabPanel>

            {/* API Keys Tab Content */}
            <TabPanel>
              <MotionBox
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5 }}
                mt={4}
              >
                <Flex justify="space-between" align="center" mb={6}>
                  <MotionHeading
                    size="lg"
                    bgGradient="linear(to-r, brand.primary, green.400)"
                    bgClip="text"
                    initial={{ opacity: 0, y: -10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.5, delay: 0.1 }}
                    fontFamily="Space Grotesk, sans-serif"
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
                    bg="#0e8144"
                    _hover={{
                      bg: "#0a6434",
                    }}
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
                                    isLoading={deletingKeyId === key.id}
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
                      bg="#0e8144"
                      _hover={{
                        bg: "#0a6434",
                      }}
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
                    <Box>
                      <Heading size="sm" mb={1} color="white">
                        Format
                      </Heading>
                      <Text>
                        API keys follow the format{" "}
                        <Code colorScheme="brand">tsra_prefix_keyBody</Code>{" "}
                        where only the prefix is stored in our database for
                        identification.
                      </Text>
                    </Box>

                    <Box>
                      <Heading size="sm" mb={1} color="white">
                        Security
                      </Heading>
                      <Text>
                        Keys are hashed using Argon2id before storage. The full
                        key is only shown once at creation time.
                      </Text>
                    </Box>

                    <Box>
                      <Heading size="sm" mb={1} color="white">
                        Expiration
                      </Heading>
                      <Text>
                        Keys have a default 30-day expiration which can be
                        customized.
                      </Text>
                    </Box>

                    <Box>
                      <Heading size="sm" mb={1} color="white">
                        Permissions
                      </Heading>
                      <Text>
                        Currently all keys have full access to your account.
                        Scoped permissions are coming soon.
                      </Text>
                    </Box>
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
                size="xl"
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
                  <ModalBody py={6} minW={"190px"}>
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
                          <Flex
                            bg="gray.900"
                            p={3}
                            justifyContent="space-between"
                            alignItems="center"
                            borderRadius="md"
                            color="whiteAlpha.900"
                            fontFamily="mono"
                            fontSize="sm"
                            position="relative"
                            mb={2}
                          >
                            <Text wordBreak="break-all" flex={1} mr={6}>
                              {newKey.key}
                            </Text>
                            <IconButton
                              aria-label="Copy API key"
                              icon={hasCopiedKey ? <FiCheck /> : <FiCopy />}
                              size="sm"
                              position="absolute"
                              top={2}
                              right={2}
                              onClick={() => {
                                void navigator.clipboard.writeText(newKey.key);
                                setHasCopiedKey(true);
                                setTimeout(() => setHasCopiedKey(false), 2000);
                                toast({
                                  title: "API key copied to clipboard",
                                  status: "success",
                                  duration: 2000,
                                  isClosable: true,
                                  position: "top-right",
                                  variant: "solid",
                                });
                              }}
                              variant="ghost"
                              color={
                                hasCopiedKey ? "green.400" : "whiteAlpha.700"
                              }
                              _hover={{ color: "brand.500" }}
                            />
                          </Flex>
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
                            borderColor="gray.600"
                            borderWidth="1px"
                            _hover={{
                              borderColor: "gray.500",
                            }}
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
                        backgroundColor="#0e8144"
                        _hover={{
                          backgroundColor: "#0a6434",
                        }}
                      >
                        Done
                      </Button>
                    ) : (
                      <Button
                        colorScheme="brand"
                        onClick={handleCreateApiKey}
                        isLoading={createApiKeyMutation.isPending}
                        backgroundColor="#0e8144"
                        _hover={{
                          backgroundColor: "#0a6434",
                        }}
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
