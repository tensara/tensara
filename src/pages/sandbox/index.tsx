import { api } from "~/utils/api";
import { useRouter } from "next/router";
import { useState, useRef, useEffect } from "react";
import type { ProgrammingLanguage } from "~/types/misc";
import { Select } from "@chakra-ui/react";
import {
  VStack,
  Heading,
  Button,
  Input,
  HStack,
  Text,
  Box,
  Grid,
  useDisclosure,
  Modal,
  ModalOverlay,
  ModalContent,
  ModalHeader,
  ModalCloseButton,
  ModalBody,
  ModalFooter,
  IconButton,
  useToast,
  Center,
  Flex,
  Spinner,
  SimpleGrid,
  Icon,
} from "@chakra-ui/react";
import { Layout } from "~/components/layout";
import { format } from "date-fns";
import { Menu, MenuButton, MenuList, MenuItem } from "@chakra-ui/react";
import {
  FiMoreVertical,
  FiPlus,
  FiCpu,
  FiCode,
  FiShare2,
  FiPackage,
} from "react-icons/fi";
import { LANGUAGE_DISPLAY_NAMES } from "~/constants/language";
import { useSession, signIn } from "next-auth/react";
import { type TRPCClientError } from "@trpc/client";
import type { AppRouter } from "~/server/api/root";
import { keyframes } from "@emotion/react";
import { type IconType } from "react-icons";

const pulseAnimation = keyframes`
  0% { opacity: 0.6; }
  50% { opacity: 0.8; }
  100% { opacity: 0.6; }
`;

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
      bg="transparent"
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

export default function SandboxHome() {
  const { data: session, status } = useSession();
  const router = useRouter();
  const utils = api.useUtils();
  const toast = useToast();

  const [name, setName] = useState("");
  const [editingId, setEditingId] = useState<string | null>(null);
  const [newName, setNewName] = useState("");
  const [newWorkspaceLanguage, setNewWorkspaceLanguage] =
    useState<ProgrammingLanguage>("cuda");
  const newWorkspaceMenuRef = useRef<HTMLButtonElement | null>(null);

  const {
    isOpen: isModalOpen,
    onOpen: openModal,
    onClose: closeModal,
  } = useDisclosure();

  const { data: workspaces, isLoading } = api.workspace.list.useQuery();

  const getLanguageDisplay = (language: ProgrammingLanguage) =>
    LANGUAGE_DISPLAY_NAMES[language] ?? language.toUpperCase();

  useEffect(() => {
    function updateWidth() {
      newWorkspaceMenuRef.current as unknown as HTMLElement | null;
    }

    if (isModalOpen) {
      // measure once when modal opens
      updateWidth();
      window.addEventListener("resize", updateWidth);
      return () => window.removeEventListener("resize", updateWidth);
    }
    // reset when modal closes
  }, [isModalOpen]);

  const createWorkspace = api.workspace.create.useMutation({
    onSuccess: async (data) => {
      await utils.workspace.list.invalidate();
      if (!session?.user?.username) return;
      const username = session.user.username;
      await router.push(`/sandbox/${username.toLowerCase()}/${data.slug}`);
    },
    onError: (error) => {
      const code = (error as TRPCClientError<AppRouter>)?.data?.code;

      if (code === "CONFLICT") {
        toast({
          title: "Workspace already exists",
          description: error.message,
          status: "error",
          duration: 4000,
        });
      } else {
        toast({
          title: "Something went wrong",
          description: error.message || "Unknown error",
          status: "error",
          duration: 4000,
        });
      }
    },
  });

  const deleteMutation = api.workspace.delete.useMutation({
    onSuccess: async () => {
      await utils.workspace.list.invalidate();
      toast({ title: "Deleted workspace", status: "success", duration: 2000 });
    },
  });

  const renameMutation = api.workspace.rename.useMutation({
    onSuccess: async () => {
      await utils.workspace.list.invalidate();
      setEditingId(null);
      toast({ title: "Renamed workspace", status: "success", duration: 2000 });
    },
    onError: (error) => {
      const typedError = error as TRPCClientError<AppRouter>;
      const code = typedError.shape?.data?.code;

      if (code === "CONFLICT") {
        toast({
          title: "Rename failed",
          description: error.message,
          status: "error",
          duration: 4000,
        });
      } else {
        toast({
          title: "Something went wrong",
          description: error.message,
          status: "error",
          duration: 4000,
        });
      }
    },
  });

  const handleCreate = () => {
    if (!name.trim()) return;
    // pass language along; server can ignore it if not supported.
    createWorkspace.mutate({ name, language: newWorkspaceLanguage });
    setName("");
  };

  if (status !== "authenticated") {
    return (
      <Layout title="Sign In">
        <Box
          w="full"
          minH="80vh"
          px={{ base: 6, md: 12 }}
          py={{ base: 12, md: 20 }}
          display="flex"
          flexDirection="column"
          alignItems="center"
          justifyContent="center"
          gap={12}
        >
          <VStack spacing={6} maxW="640px" textAlign="center">
            <Flex p={4} borderRadius="xl" color="#2ecc71">
              <Icon as={FiPackage} boxSize={16} />
            </Flex>
            <Heading fontSize={{ base: "2xl", md: "3xl" }} color="white">
              Tensara Sandbox
            </Heading>
            <Text fontSize={{ base: "md", md: "lg" }} color="gray.300">
              Disposable GPU workspaces you can spin up in seconds. Write CUDA
              C++ or Mojo kernels, experiment freely, and come back to them
              later.
            </Text>
            <VStack spacing={2}>
              <Button
                onClick={() => signIn()}
                bg="#0e8144"
                _hover={{
                  bg: "#0a6434",
                  transform: "translateY(-2px)",
                }}
                borderRadius="lg"
                px={8}
                h="46px"
              >
                Sign In to Start a Workspace
              </Button>
              <Text fontSize="xs" color="gray.500">
                Use your existing account to start a GPU workspace.
              </Text>
            </VStack>
          </VStack>

          <SimpleGrid
            columns={{ base: 1, sm: 2, md: 3 }}
            spacing={6}
            w="full"
            maxW="1200px"
          >
            <FeatureCard
              icon={FiCode}
              title="CUDA C++ & Mojo"
              description="Choose your language when creating a sandbox. Run kernels on real GPU hardware without any local setup."
            />
            <FeatureCard
              icon={FiCpu}
              title="Persistent workspaces"
              description="Keep files and experiments together across sessions. Come back to your work anytime, exactly as you left it."
            />
            <FeatureCard
              icon={FiShare2}
              title="Shareable workspaces"
              description="Every sandbox has a unique URL you can share with others. Collaborate or show off your code with a simple link."
            />
          </SimpleGrid>
        </Box>
      </Layout>
    );
  }

  return (
    <Layout title="Sandbox">
      <VStack px={8} py={12} align="start" spacing={6} w="full">
        <HStack w="full" justify="space-between">
          <Heading size="lg">My Workspaces</Heading>
          {workspaces?.length && workspaces.length > 0 && (
            <Button
              onClick={openModal}
              bg="rgba(34, 197, 94, 0.1)"
              color="rgb(34, 197, 94)"
              leftIcon={<FiPlus />}
              borderRadius="lg"
              fontWeight="semibold"
              fontSize="sm"
              px={6}
              h="40px"
              _hover={{
                bg: "rgba(34, 197, 94, 0.2)",
                transition: "all 0.5s ease",
              }}
            >
              Create Workspace
            </Button>
          )}
        </HStack>

        {isLoading ? (
          <Box
            w="100%"
            h="100%"
            bg="brand.secondary"
            borderRadius="xl"
            overflow="hidden"
            position="relative"
          >
            <Flex
              position="absolute"
              w="100%"
              h="100%"
              zIndex="1"
              flexDirection="column"
              alignItems="center"
              justifyContent="center"
            >
              <Box
                position="absolute"
                top="0"
                left="0"
                right="0"
                bottom="0"
                bgGradient="linear(to-b, brand.dark, #191919)"
                animation={`${pulseAnimation} 2s infinite ease-in-out`}
              />

              <Spinner
                size="xl"
                thickness="3px"
                speed="0.8s"
                color="brand.navbar"
                zIndex="2"
                mb="3"
              />

              <Text color="gray.300" fontFamily="mono" fontSize="sm" zIndex="2">
                Loading...
              </Text>
            </Flex>
          </Box>
        ) : workspaces?.length === 0 ? (
          <Center w="full" py={20} flexDirection="column">
            <Text fontSize="lg" color="gray.400">
              Get started with your first GPU workspace.
            </Text>
            <Text fontSize="md" color="gray.500" mt={1}>
              Make GPUs go brrr
            </Text>
            <Button
              onClick={openModal}
              mt={4}
              bg="rgba(34, 197, 94, 0.1)"
              color="rgb(34, 197, 94)"
              borderRadius="lg"
              leftIcon={<FiPlus />}
              fontWeight="semibold"
              fontSize="sm"
              px={6}
              h="40px"
              _hover={{
                bg: "rgba(34, 197, 94, 0.2)",
                transition: "all 0.5s ease",
              }}
              _active={{ bg: "rgba(34, 197, 94, 0.25)" }}
            >
              Create Workspace
            </Button>
          </Center>
        ) : (
          <Grid
            templateColumns="repeat(auto-fill, minmax(320px, 1fr))"
            gap={5}
            w="full"
          >
            {workspaces?.map((ws) => (
              <Box
                key={ws.id}
                p={5}
                borderRadius="lg"
                bg="brand.secondary"
                position="relative"
                transition="all 0.3s ease"
                _hover={{ bg: "#1e2533", cursor: "pointer" }}
              >
                <Menu>
                  <MenuButton
                    as={IconButton}
                    icon={<FiMoreVertical />}
                    size="sm"
                    variant="ghost"
                    position="absolute"
                    top="8px"
                    right="8px"
                    color="gray.400"
                    _hover={{ bg: "rgba(255,255,255,0.05)" }}
                    _active={{ bg: "rgba(255,255,255,0.1)" }}
                    _focus={{ bg: "transparent", color: "gray.400" }}
                    onClick={(e) => e.stopPropagation()}
                  />
                  <MenuList
                    bg="brand.secondary"
                    borderColor="gray.800"
                    p={0}
                    borderRadius="md"
                    minW="120px"
                  >
                    <MenuItem
                      bg="brand.secondary"
                      _hover={{ bg: "gray.700" }}
                      borderRadius="md"
                      onClick={(e) => {
                        e.stopPropagation();
                        setEditingId(ws.id);
                        setNewName(ws.name);
                      }}
                    >
                      Rename
                    </MenuItem>
                    <MenuItem
                      bg="brand.secondary"
                      _hover={{ bg: "gray.700" }}
                      borderRadius="md"
                      onClick={(e) => {
                        e.stopPropagation();
                        if (
                          confirm(
                            "Are you sure you want to delete this workspace?"
                          )
                        ) {
                          deleteMutation.mutate({ id: ws.id });
                        }
                      }}
                    >
                      Delete
                    </MenuItem>
                  </MenuList>
                </Menu>

                <Box
                  onClick={() => {
                    if (!session?.user?.username) return;
                    void router.push(
                      `/sandbox/${session.user.username.toLowerCase()}/${ws.slug}`
                    );
                  }}
                >
                  {editingId === ws.id ? (
                    <Input
                      value={newName}
                      onChange={(e) => setNewName(e.target.value)}
                      onBlur={() =>
                        renameMutation.mutate({ id: ws.id, name: newName })
                      }
                      onKeyDown={(e) => {
                        if (e.key === "Enter") {
                          renameMutation.mutate({ id: ws.id, name: newName });
                        }
                      }}
                      autoFocus
                      size="sm"
                      bg="#222"
                      color="white"
                      borderColor="#333"
                      _hover={{ borderColor: "#444" }}
                    />
                  ) : (
                    <Text fontWeight="semibold" fontSize="lg" color="white">
                      {ws.name}
                    </Text>
                  )}
                  <Text fontSize="xs" color="gray.500" mt={1}>
                    Last edited {format(ws.updatedAt, "Pp")}
                  </Text>
                </Box>
              </Box>
            ))}
          </Grid>
        )}

        <Modal isOpen={isModalOpen} onClose={closeModal} isCentered>
          <ModalOverlay />
          <ModalContent bg="brand.secondary" borderRadius="lg">
            <ModalHeader color="white">Create Workspace</ModalHeader>
            <ModalCloseButton color="white" />
            <ModalBody>
              <VStack align="stretch" spacing={3}>
                <Input
                  placeholder="Enter workspace name"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") {
                      handleCreate();
                      closeModal();
                    }
                  }}
                  bg="brand.dark"
                  color="white"
                  _hover={{ bg: "brand.dark" }}
                  _focus={{ bg: "brand.dark" }}
                />
                <Box>
                  <Text fontSize="sm" color="gray.300" mb={1}>
                    Default Language
                  </Text>
                  <Select
                    size="sm"
                    border="none"
                    bg="brand.dark"
                    color="white"
                    value={newWorkspaceLanguage}
                    onChange={(e) =>
                      setNewWorkspaceLanguage(
                        e.target.value as ProgrammingLanguage
                      )
                    }
                    borderRadius="lg"
                    h="40px"
                  >
                    {(["cuda", "mojo"] as ProgrammingLanguage[]).map((lang) => (
                      <option key={lang} value={lang}>
                        {getLanguageDisplay(lang)}
                      </option>
                    ))}
                  </Select>
                </Box>
              </VStack>
            </ModalBody>
            <ModalFooter>
              <Button
                onClick={() => {
                  handleCreate();
                  closeModal();
                }}
                bg="rgba(34, 197, 94, 0.1)"
                color="rgb(34, 197, 94)"
                borderRadius="lg"
                fontWeight="semibold"
                fontSize="sm"
                px={4}
                h="40px"
                _hover={{
                  bg: "rgba(34, 197, 94, 0.2)",
                  transition: "all 0.5s ease",
                }}
                _active={{ bg: "rgba(34, 197, 94, 0.25)" }}
              >
                Create
              </Button>
            </ModalFooter>
          </ModalContent>
        </Modal>
      </VStack>
    </Layout>
  );
}
