import { api } from "~/utils/api";
import { useRouter } from "next/router";
import { useState } from "react";
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
} from "@chakra-ui/react";
import { Layout } from "~/components/layout";
import { format } from "date-fns";
import { Menu, MenuButton, MenuList, MenuItem } from "@chakra-ui/react";
import { FiMoreVertical, FiPlus } from "react-icons/fi";
import { useSession, signIn } from "next-auth/react";
import { type TRPCClientError } from "@trpc/client";
import type { AppRouter } from "~/server/api/root";
import { keyframes } from "@emotion/react";

const pulseAnimation = keyframes`
  0% { opacity: 0.6; }
  50% { opacity: 0.8; }
  100% { opacity: 0.6; }
`;

export default function SandboxHome() {
  const { data: session, status } = useSession();
  const router = useRouter();
  const utils = api.useUtils();
  const toast = useToast();

  const [name, setName] = useState("");
  const [editingId, setEditingId] = useState<string | null>(null);
  const [newName, setNewName] = useState("");

  const {
    isOpen: isModalOpen,
    onOpen: openModal,
    onClose: closeModal,
  } = useDisclosure();

  const { data: workspaces, isLoading } = api.workspace.list.useQuery();

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
    createWorkspace.mutate({ name });
    setName("");
  };

  if (status !== "authenticated") {
    return (
      <Layout title="Sign In">
        <Center minH="70vh">
          <VStack spacing={4}>
            <Heading size="md" color="white">
              Sign in to start messing around with GPU workspaces.
            </Heading>
            <Button
              onClick={() => signIn()}
              colorScheme="green"
              borderRadius="lg"
            >
              Sign In
            </Button>
          </VStack>
        </Center>
      </Layout>
    );
  }

  return (
    <Layout title="Sandbox">
      <VStack px={8} py={12} align="start" spacing={6} w="full">
        <HStack w="full" justify="space-between">
          <Heading size="lg">My Workspaces</Heading>
          <Button
            onClick={openModal}
            bg="rgba(34, 197, 94, 0.1)"
            color="rgb(34, 197, 94)"
            leftIcon={<FiPlus />}
            borderRadius="lg"
            _hover={{
              bg: "rgba(34, 197, 94, 0.2)",
              transition: "all 0.5s ease",
            }}
          >
            Create Workspace
          </Button>
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
              fontWeight="semibold"
              fontSize="sm"
              px={6}
              h="40px"
              _hover={{
                bg: "rgba(34, 197, 94, 0.2)",
                transform: "translateY(-1px)",
              }}
              _active={{ bg: "rgba(34, 197, 94, 0.25)" }}
            >
              + Create Workspace
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
              <Input
                placeholder="Enter workspace name"
                value={name}
                onChange={(e) => setName(e.target.value)}
                bg="brand.dark"
                color="white"
                _hover={{ bg: "brand.dark" }}
                _focus={{ bg: "brand.dark" }}
              />
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
