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
} from "@chakra-ui/react";
import { Layout } from "~/components/layout";
import { format } from "date-fns";
import { Menu, MenuButton, MenuList, MenuItem } from "@chakra-ui/react";
import { FiMoreVertical, FiPlus } from "react-icons/fi";
import { useSession, signIn } from "next-auth/react";
import { TRPCClientError } from "@trpc/client";
import type { AppRouter } from "~/server/api/root";

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
            _hover={{ bg: "rgba(34, 197, 94, 0.2)" }}
          >
            Create Workspace
          </Button>
        </HStack>

        {isLoading ? (
          <Text>Loading...</Text>
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
                bg="#111"
                border="1px solid #2a2a2a"
                position="relative"
                _hover={{ borderColor: "#22c55e", cursor: "pointer" }}
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
                    onClick={(e) => e.stopPropagation()}
                  />
                  <MenuList bg="#1e1e1e" borderColor="#2a2a2a">
                    <MenuItem
                      bg="#1e1e1e"
                      _hover={{ bg: "#2a2a2a" }}
                      onClick={(e) => {
                        e.stopPropagation();
                        setEditingId(ws.id);
                        setNewName(ws.name);
                      }}
                    >
                      Rename
                    </MenuItem>
                    <MenuItem
                      bg="#1e1e1e"
                      _hover={{ bg: "#2a2a2a" }}
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
          <ModalContent bg="#1e1e1e" borderRadius="lg">
            <ModalHeader color="white">Create Workspace</ModalHeader>
            <ModalCloseButton color="white" />
            <ModalBody>
              <Input
                placeholder="Enter workspace name"
                value={name}
                onChange={(e) => setName(e.target.value)}
                bg="#111"
                color="white"
                _placeholder={{ color: "gray.500" }}
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
                px={6}
                h="40px"
                _hover={{
                  bg: "rgba(34, 197, 94, 0.2)",
                  transform: "translateY(-1px)",
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
