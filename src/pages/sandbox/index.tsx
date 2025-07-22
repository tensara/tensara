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
} from "@chakra-ui/react";
import { Layout } from "~/components/layout";
import { format } from "date-fns";
import {
  Menu,
  MenuButton,
  MenuList,
  MenuItem,
  IconButton,
  useDisclosure,
  useToast,
} from "@chakra-ui/react";
import { FiMoreVertical } from "react-icons/fi";


export default function SandboxHome() {
  const router = useRouter();
  const [name, setName] = useState("");
  const utils = api.useUtils();

  const { data: workspaces, isLoading } = api.workspace.list.useQuery();
  const createWorkspace = api.workspace.create.useMutation({
    onSuccess: async (data) => {
      await utils.workspace.list.invalidate();
      await router.push(`/sandbox/${data.slug}`);
    },
  });

  const handleCreate = () => {
    if (!name.trim()) return;
    createWorkspace.mutate({ name });
    setName("");
  };

  const toast = useToast();
const [editingId, setEditingId] = useState<string | null>(null);
const [newName, setNewName] = useState("");

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
});

  return (
    <Layout title="SandboxHome">
      <VStack px={8} py={12} align="start" spacing={6}>
        <Heading size="lg" mb={2}>
          My Workspaces
        </Heading>
        <Text color="gray.400" mb={6}>
          Create and manage your GPU code projects.
        </Text>

        <HStack mb={6}>
          <Input
            placeholder="Name your workspace..."
            value={name}
            onChange={(e) => setName(e.target.value)}
            size="md"
            borderRadius="lg"
            bg="#111"
            color="white"
            _placeholder={{ color: "gray.500" }}
            px={4}
            py={2}
            w="300px"
          />
          <Button
            onClick={handleCreate}
            bg="rgba(34, 197, 94, 0.1)"
            color="rgb(34, 197, 94)"
            borderRadius="lg"
            fontWeight="semibold"
            fontSize="sm"
            px={6}
            h="40px"
            _hover={{ bg: "rgba(34, 197, 94, 0.2)", transform: "translateY(-1px)" }}
            _active={{ bg: "rgba(34, 197, 94, 0.25)" }}
            transition="all 0.2s"
          >
            Create
          </Button>
        </HStack>

        <VStack align="stretch" spacing={3} pt={4} w="full">
          {isLoading ? (
            <Text>Loading...</Text>
          ) : (
            <Grid templateColumns="repeat(auto-fill, minmax(250px, 1fr))" gap={4}>
              {workspaces?.map((ws) => (
<Box
  key={ws.id}
  p={4}
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
          if (confirm("Are you sure you want to delete this workspace?")) {
            deleteMutation.mutate({ id: ws.id });
          }
        }}
      >
        Delete
      </MenuItem>
    </MenuList>
  </Menu>

  <Box onClick={() => router.push(`/sandbox/${ws.slug}`)}>
    {editingId === ws.id ? (
      <Input
        value={newName}
        onChange={(e) => setNewName(e.target.value)}
        onBlur={() => {
          renameMutation.mutate({ id: ws.id, name: newName });
        }}
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
      <Text fontWeight="semibold" fontSize="md" color="white">
        {ws.name}
      </Text>
    )}
    <Text fontSize="xs" color="gray.500" mt={2}>
      Last edited {format(ws.updatedAt, "Pp")}
    </Text>
  </Box>
</Box>

              ))}
            </Grid>
          )}
        </VStack>
      </VStack>
    </Layout>
  );
}
