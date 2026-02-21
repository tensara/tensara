import {
  Box,
  Text,
  Heading,
  Button,
  SimpleGrid,
  VStack,
  HStack,
  Badge,
  Spinner,
  useDisclosure,
} from "@chakra-ui/react";
import { Layout } from "~/components/layout";
import { api } from "~/utils/api";
import { useSession } from "next-auth/react";
import Link from "next/link";
import { FaPlus, FaUsers, FaBook } from "react-icons/fa";
import { CreateGroupModal } from "~/components/groups/CreateGroupModal";

const roleColor: Record<string, string> = {
  OWNER: "green",
  ADMIN: "blue",
  MEMBER: "gray",
};

export default function GroupsPage() {
  const { data: session } = useSession();
  const { data: groups, isLoading } = api.groups.getMyGroups.useQuery(undefined, {
    enabled: !!session,
  });
  const { isOpen, onOpen, onClose } = useDisclosure();

  if (!session) {
    return (
      <Layout title="Groups">
        <Box maxW="7xl" mx="auto" px={4} py={8}>
          <VStack spacing={6} align="center" py={20}>
            <Heading size="lg" color="white">
              Sign in to view your groups
            </Heading>
            <Text color="gray.400">
              Groups let you compete with friends on curated problem sets.
            </Text>
          </VStack>
        </Box>
      </Layout>
    );
  }

  if (isLoading) {
    return (
      <Layout title="Groups">
        <Box display="flex" justifyContent="center" alignItems="center" h="50vh">
          <Spinner size="xl" />
        </Box>
      </Layout>
    );
  }

  return (
    <Layout
      title="Groups"
      ogTitle="Groups | Tensara"
      ogDescription="Compete with friends on curated GPU programming problem sets."
    >
      <Box maxW="7xl" mx="auto" px={4} py={8}>
        <HStack justify="space-between" mb={6}>
          <Heading size="lg" color="white">
            My Groups
          </Heading>
          <Button
            bg="brand.primary"
            color="white"
            _hover={{ opacity: 0.9 }}
            leftIcon={<FaPlus />}
            onClick={onOpen}
          >
            Create Group
          </Button>
        </HStack>

        {groups && groups.length > 0 ? (
          <SimpleGrid columns={{ base: 1, md: 2, lg: 3 }} spacing={4}>
            {groups.map((group) => (
              <Link key={group.id} href={`/groups/${group.slug}`} passHref>
                <Box
                  bg="brand.secondary"
                  border="1px solid"
                  borderColor="whiteAlpha.100"
                  borderRadius="xl"
                  p={5}
                  cursor="pointer"
                  _hover={{
                    borderColor: "whiteAlpha.300",
                    transform: "translateY(-2px)",
                    boxShadow: "0 4px 12px rgba(0,0,0,0.3)",
                  }}
                  transition="all 0.2s"
                >
                  <VStack align="stretch" spacing={3}>
                    <HStack justify="space-between">
                      <Text color="white" fontWeight="bold" fontSize="lg" noOfLines={1}>
                        {group.name}
                      </Text>
                      <Badge colorScheme={roleColor[group.role]} fontSize="xs">
                        {group.role}
                      </Badge>
                    </HStack>

                    {group.description && (
                      <Text color="gray.400" fontSize="sm" noOfLines={2}>
                        {group.description}
                      </Text>
                    )}

                    <HStack spacing={4} pt={1}>
                      <HStack spacing={1}>
                        <FaUsers color="#a0aec0" size={14} />
                        <Text color="gray.400" fontSize="sm">
                          {group._count.members}
                        </Text>
                      </HStack>
                      <HStack spacing={1}>
                        <FaBook color="#a0aec0" size={14} />
                        <Text color="gray.400" fontSize="sm">
                          {group._count.problems} problems
                        </Text>
                      </HStack>
                    </HStack>
                  </VStack>
                </Box>
              </Link>
            ))}
          </SimpleGrid>
        ) : (
          <Box
            bg="brand.secondary"
            borderRadius="xl"
            border="1px solid"
            borderColor="whiteAlpha.100"
            py={16}
            textAlign="center"
          >
            <VStack spacing={4}>
              <FaUsers color="#a0aec0" size={40} />
              <Text color="white" fontSize="lg" fontWeight="600">
                No groups yet
              </Text>
              <Text color="gray.400">
                Create a group and invite friends to compete on problem sets.
              </Text>
              <Button
                bg="brand.primary"
                color="white"
                _hover={{ opacity: 0.9 }}
                leftIcon={<FaPlus />}
                onClick={onOpen}
              >
                Create your first group
              </Button>
            </VStack>
          </Box>
        )}
      </Box>

      <CreateGroupModal isOpen={isOpen} onClose={onClose} />
    </Layout>
  );
}
