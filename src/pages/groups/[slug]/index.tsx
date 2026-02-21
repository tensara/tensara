import {
  Box,
  Text,
  Heading,
  Button,
  HStack,
  VStack,
  Badge,
  Spinner,
  Table,
  Thead,
  Tbody,
  Tr,
  Th,
  Td,
  Tabs,
  TabList,
  Tab,
  TabPanels,
  TabPanel,
  Image,
  IconButton,
  useDisclosure,
  Tooltip,
  Menu,
  MenuButton,
  MenuList,
  MenuItem,
} from "@chakra-ui/react";
import { Layout } from "~/components/layout";
import { api } from "~/utils/api";
import { useRouter } from "next/router";
import Link from "next/link";
import {
  FaPlus,
  FaUserPlus,
  FaCog,
  FaTrash,
  FaCheckCircle,
  FaChevronDown,
} from "react-icons/fa";
import { FiArrowLeft } from "react-icons/fi";
import { AddMemberModal } from "~/components/groups/AddMemberModal";
import { AddProblemModal } from "~/components/groups/AddProblemModal";

const difficultyColor: Record<string, string> = {
  EASY: "green",
  MEDIUM: "yellow",
  HARD: "red",
  EXPERT: "purple",
};

const roleColor: Record<string, string> = {
  OWNER: "green",
  ADMIN: "blue",
  MEMBER: "gray",
};

export default function GroupDashboardPage() {
  const router = useRouter();
  const slug = router.query.slug as string;

  const { data: group, isLoading: groupLoading } = api.groups.getBySlug.useQuery(
    { slug },
    { enabled: !!slug }
  );
  const { data: problems, isLoading: problemsLoading } = api.groups.getProblems.useQuery(
    { groupSlug: slug },
    { enabled: !!slug }
  );
  const { data: members, isLoading: membersLoading } = api.groups.getMembers.useQuery(
    { groupSlug: slug },
    { enabled: !!slug }
  );

  const utils = api.useUtils();

  const removeProblem = api.groups.removeProblem.useMutation({
    onSuccess: async () => {
      await utils.groups.getProblems.invalidate({ groupSlug: slug });
      await utils.groups.getBySlug.invalidate({ slug });
    },
  });

  const removeMember = api.groups.removeMember.useMutation({
    onSuccess: async () => {
      await utils.groups.getMembers.invalidate({ groupSlug: slug });
      await utils.groups.getBySlug.invalidate({ slug });
    },
  });

  const updateRole = api.groups.updateMemberRole.useMutation({
    onSuccess: async () => {
      await utils.groups.getMembers.invalidate({ groupSlug: slug });
    },
  });

  const {
    isOpen: isMemberOpen,
    onOpen: onMemberOpen,
    onClose: onMemberClose,
  } = useDisclosure();
  const {
    isOpen: isProblemOpen,
    onOpen: onProblemOpen,
    onClose: onProblemClose,
  } = useDisclosure();

  const isAdmin = group?.currentUserRole === "OWNER" || group?.currentUserRole === "ADMIN";
  const isOwner = group?.currentUserRole === "OWNER";

  if (groupLoading) {
    return (
      <Layout title="Group">
        <Box display="flex" justifyContent="center" alignItems="center" h="50vh">
          <Spinner size="xl" />
        </Box>
      </Layout>
    );
  }

  if (!group) {
    return (
      <Layout title="Group">
        <Box maxW="7xl" mx="auto" px={4} py={8}>
          <Text color="gray.400">Group not found or you don&apos;t have access.</Text>
        </Box>
      </Layout>
    );
  }

  return (
    <Layout title={group.name} ogTitle={`${group.name} | Tensara Groups`}>
      <Box maxW="7xl" mx="auto" px={4} py={8}>
        {/* Header */}
        <VStack align="stretch" spacing={4} mb={6}>
          <Link href="/groups" passHref>
            <HStack spacing={1} color="gray.400" _hover={{ color: "white" }} cursor="pointer" w="fit-content">
              <FiArrowLeft />
              <Text fontSize="sm">Back to groups</Text>
            </HStack>
          </Link>

          <HStack justify="space-between" align="start">
            <VStack align="start" spacing={1}>
              <HStack spacing={3}>
                <Heading size="lg" color="white">
                  {group.name}
                </Heading>
                <Badge colorScheme={roleColor[group.currentUserRole]} fontSize="xs">
                  {group.currentUserRole}
                </Badge>
              </HStack>
              {group.description && (
                <Text color="gray.400" fontSize="sm">
                  {group.description}
                </Text>
              )}
              <Text color="gray.500" fontSize="xs">
                {group._count.members} members &middot; {group._count.problems} problems
              </Text>
            </VStack>

            {isAdmin && (
              <HStack spacing={2}>
                <Link href={`/groups/${slug}/settings`} passHref>
                  <IconButton
                    aria-label="Settings"
                    icon={<FaCog />}
                    variant="ghost"
                    color="gray.400"
                    _hover={{ color: "white", bg: "whiteAlpha.100" }}
                  />
                </Link>
              </HStack>
            )}
          </HStack>
        </VStack>

        {/* Tabs */}
        <Tabs variant="unstyled">
          <TabList borderBottom="1px solid" borderColor="whiteAlpha.100" mb={4}>
            <Tab
              color="gray.400"
              _selected={{ color: "white", borderBottom: "2px solid", borderColor: "brand.primary" }}
              pb={3}
              mr={4}
            >
              Problems ({group._count.problems})
            </Tab>
            <Tab
              color="gray.400"
              _selected={{ color: "white", borderBottom: "2px solid", borderColor: "brand.primary" }}
              pb={3}
            >
              Members ({group._count.members})
            </Tab>
          </TabList>

          <TabPanels>
            {/* Problems Tab */}
            <TabPanel px={0}>
              {isAdmin && (
                <Box mb={4}>
                  <Button
                    size="sm"
                    bg="brand.primary"
                    color="white"
                    _hover={{ opacity: 0.9 }}
                    leftIcon={<FaPlus />}
                    onClick={onProblemOpen}
                  >
                    Add Problem
                  </Button>
                </Box>
              )}

              {problemsLoading ? (
                <Box textAlign="center" py={12}>
                  <Spinner />
                </Box>
              ) : problems && problems.length > 0 ? (
                <Box
                  overflowX="auto"
                  borderRadius="xl"
                  bg="whiteAlpha.50"
                  border="1px solid"
                  borderColor="whiteAlpha.100"
                >
                  <Table variant="simple">
                    <Thead bg="brand.card">
                      <Tr>
                        <Th
                          color="gray.300"
                          fontSize="md"
                          py={4}
                          borderBottom="1px solid"
                          borderColor="brand.primary"
                        >
                          Problem
                        </Th>
                        <Th
                          color="gray.300"
                          fontSize="md"
                          width="140px"
                          borderBottom="1px solid"
                          borderColor="brand.primary"
                        >
                          Difficulty
                        </Th>
                        <Th
                          color="gray.300"
                          fontSize="md"
                          width="140px"
                          borderBottom="1px solid"
                          borderColor="brand.primary"
                        >
                          Solved
                        </Th>
                        <Th
                          color="gray.300"
                          fontSize="md"
                          width="160px"
                          borderBottom="1px solid"
                          borderColor="brand.primary"
                        >
                          Leaderboard
                        </Th>
                        {isAdmin && (
                          <Th
                            color="gray.300"
                            fontSize="md"
                            width="80px"
                            borderBottom="1px solid"
                            borderColor="brand.primary"
                          />
                        )}
                      </Tr>
                    </Thead>
                    <Tbody>
                      {problems.map((problem) => (
                        <Tr
                          key={problem.id}
                          bg="brand.secondary"
                          _hover={{ bg: "gray.700" }}
                          transition="all 0.15s"
                          borderBottom="1px solid"
                          borderColor="gray.800"
                        >
                          <Td color="white" borderBottom="none">
                            <HStack spacing={3}>
                              {problem.solvedByCurrentUser && (
                                <FaCheckCircle color="#4ade80" size={14} opacity={0.68} />
                              )}
                              <Link href={`/problems/${problem.slug}`}>
                                <Text _hover={{ textDecoration: "underline" }}>
                                  {problem.title}
                                </Text>
                              </Link>
                            </HStack>
                          </Td>
                          <Td borderBottom="none">
                            <Badge
                              colorScheme={difficultyColor[problem.difficulty]}
                              px={2}
                              py={0.5}
                              borderRadius="md"
                            >
                              {problem.difficulty}
                            </Badge>
                          </Td>
                          <Td borderBottom="none">
                            <Text color="gray.300" fontSize="sm">
                              {problem.solvedCount} / {problem.totalMembers}
                            </Text>
                          </Td>
                          <Td borderBottom="none">
                            <Link href={`/groups/${slug}/leaderboard/${problem.slug}`}>
                              <Button
                                size="xs"
                                variant="ghost"
                                color="brand.primary"
                                _hover={{ bg: "whiteAlpha.100" }}
                              >
                                View
                              </Button>
                            </Link>
                          </Td>
                          {isAdmin && (
                            <Td borderBottom="none">
                              <Tooltip label="Remove problem">
                                <IconButton
                                  aria-label="Remove problem"
                                  icon={<FaTrash />}
                                  size="xs"
                                  variant="ghost"
                                  color="gray.500"
                                  _hover={{ color: "red.400", bg: "whiteAlpha.100" }}
                                  onClick={() =>
                                    removeProblem.mutate({
                                      groupSlug: slug,
                                      problemSlug: problem.slug,
                                    })
                                  }
                                />
                              </Tooltip>
                            </Td>
                          )}
                        </Tr>
                      ))}
                    </Tbody>
                  </Table>
                </Box>
              ) : (
                <Box
                  bg="brand.secondary"
                  borderRadius="xl"
                  border="1px solid"
                  borderColor="whiteAlpha.100"
                  py={12}
                  textAlign="center"
                >
                  <VStack spacing={3}>
                    <Text color="gray.400">No problems added yet.</Text>
                    {isAdmin && (
                      <Button
                        size="sm"
                        bg="brand.primary"
                        color="white"
                        _hover={{ opacity: 0.9 }}
                        leftIcon={<FaPlus />}
                        onClick={onProblemOpen}
                      >
                        Add a problem
                      </Button>
                    )}
                  </VStack>
                </Box>
              )}
            </TabPanel>

            {/* Members Tab */}
            <TabPanel px={0}>
              {isAdmin && (
                <Box mb={4}>
                  <Button
                    size="sm"
                    bg="brand.primary"
                    color="white"
                    _hover={{ opacity: 0.9 }}
                    leftIcon={<FaUserPlus />}
                    onClick={onMemberOpen}
                  >
                    Add Member
                  </Button>
                </Box>
              )}

              {membersLoading ? (
                <Box textAlign="center" py={12}>
                  <Spinner />
                </Box>
              ) : members && members.length > 0 ? (
                <Box
                  overflowX="auto"
                  borderRadius="xl"
                  bg="whiteAlpha.50"
                  border="1px solid"
                  borderColor="whiteAlpha.100"
                >
                  <Table variant="simple">
                    <Thead bg="brand.card">
                      <Tr>
                        <Th
                          color="gray.300"
                          fontSize="md"
                          py={4}
                          borderBottom="1px solid"
                          borderColor="brand.primary"
                        >
                          User
                        </Th>
                        <Th
                          color="gray.300"
                          fontSize="md"
                          width="120px"
                          borderBottom="1px solid"
                          borderColor="brand.primary"
                        >
                          Role
                        </Th>
                        {isAdmin && (
                          <Th
                            color="gray.300"
                            fontSize="md"
                            width="120px"
                            borderBottom="1px solid"
                            borderColor="brand.primary"
                          />
                        )}
                      </Tr>
                    </Thead>
                    <Tbody>
                      {members.map((member) => (
                        <Tr
                          key={member.id}
                          bg="brand.secondary"
                          _hover={{ bg: "gray.700" }}
                          transition="all 0.15s"
                          borderBottom="1px solid"
                          borderColor="gray.800"
                        >
                          <Td color="white" borderBottom="none">
                            <HStack spacing={3}>
                              <Image
                                src={member.user.image ?? ""}
                                alt={member.user.username ?? ""}
                                w={8}
                                h={8}
                                borderRadius="full"
                                fallbackSrc="https://via.placeholder.com/32"
                              />
                              <Link href={`/user/${member.user.username}`}>
                                <Text _hover={{ textDecoration: "underline" }}>
                                  {member.user.username ?? member.user.name ?? "Unknown"}
                                </Text>
                              </Link>
                            </HStack>
                          </Td>
                          <Td borderBottom="none">
                            {isOwner && member.role !== "OWNER" ? (
                              <Menu>
                                <MenuButton
                                  as={Button}
                                  size="xs"
                                  rightIcon={<FaChevronDown size={8} />}
                                  variant="ghost"
                                  color="gray.300"
                                  _hover={{ bg: "whiteAlpha.100" }}
                                >
                                  <Badge colorScheme={roleColor[member.role]} fontSize="xs">
                                    {member.role}
                                  </Badge>
                                </MenuButton>
                                <MenuList
                                  bg="brand.secondary"
                                  borderColor="whiteAlpha.200"
                                  minW="120px"
                                >
                                  <MenuItem
                                    bg="brand.secondary"
                                    _hover={{ bg: "gray.700" }}
                                    color="white"
                                    fontSize="sm"
                                    onClick={() =>
                                      updateRole.mutate({
                                        groupSlug: slug,
                                        userId: member.user.id,
                                        role: "ADMIN",
                                      })
                                    }
                                  >
                                    Admin
                                  </MenuItem>
                                  <MenuItem
                                    bg="brand.secondary"
                                    _hover={{ bg: "gray.700" }}
                                    color="white"
                                    fontSize="sm"
                                    onClick={() =>
                                      updateRole.mutate({
                                        groupSlug: slug,
                                        userId: member.user.id,
                                        role: "MEMBER",
                                      })
                                    }
                                  >
                                    Member
                                  </MenuItem>
                                </MenuList>
                              </Menu>
                            ) : (
                              <Badge colorScheme={roleColor[member.role]} fontSize="xs">
                                {member.role}
                              </Badge>
                            )}
                          </Td>
                          {isAdmin && (
                            <Td borderBottom="none">
                              {member.role !== "OWNER" && (
                                <Tooltip label="Remove member">
                                  <IconButton
                                    aria-label="Remove member"
                                    icon={<FaTrash />}
                                    size="xs"
                                    variant="ghost"
                                    color="gray.500"
                                    _hover={{ color: "red.400", bg: "whiteAlpha.100" }}
                                    onClick={() =>
                                      removeMember.mutate({
                                        groupSlug: slug,
                                        userId: member.user.id,
                                      })
                                    }
                                  />
                                </Tooltip>
                              )}
                            </Td>
                          )}
                        </Tr>
                      ))}
                    </Tbody>
                  </Table>
                </Box>
              ) : null}
            </TabPanel>
          </TabPanels>
        </Tabs>
      </Box>

      <AddMemberModal isOpen={isMemberOpen} onClose={onMemberClose} groupSlug={slug} />
      <AddProblemModal
        isOpen={isProblemOpen}
        onClose={onProblemClose}
        groupSlug={slug}
        existingProblemSlugs={problems?.map((p) => p.slug) ?? []}
      />
    </Layout>
  );
}
