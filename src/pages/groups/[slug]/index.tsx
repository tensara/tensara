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
  SimpleGrid,
  Card,
  CardHeader,
  CardBody,
  Flex,
  Icon,
  Link as ChakraLink,
  useClipboard,
  Input,
  InputGroup,
  InputRightElement,
  Popover,
  PopoverTrigger,
  PopoverContent,
  PopoverBody,
} from "@chakra-ui/react";
import { Layout } from "~/components/layout";
import { api } from "~/utils/api";
import { useRouter } from "next/router";
import Link from "next/link";
import { useState } from "react";
import {
  FaPlus,
  FaUserPlus,
  FaCog,
  FaTrash,
  FaCheckCircle,
  FaChevronDown,
  FaExternalLinkAlt,
  FaExclamationCircle,
} from "react-icons/fa";
import { FiArrowLeft, FiCopy, FiCheck, FiLink } from "react-icons/fi";
import { AddMemberModal } from "~/components/groups/AddMemberModal";
import { AddProblemModal } from "~/components/groups/AddProblemModal";
import { GPU_DISPLAY_NAMES } from "~/constants/gpu";
import { LANGUAGE_DISPLAY_NAMES } from "~/constants/language";
const formatRuntime = (runtime: number | null | undefined): string => {
  if (runtime == null) return "N/A";
  if (runtime <= 1) return `${(runtime * 1000).toFixed(2)} μs`;
  if (runtime >= 1000) return `${(runtime / 1000).toFixed(2)} s`;
  return `${runtime.toFixed(2)} ms`;
};

const formatGFLOPS = (gflops: number | null | undefined): string => {
  if (gflops == null) return "N/A";
  if (gflops >= 1000) return `${(gflops / 1000).toFixed(2)} TFLOPS`;
  return `${gflops.toFixed(2)} GFLOPS`;
};

const getMedalColor = (index: number): string => {
  switch (index) {
    case 0: return "#FFD700";
    case 1: return "#C0C0C0";
    case 2: return "#CD7F32";
    default: return "white.800";
  }
};

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

  const [selectedGpu, setSelectedGpu] = useState("all");

  const { data: leaderboardCards, isLoading: leaderboardLoading } = api.groups.getLeaderboardCards.useQuery(
    { groupSlug: slug, gpuType: selectedGpu },
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

  const inviteLink = (() => {
    if (!group?.inviteCode) return "";
    const origin = typeof window !== "undefined" ? window.location.origin : "https://tensara.org";
    return `${origin}/groups/${group.slug}/join?code=${group.inviteCode}`;
  })();
  const { hasCopied, onCopy } = useClipboard(inviteLink);

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
    <Layout
      title={group.name}
      ogTitle={group.name}
      ogDescription={group.description || `A group on Tensara with ${group.memberCount} members and ${group.problemCount} GPU programming problems.`}
      ogImgSubtitle={`${group.memberCount} members · ${group.problemCount} problems`}
    >
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
                {group.memberCount} members &middot; {group.problemCount} problems
              </Text>
            </VStack>

            <HStack spacing={2}>
              {isAdmin && group.inviteCode && (
                <Popover placement="bottom-end">
                  <PopoverTrigger>
                    <Button
                      size="sm"
                      leftIcon={<FiLink />}
                      variant="outline"
                      color="gray.300"
                      borderColor="whiteAlpha.200"
                      _hover={{ bg: "whiteAlpha.100", borderColor: "whiteAlpha.400" }}
                    >
                      Invite
                    </Button>
                  </PopoverTrigger>
                  <PopoverContent
                    bg="brand.secondary"
                    border="1px solid"
                    borderColor="whiteAlpha.200"
                    w="360px"
                    _focus={{ boxShadow: "none" }}
                  >
                    <PopoverBody p={3}>
                      <Text color="gray.400" fontSize="xs" mb={2}>
                        Share this link to invite people
                      </Text>
                      <InputGroup size="sm">
                        <Input
                          value={inviteLink}
                          isReadOnly
                          bg="whiteAlpha.50"
                          color="white"
                          fontSize="xs"
                          pr="2.5rem"
                          borderColor="whiteAlpha.100"
                          _hover={{ borderColor: "whiteAlpha.300" }}
                          _focus={{ borderColor: "brand.primary", boxShadow: "none" }}
                        />
                        <InputRightElement>
                          <IconButton
                            aria-label="Copy invite link"
                            icon={hasCopied ? <FiCheck /> : <FiCopy />}
                            size="xs"
                            variant="ghost"
                            color={hasCopied ? "green.400" : "gray.400"}
                            _hover={{ color: "white" }}
                            onClick={onCopy}
                          />
                        </InputRightElement>
                      </InputGroup>
                    </PopoverBody>
                  </PopoverContent>
                </Popover>
              )}
              {isAdmin && (
                <Link href={`/groups/${slug}/settings`} passHref>
                  <IconButton
                    aria-label="Settings"
                    icon={<FaCog />}
                    variant="ghost"
                    color="gray.400"
                    _hover={{ color: "white", bg: "whiteAlpha.100" }}
                  />
                </Link>
              )}
            </HStack>
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
              Problems
            </Tab>
            <Tab
              color="gray.400"
              _selected={{ color: "white", borderBottom: "2px solid", borderColor: "brand.primary" }}
              pb={3}
              mr={4}
            >
              Members
            </Tab>
            <Tab
              color="gray.400"
              _selected={{ color: "white", borderBottom: "2px solid", borderColor: "brand.primary" }}
              pb={3}
            >
              Leaderboards
            </Tab>
          </TabList>

          <TabPanels>
            {/* Problems Tab */}
            <TabPanel px={0}>
              {isAdmin && (
                <Box mb={4}>
                  <Button
                    size="sm"
                    bg="rgba(34, 197, 94, 0.1)"
                    color="rgb(34, 197, 94)"
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
                        bg="rgba(34, 197, 94, 0.1)"
                        color="rgb(34, 197, 94)"
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
                    bg="rgba(34, 197, 94, 0.1)"
                    color="rgb(34, 197, 94)"
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

            {/* Leaderboards Tab */}
            <TabPanel px={0}>
              <HStack mb={4} justify="flex-end">
                <Menu>
                  <MenuButton
                    as={Button}
                    rightIcon={<FaChevronDown color="#d4d4d8" size={10} />}
                    bg="whiteAlpha.50"
                    _hover={{ bg: "whiteAlpha.100", borderColor: "gray.600" }}
                    _active={{ bg: "whiteAlpha.150" }}
                    _focus={{ borderColor: "blue.500", boxShadow: "none" }}
                    color="white"
                    w="200px"
                    fontWeight="normal"
                    textAlign="left"
                    justifyContent="flex-start"
                  >
                    {GPU_DISPLAY_NAMES[selectedGpu] ?? "All GPUs"}
                  </MenuButton>
                  <MenuList bg="brand.secondary" borderColor="gray.800" p={0} minW="200px">
                    {Object.entries(GPU_DISPLAY_NAMES).map(([key, value]) => (
                      <MenuItem
                        key={key}
                        onClick={() => setSelectedGpu(key)}
                        bg="brand.secondary"
                        _hover={{ bg: "gray.700" }}
                        color="white"
                        borderRadius="md"
                        fontSize="sm"
                      >
                        {value}
                      </MenuItem>
                    ))}
                  </MenuList>
                </Menu>
              </HStack>

              {leaderboardLoading ? (
                <Box textAlign="center" py={12}>
                  <Spinner />
                </Box>
              ) : leaderboardCards && leaderboardCards.length > 0 ? (
                <SimpleGrid columns={{ base: 1, md: 2, lg: 3 }} spacing={6} minChildWidth="300px">
                  {leaderboardCards.map((problem) => (
                    <Card
                      key={problem.slug}
                      bg="brand.secondary"
                      borderColor="whiteAlpha.200"
                      borderWidth={1}
                      transition="transform 0.2s, box-shadow 0.2s"
                      _hover={{ transform: "translateY(-2px)", boxShadow: "lg" }}
                    >
                      <CardHeader pb={2}>
                        <Flex gap={3}>
                          <ChakraLink
                            as={Link}
                            href={`/groups/${slug}/leaderboard/${problem.slug}`}
                            _hover={{ textDecoration: "none" }}
                          >
                            <Heading size="md" color="white" _hover={{ color: "blue.400" }}>
                              {problem.title}
                            </Heading>
                          </ChakraLink>
                          <ChakraLink href={`/problems/${problem.slug}`} isExternal>
                            <Icon
                              as={FaExternalLinkAlt}
                              color="gray.400"
                              boxSize={3}
                              _hover={{ color: "blue.400" }}
                            />
                          </ChakraLink>
                        </Flex>
                      </CardHeader>
                      <CardBody pt={2}>
                        {problem.topSubmissions.length === 0 ? (
                          <Flex
                            direction="column"
                            align="center"
                            justify="center"
                            p={4}
                            minH="120px"
                            bg="whiteAlpha.50"
                            borderRadius="md"
                          >
                            <Icon as={FaExclamationCircle} color="whiteAlpha.600" mb={2} />
                            <Text color="whiteAlpha.700" textAlign="center">
                              No submissions yet
                              {selectedGpu !== "all" ? ` for ${GPU_DISPLAY_NAMES[selectedGpu]}` : ""}
                            </Text>
                          </Flex>
                        ) : (
                          <Table variant="unstyled" size="sm">
                            <Thead>
                              <Tr>
                                <Th pl={2} color="whiteAlpha.600">Rank</Th>
                                <Th color="whiteAlpha.600">User</Th>
                                <Th isNumeric color="whiteAlpha.600">Time</Th>
                              </Tr>
                            </Thead>
                            <Tbody>
                              {problem.topSubmissions.map((submission, index) => (
                                <Tr
                                  key={submission.id}
                                  onClick={() => void router.push(`/submissions/${submission.id}`)}
                                  cursor="pointer"
                                  _hover={{ bg: "whiteAlpha.50" }}
                                  borderRadius="lg"
                                  transition="background 0.15s"
                                >
                                  <Td pl={2} borderLeftRadius="lg">
                                    <Text color="whiteAlpha.600">#{index + 1}</Text>
                                  </Td>
                                  <Td color="white">
                                    <Tooltip
                                      label={`${LANGUAGE_DISPLAY_NAMES[submission.language ?? ""] ?? "Unknown"} | ${GPU_DISPLAY_NAMES[submission.gpuType ?? ""] ?? "Unknown GPU"}`}
                                      hasArrow
                                    >
                                      <ChakraLink
                                        as={Link}
                                        href={`/user/${submission.username ?? "anonymous"}`}
                                        onClick={(e) => e.stopPropagation()}
                                        _hover={{ color: "blue.400" }}
                                      >
                                        {submission.username ?? "Anonymous"}
                                      </ChakraLink>
                                    </Tooltip>
                                  </Td>
                                  <Td isNumeric borderRightRadius="lg">
                                    <Tooltip label={formatGFLOPS(submission.gflops)} hasArrow>
                                      <Text
                                        style={{
                                          color: getMedalColor(index),
                                          fontWeight: "bold",
                                          fontSize: "0.875rem",
                                          fontVariantNumeric: "tabular-nums",
                                          display: "inline-block",
                                        }}
                                      >
                                        {formatRuntime(submission.runtime)}
                                      </Text>
                                    </Tooltip>
                                  </Td>
                                </Tr>
                              ))}
                            </Tbody>
                          </Table>
                        )}
                      </CardBody>
                    </Card>
                  ))}
                </SimpleGrid>
              ) : (
                <Box
                  bg="brand.secondary"
                  borderRadius="xl"
                  border="1px solid"
                  borderColor="whiteAlpha.100"
                  py={12}
                  textAlign="center"
                >
                  <Text color="gray.400">No problems added yet. Add problems to see leaderboards.</Text>
                </Box>
              )}
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
