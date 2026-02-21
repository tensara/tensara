import {
  Box,
  Table,
  Thead,
  Tbody,
  Tr,
  Th,
  Td,
  Text,
  Badge,
  Tooltip,
  Spinner,
  Flex,
  Heading,
  HStack,
  Button,
  Image,
  Menu,
  MenuButton,
  MenuList,
  MenuItem,
} from "@chakra-ui/react";
import { useState } from "react";
import { api } from "~/utils/api";
import { Layout } from "~/components/layout";
import { useRouter } from "next/router";
import Link from "next/link";
import { formatDistanceToNow, format } from "date-fns";
import { GPU_DISPLAY_NAMES } from "~/constants/gpu";
import { FaChevronDown } from "react-icons/fa";
import { FiArrowLeft } from "react-icons/fi";

const LanguageLogo = ({ language }: { language: string | null }) => {
  if (!language) return <Text color="gray.400">-</Text>;

  const logoMap: Record<string, { src?: string; emoji?: string; label: string }> = {
    cuda: { src: "/cuda-icon.svg", label: "CUDA C++" },
    python: { src: "/triton-logo.png", label: "Triton" },
    mojo: { emoji: "🔥", label: "Mojo" },
    cute: { emoji: "🧩", label: "CuTe DSL" },
    cutile: { emoji: "🧱", label: "CuTile" },
  };

  const logo = logoMap[language];
  if (!logo) {
    return (
      <Text fontSize="sm" color="gray.400">
        {language}
      </Text>
    );
  }

  return (
    <Tooltip label={logo.label}>
      <Flex align="center" justify="center">
        {logo.src ? (
          <Image src={logo.src} alt={logo.label} boxSize="24px" objectFit="contain" />
        ) : (
          <Text fontSize="xl" lineHeight={1}>
            {logo.emoji}
          </Text>
        )}
      </Flex>
    </Tooltip>
  );
};

export default function GroupProblemLeaderboardPage() {
  const router = useRouter();
  const { slug, problemSlug } = router.query as {
    slug: string;
    problemSlug: string;
  };

  const [selectedGpu, setSelectedGpu] = useState("all");

  const { data: group } = api.groups.getBySlug.useQuery(
    { slug },
    { enabled: !!slug }
  );

  const { data: problem } = api.problems.getById.useQuery(
    { slug: problemSlug },
    { enabled: !!problemSlug }
  );

  const { data: entries, isLoading } = api.groups.getProblemLeaderboard.useQuery(
    { groupSlug: slug, problemSlug, gpuType: selectedGpu },
    { enabled: !!slug && !!problemSlug }
  );

  if (isLoading) {
    return (
      <Layout title="Group Leaderboard">
        <Box display="flex" justifyContent="center" alignItems="center" h="50vh">
          <Spinner size="xl" />
        </Box>
      </Layout>
    );
  }

  return (
    <Layout
      title={
        problem?.title && group?.name
          ? `${problem.title} — ${group.name}`
          : "Group Leaderboard"
      }
    >
      <Box maxW="7xl" mx="auto" px={4} py={8}>
        <Flex direction="column" gap={6}>
          <Link href={`/groups/${slug}`} passHref>
            <HStack
              spacing={1}
              color="gray.400"
              _hover={{ color: "white" }}
              cursor="pointer"
              w="fit-content"
            >
              <FiArrowLeft />
              <Text fontSize="sm">Back to {group?.name ?? "group"}</Text>
            </HStack>
          </Link>

          <HStack justify="space-between" align="center">
            <Heading size="lg" color="white">
              {problem?.title ?? "Leaderboard"}
            </Heading>

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
              <MenuList bg="brand.secondary" borderColor="gray.800" p={0}>
                {Object.entries(GPU_DISPLAY_NAMES).map(([key, value]) => (
                  <MenuItem
                    key={key}
                    onClick={() => setSelectedGpu(key)}
                    bg="brand.secondary"
                    _hover={{ bg: "gray.700" }}
                    color="white"
                    borderRadius="md"
                  >
                    {value}
                  </MenuItem>
                ))}
              </MenuList>
            </Menu>
          </HStack>

          {entries && entries.length > 0 ? (
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
                      borderBottom="1px solid"
                      borderColor="brand.primary"
                      py={4}
                      w="60px"
                    >
                      Rank
                    </Th>
                    <Th
                      color="gray.300"
                      borderBottom="1px solid"
                      borderColor="brand.primary"
                      py={4}
                    >
                      User
                    </Th>
                    <Th
                      color="gray.300"
                      borderBottom="1px solid"
                      borderColor="brand.primary"
                      py={4}
                      isNumeric
                    >
                      Runtime (ms)
                    </Th>
                    <Th
                      color="gray.300"
                      borderBottom="1px solid"
                      borderColor="brand.primary"
                      py={4}
                      isNumeric
                    >
                      GFLOP/s
                    </Th>
                    {selectedGpu === "all" && (
                      <Th
                        color="gray.300"
                        borderBottom="1px solid"
                        borderColor="brand.primary"
                        py={4}
                      >
                        GPU
                      </Th>
                    )}
                    <Th
                      color="gray.300"
                      borderBottom="1px solid"
                      borderColor="brand.primary"
                      py={4}
                    >
                      Language
                    </Th>
                    <Th
                      color="gray.300"
                      borderBottom="1px solid"
                      borderColor="brand.primary"
                      py={4}
                    >
                      Submitted
                    </Th>
                  </Tr>
                </Thead>
                <Tbody>
                  {entries.map((entry) => {
                    const medalColor =
                      entry.rank <= 3
                        ? entry.rank === 1
                          ? "#FFD700"
                          : entry.rank === 2
                            ? "#C0C0C0"
                            : "#CD7F32"
                        : undefined;

                    return (
                      <Tr
                        key={entry.submissionId}
                        cursor="pointer"
                        onClick={() => void router.push(`/submissions/${entry.submissionId}`)}
                        _hover={{ bg: "whiteAlpha.100" }}
                        borderBottom="1px solid"
                        borderColor="whiteAlpha.100"
                        bg={
                          medalColor
                            ? `rgba(${medalColor
                                .replace("#", "")
                                .match(/../g)
                                ?.map((hex) => parseInt(hex, 16))
                                .join(",") ?? "0,0,0"}, 0.08)`
                            : "brand.secondary"
                        }
                      >
                        <Td borderBottom="none" py={3} w="60px">
                          <Text
                            color={medalColor ?? "gray.300"}
                            fontWeight={medalColor ? "bold" : "normal"}
                          >
                            {entry.rank}
                          </Text>
                        </Td>
                        <Td borderBottom="none" py={3}>
                          <HStack spacing={3}>
                            <Image
                              src={entry.image ?? ""}
                              alt={entry.username ?? ""}
                              w={7}
                              h={7}
                              borderRadius="full"
                              fallbackSrc="https://via.placeholder.com/28"
                            />
                            <Text
                              color={medalColor ?? "white"}
                              fontWeight={medalColor ? "bold" : "normal"}
                            >
                              {entry.username ?? "Anonymous"}
                            </Text>
                          </HStack>
                        </Td>
                        <Td borderBottom="none" py={3} isNumeric>
                          <Text
                            color={medalColor ?? "white"}
                            fontWeight={entry.rank <= 3 ? "bold" : "normal"}
                            style={{ fontVariantNumeric: "tabular-nums" }}
                          >
                            {entry.runtime?.toFixed(2) ?? "N/A"}
                          </Text>
                        </Td>
                        <Td borderBottom="none" py={3} isNumeric>
                          <Text
                            color="gray.300"
                            style={{ fontVariantNumeric: "tabular-nums" }}
                          >
                            {entry.gflops?.toFixed(1) ?? "-"}
                          </Text>
                        </Td>
                        {selectedGpu === "all" && (
                          <Td borderBottom="none" py={3}>
                            <Badge
                              bg="whiteAlpha.200"
                              color="white"
                              px={2}
                              py={0.5}
                              borderRadius="md"
                              fontSize="xs"
                            >
                              {entry.gpuType}
                            </Badge>
                          </Td>
                        )}
                        <Td borderBottom="none" py={3}>
                          <LanguageLogo language={entry.language} />
                        </Td>
                        <Td borderBottom="none" py={3}>
                          <Tooltip
                            label={format(
                              new Date(entry.createdAt),
                              "MM/dd/yyyy, h:mm:ss a"
                            )}
                          >
                            <Text color="gray.400" fontSize="sm">
                              {formatDistanceToNow(new Date(entry.createdAt))} ago
                            </Text>
                          </Tooltip>
                        </Td>
                      </Tr>
                    );
                  })}
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
              <Text color="gray.400">
                No submissions yet
                {selectedGpu !== "all" && ` for ${GPU_DISPLAY_NAMES[selectedGpu]}`}.
              </Text>
            </Box>
          )}
        </Flex>
      </Box>
    </Layout>
  );
}
