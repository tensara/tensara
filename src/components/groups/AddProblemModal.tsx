import {
  Modal,
  ModalOverlay,
  ModalContent,
  ModalHeader,
  ModalBody,
  ModalCloseButton,
  Input,
  InputGroup,
  InputLeftElement,
  VStack,
  HStack,
  Text,
  Badge,
  Box,
  Spinner,
} from "@chakra-ui/react";
import { useState, useMemo } from "react";
import { api } from "~/utils/api";
import { FaSearch } from "react-icons/fa";

const difficultyColor: Record<string, string> = {
  EASY: "green",
  MEDIUM: "yellow",
  HARD: "red",
  EXPERT: "purple",
};

interface AddProblemModalProps {
  isOpen: boolean;
  onClose: () => void;
  groupSlug: string;
  existingProblemSlugs: string[];
}

export function AddProblemModal({
  isOpen,
  onClose,
  groupSlug,
  existingProblemSlugs,
}: AddProblemModalProps) {
  const [search, setSearch] = useState("");
  const utils = api.useUtils();

  const { data: allProblems, isLoading } = api.problems.getAll.useQuery(undefined, {
    enabled: isOpen,
  });

  const addProblem = api.groups.addProblem.useMutation({
    onSuccess: async () => {
      await utils.groups.getProblems.invalidate({ groupSlug });
      await utils.groups.getBySlug.invalidate({ slug: groupSlug });
    },
  });

  const filtered = useMemo(() => {
    if (!allProblems) return [];
    const existing = new Set(existingProblemSlugs);
    return allProblems.filter(
      (p) =>
        !existing.has(p.slug) &&
        p.title.toLowerCase().includes(search.toLowerCase())
    );
  }, [allProblems, existingProblemSlugs, search]);

  const handleAdd = (problemSlug: string) => {
    addProblem.mutate({ groupSlug, problemSlug });
  };

  return (
    <Modal isOpen={isOpen} onClose={onClose} isCentered size="lg" scrollBehavior="inside">
      <ModalOverlay bg="blackAlpha.700" />
      <ModalContent bg="brand.secondary" border="1px solid" borderColor="whiteAlpha.100" maxH="70vh">
        <ModalHeader color="white">Add Problem</ModalHeader>
        <ModalCloseButton color="white" />
        <ModalBody pb={6}>
          <VStack spacing={4} align="stretch">
            <InputGroup>
              <InputLeftElement pointerEvents="none">
                <FaSearch color="#a0aec0" />
              </InputLeftElement>
              <Input
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                placeholder="Search problems..."
                bg="whiteAlpha.50"
                color="white"
                _hover={{ borderColor: "gray.600" }}
                _focus={{ borderColor: "brand.primary", boxShadow: "none" }}
              />
            </InputGroup>

            {isLoading ? (
              <Box textAlign="center" py={8}>
                <Spinner />
              </Box>
            ) : filtered.length === 0 ? (
              <Text color="gray.400" textAlign="center" py={8}>
                {search ? "No matching problems found" : "All problems have been added"}
              </Text>
            ) : (
              filtered.map((problem) => (
                <Box
                  key={problem.id}
                  px={4}
                  py={3}
                  borderRadius="lg"
                  bg="whiteAlpha.50"
                  _hover={{ bg: "whiteAlpha.100", cursor: "pointer" }}
                  transition="all 0.15s"
                  onClick={() => handleAdd(problem.slug)}
                >
                  <HStack justify="space-between">
                    <VStack align="start" spacing={0}>
                      <Text color="white" fontWeight="medium">
                        {problem.title}
                      </Text>
                    </VStack>
                    <Badge colorScheme={difficultyColor[problem.difficulty]} fontSize="xs">
                      {problem.difficulty}
                    </Badge>
                  </HStack>
                </Box>
              ))
            )}
          </VStack>
        </ModalBody>
      </ModalContent>
    </Modal>
  );
}
