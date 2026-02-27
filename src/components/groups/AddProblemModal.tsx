import {
  Modal,
  ModalOverlay,
  ModalContent,
  ModalHeader,
  ModalBody,
  ModalFooter,
  ModalCloseButton,
  Input,
  InputGroup,
  InputLeftElement,
  VStack,
  HStack,
  Text,
  Badge,
  Box,
  Button,
  Checkbox,
  Spinner,
} from "@chakra-ui/react";
import { useState, useMemo, useCallback } from "react";
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
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const utils = api.useUtils();

  const { data: allProblems, isLoading } = api.problems.getAll.useQuery(
    undefined,
    {
      enabled: isOpen,
    }
  );

  const addProblems = api.groups.addProblems.useMutation({
    onSuccess: async () => {
      await utils.groups.getProblems.invalidate({ groupSlug });
      await utils.groups.getBySlug.invalidate({ slug: groupSlug });
      handleClose();
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

  const toggleProblem = useCallback((slug: string) => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(slug)) {
        next.delete(slug);
      } else {
        next.add(slug);
      }
      return next;
    });
  }, []);

  const handleClose = () => {
    setSearch("");
    setSelected(new Set());
    onClose();
  };

  const handleAdd = () => {
    if (selected.size === 0) return;
    addProblems.mutate({ groupSlug, problemSlugs: Array.from(selected) });
  };

  return (
    <Modal
      isOpen={isOpen}
      onClose={handleClose}
      isCentered
      size="lg"
      scrollBehavior="inside"
    >
      <ModalOverlay bg="blackAlpha.700" />
      <ModalContent
        bg="brand.secondary"
        border="1px solid"
        borderColor="whiteAlpha.100"
        maxH="75vh"
      >
        <ModalHeader color="white">
          Add Problems
          {selected.size > 0 && (
            <Badge
              ml={2}
              colorScheme="green"
              fontSize="sm"
              verticalAlign="middle"
            >
              {selected.size} selected
            </Badge>
          )}
        </ModalHeader>
        <ModalCloseButton color="white" />
        <ModalBody pb={2}>
          <VStack spacing={3} align="stretch">
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
                {search
                  ? "No matching problems found"
                  : "All problems have been added"}
              </Text>
            ) : (
              filtered.map((problem) => {
                const isChecked = selected.has(problem.slug);
                return (
                  <Box
                    key={problem.id}
                    px={4}
                    py={3}
                    borderRadius="lg"
                    bg={isChecked ? "whiteAlpha.100" : "whiteAlpha.50"}
                    border="1px solid"
                    borderColor={isChecked ? "brand.primary" : "transparent"}
                    _hover={{ bg: "whiteAlpha.100", cursor: "pointer" }}
                    transition="all 0.15s"
                    onClick={() => toggleProblem(problem.slug)}
                  >
                    <HStack justify="space-between">
                      <HStack spacing={3}>
                        <Checkbox
                          isChecked={isChecked}
                          onChange={() => toggleProblem(problem.slug)}
                          colorScheme="green"
                          borderColor="gray.500"
                          onClick={(e) => e.stopPropagation()}
                        />
                        <Text color="white" fontWeight="medium">
                          {problem.title}
                        </Text>
                      </HStack>
                      <Badge
                        colorScheme={difficultyColor[problem.difficulty]}
                        fontSize="xs"
                      >
                        {problem.difficulty}
                      </Badge>
                    </HStack>
                  </Box>
                );
              })
            )}
          </VStack>
        </ModalBody>
        <ModalFooter borderTop="1px solid" borderColor="whiteAlpha.100">
          <Button variant="ghost" color="gray.400" mr={3} onClick={handleClose}>
            Cancel
          </Button>
          <Button
            bg="brand.primary"
            color="white"
            _hover={{ opacity: 0.9 }}
            onClick={handleAdd}
            isLoading={addProblems.isPending}
            isDisabled={selected.size === 0}
          >
            Add{" "}
            {selected.size > 0
              ? `${selected.size} problem${selected.size > 1 ? "s" : ""}`
              : ""}
          </Button>
        </ModalFooter>
      </ModalContent>
    </Modal>
  );
}
