import {
  Modal,
  ModalOverlay,
  ModalContent,
  ModalHeader,
  ModalBody,
  ModalFooter,
  ModalCloseButton,
  FormControl,
  FormLabel,
  FormHelperText,
  Input,
  Textarea,
  Button,
  VStack,
  Text,
} from "@chakra-ui/react";
import { useState, useEffect } from "react";
import { api } from "~/utils/api";
import { useRouter } from "next/router";

function toSlug(name: string) {
  return name
    .toLowerCase()
    .replace(/[^a-z0-9\s-]/g, "")
    .replace(/\s+/g, "-")
    .replace(/-+/g, "-")
    .replace(/^-|-$/g, "")
    .slice(0, 48);
}

interface CreateGroupModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export function CreateGroupModal({ isOpen, onClose }: CreateGroupModalProps) {
  const router = useRouter();
  const utils = api.useUtils();
  const [name, setName] = useState("");
  const [slug, setSlug] = useState("");
  const [slugManuallyEdited, setSlugManuallyEdited] = useState(false);
  const [description, setDescription] = useState("");

  useEffect(() => {
    if (!slugManuallyEdited) {
      setSlug(toSlug(name));
    }
  }, [name, slugManuallyEdited]);

  const createGroup = api.groups.create.useMutation({
    onSuccess: async (group) => {
      await utils.groups.getMyGroups.invalidate();
      onClose();
      setName("");
      setSlug("");
      setDescription("");
      setSlugManuallyEdited(false);
      void router.push(`/groups/${group.slug}`);
    },
  });

  const handleSubmit = () => {
    if (!name.trim() || !slug.trim()) return;
    createGroup.mutate({
      name: name.trim(),
      slug: slug.trim(),
      description: description.trim() || undefined,
    });
  };

  return (
    <Modal isOpen={isOpen} onClose={onClose} isCentered size="md">
      <ModalOverlay bg="blackAlpha.700" />
      <ModalContent bg="brand.secondary" border="1px solid" borderColor="whiteAlpha.100">
        <ModalHeader color="white">Create a Group</ModalHeader>
        <ModalCloseButton color="white" />
        <ModalBody>
          <VStack spacing={4}>
            <FormControl isRequired>
              <FormLabel color="gray.300">Name</FormLabel>
              <Input
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="My Study Group"
                bg="whiteAlpha.50"
                color="white"
                _hover={{ borderColor: "gray.600" }}
                _focus={{ borderColor: "brand.primary", boxShadow: "none" }}
                maxLength={100}
              />
            </FormControl>

            <FormControl isRequired>
              <FormLabel color="gray.300">Slug</FormLabel>
              <Input
                value={slug}
                onChange={(e) => {
                  setSlug(e.target.value);
                  setSlugManuallyEdited(true);
                }}
                placeholder="my-study-group"
                bg="whiteAlpha.50"
                color="white"
                _hover={{ borderColor: "gray.600" }}
                _focus={{ borderColor: "brand.primary", boxShadow: "none" }}
                maxLength={48}
              />
              <FormHelperText color="gray.500">
                tensara.org/groups/{slug || "..."}
              </FormHelperText>
            </FormControl>

            <FormControl>
              <FormLabel color="gray.300">Description</FormLabel>
              <Textarea
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                placeholder="Optional description..."
                bg="whiteAlpha.50"
                color="white"
                _hover={{ borderColor: "gray.600" }}
                _focus={{ borderColor: "brand.primary", boxShadow: "none" }}
                rows={3}
                maxLength={500}
              />
            </FormControl>

            {createGroup.error && (
              <Text color="red.400" fontSize="sm">
                {createGroup.error.message}
              </Text>
            )}
          </VStack>
        </ModalBody>
        <ModalFooter>
          <Button variant="ghost" color="gray.400" mr={3} onClick={onClose}>
            Cancel
          </Button>
          <Button
            bg="brand.primary"
            color="white"
            _hover={{ opacity: 0.9 }}
            onClick={handleSubmit}
            isLoading={createGroup.isPending}
            isDisabled={!name.trim() || !slug.trim()}
          >
            Create Group
          </Button>
        </ModalFooter>
      </ModalContent>
    </Modal>
  );
}
