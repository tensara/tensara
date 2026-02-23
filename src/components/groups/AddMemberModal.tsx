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
  Input,
  Button,
  Text,
} from "@chakra-ui/react";
import { useState } from "react";
import { api } from "~/utils/api";

interface AddMemberModalProps {
  isOpen: boolean;
  onClose: () => void;
  groupSlug: string;
}

export function AddMemberModal({ isOpen, onClose, groupSlug }: AddMemberModalProps) {
  const [username, setUsername] = useState("");
  const utils = api.useUtils();

  const addMember = api.groups.addMember.useMutation({
    onSuccess: async () => {
      await utils.groups.getMembers.invalidate({ groupSlug });
      await utils.groups.getBySlug.invalidate({ slug: groupSlug });
      onClose();
      setUsername("");
    },
  });

  const handleSubmit = () => {
    if (!username.trim()) return;
    addMember.mutate({ groupSlug, username: username.trim() });
  };

  return (
    <Modal isOpen={isOpen} onClose={onClose} isCentered size="sm">
      <ModalOverlay bg="blackAlpha.700" />
      <ModalContent bg="brand.secondary" border="1px solid" borderColor="whiteAlpha.100">
        <ModalHeader color="white">Add Member</ModalHeader>
        <ModalCloseButton color="white" />
        <ModalBody>
          <FormControl>
            <FormLabel color="gray.300">Username</FormLabel>
            <Input
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              placeholder="Enter GitHub username"
              bg="whiteAlpha.50"
              color="white"
              _hover={{ borderColor: "gray.600" }}
              _focus={{ borderColor: "brand.primary", boxShadow: "none" }}
              onKeyDown={(e) => e.key === "Enter" && handleSubmit()}
            />
          </FormControl>
          {addMember.error && (
            <Text color="red.400" fontSize="sm" mt={2}>
              {addMember.error.message}
            </Text>
          )}
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
            isLoading={addMember.isPending}
            isDisabled={!username.trim()}
          >
            Add
          </Button>
        </ModalFooter>
      </ModalContent>
    </Modal>
  );
}
