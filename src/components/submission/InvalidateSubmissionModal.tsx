import {
  Modal,
  ModalOverlay,
  ModalContent,
  ModalHeader,
  ModalFooter,
  ModalBody,
  ModalCloseButton,
  Button,
  Text,
} from "@chakra-ui/react";

interface InvalidateSubmissionModalProps {
  isOpen: boolean;
  onClose: () => void;
  onInvalidate: () => void;
  isLoading?: boolean;
}

export function InvalidateSubmissionModal({
  isOpen,
  onClose,
  onInvalidate,
  isLoading = false,
}: InvalidateSubmissionModalProps) {
  return (
    <Modal isOpen={isOpen} onClose={onClose} isCentered>
      <ModalOverlay bg="blackAlpha.800" backdropFilter="blur(5px)" />
      <ModalContent
        bg="brand.secondary"
        borderColor="whiteAlpha.100"
        borderWidth={1}
        mx={4}
        maxW="md"
      >
        <ModalHeader color="white">Invalidate Submission</ModalHeader>
        <ModalCloseButton color="gray.400" />
        <ModalBody>
          <Text color="gray.300">
            Are you sure you want to invalidate this submission? It will no
            longer count toward public or competitive views, including the
            leaderboard.
          </Text>
        </ModalBody>

        <ModalFooter gap={3}>
          <Button
            variant="ghost"
            onClick={onClose}
            color="gray.300"
            _hover={{ bg: "whiteAlpha.100" }}
            isDisabled={isLoading}
          >
            Cancel
          </Button>
          <Button
            bg="rgba(239, 68, 68, 0.12)"
            color="rgb(252, 165, 165)"
            _hover={{
              bg: "rgba(239, 68, 68, 0.2)",
            }}
            onClick={onInvalidate}
            isLoading={isLoading}
            loadingText="Invalidating"
          >
            Invalidate Submission
          </Button>
        </ModalFooter>
      </ModalContent>
    </Modal>
  );
}
