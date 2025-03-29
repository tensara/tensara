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

interface ResetCodeModalProps {
  isOpen: boolean;
  onClose: () => void;
  onReset: () => void;
}

const ResetCodeModal = ({ isOpen, onClose, onReset }: ResetCodeModalProps) => {
  return (
    <Modal isOpen={isOpen} onClose={onClose} isCentered>
      <ModalOverlay bg="blackAlpha.800" backdropFilter="blur(5px)" />
      <ModalContent
        bg="gray.800"
        borderColor="whiteAlpha.100"
        borderWidth={1}
        mx={4}
        maxW="md"
      >
        <ModalHeader color="white">Reset Code</ModalHeader>
        <ModalCloseButton color="gray.400" />
        <ModalBody>
          <Text color="gray.300">
            Are you sure you want to reset to the starter code? Your changes
            will be lost.
          </Text>
        </ModalBody>

        <ModalFooter gap={3}>
          <Button
            variant="ghost"
            onClick={onClose}
            color="gray.300"
            _hover={{ bg: "whiteAlpha.100" }}
          >
            Cancel
          </Button>
          <Button
            bg="rgba(34, 197, 94, 0.1)"
            color="rgb(34, 197, 94)"
            _hover={{
              bg: "rgba(34, 197, 94, 0.2)",
            }}
            onClick={() => {
              onReset();
              onClose();
            }}
          >
            Reset Code
          </Button>
        </ModalFooter>
      </ModalContent>
    </Modal>
  );
};

export default ResetCodeModal;
