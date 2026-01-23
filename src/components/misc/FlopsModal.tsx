import {
  Modal,
  ModalOverlay,
  ModalContent,
  ModalHeader,
  ModalBody,
  ModalCloseButton,
  Box,
  Text,
  Link as ChakraLink,
} from "@chakra-ui/react";
import Editor from "@monaco-editor/react";

interface FlopsModalProps {
  isOpen: boolean;
  onClose: () => void;
  problemSlug: string;
  getFlops: string | null | undefined;
}

export const FlopsModal = ({
  isOpen,
  onClose,
  problemSlug,
  getFlops,
}: FlopsModalProps) => {
  return (
    <Modal isOpen={isOpen} onClose={onClose} isCentered size="4xl">
      <ModalOverlay bg="blackAlpha.800" backdropFilter="blur(5px)" />
      <ModalContent
        bg="brand.secondary"
        borderColor="whiteAlpha.100"
        borderWidth={1}
      >
        <ModalHeader color="white">FLOPs Calculation</ModalHeader>
        <ModalCloseButton color="gray.400" />
        <ModalBody pb={6}>
          <Box>
            <Text
              fontSize="sm"
              color="blue.300"
              as="span"
              fontWeight="semibold"
            >
              <ChakraLink
                href={`https://github.com/tensara/problems/blob/main/problems/${problemSlug}/def.py`}
                target="_blank"
                color="blue.300"
                fontWeight="semibold"
                textDecoration="underline"
              >
                View Problem Source
              </ChakraLink>
              <Text as="span" color="gray.400" mx={2}>
                |
              </Text>
              <Text as="span" color="gray.100">
                Found an error?&nbsp;
                <ChakraLink
                  href="https://github.com/tensara/problems/issues"
                  target="_blank"
                  color="blue.300"
                  textDecoration="underline"
                >
                  Report it here
                </ChakraLink>
              </Text>
            </Text>
          </Box>
          {getFlops && (
            <Box
              borderRadius="md"
              overflow="hidden"
              border="1px solid"
              borderColor="whiteAlpha.200"
              mt={4}
            >
              {(() => {
                const lines = getFlops.split("\n").length;
                const height = Math.min(lines * 20 + 100, 600);
                return (
                  <Editor
                    height={height}
                    language="python"
                    value={getFlops}
                    theme="tensara-dark"
                    options={{
                      readOnly: true,
                      minimap: { enabled: false },
                      fontSize: 14,
                      lineNumbers: "on",
                      scrollBeyondLastLine: false,
                      fontFamily: "JetBrains Mono, monospace",
                    }}
                  />
                );
              })()}
            </Box>
          )}
        </ModalBody>
      </ModalContent>
    </Modal>
  );
};
