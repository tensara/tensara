import {
  Modal,
  ModalOverlay,
  ModalContent,
  ModalHeader,
  ModalBody,
  ModalCloseButton,
  IconButton,
  useDisclosure,
  Text,
  Box,
  Table,
  Tbody,
  Tr,
  Td,
  Tooltip,
} from "@chakra-ui/react";
import { FaInfoCircle } from "react-icons/fa";
import {
  PYTHON_VERSION,
  NVCC_CMD,
  TRITON_VERSION,
  CUDA_RUNTIME_VERSION,
  CUDA_DRIVER_VERSION,
  MOJO_CMD,
} from "~/constants/deviceQuery";

export const LanguageInfoModal = () => {
  const { isOpen, onOpen, onClose } = useDisclosure();

  const TableWrapper = ({ children }: { children: React.ReactNode }) => (
    <Table
      variant="unstyled"
      size="sm"
      sx={{
        "tr:not(:last-child)": {
          borderBottom: "1px solid",
          borderColor: "whiteAlpha.100",
        },
        td: {
          py: 2,
        },
      }}
    >
      {children}
    </Table>
  );

  return (
    <>
      <IconButton
        aria-label="Framework Information"
        icon={<FaInfoCircle />}
        size="sm"
        variant="ghost"
        onClick={onOpen}
        color="gray.400"
        _hover={{ color: "white", bg: "transparent" }}
        bg="transparent"
      />

      <Modal isOpen={isOpen} onClose={onClose} isCentered size="xl">
        <ModalOverlay bg="blackAlpha.800" backdropFilter="blur(5px)" />
        <ModalContent
          bg="brand.secondary"
          borderColor="whiteAlpha.100"
          borderWidth={1}
        >
          <ModalHeader color="white">
            Framework Information
            <Text fontSize="sm" color="gray.400" mt={1}>
              CUDA Driver Version: {CUDA_DRIVER_VERSION} | Runtime Version:{" "}
              {CUDA_RUNTIME_VERSION}
            </Text>
          </ModalHeader>
          <ModalCloseButton color="gray.400" />
          <ModalBody pb={6}>
            <Box mb={6}>
              <Text fontWeight="semibold" mb={3} color="gray.300">
                CUDA C++
              </Text>
              <TableWrapper>
                <Tbody>
                  <Tr>
                    <Td color="gray.400" pl={0}>
                      NVCC Command
                    </Td>
                    <Td color="white">
                      <Tooltip
                        label="-arch and -code are GPU dependent"
                        bg="gray.700"
                        color="gray.200"
                        fontSize="sm"
                      >
                        <code style={{ cursor: "pointer" }}>{NVCC_CMD}</code>
                      </Tooltip>
                    </Td>
                  </Tr>
                </Tbody>
              </TableWrapper>
            </Box>

            <Box mb={6}>
              <Text fontWeight="semibold" mb={3} color="gray.300">
                Python (Triton)
              </Text>
              <TableWrapper>
                <Tbody>
                  <Tr>
                    <Td color="gray.400" pl={0}>
                      Python Version
                    </Td>
                    <Td color="white" textAlign="left">
                      {PYTHON_VERSION}
                    </Td>
                  </Tr>
                  <Tr>
                    <Td color="gray.400" pl={0}>
                      Triton Version
                    </Td>
                    <Td color="white" textAlign="left">
                      {TRITON_VERSION}
                    </Td>
                  </Tr>
                </Tbody>
              </TableWrapper>
            </Box>

            <Box>
              <Text fontWeight="semibold" mb={3} color="gray.300">
                Mojo
              </Text>
              <TableWrapper>
                <Tbody>
                  <Tr>
                    <Td color="gray.400" pl={0}>
                      Build Command
                    </Td>
                    <Td color="white">
                      <code style={{ cursor: "pointer" }}>{MOJO_CMD}</code>
                    </Td>
                  </Tr>
                </Tbody>
              </TableWrapper>
            </Box>
          </ModalBody>
        </ModalContent>
      </Modal>
    </>
  );
};
