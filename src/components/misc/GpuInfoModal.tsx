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
  VStack,
  Tabs,
  TabList,
  TabPanels,
  Tab,
  TabPanel,
  Grid,
  GridItem,
} from "@chakra-ui/react";
import { FaInfoCircle } from "react-icons/fa";

import {
  CUDA_DRIVER_VERSION,
  CUDA_RUNTIME_VERSION,
  DEVICE_QUERY_GPU_MAP,
} from "~/constants/deviceQuery";
import { GPU_DISPLAY_NAMES } from "~/constants/gpu";

export const GpuInfoModal = () => {
  const { isOpen, onOpen, onClose } = useDisclosure();

  const formatBytes = (bytes: number, decimals = 2) => {
    if (bytes === 0) return "0 Bytes";
    const k = 1024;
    const sizes = ["Bytes", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return `${parseFloat((bytes / Math.pow(k, i)).toFixed(decimals))} ${sizes[i]}`;
  };

  const formatArray = (arr: number[]) => arr.join(" × ");

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

  const renderGpuInfo = (gpuType: keyof typeof DEVICE_QUERY_GPU_MAP) => {
    const gpuInfo = DEVICE_QUERY_GPU_MAP[gpuType]!;

    return (
      <Grid templateColumns="repeat(2, 1fr)" gap={8}>
        <GridItem>
          <VStack spacing={8} align="stretch">
            <Box>
              <Text fontWeight="semibold" mb={3} color="gray.300">
                Core Specifications
              </Text>
              <TableWrapper>
                <Tbody>
                  <Tr>
                    <Td color="gray.400" pl={0}>
                      GPU Model
                    </Td>
                    <Td color="white" textAlign="right">
                      {gpuInfo.name}
                    </Td>
                  </Tr>
                  <Tr>
                    <Td color="gray.400" pl={0}>
                      CUDA Cores
                    </Td>
                    <Td color="white" textAlign="right" isNumeric>
                      {gpuInfo.totalCUDACores.toLocaleString()}
                    </Td>
                  </Tr>
                  <Tr>
                    <Td color="gray.400" pl={0}>
                      Multiprocessors
                    </Td>
                    <Td color="white" textAlign="right" isNumeric>
                      {gpuInfo.multiprocessors} ({gpuInfo.cudaCoresPerMP} cores
                      each)
                    </Td>
                  </Tr>
                  <Tr>
                    <Td color="gray.400" pl={0}>
                      GPU Clock Speed
                    </Td>
                    <Td color="white" textAlign="right" isNumeric>
                      {gpuInfo.gpuMaxClockRate} MHz
                    </Td>
                  </Tr>
                </Tbody>
              </TableWrapper>
            </Box>

            <Box>
              <Text fontWeight="semibold" mb={3} color="gray.300">
                Memory Specifications
              </Text>
              <TableWrapper>
                <Tbody>
                  <Tr>
                    <Td color="gray.400" pl={0}>
                      Global Memory
                    </Td>
                    <Td color="white" textAlign="right" isNumeric>
                      {formatBytes(gpuInfo.globalMemory)}
                    </Td>
                  </Tr>
                  <Tr>
                    <Td color="gray.400" pl={0}>
                      Memory Clock
                    </Td>
                    <Td color="white" textAlign="right" isNumeric>
                      {gpuInfo.memoryClockRate} MHz
                    </Td>
                  </Tr>
                  <Tr>
                    <Td color="gray.400" pl={0}>
                      Memory Bus Width
                    </Td>
                    <Td color="white" textAlign="right" isNumeric>
                      {gpuInfo.memoryBusWidth} bits
                    </Td>
                  </Tr>
                  <Tr>
                    <Td color="gray.400" pl={0}>
                      L2 Cache Size
                    </Td>
                    <Td color="white" textAlign="right" isNumeric>
                      {formatBytes(gpuInfo.l2CacheSize)}
                    </Td>
                  </Tr>
                </Tbody>
              </TableWrapper>
            </Box>

            <Box>
              <Text fontWeight="semibold" mb={3} color="gray.300">
                CUDA Capabilities
              </Text>
              <TableWrapper>
                <Tbody>
                  <Tr>
                    <Td color="gray.400" pl={0}>
                      Compute Capability
                    </Td>
                    <Td color="white" textAlign="right">
                      {gpuInfo.cudaCapability.major}.
                      {gpuInfo.cudaCapability.minor}
                    </Td>
                  </Tr>
                  <Tr>
                    <Td color="gray.400" pl={0}>
                      Shared Memory per Block
                    </Td>
                    <Td color="white" textAlign="right" isNumeric>
                      {formatBytes(gpuInfo.memory.sharedMemoryPerBlock)}
                    </Td>
                  </Tr>
                  <Tr>
                    <Td color="gray.400" pl={0}>
                      Shared Memory per MP
                    </Td>
                    <Td color="white" textAlign="right" isNumeric>
                      {formatBytes(gpuInfo.memory.sharedMemoryPerMP)}
                    </Td>
                  </Tr>
                  <Tr>
                    <Td color="gray.400" pl={0}>
                      Constant Memory
                    </Td>
                    <Td color="white" textAlign="right" isNumeric>
                      {formatBytes(gpuInfo.memory.constantMemory)}
                    </Td>
                  </Tr>
                </Tbody>
              </TableWrapper>
            </Box>
          </VStack>
        </GridItem>

        <GridItem>
          <VStack spacing={8} align="stretch">
            <Box>
              <Text fontWeight="semibold" mb={3} color="gray.300">
                Thread & Block Specifications
              </Text>
              <TableWrapper>
                <Tbody>
                  <Tr>
                    <Td color="gray.400" pl={0}>
                      Warp Size
                    </Td>
                    <Td color="white" textAlign="right" isNumeric>
                      {gpuInfo.warpSize} threads
                    </Td>
                  </Tr>
                  <Tr>
                    <Td color="gray.400" pl={0}>
                      Max Threads per MP
                    </Td>
                    <Td color="white" textAlign="right" isNumeric>
                      {gpuInfo.threads.maxPerMP}
                    </Td>
                  </Tr>
                  <Tr>
                    <Td color="gray.400" pl={0}>
                      Max Threads per Block
                    </Td>
                    <Td color="white" textAlign="right" isNumeric>
                      {gpuInfo.threads.maxPerBlock}
                    </Td>
                  </Tr>
                  <Tr>
                    <Td color="gray.400" pl={0}>
                      Max Block Dimensions
                    </Td>
                    <Td color="white" textAlign="right" isNumeric>
                      {formatArray(gpuInfo.threads.maxBlockDim)}
                    </Td>
                  </Tr>
                  <Tr>
                    <Td color="gray.400" pl={0}>
                      Max Grid Dimensions
                    </Td>
                    <Td color="white" textAlign="right" isNumeric>
                      {formatArray(gpuInfo.threads.maxGridDim)}
                    </Td>
                  </Tr>
                </Tbody>
              </TableWrapper>
            </Box>

            <Box>
              <Text fontWeight="semibold" mb={3} color="gray.300">
                Texture Capabilities
              </Text>
              <TableWrapper>
                <Tbody>
                  <Tr>
                    <Td color="gray.400" pl={0}>
                      Max 1D Texture
                    </Td>
                    <Td color="white" textAlign="right" isNumeric>
                      {gpuInfo.textureDimensions.max1D}
                    </Td>
                  </Tr>
                  <Tr>
                    <Td color="gray.400" pl={0}>
                      Max 2D Texture
                    </Td>
                    <Td color="white" textAlign="right" isNumeric>
                      {formatArray(gpuInfo.textureDimensions.max2D)}
                    </Td>
                  </Tr>
                  <Tr>
                    <Td color="gray.400" pl={0}>
                      Max 3D Texture
                    </Td>
                    <Td color="white" textAlign="right" isNumeric>
                      {formatArray(gpuInfo.textureDimensions.max3D)}
                    </Td>
                  </Tr>
                  <Tr>
                    <Td color="gray.400" pl={0}>
                      Texture Alignment
                    </Td>
                    <Td color="white" textAlign="right" isNumeric>
                      {formatBytes(gpuInfo.textureAlignment)}
                    </Td>
                  </Tr>
                </Tbody>
              </TableWrapper>
            </Box>
          </VStack>
        </GridItem>
      </Grid>
    );
  };

  return (
    <>
      <IconButton
        aria-label="GPU Information"
        icon={<FaInfoCircle />}
        size="sm"
        variant="ghost"
        onClick={onOpen}
        color="gray.400"
        _hover={{ color: "white", bg: "transparent" }}
        bg="transparent"
      />

      <Modal isOpen={isOpen} onClose={onClose} isCentered size="6xl">
        <ModalOverlay bg="blackAlpha.800" backdropFilter="blur(5px)" />
        <ModalContent
          bg="gray.800"
          borderColor="whiteAlpha.100"
          borderWidth={1}
          maxH="90vh"
        >
          <ModalHeader color="white" px={8} pt={6}>
            GPU Specifications
            <Text fontSize="sm" color="gray.400" mt={1}>
              CUDA Driver Version: {CUDA_DRIVER_VERSION} | Runtime Version:{" "}
              {CUDA_RUNTIME_VERSION}
            </Text>
          </ModalHeader>
          <ModalCloseButton color="gray.400" mr={2} mt={2} />
          <ModalBody pb={8} px={8} overflowY="auto">
            <Tabs variant="line" colorScheme="green">
              <TabList borderBottomColor="whiteAlpha.200" mb={6}>
                {Object.entries(DEVICE_QUERY_GPU_MAP).map(([key]) => (
                  <Tab
                    key={key}
                    color="gray.400"
                    _selected={{
                      color: "white",
                      borderColor: "green.400",
                      bg: "transparent",
                    }}
                    _hover={{
                      color: "white",
                      bg: "transparent",
                    }}
                  >
                    {GPU_DISPLAY_NAMES[key]}
                  </Tab>
                ))}
              </TabList>
              <TabPanels>
                {Object.keys(DEVICE_QUERY_GPU_MAP).map((key) => (
                  <TabPanel key={key} p={0}>
                    {renderGpuInfo(key)}
                  </TabPanel>
                ))}
              </TabPanels>
            </Tabs>
          </ModalBody>
        </ModalContent>
      </Modal>
    </>
  );
};
