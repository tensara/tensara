import React, { useState } from "react";
import {
  Box,
  Text,
  VStack,
  HStack,
  Button,
  Icon,
  Heading,
  Menu,
  MenuButton,
  MenuList,
  MenuItem,
  IconButton,
  Tooltip,
} from "@chakra-ui/react";

import {
  FaPlay,
  FaSave,
  FaShare,
  FaChevronDown,
  FaExclamationCircle,
} from "react-icons/fa";

import { Layout } from "~/components/layout";
import CodeEditor from "~/components/problem/CodeEditor";
import { type ProgrammingLanguage } from "~/types/misc";

export default function SandboxIndex() {
  const [selectedGpu, setSelectedGpu] = useState("Tesla T4");
  const [selectedLanguage, setSelectedLanguage] = useState("python");
  const [isRunning] = useState(false);

  const [code, setCode] = useState(`import torch
import numpy as np

# GPU-accelerated computation example
def matrix_multiply_gpu():
    # Create random matrices
    a = torch.randn(1000, 1000).cuda()
    b = torch.randn(1000, 1000).cuda()
    
    # Perform matrix multiplication
    result = torch.matmul(a, b)
    
    print(f"Matrix shape: {result.shape}")
    print(f"GPU memory used: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    return result

# Run the computation
if __name__ == "__main__":
    if torch.cuda.is_available():
        print("🚀 GPU detected! Running computation...")
        result = matrix_multiply_gpu()
        print("✅ Computation completed successfully!")
    else:
        print("❌ No GPU available")`);

  const mobileWarning = (
    <Box
      display={{ base: "block", md: "none" }}
      w="100%"
      p={6}
      bg="whiteAlpha.50"
      borderRadius="xl"
      mb={4}
    >
      <VStack spacing={4} align="center">
        <Icon as={FaExclamationCircle} boxSize={10} color="yellow.400" />
        <Heading size="md" textAlign="center">
          Desktop Required for GPU Computing
        </Heading>
        <Text textAlign="center" color="whiteAlpha.800">
          For the best experience with GPU acceleration, please switch to a
          desktop device.
        </Text>
      </VStack>
    </Box>
  );

  return (
    <Layout
      title="Tensara Sandbox"
      ogTitle="Tensara Sandbox - Interactive Computing Environment"
      ogImgSubtitle="GPU Computing | Tensara"
    >
      <Box
        bg="brand.secondary"
        borderRadius="xl"
        border="1px solid"
        borderColor="gray.800"
        h="100%"
        p={{ base: 3, md: 4 }}
        overflow="auto"
      >
        {mobileWarning}

        <VStack w="100%" h="100%" spacing={4}>
          {/* Header */}
          <HStack justify="space-between" w="100%">
            <HStack spacing={4}>
              <Heading size="lg" color="whiteAlpha.900">
                Tensara Sandbox
              </Heading>
            </HStack>

            <HStack spacing={3}>
              <Text fontSize="sm" color="whiteAlpha.600">
                GPU Type
              </Text>
              <Menu>
                <MenuButton
                  as={Button}
                  rightIcon={<FaChevronDown />}
                  size="sm"
                  variant="outline"
                  borderColor="whiteAlpha.300"
                  color="whiteAlpha.900"
                  _hover={{ bg: "whiteAlpha.100" }}
                >
                  {selectedGpu}
                </MenuButton>
                <MenuList bg="gray.800" borderColor="gray.700">
                  {["Tesla T4", "Tesla V100", "A100 SXM4", "H100 SXM5"].map(
                    (gpu) => (
                      <MenuItem
                        key={gpu}
                        onClick={() => setSelectedGpu(gpu)}
                        bg="gray.800"
                        _hover={{ bg: "gray.700" }}
                        color="whiteAlpha.900"
                      >
                        {gpu}
                      </MenuItem>
                    )
                  )}
                </MenuList>
              </Menu>

              <Text fontSize="sm" color="whiteAlpha.600">
                Language
              </Text>
              <Menu>
                <MenuButton
                  as={Button}
                  rightIcon={<FaChevronDown />}
                  size="sm"
                  variant="outline"
                  borderColor="whiteAlpha.300"
                  color="whiteAlpha.900"
                  _hover={{ bg: "whiteAlpha.100" }}
                >
                  {selectedLanguage === "python" ? "Python" : "C++/CUDA"}
                </MenuButton>
                <MenuList bg="gray.800" borderColor="gray.700">
                  <MenuItem
                    onClick={() => setSelectedLanguage("python")}
                    bg="gray.800"
                    _hover={{ bg: "gray.700" }}
                    color="whiteAlpha.900"
                  >
                    Python
                  </MenuItem>
                  <MenuItem
                    onClick={() => setSelectedLanguage("cpp")}
                    bg="gray.800"
                    _hover={{ bg: "gray.700" }}
                    color="whiteAlpha.900"
                  >
                    C++/CUDA
                  </MenuItem>
                </MenuList>
              </Menu>

              <Tooltip label="Save file">
                <IconButton
                  aria-label="Save"
                  icon={<FaSave />}
                  size="sm"
                  variant="ghost"
                  color="whiteAlpha.700"
                  _hover={{ bg: "whiteAlpha.100", color: "whiteAlpha.900" }}
                />
              </Tooltip>

              <Tooltip label="Share">
                <IconButton
                  aria-label="Share"
                  icon={<FaShare />}
                  size="sm"
                  variant="ghost"
                  color="whiteAlpha.700"
                  _hover={{ bg: "whiteAlpha.100", color: "whiteAlpha.900" }}
                />
              </Tooltip>

              <Button
                leftIcon={<FaPlay />}
                size="sm"
                colorScheme="green"
                variant="solid"
                _hover={{ transform: "scale(1.05)" }}
                transition="all 0.2s"
                isLoading={isRunning}
              >
                Run
              </Button>
            </HStack>
          </HStack>

          {/* Code Editor */}
          <Box w="100%" h="100%" minH="400px">
            <CodeEditor
              code={code}
              setCode={setCode}
              selectedLanguage={selectedLanguage as ProgrammingLanguage}
            />
          </Box>
        </VStack>
      </Box>
    </Layout>
  );
}
