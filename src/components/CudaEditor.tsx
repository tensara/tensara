import React, { useState, useEffect, useMemo } from "react";
import { Box, Flex, Text, Container } from "@chakra-ui/react";

const AnimatedCudaEditor = () => {
  const initialCode = useMemo(
    () => [
      "#include <cuda_runtime.h>",
      "",
      "// Note: input vectors A, B and output C are all device pointers",
      'extern "C" void vectorAdd(const float* A, const float* B, float* C, int N) {',
      "}",
    ],
    []
  );

  const finalCode = useMemo(
    () => [
      "#include <cuda_runtime.h>",
      "",
      "// Note: input vectors A, B and output C are all device pointers",
      'extern "C" void vectorAdd(const float* A, const float* B, float* C, int N) {',
      "  // Get thread index",
      "  const int i = blockIdx.x * blockDim.x + threadIdx.x;",
      "",
      "  if (i < N) {",
      "    // Perform vector addition",
      "    C[i] = A[i] + B[i];",
      "  }",
      "}",
    ],
    []
  );

  // Calculate indentation for each line
  const lineIndents = useMemo(() => {
    return finalCode.map((line) => {
      const match = /^(\s*)/.exec(line);
      return match ? match[0].length : 0;
    });
  }, [finalCode]);

  // Prepare the final structure with empty lines to maintain fixed height
  const emptyLines = Array.from(
    { length: finalCode.length - initialCode.length },
    () => ""
  );
  const [displayText, setDisplayText] = useState([
    ...initialCode,
    ...emptyLines,
  ]);
  const [currentLine, setCurrentLine] = useState(4);
  const [currentChar, setCurrentChar] = useState(0);
  const [typingComplete, setTypingComplete] = useState(false);
  const [isVisible, setIsVisible] = useState(false);

  // Add fade-in effect when component mounts
  useEffect(() => {
    setTimeout(() => {
      setIsVisible(true);
    }, 300);
  }, []);

  useEffect(() => {
    if (typingComplete) return;

    const typingInterval = setInterval(() => {
      if (currentLine >= finalCode.length) {
        clearInterval(typingInterval);
        setTypingComplete(true);
        return;
      }

      if (currentLine < 4) {
        // Skip initial lines that are already present
        setCurrentLine(4);
        return;
      }

      if (currentLine === 4 && currentChar === 0) {
        // Start by replacing the closing bracket with a newline
        setDisplayText((prev) => {
          const newText = [...prev];
          newText[4] = "";
          return newText;
        });
        setCurrentChar(1);
        return;
      }

      const lineToType = finalCode[currentLine] ?? "";

      if (currentChar === 0) {
        // Start a new line but maintain fixed structure
        setDisplayText((prev) => {
          const newText = [...prev];
          newText[currentLine] = "";
          return newText;
        });
      }

      if (currentChar < lineToType.length) {
        // Add next character to current line
        setDisplayText((prev) => {
          const newText = [...prev];
          newText[currentLine] = lineToType.substring(0, currentChar + 1);
          return newText;
        });
        setCurrentChar(currentChar + 1);
      } else {
        // Move to next line
        setCurrentLine(currentLine + 1);
        setCurrentChar(0);
      }
    }, 50); // Adjust typing speed here

    return () => clearInterval(typingInterval);
  }, [currentLine, currentChar, typingComplete, finalCode]);

  return (
    <Container minW="lg">
      <Flex
        justifyContent="center"
        alignItems="center"
        flexDirection="column"
        opacity={isVisible ? 1 : 0}
      >
        <Flex
          mb={4}
          w="full"
          justifyContent="space-evenly"
          maxW={{ base: "full", lg: "90%" }}
        >
          <Box>
            <Text fontWeight="medium" color="whiteAlpha.800">
              GPU Type
            </Text>
            <Box
              bg="brand.secondary"
              px={4}
              py={2}
              borderRadius="md"
              mt={1}
              w="150px"
              color="whiteAlpha.800"
            >
              NVIDIA H100
            </Box>
          </Box>

          <Box>
            <Text fontWeight="medium" color="whiteAlpha.800">
              Language
            </Text>
            <Box
              bg="brand.secondary"
              px={4}
              py={2}
              borderRadius="md"
              mt={1}
              w="150px"
              color="whiteAlpha.800"
            >
              CUDA C++
            </Box>
          </Box>

          <Box>
            <Text fontWeight="medium" color="whiteAlpha.800">
              Data Type
            </Text>
            <Box
              bg="brand.secondary"
              px={4}
              py={2}
              borderRadius="md"
              mt={1}
              w="150px"
              color="whiteAlpha.800"
            >
              float32
            </Box>
          </Box>
        </Flex>

        {/* Terminal */}
        <Box
          bg="#111111"
          borderRadius="lg"
          overflow="hidden"
          borderWidth="1px"
          borderColor="whiteAlpha.100"
          boxShadow="0 8px 30px rgba(0, 0, 0, 0.25)"
          position="relative"
          w="full"
          maxW={{ base: "full", lg: "90%" }}
          h="380px" // Increased height to accommodate terminal header
        >
          {/* Terminal header */}
          <Flex
            bg="rgba(22, 27, 34, 1)"
            p={2}
            borderBottomWidth="1px"
            borderColor="whiteAlpha.200"
            alignItems="center"
          >
            <Text
              color="whiteAlpha.900"
              fontSize="xs"
              fontFamily="mono"
              flex="1"
              textAlign="center"
              fontWeight="medium"
            >
              vector-add.cu
            </Text>
          </Flex>

          {/* Code content */}
          <Box
            p={4}
            fontFamily="mono"
            fontSize="sm"
            position="relative"
            h="calc(100% - 40px)" // Adjust for terminal header
          >
            {displayText.map((line, lineIndex) => (
              <Flex key={lineIndex} mb={1}>
                <Text
                  pl={
                    lineIndents[lineIndex]
                      ? `${lineIndents[lineIndex] * 0.5}rem`
                      : 0
                  }
                  color={
                    line.startsWith("//")
                      ? "#9CA3AF"
                      : line.includes("extern") || line.includes("#include")
                        ? "#63B3ED"
                        : "whiteAlpha.800"
                  }
                >
                  {line
                    .trim()
                    .split(" ")
                    .map((word, wordIndex, array) => {
                      // Skip empty lines or process words
                      if (!line) return null;

                      const isKeyword = [
                        "int",
                        "float",
                        "void",
                        "if",
                        "for",
                        "float*",
                      ].includes(word);
                      const isFunction = word === "vectorAdd";
                      const isInclude = word === "#include";
                      const isDirective = word.startsWith("//");

                      return (
                        <React.Fragment key={wordIndex}>
                          <Text
                            as="span"
                            color={
                              isKeyword
                                ? "#63ed78"
                                : isFunction
                                  ? "#6fb5e1"
                                  : isInclude
                                    ? "#63B3ED"
                                    : isDirective
                                      ? "#9CA3AF"
                                      : undefined
                            }
                          >
                            {word}
                          </Text>
                          {wordIndex < array.length - 1 && " "}
                        </React.Fragment>
                      );
                    })}
                  {lineIndex === currentLine && !typingComplete && (
                    <Box
                      as="span"
                      display="inline-block"
                      bg="#2ecc71"
                      w="2px"
                      h="14px"
                      ml={1}
                      position="relative"
                      top="2px"
                      sx={{
                        animation: "blink 1s infinite",
                        "@keyframes blink": {
                          "0%": { opacity: 1 },
                          "50%": { opacity: 0 },
                          "100%": { opacity: 1 },
                        },
                      }}
                    />
                  )}
                </Text>
              </Flex>
            ))}
          </Box>

          {/* Terminal bottom glow effect */}
          <Box
            position="absolute"
            bottom="0"
            left="0"
            right="0"
            height="40px"
            background="linear-gradient(to top, rgba(15, 23, 42, 0.5), transparent)"
            pointerEvents="none"
          />
        </Box>
      </Flex>
    </Container>
  );
};

export default AnimatedCudaEditor;
