import React, { useState, useEffect, useMemo } from "react";
import { Box, Flex, Text, Container } from "@chakra-ui/react";
import { motion } from "framer-motion";

// Define props interface
interface AnimatedCudaEditorProps {
  onTypingComplete?: () => void; // Optional callback when typing finishes
  isFadingOut?: boolean; // Controls the fade-out animation
}

const AnimatedCudaEditor: React.FC<AnimatedCudaEditorProps> = ({
  onTypingComplete,
  isFadingOut = false, // Default to not fading out
}) => {
  const initialCode = useMemo(
    () => [
      "#include <cuda_runtime.h>",
      "",
      "",
      "__global__ void vectorAdd(const float* A, const float* B, float* C, int N) {",
      "}",
    ],
    []
  );

  const finalCode = useMemo(
    () => [
      "#include <cuda_runtime.h>",
      "",
      "",
      "__global__ void vectorAdd(const float* A, const float* B, float* C, int N) {",
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
    if (typingComplete) {
      // Call the callback only once when typing completes
      if (onTypingComplete) {
        onTypingComplete();
      }
      return; // Stop the effect if typing is complete
    }

    const typingInterval = setInterval(() => {
      if (currentLine >= finalCode.length) {
        clearInterval(typingInterval);
        setTypingComplete(true); // Set complete flag, callback handled above
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
  }, [currentLine, currentChar, typingComplete, finalCode, onTypingComplete]); // Add onTypingComplete to dependency array

  const editorVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: { opacity: 1, y: 0, transition: { duration: 0.5 } },
    fadingOut: { opacity: 0, y: -20, transition: { duration: 0.5 } },
  };

  return (
    <Container minW="lg">
      <motion.div
        initial="hidden"
        animate={isFadingOut ? "fadingOut" : isVisible ? "visible" : "hidden"}
        variants={editorVariants}
      >
        <Flex
          justifyContent="center"
          alignItems="center"
          flexDirection="column"
          // Opacity is now handled by framer-motion
        >
          {/* Terminal */}
          <Box
            bg="#111111"
            borderRadius="lg"
            overflow="hidden"
            borderWidth="1px"
            borderColor="whiteAlpha.100"
            boxShadow="0 10px 40px rgba(14, 129, 68, 0.2)" // Enhanced shadow
            position="relative"
            w="full"
            maxW={{ base: "full", lg: "90%" }}
            h="380px" // Increased height to accommodate terminal header
            // Removed _after pseudo-element for glow - will be handled by parent
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

            {/* Removed old simple glow effect, replaced by _after pseudo-element */}
          </Box>
        </Flex>
      </motion.div>
    </Container>
  );
};

export default AnimatedCudaEditor;
