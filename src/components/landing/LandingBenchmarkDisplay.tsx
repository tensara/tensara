import React, { useState, useEffect, useRef } from "react";
import {
  Box,
  Heading,
  Text,
  HStack,
  VStack,
  Table,
  Thead,
  Tbody,
  Tr, // Use standard Tr
  Th,
  Td,
  Icon,
  // Collapse, // Still removed
  IconButton,
  Spinner,
  Center,
} from "@chakra-ui/react";
import { motion, AnimatePresence } from "framer-motion"; // Keep imports for MotionBox/footer
import { FaCheck, FaChevronUp, FaChevronDown } from "react-icons/fa";

// Define the structure for a single dummy benchmark result
interface DummyBenchmarkResult {
  id: string;
  name: string;
  runtime_ms: number;
  gflops: number;
}

// Define props for the component
interface LandingBenchmarkDisplayProps {
  isVisible: boolean; // Controls the fade-in animation
  dummyData: DummyBenchmarkResult[]; // Array of dummy results
}

const MotionBox = motion(Box);
const MotionTr = motion(Tr);

const LandingBenchmarkDisplay: React.FC<LandingBenchmarkDisplayProps> = ({
  isVisible,
  dummyData,
}) => {
  const [isTableOpen, setIsTableOpen] = useState(true);
  const [loadedRowCount, setLoadedRowCount] = useState(0);
  const [showFooter, setShowFooter] = useState(false);
  const animationStartedRef = useRef(false);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);
  const footerTimerRef = useRef<NodeJS.Timeout | null>(null);

  // Effect for row-by-row loading animation - Unconditional Start Logic
  useEffect(() => {
    const cleanup = () => {
      console.log("Running effect cleanup");
      if (intervalRef.current) clearInterval(intervalRef.current);
      if (footerTimerRef.current) clearTimeout(footerTimerRef.current);
      intervalRef.current = null;
      footerTimerRef.current = null;
      animationStartedRef.current = false;
    };

    console.log("Effect running");

    if (!animationStartedRef.current) {
      console.log(
        "Starting animation setup (setState enabled, no MotionTr)..."
      );
      cleanup();
      animationStartedRef.current = true;
      setLoadedRowCount(0); // Reset state
      setShowFooter(false);

      intervalRef.current = setInterval(() => {
        console.log("Interval tick");
        setLoadedRowCount((prevCount) => {
          const nextCount = prevCount + 1;
          console.log("Attempting to set loadedRowCount to:", nextCount); // Log attempt
          if (nextCount >= dummyData.length) {
            console.log(
              "Last row reached, clearing interval and scheduling footer"
            );
            if (intervalRef.current) {
              clearInterval(intervalRef.current);
              intervalRef.current = null;
            }
            footerTimerRef.current = setTimeout(() => {
              console.log("Setting showFooter to true");
              setShowFooter(true);
            }, 100);
            return dummyData.length;
          }
          return nextCount;
        });
      }, 300);
    } else {
      console.log("Animation already started flag is true, skipping setup.");
    }

    return cleanup;
  }, [dummyData.length]); // Depend only on data length

  // *** ADDED: Effect to specifically log loadedRowCount changes ***
  useEffect(() => {
    console.log(`State changed: loadedRowCount is now ${loadedRowCount}`);
  }, [loadedRowCount]);

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: { opacity: 1, transition: { duration: 0.3 } },
  };

  const rowVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: {
      opacity: 1,
      y: 0,
      transition: {
        duration: 0.3,
        type: "spring",
        stiffness: 100,
        opacity: { duration: 0.5 },
      },
    },
  };

  const hasAnimationStarted = animationStartedRef.current;
  console.log("Rendering component", {
    loadedRowCount,
    showFooter,
    hasAnimationStarted,
  });

  return (
    <MotionBox
      w="full"
      maxW={{ base: "50%", lg: "70%" }}
      variants={containerVariants}
      initial="hidden"
      animate={isVisible ? "visible" : "hidden"}
      position="relative"
      mx="auto"
    >
      {/* Main content box */}
      <Box bg="whiteAlpha.50" borderRadius="xl" overflow="hidden">
        <VStack
          spacing={0}
          align="stretch"
          divider={<Box borderBottomWidth={1} borderColor="whiteAlpha.100" />}
        >
          {/* Header */}
          <HStack
            justify="space-between"
            px={6}
            py={4}
            cursor={dummyData.length > 0 && showFooter ? "pointer" : "default"}
            onClick={() =>
              dummyData.length > 0 && showFooter && setIsTableOpen(!isTableOpen)
            }
          >
            <HStack spacing={2}>
              <Heading size="sm" fontWeight="semibold" color="white">
                Benchmark Results
              </Heading>
              {hasAnimationStarted && !showFooter && (
                <Spinner size="xs" color="gray.300" ml={2} />
              )}
            </HStack>
          </HStack>

          {/* Benchmark Table - No Collapse */}
          <Box
            display={isTableOpen ? "block" : "none"}
            minH={
              loadedRowCount === 0 && hasAnimationStarted ? "100px" : "auto"
            }
          >
            <Table variant="unstyled" size="sm">
              <Thead bg="whiteAlpha.100">
                <Tr>
                  <Th color="whiteAlpha.700" py={3}>
                    Test Case
                  </Th>
                  <Th color="whiteAlpha.700" py={3} isNumeric>
                    Runtime
                  </Th>
                  <Th color="whiteAlpha.700" py={3} isNumeric>
                    Performance
                  </Th>
                </Tr>
              </Thead>
              <Tbody>
                {/* Using AnimatePresence for row animations */}
                <AnimatePresence>
                  {dummyData.slice(0, loadedRowCount).map((result, index) => (
                    <MotionTr
                      key={result.id}
                      _hover={{ bg: "whiteAlpha.100" }}
                      variants={rowVariants}
                      initial="hidden"
                      animate="visible"
                      custom={index}
                    >
                      <Td py={3}>
                        <HStack spacing={2}>
                          <Icon as={FaCheck} color="green.300" boxSize={4} />
                          <Text color="white">{result.name}</Text>
                        </HStack>
                      </Td>
                      <Td py={3} isNumeric>
                        <Text color="white">
                          {result.runtime_ms.toFixed(2)} ms
                        </Text>
                      </Td>
                      <Td py={3} isNumeric>
                        <Text color="white">
                          {result.gflops.toFixed(2)} GFLOPS
                        </Text>
                      </Td>
                    </MotionTr>
                  ))}
                </AnimatePresence>
              </Tbody>
            </Table>
          </Box>
        </VStack>
      </Box>

      {/* Footer: Average Stats Box - Animates in */}
      <AnimatePresence>
        {showFooter && (
          <MotionBox
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.3 }}
            mt={4}
          >
            <Box bg="whiteAlpha.50" p={4} borderRadius="xl">
              <HStack justify="space-between">
                <Box>
                  <Text color="whiteAlpha.700" mb={1} fontSize="sm">
                    Average Performance
                  </Text>
                  <Heading size="md" color="white">
                    {(
                      dummyData.reduce((sum, r) => sum + r.gflops, 0) /
                      (dummyData.length || 1)
                    ) // Avoid division by zero
                      .toFixed(2)}{" "}
                    GFLOPS
                  </Heading>
                </Box>
                <Box>
                  <Text color="whiteAlpha.700" mb={1} fontSize="sm">
                    Average Runtime
                  </Text>
                  <Heading size="md" color="white">
                    {(
                      dummyData.reduce((sum, r) => sum + r.runtime_ms, 0) /
                      (dummyData.length || 1)
                    ) // Avoid division by zero
                      .toFixed(2)}{" "}
                    ms
                  </Heading>
                </Box>
              </HStack>
            </Box>
          </MotionBox>
        )}
      </AnimatePresence>
    </MotionBox>
  );
};

export default LandingBenchmarkDisplay;
