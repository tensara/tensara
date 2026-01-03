/**
 * AmdProvisioningCard.tsx
 *
 * Animated card component shown during AMD VM provisioning.
 * Features:
 * - AMD logo with pulse glow animation
 * - Time-based progress bar (capped at 95%)
 * - Elapsed time display
 * - Rotating "Did You Know?" tips about GPU programming
 * - Connection status indicator (heartbeat-based)
 * - Cancel button with confirmation dialog
 */

import { useState, useEffect, useRef } from "react";
import {
  Box,
  VStack,
  HStack,
  Text,
  Progress,
  Icon,
  Button,
  AlertDialog,
  AlertDialogBody,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogContent,
  AlertDialogOverlay,
  useDisclosure,
} from "@chakra-ui/react";
import { keyframes } from "@emotion/react";
import { SiAmd } from "react-icons/si";
import { FaCircle, FaTimes } from "react-icons/fa";
import { getEstimatedWaitTime, isVmLikelyWarm } from "~/utils/amdVmStatus";

// GPU programming tips - rotates every 10-15 seconds
const AMD_GPU_TIPS = [
  // AMD MI300X Hardware Facts
  "AMD MI300X has 192GB HBM3 memory - 2.4x more than H100's 80GB!",
  "MI300X delivers up to 1.3 PFLOPS of FP16 performance.",
  "MI300X uses a chiplet design with 8 XCDs (Accelerator Complex Dies).",
  "Each MI300X has 304 compute units and 19,456 stream processors.",
  "MI300X memory bandwidth is 5.3 TB/s - great for memory-bound kernels.",

  // HIP/ROCm Programming
  "HIP is highly compatible with CUDA - most kernels port with minimal changes.",
  "Use `hipify-perl` or `hipify-clang` to auto-convert CUDA code to HIP.",
  "HIP uses `hipMalloc`, `hipMemcpy`, `hipFree` - same patterns as CUDA.",
  "ROCm's `rocprof` is the AMD equivalent of NVIDIA's `nsight` profiler.",
  "HIP supports both AMD and NVIDIA GPUs - write once, run on both!",

  // GPU Programming Best Practices
  "Coalesced memory access can improve bandwidth utilization by 10x or more.",
  "Shared memory (LDS on AMD) is ~100x faster than global memory.",
  "Occupancy isn't everything - sometimes fewer threads with more registers wins.",
  "Use `__restrict__` pointers to help the compiler optimize memory access.",
  "Warp/wavefront divergence kills performance - keep branches uniform.",

  // AMD-Specific Optimization
  "AMD wavefronts are 64 threads (vs NVIDIA's 32-thread warps).",
  "MI300X has 256KB of LDS (shared memory) per compute unit.",
  "Use `__builtin_amdgcn_readfirstlane` for efficient warp-level reductions.",
  "ROCm's `rocBLAS` and `hipBLAS` are drop-in replacements for cuBLAS.",
  "The `--offload-arch=gfx942` flag targets MI300X specifically.",

  // Performance & Debugging
  "Your next submission will start much faster - VMs stay warm for 10 minutes!",
  "Profile before optimizing - measure, don't guess.",
  "Register pressure often limits occupancy more than shared memory.",
  "Use `rocm-smi` to monitor GPU utilization, temperature, and memory.",
  "Async memory transfers (`hipMemcpyAsync`) can overlap compute and data movement.",
];

// Pulse glow animation for AMD logo
const pulseGlow = keyframes`
  0%, 100% {
    filter: drop-shadow(0 0 8px rgba(237, 28, 36, 0.4));
    transform: scale(1);
  }
  50% {
    filter: drop-shadow(0 0 20px rgba(237, 28, 36, 0.8));
    transform: scale(1.05);
  }
`;

// Subtle pulse for connection indicator
const connectionPulse = keyframes`
  0%, 100% {
    opacity: 1;
  }
  50% {
    opacity: 0.5;
  }
`;

interface AmdProvisioningCardProps {
  elapsedSeconds: number;
  lastHeartbeat?: number | null;
  onCancel?: () => void;
}

const AmdProvisioningCard = ({
  elapsedSeconds,
  lastHeartbeat,
  onCancel,
}: AmdProvisioningCardProps) => {
  const [currentTipIndex, setCurrentTipIndex] = useState(0);
  const [isWarm] = useState(() => isVmLikelyWarm());
  const estimatedTotal = getEstimatedWaitTime();

  // Cancel confirmation dialog
  const { isOpen, onOpen, onClose } = useDisclosure();
  const cancelRef = useRef<HTMLButtonElement>(null);

  const handleConfirmCancel = () => {
    onClose();
    onCancel?.();
  };

  // Rotate tips every 25 seconds
  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentTipIndex((prev) => (prev + 1) % AMD_GPU_TIPS.length);
    }, 25000);
    return () => clearInterval(interval);
  }, []);

  // Calculate progress (capped at 95%)
  const progress = Math.min((elapsedSeconds / estimatedTotal) * 100, 95);

  // Check connection status (heartbeat within last 30 seconds)
  const isConnected =
    lastHeartbeat !== null &&
    lastHeartbeat !== undefined &&
    Date.now() - lastHeartbeat < 30000;

  // Format elapsed time as mm:ss
  const formatTime = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, "0")}`;
  };

  return (
    <Box
      bg="whiteAlpha.50"
      borderRadius="xl"
      p={6}
      borderWidth={1}
      borderColor="whiteAlpha.100"
    >
      <VStack spacing={5} align="stretch">
        {/* Header with AMD logo */}
        <HStack spacing={4} justify="center">
          <Box animation={`${pulseGlow} 2s ease-in-out infinite`}>
            <Icon as={SiAmd} boxSize={10} color="#ED1C24" />
          </Box>
          <VStack align="start" spacing={0}>
            <Text fontSize="lg" fontWeight="semibold" color="white">
              Provisioning AMD MI300X
            </Text>
            <Text fontSize="sm" color="whiteAlpha.700">
              {isWarm ? "Warm VM - faster startup" : "Cold start - please wait"}
            </Text>
          </VStack>
        </HStack>

        {/* Progress section */}
        <VStack spacing={2} align="stretch">
          <HStack justify="space-between">
            <Text fontSize="sm" color="whiteAlpha.700">
              {isWarm ? "Estimated: ~1 min" : "Estimated: ~7 min"}
            </Text>
            <Text fontSize="sm" color="white" fontWeight="medium">
              {formatTime(elapsedSeconds)}
            </Text>
          </HStack>
          <Progress
            value={progress}
            size="sm"
            colorScheme="red"
            bg="whiteAlpha.200"
            borderRadius="full"
            sx={{
              "& > div": {
                background: "linear-gradient(90deg, #ED1C24 0%, #FF6B6B 100%)",
                transition: "width 0.5s ease-out",
              },
            }}
          />
        </VStack>

        {/* Did You Know tip */}
        <Box
          bg="whiteAlpha.50"
          borderRadius="lg"
          p={4}
          borderLeftWidth={3}
          borderLeftColor="#ED1C24"
        >
          <Text fontSize="xs" color="#ED1C24" fontWeight="semibold" mb={1}>
            DID YOU KNOW?
          </Text>
          <Text
            fontSize="sm"
            color="whiteAlpha.900"
            key={currentTipIndex}
            sx={{
              animation: "fadeIn 0.5s ease-in-out",
              "@keyframes fadeIn": {
                "0%": { opacity: 0, transform: "translateY(5px)" },
                "100%": { opacity: 1, transform: "translateY(0)" },
              },
            }}
          >
            {AMD_GPU_TIPS[currentTipIndex]}
          </Text>
        </Box>

        {/* Connection status */}
        <HStack justify="center" spacing={2}>
          <Icon
            as={FaCircle}
            boxSize={2}
            color={isConnected ? "green.400" : "yellow.400"}
            animation={
              isConnected
                ? `${connectionPulse} 2s ease-in-out infinite`
                : "none"
            }
          />
          <Text fontSize="xs" color="whiteAlpha.600">
            {isConnected ? "Connected to runner" : "Waiting for connection..."}
          </Text>
        </HStack>

        {/* Cancel button */}
        {onCancel && (
          <Button
            size="sm"
            variant="ghost"
            color="whiteAlpha.500"
            _hover={{ color: "whiteAlpha.800", bg: "whiteAlpha.100" }}
            leftIcon={<Icon as={FaTimes} boxSize={3} />}
            onClick={onOpen}
            alignSelf="center"
          >
            Cancel submission
          </Button>
        )}
      </VStack>

      {/* Cancel confirmation dialog */}
      <AlertDialog
        isOpen={isOpen}
        leastDestructiveRef={cancelRef}
        onClose={onClose}
        isCentered
      >
        <AlertDialogOverlay bg="blackAlpha.700" backdropFilter="blur(4px)">
          <AlertDialogContent
            bg="brand.secondary"
            borderColor="whiteAlpha.200"
            borderWidth={1}
          >
            <AlertDialogHeader fontSize="lg" fontWeight="bold" color="white">
              Cancel Submission?
            </AlertDialogHeader>

            <AlertDialogBody color="whiteAlpha.800">
              This will stop the VM provisioning and cancel your submission. You
              can submit again at any time.
            </AlertDialogBody>

            <AlertDialogFooter gap={3}>
              <Button
                ref={cancelRef}
                onClick={onClose}
                variant="ghost"
                color="whiteAlpha.700"
              >
                Keep waiting
              </Button>
              <Button colorScheme="red" onClick={handleConfirmCancel}>
                Cancel submission
              </Button>
            </AlertDialogFooter>
          </AlertDialogContent>
        </AlertDialogOverlay>
      </AlertDialog>
    </Box>
  );
};

export default AmdProvisioningCard;
