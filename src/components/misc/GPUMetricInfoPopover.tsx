import {
  Popover,
  PopoverTrigger,
  PopoverContent,
  PopoverHeader,
  PopoverBody,
  PopoverArrow,
  VStack,
  HStack,
  Text,
  Icon,
  Box,
  Divider,
} from "@chakra-ui/react";
import { FiInfo } from "react-icons/fi";
import { type ReactNode } from "react";

interface GPUMetricInfoPopoverProps {
  children: ReactNode;
}

/**
 * Popover component explaining GPU metrics collected during benchmarking.
 * Shows information about temperature, SM clock, P-state, and throttle reasons.
 */
export const GPUMetricInfoPopover = ({
  children,
}: GPUMetricInfoPopoverProps) => {
  return (
    <Popover trigger="hover" placement="top">
      <PopoverTrigger>{children}</PopoverTrigger>
      <PopoverContent
        bg="gray.800"
        borderColor="whiteAlpha.200"
        maxW="400px"
        _focus={{ boxShadow: "none" }}
      >
        <PopoverArrow bg="gray.800" />
        <PopoverHeader
          borderBottomColor="whiteAlpha.200"
          fontWeight="semibold"
          color="white"
        >
          GPU Metrics
        </PopoverHeader>
        <PopoverBody>
          <VStack align="stretch" spacing={3} fontSize="sm">
            <Text color="whiteAlpha.800">
              These metrics are sampled every ~5ms during kernel execution to
              help identify performance anomalies.
            </Text>

            <Divider borderColor="whiteAlpha.200" />

            <Box>
              <Text fontWeight="medium" color="blue.300" mb={1}>
                Temperature (°C)
              </Text>
              <Text color="whiteAlpha.700">
                GPU core temperature. High temperatures (&gt;80°C) may cause
                thermal throttling.
              </Text>
            </Box>

            <Box>
              <Text fontWeight="medium" color="green.300" mb={1}>
                SM Clock (MHz)
              </Text>
              <Text color="whiteAlpha.700">
                Streaming multiprocessor clock speed. Consistent clocks indicate
                stable performance.
              </Text>
            </Box>

            <Box>
              <Text fontWeight="medium" color="purple.300" mb={1}>
                P-State
              </Text>
              <Text color="whiteAlpha.700">
                Performance state (0 = max performance). Higher values indicate
                power saving modes.
              </Text>
            </Box>

            <Box>
              <Text fontWeight="medium" color="orange.300" mb={1}>
                Throttle Reasons
              </Text>
              <Text color="whiteAlpha.700">
                Indicates if the GPU is being throttled due to power, thermal,
                or other limits.
              </Text>
            </Box>
          </VStack>
        </PopoverBody>
      </PopoverContent>
    </Popover>
  );
};

/**
 * Icon button that triggers the GPU metric info popover.
 */
export const GPUMetricInfoIcon = () => {
  return (
    <GPUMetricInfoPopover>
      <Box as="span" cursor="pointer" display="inline-flex" alignItems="center">
        <Icon
          as={FiInfo}
          color="whiteAlpha.600"
          _hover={{ color: "whiteAlpha.800" }}
          boxSize={4}
        />
      </Box>
    </GPUMetricInfoPopover>
  );
};

/**
 * Decode throttle reason bitmask to human-readable reasons.
 * Based on NVML throttle reason codes.
 */
export const decodeThrottleReasons = (bitmask: number): string[] => {
  const reasons: string[] = [];

  // NVML throttle reason codes
  if (bitmask & 0x0000000000000001) reasons.push("Idle");
  if (bitmask & 0x0000000000000002) reasons.push("Application clocks");
  if (bitmask & 0x0000000000000004) reasons.push("SW Power Cap");
  if (bitmask & 0x0000000000000008) reasons.push("HW Slowdown");
  if (bitmask & 0x0000000000000010) reasons.push("Sync boost");
  if (bitmask & 0x0000000000000020) reasons.push("SW Thermal slowdown");
  if (bitmask & 0x0000000000000040) reasons.push("HW Thermal slowdown");
  if (bitmask & 0x0000000000000080) reasons.push("HW Power brake");
  if (bitmask & 0x0000000000000100) reasons.push("Display clock setting");

  // If none of the above, check for "None" (no throttling)
  if (reasons.length === 0 && bitmask === 0) {
    return ["None"];
  }

  return reasons.length > 0 ? reasons : ["Unknown"];
};

/**
 * Format throttle reasons for display.
 */
export const formatThrottleReasons = (bitmask: number): string => {
  if (bitmask === 0) return "None";
  const reasons = decodeThrottleReasons(bitmask);
  return reasons.join(", ");
};

/**
 * Get color based on temperature value.
 */
export const getTempColor = (temp: number): string => {
  if (temp < 60) return "green.400";
  if (temp < 75) return "yellow.400";
  if (temp < 85) return "orange.400";
  return "red.400";
};

/**
 * Get color based on P-state value.
 */
export const getPStateColor = (pstate: number): string => {
  if (pstate === 0) return "green.400";
  if (pstate <= 2) return "yellow.400";
  return "orange.400";
};

/**
 * Get color for throttle indicator.
 */
export const getThrottleColor = (bitmask: number): string => {
  if (bitmask === 0) return "green.400";
  // Idle throttle is not concerning
  if (bitmask === 1) return "gray.400";
  return "orange.400";
};

export default GPUMetricInfoPopover;
