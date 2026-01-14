import {
  Popover,
  PopoverTrigger,
  PopoverContent,
  PopoverHeader,
  PopoverBody,
  PopoverArrow,
  VStack,
  Text,
  Icon,
  Box,
  Divider,
  IconButton,
} from "@chakra-ui/react";
import { FiInfo } from "react-icons/fi";
import { FaInfoCircle } from "react-icons/fa";
import { type ReactNode } from "react";

// ============================================================================
// NEW: Per-metric info popover (used in SubmissionResults.tsx)
// ============================================================================

export type GPUMetricType = "temperature" | "smClock" | "pState";

interface GPUMetricInfoPopoverProps {
  metric: GPUMetricType;
}

const METRIC_INFO: Record<
  GPUMetricType,
  { title: string; description: string }
> = {
  temperature: {
    title: "GPU Temperature",
    description:
      "GPU core temperature during kernel execution. Sustained high temperatures (>80°C) may trigger thermal throttling, reducing performance. Hover over each test case to see the temperature range.",
  },
  smClock: {
    title: "SM Clock Speed",
    description:
      "Streaming Multiprocessor clock speed in MHz. The clock may dynamically adjust based on power and thermal conditions. Hover over each test case to see the clock speed range.",
  },
  pState: {
    title: "Performance State",
    description:
      "GPU performance state where P0 = maximum performance. Lower numbers indicate higher performance modes. Higher P-states (P1, P2, etc.) indicate the GPU is in a power-saving mode.",
  },
};

/**
 * Per-metric info popover with individual info icon.
 * Used in SubmissionResults.tsx for Temperature, SM Clock, etc.
 */
export const GPUMetricInfoPopover = ({ metric }: GPUMetricInfoPopoverProps) => {
  const info = METRIC_INFO[metric];

  return (
    <Popover placement="top" trigger="click">
      <PopoverTrigger>
        <IconButton
          aria-label={`${info.title} Information`}
          icon={<FaInfoCircle />}
          size="xs"
          variant="ghost"
          color="gray.500"
          _hover={{ color: "white", bg: "transparent" }}
          bg="transparent"
          minW="auto"
          h="auto"
          p={0}
          ml={1}
        />
      </PopoverTrigger>
      <PopoverContent
        bg="brand.secondary"
        borderColor="whiteAlpha.200"
        w="280px"
        _focus={{ boxShadow: "none" }}
      >
        <PopoverBody p={4}>
          <Text fontWeight="semibold" color="white" mb={2}>
            {info.title}
          </Text>
          <Text fontSize="sm" color="whiteAlpha.700" lineHeight="tall">
            {info.description}
          </Text>
        </PopoverBody>
      </PopoverContent>
    </Popover>
  );
};

// ============================================================================
// LEGACY: Full GPU metrics popover (used in [id].tsx submission detail page)
// ============================================================================

interface GPUMetricInfoPopoverLegacyProps {
  children: ReactNode;
}

/**
 * Legacy popover component that wraps children and shows all GPU metrics info.
 * Used in submission detail page ([id].tsx).
 */
export const GPUMetricInfoPopoverLegacy = ({
  children,
}: GPUMetricInfoPopoverLegacyProps) => {
  return (
    <Popover trigger="hover" placement="top">
      <PopoverTrigger>{children}</PopoverTrigger>
      <PopoverContent
        bg="brand.secondary"
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
 * Icon button that triggers the legacy GPU metric info popover.
 * Used in submission detail page ([id].tsx).
 */
export const GPUMetricInfoIcon = () => {
  return (
    <GPUMetricInfoPopoverLegacy>
      <Box as="span" cursor="pointer" display="inline-flex" alignItems="center">
        <Icon
          as={FiInfo}
          color="whiteAlpha.600"
          _hover={{ color: "whiteAlpha.800" }}
          boxSize={4}
        />
      </Box>
    </GPUMetricInfoPopoverLegacy>
  );
};

// ============================================================================
// Helper functions for GPU metrics display
// ============================================================================

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
