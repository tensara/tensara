import {
  Popover,
  PopoverTrigger,
  PopoverContent,
  PopoverBody,
  IconButton,
  Text,
} from "@chakra-ui/react";
import { FaInfoCircle } from "react-icons/fa";

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
      "GPU core temperature during kernel execution. Higher temperatures indicate heavy load. Sustained high temperatures (>80°C) may trigger thermal throttling, reducing performance.",
  },
  smClock: {
    title: "SM Clock Speed",
    description:
      "Streaming Multiprocessor clock speed in MHz. Higher values indicate the GPU is running at higher performance levels. The clock may dynamically adjust based on power and thermal conditions.",
  },
  pState: {
    title: "Performance State",
    description:
      "GPU performance state where P0 = maximum performance. Lower numbers indicate higher performance modes. Higher P-states (P1, P2, etc.) indicate the GPU is in a power-saving mode.",
  },
};

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
        bg="gray.800"
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
