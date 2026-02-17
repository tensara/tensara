import {
  HStack,
  Button,
  Flex,
  Menu,
  MenuButton,
  MenuList,
  MenuItem,
  Text,
  IconButton,
  Tooltip,
} from "@chakra-ui/react";
import { FaChevronDown } from "react-icons/fa";
import { IoRepeat } from "react-icons/io5";
import { GPU_DISPLAY_NAMES } from "~/constants/gpu";
import { LANGUAGE_DISPLAY_NAMES } from "~/constants/language";
import type { ProgrammingLanguage } from "~/types/misc";

interface PuzzleToolbarProps {
  selectedLanguage: ProgrammingLanguage;
  setSelectedLanguage: (lang: ProgrammingLanguage) => void;
  selectedGpuType: string;
  setSelectedGpuType: (gpu: string) => void;
  isCodeDirty: boolean;
  onResetClick: () => void;
  onRun: () => void;
  isRunning: boolean;
  onSubmit: () => void;
  isSubmitting: boolean;
}

const languageOptions = Object.entries(LANGUAGE_DISPLAY_NAMES).filter(
  ([k]) => k !== "all"
);
const gpuOptions = Object.entries(GPU_DISPLAY_NAMES).filter(
  ([k]) => k !== "all"
);

export default function PuzzleToolbar({
  selectedLanguage,
  setSelectedLanguage,
  selectedGpuType,
  setSelectedGpuType,
  isCodeDirty,
  onResetClick,
  onRun,
  isRunning,
  onSubmit,
  isSubmitting,
}: PuzzleToolbarProps) {
  return (
    <Flex
      w="100%"
      justify="space-between"
      align="center"
      px={3}
      py={2}
      bg="whiteAlpha.50"
      borderRadius="lg"
      flexShrink={0}
    >
      <HStack spacing={2}>
        <Menu>
          <MenuButton
            as={Button}
            size="xs"
            variant="ghost"
            color="whiteAlpha.800"
            rightIcon={<FaChevronDown />}
          >
            {LANGUAGE_DISPLAY_NAMES[selectedLanguage] ?? selectedLanguage}
          </MenuButton>
          <MenuList bg="gray.800" borderColor="whiteAlpha.200" minW="140px">
            {languageOptions.map(([key, label]) => (
              <MenuItem
                key={key}
                bg="transparent"
                _hover={{ bg: "whiteAlpha.100" }}
                fontSize="sm"
                onClick={() => setSelectedLanguage(key as ProgrammingLanguage)}
              >
                {label}
              </MenuItem>
            ))}
          </MenuList>
        </Menu>

        <Menu>
          <MenuButton
            as={Button}
            size="xs"
            variant="ghost"
            color="whiteAlpha.800"
            rightIcon={<FaChevronDown />}
          >
            {GPU_DISPLAY_NAMES[selectedGpuType] ?? selectedGpuType}
          </MenuButton>
          <MenuList bg="gray.800" borderColor="whiteAlpha.200" minW="140px">
            {gpuOptions.map(([key, label]) => (
              <MenuItem
                key={key}
                bg="transparent"
                _hover={{ bg: "whiteAlpha.100" }}
                fontSize="sm"
                onClick={() => setSelectedGpuType(key)}
              >
                {label}
              </MenuItem>
            ))}
          </MenuList>
        </Menu>

        {isCodeDirty && (
          <Tooltip label="Reset code">
            <IconButton
              aria-label="Reset code"
              icon={<IoRepeat />}
              size="xs"
              variant="ghost"
              color="whiteAlpha.600"
              onClick={onResetClick}
            />
          </Tooltip>
        )}
      </HStack>

      <HStack spacing={2}>
        <Button
          size="xs"
          variant="outline"
          borderColor="whiteAlpha.200"
          color="whiteAlpha.800"
          onClick={onRun}
          isLoading={isRunning}
          loadingText="Running"
          _hover={{ bg: "whiteAlpha.100" }}
        >
          Run
        </Button>
        <Button
          size="xs"
          bg="brand.primary"
          color="white"
          onClick={onSubmit}
          isLoading={isSubmitting}
          loadingText="Submitting"
          _hover={{ opacity: 0.9 }}
        >
          Submit
        </Button>
      </HStack>
    </Flex>
  );
}
