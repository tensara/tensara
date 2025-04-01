import { useRef } from "react";
import {
  HStack,
  Box,
  Text,
  Select,
  Button,
  IconButton,
  Flex,
} from "@chakra-ui/react";

import { type DataType, type ProgrammingLanguage } from "~/types/misc";

import { GpuInfoModal } from "~/components/misc/GpuInfoModal";
import { LanguageInfoModal } from "~/components/misc/LanguageInfoModal";

import { InfoIcon, RepeatIcon } from "@chakra-ui/icons";

import { GPU_DISPLAY_NAMES } from "~/constants/gpu";

interface SubmissionFormProps {
  selectedGpuType: string;
  setSelectedGpuType: (gpuType: string) => void;
  selectedLanguage: ProgrammingLanguage;
  setSelectedLanguage: (language: ProgrammingLanguage) => void;
  selectedDataType: DataType;
  setSelectedDataType: (dataType: DataType) => void;
  isCodeDirty: boolean;
  onResetClick: () => void;
  onSubmit: () => void;
  isSubmitting: boolean;
  isGpuSelectionDisabled: boolean;
  isLanguageSelectionDisabled: boolean;
  isDataTypeSelectionDisabled: boolean;
}

const SubmissionForm = ({
  selectedGpuType,
  setSelectedGpuType,
  selectedLanguage,
  setSelectedLanguage,
  selectedDataType,
  setSelectedDataType,
  isCodeDirty,
  onResetClick,
  onSubmit,
  isSubmitting,
  isGpuSelectionDisabled,
  isLanguageSelectionDisabled,
  isDataTypeSelectionDisabled,
}: SubmissionFormProps) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const buttonContainerRef = useRef<HTMLDivElement>(null);

  return (
    <Flex
      ref={containerRef}
      id="form-container"
      w="100%"
      justifyContent="space-between"
      alignItems={{ base: "flex-start", md: "center" }}
      flexDirection={{ base: "column", md: "row" }}
      gap={3}
    >
      <HStack spacing={2} flexWrap="wrap" gap={2}>
        <Box>
          <Text fontSize="sm" color="whiteAlpha.700">
            GPU Type
            <GpuInfoModal />
          </Text>
          <Select
            size="sm"
            bg="whiteAlpha.50"
            borderColor="whiteAlpha.200"
            _hover={{ borderColor: "whiteAlpha.300" }}
            w={{ base: "110px", md: "140px" }}
            value={selectedGpuType}
            onChange={(e) => setSelectedGpuType(e.target.value)}
            borderRadius="full"
            sx={{
              "& > option": {
                bg: "gray.800",
              },
            }}
            isDisabled={isGpuSelectionDisabled}
          >
            {Object.entries(GPU_DISPLAY_NAMES)
              .filter(([key]) => key !== "all")
              .map(([key, value]) => (
                <option key={key} value={key}>
                  {value}
                </option>
              ))}
          </Select>
        </Box>
        <Box>
          <Text fontSize="sm" color="whiteAlpha.700">
            Language
            <LanguageInfoModal />
          </Text>
          <Select
            size="sm"
            bg="whiteAlpha.50"
            borderColor="whiteAlpha.200"
            _hover={{ borderColor: "whiteAlpha.300" }}
            onChange={(e) =>
              setSelectedLanguage(e.target.value as ProgrammingLanguage)
            }
            value={selectedLanguage}
            w={{ base: "110px", md: "140px" }}
            defaultValue="cuda"
            borderRadius="full"
            sx={{
              "& > option": {
                bg: "gray.800",
              },
            }}
            isDisabled={isLanguageSelectionDisabled}
          >
            <option value="cuda">CUDA C++</option>
            <option value="python">Python (Triton)</option>
          </Select>
        </Box>
        <Box>
          <Text fontSize="sm" color="whiteAlpha.700">
            Data Type
            {/* dummy button to align -- terrible hack */}
            <IconButton
              aria-label="Data Type Information"
              icon={<InfoIcon />}
              size="sm"
              variant="ghost"
              color="transparent"
              visibility="hidden"
              bg="transparent"
            />
          </Text>
          <Select
            size="sm"
            bg="whiteAlpha.50"
            borderColor="whiteAlpha.200"
            _hover={{ borderColor: "whiteAlpha.300" }}
            w={{ base: "110px", md: "140px" }}
            value={selectedDataType}
            onChange={(e) => setSelectedDataType(e.target.value as DataType)}
            borderRadius="full"
            sx={{
              "& > option": {
                bg: "gray.800",
              },
            }}
            isDisabled={isDataTypeSelectionDisabled}
          >
            <option value="float32">float32</option>
            <option value="float16" disabled>
              float16
            </option>
            <option value="int32" disabled>
              int32
            </option>
            <option value="int16" disabled>
              int16
            </option>
          </Select>
        </Box>
      </HStack>

      <HStack
        ref={buttonContainerRef}
        spacing={2}
        mt={{ base: 2, md: 0 }}
        minW="90px"
        flexShrink={0}
      >
        {isCodeDirty && (
          <>
            <Button
              size="md"
              variant="ghost"
              onClick={onResetClick}
              borderRadius="full"
              height="40px"
              fontSize="sm"
              fontWeight="semibold"
              color="gray.300"
              leftIcon={<RepeatIcon />}
              iconSpacing={2}
              px={4}
              _hover={{
                bg: "whiteAlpha.50",
                color: "white",
              }}
            >
              Reset Code
            </Button>
          </>
        )}
        <Button
          bg="rgba(34, 197, 94, 0.1)"
          color="rgb(34, 197, 94)"
          size="md"
          onClick={onSubmit}
          isLoading={isSubmitting}
          loadingText="Submit"
          spinner={<></>}
          disabled={isSubmitting}
          borderRadius="full"
          height="40px"
          fontSize="sm"
          fontWeight="semibold"
          px={{ base: 4, md: 6 }}
          minW="80px"
          _hover={{
            bg: "rgba(34, 197, 94, 0.2)",
            transform: "translateY(-1px)",
          }}
          _active={{
            bg: "rgba(34, 197, 94, 0.25)",
          }}
          transition="all 0.2s"
        >
          Submit
        </Button>
      </HStack>
    </Flex>
  );
};

export default SubmissionForm;
