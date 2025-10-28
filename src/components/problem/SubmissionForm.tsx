import { useRef } from "react";
import {
  HStack,
  Box,
  Text,
  Button,
  IconButton,
  Flex,
  Menu,
  MenuButton,
  MenuList,
  MenuItem,
  Badge,
} from "@chakra-ui/react";
import { type DataType, type ProgrammingLanguage } from "~/types/misc";

import { GpuInfoModal } from "~/components/misc/GpuInfoModal";
import { LanguageInfoModal } from "~/components/misc/LanguageInfoModal";

import { IoRepeat } from "react-icons/io5";
import { FaInfoCircle, FaChevronDown } from "react-icons/fa";

import { GPU_DISPLAY_NAMES } from "~/constants/gpu";
import { LANGUAGE_DISPLAY_NAMES } from "~/constants/language";

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
  onRun?: () => void;
  isRunning?: boolean;
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
  onRun,
  isRunning,
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
          <Menu>
            <MenuButton
              size="sm"
              as={Button}
              rightIcon={<FaChevronDown size={12} color="#a1a1aa" />}
              bg="whiteAlpha.50"
              _hover={{ bg: "whiteAlpha.100", borderColor: "gray.600" }}
              _active={{ bg: "whiteAlpha.150" }}
              _focus={{ borderColor: "blue.500", boxShadow: "none" }}
              color="white"
              w={{ base: "110px", md: "140px" }}
              fontWeight="normal"
              textAlign="left"
              justifyContent="flex-start"
              borderRadius="lg"
            >
              {GPU_DISPLAY_NAMES[selectedGpuType]}
            </MenuButton>
            <MenuList
              bg="brand.secondary"
              borderColor="gray.800"
              p={0}
              borderRadius="lg"
              minW="140px"
            >
              {Object.entries(GPU_DISPLAY_NAMES)
                .filter(([key]) => key !== "all")
                .map(([key, value]) => (
                  <MenuItem
                    key={key}
                    onClick={() => {
                      setSelectedGpuType(key);
                    }}
                    bg="brand.secondary"
                    _hover={{ bg: "gray.700" }}
                    color="white"
                    borderRadius="lg"
                    fontSize="sm"
                  >
                    {value}
                  </MenuItem>
                ))}
            </MenuList>
          </Menu>
        </Box>
        <Box>
          <Text fontSize="sm" color="whiteAlpha.700">
            Language
            <LanguageInfoModal />
          </Text>
          <Menu>
            <MenuButton
              size="sm"
              as={Button}
              rightIcon={<FaChevronDown size={12} color="#a1a1aa" />}
              bg="whiteAlpha.50"
              _hover={{ bg: "whiteAlpha.100", borderColor: "gray.600" }}
              _active={{ bg: "whiteAlpha.150" }}
              _focus={{ borderColor: "blue.500", boxShadow: "none" }}
              color="white"
              w={{ base: "110px", md: "140px" }}
              fontWeight="normal"
              textAlign="left"
              justifyContent="flex-start"
              borderRadius="lg"
            >
              {LANGUAGE_DISPLAY_NAMES[selectedLanguage]}
            </MenuButton>
            <MenuList
              bg="brand.secondary"
              borderColor="gray.800"
              p={0}
              borderRadius="lg"
              minW="140px"
            >
              <MenuItem
                key="cuda"
                onClick={() => setSelectedLanguage("cuda")}
                bg="brand.secondary"
                _hover={{ bg: "gray.700" }}
                color="white"
                borderRadius="lg"
                fontSize="sm"
              >
                CUDA C++
              </MenuItem>
              <MenuItem
                key="python"
                onClick={() => setSelectedLanguage("python")}
                bg="brand.secondary"
                _hover={{ bg: "gray.700" }}
                color="white"
                borderRadius="lg"
                fontSize="sm"
              >
                Triton
              </MenuItem>
              <MenuItem
                key="mojo"
                onClick={() => setSelectedLanguage("mojo")}
                bg="brand.secondary"
                _hover={{ bg: "gray.700" }}
                color="white"
                borderRadius="lg"
                fontSize="sm"
              >
                Mojo
              </MenuItem>
              <MenuItem
                key="cute"
                onClick={() => setSelectedLanguage("cute")}
                bg="brand.secondary"
                _hover={{ bg: "gray.700" }}
                color="white"
                borderRadius="lg"
                fontSize="sm"
              >
                CuTe DSL
                <Badge
                  ml={2}
                  colorScheme="cyan"
                  fontSize="2xs"
                  variant="subtle"
                  rounded="sm"
                >
                  NEW
                </Badge>
              </MenuItem>
            </MenuList>
          </Menu>
        </Box>
        <Box>
          <Text fontSize="sm" color="whiteAlpha.700">
            Data Type
            {/* dummy button to align -- terrible hack */}
            <IconButton
              aria-label="Data Type Information"
              icon={<FaInfoCircle />}
              size="sm"
              variant="ghost"
              color="transparent"
              visibility="hidden"
              bg="transparent"
            />
          </Text>
          <Menu>
            <MenuButton
              size="sm"
              as={Button}
              rightIcon={<FaChevronDown size={12} color="#a1a1aa" />}
              bg="whiteAlpha.50"
              _hover={{ bg: "whiteAlpha.100", borderColor: "gray.600" }}
              _active={{ bg: "whiteAlpha.150" }}
              _focus={{ borderColor: "blue.500", boxShadow: "none" }}
              color="white"
              w={{ base: "110px", md: "140px" }}
              fontWeight="normal"
              textAlign="left"
              justifyContent="flex-start"
              borderRadius="lg"
            >
              {selectedDataType}
            </MenuButton>
            <MenuList
              bg="brand.secondary"
              borderColor="gray.800"
              p={0}
              borderRadius="lg"
              minW="140px"
            >
              <MenuItem
                key="float32"
                onClick={() => setSelectedDataType("float32")}
                bg="brand.secondary"
                _hover={{ bg: "gray.700" }}
                color="white"
                borderRadius="lg"
                fontSize="sm"
              >
                float32
              </MenuItem>
              <MenuItem
                key="float16"
                onClick={() => setSelectedDataType("float16")}
                bg="brand.secondary"
                _hover={{ bg: "gray.700" }}
                color="white"
                isDisabled={true}
                borderRadius="lg"
                fontSize="sm"
              >
                float16
              </MenuItem>
              <MenuItem
                key="int32"
                onClick={() => setSelectedDataType("int32")}
                bg="brand.secondary"
                _hover={{ bg: "gray.700" }}
                color="white"
                isDisabled={true}
                borderRadius="lg"
                fontSize="sm"
              >
                int32
              </MenuItem>
              <MenuItem
                key="int16"
                onClick={() => setSelectedDataType("int16")}
                bg="brand.secondary"
                _hover={{ bg: "gray.700" }}
                color="white"
                isDisabled={true}
                borderRadius="lg"
                fontSize="sm"
              >
                int16
              </MenuItem>
            </MenuList>
          </Menu>
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
              borderRadius="lg"
              height="40px"
              fontSize="sm"
              fontWeight="semibold"
              color="gray.300"
              leftIcon={<IoRepeat size={16} />}
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
          bg="rgba(59, 130, 246, 0.1)"
          color="rgb(59, 130, 246)"
          size="md"
          onClick={onRun}
          isLoading={isRunning}
          loadingText="Run"
          spinner={<></>}
          disabled={isRunning}
          borderRadius="lg"
          height="40px"
          fontSize="sm"
          fontWeight="semibold"
          px={{ base: 2, md: 6 }}
          minW="70px"
          _hover={{
            bg: "rgba(59, 130, 246, 0.2)",
            transform: "translateY(-1px)",
          }}
          _active={{
            bg: "rgba(59, 130, 246, 0.25)",
          }}
          transition="all 0.2s"
        >
          Run
        </Button>
        <Button
          bg="rgba(34, 197, 94, 0.1)"
          color="rgb(34, 197, 94)"
          size="md"
          onClick={onSubmit}
          isLoading={isSubmitting}
          loadingText="Submit"
          spinner={<></>}
          disabled={isSubmitting}
          borderRadius="lg"
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
