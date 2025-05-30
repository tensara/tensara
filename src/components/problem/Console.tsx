import { useRef, useState } from "react";
import { Box, Text, VStack, HStack } from "@chakra-ui/react";

const getColorForLine = (
  line: string
): { label: string; color: string; bg: string } => {
  if (line.startsWith("Error:"))
    return { label: "ERROR", color: "red.500", bg: "red.900" };
  if (line.startsWith("Result: PASSED"))
    return { label: "RESULT", color: "green.400", bg: "green.900" };
  if (line.startsWith("Result: FAILED"))
    return { label: "RESULT", color: "red.500", bg: "red.900" };
  if (line.startsWith("Input"))
    return { label: "INPUT", color: "gray.300", bg: "gray.800" };
  if (line.startsWith("Output"))
    return { label: "OUTPUT", color: "gray.300", bg: "gray.800" };
  if (line.startsWith("Stdout"))
    return { label: "STDOUT", color: "gray.300", bg: "gray.800" };
  if (line.startsWith("Stderr"))
    return { label: "STDERR", color: "gray.300", bg: "gray.800" };
  if (
    line.startsWith("Status:") ||
    line.startsWith("IN_QUEUE") ||
    line.startsWith("COMPILING")
  )
    return { label: "STATUS", color: "gray.300", bg: "gray.800" };
  return { label: "LOG", color: "gray.400", bg: "gray.900" };
};

type ConsoleProps = {
  output: string[];
};

const ResizableConsole = ({ output }: ConsoleProps) => {
  const [height, setHeight] = useState(200);
  const isDragging = useRef(false);

  const handleMouseDown = (e: React.MouseEvent) => {
    isDragging.current = true;
    const startY = e.clientY;
    const startHeight = height;

    const onMouseMove = (moveEvent: MouseEvent) => {
      const deltaY = moveEvent.clientY - startY;
      setHeight(Math.max(100, startHeight - deltaY));
    };

    const onMouseUp = () => {
      isDragging.current = false;
      document.removeEventListener("mousemove", onMouseMove);
      document.removeEventListener("mouseup", onMouseUp);
    };

    document.addEventListener("mousemove", onMouseMove);
    document.addEventListener("mouseup", onMouseUp);
  };

  return (
    <Box
      w="100%"
      border="1px solid"
      borderColor="gray.700"
      borderRadius="md"
      fontFamily="mono"
      fontSize="sm"
      bg="gray.900"
    >
      <Box
        cursor="ns-resize"
        h="6px"
        bg="gray.600"
        borderTopRadius="md"
        onMouseDown={handleMouseDown}
      />
      <Box
        color="white"
        p={3}
        overflowY="auto"
        maxH="400px"
        minH="100px"
        h={`${height}px`}
        borderBottomRadius="md"
      >
        {output.length === 0 ? (
          <Text color="gray.500">Console output will appear here...</Text>
        ) : (
          <VStack align="start" spacing={2}>
            {output.map((line, i) => {
              const { label, color, bg } = getColorForLine(line);
              return (
                <HStack key={i} align="start" spacing={2} w="full">
                  <Box
                    px={2}
                    py={0.5}
                    minW="70px"
                    fontWeight="bold"
                    color={color}
                    bg={bg}
                    borderRadius="sm"
                    fontSize="xs"
                    textAlign="center"
                  >
                    {label}
                  </Box>
                  <Text whiteSpace="pre-wrap" color="gray.200">
                    {line}
                  </Text>
                </HStack>
              );
            })}
          </VStack>
        )}
      </Box>
    </Box>
  );
};

export default ResizableConsole;
