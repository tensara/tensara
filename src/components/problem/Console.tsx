import { useRef, useState } from "react";
import { Box, Text, VStack } from "@chakra-ui/react";
import { motion, useDragControls } from "framer-motion";

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
    <Box w="100%" border="1px solid" borderColor="gray.700" borderRadius="md">
      {/* Drag handle */}
      <Box
        cursor="ns-resize"
        h="6px"
        bg="gray.600"
        borderTopRadius="md"
        onMouseDown={handleMouseDown}
      />
      {/* Console area */}
      <Box
        bg="gray.900"
        color="white"
        p={3}
        fontFamily="mono"
        fontSize="sm"
        overflowY="auto"
        maxH="400px"
        minH="100px"
        h={`${height}px`}
        borderBottomRadius="md"
      >
        {output.length === 0 ? (
          <Text color="gray.500">Console output will appear here...</Text>
        ) : (
          <VStack align="start" spacing={1}>
            {output.map((line, i) => (
              <Text key={i}>{line}</Text>
            ))}
          </VStack>
        )}
      </Box>
    </Box>
  );
};

export default ResizableConsole;
