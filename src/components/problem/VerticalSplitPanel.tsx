import React, { useCallback, useState, useEffect } from "react";
import { Box } from "@chakra-ui/react";

interface VerticalSplitPanelProps {
  topContent: React.ReactNode;
  bottomContent: React.ReactNode;
  initialRatio?: number;
  minTopHeight?: number;
  minBottomHeight?: number;
}

const VerticalSplitPanel = ({
  topContent,
  bottomContent,
  initialRatio = 70,
  minTopHeight = 30,
  minBottomHeight = 20,
}: VerticalSplitPanelProps) => {
  const [splitRatio, setSplitRatio] = useState(initialRatio);
  const [isResizing, setIsResizing] = useState(false);

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    setIsResizing(true);
  }, []);

  const handleMouseMove = useCallback(
    (e: MouseEvent) => {
      if (!isResizing) return;

      const containerRect = document
        .getElementById("vertical-split-container")
        ?.getBoundingClientRect();

      if (!containerRect) return;

      const containerHeight = containerRect.height;
      const mouseY = e.clientY - containerRect.top;
      let newRatio = (mouseY / containerHeight) * 100;

      // Apply min-height constraints
      const minTopPixels = (containerHeight * minTopHeight) / 100;
      const minBottomPixels = (containerHeight * minBottomHeight) / 100;

      if (mouseY < minTopPixels) {
        newRatio = minTopHeight;
      } else if (mouseY > containerHeight - minBottomPixels) {
        newRatio = 100 - minBottomHeight;
      }

      setSplitRatio(newRatio);
    },
    [isResizing, minTopHeight, minBottomHeight]
  );

  const handleMouseUp = useCallback(() => {
    setIsResizing(false);
  }, []);

  useEffect(() => {
    if (isResizing) {
      document.addEventListener("mousemove", handleMouseMove);
      document.addEventListener("mouseup", handleMouseUp);
    } else {
      document.removeEventListener("mousemove", handleMouseMove);
      document.removeEventListener("mouseup", handleMouseUp);
    }

    return () => {
      document.removeEventListener("mousemove", handleMouseMove);
      document.removeEventListener("mouseup", handleMouseUp);
    };
  }, [isResizing, handleMouseMove, handleMouseUp]);

  return (
    <Box
      id="vertical-split-container"
      display="flex"
      flexDirection="column"
      h="100%"
      gap={1}
    >
      {/* Top Panel */}
      <Box flex={`${splitRatio}`} overflow="hidden" minH={0}>
        {topContent}
      </Box>

      {/* Resizer Handle */}
      <Box
        h="2px"
        cursor="row-resize"
        display="flex"
        alignItems="center"
        justifyContent="center"
        onClick={(e) => e.stopPropagation()}
        onMouseDown={handleMouseDown}
        py={0}
      >
        <Box
          height="2px"
          width={isResizing ? "60px" : "40px"}
          bg="whiteAlpha.200"
          borderRadius="full"
          transition="all 0.2s ease"
          _hover={{ bg: "whiteAlpha.400" }}
        />
      </Box>

      {/* Bottom Panel */}
      <Box flex={`${100 - splitRatio}`} overflow="hidden" minH={0}>
        {bottomContent}
      </Box>
    </Box>
  );
};

export default VerticalSplitPanel;
