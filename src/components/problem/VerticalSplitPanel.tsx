import React, { useCallback, useState, useEffect } from "react";
import { Box } from "@chakra-ui/react";

interface VerticalSplitPanelProps {
  topContent: React.ReactNode;
  bottomContent: React.ReactNode;
  initialRatio?: number;
  minTopHeight?: number;
  minBottomHeight?: number;
  splitRatio?: number;
  onSplitRatioChange?: (ratio: number) => void;
  containerId?: string;
  allowCollapse?: boolean;
  snapOffsetPx?: number;
  collapsedTopLabel?: string;
  collapsedBottomLabel?: string;
}

const VerticalSplitPanel = ({
  topContent,
  bottomContent,
  initialRatio = 70,
  minTopHeight = 30,
  minBottomHeight = 20,
  splitRatio: controlledRatio,
  onSplitRatioChange,
  containerId = "vertical-split-container",
  allowCollapse = false,
  snapOffsetPx = 24,
  collapsedTopLabel,
  collapsedBottomLabel,
}: VerticalSplitPanelProps) => {
  const [uncontrolledRatio, setUncontrolledRatio] = useState(initialRatio);
  const [isResizing, setIsResizing] = useState(false);
  const splitRatio = controlledRatio ?? uncontrolledRatio;
  const isCollapsedTop = allowCollapse && splitRatio <= 0.5;
  const isCollapsedBottom = allowCollapse && splitRatio >= 99.5;

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    setIsResizing(true);
  }, []);

  const handleMouseMove = useCallback(
    (e: MouseEvent) => {
      if (!isResizing) return;

      const containerRect = document
        .getElementById(containerId)
        ?.getBoundingClientRect();

      if (!containerRect) return;

      const containerHeight = containerRect.height;
      const mouseY = e.clientY - containerRect.top;
      let newRatio = (mouseY / containerHeight) * 100;

      if (allowCollapse) {
        if (mouseY <= snapOffsetPx) {
          newRatio = 0;
        } else if (mouseY >= containerHeight - snapOffsetPx) {
          newRatio = 100;
        }
      }

      // Apply min-height constraints
      const minTopPixels = (containerHeight * minTopHeight) / 100;
      const minBottomPixels = (containerHeight * minBottomHeight) / 100;

      if (newRatio !== 0 && newRatio !== 100 && mouseY < minTopPixels) {
        newRatio = minTopHeight;
      } else if (
        newRatio !== 0 &&
        newRatio !== 100 &&
        mouseY > containerHeight - minBottomPixels
      ) {
        newRatio = 100 - minBottomHeight;
      }

      onSplitRatioChange?.(newRatio);
      if (controlledRatio === undefined) {
        setUncontrolledRatio(newRatio);
      }
    },
    [
      isResizing,
      minTopHeight,
      minBottomHeight,
      containerId,
      onSplitRatioChange,
      controlledRatio,
      allowCollapse,
      snapOffsetPx,
    ]
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
      id={containerId}
      display="flex"
      flexDirection="column"
      h="100%"
      gap={0}
    >
      {/* Top Panel */}
      <Box
        flex={`${splitRatio}`}
        overflow="hidden"
        minH={0}
        display={isCollapsedTop ? "none" : "block"}
      >
        {topContent}
      </Box>

      {/* Resizer Handle */}
      <Box
        h={isCollapsedTop || isCollapsedBottom ? "18px" : "10px"}
        cursor="row-resize"
        display="flex"
        alignItems="center"
        justifyContent="center"
        onClick={(e) => e.stopPropagation()}
        onMouseDown={handleMouseDown}
        py={0}
        bg={
          isCollapsedTop || isCollapsedBottom ? "whiteAlpha.50" : "transparent"
        }
        _hover={{ bg: "whiteAlpha.50" }}
        position="relative"
      >
        <Box
          height="2px"
          width="100%"
          bg={isResizing ? "whiteAlpha.300" : "whiteAlpha.100"}
          borderRadius="full"
          transition="all 0.2s ease"
          _hover={{ bg: "whiteAlpha.300" }}
        />
        {(isCollapsedTop || isCollapsedBottom) &&
          !isResizing &&
          (isCollapsedTop ? collapsedTopLabel : collapsedBottomLabel) && (
            <Box
              position="absolute"
              top="50%"
              left="50%"
              transform="translate(-50%, -50%)"
              fontSize="10px"
              fontWeight="600"
              color="gray.400"
              pointerEvents="none"
              userSelect="none"
              letterSpacing="0.06em"
              textTransform="uppercase"
              bg="brand.secondary"
              px={2}
              py={0.5}
              borderRadius="md"
              border="1px solid"
              borderColor="whiteAlpha.100"
            >
              {isCollapsedTop ? collapsedTopLabel : collapsedBottomLabel}
            </Box>
          )}
      </Box>

      {/* Bottom Panel */}
      <Box
        flex={`${100 - splitRatio}`}
        overflow="hidden"
        minH={0}
        display={isCollapsedBottom ? "none" : "block"}
      >
        {bottomContent}
      </Box>
    </Box>
  );
};

export default VerticalSplitPanel;
