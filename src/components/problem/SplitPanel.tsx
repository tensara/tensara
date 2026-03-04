import React, {
  useCallback,
  useState,
  useEffect,
  createContext,
  useContext,
} from "react";
import { Box } from "@chakra-ui/react";
import { FaChevronLeft, FaChevronRight } from "react-icons/fa";

// Create context for split ratio
export const SplitPanelContext = createContext<{
  splitRatio: number;
}>({
  splitRatio: 30,
});

// Custom hook to use split ratio
export const useSplitPanel = () => useContext(SplitPanelContext);

interface SplitPanelProps {
  leftContent: React.ReactNode;
  rightContent: React.ReactNode;
  initialRatio?: number;
  minLeftWidth?: number;
  minRightWidth?: number;
  splitRatio?: number;
  onSplitRatioChange?: (ratio: number) => void;
  containerId?: string;
  allowCollapse?: boolean;
  snapOffsetPx?: number;
  resizerLineInsetTopPx?: number;
  collapsedLeftLabel?: string;
  collapsedRightLabel?: string;
}

const SplitPanel = ({
  leftContent,
  rightContent,
  initialRatio = 30,
  minLeftWidth = 30,
  minRightWidth = 50,
  splitRatio: controlledRatio,
  onSplitRatioChange,
  containerId = "split-container",
  allowCollapse = false,
  snapOffsetPx = 28,
  resizerLineInsetTopPx = 0,
  collapsedLeftLabel,
  collapsedRightLabel,
}: SplitPanelProps) => {
  const [uncontrolledRatio, setUncontrolledRatio] = useState(initialRatio);
  const [isResizing, setIsResizing] = useState(false);
  const splitRatio = controlledRatio ?? uncontrolledRatio;
  const isCollapsedLeft = allowCollapse && splitRatio <= 0.5;
  const isCollapsedRight = allowCollapse && splitRatio >= 99.5;
  const setRatio = useCallback(
    (ratio: number) => {
      onSplitRatioChange?.(ratio);
      if (controlledRatio === undefined) {
        setUncontrolledRatio(ratio);
      }
    },
    [onSplitRatioChange, controlledRatio]
  );

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

      const containerWidth = containerRect.width;
      const mouseX = e.clientX - containerRect.left;
      let newRatio = (mouseX / containerWidth) * 100;

      if (allowCollapse) {
        if (mouseX <= snapOffsetPx) {
          newRatio = 0;
        } else if (mouseX >= containerWidth - snapOffsetPx) {
          newRatio = 100;
        }
      }

      // Apply min-width constraints
      const minLeftPixels = (containerWidth * minLeftWidth) / 100;
      const minRightPixels = (containerWidth * minRightWidth) / 100;

      if (newRatio !== 0 && newRatio !== 100 && mouseX < minLeftPixels) {
        newRatio = minLeftWidth;
      } else if (
        newRatio !== 0 &&
        newRatio !== 100 &&
        mouseX > containerWidth - minRightPixels
      ) {
        newRatio = 100 - minRightWidth;
      }

      setRatio(newRatio);
    },
    [
      isResizing,
      minLeftWidth,
      minRightWidth,
      containerId,
      setRatio,
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
    <SplitPanelContext.Provider value={{ splitRatio }}>
      <Box
        id={containerId}
        display="flex"
        flexDirection={{ base: "column", md: "row" }}
        h="100%"
        position="relative"
      >
        {/* Left Panel */}
        <Box
          w={{ base: "100%", md: `${splitRatio}%` }}
          h={{ base: "auto", md: "100%" }}
          overflow="hidden"
          minH={0}
          pr={{ base: 0, md: 1.5 }}
          mb={{ base: 4, md: 0 }}
          maxH={{ base: "auto", md: "100%" }}
          display={{ base: "block", md: isCollapsedLeft ? "none" : "block" }}
        >
          {leftContent}
        </Box>

        {/* Resizer Handle - Only visible on desktop */}
        <Box
          display={{ base: "none", md: "block" }}
          position="absolute"
          left={
            isCollapsedLeft
              ? "0%"
              : isCollapsedRight
                ? "100%"
                : `${splitRatio}%`
          }
          transform={
            isCollapsedLeft
              ? "translateX(0)"
              : isCollapsedRight
                ? "translateX(-100%)"
                : "translateX(-50%)"
          }
          width={isCollapsedLeft || isCollapsedRight ? "22px" : "10px"}
          height="100%"
          cursor="col-resize"
          zIndex={2}
          onClick={(e) => e.stopPropagation()}
          onMouseDown={handleMouseDown}
          bg={
            isCollapsedLeft || isCollapsedRight
              ? "whiteAlpha.50"
              : "transparent"
          }
          _hover={{ bg: "whiteAlpha.50" }}
        >
          <Box
            position="absolute"
            left={isCollapsedLeft ? "6px" : isCollapsedRight ? "8px" : "50%"}
            top={`${Math.max(0, resizerLineInsetTopPx)}px`}
            transform={
              isCollapsedLeft || isCollapsedRight ? "none" : "translateX(-50%)"
            }
            width="2px"
            height={
              resizerLineInsetTopPx > 0
                ? `calc(100% - ${Math.max(0, resizerLineInsetTopPx)}px)`
                : "100%"
            }
            bg={"whiteAlpha.100"}
            borderRadius="full"
            transition="all 0.2s ease"
            _hover={{ bg: "whiteAlpha.300" }}
            boxShadow={
              isResizing ? "0 0 0 1px rgba(78, 201, 176, 0.35)" : "none"
            }
          />
          {(isCollapsedLeft || isCollapsedRight) &&
            !isResizing &&
            (isCollapsedLeft ? collapsedLeftLabel : collapsedRightLabel) && (
              <Box
                position="absolute"
                inset={0}
                display="flex"
                alignItems="center"
                justifyContent="center"
                pointerEvents="none"
                userSelect="none"
              >
                <Box
                  as="span"
                  fontSize="10px"
                  fontWeight="600"
                  color="gray.400"
                  transform="rotate(-90deg)"
                  whiteSpace="nowrap"
                  letterSpacing="0.06em"
                  textTransform="uppercase"
                >
                  {isCollapsedLeft ? collapsedLeftLabel : collapsedRightLabel}
                </Box>
              </Box>
            )}
          {allowCollapse && !isResizing && (
            <Box
              as="button"
              type="button"
              position="absolute"
              top="8px"
              left="50%"
              transform="translateX(-50%)"
              w="16px"
              h="16px"
              borderRadius="full"
              border="1px solid"
              borderColor="whiteAlpha.300"
              color="gray.300"
              fontSize="11px"
              lineHeight="1"
              display="flex"
              alignItems="center"
              justifyContent="center"
              bg="rgba(17, 17, 17, 0.9)"
              _hover={{ bg: "whiteAlpha.200", color: "white" }}
              pointerEvents="auto"
              onMouseDown={(e: React.MouseEvent) => e.stopPropagation()}
              onClick={(e: React.MouseEvent) => {
                e.stopPropagation();
                const minOpen = Math.max(0, minLeftWidth);
                const maxOpen = Math.min(100, 100 - minRightWidth);
                const fallback = Math.min(
                  maxOpen,
                  Math.max(minOpen, initialRatio)
                );

                if (isCollapsedLeft) {
                  setRatio(fallback);
                  return;
                }

                setRatio(0);
              }}
              aria-label={
                isCollapsedLeft ? "Expand problem panel" : "Expand editor"
              }
            >
              {isCollapsedLeft ? (
                <FaChevronRight size={9} />
              ) : (
                <FaChevronLeft size={9} />
              )}
            </Box>
          )}
        </Box>

        {/* Right Panel */}
        <Box
          display={{ base: "none", md: isCollapsedRight ? "none" : "block" }}
          w={{ base: "100%", md: `${100 - splitRatio}%` }}
          h={{ base: "auto", md: "100%" }}
          minH={{ base: "50vh", md: "auto" }}
          pl={{ base: 0, md: 1.5 }}
          overflow="hidden"
        >
          {rightContent}
        </Box>
      </Box>
    </SplitPanelContext.Provider>
  );
};

export default SplitPanel;
