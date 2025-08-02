import React, {
  useCallback,
  useState,
  useEffect,
  createContext,
  useContext,
} from "react";
import { Box } from "@chakra-ui/react";

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
}

const SplitPanel = ({
  leftContent,
  rightContent,
  initialRatio = 30,
  minLeftWidth = 30,
  minRightWidth = 50,
}: SplitPanelProps) => {
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
        .getElementById("split-container")
        ?.getBoundingClientRect();

      if (!containerRect) return;

      const containerWidth = containerRect.width;
      const mouseX = e.clientX - containerRect.left;
      let newRatio = (mouseX / containerWidth) * 100;

      // Apply min-width constraints
      const minLeftPixels = (containerWidth * minLeftWidth) / 100;
      const minRightPixels = (containerWidth * minRightWidth) / 100;

      if (mouseX < minLeftPixels) {
        newRatio = minLeftWidth;
      } else if (mouseX > containerWidth - minRightPixels) {
        newRatio = 100 - minRightWidth;
      }

      setSplitRatio(newRatio);
    },
    [isResizing, minLeftWidth, minRightWidth]
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
        id="split-container"
        display="flex"
        flexDirection={{ base: "column", md: "row" }}
        h="100%"
        maxH="calc(100vh - 120px)"
        position="relative"
      >
        {/* Left Panel */}
        <Box
          w={{ base: "100%", md: `${splitRatio}%` }}
          h={{ base: "auto", md: "100%" }}
          overflowY="auto"
          pr={{ base: 0, md: 4 }}
          mb={{ base: 4, md: 0 }}
          maxH={{ base: "auto", md: "100%" }}
        >
          {leftContent}
        </Box>

        {/* Resizer Handle - Only visible on desktop */}
        <Box
          display={{ base: "none", md: "block" }}
          position="absolute"
          left={`${splitRatio}%`}
          transform="translateX(-50%)"
          width="6px"
          height="100%"
          cursor="col-resize"
          zIndex={2}
          onClick={(e) => e.stopPropagation()}
          onMouseDown={handleMouseDown}
        >
          <Box
            position="absolute"
            left="50%"
            top="50%"
            transform="translate(-50%, -50%)"
            width="6px"
            height={isResizing ? "120px" : "80px"}
            bg={"whiteAlpha.200"}
            borderRadius="full"
            transition="all 0.2s ease"
            boxShadow={isResizing ? "0 0 10px 2px brand.dark" : "none"}
          />

          {/* GPU Icon */}
          <Box
            position="absolute"
            left="50%"
            top="calc(50% + 50px)"
            transform="translate(-50%, -50%)"
            width="24px"
            height="24px"
            opacity={isResizing ? 0.9 : 0.5}
            transition="all 0.3s ease"
            _hover={{ opacity: 0.9 }}
            filter={isResizing ? "brightness(1.5)" : "brightness(1)"}
          ></Box>
        </Box>

        {/* Right Panel */}
        <Box
          display={{ base: "none", md: "block" }}
          w={{ base: "100%", md: `${100 - splitRatio}%` }}
          h={{ base: "auto", md: "100%" }}
          minH={{ base: "50vh", md: "auto" }}
          pl={{ base: 0, md: 4 }}
        >
          {rightContent}
        </Box>
      </Box>
    </SplitPanelContext.Provider>
  );
};

export default SplitPanel;
