// useSplitPanel.ts
import { useState, useCallback, useEffect } from "react";

interface SplitPanelOptions {
  initialRatio?: number;
  minRatio?: number;
  maxRatio?: number;
  containerId?: string;
}

export function useSplitPanel({
  initialRatio = 40,
  minRatio = 35,
  maxRatio = 52,
  containerId = "split-container",
}: SplitPanelOptions = {}) {
  const [splitRatio, setSplitRatio] = useState<number>(initialRatio);
  const [isDragging, setIsDragging] = useState<boolean>(false);

  const handleMouseDown = useCallback(() => {
    setIsDragging(true);
    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";
  }, []);

  const handleMouseUp = useCallback(() => {
    setIsDragging(false);
    document.body.style.cursor = "default";
    document.body.style.userSelect = "auto";
  }, []);

  useEffect(() => {
    if (isDragging) {
      const handleMouseMove = (e: MouseEvent) => {
        if (!isDragging) return;

        const container = document.getElementById(containerId);
        if (!container) return;

        const containerRect = container.getBoundingClientRect();
        const newRatio =
          ((e.clientX - containerRect.left) / containerRect.width) * 100;

        // Clamp the ratio within min and max bounds
        const clampedRatio = Math.min(Math.max(newRatio, minRatio), maxRatio);
        setSplitRatio(clampedRatio);
      };

      window.addEventListener("mousemove", handleMouseMove);
      window.addEventListener("mouseup", handleMouseUp);

      return () => {
        window.removeEventListener("mousemove", handleMouseMove);
        window.removeEventListener("mouseup", handleMouseUp);
      };
    }
  }, [isDragging, handleMouseUp, containerId, minRatio, maxRatio]);

  return {
    splitRatio,
    isDragging,
    handleMouseDown,
    handleMouseUp,
  };
}
