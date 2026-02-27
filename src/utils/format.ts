export const formatRuntime = (runtime: number | null | undefined): string => {
  if (runtime == null) return "N/A";
  if (runtime <= 1) {
    const microseconds = runtime * 1000;
    return `${microseconds.toFixed(2)} μs`;
  }
  if (runtime >= 1000) {
    const seconds = runtime / 1000;
    return `${seconds.toFixed(2)} s`;
  }
  return `${runtime.toFixed(2)} ms`;
};
