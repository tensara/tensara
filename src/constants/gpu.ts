export const GPU_DISPLAY_NAMES: Record<string, string> = {
  all: "All GPUs",
  T4: "Tesla T4",
  H100: "NVIDIA H100",
  H200: "NVIDIA H200",
  B200: "NVIDIA B200",
  "A100-80GB": "NVIDIA A100",
  A10G: "NVIDIA A10G",
  L40S: "NVIDIA L40S",
  L4: "NVIDIA L4",
  MI300X: "AMD MI300X",
} as const;

export const gpuTypes = Object.keys(GPU_DISPLAY_NAMES);

export const GPU_DISPLAY_ON_PROFILE = {
  T4: "T4",
  H100: "H100",
  H200: "H200",
  B200: "B200",
  "A100-80GB": "A100",
  A10G: "A10G",
  L40S: "L40S",
  L4: "L4",
  MI300X: "MI300X",
  none: "N/A",
} as const;
