export const GPU_DISPLAY_NAMES: Record<string, string> = {
  "all": "All GPUs",
  "T4": "NVIDIA T4",
  "H100": "NVIDIA H100",
  "A100-80GB": "NVIDIA A100-80GB",
  "A10G": "NVIDIA A10G",
  "L40S": "NVIDIA L40S",
  "L4": "NVIDIA L4"
} as const;

export const GPU_TYPES = ["T4", "H100", "A100-80GB", "A10G", "L40S", "L4"] as const; 