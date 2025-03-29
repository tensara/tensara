export const LANGUAGE_DISPLAY_NAMES: Record<string, string> = {
  "all": "All Languages",
  "cuda": "CUDA C++",
  "python": "Python (Triton)",
} as const;

export const LANGUAGE_PROFILE_DISPLAY_NAMES: Record<string, string> = {
  "cuda": "CUDA",
  "python": "Triton",
} as const;

export const IS_DISABLED_LANGUAGE: Record<string, boolean> = {
  "cuda": false,
  "python": true,
} as const;