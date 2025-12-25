export const LANGUAGE_DISPLAY_NAMES: Record<string, string> = {
  all: "All Languages",
  cuda: "CUDA C++",
  python: "Triton",
  mojo: "Mojo",
  cute: "CuTe DSL",
  hip: "HIP C++",
} as const;

export const LANGUAGE_PROFILE_DISPLAY_NAMES: Record<string, string> = {
  cuda: "CUDA",
  python: "Triton",
  mojo: "Mojo",
  cute: "CuTe",
  hip: "HIP",
} as const;
