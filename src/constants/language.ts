export const LANGUAGE_DISPLAY_NAMES: Record<string, string> = {
  all: "All Languages",
  cuda: "CUDA C++",
  python: "Triton",
  mojo: "Mojo",
  cute: "CuTe DSL",
  cutile: "cuTile Python",
} as const;

export const LANGUAGE_PROFILE_DISPLAY_NAMES: Record<string, string> = {
  cuda: "CUDA",
  python: "Triton",
  mojo: "Mojo",
  cute: "CuTe",
  cutile: "cuTile",
} as const;
