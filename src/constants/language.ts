import { GPU_DISPLAY_NAMES } from "./gpu";

export const LANGUAGE_DISPLAY_NAMES: Record<string, string> = {
  all: "All Languages",
  cuda: "CUDA C++",
  python: "Triton",
  pyptx: "PyPTX",
  mojo: "Mojo",
  cute: "CuTe DSL",
  cutile: "cuTile Python",
} as const;

export const LANGUAGE_PROFILE_DISPLAY_NAMES: Record<string, string> = {
  cuda: "CUDA",
  python: "Triton",
  pyptx: "PyPTX",
  mojo: "Mojo",
  cute: "CuTe",
  cutile: "cuTile",
} as const;

export const PYPTX_SUPPORTED_GPUS = ["H100", "H200", "B200"] as const;

export function isLanguageSupportedOnGpu(
  language: string,
  gpuType: string | null | undefined
): boolean {
  const normalized = (language ?? "").toLowerCase();

  if (normalized === "cutile") {
    return gpuType === "B200";
  }

  if (normalized === "pyptx") {
    return PYPTX_SUPPORTED_GPUS.includes(
      gpuType as (typeof PYPTX_SUPPORTED_GPUS)[number]
    );
  }

  return true;
}

export function getSupportedGpusForLanguage(
  language: string,
  gpuTypes: string[]
): string[] {
  return gpuTypes.filter((gpuType) =>
    isLanguageSupportedOnGpu(language, gpuType)
  );
}

export function getLanguageGpuSupportError(
  language: string,
  gpuType: string | null | undefined
): string | null {
  const normalized = (language ?? "").toLowerCase();
  if (isLanguageSupportedOnGpu(normalized, gpuType)) {
    return null;
  }

  if (normalized === "cutile") {
    return "cuTile Python submissions require the NVIDIA B200 GPU.";
  }

  if (normalized === "pyptx") {
    const supported = PYPTX_SUPPORTED_GPUS.map(
      (gpu) => GPU_DISPLAY_NAMES[gpu]
    ).join(", ");
    return `PyPTX submissions require one of: ${supported}.`;
  }

  return `${LANGUAGE_DISPLAY_NAMES[normalized] ?? language} is not supported on ${gpuType ?? "this GPU"}.`;
}
