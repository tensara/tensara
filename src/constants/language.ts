import { GPU_DISPLAY_NAMES } from "./gpu";

export type LanguageResource = {
  label: string;
  href: string;
  isExternal?: boolean;
};

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

export const LANGUAGE_RESOURCES: Record<string, LanguageResource[]> = {
  pyptx: [
    { label: "Tensara blog", href: "https://tensara.org/blog/submitting-kernels-on-tensara-with-pyptx-1623386" },
    {
      label: "PyPTX docs",
      href: "https://pyptx.dev/getting-started/",
      isExternal: true,
    },
  ],
  cute: [
    {
      label: "CuTe quickstart",
      href: "https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/quick_start.html",
      isExternal: true,
    },
    {
      label: "CuTe docs",
      href: "https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl.html",
      isExternal: true,
    },
  ],
  cutile: [
    {
      label: "cuTile quickstart",
      href: "https://docs.nvidia.com/cuda/cutile-python/quickstart.html",
      isExternal: true,
    },
    {
      label: "cuTile docs",
      href: "https://docs.nvidia.com/cuda/cutile-python/",
      isExternal: true,
    },
  ],
} as const;

export function getLanguageResources(language: string): LanguageResource[] {
  const normalized = (language ?? "").toLowerCase();
  return LANGUAGE_RESOURCES[normalized] ?? [];
}

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
