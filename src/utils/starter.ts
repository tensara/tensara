import { type Parameter } from "~/types/problem";
import { type DataType } from "~/types/misc";
import {
  CPP_TYPES,
  PYTHON_TYPES,
  MOJO_TYPES,
  PYTHON_MISC_TYPES,
  MOJO_MISC_TYPES,
  CUTE_TYPES,
  CUTE_MISC_TYPES,
  CPP_TYPES as HIP_TYPES,
} from "~/constants/datatypes";
import { FORBIDDEN_PATTERNS } from "~/constants/forbidden";

export const generateStarterCode = (
  parameters: Parameter[],
  language: string,
  dataType: DataType
) => {
  // Safety check: return empty string if parameters is undefined or not an array
  if (!parameters || !Array.isArray(parameters)) {
    return "";
  }

  if (language === "cuda") {
    const names = parameters
      .map((parameter: Parameter) =>
        parameter.pointer === "true" ? parameter.name : null
      )
      .filter(Boolean);
    const paramStr = parameters
      .map(
        (parameter: Parameter) =>
          `${parameter.const === "true" ? "const " : ""}${parameter.type === "[VAR]" ? CPP_TYPES[dataType] : parameter.type}${parameter.pointer === "true" ? "*" : ""} ${parameter.name}`
      )
      .join(", ");
    return `#include <cuda_runtime.h>

// Note: ${names.join(", ")} are all device pointers to ${dataType} arrays
extern "C" void solution(${paramStr}) {
}`;
  }
  if (language === "python") {
    const names = parameters
      .map((parameter: Parameter) =>
        parameter.pointer === "true" ? parameter.name : null
      )
      .filter(Boolean);
    const paramStr = parameters
      .map(
        (parameter: Parameter) =>
          `${parameter.name}${parameter.pointer === "true" ? "" : parameter.type === "[VAR]" ? `: ${PYTHON_TYPES[dataType]}` : `: ${PYTHON_MISC_TYPES[parameter.type]}`}`
      )
      .join(", ");
    return `import triton
import triton.language as tl

# Note: ${names.join(", ")} are all ${dataType} device tensors
def solution(${paramStr}):
  `;
  }
  if (language === "mojo") {
    const names = parameters
      .map((parameter: Parameter) =>
        parameter.pointer === "true" ? parameter.name : null
      )
      .filter(Boolean);
    const paramStr = parameters
      .map(
        (parameter: Parameter) =>
          `${parameter.name}: ${parameter.pointer === "true" ? `UnsafePointer[${MOJO_TYPES[dataType]}]` : parameter.type === "[VAR]" ? MOJO_TYPES[dataType] : MOJO_MISC_TYPES[parameter.type]}`
      )
      .join(", ");
    return `from gpu.host import DeviceContext
from gpu.id import block_dim, block_idx, thread_idx
from memory import UnsafePointer

# Note: ${names.join(", ")} are all device pointers to ${dataType} arrays
@export
fn solution(${paramStr}) raises:
  `;
  }
  if (language == "cute") {
    const names = parameters
      .map((parameter: Parameter) =>
        parameter.pointer === "true" ? parameter.name : null
      )
      .filter(Boolean);
    const paramStr = parameters
      .map(
        (parameter: Parameter) =>
          `${parameter.name}${parameter.pointer === "true" ? `: cute.Tensor` : parameter.type === "[VAR]" ? `: ${CUTE_TYPES[dataType]}` : `: ${CUTE_MISC_TYPES[parameter.type]}`}`
      )
      .filter(Boolean)
      .join(", ");
    return `import cutlass
import cutlass.cute as cute

# Note: ${names.join(", ")} are all device tensors
@cute.jit
def solution(${paramStr}):
  `;
  }
  if (language === "hip") {
    const names = parameters
      .map((parameter: Parameter) =>
        parameter.pointer === "true" ? parameter.name : null
      )
      .filter(Boolean);
    const paramStr = parameters
      .map(
        (parameter: Parameter) =>
          `${parameter.const === "true" ? "const " : ""}${parameter.type === "[VAR]" ? HIP_TYPES[dataType] : parameter.type}${parameter.pointer === "true" ? "*" : ""} ${parameter.name}`
      )
      .join(", ");
    return `#include <hip/hip_runtime.h>

// Note: ${names.join(", ")} are all device pointers to ${dataType} arrays
extern "C" void solution(${paramStr}) {
}`;
  }
  return "";
};

/**
 * Map submission language to the key used in FORBIDDEN_PATTERNS
 */
function mapSubmissionLanguage(lang: string): string {
  if (!lang) return lang;
  const l = lang.toLowerCase();
  if (l === "triton" || l === "python" || l === "cute") return "python";
  if (l === "cuda" || l === "c++" || l === "cpp") return "cuda";
  if (l === "mojo") return "mojo";
  return l;
}

/**
 * Strip comments and string literals so that commented-out occurrences are ignored
 */
function stripCommentsAndStrings(s: string, languageKey: string): string {
  if (!s) return s;

  // Python: remove triple-quoted strings, normal strings, and line comments (#)
  if (languageKey === "python") {
    s = s.replace(/"""[\s\S]*?"""/g, "");
    s = s.replace(/'''[\s\S]*?'''/g, "");
    s = s.replace(/"(?:\\.|[^"\\])*"/g, "");
    s = s.replace(/'(?:\\.|[^'\\])*'/g, "");
    s = s.replace(/#.*$/gm, "");
    return s;
  }

  // C-like / Mojo: remove double/single-quoted strings and C-style comments
  s = s.replace(/"(?:\\.|[^"\\])*"/g, "");
  s = s.replace(/'(?:\\.|[^'\\])*'/g, "");
  s = s.replace(/\/\*[\s\S]*?\*\//g, "");
  s = s.replace(/\/\/.*$/gm, "");
  return s;
}

/**
 * Check if code matches any forbidden patterns for the given language
 * Returns the matched pattern if found, null otherwise
 */
function findForbiddenMatch(lang: string, src: string): string | null {
  if (!src || typeof src !== "string") return null;

  const mapped = mapSubmissionLanguage(lang);
  const list = FORBIDDEN_PATTERNS[mapped] ?? [];

  const cleaned = stripCommentsAndStrings(src, mapped);

  for (const p of list) {
    try {
      const re = new RegExp(p, "i");
      if (re.test(cleaned)) return p;
    } catch (e) {
      // ignore invalid regex entries
      console.warn("Invalid forbidden pattern", p, e);
    }
  }
  return null;
}

/**
 * Validate code for forbidden patterns and disallowed usage
 */
export function validateCode(
  code: string,
  language: string
): { valid: boolean; error: string; details?: string } {
  // legacy validation checks (kept for backward compatibility and specific error messages)
  if (language === "python" || language === "triton" || language === "cute") {
    if (code.includes("torch.") || code.includes("import torch")) {
      return { valid: false, error: "You cannot use PyTorch in the code!" };
    }
    if (/exec\s*\(\s*[^)]*\)/.test(code)) {
      return { valid: false, error: "You cannot use exec() in the code!" };
    }
  }

  // Check forbidden patterns from config
  const matched = findForbiddenMatch(language, code);
  if (matched) {
    return {
      valid: false,
      error: "Forbidden usage detected",
      details: `Matched forbidden pattern: ${matched}`,
    };
  }

  return { valid: true, error: "" };
}
