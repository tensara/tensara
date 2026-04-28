import { type Parameter } from "~/types/problem";
import {
  CPP_TYPES,
  PYTHON_TYPES,
  MOJO_TYPES,
  MOJO_DTYPE_CONST,
  CUTE_TYPES,
} from "~/constants/datatypes";
import { FORBIDDEN_PATTERNS } from "~/constants/forbidden";

/** Parameter.type is always C-style (float, int, size_t, uint64_t, etc.). */
function resolveCppType(type: string): string {
  return CPP_TYPES[type] ?? type;
}

function resolvePythonType(type: string): string {
  return PYTHON_TYPES[type] ?? type;
}

function resolveMojoType(type: string): string {
  return MOJO_TYPES[type] ?? type;
}

function resolveCuteType(type: string): string {
  return CUTE_TYPES[type] ?? type;
}

export const generateStarterCode = (
  parameters: Parameter[],
  language: string
) => {
  if (language === "cuda") {
    const names = parameters
      .map((parameter: Parameter) =>
        parameter.pointer === "true" ? parameter.name : null
      )
      .filter(Boolean);
    const paramStr = parameters
      .map(
        (parameter: Parameter) =>
          `${parameter.const === "true" ? "const " : ""}${resolveCppType(parameter.type)}${parameter.pointer === "true" ? "*" : ""} ${parameter.name}`
      )
      .join(", ");
    const paramTypes = new Set(parameters.map((p) => p.type));
    const extraIncludes: string[] = [];
    if (paramTypes.has("float16")) extraIncludes.push("#include <cuda_fp16.h>");
    if (paramTypes.has("float4")) extraIncludes.push("#include <cuda_fp4.h>");
    if (paramTypes.has("float8")) extraIncludes.push("#include <cuda_fp8.h>");
    if (paramTypes.has("uint8_t")) extraIncludes.push("#include <cstdint>");
    if (paramTypes.has("float4") || paramTypes.has("float8")) {
      extraIncludes.push("#include <cstdint>");
    }
    const includesBlock = ["#include <cuda_runtime.h>", ...extraIncludes].join(
      "\n"
    );
    return `${includesBlock}

// Note: ${names.join(", ")} are device pointers
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
          `${parameter.name}${parameter.pointer === "true" ? "" : `: ${resolvePythonType(parameter.type)}`}`
      )
      .join(", ");
    return `import triton
import triton.language as tl

# Note: ${names.join(", ")} are device tensors
def solution(${paramStr}):
    `;
  }
  if (language === "pyptx") {
    const names = parameters
      .map((parameter: Parameter) =>
        parameter.pointer === "true" ? parameter.name : null
      )
      .filter(Boolean);
    const paramStr = parameters
      .map(
        (parameter: Parameter) =>
          `${parameter.name}${parameter.pointer === "true" ? "" : `: ${resolvePythonType(parameter.type)}`}`
      )
      .join(", ");
    return `from pyptx import kernel, ptx, reg, Tile
from pyptx.types import f32, u32


# Note: ${names.join(", ")} are device tensors.
def solution(${paramStr}):
    # for H100/H200 use arch="sm_90a"; for B200 use arch="sm_100a".
    `;
  }
  if (language === "mojo") {
    const pointerParams = parameters.filter((p) => p.pointer === "true");
    const names = pointerParams.map((p) => p.name).filter(Boolean);
    const uniquePtrTypes = [...new Set(pointerParams.map((p) => p.type))];
    const usesDType = pointerParams.length > 0;

    const dtypeVarForType = (t: string) => `dtype_${t}`;
    const dtypeBlock = usesDType
      ? uniquePtrTypes
          .map(
            (t) =>
              `comptime ${dtypeVarForType(t)} = ${MOJO_DTYPE_CONST[t] ?? "DType.float32"}`
          )
          .join("\n")
      : "";

    const paramStr = parameters
      .map((p) =>
        p.pointer === "true"
          ? `${p.name}_addr: Int`
          : `${p.name}: ${resolveMojoType(p.type)}`
      )
      .join(", ");

    const ptrTypeComment =
      uniquePtrTypes.length === 1
        ? `${uniquePtrTypes[0]} arrays`
        : "mixed-dtype arrays";

    const pointerSetup = pointerParams
      .map((p) => {
        const varName = p.name.startsWith("d_") ? p.name.slice(2) : p.name;
        const dtypeVar = dtypeVarForType(p.type);
        return `    ${varName} = UnsafePointer[Scalar[${dtypeVar}], MutExternalOrigin](unsafe_from_address=${p.name}_addr)`;
      })
      .join("\n");

    return `from gpu import thread_idx, block_idx, block_dim
from gpu.host import DeviceContext
from memory import UnsafePointer 

${dtypeBlock ? dtypeBlock : ""}

# Note: ${names.join(", ")} are device pointers to ${ptrTypeComment}
@export
def solution(${paramStr}) raises:
${pointerSetup ? pointerSetup + "\n" : ""}    `;
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
          `${parameter.name}${parameter.pointer === "true" ? `: cute.Tensor` : `: ${resolveCuteType(parameter.type)}`}`
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
  if (language === "cutile") {
    const names = parameters
      .map((parameter: Parameter) =>
        parameter.pointer === "true" ? parameter.name : null
      )
      .filter(Boolean);
    const paramStr = parameters
      .map(
        (parameter: Parameter) =>
          `${parameter.name}${parameter.pointer === "true" ? "" : `: ${resolvePythonType(parameter.type)}`}`
      )
      .join(", ");
    return `import cuda.tile as ct
import cupy

# You can use cupy.cuda.get_current_stream() to get the current stream to launch cuTile kernels.
# Note: ${names.join(", ")} are device tensors
def solution(${paramStr}):
    `;
  }
  return "";
};

/**
 * Map submission language to the key used in FORBIDDEN_PATTERNS
 */
function mapSubmissionLanguage(lang: string): string {
  if (!lang) return lang;
  const l = lang.toLowerCase();
  if (l === "triton" || l === "python" || l === "cute" || l === "cutile")
    return "python";
  if (l === "pyptx") return "python";
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
  if (
    language === "python" ||
    language === "triton" ||
    language === "pyptx" ||
    language === "cute" ||
    language === "cutile"
  ) {
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
