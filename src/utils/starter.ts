import { type Parameter } from "~/types/problem";
import { type DataType } from "~/types/misc";
import { CPP_TYPES, PYTHON_TYPES } from "~/constants/datatypes";

export const generateStarterCode = (
  parameters: Parameter[],
  language: string,
  dataType: DataType
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
          `${parameter.name}${parameter.pointer === "true" ? "" : parameter.type === "[VAR]" ? `: ${PYTHON_TYPES[dataType]}` : ": int"}`
      )
      .join(", ");
    return `import triton
import triton.language as tl

# Note: ${names.join(", ")} are all ${dataType} device tensors
def solution(${paramStr}):
  `;
  }
  return "";
};

export function validateCode(
  code: string,
  language: string
): { valid: boolean; error: string } {
  if (language === "python") {
    if (code.includes("torch.") || code.includes("import torch")) {
      return { valid: false, error: "You cannot use PyTorch in the code!" };
    }
    if (/exec\s*\(\s*[^)]*\)/.test(code)) {
      return { valid: false, error: "You cannot use exec() in the code!" };
    }
  }
  return { valid: true, error: "" };
}
