import { Parameter } from "~/types/problem";
import { DataType } from "~/types/misc";

const CPP_TYPES: Record<DataType, string> = {
  "float32": "float",
  "float16": "double",
  "int32": "int",
  "int16": "short"
} as const;

export const generateStarterCode = (parameters: Parameter[], language: string, dataType: DataType) => {
  if (language === "cuda") {
    const names = parameters.map((parameter: Parameter) => parameter.pointer === "true" ? parameter.name : null).filter(Boolean);
    const paramStr = parameters.map((parameter: Parameter) => 
      `${parameter.const === "true" ? "const " : ""}${parameter.type === "[VAR]" ? CPP_TYPES[dataType] : parameter.type}${parameter.pointer === "true" ? "*" : ""} ${parameter.name}`
    ).join(", ");
    return `#include <cuda_runtime.h>

// Note: ${names.join(", ")} are all device pointers to ${dataType} arrays
extern "C" void solution(${paramStr}) {    
}`
  }
  return "";
};
