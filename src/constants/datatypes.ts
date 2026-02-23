/**
 * Parameter types are always C-style (float, int, size_t, uint64_t, etc.).
 * These maps convert C-style type names to each language's syntax.
 */

export const CPP_TYPES: Record<string, string> = {
  float: "float",
  double: "double",
  float16: "__half",
  float8: "uint8_t",
  float4: "uint8_t",
  int: "int",
  uint8_t: "uint8_t",
  size_t: "size_t",
  uint32_t: "uint32_t",
  uint64_t: "uint64_t",
} as const;

export const PYTHON_TYPES: Record<string, string> = {
  float: "float",
  double: "float",
  float16: "float",
  float8: "float",
  float4: "float",
  int: "int",
  uint8_t: "int",
  size_t: "int",
  uint32_t: "int",
  uint64_t: "int",
} as const;

export const MOJO_TYPES: Record<string, string> = {
  float: "Float32",
  double: "Float64",
  float16: "Float16",
  float8: "Float8_e4m3fn",
  float4: "UInt8",
  int: "Int32",
  uint8_t: "UInt8",
  size_t: "Int64",
  uint32_t: "UInt32",
  uint64_t: "UInt64",
} as const;

/** C-style type -> Mojo DType constant for comptime dtype (pointer element type). */
export const MOJO_DTYPE_CONST: Record<string, string> = {
  float: "DType.float32",
  double: "DType.float64",
  float16: "DType.float16",
  float8: "DType.float8_e4m3fn",
  float4: "DType.uint8",
  int: "DType.int32",
  uint8_t: "DType.uint8",
  size_t: "DType.int64",
  uint32_t: "DType.uint32",
  uint64_t: "DType.uint64",
} as const;

export const CUTE_TYPES: Record<string, string> = {
  float: "cute.Float32",
  double: "cute.Float64",
  float16: "cute.Float16",
  float8: "cute.Float8E4M3",
  float4: "cute.UInt8",
  int: "cute.Int32",
  uint8_t: "cute.UInt8",
  size_t: "cute.Int64",
  uint32_t: "cute.UInt32",
  uint64_t: "cute.UInt64",
} as const;
