/**
 * Parameter types are always C-style (float, int, size_t, uint64_t, etc.).
 * These maps convert C-style type names to each language's syntax.
 */
export const CPP_TYPES: Record<string, string> = {
  float: "float",
  int: "int",
  size_t: "size_t",
  uint64_t: "uint64_t",
} as const;

export const PYTHON_TYPES: Record<string, string> = {
  float: "float",
  int: "int",
  size_t: "int",
  uint64_t: "int",
} as const;

export const MOJO_TYPES: Record<string, string> = {
  float: "Float32",
  int: "Int32",
  size_t: "Int32",
  uint64_t: "Int32",
} as const;

/** C-style type -> Mojo DType constant for comptime dtype (pointer element type). */
export const MOJO_DTYPE_CONST: Record<string, string> = {
  float: "DType.float32",
  int: "DType.int32",
  size_t: "DType.int32",
  uint64_t: "DType.int32",
} as const;

export const CUTE_TYPES: Record<string, string> = {
  float: "cute.Float32",
  int: "cute.Int32",
  size_t: "cute.Int32",
  uint64_t: "cute.Int32",
} as const;
