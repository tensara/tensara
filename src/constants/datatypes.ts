import { type DataType } from "~/types/misc";

export const CPP_TYPES: Record<DataType, string> = {
  float32: "float",
  float16: "double",
  int32: "int",
  int16: "short",
} as const;

export const PYTHON_TYPES: Record<DataType, string> = {
  float32: "float",
  float16: "float16",
  int32: "int",
  int16: "int16",
} as const;

export const PYTHON_MISC_TYPES: Record<string, string> = {
  float: "float",
  int: "int",
  size_t: "int",
} as const;

export const MOJO_TYPES: Record<DataType, string> = {
  float32: "Float32",
  float16: "Float16",
  int32: "Int32",
  int16: "Int16",
} as const;

export const MOJO_DTYPE_CONST: Record<DataType, string> = {
  float32: "DType.float32",
  float16: "DType.float16",
  int32: "DType.int32",
  int16: "DType.int16",
} as const;

export const MOJO_MISC_TYPES: Record<string, string> = {
  float: "Float32",
  int: "Int32",
  size_t: "Int32",
} as const;

export const CUTE_TYPES: Record<DataType, string> = {
  float32: "cute.Float32",
  float16: "cute.Float16",
  int32: "cute.Int32",
  int16: "cute.Int16",
} as const;

export const CUTE_MISC_TYPES: Record<string, string> = {
  float: "cute.Float32",
  int: "cute.Int32",
  size_t: "cute.Int32",
} as const;
