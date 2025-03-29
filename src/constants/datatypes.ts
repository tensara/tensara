import { type DataType } from "~/types/misc";

export const DATA_TYPE_DISPLAY_NAMES: Record<DataType, string> = {
    "float32": "float32",
    "float16": "float16",
    "int32": "int32",
    "int16": "int16",
} as const;

export const IS_DISABLED_DATA_TYPE: Record<DataType, boolean> = {
    "float32": false,
    "float16": true,
    "int32": true,
    "int16": true,
} as const;

export const CPP_TYPES: Record<DataType, string> = {
    "float32": "float",
    "float16": "double",
    "int32": "int",
    "int16": "short"
} as const;
  
export const PYTHON_TYPES: Record<DataType, string> = {
    "float32": "float",
    "float16": "float16",
    "int32": "int",
    "int16": "int16"
} as const;
  