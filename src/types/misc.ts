import { type SandboxStatusType } from "./submission";

export type ProgrammingLanguage = "cuda" | "python" | "mojo" | "cute" | "hip";

export type DataType = "float16" | "float32" | "int32" | "int16";

export interface ApiKey {
  id: string;
  name: string;
  key: string;
  createdAt: Date;
  expiresAt: Date;
}

export interface SandboxFile {
  name: string;
  content: string;
}

export interface SandboxOutput {
  status: SandboxStatusType;
  stdout?: string;
  stderr?: string;
  return_code?: number;
}
