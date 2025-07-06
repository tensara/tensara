export type ProgrammingLanguage = "cuda" | "python" | "mojo";

export type DataType = "float16" | "float32" | "int32" | "int16";

export interface ApiKey {
  id: string;
  name: string;
  key: string;
  createdAt: Date;
  expiresAt: Date;
}
