export interface Parameter {
  name: string
  type: string
  const: string
  pointer: string
}

export interface Problem {
  id: string;
  title: string;
  slug: string;
  description: string;
  difficulty: string;
  parameters: Parameter[];
}

export type DebugInfo = {
  max_difference?: number;
  mean_difference?: number;
  sample_differences?: Record<string, {
    expected: number;
    actual: number;
    diff: number;
  }>;
  message?: string;
};
