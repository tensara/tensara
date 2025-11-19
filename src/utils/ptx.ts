export type PtxSourceMap = Record<number, number[]>;

const FILE_REGEX = /^\.file\s+(\d+)\s+"([^"]+)"/;
const LOC_REGEX = /^\.loc\s+(\d+)\s+(\d+)/;

/**
 * Parse PTX source and build a mapping from CUDA source lines to PTX lines.
 * Uses `.file` and `.loc` directives emitted by NVCC (`-lineinfo` flag).
 */
export function createPtxSourceMap(ptxContent: string): PtxSourceMap {
  const sourceMap: PtxSourceMap = {};
  const fileTable: Record<number, string> = {};
  const preferredFiles = new Set<number>();
  let primaryFileId: number | null = null;

  let currentFileId: number | null = null;
  let currentSourceLine: number | null = null;

  const lines = ptxContent.split(/\r?\n/);

  lines.forEach((line, index) => {
    const trimmed = line.trim();
    if (!trimmed) {
      return;
    }

    const fileMatch = trimmed.match(FILE_REGEX);
    if (fileMatch) {
      const fileId = Number(fileMatch[1]);
      const filePath = fileMatch[2];
      if (!filePath) {
        return;
      }
      fileTable[fileId] = filePath;
      const normalizedPath = filePath.toLowerCase();
      if (primaryFileId === null) {
        primaryFileId = fileId;
      }

      const isCudaFile =
        normalizedPath.endsWith(".cu") ||
        normalizedPath.endsWith(".cuh") ||
        normalizedPath.includes(".cu.");

      if (isCudaFile) {
        preferredFiles.add(fileId);
        primaryFileId = fileId;
      }
      return;
    }

    const locMatch = trimmed.match(LOC_REGEX);
    if (locMatch) {
      currentFileId = Number(locMatch[1]);
      currentSourceLine = Number(locMatch[2]);
      return;
    }

    // Only map instructions associated with the primary CUDA file.
    if (currentSourceLine == null || currentFileId == null) {
      return;
    }

    const isPreferred =
      preferredFiles.size > 0
        ? preferredFiles.has(currentFileId)
        : primaryFileId === null || primaryFileId === currentFileId;

    if (!isPreferred) {
      return;
    }

    if (!sourceMap[currentSourceLine]) {
      sourceMap[currentSourceLine] = [];
    }
    sourceMap[currentSourceLine]?.push(index + 1);
  });

  return sourceMap;
}
