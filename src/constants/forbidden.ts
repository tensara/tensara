export const FORBIDDEN_PATTERNS: Record<string, string[]> = {
  cuda: [
    "#\\s*include\\s*<thrust/",
    "\\bthrust::",
    "\\bstd::sort\\b",
    "\\bstd::stable_sort\\b",
    "\\bqsort\\s*\\(",
  ],
  python: [
    "\\bimport\\s+thrust\\b",
    "\\bfrom\\s+thrust\\b",
    "\\b(?:tl|torch)\\s*\\.\\s*(?:sort|topk)\\b",
    "\\beval\\s*\\(",
    "\\bexec\\s*\\(",
    "\\bopen\\s*\\(",
    "__import__",
    "\\bimportlib\\s*\\.",
    "from\\s+[\\w\\.]*builtin\\s+import\\s+sort",
  ],
  mojo: [
    "from\\s+builtin\\.sort\\s+import\\s+sort",
    "\\bbuiltin\\.sort\\.sort\\b",
    "\\bsort\\s*\\(",
  ],
};

