const LOCAL_STORAGE_PREFIX = "problem_solution_";

export const getSolutionKey = (
  slug: string,
  language: string,
  dataType: string
): string => `${LOCAL_STORAGE_PREFIX}${slug}_${language}_${dataType}`;

export const saveSolutionToStorage = (
  slug: string,
  code: string,
  language: string,
  dataType: string
): void => {
  if (typeof window === "undefined") return;
  localStorage.setItem(getSolutionKey(slug, language, dataType), code);
};

export const loadSolutionFromStorage = (
  slug: string,
  language: string,
  dataType: string
): string | null => {
  if (typeof window === "undefined") return null;
  return localStorage.getItem(getSolutionKey(slug, language, dataType));
};
