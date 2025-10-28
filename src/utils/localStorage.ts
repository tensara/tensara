const LOCAL_STORAGE_PREFIX = "problem_solution_";
const PREFERENCES_PREFIX = "problem_preferences_";

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

interface ProblemPreferences {
  language: string;
  dataType: string;
  gpuType: string;
}

export const savePreferences = (
  slug: string,
  preferences: ProblemPreferences
): void => {
  if (typeof window === "undefined") return;
  localStorage.setItem(
    `${PREFERENCES_PREFIX}${slug}`,
    JSON.stringify(preferences)
  );
};

export const loadPreferences = (slug: string): ProblemPreferences | null => {
  if (typeof window === "undefined") return null;
  const stored = localStorage.getItem(`${PREFERENCES_PREFIX}${slug}`);
  if (!stored) return null;
  try {
    return JSON.parse(stored) as ProblemPreferences;
  } catch {
    return null;
  }
};
