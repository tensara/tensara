const LOCAL_STORAGE_PREFIX = "problem_solution_";
const PREFERENCES_PREFIX = "problem_preferences_";
const BLOG_POST_SNAPSHOT_PREFIX = "blogPost_";
const EDITOR_VIM_MODE_KEY = "editor_vim_mode";

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

export const saveVimModePreference = (enabled: boolean): void => {
  if (typeof window === "undefined") return;
  localStorage.setItem(EDITOR_VIM_MODE_KEY, enabled ? "true" : "false");
};

export const loadVimModePreference = (): boolean | null => {
  if (typeof window === "undefined") return null;
  const stored = localStorage.getItem(EDITOR_VIM_MODE_KEY);
  if (stored == null) return null;
  return stored === "true";
};

export interface BlogPostSnapshot {
  title: string;
  content: string;
  tags: string[];
}

export const saveBlogPostSnapshot = (
  id: string,
  snapshot: BlogPostSnapshot
): void => {
  if (typeof window === "undefined") return;
  try {
    localStorage.setItem(
      `${BLOG_POST_SNAPSHOT_PREFIX}${id}`,
      JSON.stringify(snapshot)
    );
  } catch (err) {
    console.warn("Failed to save blog post snapshot to localStorage:", err);
  }
};

export const loadBlogPostSnapshot = (id: string): BlogPostSnapshot | null => {
  if (typeof window === "undefined") return null;
  try {
    const stored = localStorage.getItem(`${BLOG_POST_SNAPSHOT_PREFIX}${id}`);
    if (!stored) return null;
    return JSON.parse(stored) as BlogPostSnapshot;
  } catch {
    return null;
  }
};

export const clearBlogPostSnapshot = (id: string): void => {
  if (typeof window === "undefined") return;
  try {
    localStorage.removeItem(`${BLOG_POST_SNAPSHOT_PREFIX}${id}`);
  } catch (err) {
    console.warn("Failed to clear blog post snapshot from localStorage:", err);
  }
};

const BLOG_ACTIVE_TAB_KEY = "blog_active_tab";

export const saveBlogActiveTab = (
  tab: "all" | "myPosts" | "myDrafts"
): void => {
  if (typeof window === "undefined") return;
  try {
    localStorage.setItem(BLOG_ACTIVE_TAB_KEY, tab);
  } catch (err) {
    console.warn("Failed to save blog active tab to localStorage:", err);
  }
};

export const loadBlogActiveTab = (): "all" | "myPosts" | "myDrafts" | null => {
  if (typeof window === "undefined") return null;
  try {
    const stored = localStorage.getItem(BLOG_ACTIVE_TAB_KEY);
    if (!stored) return null;
    if (stored === "all" || stored === "myPosts" || stored === "myDrafts") {
      return stored;
    }
    return null;
  } catch {
    return null;
  }
};
