import { useState, useEffect, useCallback, useMemo } from "react";
import {
  saveSolutionToStorage,
  loadSolutionFromStorage,
  getSolutionKey,
  loadPreferences,
} from "~/utils/localStorage";
import { generateStarterCode } from "~/utils/starter";
import { type ProgrammingLanguage } from "~/types/misc";
import { type Parameter } from "~/types/problem";
import { DEFAULT_LANGUAGE } from "~/constants/problem";
import { type Problem } from "@prisma/client";

export function useCodePersistence(
  slug: string | undefined,
  problem: Problem,
  initialLanguage: ProgrammingLanguage = DEFAULT_LANGUAGE
) {
  const [code, setCode] = useState<string>("");
  const [selectedLanguage, setSelectedLanguage] =
    useState<ProgrammingLanguage>(initialLanguage);
  const [savedGpuType, setSavedGpuType] = useState<string | undefined>(
    undefined
  );
  const [isCodeDirty, setIsCodeDirty] = useState<boolean>(false);
  const [starterCode, setStarterCode] = useState<string>("");
  const [hasSetInitialCode, setHasSetInitialCode] = useState<boolean>(false);
  const [hasLoadedPreferences, setHasLoadedPreferences] =
    useState<boolean>(false);

  // Load preferences from localStorage on client side only
  useEffect(() => {
    if (!hasLoadedPreferences && slug) {
      const savedPreferences = loadPreferences(slug);
      if (savedPreferences) {
        if (savedPreferences.language) {
          setSelectedLanguage(savedPreferences.language as ProgrammingLanguage);
        }
        if (savedPreferences.gpuType) {
          setSavedGpuType(savedPreferences.gpuType);
        }
      }
      setHasLoadedPreferences(true);
    }
  }, [slug, hasLoadedPreferences]);

  const memoizedStarterCode = useMemo(
    () =>
      generateStarterCode(
        problem?.parameters as unknown as Parameter[],
        selectedLanguage
      ),
    [problem?.parameters, selectedLanguage]
  );

  // Load starter code based on language and data type
  useEffect(() => {
    if (!slug || !problem?.parameters) return;

    if (memoizedStarterCode) {
      setStarterCode(memoizedStarterCode);
      const savedSolution = loadSolutionFromStorage(slug, selectedLanguage);
      if (savedSolution) {
        setCode(savedSolution);
      } else {
        setCode(memoizedStarterCode);
        saveSolutionToStorage(slug, memoizedStarterCode, selectedLanguage);
      }
    }
  }, [selectedLanguage, problem?.parameters, slug, memoizedStarterCode]);

  // Check if code is dirty (different from starter code)
  useEffect(() => {
    if (starterCode) {
      setIsCodeDirty(code !== starterCode);
    }
  }, [code, starterCode]);

  // Save code to localStorage when it changes
  useEffect(() => {
    if (code && slug) {
      saveSolutionToStorage(slug, code, selectedLanguage);
    }
  }, [code, slug, selectedLanguage]);

  // Initial code setup
  useEffect(() => {
    if (!hasSetInitialCode && slug) {
      const savedSolution = loadSolutionFromStorage(slug, selectedLanguage);
      if (savedSolution) {
        setCode(savedSolution);
        setHasSetInitialCode(true);
      } else if (starterCode) {
        setCode(starterCode);
        setHasSetInitialCode(true);
      }
    }
  }, [slug, hasSetInitialCode, starterCode, selectedLanguage]);

  const handleReset = useCallback(() => {
    if (starterCode && slug) {
      setCode(starterCode);
      localStorage.removeItem(getSolutionKey(slug, selectedLanguage));
    }
  }, [starterCode, slug, selectedLanguage]);

  return {
    code,
    setCode,
    selectedLanguage,
    setSelectedLanguage,
    isCodeDirty,
    handleReset,
    starterCode,
    savedGpuType,
    hasLoadedPreferences,
  };
}
