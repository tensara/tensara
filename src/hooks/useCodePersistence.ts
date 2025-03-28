import { useState, useEffect, useCallback, useMemo } from 'react';
import { saveSolutionToStorage, loadSolutionFromStorage, getSolutionKey } from '~/utils/localStorage';
import { generateStarterCode } from '~/utils/starter';
import { type ProgrammingLanguage, type DataType } from '~/types/misc'; 
import { type Parameter } from '~/types/problem';
import { DEFAULT_LANGUAGE, DEFAULT_DATA_TYPE } from '~/constants/problem';
import { type Problem } from "@prisma/client";

export function useCodePersistence(
  slug: string | undefined, 
  problem: Problem, 
  initialLanguage: ProgrammingLanguage = DEFAULT_LANGUAGE, 
  initialDataType: DataType = DEFAULT_DATA_TYPE
) {
  const [code, setCode] = useState<string>("");
  const [selectedLanguage, setSelectedLanguage] = useState<ProgrammingLanguage>(initialLanguage);
  const [selectedDataType, setSelectedDataType] = useState<DataType>(initialDataType);
  const [isCodeDirty, setIsCodeDirty] = useState<boolean>(false);
  const [starterCode, setStarterCode] = useState<string>("");
  const [hasSetInitialCode, setHasSetInitialCode] = useState<boolean>(false);


  const memoizedStarterCode = useMemo(() => {
    console.log("PRITNING THIS", problem)
    return generateStarterCode(problem?.parameters as unknown as Parameter[], selectedLanguage, selectedDataType);
  }, [problem?.parameters, selectedLanguage, selectedDataType]);
  
  // Load starter code based on language and data type
  useEffect(() => {
    if (!slug || !problem?.parameters) return;

    if (memoizedStarterCode) {
      setStarterCode(memoizedStarterCode);
      const savedSolution = loadSolutionFromStorage(slug, selectedLanguage, selectedDataType);
      if (savedSolution) {
        setCode(savedSolution);
      } else {
        setCode(memoizedStarterCode);
        saveSolutionToStorage(slug, memoizedStarterCode, selectedLanguage, selectedDataType);
      }
    }
  }, [selectedLanguage, selectedDataType, problem?.parameters, slug, memoizedStarterCode]);

  // Check if code is dirty (different from starter code)
  useEffect(() => {
    if (starterCode) {
      setIsCodeDirty(code !== starterCode);
    }
  }, [code, starterCode]);

  // Save code to localStorage when it changes
  useEffect(() => {
    if (code && slug) {
      saveSolutionToStorage(slug, code, selectedLanguage, selectedDataType);
    }
  }, [code, slug, selectedLanguage, selectedDataType]);

  // Initial code setup
  useEffect(() => {
    if (!hasSetInitialCode && slug) {
      const savedSolution = loadSolutionFromStorage(slug, selectedLanguage, selectedDataType);
      if (savedSolution) {
        setCode(savedSolution);
        setHasSetInitialCode(true);
      } else if (starterCode) {
        setCode(starterCode);
        setHasSetInitialCode(true);
      }
    }
  }, [slug, hasSetInitialCode, starterCode, selectedLanguage, selectedDataType]);

  const handleReset = useCallback(() => {
    if (starterCode && slug) {
      setCode(starterCode);
      localStorage.removeItem(getSolutionKey(slug, selectedLanguage, selectedDataType));
    }
  }, [starterCode, slug, selectedLanguage, selectedDataType]);

  return {
    code,
    setCode,
    selectedLanguage,
    setSelectedLanguage,
    selectedDataType,
    setSelectedDataType,
    isCodeDirty,
    handleReset,
    starterCode
  };
}