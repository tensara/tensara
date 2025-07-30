import { useState, useEffect, useCallback, useRef } from "react";
import { generateStarterCode } from "~/utils/starter";
import { DEFAULT_DATA_TYPE } from "~/constants/problem";
import type { Difficulty } from "~/constants/problem";

const CACHE_KEY = "contribution-form-cache";

const pythonStarterCode = generateStarterCode([], "python", DEFAULT_DATA_TYPE);

const descriptionTemplate = `## Problem Statement
Your problem statement goes here. Be descriptive and clear.

## Input
- Describe the input format here.

## Output
- Describe the expected output format here.

## Notes
- Add any constraints, edge cases, or other notes here.`;

interface Parameter {
  name: string;
  type: string;
  pointer: boolean;
  const: boolean;
}

interface ContributionFormState {
  title: string;
  slug: string;
  description: string;
  difficulty: Difficulty;
  tags: string[];
  referenceSolutionCode: string;
  testCases: { input: string; output: string }[];
  parameters: Parameter[];
  flops: string;
}

const defaultInitialState: ContributionFormState = {
  title: "",
  slug: "",
  description: descriptionTemplate,
  difficulty: "medium",
  tags: [],
  referenceSolutionCode: pythonStarterCode,
  testCases: [{ input: "", output: "" }],
  parameters: [],
  flops: "",
};

export const useContributionFormPersistence = () => {
  const [formState, setFormState] =
    useState<ContributionFormState>(defaultInitialState);
  const isMounted = useRef(false);

  // Load state from localStorage on initial client-side render
  useEffect(() => {
    try {
      const cachedData = localStorage.getItem(CACHE_KEY);
      if (cachedData) {
        const loadedState = JSON.parse(
          cachedData
        ) as Partial<ContributionFormState>;
        setFormState({ ...defaultInitialState, ...loadedState });
      }
    } catch (error) {
      console.error("Failed to load from localStorage", error);
    }
  }, []);

  useEffect(() => {
    if (isMounted.current) {
      try {
        localStorage.setItem(CACHE_KEY, JSON.stringify(formState));
      } catch (error) {
        console.error("Failed to save to localStorage", error);
      }
    } else {
      isMounted.current = true;
    }
  }, [formState]);

  const handleReset = useCallback(() => {
    setFormState(defaultInitialState);
    try {
      localStorage.removeItem(CACHE_KEY);
    } catch (error) {
      console.error("Failed to remove from localStorage", error);
    }
  }, []);

  const setTitle = (title: string) =>
    setFormState((prevState) => ({ ...prevState, title }));
  const setSlug = (slug: string) =>
    setFormState((prevState) => ({ ...prevState, slug }));
  const setDescription = (description: string) =>
    setFormState((prevState) => ({ ...prevState, description }));
  const setDifficulty = (difficulty: Difficulty) =>
    setFormState((prevState) => ({ ...prevState, difficulty }));
  const setTags = (tags: string[]) =>
    setFormState((prevState) => ({ ...prevState, tags }));
  const setReferenceSolutionCode = (referenceSolutionCode: string) =>
    setFormState((prevState) => ({ ...prevState, referenceSolutionCode }));
  const setFlops = (flops: string) =>
    setFormState((prevState) => ({ ...prevState, flops }));

  const addTestCase = useCallback(() => {
    setFormState((prevState) => ({
      ...prevState,
      testCases: [...prevState.testCases, { input: "", output: "" }],
    }));
  }, []);

  const updateTestCase = useCallback(
    (
      index: number,
      newTestCase: Partial<{ input: string; output: string }>
    ) => {
      setFormState((prevState) => {
        const newTestCases = [...prevState.testCases];
        const currentTestCase = newTestCases[index];
        if (currentTestCase) {
          newTestCases[index] = { ...currentTestCase, ...newTestCase };
        }
        return { ...prevState, testCases: newTestCases };
      });
    },
    []
  );

  const removeTestCase = useCallback((index: number) => {
    setFormState((prevState) => ({
      ...prevState,
      testCases: prevState.testCases.filter((_, i) => i !== index),
    }));
  }, []);

  const addParameter = useCallback(() => {
    setFormState((prevState) => ({
      ...prevState,
      parameters: [
        ...prevState.parameters,
        { name: "", type: "", pointer: false, const: false },
      ],
    }));
  }, []);

  const updateParameter = useCallback(
    (index: number, newParameter: Partial<Parameter>) => {
      setFormState((prevState) => {
        const newParameters = [...prevState.parameters];
        const currentParameter = newParameters[index];
        if (currentParameter) {
          newParameters[index] = { ...currentParameter, ...newParameter };
        }
        return { ...prevState, parameters: newParameters };
      });
    },
    []
  );

  const removeParameter = useCallback((index: number) => {
    setFormState((prevState) => ({
      ...prevState,
      parameters: prevState.parameters.filter((_, i) => i !== index),
    }));
  }, []);

  return {
    ...formState,
    setTitle,
    setSlug,
    setDescription,
    setDifficulty,
    setTags,
    setReferenceSolutionCode,
    setFlops,
    addTestCase,
    updateTestCase,
    removeTestCase,
    addParameter,
    updateParameter,
    removeParameter,
    handleReset,
  };
};
