export interface Puzzle {
  id: string;
  title: string;
  description: string;
  difficulty: "beginner" | "intermediate" | "advanced";
  author: string;
  puzzleMd?: string;
  hints?: string[];
  startingCode?: {
    cuda: string;
    python: string;
  };
  nextPuzzleId?: string;
  prevPuzzleId?: string;
  number?: number;
}

export const gpuPuzzles: Record<string, Puzzle> = {
  "vector-add": {
    id: "vector-add",
    title: "Puzzle 1: Map",
    description:
      "Learn the basics of GPU programming by implementing vector addition",
    difficulty: "beginner",
    author: "Sasha Rush",
    puzzleMd: `# Puzzle 1: Map - Vector Addition

## Introduction
Welcome to your first GPU programming challenge! In this puzzle, you'll learn the fundamental concept of mapping operations across parallel threads - one of the core patterns in GPU programming.

## Problem
Implement a "kernel" (GPU function) that adds 10 to each position of vector \`a\` and stores it in vector \`out\`. 

## Key Concepts
- You have 1 thread per position in the vector
- Each thread needs to know which element it's responsible for
- Apply the same operation to each element independently

## Helpful Tips
Think of the function \`call\` as being run 1 time for each thread. The only difference is that \`cuda.threadIdx.x\` changes each time.`,
    hints: [
      "Use cuda.threadIdx.x to get the current thread index - this tells you which element of the array your thread is responsible for.",
      "Remember that GPU programming is about parallelism - each thread works on one element simultaneously with all other threads.",
      "The operation is straightforward: out[i] = a[i] + 10, where i is your thread index.",
      "Always ensure your thread index doesn't exceed the array bounds in real-world applications.",
    ],
    startingCode: {
      cuda: `#include <cuda_runtime.h>  // Note: input, output are all device pointers to float32 arrays
extern "C" void solution(const float* input, float* output) {
    // TODO: Implement vector addition (add 10 to each element)
}`,
      python: `import triton
import triton.language as tl

@triton.jit
def map_kernel(
    out_ptr,
    a_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the program ID
    pid = tl.program_id(axis=0)
    # Compute the offsets
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Create a mask to handle the case where BLOCK_SIZE is larger than n_elements
    mask = offsets < n_elements
    # Load the input
    a = tl.load(a_ptr + offsets, mask=mask)
    
    # TODO: Implement the operation (add 10 to each element)
    
    # Write the output
    tl.store(out_ptr + offsets, out, mask=mask)`,
    },
    nextPuzzleId: "vector-zip",
  },

  // Additional puzzles can be added here
  "vector-zip": {
    id: "vector-zip",
    title: "Puzzle 2: Zip",
    description: "Learn to combine two vectors using element-wise operations",
    difficulty: "beginner",
    author: "Sasha Rush",
    puzzleMd: `# Puzzle 2: Zip - Element-wise Vector Operations

## Introduction
This puzzle builds on your knowledge of mapping by introducing operations that combine two vectors element by element.

## The Challenge
Implement a kernel that adds corresponding elements from vectors \`a\` and \`b\` and stores the result in vector \`out\`.

## Key Concepts
- **Element-wise Operations**: Combining corresponding elements from multiple arrays
- **Parallel Data Processing**: Each thread handles one element from each input array

Remember that each thread still works independently, but now processes data from multiple sources.`,
    hints: [
      "Load both input arrays using the same thread index mechanism you learned in Puzzle 1",
      "For each thread, add the corresponding elements: out[i] = a[i] + b[i]",
      "The pattern of one thread per element remains the same, but now each thread reads two values",
    ],
    startingCode: {
      cuda: `#include <cuda_runtime.h>
extern "C" void solution(const float* a, const float* b, float* output) {
    // TODO: Implement element-wise addition of vectors a and b
}`,
      python: `import triton
import triton.language as tl

@triton.jit
def zip_kernel(
    out_ptr,
    a_ptr,
    b_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the program ID
    pid = tl.program_id(axis=0)
    # Compute the offsets
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Create a mask
    mask = offsets < n_elements
    
    # TODO: Load inputs and implement element-wise addition
    
    # Write the output
    tl.store(out_ptr + offsets, out, mask=mask)`,
    },
    nextPuzzleId: "vector-reduce",
  },
};
