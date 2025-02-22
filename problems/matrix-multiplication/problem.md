---
slug: "matrix-multiplication"
title: "Matrix Multiplication"
difficulty: "MEDIUM"
author: "sarthak"
tags: ["cuda-basics", "parallel-computing"]
---

## Problem Statement
Implement a CUDA kernel to perform matrix multiplication of two matrices:
`C[i,j] = Σ(A[i,k] * B[k,j])` for k = 0 to K-1

## Input Specifications
- Matrix A is of size `M x K`
- Matrix B is of size `K x N`
- M, N, and K will be between 32 and 4096
- All matrices contain single-precision floating point values
- Input matrices are stored in row-major order

## Output Requirements
- Matrix C of size `M x N` must contain the matrix multiplication result
- Must handle non-uniform grid sizes efficiently
- Implementation should optimize for memory coalescing and shared memory usage
