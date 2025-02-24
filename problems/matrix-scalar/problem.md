---
slug: "matrix-scalar"
title: "Matrix Scalar Multiplication"
difficulty: "EASY"
author: "sarthak"
tags: ["cuda-basics", "parallel-computing"]
---

Implement a CUDA kernel to perform multiplication of a matrix and a scalar:
$C_{i,j} = A_{i,j} \cdot s$ where $s$ is the scalar value

## Input:
- Matrix $A$ of size $M \times N$
- Scalar value $s$

## Output:
- Matrix $C$ of size $M \times N$ where each element is multiplied by the scalar