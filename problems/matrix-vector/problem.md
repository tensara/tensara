---
slug: "matrix-vector"
title: "Matrix Vector Multiplication"
difficulty: "EASY"
author: "sarthak"
tags: ["cuda-basics", "parallel-computing"]
parameters:
  - name: "input_a"
    type: "[VAR]"
    pointer: "true"
    const: "true"
  
  - name: "input_b"
    type: "[VAR]"
    pointer: "true"
    const: "true"

  - name: "output_c" 
    type: "[VAR]"
    pointer: "true"
    const: "false"

  - name: "m" 
    type: "size_t"
    pointer: "false"
    constant: "false"
    
  - name: "k"
    type: "size_t"
    pointer: "false"
    constant: "false"
---

Perform multiplication of a matrix and a vector:
$$
C[i] = \sum_{k=0}^{K-1} A[i][k] \cdot B[k]
$$

## Input:
- Matrix $A$ of size $M \times K$
- Vector $B$ of size $K \times 1$

## Output:
- Vector $C = AB$ of size $M \times 1$

## Notes:
- Matrix $\text{A}$ is stored in row-major order