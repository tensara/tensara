---
slug: "square-matmul"
title: "Square Matrix Multiplication"
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

  - name: "n" 
    type: "size_t"
    pointer: "false"
    constant: "false"
    
---

Perform multiplication of two square matrices:
$$
C[i][j] = \sum_{k=0}^{N-1} A[i][k] \cdot B[k][j]
$$

## Input
- Matrix $A$ of size $N \times N$
- Matrix $B$ of size $N \times N$ 

## Output
- Matrix $C = AB$ of size $N \times N$

## Notes:
- All matrices $\text{A}$, $\text{B}$, and $\text{C}$ are stored in row-major order